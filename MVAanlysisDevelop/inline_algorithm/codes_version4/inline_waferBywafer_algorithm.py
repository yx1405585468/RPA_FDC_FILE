import json
import pyspark
import requests
import pymysql
import pandas as pd
from pca import pca
from functools import reduce
from pyspark.sql import DataFrame
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from backend_spark.doris_common.doris_client import DorisClient
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, IntegerType, FloatType
from pyspark.sql.functions import pandas_udf, PandasUDFType, countDistinct, lit, col, when

client81 = pymysql.connect(user='root', password='Nexchip@123', port=9030, host='10.52.199.81')
data_count = 0
# doris 数据库连接
# client91 = DorisClient("10.52.199.91", 18030, 9030, user="ikas_user", password="Ikas_user@123", data_base="ODS_EDA",
#                      mem_limit="68719476736")

# client81 = DorisClient("10.52.199.81", 18030, 9030, user="root", password="Nexchip@123", data_base="ODS_EDA",
#                      mem_limit="68719476736")


####################################################################################
###################################解析sql 的辅助函数###################################
####################################################################################
def read_sql1(sql_stat, read_client=client81, session=spark):
    # df1 = read_client.doris_read(session, sql_stat)
    df1 = spark.createDataFrame(pd.read_sql(sql=sql_stat, con=client81))
    return df1


def read_sql(sql_stat, read_client=client81, session=spark):
    # df1 = read_client.doris_read(session, sql_stat)
    ds = pd.read_sql(sql=sql_stat, con=client81)
    if ds is None or len(ds) == 0:
        empty_schema = StructType([
            StructField("WAFER_ID", StringType(), True),
            StructField("OPE_NO", StringType(), True),
            StructField("INLINE_PARAMETER_ID", StringType(), True),
            StructField("PRODUCT_ID", StringType(), True),
            StructField("AVERAGE", StringType(), True),
            StructField("STD_DEV", StringType(), True),
            StructField("MEASURE_TIME", TimestampType(), True),
            StructField("AVG_SPEC_CHK_RESULT", StringType(), True),
            StructField("label", StringType(), True)
        ])
        df1 = spark.createDataFrame([], empty_schema)
        # print(df1)
        # null_dataframe = pd.DataFrame(
        # {
        # "WAFER_ID" : [""],
        # "OPE_NO" : [""],
        # "INLINE_PARAMETER_ID" : [""],
        # "PRODUCT_ID" : [""],
        # "AVERAGE" : [0.0],
        # "STD_DEV" : [0.0],
        # "MEASURE_TIME" : [""],
        # "AVG_SPEC_CHK_RESULT" : [""],
        # "label" : [0],
        # }
        # )
        # df1 = spark.createDataFrame(null_dataframe)
    else:
        df1 = spark.createDataFrame(ds)
        data_count = 1
    return df1


def process_like(key: str, value: list[str]) -> str:
    # 处理模糊条件的匹配: (key like 'aa%' or key like "bb%")
    key = keyword_map_from_json_to_table.get(key)
    v_join = ' or '.join([f"{key} like  '{v.replace('*', '%')}' " for v in value])
    return "({})".format(v_join)


def process_not_like(key: str, value: list[str]) -> str:
    # 处理非模糊条件的匹配:key in ('aa', 'bb')
    key = keyword_map_from_json_to_table.get(key)
    v_join = ",".join([f"'{v}'" for v in value])
    return "{} in ({})".format(key, v_join)


def test_not_like():
    result = (process_not_like("tool_name", ["aa", "bb", "cc"]))
    assert "tool_name in ('aa','bb','cc')" == result, "not like 验证失败"


def test_like():
    result = process_like("tool_name", ["aa*", "bb*", "cc*"])

    assert "(tool_name like  'aa%'  or tool_name like  'bb%'  or tool_name like  'cc%' )" == result, "like 验证失败"


def process_one_keyword(key, value: list[str]) -> Optional[str]:
    if len(value) == 0:
        return None

    not_like_list = [v for v in value if "*" not in v]
    like_list = [v for v in value if "*" in v]

    # 处理模糊条件
    if len(not_like_list) != 0:
        not_like_sql_str = process_not_like(key, not_like_list)
    else:
        not_like_sql_str = ""

    # 处理非模糊条件

    if len(like_list) != 0:
        like_sql_str = process_like(key, like_list)
    else:
        like_sql_str = ""

    # 去除为一个元素为空字符串的情况的情况的情况
    concat_sql_str_list = [sql_str for sql_str in [like_sql_str, not_like_sql_str] if len(sql_str) != 0]
    # 使用or 操作 单字段过滤 的like 和 not like 语句
    return "(" + " or ".join(concat_sql_str_list) + ")"


def check_time_start_end(min_time, max_time):
    if min_time is not None and max_time is not None:
        pass
    else:
        raise ValueError("起始时间和结束时间必须全填")


def get_time_selection_sql(time_keyword, max_time=None, min_time=None):
    """
    获取时间区间的筛选的sql, 起始时间和结束时间都是可选的
    :param time_keyword:
    :param max_time:
    :param min_time:
    :return:
    """
    # 根据取值，生成单个时间过滤条件
    if min_time:
        time_part_min = f"{time_keyword} > '{min_time}'"
    else:
        time_part_min = " "

    if max_time:
        time_part_max = f"{time_keyword} <= '{max_time}'"
    else:
        time_part_max = " "

    # 如果存在，拼接多个查询条件，或者只保留一个过滤条件
    if (max_time is not None) and (min_time is not None):
        time_sql = f' {time_part_min} and {time_part_max}'
    elif (max_time is None) and (min_time is None):
        time_sql = " "
    else:
        time_sql = time_part_max if max_time else time_part_min

    return time_sql


def concat_time_filter_sql_with_other_keyword_sql(time_filter_sql: str, other_keyword_sql: str) -> str:
    """
    拼接时间过滤条件与非时间过滤条件
    :param time_filter_sql:
    :param other_keyword_sql:
    :return:
    """
    time_strip = time_filter_sql.strip()
    other_strip = other_keyword_sql.strip()
    if len(time_strip) == 0 and len(other_strip) == 0:
        return ""
    elif len(time_strip) != 0 and len(other_strip) == 0:
        return time_filter_sql
    elif len(time_strip) == 0 and len(other_strip) != 0:
        return other_keyword_sql
    else:
        return f'{time_filter_sql} and {other_keyword_sql}'


def trans_select_condition_to_sql_with_label(select_condition_dict: dict, table_name: str) -> str:
    # 查询条件转sql,并打标签，label '0': good wafer, '1': bad wafer
    filter_sql_list = []
    for keyword, value in select_condition_dict.items():
        if keyword in ["productId", "operNo"]:
            sql_filter_one_keyword = process_one_keyword(keyword, value)
            if sql_filter_one_keyword is not None:
                filter_sql_list.append(sql_filter_one_keyword)

    # 处理时间区间
    time_bin = select_condition_dict.get("dateRange")

    if len(time_bin) > 0:
        time_bin_dict = time_bin[0]
        min_time = time_bin_dict.get("start")
        max_time = time_bin_dict.get("end")
    else:
        min_time = None
        max_time = None

        # 去除时间检查，时间范围为可选输入
    # 检查起始时间和结束时间全部非空
    # check_time_start_end(min_time, max_time)

    # 处理waferId
    waferId = select_condition_dict.get("waferId")
    good_wafer_list = waferId.get("good")
    bad_wafer_list = waferId.get("bad")
    upload_id = select_condition_dict.get("uploadId")

    # 根据time 过滤条件,生成sql
    time_filter_sql = get_time_selection_sql(time_keyword=keyword_map_from_json_to_table.get('dateRange'),
                                             max_time=max_time, min_time=min_time)

    if upload_id is not None and len(upload_id) > 0:

        # good wafer, bad wafe 均有指定，需要从层层字段的过滤的条件下选择
        # good_wafer_filter_sql = process_one_keyword("waferId", good_wafer_list)
        # bad_wafer_filter_sql = process_one_keyword("waferId", bad_wafer_list)
        # # or 拼接
        # wafer_filter_sql = " or ".join([good_wafer_filter_sql, bad_wafer_filter_sql])
        # wafer_filter_sql = f"({wafer_filter_sql})"
        # # 加入wafer 过滤条件
        # filter_sql_list.append(wafer_filter_sql)
        other_keyword_filter = " and ".join(filter_sql_list)

        case_when_statment = f"""  (case
		when d2.GB_FLAG = 'good' then 0 
		else 1
	    end ) as label
        """

        filter_sql_concat = concat_time_filter_sql_with_other_keyword_sql(time_filter_sql, other_keyword_filter)

        if filter_sql_concat != '':
            select_sql = f"""select *, {case_when_statment} from {table_name} d1  
            join etl.UPLOADED_WAFER d2
        on d1.WAFER_ID = d2.WAFER_ID

            where {filter_sql_concat} and d2.UPLOAD_ID = '{upload_id}'"""
        else:
            select_sql = f"""select *, {case_when_statment} from {table_name} d1  
            join etl.UPLOADED_WAFER d2
        on d1.WAFER_ID = d2.WAFER_ID where d2.UPLOAD_ID = '{upload_id}'"""


    elif len(good_wafer_list) + len(bad_wafer_list) == 0:
        raise ValueError("good, bad wafer 至少选择一个")
    elif len(good_wafer_list) > 0 and len(bad_wafer_list) > 0:
        # good wafer, bad wafe 均有指定，需要从层层字段的过滤的条件下选择
        good_wafer_filter_sql = process_one_keyword("waferId", good_wafer_list)
        bad_wafer_filter_sql = process_one_keyword("waferId", bad_wafer_list)
        # or 拼接
        wafer_filter_sql = " or ".join([good_wafer_filter_sql, bad_wafer_filter_sql])
        wafer_filter_sql = f"({wafer_filter_sql})"
        # 加入wafer 过滤条件
        filter_sql_list.append(wafer_filter_sql)
        other_keyword_filter = " and ".join(filter_sql_list)

        case_when_statment = f"""(
        case 
        when {good_wafer_filter_sql} then 0
        else 1 
        end 
        ) label
        """

        filter_sql_concat = concat_time_filter_sql_with_other_keyword_sql(time_filter_sql, other_keyword_filter)
        if filter_sql_concat != '':
            select_sql = f"select *, {case_when_statment} from {table_name} where {filter_sql_concat}"
        else:
            select_sql = f"select *, {case_when_statment} from {table_name}"



    elif len(good_wafer_list) > 0 and len(bad_wafer_list) == 0:
        # 选good, 剩余为 bad
        other_keyword_filter = " and ".join(filter_sql_list)
        good_wafer_filter_sql = process_one_keyword("waferId", good_wafer_list)

        case_when_statment = f"(case when {good_wafer_filter_sql} then 0 else 1  end ) label"
        filter_sql_concat = concat_time_filter_sql_with_other_keyword_sql(time_filter_sql, other_keyword_filter)
        select_sql = f"select *, {case_when_statment} from {table_name} where {filter_sql_concat}"

    elif len(good_wafer_list) == 0 and len(bad_wafer_list) > 0:
        # 选bad, 剩余为good
        other_keyword_filter = " and ".join(filter_sql_list)
        bad_wafer_filter_sql = process_one_keyword("waferId", bad_wafer_list)
        case_when_statment = f"""(case when {bad_wafer_filter_sql} then 1 else 0 end ) label"""
        filter_sql_concat = concat_time_filter_sql_with_other_keyword_sql(time_filter_sql, other_keyword_filter)
        select_sql = f"select *, {case_when_statment} from {table_name} where {filter_sql_concat}"
        # case1 stat results 表没有case_info 时间列，暂时去掉
        # select_sql = f"select *, {case_when_statment} from {table_name} where {other_keyword_filter}"
    select_keywords = """d1.WAFER_ID,
    d1.OPE_NO,
    d1.INLINE_PARAMETER_ID,
    d1.PRODUCT_ID,
    d1.LOT_ID,
    d1.`AVERAGE`,
    d1.STD_DEV,
    d1.MEASURE_TIME,
    d1.AVG_SPEC_CHK_RESULT"""
    select_sql = select_sql.replace("*", select_keywords)
    print(select_sql)
    return select_sql


def get_data_from_doris(select_condition_list, table_name):
    select_df_list = [read_sql(trans_select_condition_to_sql_with_label(select_condition_dict, table_name)) for
                      select_condition_dict in select_condition_list]
    # 多个进行union
    df1 = reduce(DataFrame.unionAll, select_df_list)
    return df1


############################################################################
##############################从kafka消息读取需要的资料#########################
############################################################################
def get_some_info(df: pd.DataFrame):
    if len(df) > 0:
        df = df.head(1)

    request_id = df["requestId"].values[0]
    request_params = df["requestParam"].values[0]
    # 避免存在单引号，因为json 引号只有双引号
    request_params = request_params.replace('\'', "\"")
    parse_dict = json.loads(request_params)
    return parse_dict, request_id


####################################################################################
#####################################结果写回数据库####################################
####################################################################################
def doris_stream_load_from_df(df, engine, table, is_json=True, chunksize=100000, partitions=None):
    engine_url = engine.url
    url = 'http://%s:18030/api/%s/%s/_stream_load' % (engine_url.host, engine_url.database, table)

    format_str = 'csv' if not is_json else 'json'
    headers = {
        'Content-Type': 'text/plain; charset=UTF-8',
        'format': format_str,
        'Expect': '100-continue'
    }
    if is_json:
        headers['strip_outer_array'] = 'true'
        headers['read_json_by_line'] = 'true'
    else:
        headers['column_separator'] = '@'

    if partitions:
        headers['partitions'] = partitions

    auth = requests.auth.HTTPBasicAuth(engine_url.username, engine_url.password)
    session = requests.sessions.Session()
    session.should_strip_auth = lambda old_url, new_url: False

    l = len(df)
    if l > 0:
        if chunksize and chunksize < l:
            batches = l // chunksize
            if l % chunksize > 0:
                batches += 1
            for i in range(batches):
                si = i * chunksize
                ei = min(si + chunksize, l)
                sub = df[si:ei]
                do_doris_stream_load_from_df(sub, session, url, headers, auth, is_json)
        else:
            do_doris_stream_load_from_df(df, session, url, headers, auth, is_json)


def do_doris_stream_load_from_df(df, session, url, headers, auth, is_json=False):
    data = df.to_csv(header=False, index=False, sep='@') if not is_json else df.to_json(orient='records',
                                                                                        date_format='iso')
    # print(data)

    resp = session.request(
        'PUT',
        url=url,
        data=data.encode('utf-8'),
        headers=headers,
        auth=auth
    )
    print(resp.reason, resp.text)
    check_stream_load_response(resp.text)


def check_stream_load_response(resp_text):
    resp = json.loads(resp_text)
    if resp['Status'] not in ["Success", "Publish Timeout"]:
        raise Exception(resp['Message'])


####################################################################################
#####################################INLINE算法####################################
####################################################################################
class DataPreprocessorForInline:
    def __init__(self, df: pyspark.sql.dataframe, columns_list: list[str], certain_column: str,
                 key_words: list[str], convert_to_numeric_list: list[str]) -> pyspark.sql.dataframe:
        self.df = df
        self.columns_list = columns_list
        self.certain_column = certain_column
        self.key_words = key_words
        self.convert_to_numeric_list = convert_to_numeric_list

    def select_columns(self):
        return self.df.select(self.columns_list)

    def exclude_some_data(self, df):
        key_words_str = '|'.join(self.key_words)
        df_filtered = df.filter(~col(self.certain_column).rlike(key_words_str))
        return df_filtered

    def pre_process(self, df):
        for column in self.convert_to_numeric_list:
            df = df.withColumn(column, col(column).cast('double'))
        if 'SITE_COUNT' in self.convert_to_numeric_list:
            self.convert_to_numeric_list.remove('SITE_COUNT')
        df = df.dropna(subset=self.convert_to_numeric_list, how='all')
        return df

    def run(self):
        df_select = self.select_columns()
        df_esd = self.exclude_some_data(df=df_select)
        df_pp = self.pre_process(df=df_esd)
        return df_pp


class GetTrainDataForInline:
    def __init__(self, df: pyspark.sql.dataframe, grpby_list: list[str]) -> pyspark.sql.dataframe:
        self.df_run = df
        self.grpby_list = grpby_list

    def commonality_analysis(self):
        grps = (self.df_run.groupBy(self.grpby_list)
                .agg(countDistinct('WAFER_ID').alias('wafer_count'),
                     countDistinct('WAFER_ID', when(self.df_run['label'] == 0, 1)).alias('good_num'),
                     countDistinct('WAFER_ID', when(self.df_run['label'] == 1, 1)).alias('bad_num'))
                .na.fill(0)
                .orderBy(['bad_num', 'good_num'], ascending=False))
        if grps.count() == 1:
            return grps
        else:
            grps = grps.filter("bad_num > 1 AND wafer_count > 2")
            return grps

    @staticmethod
    def get_data_list(common_res):
        data_list = common_res.select(['OPE_NO']).collect()
        data_dict_list = [row.asDict() for row in data_list]
        return data_dict_list

    def get_train_data(self, data_dict_list):
        oper = data_dict_list[0]['OPE_NO']
        df_s = self.df_run.filter("OPE_NO == '{}'".format(oper))
        for i in range(1, len(data_dict_list)):
            oper = data_dict_list[i]['OPE_NO']
            df_m = self.df_run.filter("OPE_NO == '{}'".format(oper))
            df_s = df_s.union(df_m)
        return df_s

    def run(self):
        common_res = self.commonality_analysis()
        print("common_res.count:", common_res.count())
        data_dict_list = self.get_data_list(common_res)
        train_data = self.get_train_data(data_dict_list)
        return train_data


def process_missing_values(df: pyspark.sql.dataframe, columns_to_process: list[str], threshold: float = 0.6) -> pyspark.sql.dataframe:
    df_processed = df.copy()
    for column in columns_to_process:
        missing_percentage = df[column].isnull().mean()
        if missing_percentage > threshold:
            df_processed = df_processed.drop(columns=[column])
        else:
            df_processed[column] = df_processed[column].fillna(df_processed[column].mean())
    return df_processed


def get_pivot_table(df: pyspark.sql.dataframe, columns_to_process: list[str]) -> pyspark.sql.dataframe:
    df_specific_operno = process_missing_values(df=df, columns_to_process=columns_to_process, threshold=0.6)

    values_list = df_specific_operno.columns.difference(['WAFER_ID', 'OPE_NO', 'INLINE_PARAMETER_ID', 'SITE_COUNT', 'label'])
    pivot_result = df_specific_operno.pivot_table(index=['WAFER_ID', 'label'],
                                                  columns=['OPE_NO', 'INLINE_PARAMETER_ID'],
                                                  values=values_list)
    pivot_result.columns = pivot_result.columns.map('#'.join)
    pivot_result = process_missing_values(df=pivot_result, columns_to_process=pivot_result.columns, threshold=0.6)
    pivot_result = pivot_result.reset_index(drop=False)
    return pivot_result


def fit_pca_model(df: pyspark.sql.dataframe, by: list[str], columns_to_process: list[str]) -> pyspark.sql.dataframe:
    schema_all = StructType([StructField("features", StringType(), True),
                             StructField("importance", FloatType(), True)])

    @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
    def get_model_result(df_run):
        pivot_result = get_pivot_table(df=df_run, columns_to_process=columns_to_process)
        # 定义自变量
        x_train = pivot_result[pivot_result.columns.difference(['WAFER_ID', 'label']).tolist()]
        n_components = min(min(x_train.shape)-1, 5)

        model = pca(n_components=n_components, verbose=None)
        results = model.fit_transform(x_train)
        res_top = results['topfeat']
        res_top_select = res_top[res_top['type'] == 'best'][['feature', 'loading']]
        res_top_select = res_top_select.drop_duplicates()
        res_top_select['importance'] = abs(res_top_select['loading'])
        res_top_select = res_top_select.rename(columns={'feature': 'features'})
        res_top_select = res_top_select.drop("loading", axis=1)
        return res_top_select
    return df.groupby(by).apply(get_model_result)


def split_features(df: pd.DataFrame, index: int) -> str:
    return df['features'].apply(lambda x: x.split('#')[index])


def get_split_features(df: pd.DataFrame) -> pd.DataFrame:
    df['STATISTIC_RESULT'] = split_features(df, 0)
    df['OPE_NO'] = split_features(df, 1)
    df['INLINE_PARAMETER_ID'] = split_features(df, 2)
    df = df.drop(['features', 'STATISTIC_RESULT'], axis=1).reset_index(drop=True)
    return df


def split_calculate_features(df: pyspark.sql.dataframe, by: str) -> pyspark.sql.dataframe:
    schema_all = StructType([StructField("OPE_NO", StringType(), True),
                             StructField("INLINE_PARAMETER_ID", StringType(), True),
                             StructField("importance", FloatType(), True)])

    @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
    def get_model_result(df_run):
        split_table = get_split_features(df_run)
        split_table_grpby = split_table.groupby(['OPE_NO', 'INLINE_PARAMETER_ID'])['importance'].sum().reset_index(
            drop=False)
        split_table_grpby = split_table_grpby.sort_values('importance', ascending=False).reset_index(drop=True)
        return split_table_grpby
    return df.groupby(by).apply(get_model_result)


def add_certain_column(df: pyspark.sql.dataframe, by: str, request_id: str) -> pyspark.sql.dataframe:
    schema_all = StructType([
        StructField("OPER_NO", StringType(), True),
        StructField("INLINE_PARAMETER_ID", StringType(), True),
        StructField("AVG_SPEC_CHK_RESULT_COUNT", FloatType(), True),
        StructField("weight", FloatType(), True),
        StructField("request_id", StringType(), True),
        StructField("weight_percent", FloatType(), True),
        StructField("index_no", IntegerType(), True)])

    @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
    def get_result(final_res):
        # 计算weight, 归一化
        final_res['importance'] = final_res['importance'].astype(float)
        final_res = final_res.query("importance > 0")
        final_res['weight'] = final_res['importance'] / final_res['importance'].sum()
        final_res['weight_percent'] = final_res['weight'] * 100
        final_res = final_res.sort_values('weight', ascending=False)
        # 增加列
        final_res['index_no'] = [i + 1 for i in range(len(final_res))]
        final_res['request_id'] = request_id
        final_res['AVG_SPEC_CHK_RESULT_COUNT'] = 0.0
        final_res = final_res.rename(columns={'OPE_NO': 'OPER_NO'})
        return final_res.drop(['importance', 'add'], axis=1)
    return df.groupby(by).apply(get_result)


####################################################################################
#####################################正式调用以上函数##################################
####################################################################################
# 1. 解析json 为字典， df1 为kafka 输入的结果数据
df2 = df1.toPandas()
parse_dict, request_id = get_some_info(df2)

# 2. 从kafka 关键字映射都具体数据源中的字段,没有的可以删除
keyword_map_from_json_to_table: dict = {
    "waferId": "WAFER_ID",
    "dateRange": "MEASURE_TIME",
    "productId": "PRODUCT_ID",
    "operNo": "OPE_NO",
    # "inlineParameterId": 'INLINE_PARAMETER_ID',
    # "eqp": "TOOL_NAME",
    "lot": "LOT_ID",
    # "recipeName": "RECIPE_NAME"
}
select_condition_list = parse_dict

# 3. 查询表名, 需按实际情况修改
table_name = "ODS_EDA.ODS_INLINE_WAFER_SUMMARY"

# 主程序
try:
    # 1. 从数据库中获取数据
    df1 = get_data_from_doris(select_condition_list=select_condition_list, table_name=table_name)
    print(df1.count())
    if df1.count() == 0:
        msg = '解析SQL获取数据异常: 数据库中可能没有数据!'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        raise ValueError

    # 2. 数据预处理
    dp = DataPreprocessorForInline(df=df1,
                                   columns_list=['WAFER_ID', 'OPE_NO', 'INLINE_PARAMETER_ID', 'AVERAGE', 'MAX_VAL',
                                                 'MEDIAN', 'MIN_VAL', 'STD_DEV', 'PERCENTILE_25', 'PERCENTILE_75',
                                                 'SITE_COUNT', 'label'],
                                   certain_column='INLINE_PARAMETER_ID',
                                   key_words=['CXS', 'CYS', 'FDS'],
                                   convert_to_numeric_list=['AVERAGE', 'MAX_VAL', 'MEDIAN', 'MIN_VAL', 'STD_DEV',
                                                            'PERCENTILE_25', 'PERCENTILE_75', 'SITE_COUNT'])
    df_pp_ = dp.run()
    if df_pp_.count() == 0:
        msg = '预处理后的数据为空'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        raise ValueError

    # 3. 获取训练数据
    gtd = GetTrainDataForInline(df=df_pp_, grpby_list=['OPE_NO'])
    df_run_ = gtd.run()
    if df_run_.count() == 0:
        msg = '训练数据为空'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        raise ValueError

    # 4. 训练模型
    res = fit_pca_model(df=df_run_, by=['OPE_NO'], columns_to_process=['AVERAGE', 'MAX_VAL', 'MEDIAN', 'MIN_VAL',
                                                                       'STD_DEV', 'PERCENTILE_25', 'PERCENTILE_75'])
    if res.count() == 0:
        msg = '该场景下数据库中暂无充足的数据(真实BAD个数可能为0或真实GOOD个数可能为0)'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        raise ValueError

    # 5. 特征处理和排序
    res_add = res.withColumn('add', lit(0))
    final_res = split_calculate_features(df=res_add, by='add')
    final_res = final_res.withColumn('add', lit(0))
    final_res_add = add_certain_column(df=final_res, by='add', request_id=request_id)
    if final_res_add.count() == 0:
        msg = '算法结果增加列暂时异常'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        raise ValueError
    else:
        # final_res_add 是最后的结果，要写回数据库
        ddd = final_res_add.toPandas()
        user = "root"
        host = "10.52.199.81"
        password = "Nexchip%40123"
        db = "etl"
        port = 9030
        engine = create_engine("mysql+pymysql://{user}:{password}@{host}:{port}/{db}".format(user=user,
                                                                                             password=password,
                                                                                             host=host,
                                                                                             port=port,
                                                                                             db=db))
        doris_stream_load_from_df(ddd, engine, "inline_results")

        # 最后运行成功，输出以下kafka信息
        df_kafka = pd.DataFrame({'code': 0, 'msg': '运行成功', 'requestId': request_id}, index=[1])
        df1 = spark.createDataFrame(df_kafka)

except ValueError as ve:
    pass

except Exception as e:
    df_kafka = pd.DataFrame({"code": 1, "msg": f"主程序发生异常: {str(e)}", "requestId": request_id}, index=[0])
    df1 = spark.createDataFrame(df_kafka)

