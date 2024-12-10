import json
import requests
import pymysql
import numpy as np
import pandas as pd
import pyspark.pandas as ps
import pyspark.sql.functions as F

from scipy import stats
from functools import reduce
from pyspark.sql import DataFrame
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from backend_spark.doris_common.doris_client import DorisClient
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, IntegerType, FloatType
from pyspark.sql.functions import pandas_udf, PandasUDFType, monotonically_increasing_id, lit, col

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
def pre_process(df, agg_list):
    new_df = df.dropna(subset=['OPE_NO', 'INLINE_PARAMETER_ID']).copy()

    # 数据类型转换
    convert_to_numeric_list = agg_list + ['label', 'AVG_SPEC_CHK_RESULT']
    new_df[convert_to_numeric_list] = new_df[convert_to_numeric_list].apply(pd.to_numeric)

    # 缺失值填充
    new_df[agg_list] = new_df[agg_list].apply(lambda x: x.fillna(x.mean()))

    sum_spec_chk_res = new_df['AVG_SPEC_CHK_RESULT'].sum()
    sum_spec_chk_res = 0.0 if pd.isna(sum_spec_chk_res) else sum_spec_chk_res
    return sum_spec_chk_res, new_df


def dis_gravity(df, cols):
    """分类重心距离"""
    df_good = df.query('label == 0')
    df_bad = df.query('label == 1')

    feature_cols = []
    for col in cols:
        sample_good_count = len(df_good[col].dropna())
        sample_bad_count = len(df_bad[col].dropna())
        if df[col].nunique() > 1 and sample_bad_count > 0 and sample_good_count > 0:
            feature_cols.append(col)

    if len(feature_cols) == 0:
        return -1

    x_good = []
    x_bad = []
    for col in feature_cols:
        x_good.append(df_good[col].dropna().mean())
        x_bad.append(df_bad[col].dropna().mean())

    x_good = np.array(x_good).reshape(-1)
    x_bad = np.array(x_bad).reshape(-1)
    score = np.linalg.norm(x_good - x_bad)
    return score


def fit_inline_algorithm(df, by, agg_list):
    schema_all = StructType([StructField("OPE_NO", StringType(), True),
                             StructField("weight", FloatType(), True),
                             StructField("INLINE_PARAMETER_ID", StringType(), True),
                             StructField("AVG_SPEC_CHK_RESULT_COUNT", FloatType(), True)])

    @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
    def get_model_result(df):
        sum_spec_chk_res, new_df = pre_process(df, agg_list)
        score = dis_gravity(new_df, agg_list)

        # 增加一些字段信息
        new_df = new_df[['OPE_NO', 'INLINE_PARAMETER_ID']].head(1)
        new_df['weight'] = score
        new_df['AVG_SPEC_CHK_RESULT_COUNT'] = sum_spec_chk_res
        new_df = new_df[['OPE_NO', 'INLINE_PARAMETER_ID', 'weight', 'AVG_SPEC_CHK_RESULT_COUNT']]
        return new_df

    return df.groupby(by).apply(get_model_result)


def add_certain_column(df, by, request_id):
    """
    param df: 最后的建模结果
    param by: 分组字段, 手动增加一列add
    param request_id: 传入的request_id
    return: 最后的建模结果增加特定的列
    """
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
        final_res['weight'] = final_res['weight'].astype(float)
        final_res = final_res.query("weight > 0")
        final_res['weight'] = final_res['weight'] / final_res['weight'].sum()
        final_res['weight_percent'] = final_res['weight'] * 100

        final_res['request_id'] = request_id
        final_res = final_res.sort_values('weight', ascending=False)
        final_res['index_no'] = [i + 1 for i in range(len(final_res))]

        final_res['AVG_SPEC_CHK_RESULT_COUNT'] = final_res['AVG_SPEC_CHK_RESULT_COUNT'].fillna(0)
        final_res = final_res.drop('add', axis=1)
        return final_res

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
    df1 = df1.select("OPE_NO", "INLINE_PARAMETER_ID", "AVERAGE", 'STD_DEV', 'AVG_SPEC_CHK_RESULT', 'label')
    if df1.count() == 0:
        msg = '解析SQL获取数据异常: 数据库中可能没有数据!'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        raise ValueError

    # 2. fit Inline算法
    res = (fit_inline_algorithm(df=df1.repartition(10, 'OPE_NO', 'INLINE_PARAMETER_ID'),
                                by=['OPE_NO', 'INLINE_PARAMETER_ID'],
                                agg_list=['AVERAGE', 'STD_DEV'])
           .withColumnRenamed(existing="OPE_NO", new="OPER_NO")
           .withColumn('add', lit(0)))
    if res.count() == 0:
        msg = '该场景下数据库中暂无充足的数据(真实BAD个数可能为0或真实GOOD个数可能为0)'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        raise ValueError

    # 3. 增加特定的列，算法结果处理
    final_res = add_certain_column(df=res, by='add', request_id=request_id)
    if final_res.count() == 0:
        msg = '算法结果增加列暂时异常'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        raise ValueError
    else:
        # 4.final_res 是最后的结果，要写回数据库
        ddd = final_res.toPandas()
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