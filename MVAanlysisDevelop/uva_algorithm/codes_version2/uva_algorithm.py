import pandas as pd
import requests
import json
from sqlalchemy import create_engine
from pca import pca
from pyspark.sql.functions import pandas_udf, PandasUDFType, max, col, countDistinct, when, rank, lit
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from pyspark.sql.window import Window

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import ClusterCentroids

from backend_spark.doris_common.doris_client import DorisClient
from functools import reduce
from pyspark.sql import DataFrame
from typing import Optional

#######################################解析SQL########################################
# doris 数据库连接
client = DorisClient("10.52.199.81", 18030, 9030, user="root", password="Nexchip@123", data_base="etl",
                     mem_limit="68719476736")

"""解析sql 的辅助函数"""


def read_sql(sql_stat, read_client=client, session=spark):
    df1 = read_client.doris_read(session, sql_stat)
    return df1


def process_like(key: str, value: list[str]) -> str:
    # 处理模糊条件的匹配: (key like 'aa%' or key like "bb%")
    key = keyword_map_from_json_to_table.get(key)
    v_join = ' or '.join([f"d1.{key} like  '{v.replace('*', '%')}' " for v in value])
    return "({})".format(v_join)


def process_not_like(key: str, value: list[str]) -> str:
    # 处理非模糊条件的匹配:key in ('aa', 'bb')
    key = keyword_map_from_json_to_table.get(key)
    v_join = ",".join([f"'{v}'" for v in value])
    return "d1.{} in ({})".format(key, v_join)


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
        time_part_min = f"d1.{time_keyword} >= '{min_time}'"
    else:
        time_part_min = " "

    if max_time:
        time_part_max = f"d1.{time_keyword} < '{max_time}'"
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
        if keyword not in ["dateRange", "waferId", "uploadId", "mergeProdg1"]:
            sql_filter_one_keyword = process_one_keyword(keyword, value)
            if sql_filter_one_keyword is not None:
                filter_sql_list.append(sql_filter_one_keyword)

    # 处理时间区间
    time_bin = select_condition_dict.get("dateRange")

    if len(time_bin) == 1:  # list[dict]
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
    # upload_id = '20231116152808771'

    # 根据time 过滤条件,生成sql
    time_filter_sql = get_time_selection_sql(time_keyword=keyword_map_from_json_to_table.get('dateRange'),
                                             max_time=max_time, min_time=min_time)

    # if len(good_wafer_list) > 0 and len(bad_wafer_list) > 0:
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

    else:
        raise ValueError("good bad wafer 都必须选！")

    # elif len(good_wafer_list) > 0 and len(bad_wafer_list) == 0:
    #     # 选good, 剩余为 bad

    #     other_keyword_filter = " and ".join(filter_sql_list)
    #     good_wafer_filter_sql = process_one_keyword("waferId", good_wafer_list)

    #     case_when_statment = f"(case when {good_wafer_filter_sql} then 0 else 1  end ) label"
    #     filter_sql_concat = concat_time_filter_sql_with_other_keyword_sql(time_filter_sql, other_keyword_filter)
    #     select_sql = f"select *, {case_when_statment} from {table_name} where {filter_sql_concat}"

    # elif len(good_wafer_list) == 0 and len(bad_wafer_list) > 0:
    #     # 选bad, 剩余为good
    #     other_keyword_filter = " and ".join(filter_sql_list)
    #     bad_wafer_filter_sql = process_one_keyword("waferId", bad_wafer_list)
    #     case_when_statment = f"""(case when {bad_wafer_filter_sql} then 1 else 0 end ) label"""
    #     filter_sql_concat = concat_time_filter_sql_with_other_keyword_sql(time_filter_sql, other_keyword_filter)
    #     select_sql = f"select *, {case_when_statment} from {table_name} where {filter_sql_concat}"
    #     # case1 stat results 表没有case_info 时间列，暂时去掉
    # select_sql = f"select *, {case_when_statment} from {table_name} where {other_keyword_filter}"

    # print(select_sql)
    select_sql = select_sql.replace("*",
                                    "d1.WAFER_ID, d1.TOOL_ID, d1.RUN_ID, d1.EQP_NAME, d1.PRODUCT_ID, d1.PRODG1, d1.TOOL_NAME, d1.LOT_ID, d1.RECIPE_NAME, d1.OPER_NO, d1.parametric_name, d1.CASE_INFO, d1.STATUS, d1.STATISTIC_RESULT")
    if table_name == "etl.DWD_POC_CASE_FD_UVA_DATA_TEST":
        select_sql = f"{select_sql} and d1.STATUS != 'ERROR'"
    print("select_sql", select_sql)
    return select_sql


def get_data_from_doris(select_condition_list, table_name):
    try:
        select_df_list = [read_sql(trans_select_condition_to_sql_with_label(select_condition_dict, table_name)) for
                          select_condition_dict in select_condition_list]
        # 多个进行union
        df1 = reduce(DataFrame.unionAll, select_df_list)
        return df1
    except Exception as e:
        return None


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
    merge_prodg1 = parse_dict[0]['mergeProdg1']

    if merge_prodg1 == '1':
        grpby_list = ['OPER_NO', 'TOOL_NAME']
    elif merge_prodg1 == '0':
        grpby_list = ['PRODG1', 'OPER_NO', 'TOOL_NAME']
    else:
        raise ValueError
    return parse_dict, request_id, grpby_list



############################################################################
##################################FDC数据预处理###############################
############################################################################
def _pre_process(df):
    """
    param df: 从数据库中读取出来的某个CASE数据
    return: 数据预处理，后面要根据实际情况统一添加
    """
    # 只选出会用到的列
    df = df.select('WAFER_ID', 'TOOL_ID', 'RUN_ID', 'EQP_NAME', 'PRODUCT_ID', 'PRODG1', 'TOOL_NAME',
                   'OPER_NO', 'parametric_name', 'STATISTIC_RESULT', 'label')
    # 剔除NA值
    df = df.filter(col('STATISTIC_RESULT').isNotNull())
    # 按照所有的行进行去重
    df1 = df.dropDuplicates()
    # 选最新的RUN
    df2 = df1.groupBy('WAFER_ID', 'OPER_NO', 'TOOL_ID').agg(max('RUN_ID').alias('RUN_ID'))
    df_run = df1.join(df2.dropDuplicates(subset=['WAFER_ID', 'OPER_NO', 'TOOL_ID', 'RUN_ID']),
                      on=['WAFER_ID', 'OPER_NO', 'TOOL_ID', 'RUN_ID'], how='inner')
    return df_run
# [
#   WAFER_ID: {good:, bad:},
#   TOOL_NAME:'1F.GEO',
# ]
# pipeline _pre_process 固定 传参 组合

def commonality_analysis(df_run, grpby_list):
    """
    param df_run: 数据预处理后的数据
    return: 共性分析后的结果， 返回bad wafer前十的组合
    """
    grps = (df_run.groupBy(grpby_list)
            .agg(countDistinct('WAFER_ID').alias('wafer_count'),
                 countDistinct('WAFER_ID', when(df_run['label'] == 0, 1)).alias('good_num'),
                 countDistinct('WAFER_ID', when(df_run['label'] == 1, 1)).alias('bad_num'))
            .orderBy('bad_num', ascending=False))

    # 单站点+单腔室的情况
    if grps.count() == 1:
        return grps
    else:
        grps = grps.filter(grps['bad_num'] > 0)
        window_sep = Window().orderBy(col("bad_num").desc())
        ranked_df = grps.withColumn("rank", rank().over(window_sep))
        grpss = ranked_df.filter(col("rank") <= 10).drop("rank")
        return grpss



# common_res = commonality_analysis(df_run)


############################################################################
#################################获取样本数据#################################
############################################################################
def get_data_list(common_res, grpby_list, big_or_small='big'):
    """
    param common_res: 共性分析后的结果, 按照大样本或者小样本条件筛选出组合
    param grpby_list: 按照PRODG1+OPER_NO+TOOL_NAME分组或OPER_NO+TOOL_NAME分组
    param big_or_small: big或者small
    return: 对应组合的字典形式, 包在一个大列表中
    """
    assert big_or_small in ['big', 'small'], "只能选择big或者small, 请检查拼写"
    if big_or_small == 'big':
        good_bad_grps = common_res.filter("good_num >= 3 AND bad_num >= 3")
    else:
        good_bad_grps = common_res.filter("bad_num >= 1 AND wafer_count >=2")
    good_bad_grps = good_bad_grps.orderBy(col("bad_num").desc(), col("wafer_count").desc(),
                                          col("good_num").desc()).limit(5)

    if 'PRODG1' in grpby_list:
        data_list = good_bad_grps['PRODG1', 'OPER_NO', 'TOOL_NAME'].collect()
    else:
        data_list = good_bad_grps['OPER_NO', 'TOOL_NAME'].collect()

    data_dict_list = [row.asDict() for row in data_list]
    return data_dict_list



def get_train_data(df_run, data_dict_list):
    """
    param df_run: 数据预处理后的数据
    param data_dict: 筛选后的字典结果
    return: 从原始数据中过滤出真正用来建模的组合数据
    """
    if len(data_dict_list[0]) == 3:
        prod, oper, tool = data_dict_list[0]['PRODG1'], data_dict_list[0]['OPER_NO'], data_dict_list[0]['TOOL_NAME']
        df_s = df_run.filter("PRODG1 == '{}' AND OPER_NO == '{}' AND TOOL_NAME == '{}'".format(prod, oper, tool))
        for i in range(1, len(data_dict_list)):
            prod, oper, tool = data_dict_list[i]['PRODG1'], data_dict_list[i]['OPER_NO'], data_dict_list[i][
                'TOOL_NAME']
            df_m = df_run.filter(
                "PRODG1 == '{}' AND OPER_NO == '{}' and TOOL_NAME == '{}'".format(prod, oper, tool))
            df_s = df_s.union(df_m)
    else:
        oper, tool = data_dict_list[0]['OPER_NO'], data_dict_list[0]['TOOL_NAME']
        df_s = df_run.filter("OPER_NO == '{}' AND TOOL_NAME == '{}'".format(oper, tool))
        for i in range(1, len(data_dict_list)):
            oper, tool = data_dict_list[i]['OPER_NO'], data_dict_list[i]['TOOL_NAME']
            df_m = df_run.filter("OPER_NO == '{}' and TOOL_NAME == '{}'".format(oper, tool))
            df_s = df_s.union(df_m)
    return df_s



# data_dict_bs = get_data_dict_big_sample(common_res)
# df_run_bs = get_train_data_big_sample(df_run, data_dict_bs)
# print(data_dict_bs)
# print(df_run_bs.count())

############################################################################
#########################获取传入的整个数据中的所有bad_wafer个数##################
############################################################################
def get_all_bad_wafer_num(df):
    """
    param df: 筛选后的数据
    return: 数据中所有bad_wafer的数量
    """
    return df.filter("label == 1").select('WAFER_ID').distinct().count()


# print(get_all_bad_wafer_num(df_run_bs))


############################################################################
#####################对good>=3和bad>=3的数据，用rf建模##########################
############################################################################
def get_pivot_table(df, by):
    """
    param df: 组合的数据
    param by: 分组字段
    return: 表格透视后的结果
    """
    if len(by) == 3:
        df_pivot = df.dropna(axis=0).pivot_table(index=['WAFER_ID', 'label'],
                                                 columns=['OPER_NO', 'TOOL_NAME', 'parametric_name', 'PRODG1'],
                                                 values=['STATISTIC_RESULT'])
    else:
        df_pivot = df.dropna(axis=0).pivot_table(index=['WAFER_ID', 'label'],
                                                 columns=['OPER_NO', 'TOOL_NAME', 'parametric_name'],
                                                 values=['STATISTIC_RESULT'])
    df_pivot.columns = df_pivot.columns.map('#'.join)
    df_pivot = df_pivot.fillna(df_pivot.mean()).reset_index(drop=False)
    return df_pivot



def fit_rf_big_sample(df, by):
    """
    param df: 大样本组合的数据
    param by: 分组字段
    return: RandomForest建模后的结果
    """
    schema_all = StructType([
        StructField("PRODG1", StringType(), True),
        StructField("OPER_NO", StringType(), True),
        StructField("TOOL_NAME", StringType(), True),
        StructField("bad_wafer", IntegerType(), True),
        StructField("roc_auc_score", FloatType(), True),
        StructField("features", StringType(), True),
        StructField("importance", FloatType(), True)])

    @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
    def get_model_result(df_run):
        # 表格透视
        df_pivot = get_pivot_table(df=df_run, by=by)

        # 定义自变量和因变量
        X_train = df_pivot[df_pivot.columns.difference(['WAFER_ID', 'label']).tolist()]
        y_train = df_pivot[['label']]

        z_ratio = y_train.value_counts(normalize=True)
        good_ratio = z_ratio[0]
        bad_ratio = z_ratio[1]
        if abs(good_ratio - bad_ratio) > 0.7:
            undersampler = ClusterCentroids(random_state=101)
            X_train, y_train = undersampler.fit_resample(X_train, y_train)

        # 网格搜索
        pipe = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(random_state=99))])
        param_grid = {'model__n_estimators': [*range(50, 100, 10)],
                      'model__max_depth': [*range(10, 50, 10)]}
        grid = GridSearchCV(estimator=pipe, scoring='roc_auc', param_grid=param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train.values, y_train.values.ravel())
        roc_auc_score_ = grid.best_score_

        # 特征重要度、结果汇总
        small_importance_res = pd.DataFrame({
            'features': X_train.columns,
            'importance': grid.best_estimator_.steps[2][1].feature_importances_}).sort_values(by='importance',
                                                                                              ascending=False)
        if len(by) == 3:
            small_sample_res = pd.DataFrame({
                'PRODG1': df_run['PRODG1'].unique(),
                'OPER_NO': df_run['OPER_NO'].unique(),
                'TOOL_NAME': df_run['TOOL_NAME'].unique(),
                'bad_wafer': sum(df_pivot['label']),
                'roc_auc_score': roc_auc_score_})
        else:
            PRODG1 = 'grplen2'
            small_sample_res = pd.DataFrame({
                'PRODG1': PRODG1,
                'OPER_NO': df_run['OPER_NO'].unique(),
                'TOOL_NAME': df_run['TOOL_NAME'].unique(),
                'bad_wafer': sum(df_pivot['label']),
                'roc_auc_score': roc_auc_score_})
        return pd.concat([small_importance_res, small_sample_res])

    return df.groupby(by).apply(get_model_result)



# res = fit_rf_big_sample(df=df_run_bs, by=['PRODG1', 'OPER_NO', 'TOOL_NAME'])


#####################################################################################
#########################对good>=3和bad>=3建模后的结果进行整合############################
#####################################################################################
def split_score_big_sample(df, by):
    """
    param df: RandomForest建模后的结果
    param by: 分组字段
    return: roc_auc分数结果
    """
    schema_all = StructType([StructField("PRODG1", StringType(), True),
                             StructField("OPER_NO", StringType(), True),
                             StructField("TOOL_NAME", StringType(), True),
                             StructField("bad_wafer", IntegerType(), True),
                             StructField("roc_auc_score", FloatType(), True)])

    @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
    def get_result(model_results):
        sample_res = model_results[['PRODG1', 'OPER_NO', 'TOOL_NAME', 'bad_wafer', 'roc_auc_score']].dropna(axis=0)
        sample_res = sample_res[sample_res['roc_auc_score'] > 0.6]
        return sample_res

    return df.groupby(by).apply(get_result)



def split_features(df, index) -> str:
    """
    param df: RandomForest建模后的feature_importance_table
    param index: 顺序值
    return: 字段属性值
    """
    return df['features'].apply(lambda x: x.split('#')[index])


def get_split_feature_importance_table(df, by):
    """
    param df: RandomForest建模后的feature_importance_table
    param by: OPER_NO+TOOL_NAME+PRODG1或者OPER_NO+TOOL_NAME
    return: 分裂features后的表
    """
    df['STATISTIC_RESULT'] = split_features(df, 0)
    df['OPER_NO'] = split_features(df, 1)
    df['TOOL_NAME'] = split_features(df, 2)
    df['parametric_name'] = split_features(df, 3)
    df['step'] = split_features(df, 4)
    df['stats'] = split_features(df, 5)

    if 'PRODG1' in by:
        df['PRODG1'] = split_features(df, 6)
    else:
        df = df.assign(PRODG1='grplen2')

    df = df.drop(['features', 'STATISTIC_RESULT'], axis=1).reset_index(drop=True)
    return df



def add_feature_stats(df):
    """
    param df: 经过处理后的feature_importance_table
    return: 新增一列，含有参数的所有统计特征:feature_stats
    """
    feature_stats = df.groupby(['PRODG1', 'OPER_NO', 'TOOL_NAME', 'parametric_name', 'step'])[
        'stats'].unique().reset_index()
    feature_stats['stats'] = [feature_stats['stats'].iloc[i].tolist() for i in range(len(feature_stats))]
    feature_stats['stats'] = feature_stats['stats'].apply(lambda x: "#".join(x))
    feature_stats = feature_stats.assign(
        parametric_name=lambda x: x['parametric_name'] + str('#') + x['step']).drop('step', axis=1)
    return feature_stats



def split_calculate_features_big_sample(df, by):
    """
    param df: RandomForest建模后的结果
    param by: 分组字段
    return: features和importance结果
    """
    schema_all = StructType([
        StructField("PRODG1", StringType(), True),
        StructField("OPER_NO", StringType(), True),
        StructField("TOOL_NAME", StringType(), True),
        StructField("parametric_name", StringType(), True),
        StructField("importance", FloatType(), True),
        StructField("stats", StringType(), True)])

    @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
    def get_result(model_results):
        # 先从随机森林的模型结果中取出包含features和importance的dataframe
        feature_importance_table = model_results[['features', 'importance']].dropna(axis=0)
        # 分裂features
        feature_importance_res_split = get_split_feature_importance_table(feature_importance_table, by)

        # 去除importance为0的组合
        feature_importance_res_split_drop = feature_importance_res_split.query("importance > 0").reset_index(
            drop=True)

        # 取每一种组合结果的前60%或者100%
        feature_importance_res_split_nlargest = (
            feature_importance_res_split_drop.groupby(by=['PRODG1', 'OPER_NO', 'TOOL_NAME'])
            .apply(lambda x: x.nlargest(int(x.shape[0] * 0.6), 'importance') if x.shape[0] > 1 else x.nlargest(
                int(x.shape[0] * 1), 'importance'))
            .reset_index(drop=True))
        # 新增一列，含有参数的所有统计特征:feature_stats
        feature_stats = add_feature_stats(feature_importance_res_split_drop)

        # 对同一种组合里的同一个参数进行求和:feature_importance_groupby
        feature_importance_groupby = (
            feature_importance_res_split_nlargest.groupby(['PRODG1', 'OPER_NO', 'TOOL_NAME',
                                                           'parametric_name', 'step'])[
                'importance'].sum().reset_index())
        feature_importance_groupby = feature_importance_groupby.assign(
            parametric_name=lambda x: x['parametric_name'] + str('#') + x['step']).drop('step', axis=1)

        # feature_stats和feature_importance_groupby连接
        grpby_stats = pd.merge(feature_stats, feature_importance_groupby,
                               on=['PRODG1', 'OPER_NO', 'TOOL_NAME', 'parametric_name']).dropna().reset_index(
            drop=True)
        return grpby_stats
    return df.groupby(by).apply(get_result)



def get_finall_results_big_sample(s_res, f_res, bad_wafer_num):
    """
    param s_res: roc_auc分数结果
    param f_res: features和importance结果
    param bad_wafer_num: 数据中所有bad_wafer的数量
    return: 最后的建模结果
    """
    # feature_importance_groupby和sample_res连接
    roc_auc_score_all = s_res.agg({"roc_auc_score": "sum"}).collect()[0][0]
    s_res = s_res.withColumn("roc_auc_score_ratio", col("roc_auc_score") / roc_auc_score_all)
    s_res = s_res.withColumn("bad_ratio", col("bad_wafer") / bad_wafer_num)

    df_merge = s_res.join(f_res, on=['PRODG1', 'OPER_NO', 'TOOL_NAME'], how='left')
    df_merge = df_merge.withColumn('weight_original',
                                   col('roc_auc_score_ratio') * col('bad_ratio') * col('importance'))

    # 最后再次进行一次归一化
    weight_all = df_merge.agg({"weight_original": "sum"}).collect()[0][0]
    df_merge = df_merge.withColumn("weight", col("weight_original") / weight_all)

    df_merge = df_merge.select(['PRODG1', 'OPER_NO', 'TOOL_NAME',
                                'parametric_name', 'weight', 'stats']).orderBy('weight', ascending=False)
    return df_merge



#####################################################################################
#############################将建模后的结果增加特定的列####################################
#####################################################################################
def add_certain_column(df, by, request_id):
    """
    param df: 最后的建模结果
    param by: 分组字段, 手动增加一列add
    param request_id: 传入的request_id
    return: 最后的建模结果增加特定的列
    """
    schema_all = StructType([
        StructField("PRODG1", StringType(), True),
        StructField("OPER_NO", StringType(), True),
        StructField("TOOL_NAME", StringType(), True),
        StructField("stats", StringType(), True),
        StructField("parametric_name", StringType(), True),
        StructField("weight", FloatType(), True),
        StructField("request_id", StringType(), True),
        StructField("weight_percent", FloatType(), True),
        StructField("index_no", IntegerType(), True)])

    @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
    def get_result(final_res):
        final_res['weight'] = final_res['weight'].astype(float)
        final_res = final_res.query("weight > 0")
        final_res['request_id'] = request_id
        final_res['weight_percent'] = final_res['weight'] * 100
        final_res = final_res.sort_values('weight', ascending=False)
        final_res['index_no'] = [i + 1 for i in range(len(final_res))]
        final_res = final_res.drop('add', axis=1)
        # final_res['parametric_name'] = final_res['parametric_name'].str.replace("_", "+")
        final_res['PRODG1'] = final_res['PRODG1'].apply(lambda x: None if x == 'grplen2' else x)
        return final_res
    return df.groupby(by).apply(get_result)



##########################################################################################
#######################################对bad>=1的数据，用pca建模##############################
##########################################################################################
def fit_pca_small_sample(df, by):
    """
    param df: 小样本组合的数据
    param by: 分组字段
    return: PCA建模后的结果
    """
    schema_all = StructType([StructField("PRODG1", StringType(), True),
                             StructField("OPER_NO", StringType(), True),
                             StructField("TOOL_NAME", StringType(), True),
                             StructField("features", StringType(), True),
                             StructField("importance", FloatType(), True),
                             StructField("bad_wafer", IntegerType(), True)])

    @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
    def get_model_result(df_run):
        df_pivot = get_pivot_table(df=df_run, by=by)

        # 定义自变量
        x_train = df_pivot[df_pivot.columns.difference(['WAFER_ID', 'label']).tolist()]

        # 建立模型
        model = pca(n_components=0.8, verbose=None)
        results = model.fit_transform(x_train)
        res_top = results['topfeat']
        res_top_select = res_top[res_top['type'] == 'best'][['feature', 'loading']]
        res_top_select['importance'] = abs(res_top_select['loading'])
        res_top_select = res_top_select.rename(columns={'feature': 'features'})
        res_top_select = res_top_select.drop("loading", axis=1)

        # 增加一些字段信息
        res_top_select['bad_wafer'] = sum(df_pivot['label'])
        res_top_select['OPER_NO'] = df_run['OPER_NO'].values[0]
        res_top_select['TOOL_NAME'] = df_run['TOOL_NAME'].values[0]
        if len(by) == 3:
            res_top_select['PRODG1'] = df_run['PRODG1'].values[0]
        else:
            res_top_select['PRODG1'] = 'grplen2'
        return res_top_select
    return df.groupby(by).apply(get_model_result)



#####################################################################################
##################################对bad>=1建模后的结果进行整合############################
#####################################################################################
def split_calculate_features_small_sample(df, by):
    """
    param df: PCA建模后的结果
    param by: 分组字段
    return: features和importance结果
    """
    schema_all = StructType([StructField("PRODG1", StringType(), True),
                             StructField("OPER_NO", StringType(), True),
                             StructField("TOOL_NAME", StringType(), True),
                             StructField("parametric_name", StringType(), True),
                             StructField("importance", FloatType(), True),
                             StructField("bad_wafer", FloatType(), True),
                             StructField("stats", StringType(), True)])

    @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
    def get_result(model_results):
        feature_importance_table = model_results[['features', 'importance', 'bad_wafer']].dropna(axis=0)
        # 分裂features
        feature_importance_res_split = get_split_feature_importance_table(feature_importance_table, by)

        # 新增一列，含有参数的所有统计特征:feature_stats
        feature_stats = add_feature_stats(feature_importance_res_split)

        # 对同一种组合里的同一个参数进行求和:feature_importance_groupby
        feature_importance_groupby = (
            feature_importance_res_split.groupby(['PRODG1', 'OPER_NO', 'TOOL_NAME', 'bad_wafer',
                                                  'parametric_name', 'step'])['importance'].sum().reset_index())
        feature_importance_groupby = feature_importance_groupby.assign(
            parametric_name=lambda x: x['parametric_name'] + str('#') + x['step']).drop('step', axis=1)

        # feature_stats和feature_importance_groupby连接
        grpby_stats = pd.merge(feature_stats, feature_importance_groupby,
                               on=['PRODG1', 'OPER_NO', 'TOOL_NAME', 'parametric_name']).dropna().reset_index(
            drop=True)
        return grpby_stats
    return df.groupby(by).apply(get_result)



def get_finall_results_small_sample(f_res, bad_wafer_num):
    """
    param s_res: roc_auc分数结果
    param f_res: features和importance结果
    param bad_wafer_num: 数据中所有bad_wafer的数量
    return: 最后的建模结果
    """
    f_res = f_res.withColumn("bad_ratio", col("bad_wafer") / bad_wafer_num)
    df_merge = f_res.withColumn('weight_original', col('importance') * col('bad_ratio'))

    # 最后再次进行一次归一化
    weight_all = df_merge.agg({"weight_original": "sum"}).collect()[0][0]
    df_merge = df_merge.withColumn("weight", col("weight_original") / weight_all)

    df_merge = df_merge.select(['PRODG1', 'OPER_NO', 'TOOL_NAME',
                                'parametric_name', 'weight', 'stats']).orderBy('weight', ascending=False)
    return df_merge



##########################################################################################
#######################################健壮性检查###########################################
##########################################################################################
def fit_big_data_model(df_run, data_dict_list_bs, grpby_list, request_id):
    df1 = None
    final_res_add_columns = None
    df_run_bs = get_train_data(df_run, data_dict_list_bs)
    if df_run_bs.count() == 0:
        msg = '数据库中暂无此类数据!'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, final_res_add_columns

    # 4. 获取所有bad wafer数量
    bad_wafer_num_big_sample = get_all_bad_wafer_num(df_run_bs)
    if bad_wafer_num_big_sample < 3:
        msg = '数据库中实际BAD_WAFER数量小于3片, 请提供更多的BAD_WAFER数量!'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, final_res_add_columns

    # 5. 对挑选出的大样本数据进行建模
    res = fit_rf_big_sample(df=df_run_bs, by=grpby_list)
    if res.count() == 0:
        msg = '算法内部暂时异常!'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, final_res_add_columns

    # 6. 将建模结果进行整合
    s_res = split_score_big_sample(df=res, by=['PRODG1', 'OPER_NO', 'TOOL_NAME'])
    if s_res.count() == 0:
        msg = '算法运行评分结果较低, 暂无输出, 建议增加BAD_WAFER数量'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, final_res_add_columns

    f_res = split_calculate_features_big_sample(df=res, by=grpby_list)
    if f_res.count() == 0:
        msg = '算法结果求和暂时异常'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, final_res_add_columns

    model_res_bs = get_finall_results_big_sample(s_res=s_res, f_res=f_res, bad_wafer_num=bad_wafer_num_big_sample)
    if model_res_bs.count() == 0:
        msg = '算法结果拼接暂时异常'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, final_res_add_columns

    # 7. 增加特定的列
    final_res_bs = model_res_bs.withColumn('add', lit(0))
    final_res_add_columns = add_certain_column(df=final_res_bs, by='add', request_id=request_id)
    if final_res_add_columns.count() == 0:
        msg = '算法结果增加列暂时异常'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, final_res_add_columns
    else:
        return df1, final_res_add_columns


def fit_small_data_model(df_run, common_res, grpby_list, request_id):
    df1 = None
    final_res_add_columns = None

    data_dict_list_ss = get_data_list(common_res=common_res, grpby_list=grpby_list, big_or_small='small')
    print("data_dict_list_ss:", data_dict_list_ss)
    if len(data_dict_list_ss) == 0:
        msg = '该查询条件下数据库中实际BAD_WAFER数量为0, 无法分析'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, final_res_add_columns

    df_run_ss = get_train_data(df_run=df_run, data_dict_list=data_dict_list_ss)
    if df_run_ss.count() == 0:
        msg = '数据库中暂无此类数据!'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, final_res_add_columns

    bad_wafer_num_small_sample = get_all_bad_wafer_num(df_run_ss)
    if bad_wafer_num_small_sample < 1:
        msg = '该查询条件下数据库中实际BAD_WAFER数量小于1片, 请提供更多的BAD_WAFER数量!'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, final_res_add_columns

    res = fit_pca_small_sample(df=df_run_ss, by=grpby_list)
    if res.count() == 0:
        msg = '算法内部暂时异常!'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, final_res_add_columns

    f_res = split_calculate_features_small_sample(df=res, by=grpby_list)
    if f_res.count() == 0:
        msg = '算法结果求和暂时异常'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, final_res_add_columns

    model_res_ss = get_finall_results_small_sample(f_res=f_res, bad_wafer_num=bad_wafer_num_small_sample)
    if model_res_ss.count() == 0:
        msg = '算法结果拼接暂时异常'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, final_res_add_columns

    final_res_ss = model_res_ss.withColumn('add', lit(0))
    final_res_add_columns = add_certain_column(df=final_res_ss, by='add', request_id=request_id)
    if final_res_add_columns.count() == 0:
        msg = '算法结果增加列暂时异常'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, final_res_add_columns
    else:
        return df1, final_res_add_columns


#####################################################################################
################################将最后的结果写回数据库####################################
#####################################################################################
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


##########################################################################################
#######################################正式调用以上函数#######################################
##########################################################################################

# 1. 解析json 为字典， df1为kafka输入的结果数据，获取到parse_dict, request_id, grpby_list
df2 = df1.toPandas()
parse_dict, request_id, grpby_list = get_some_info(df2)
# print(type(parse_dict))
# print(grpby_list)

# 2. 从kafka 关键字映射都具体数据源中的字段,没有的可以删除
keyword_map_from_json_to_table: dict = {
    "prodg1": "PRODG1",
    "waferId": "WAFER_ID",
    "dateRange": "START_TIME",
    "productId": "PRODUCT_ID",
    "operNo": "OPER_NO",
    "eqp": "EQP_NAME",
    "tool": "TOOL_NAME",
    "lot": "LOT_ID",
    "recipeName": "RECIPE_NAME"}

# 3. 获取查询条件list
select_condition_list = parse_dict

# 4. 指定查询表名, 根据实际情况需要修改
table_name = "etl.DWD_POC_CASE_FD_UVA_DATA_TEST"

# 主程序
try:
    # 从数据库中获取数据
    df1 = get_data_from_doris(select_condition_list=select_condition_list, table_name=table_name)
    if df1.count() == 0:
        msg = '解析SQL获取数据异常: 数据库中可能没有数据!'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        raise ValueError

    # 1. 数据预处理
    print(df1.count())
    df_run = _pre_process(df1)
    print(df_run.count())
    if df_run.count() == 0:
        msg = '该条件下数据库中暂无数据，请检查！'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        raise ValueError

    # 2. 进行共性分析
    common_res = commonality_analysis(df_run, grpby_list)
    common_res.show()
    if common_res.count() == 0:
        msg = '共性分析结果异常!'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        raise ValueError

    # 3. 挑选出数据：bad和good要同时大于3
    data_dict_list_bs = get_data_list(common_res, grpby_list, big_or_small='big')
    print("data_dict_list_bs:", data_dict_list_bs)
    if len(data_dict_list_bs) != 0:
        df1, final_res_add_columns = fit_big_data_model(df_run, data_dict_list_bs, grpby_list, request_id)
    # 如果不大于3
    else:
        print("小样本算法调用")
        df1, final_res_add_columns = fit_small_data_model(df_run, common_res, grpby_list, request_id)

    # final_res_add_columns 是最后的结果，要写回数据库
    ddd = final_res_add_columns.toPandas()
    user ="root"
    host = "10.52.199.81"
    password = "Nexchip%40123"
    db = "etl"
    port = 9030
    engine = create_engine("mysql+pymysql://{user}:{password}@{host}:{port}/{db}".format(user = user,
                                                                                        password = password,
                                                                                        host = host,
                                                                                        port = port,
                                                                                        db = db))
    doris_stream_load_from_df(ddd, engine, "results")

    # 最终成功的话，就会输出下面这条
    df_kafka = pd.DataFrame({"code": 0, "msg": "运行成功", "requestId": request_id}, index=[0])
    df1 = spark.createDataFrame(df_kafka)

except ValueError as ve:
    pass

except Exception as e:
    df_kafka = pd.DataFrame({"code": 1, "msg": f"主程序发生异常: {str(e)}", "requestId": request_id}, index=[0])
    df1 = spark.createDataFrame(df_kafka)


