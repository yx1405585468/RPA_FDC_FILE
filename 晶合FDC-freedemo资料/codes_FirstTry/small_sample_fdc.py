import pyspark.pandas as ps
from pca import pca
from pyspark.sql.window import Window
from pyspark.sql.functions import pandas_udf, PandasUDFType, countDistinct, when, col, rank, lit
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType


def commonality_analysis(df_run):
    """
    :param df_run: 从数据库中读取出来的某个CASE的数据
    :return: 共性分析后的结果，返回bad wafer前十的组合
    """
    grps = (df_run.groupBy(['PRODG1', 'OPER_NO', 'TOOL_NAME']).agg(countDistinct('WAFER_ID').alias('wafer_count'),
                                                                   countDistinct('WAFER_ID', when(df_run['label'] == 0,
                                                                                                  1)).alias('good_num'),
                                                                   countDistinct('WAFER_ID', when(df_run['label'] == 1,
                                                                                                  1)).alias('bad_num'))
            .orderBy('bad_num', ascending=False))
    # 单站点+单腔室的情况
    if grps.count() == 1:
        return grps
    else:
        grps_filter = grps.filter(grps['bad_num'] > 0)
        window_sep = Window().orderBy(col('bad_num').desc())
        ranked_df = grps_filter.withColumn("rank", rank().over(window_sep))
        grpss = ranked_df.filter(col('rank') <= 10).drop("rank")
    return grpss


def get_all_bad_wafer_num(df):
    """
    :param df: 从数据库中读取出来的某个CASE的数据
    :return: 这个CASE里的bad wafer数量
    """
    return df.filter("label == 1").select('WAFER_ID').distinct().count()


def get_data_dict_small_sample(common_res):
    """
    :param common_res: 共性分析的结果，按照不满足good_num > 3 AND bad_num > 3的条件，筛选出组合
    :return: 对应组合的字典形式
    """
    good_bad_grps = common_res.filter("NOT (good_num > 3 AND bad_num > 3)")
    data_list = good_bad_grps['PRODG1', 'OPER_NO', 'TOOL_NAME'].collect()
    data_dict = [row.asDict() for row in data_list]
    return data_dict


def get_train_data_small_sample(df_run, data_dict):
    """
    :param df_run: 从数据库中读取出来的某个CASE的数据
    :param data_dict: 筛选后的字典结果(get_data_dict_small_sample)
    :return: 从原始数据中过滤出小样本组合的数据
    """
    prod, oper, tool = data_dict[0]['PRODG1'], data_dict[0]['OPER_NO'], data_dict[0]['TOOL_NAME']
    df_s = df_run.filter("PRODG1 == '{}' AND OPER_NO == '{}' AND TOOL_NAME == '{}'".format(prod, oper, tool))

    for i in range(1, len(data_dict)):
        prod, oper, tool = data_dict[i]['PRODG1'], data_dict[i]['OPER_NO'], data_dict[i]['TOOL_NAME']
        df_m = df_run.filter("PRODG1 == '{}' AND OPER_NO == '{}' AND TOOL_NAME == '{}'".format(prod, oper, tool))
        df_s = df_s.union(df_m)
    return df_s


def fit_pca_small_sample(df, by):
    """
    :param df: 小样本组合的数据
    :param by: 分组字段
    :return: PCA建模后的结果
    """
    schema_all = StructType(
        [StructField("feature", StringType(), True),
         StructField("loading", FloatType(), True),
         StructField("bad_wafer", IntegerType(), True)])

    @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
    def get_model_results(df_run):
        df_pivot = df_run.dropna(axis=0).pivot_table(index=['WAFER_ID', 'label'],
                                                     columns=['PRODG1', 'OPER_NO', 'TOOL_NAME', 'parametric_name'],
                                                     values=['mean', 'std', 'min', 'q25', 'median', 'q75', 'max',
                                                             'range1'])
        df_pivot.columns = df_pivot.columns.map('#'.join)
        df_pivot = df_pivot.fillna(df_pivot.mean()).reset_index(drop=False)

        # 定义自变量
        x_train = df_pivot[df_pivot.columns.difference(['WAFER_ID', 'label']).tolist()]

        # 建立模型
        model = pca(n_components=0.8, verbose=None)
        results = model.fit_transform(x_train)
        res_top = results['topfeat']
        res_top_select = res_top[res_top['type'] == 'best'][['feature', 'loading']]
        res_top_select['loading'] = abs(res_top_select['loading'])
        res_top_select['bad_wafer'] = sum(df_pivot['label'])
        return res_top_select

    return df.groupby(by).apply(get_model_results)


def split_feature_importance_small_sample(df, by, all_bad_num):
    """
    :param df: PCA建模后的数据
    :param by: 手动增加的一个分组字段，grp=1
    :param all_bad_num: 整个数据集下的所有bad wafer数量
    :return: 最后的结果
    """
    schema_all = StructType([StructField("PRODG1", StringType(), True),
                             StructField("OPER_NO", StringType(), True),
                             StructField("TOOL_NAME", StringType(), True),
                             StructField("parametric_name", StringType(), True),
                             StructField("weight", FloatType(), True)])

    @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
    def get_split_results(feature_importance_table):
        # 将feature_importance_res结果拆开
        feature_importance_table['STATISTICAL_RESULT'] = feature_importance_table['feature'].apply(lambda x: x.split("#")[0])
        feature_importance_table['PRODG1'] = feature_importance_table['feature'].apply(lambda x: x.split("#")[1])
        feature_importance_table['OPER_NO'] = feature_importance_table['feature'].apply(lambda x: x.split("#")[2])
        feature_importance_table['TOOL_NAME'] = feature_importance_table['feature'].apply(lambda x: x.split("#")[3])
        feature_importance_table['parametric_name'] = feature_importance_table['feature'].apply(lambda x: x.split("#")[4])
        feature_importance_table['weight'] = (feature_importance_table['bad_wafer'] / all_bad_num) * \
                                              feature_importance_table['loading']
        # 对同一种组合里的同一个参数进行求和
        feature_importance_groupby = (feature_importance_table.groupby(['PRODG1', 'OPER_NO', 'TOOL_NAME', 'parametric_name'])['weight']
                                                              .mean()
                                                              .reset_index())
        return feature_importance_groupby

    return df.groupby(by).apply(get_split_results)


if __name__ == "__main__":
    import warnings
    import pandas as pd
    import os
    warnings.filterwarnings('ignore')

    # local模式
    # import findspark
    # findspark.init()
    # spark = SparkSession \
    #     .builder \
    #     .appName("ywj") \
    #     .config('spark.sql.session.timeZone', 'Asia/Shanghai') \
    #     .master("local[*]") \
    #     .getOrCreate()

    # spark集群模式
    from pyspark.sql import SparkSession

    os.environ['PYSPARK_PYTHON'] = '/usr/local/python-3.9.13/bin/python3'
    spark = SparkSession.builder \
        .appName("pandas_udf") \
        .config('spark.sql.session.timeZone', 'Asia/Shanghai') \
        .config("spark.scheduler.mode", "FAIR") \
        .config('spark.driver.memory', '1024m') \
        .config('spark.driver.cores', '3') \
        .config('spark.executor.memory', '1024m') \
        .config('spark.executor.cores', '1') \
        .config('spark.cores.max', '2') \
        .config('spark.driver.host', '192.168.22.28') \
        .master("spark://192.168.12.47:7077,192.168.12.48:7077") \
        .getOrCreate()

    # 读取数据
    df_run_ = pd.read_csv('C:/Users/yang.wenjun/Desktop/晶合FDC-freedemo资料/df_run.csv', index_col=0)
    df_run_spark = ps.from_pandas(df_run_).to_spark()

    # 共性分析
    oper_tool_groups = commonality_analysis(df_run_spark)
    oper_tool_groups.show()

    # 获取小样本组合下的数据
    small_data_dict = get_data_dict_small_sample(oper_tool_groups)
    print(small_data_dict)
    df_run_spark_small_sample = get_train_data_small_sample(df_run_spark, small_data_dict)
    print(df_run_spark_small_sample.count())

    # 对小样本组合下的数据，建模
    res = fit_pca_small_sample(df=df_run_spark_small_sample, by=['OPER_NO', 'TOOL_NAME'])
    res.show()

    # 获取该case中的所有bad_wafer数量
    all_bad_num_ = get_all_bad_wafer_num(df_run_spark_small_sample)
    print(all_bad_num_)

    # 将建模后的数据拆开得到最后的结果, 先新增一列grp作为分组字段
    res = res.withColumn('grp', lit(1))
    feature_res = split_feature_importance_small_sample(df=res, by='grp', all_bad_num=all_bad_num_)
    feature_res.show()
