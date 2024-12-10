import pandas as pd
import pyspark.pandas as ps
from imblearn.under_sampling import ClusterCentroids
from pyspark.sql.window import Window
from pyspark.sql.functions import pandas_udf, PandasUDFType, countDistinct, when, col, rank
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def get_data_dict_big_sample(common_res):
    """
    :param common_res: 共性分析的结果，按照满足good_num > 3 AND bad_num > 3的条件，筛选出组合
    :return: 对应组合的字典形式
    """
    good_bad_grps = common_res.filter("good_num > 3 AND bad_num > 3")
    data_list = good_bad_grps['PRODG1', 'OPER_NO', 'TOOL_NAME'].collect()
    data_dict = [row.asDict() for row in data_list]
    return data_dict


def get_train_data_big_sample(df_run, data_dict):
    """
    :param df_run: 从数据库中读取出来的某个CASE的数据
    :param data_dict: 筛选后的字典结果(get_data_dict_big_sample)
    :return: 从原始数据中过滤出大样本组合的数据
    """
    prod, oper, tool = data_dict[0]['PRODG1'], data_dict[0]['OPER_NO'], data_dict[0]['TOOL_NAME']
    df_s = df_run.filter("PRODG1 == '{}' AND OPER_NO == '{}' AND TOOL_NAME == '{}'".format(prod, oper, tool))

    for i in range(1, len(data_dict)):
        prod, oper, tool = data_dict[i]['PRODG1'], data_dict[i]['OPER_NO'], data_dict[i]['TOOL_NAME']
        df_m = df_run.filter("PRODG1 == '{}' AND OPER_NO == '{}' AND TOOL_NAME == '{}'".format(prod, oper, tool))
        df_s = df_s.union(df_m)
    return df_s


def fit_rf_big_sample(df, by):
    """
    :param df: 大样本组合的数据
    :param by: 分组字段
    :return: RandomForest建模后的结果
    """
    schema_all = StructType(
        [StructField("PRODG1", StringType(), True),
         StructField("OPER_NO", StringType(), True),
         StructField("TOOL_NAME", StringType(), True),
         StructField("bad_wafer", IntegerType(), True),
         StructField("roc_auc_score", FloatType(), True),
         StructField("features", StringType(), True),
         StructField("importance", FloatType(), True)])

    @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
    def get_model_results(df_run):
        df_pivot = df_run.dropna(axis=0).pivot_table(index=['WAFER_ID', 'label'],
                                                     columns=['PRODG1', 'OPER_NO', 'TOOL_NAME', 'parametric_name'],
                                                     values=['mean', 'std', 'min', 'q25', 'median', 'q75', 'max',
                                                             'range1'])
        df_pivot.columns = df_pivot.columns.map('#'.join)
        df_pivot = df_pivot.fillna(df_pivot.mean()).reset_index(drop=False)

        # 定义自变量和因变量
        x_train = df_pivot[df_pivot.columns.difference(['WAFER_ID', 'label']).tolist()]
        y_train = df_pivot[['label']]

        z_ratio = y_train.value_counts(normalize=True)
        good_ratio, bad_ratio = z_ratio[0], z_ratio[1]
        if abs(good_ratio - bad_ratio) > 0.7:
            undersampler = ClusterCentroids(ramdom_state=2)
            x_train, y_train = undersampler.fit_resample(x_train, y_train)

        # 网格搜索&随机森林
        pipe = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
                   ('scaler', StandardScaler()),
                   ('model', RandomForestClassifier())])
        param_grid = {'model__n_estimators': [*range(50, 100, 10)], 'model__max_depth': [*range(10, 50, 10)]}
        grid = GridSearchCV(estimator=pipe, scoring='roc_auc', param_grid=param_grid, cv=3, n_jobs=-1)
        grid.fit(x_train.values, y_train.values.ravel())
        roc_auc_score_ = grid.best_score_

        # 特征重要度、结果汇总
        small_importance_res = pd.DataFrame({
            'features': x_train.columns,
            'importance': grid.best_estimator_.steps[2][1].feature_importances_}).sort_values(by='importance',
                                                                                              ascending=False)

        small_sample_res = pd.DataFrame({
            'PRODG1': df_run['PRODG1'].unique(),
            'OPER_NO': df_run['OPER_NO'].unique(),
            'TOOL_NAME': df_run['TOOL_NAME'].unique(),
            'bad_wafer': sum(df_pivot['label']),
            'roc_auc_score': roc_auc_score_})
        return pd.concat([small_importance_res, small_sample_res])

    return df.groupby(by).apply(get_model_results)


def split_score_big_sample(df, by):
    """
    :param df: RandomForest建模后的结果
    :param by: 分组字段
    :return: roc_auc分数结果
    """
    schema_all = StructType([StructField("PRODG1", StringType(), True),
                             StructField("OPER_NO", StringType(), True),
                             StructField("TOOL_NAME", StringType(), True),
                             StructField("bad_wafer", IntegerType(), True),
                             StructField("roc_auc_score", FloatType(), True)])

    @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
    def get_result(model_results):
        sample_res = model_results[['PRODG1', 'OPER_NO', 'TOOL_NAME', 'bad_wafer', 'roc_auc_score']].dropna(axis=0)
        sample_res = sample_res[sample_res['roc_auc_score'] > 0.7]
        return sample_res

    return df.groupby(by).apply(get_result)


def split_calculate_features_big_sample(df, by):
    """
    :param df: RandomForest建模后的结果
    :param by: 分组字段
    :return: features和importance结果
    """
    schema_all = StructType([StructField("PRODG1", StringType(), True),
                             StructField("OPER_NO", StringType(), True),
                             StructField("TOOL_NAME", StringType(), True),
                             StructField("parametric_name", StringType(), True),
                             StructField("importance", FloatType(), True)])

    @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
    def get_result(model_results):
        # 选出包含features和importance的数据
        feature_importance_table = model_results[['features', 'importance']].dropna(axis=0)
        # 对其进行split
        feature_importance_table['STATISTICAL_RESULT'] = feature_importance_table['features'].apply(
            lambda x: x.split("#")[0])
        feature_importance_table['PRODG1'] = feature_importance_table['features'].apply(lambda x: x.split("#")[1])
        feature_importance_table['OPER_NO'] = feature_importance_table['features'].apply(lambda x: x.split("#")[2])
        feature_importance_table['TOOL_NAME'] = feature_importance_table['features'].apply(lambda x: x.split("#")[3])
        feature_importance_table['parametric_name'] = feature_importance_table['features'].apply(
            lambda x: x.split("#")[4])
        feature_importance_res_split = feature_importance_table.drop(['features', 'STATISTICAL_RESULT'],
                                                                     axis=1).reset_index(drop=True)

        # 去除importance为0的组合
        feature_importance_res_split_drop = feature_importance_res_split[
            feature_importance_res_split['importance'] != 0].reset_index(drop=True)

        # 取每一种组合的前60%
        feature_importance_res_split_nlargest = (
            feature_importance_res_split_drop.groupby(by=['PRODG1', 'OPER_NO', 'TOOL_NAME'])
            .apply(lambda x: x.nlargest(int(x.shape[0] * 0.6), 'importance'))
            .reset_index(drop=True))
        # 对同一种组合里的同一种参数进行求和
        feature_importance_groupby = (feature_importance_res_split_nlargest.groupby(['PRODG1', 'OPER_NO', 'TOOL_NAME',
                                                                                    'parametric_name'])['importance']
                                                                          .sum().reset_index())
        return feature_importance_groupby

    return df.groupby(by).apply(get_result)


def get_final_results_big_sample(s_res, f_res, bad_wafer_num):
    """
    :param s_res: roc_auc分数结果
    :param f_res: features和importance结果
    :param bad_wafer_num: 该情况下数据中的所有bad_wafer数量
    :return: 最后的结果
    """
    df_merge = s_res.join(f_res, on=['PRODG1', 'OPER_NO', 'TOOL_NAME'], how='left')
    df_merge = df_merge.withColumn('bad_ratio', col('bad_wafer') / bad_wafer_num)
    df_merge = (df_merge.withColumn('weight', col('roc_auc_score') * col('importance') * col('bad_ratio'))
                .orderBy('weight', ascending=False)
                .select(['PRODG1', 'OPER_NO', 'TOOL_NAME', 'parametric_name', 'weight']))
    return df_merge


if __name__ == "__main__":
    import warnings
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

    # 获取大样本组合下的数据
    big_data_dict = get_data_dict_big_sample(oper_tool_groups)
    print(big_data_dict)
    df_run_spark_big_sample = get_train_data_big_sample(df_run_spark, big_data_dict)
    print(df_run_spark_big_sample.count())

    # 对大样本组合下的数据，建模
    res = fit_rf_big_sample(df=df_run_spark_big_sample, by=['PRODG1', 'OPER_NO', 'TOOL_NAME'])
    res.show()

    # 获取该case中的所有bad_wafer数量
    all_bad_num_ = get_all_bad_wafer_num(df_run_spark_big_sample)
    print(all_bad_num_)

    # 将建模后的数据拆开得到最后的结果, 先新增一列grp作为分组字段
    s_res_ = split_score_big_sample(df=res, by=['PRODG1', 'OPER_NO', 'TOOL_NAME'])
    f_res_ = split_calculate_features_big_sample(df=res, by=['PRODG1', 'OPER_NO', 'TOOL_NAME'])
    s_res_.show()
    f_res_.show()

    final_res = get_final_results_big_sample(s_res_, f_res_, all_bad_num_)
    final_res.show()
