import pandas as pd
import pyspark.sql.dataframe
import requests
import json

from pca import pca
from pyspark.sql.functions import pandas_udf, PandasUDFType, max, col, countDistinct, when, rank, lit
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from pyspark.sql.window import Window

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import ClusterCentroids

from functools import reduce
from pyspark.sql import DataFrame
from typing import Optional, List, Dict, Union
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, StructType, StructField
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from imblearn.under_sampling import ClusterCentroids
from pyspark.sql.functions import pandas_udf, PandasUDFType


# def get_some_info(df: pd.DataFrame):
#     """
#     从kafka消息读取需要的资料, 需要根据接口重新写
#     """
#     if len(df) > 0:
#         df = df.head(1)
#
#     request_id = df["requestId"].values[0]
#     request_params = df["requestParam"].values[0]
#     # 避免存在单引号，因为json引号只有双引号
#     request_params = request_params.replace('\'', "\"")
#     parse_dict = json.loads(request_params)
#     merge_prodg1 = parse_dict[0]['mergeProdg1']
#     try:
#         merge_operno = list(parse_dict[0]['mergeOperno'])
#     except KeyError:
#         merge_operno = None
#
#     if merge_prodg1 == '1':
#         grpby_list = ['OPER_NO', 'TOOL_NAME']
#     elif merge_prodg1 == '0':
#         grpby_list = ['PRODG1', 'OPER_NO', 'TOOL_NAME']
#     else:
#         raise ValueError
#     return parse_dict, request_id, grpby_list, merge_operno


def integrate_operno(df: pyspark.sql.dataframe, merge_operno_list: List[Dict[str, List[str]]]) -> pyspark.sql.dataframe:
    """
    Integrate the 'OPER_NO' column in the DataFrame based on the provided merge_operno_list.
    :param df: The input DataFrame.
    :param merge_operno_list: A list of dictionaries where each dictionary contains values to be merged.
           Example: [{'2F.CDS10_XX.TDS01': ['2F.CDS10', 'XX.TDS01']},
                     {'2F.CDS20_XX.CDS20': ['2F.CDS20', 'XX.CDS20']}]
    :return: DataFrame with 'OPER_NO' column integrated according to the merge_operno_list.
    """
    if merge_operno_list is not None and len(merge_operno_list) > 0:
        # Extract values from each dictionary in merge_operno_list and create a list
        values_to_replace = [list(rule.values())[0] for rule in merge_operno_list]
        # Concatenate values from each dictionary
        merged_values = ["_".join(list(rule.values())[0]) for rule in merge_operno_list]

        # Replace values in 'OPER_NO' column based on the rules defined in merge_operno_list
        for values, replacement_value in zip(values_to_replace, merged_values):
            df = df.withColumn("OPER_NO",
                               when(col("OPER_NO").isin(values), replacement_value).otherwise(col("OPER_NO")))
        return df
    else:
        return df


def _pre_process(df: pyspark.sql.dataframe) -> pyspark.sql.dataframe:
    """
    Preprocess the data extracted from the database for a specific CASE.
    :param df: Data for a specific CASE retrieved from the database.
    :return: Preprocessed data with relevant columns and filters applied.
    """
    # Select only the columns that will be used
    df = df.select('WAFER_ID', 'TOOL_ID', 'RUN_ID', 'EQP_NAME', 'PRODUCT_ID', 'PRODG1', 'TOOL_NAME',
                   'OPER_NO', 'parametric_name', 'STATISTIC_RESULT', 'label')
    # Remove rows with missing values in 'STATISTIC_RESULT' column
    df = df.filter(col('STATISTIC_RESULT').isNotNull())
    # Drop duplicates based on all columns
    df1 = df.dropDuplicates()
    # Select the rows with the latest 'RUN_ID' for each combination of 'WAFER_ID', 'OPER_NO', 'TOOL_ID'
    df2 = df1.groupBy('WAFER_ID', 'OPER_NO', 'TOOL_ID').agg(max('RUN_ID').alias('RUN_ID'))
    df_run = df1.join(df2.dropDuplicates(subset=['WAFER_ID', 'OPER_NO', 'TOOL_ID', 'RUN_ID']),
                      on=['WAFER_ID', 'OPER_NO', 'TOOL_ID', 'RUN_ID'], how='inner')
    return df_run


def commonality_analysis(df_run: pyspark.sql.dataframe, grpby_list: List[str]) -> pyspark.sql.dataframe:
    """
    Perform commonality analysis on preprocessed data.
    :param df_run: Preprocessed data after data preprocessing.
    :param grpby_list: List of columns ['PRODG1', 'EQP_NAME', 'OPER_NO', 'PRODUCT_ID', 'TOOL_NAME'] for grouping.
            Example: grpby_list = ['PRODG1', 'TOOL_NAME', 'OPER_NO'], grpby_list = ['PRODUCT_ID', 'OPER_NO']
    :return: Results of commonality analysis, showing the top ten combinations with the highest number of bad wafers.
    """
    grps = (df_run.groupBy(grpby_list)
            .agg(countDistinct('WAFER_ID').alias('wafer_count'),
                 countDistinct('WAFER_ID', when(df_run['label'] == 0, 1)).alias('good_num'),
                 countDistinct('WAFER_ID', when(df_run['label'] == 1, 1)).alias('bad_num'))
            .orderBy('bad_num', ascending=False))

    # Handle the case of a single OPER_NO or single TOOL_NAME
    if grps.count() == 1:
        return grps
    else:
        # Filter out groups with no bad wafers
        grps = grps.filter(grps['bad_num'] > 0)
        # Rank the groups based on the number of bad wafers
        window_sep = Window().orderBy(col("bad_num").desc())
        ranked_df = grps.withColumn("rank", rank().over(window_sep))
        # Select the top ten groups and remove the 'rank' column
        grpss = ranked_df.filter(col("rank") <= 10).drop("rank")
        return grpss


def get_data_list(common_res: pyspark.sql.dataframe,
                  grpby_list: List[str],
                  big_or_small: str = 'big') -> List[Dict[str, str]]:
    """
    Get a list of dictionaries for corresponding groups based on commonality analysis.

    :param common_res: Result of commonality analysis.
    :param grpby_list:  List of columns ['PRODG1', 'EQP_NAME', 'OPER_NO', 'PRODUCT_ID', 'TOOL_NAME'] for grouping.
    :param big_or_small: 'big' or 'small'.
    :return: List of dictionaries for corresponding groups.
            Example: [{'OPER_NO': '1F.EEK10', 'PRODUCT_ID': 'AFKN2J01N.0U01'},
                      {'OPER_NO': '1F.EEK10', 'PRODUCT_ID': 'AFKN4X01N.0B01'},
                      {'OPER_NO': '1F.EEK10', 'PRODUCT_ID': 'AFGN1501N.0C02'}]
    """
    assert big_or_small in ['big', 'small'], "Choose only 'big' or 'small'. Please check the spelling."

    # Filter groups based on big or small sample conditions
    if big_or_small == 'big':
        good_bad_grps = common_res.filter("good_num >= 3 AND bad_num >= 3")
    else:
        good_bad_grps = common_res.filter("bad_num >= 1 AND wafer_count >= 2")

    # Order the results and limit to the top 10 groups
    good_bad_grps = good_bad_grps.orderBy(col("bad_num").desc(), col("wafer_count").desc(),
                                          col("good_num").desc()).limit(10)

    # Collect the data and convert it into a list of dictionaries
    data_list = good_bad_grps[grpby_list].collect()
    data_dict_list = [row.asDict() for row in data_list]
    return data_dict_list


def get_train_data(df_run: pyspark.sql.dataframe, data_dict_list: List[Dict[str, str]]) -> pyspark.sql.dataframe:
    """
    Get the actual combination data for modeling from the original data.

    :param df_run: Preprocessed data after data preprocessing.
    :param data_dict_list: List of dictionaries with filtering conditions.
           Example: [{'OPER_NO': '1F.EEK10', 'PRODUCT_ID': 'AFKN2J01N.0U01'},
                      {'OPER_NO': '1F.EEK10', 'PRODUCT_ID': 'AFKN4X01N.0B01'},
                      {'OPER_NO': '1F.EEK10', 'PRODUCT_ID': 'AFGN1501N.0C02'}]
    :return: Filtered data for modeling.
    """
    # Get the filtering conditions for the first data dictionary
    first_data_dict = data_dict_list[0]
    conditions = " AND ".join(["{} == '{}'".format(col, first_data_dict[col]) for col in first_data_dict])
    # Filter the data for the first condition
    df_s = df_run.filter(conditions)

    # Loop through the remaining data dictionaries and filter the data accordingly
    for i in range(1, len(data_dict_list)):
        data_dict = data_dict_list[i]
        conditions = " AND ".join(["{} == '{}'".format(col_, data_dict[col_]) for col_ in data_dict])
        df_m = df_run.filter(conditions)
        df_s = df_s.union(df_m)
    return df_s


def get_all_bad_wafer_num(df: pyspark.sql.dataframe) -> int:
    """
    Get the number of distinct bad WAFER in the DataFrame.
    """
    return df.filter("label == 1").select('WAFER_ID').distinct().count()


def get_pivot_table(df: pd.DataFrame, grpby_list: List[str]) -> pd.DataFrame:
    """
    Pivot the DataFrame based on specified grouping columns.

    Parameters:
    - df: Data for modeling.
    - grpby_list: List of grouping columns.

    Returns:
    - DataFrame: Result of pivoting the table.
    """
    index_cols = ['WAFER_ID', 'label']
    columns_cols = grpby_list + ['parametric_name']
    df_pivot = df.dropna(axis=0).pivot_table(index=index_cols,
                                             columns=columns_cols,
                                             values=['STATISTIC_RESULT'])
    df_pivot.columns = df_pivot.columns.map('#'.join)
    df_pivot = df_pivot.fillna(df_pivot.mean()).reset_index(drop=False)
    return df_pivot


def fit_rf_big_sample(df: pyspark.sql.dataframe, grpby_list: List[str]) -> pyspark.sql.dataframe:
    """
    Fit a RandomForest model on the train data. It is for large sample method(good_wafer_num >= 3 AND bad_wafer_num >= 3)

    Parameters:
    - df: Data for modeling.
    - grpby_list: List of grouping columns.

    Returns:
    - DataFrame: Combined dataframe of roc_auc_score result and feature importance after RandomForest modeling.
    """
    # Dynamically build schema according to the grpby_list
    struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
    struct_fields.extend([StructField("bad_wafer", IntegerType(), True),
                          StructField("roc_auc_score", FloatType(), True),
                          StructField("features", StringType(), True),
                          StructField("importance", FloatType(), True)])
    schema_all = StructType(struct_fields)

    @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
    def get_model_result(df_run: pd.DataFrame) -> pd.DataFrame:
        # Pivot the table
        df_pivot = get_pivot_table(df=df_run, grpby_list=grpby_list)

        # Define independent and dependent variables
        x_train = df_pivot[df_pivot.columns.difference(['WAFER_ID', 'label']).tolist()]
        y_train = df_pivot[['label']]

        z_ratio = y_train.value_counts(normalize=True)
        good_ratio = z_ratio[0]
        bad_ratio = z_ratio[1]
        if abs(good_ratio - bad_ratio) > 0.7:
            undersampler = ClusterCentroids(random_state=101)
            x_train, y_train = undersampler.fit_resample(x_train, y_train)

        # Grid search
        pipe = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(random_state=2024))])
        param_grid = {'model__n_estimators': [*range(50, 100, 10)],
                      'model__max_depth': [*range(10, 50, 10)]}
        grid = GridSearchCV(estimator=pipe, scoring='roc_auc', param_grid=param_grid, cv=3, n_jobs=-1)
        grid.fit(x_train.values, y_train.values.ravel())
        roc_auc_score_ = grid.best_score_

        # Feature importance and result summary
        small_importance_res = pd.DataFrame({'features': x_train.columns,
                                             'importance': grid.best_estimator_.steps[2][1].feature_importances_})

        sample_res_dict = {'bad_wafer': sum(df_pivot['label']),
                           'roc_auc_score': roc_auc_score_}
        sample_res_dict.update({col_: df_run[col_].unique() for col_ in grpby_list})
        small_sample_res = pd.DataFrame(sample_res_dict)
        return pd.concat([small_importance_res, small_sample_res])
    return df.groupby(grpby_list).apply(get_model_result)


def split_score_big_sample(df: pyspark.sql.dataframe, grpby_list: List[str]) -> pyspark.sql.dataframe:
    """
    Split the ROC AUC scores based on the specified grouping columns.

    Parameters:
    - df: Results after RandomForest modeling.
    - grpby_list: List of grouping columns.

    Returns:
    - DataFrame: ROC AUC scores result with each element in grpby_list as columns.
    """
    struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
    struct_fields.extend([StructField("bad_wafer", IntegerType(), True),
                          StructField("roc_auc_score", FloatType(), True)])
    schema_all = StructType(struct_fields)

    @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
    def get_result(model_results: pd.DataFrame) -> pd.DataFrame:
        select_expr = grpby_list + ['bad_wafer', 'roc_auc_score']
        sample_res = model_results[select_expr].dropna(axis=0)
        sample_res = sample_res[sample_res['roc_auc_score'] > 0.6]
        return sample_res

    return df.groupby(grpby_list).apply(get_result)


def split_features(df: pd.DataFrame, index: int) -> str:
    """
    Split the 'features' column based on the specified index.

    Parameters:
    - df: RandomForest modeling results with 'features' column.
    - index: Order value.

    Returns:
    - str: Field attribute value.
    """
    return df['features'].apply(lambda x: x.split('#')[index])


def get_split_feature_importance_table(df: pd.DataFrame, grpby_list: List[str]) -> pd.DataFrame:
    """
    Get the table after splitting the 'features' column based on the specified grouping columns.

    Parameters:
    - df: RandomForest modeling results with 'features' column.
    - grpby_list: List of grouping columns.

    Returns:
    - DataFrame: Table after splitting features.
    """
    n_feats = len(grpby_list)
    for i in range(n_feats):
        df[grpby_list[i]] = split_features(df, i + 1)

    df['parametric_name'] = split_features(df, n_feats + 1)
    df['step'] = split_features(df, n_feats + 2)
    df['stats'] = split_features(df, n_feats + 3)
    df = df.drop(['features'], axis=1).reset_index(drop=True)
    return df


def add_feature_stats(df: pd.DataFrame, grpby_list: List[str]) -> pd.DataFrame:
    """
    Add a column with all statistical features of parameters.

    Parameters:
    - df: Feature importance table after processing.
    - grpby_list: List of grouping columns.

    Returns:
    - DataFrame: New column containing all statistical features: 'feature_stats'.
    """
    grpby_list_extend = grpby_list + ['parametric_name', 'step']
    feature_stats = df.groupby(grpby_list_extend)['stats'].unique().reset_index()
    feature_stats['stats'] = [feature_stats['stats'].iloc[i].tolist() for i in range(len(feature_stats))]
    feature_stats['stats'] = feature_stats['stats'].apply(lambda x: "#".join(x))
    feature_stats = feature_stats.assign(parametric_name=lambda x: x['parametric_name'] + str('#') + x['step']).drop(
        'step', axis=1)
    return feature_stats


def split_calculate_features_big_sample(df: pyspark.sql.dataframe, grpby_list: List[str]) -> pyspark.sql.dataframe:
    """
    Split and calculate features based on the specified grouping columns.

    Parameters:
    - df: Results after RandomForest modeling.
    - grpby_list: List of grouping columns.

    Returns:
    - DataFrame: Features importance results.
    """
    # Dynamically build schema
    struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
    struct_fields.extend([StructField("parametric_name", StringType(), True),
                          StructField("importance", FloatType(), True),
                          StructField("stats", StringType(), True)])
    schema_all = StructType(struct_fields)

    @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
    def get_result(model_results: pd.DataFrame) -> pd.DataFrame:
        # Extract 'features' and 'importance' from the RandomForest model results
        feature_importance_table = model_results[['features', 'importance']].dropna(axis=0)

        # Split features
        feature_importance_res_split = get_split_feature_importance_table(df=feature_importance_table,
                                                                          grpby_list=grpby_list)

        # Remove combinations with importance equal to 0
        feature_importance_res_split_drop = feature_importance_res_split.query("importance > 0").reset_index(drop=True)

        # Take the top 60% or 100% of each combination result
        feature_importance_res_split_nlargest = (feature_importance_res_split_drop.groupby(by=grpby_list)
                                                 .apply(
            lambda x: x.nlargest(int(x.shape[0] * 0.6), 'importance') if x.shape[0] > 1 else x.nlargest(
                int(x.shape[0] * 1), 'importance'))
                                                 .reset_index(drop=True))

        # Add a column with all statistical features: 'feature_stats'
        feature_stats = add_feature_stats(df=feature_importance_res_split_drop, grpby_list=grpby_list)

        # Sum the importance for the same combination and parameter: 'feature_importance_groupby'
        feature_importance_groupby = (
            feature_importance_res_split_nlargest.groupby(grpby_list + ['parametric_name', 'step'])['importance']
            .sum().reset_index())
        feature_importance_groupby = (
            feature_importance_groupby.assign(parametric_name=lambda x: x['parametric_name'] + str('#') + x['step'])
            .drop('step', axis=1))

        # Connect 'feature_stats' and 'feature_importance_groupby'
        grpby_stats = pd.merge(feature_stats, feature_importance_groupby,
                               on=grpby_list + ['parametric_name']).dropna().reset_index(drop=True)
        return grpby_stats
    return df.groupby(grpby_list).apply(get_result)


def get_final_results_big_sample(s_res: pyspark.sql.dataframe, f_res: pyspark.sql.dataframe, grpby_list: List[str],
                                  bad_wafer_num: int) -> pyspark.sql.dataframe:
    """
    Get the final modeling results.

    Parameters:
    - s_res: ROC AUC scores result.
    - f_res: Features importance result.
    - grpby_list: List of grouping columns.
    - bad_wafer_num: Number of bad wafers in the data.

    Returns:
    - DataFrame: Final modeling result.
    """
    roc_auc_score_all = s_res.agg({"roc_auc_score": "sum"}).collect()[0][0]
    s_res = s_res.withColumn("roc_auc_score_ratio", col("roc_auc_score") / roc_auc_score_all)
    s_res = s_res.withColumn("bad_ratio", col("bad_wafer") / bad_wafer_num)

    df_merge = s_res.join(f_res, on=grpby_list, how='left')
    df_merge = df_merge.withColumn('weight_original', col('roc_auc_score_ratio') * col('bad_ratio') * col('importance'))

    # Normalize again
    weight_all = df_merge.agg({"weight_original": "sum"}).collect()[0][0]
    df_merge = df_merge.withColumn("weight", col("weight_original") / weight_all)
    df_merge = df_merge.select(grpby_list + ['parametric_name', 'weight', 'stats']).orderBy('weight', ascending=False)
    return df_merge


def add_certain_column(df: pyspark.sql.dataframe, by: str, request_id: str, grpby_list: List[str]) -> pyspark.sql.dataframe:
    """
    Add specific columns to the final modeling results.

    Parameters:
    - df: Final modeling result.
    - by: Grouping column, manually add a column 'add'.
    - request_id: Request ID passed in.
    - grpby_list:  List of grouping columns.

    Returns:
    - DataFrame: Final modeling result with specific columns added.
    """
    # Dynamically build schema_all
    struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
    struct_fields.extend([StructField("stats", StringType(), True),
                          StructField("parametric_name", StringType(), True),
                          StructField("weight", FloatType(), True),
                          StructField("request_id", StringType(), True),
                          StructField("weight_percent", FloatType(), True),
                          StructField("index_no", IntegerType(), True)])
    schema_all = StructType(struct_fields)

    @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
    def get_result(final_res: pd.DataFrame) -> pd.DataFrame:
        final_res['weight'] = final_res['weight'].astype(float)
        final_res = final_res.query("weight > 0")
        final_res['request_id'] = request_id
        final_res['weight_percent'] = final_res['weight'] * 100
        final_res = final_res.sort_values('weight', ascending=False)
        final_res['index_no'] = [i + 1 for i in range(len(final_res))]
        final_res = final_res.drop('add', axis=1)
        return final_res
    return df.groupby(by).apply(get_result)




# 大样本数据模型整合
def fit_big_data_model(df_run, data_dict_list_bs, grpby_list, request_id):
    df1 = None
    df2 = None

    # 1. 获取用于建模的大样本数据
    df_run_bs = get_train_data(df_run, data_dict_list_bs)
    if df_run_bs.count() == 0:
        msg = '数据库中暂无此类数据!'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, df2

    # 2. 获取所有bad wafer数量
    bad_wafer_num_big_sample = get_all_bad_wafer_num(df_run_bs)
    if bad_wafer_num_big_sample < 3:
        msg = '数据库中实际BAD_WAFER数量小于3片, 请提供更多的BAD_WAFER数量!'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, df2

    # 3. 对挑选出的大样本数据进行建模
    res = fit_rf_big_sample(df=df_run_bs, by=grpby_list)
    if res.count() == 0:
        msg = '算法内部暂时异常!'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, df2

    # 4. 将建模结果进行整合
    s_res = split_score_big_sample(df=res, by=['PRODG1', 'OPER_NO', 'TOOL_NAME'])
    if s_res.count() == 0:
        msg = '算法运行评分结果较低, 暂无输出, 建议增加BAD_WAFER数量'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, df2

    f_res = split_calculate_features_big_sample(df=res, by=grpby_list)
    if f_res.count() == 0:
        msg = '算法结果求和暂时异常'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, df2

    model_res_bs = get_finall_results_big_sample(s_res=s_res, f_res=f_res, bad_wafer_num=bad_wafer_num_big_sample)
    if model_res_bs.count() == 0:
        msg = '算法结果拼接暂时异常'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, df2

    # 7. 增加特定的列
    final_res_bs = model_res_bs.withColumn('add', lit(0))
    final_res_add_columns = add_certain_column(df=final_res_bs, by='add', request_id=request_id)
    if final_res_add_columns.count() == 0:
        msg = '算法结果增加列暂时异常'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, df2
    else:
        return df1, final_res_add_columns


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
        # 由于是小样本，再重新copy一份制造多一点数据传给PCA模型
        df_pivot_copy = df_pivot.copy()
        df_pivot_all = pd.concat([df_pivot, df_pivot_copy], axis=0)

        # 定义自变量
        x_train = df_pivot_all[df_pivot_all.columns.difference(['WAFER_ID', 'label']).tolist()]

        # 建立模型，传入给PCA的n_components选择x_train.shape中的最小值-1；
        # 选择是70%或者80%，出来的特征很有可能只是一两个
        model = pca(n_components=min(x_train.shape[0], x_train.shape[1]) - 1, verbose=None)
        results = model.fit_transform(x_train)
        res_top = results['topfeat']
        res_top_select = res_top[res_top['type'] == 'best'][['feature', 'loading']]
        res_top_select = res_top_select.drop_duplicates()
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
                               on=['PRODG1', 'OPER_NO', 'TOOL_NAME', 'parametric_name']).dropna().reset_index(drop=True)
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


# 小样本数据模型整合
def fit_small_data_model(df_run, common_res, grpby_list, request_id):
    df1 = None
    df2 = None

    data_dict_list_ss = get_data_list(common_res=common_res, grpby_list=grpby_list, big_or_small='small')
    print("data_dict_list_ss:", data_dict_list_ss)
    if len(data_dict_list_ss) == 0:
        msg = '该查询条件下数据库中实际BAD_WAFER数量为0, 无法分析'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, df2

    df_run_ss = get_train_data(df_run=df_run, data_dict_list=data_dict_list_ss)
    if df_run_ss.count() == 0:
        msg = '数据库中暂无此类数据!'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, df2

    bad_wafer_num_small_sample = get_all_bad_wafer_num(df_run_ss)
    if bad_wafer_num_small_sample < 1:
        msg = '该查询条件下数据库中实际BAD_WAFER数量小于1片, 请提供更多的BAD_WAFER数量!'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, df2

    res = fit_pca_small_sample(df=df_run_ss, by=grpby_list)
    if res.count() == 0:
        msg = '算法内部暂时异常!'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, df2

    f_res = split_calculate_features_small_sample(df=res, by=grpby_list)
    if f_res.count() == 0:
        msg = '算法结果求和暂时异常'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, df2

    model_res_ss = get_finall_results_small_sample(f_res=f_res, bad_wafer_num=bad_wafer_num_small_sample)
    if model_res_ss.count() == 0:
        msg = '算法结果拼接暂时异常'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, df2

    final_res_ss = model_res_ss.withColumn('add', lit(0))
    final_res_add_columns = add_certain_column(df=final_res_ss, by='add', request_id=request_id)
    if final_res_add_columns.count() == 0:
        msg = '算法结果增加列暂时异常'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        return df1, df2
    else:
        return df1, final_res_add_columns


##########################################################################################
#######################################正式调用以上函数#######################################
##########################################################################################
# request_id = 'sdd'
# grpby_list = ['OPER_NO', 'TOOL_NAME']

# 1. 解析json 为字典， df1为kafka输入的结果数据，获取到parse_dict, request_id, grpby_list, merge_operno
df2 = df1.toPandas()
parse_dict, request_id, grpby_list, merge_operno = get_some_info(df2)
print("parse_dict是：", parse_dict)
print("parse_dict的类型是：", type(parse_dict))
print("request_id是：", request_id)
print("grpby_list是：", grpby_list)
print("merge_operno是：", merge_operno)

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
    print(df1.count())
    if df1.count() == 0:
        msg = '解析SQL获取数据异常: 数据库中可能没有数据!'
        df_kafka = pd.DataFrame({"code": 1, "msg": f'{msg}', "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)
        raise ValueError

    # 1. 站点融合和数据预处理
    df1 = integrate_operno(df=df1, merge_operno_list=merge_operno)
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
        print("****************大样本算法调用****************")
        df1, final_res_add_columns = fit_big_data_model(df_run, data_dict_list_bs, grpby_list, request_id)
    else:
        print("****************小样本算法调用****************")
        df1, final_res_add_columns = fit_small_data_model(df_run, common_res, grpby_list, request_id)

    if df1 is not None:
        raise ValueError
    else:
        # final_res_add_columns 是最后的结果，要写回数据库
        # ddd = final_res_add_columns.toPandas()
        # user ="root"
        # host = "10.52.199.81"
        # password = "Nexchip%40123"
        # db = "etl"
        # port = 9030
        # engine = create_engine("mysql+pymysql://{user}:{password}@{host}:{port}/{db}".format(user = user,
        #                                                                                     password = password,
        #                                                                                     host = host,
        #                                                                                     port = port,
        #                                                                                     db = db))
        # doris_stream_load_from_df(ddd, engine, "results")

        # # 最终成功的话，就会输出下面这条
        print("运行成功")
        df_kafka = pd.DataFrame({"code": 0, "msg": "运行成功", "requestId": request_id}, index=[0])
        df1 = spark.createDataFrame(df_kafka)

except ValueError as ve:
    pass

except Exception as e:
    df_kafka = pd.DataFrame({"code": 1, "msg": f"主程序发生异常: {str(e)}", "requestId": request_id}, index=[0])
    df1 = spark.createDataFrame(df_kafka)

print("最终的df1是：")
print(type(df1))
df1.show()

print("最终的算法结果是：")
print(type(final_res_add_columns))
final_res_add_columns.show()
