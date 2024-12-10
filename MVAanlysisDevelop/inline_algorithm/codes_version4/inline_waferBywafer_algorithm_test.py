import json

import pyspark
import requests
import pymysql
import numpy as np
import pandas as pd
import pyspark.pandas as ps
import pyspark.sql.functions as F
from pca import pca

from scipy import stats
from functools import reduce
from pyspark.sql import DataFrame
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, IntegerType, FloatType
from pyspark.sql.functions import pandas_udf, PandasUDFType, monotonically_increasing_id, lit, col, countDistinct, when


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


if __name__ == "__main__":
    import os
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

    # 1. 读取数据
    df_pandas = pd.read_csv(
        "D:/Jupyterfiles/晶合MVAFDC_general开发/MVAanlysisDevelop/inline_algorithm/inline_case5_label.csv")
    df_spark = ps.from_pandas(df_pandas).to_spark()
    num_rows = df_spark.count()
    num_columns = len(df_spark.columns)
    print(f"df_spark shape: ({num_rows}, {num_columns})")

    # 2. 数据预处理
    dp = DataPreprocessorForInline(df=df_spark,
                                     columns_list=['WAFER_ID', 'OPE_NO', 'INLINE_PARAMETER_ID', 'AVERAGE', 'MAX_VAL',
                                                   'MEDIAN', 'MIN_VAL', 'STD_DEV', 'PERCENTILE_25', 'PERCENTILE_75', 'SITE_COUNT', 'label'],
                                     certain_column='INLINE_PARAMETER_ID',
                                     key_words=['CXS', 'CYS', 'FDS'],
                                     convert_to_numeric_list=['AVERAGE', 'MAX_VAL', 'MEDIAN', 'MIN_VAL', 'STD_DEV',
                                                              'PERCENTILE_25', 'PERCENTILE_75', 'SITE_COUNT'])
    df_pp_ = dp.run()
    num_rows = df_pp_.count()
    num_columns = len(df_pp_.columns)
    print(f"df_pp_ shape: ({num_rows}, {num_columns})")

    # 3. 获取训练数据
    gtd = GetTrainDataForInline(df=df_pp_, grpby_list=['OPE_NO'])
    df_run_ = gtd.run()
    num_rows = df_run_.count()
    num_columns = len(df_run_.columns)
    print(f"df_run_ shape: ({num_rows}, {num_columns})")

    # 4. 训练模型
    res = fit_pca_model(df=df_run_, by=['OPE_NO'], columns_to_process=['AVERAGE', 'MAX_VAL', 'MEDIAN', 'MIN_VAL',
                                                                       'STD_DEV', 'PERCENTILE_25', 'PERCENTILE_75'])
    num_rows = res.count()
    num_columns = len(res.columns)
    print(f"res shape: ({num_rows}, {num_columns})")
    res.show()

    # 5. 特征处理和排序
    res_add = res.withColumn('add', lit(0))
    final_res = split_calculate_features(df=res_add, by='add')
    final_res = final_res.withColumn('add', lit(0))
    final_res_add = add_certain_column(df=final_res, by='add', request_id='855s')
    num_rows = final_res_add.count()
    num_columns = len(final_res_add.columns)
    print(f"final_res_add shape: ({num_rows}, {num_columns})")
    final_res_add.show()



