import pyspark
import pandas as pd
import pyspark.pandas as ps
from pyspark.sql.functions import col, countDistinct, when


class DataPreprocessorForInline:
    def __init__(self,
                 df: pyspark.sql.dataframe,
                 columns_list: list[str],
                 certain_column: str,
                 key_words: list[str],
                 convert_to_numeric_list: list[str]):
        self.df = df
        self.columns_list = columns_list
        self.certain_column = certain_column
        self.key_words = key_words
        self.convert_to_numeric_list = convert_to_numeric_list

    def select_columns(self, df):
        return df.select(self.columns_list)

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
        df_select = self.select_columns(df=self.df)
        df_esd = self.exclude_some_data(df=df_select)
        df_pp = self.pre_process(df=df_esd)
        return df_pp


class GetTrainDataForInline:
    def __init__(self, df: pyspark.sql.dataframe, grpby_list: list[str]):
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

    def get_data_list(self, common_res):
        data_list = common_res.select(self.grpby_list).collect()
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
        data_dict_list = self.get_data_list(common_res)
        train_data = self.get_train_data(data_dict_list)
        return train_data


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
    df_pandas = pd.read_csv("D:/Jupyterfiles/晶合MVAFDC_general开发/MVAanlysisDevelop/inline_algorithm/inline_case5_label.csv")
    df_spark = ps.from_pandas(df_pandas).to_spark()
    num_rows = df_spark.count()
    num_columns = len(df_spark.columns)
    print(f"df_spark shape: ({num_rows}, {num_columns})")

    # 2. 数据预处理
    dp = DataPreprocessorForInline(df=df_spark,
                                   columns_list=['WAFER_ID', 'OPE_NO', 'INLINE_PARAMETER_ID', 'AVERAGE', 'MAX_VAL', 'MEDIAN',
                                                 'MIN_VAL', 'STD_DEV', 'PERCENTILE_25', 'PERCENTILE_75', 'SITE_COUNT', 'label'],
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
