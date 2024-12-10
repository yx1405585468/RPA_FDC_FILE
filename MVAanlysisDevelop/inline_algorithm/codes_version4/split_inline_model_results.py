import pyspark
import pandas as pd
import pyspark.pandas as ps
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.sql.functions import pandas_udf, PandasUDFType, lit


class SplitInlineModelResults:
    def __init__(self, df: pyspark.sql.dataframe, request_id: str):
        self.df = df
        self.request_id = request_id

    @staticmethod
    def split_features(df: pd.DataFrame, index: int) -> str:
        return df['features'].apply(lambda x: x.split('#')[index])

    @staticmethod
    def get_split_features(df: pd.DataFrame) -> pd.DataFrame:
        df['STATISTIC_RESULT'] = SplitInlineModelResults.split_features(df, 0)
        df['OPE_NO'] = SplitInlineModelResults.split_features(df, 1)
        df['INLINE_PARAMETER_ID'] = SplitInlineModelResults.split_features(df, 2)
        df = df.drop(['features', 'STATISTIC_RESULT'], axis=1).reset_index(drop=True)
        return df

    @staticmethod
    def split_calculate_features(df: pyspark.sql.dataframe, by: str) -> pyspark.sql.dataframe:
        schema_all = StructType([StructField("OPE_NO", StringType(), True),
                                 StructField("INLINE_PARAMETER_ID", StringType(), True),
                                 StructField("importance", FloatType(), True)])

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run):
            split_table = SplitInlineModelResults.get_split_features(df_run)
            split_table_grpby = split_table.groupby(['OPE_NO', 'INLINE_PARAMETER_ID'])['importance'].sum().reset_index(drop=False)
            return split_table_grpby
        return df.groupby(by).apply(get_model_result)

    @staticmethod
    def add_certain_column(df: pyspark.sql.dataframe, by: str) -> pyspark.sql.dataframe:
        schema_all = StructType([StructField("OPER_NO", StringType(), True),
                                StructField("INLINE_PARAMETER_ID", StringType(), True),
                                StructField("AVG_SPEC_CHK_RESULT_COUNT", FloatType(), True),
                                StructField("weight", FloatType(), True),
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
            final_res['AVG_SPEC_CHK_RESULT_COUNT'] = 0.0
            final_res = final_res.rename(columns={'OPE_NO': 'OPER_NO'})
            final_res = final_res.drop(['importance', 'temp'], axis=1)
            return final_res
        return df.groupby(by).apply(get_result)

    def run(self):
        df = self.df.withColumn('temp', lit(0))
        res = self.split_calculate_features(df=df, by='temp')
        res = res.withColumn('temp', lit(1))
        final_res = self.add_certain_column(df=res, by='temp')
        final_res = final_res.withColumn('request_id', lit(self.request_id))
        return final_res


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
    df_pandas = pd.read_csv(filepath_or_buffer="inline_model_by_wafer_res1.csv")
    df_spark = ps.from_pandas(df_pandas).to_spark()
    num_rows = df_spark.count()
    num_columns = len(df_spark.columns)
    print(f"df_spark shape: ({num_rows}, {num_columns})")

    # 2. 整理结果
    final_res_ = SplitInlineModelResults(df=df_spark, request_id='855s').run()
    num_rows = final_res_.count()
    num_columns = len(final_res_.columns)
    print(f"final_res_ shape: ({num_rows}, {num_columns})")
    final_res_.show()
