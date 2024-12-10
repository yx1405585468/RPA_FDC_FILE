# encoding:utf-8
"""
compare with go through time(Process) ,by wafer 算法
"""
import pyspark
from pyspark.sql.functions import count
import pandas as pd
from src.correlation.common_process.data_preprocessing import ProcessBaseDFByWAFER, MergeOneColumnMultiValuesIntoNewOne
from typing import List, Dict,Optional
from pyspark.sql.functions import pandas_udf, PandasUDFType, lit, col
from pyspark.sql.types import StringType, DoubleType, StructType, StructField, FloatType
from src.correlation.common_process.corr_base_alogrithm import CorrelationDetectAlgorithm


class CorrCompareProcessByWaferAlgorithm:

    @staticmethod
    def run(base_df: pyspark.sql.DataFrame, source_df: pyspark.sql.DataFrame, request_id: str,
            config_dict: dict) -> pyspark.sql.DataFrame:
        # 选择元素，处理None值
        select_config_dict = {
            k: list(v) if v is not None else list()  # optional类型 解包， None => 空列表
            for k, v in config_dict.items()
            if k in ['group_by_list', 'merge_prodg1_list', 'merge_product_list',
                     ]
        }


        return CorrCompareProcessByWaferAlgorithm.__run_with_kwargs__(base_df, source_df, request_id,
                                                                     **select_config_dict)

    @staticmethod
    def __run_with_kwargs__(base_df: pyspark.sql.DataFrame, source_df: pyspark.sql.DataFrame, request_id: str,
                            group_by_list: List[str],
                            merge_prodg1_list: List[Dict[str, List[str]]],
                            merge_product_list: List[Dict[str, List[str]]],
                            ) -> Optional[pyspark.sql.DataFrame]:

       # 异常数据处理
       base_df, base_col = ProcessBaseDFByWAFER.process_by_wafer(base_df)


       # source_df 处理
       source_df= source_df.select( ['PRODG1', "PRODUCT_ID",'EQP_NAME','CHAMBER_NAME',"WAFER_ID"]).dropna(how='any')
       # 处理合并选项
       merge_option_dict = {
       "PRODUCT_ID": merge_product_list,
        "PRODG1": merge_prodg1_list,
       }
       source_df = MergeOneColumnMultiValuesIntoNewOne.integrate_columns(source_df, **merge_option_dict)

       if 'OPE_NO' in group_by_list:
           group_by_list.remove('OPE_NO')

       # pandas Udf
       schema = StructType(
           [StructField(col_, StringType(), True) for col_ in group_by_list ] +
           [
               StructField("WEIGHT", DoubleType(), True),
               StructField("COUNT", FloatType(), True),

           ])


       @pandas_udf(returnType=schema, functionType=PandasUDFType.GROUPED_MAP)
       def get_corr_weight(data: pd.DataFrame) -> pd.DataFrame:

           data['COUNT'] = len(data)
           if data["COUNT"].values[0] == 1:
               weight = 0.0
           else:
               weight = CorrelationDetectAlgorithm.get_corr_func(x=data[base_col], y=data['WAFER_COUNT'])
           # 返回一行分组字段 + parameter_name
           return data.head(1).assign(WEIGHT=weight).filter(
               items=group_by_list + [ "WEIGHT", "COUNT"])



       # group by list 至少里面还有 EQP_NAME 或者 (EQP_NAME, CHAMBER_NAME)) 才能进行group by
       count_df = source_df.groupBy(group_by_list+["WAFER_ID"]).agg(count('WAFER_ID').alias("WAFER_COUNT"))
       result = (count_df
                 .join(base_df, on=["WAFER_ID"], how="left")
                 .dropna(subset=['WAFER_COUNT', base_col], how='any') #  去除wafer id两边计数存在至少一个为空的情况
            .groupBy(group_by_list).apply(
                    get_corr_weight
       )
    )
       result = (result
                 .withColumn("STATS", lit("GO_THROUGH_TIME"))
                 .withColumn("REQUEST_ID", lit(request_id))  # 赋值request_id 信息
                 .withColumn("ANALYSIS_NAME", lit("PROCESS"))  # 赋值分析类型

                 )

       return result






if __name__ == '__main__':
    import pyspark
    from pyspark.sql import SparkSession
    import pandas as pd
    import pyspark.pandas as ps
    from time import time
    from pyspark.sql.functions import col as F_col,countDistinct, count,first, sum
    import numpy as np
    def timer(func):
        def func_wrapper(*args, **kwargs):
            time_start = time()
            result = func(*args, **kwargs)
            time_end = time()
            time_spend = time_end - time_start
            print('%s cost time: %.3f s' % (func.__name__, time_spend))

            return result

        return func_wrapper


    def get_local_spark():
        import findspark
        import warnings
        warnings.filterwarnings("ignore")
        spark_home = r"D:\software\spark-3.3.0-bin-hadoop3"

        python_path = r"D:\software\Anaconda3\envs\python39\python.exe"
        findspark.init(spark_home, python_path)
        # findspark.init()
        # os.environ['PYSPARK_PYTHON'] = "/usr/local/python-3.9.13/bin/python3"
        spark = SparkSession.builder.appName("example").getOrCreate()
        return spark

    spark = get_local_spark()

    @timer
    def test_inline_go_through_time(spark:SparkSession):
        df_pandas = pd.read_csv(
            r"D:\xxs_project\2024\RCA根因分析\test_data\wafer_condition_select.csv").rename(columns={"OPER_NO": "OPE_NO"})
        #
        base_wafer_id_list = df_pandas.WAFER_ID.unique().tolist()[:10]
        df_pandas = df_pandas[df_pandas.WAFER_ID.isin(base_wafer_id_list)]
        df_spark = ps.from_pandas(df_pandas).to_spark()

        base_df = pd.DataFrame({
            "WAFER_ID": base_wafer_id_list,
            "value": np.random.rand(len(base_wafer_id_list))
        })

        base_df_spark = ps.from_pandas(base_df).to_spark()

        json_loads_dict = {"requestId": "269",
                           "process": {
                               "group_by_list": ['PRODG1', 'PRODUCT_ID', "OPE_NO", "EQP_NAME", "CHAMBER_NAME"],
                               # "merge_prodg1_list": [{'xx1':['L15KD03A', 'L11TG07A']}],
                               "merge_prodg1_list": None,
                               "merge_product_list": None,

                           }
                           }

        result = CorrCompareProcessByWaferAlgorithm.run(
            base_df=base_df_spark,
            source_df=df_spark,
            request_id="269",
            config_dict=json_loads_dict.get("process")
        )
        result.orderBy(col("WEIGHT").desc()).show()


        # df_spark = df_spark.select(
        #      'PRODG1',
        #     "PRODUCT_ID",
        #     'OPE_NO',
        #     'EQP_NAME',
        #     'CHAMBER_NAME',
        #     "WAFER_ID",
        #     'START_TIME',
        # # ).orderBy(F_col("START_TIME").asc())
        # df_spark.show()
        # print(f"df_spark shape: ({df_spark.count()}, {len(df_spark.columns)})")

        df_spark = df_spark.groupBy('PRODG1', 'PRODUCT_ID',  'EQP_NAME', 'CHAMBER_NAME', 'WAFER_ID').agg(count('WAFER_ID').alias("WAFER_COUNT"))
        df_spark.show()


    test_inline_go_through_time(spark)