# encoding:utf-8
import pyspark
import pandas as pd
from pyspark.sql import SparkSession
from src.correlation.common_process.data_preprocessing import (
    ProcessBaseDFByWAFER,
    MergeOneColumnMultiValuesIntoNewOne,
)
from pyspark import StorageLevel
from src.correlation.common_process.corr_base_alogrithm import (
    CorrelationDetectAlgorithm,
)
from typing import List, Dict, Optional
from pyspark.sql.functions import count, PandasUDFType, pandas_udf, lit
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window


class CorrCompareQTimeByWaferAlgorithm(
    ProcessBaseDFByWAFER, CorrelationDetectAlgorithm
):

    @staticmethod
    def run(
        base_df: pyspark.sql.DataFrame,
        source_df: pyspark.sql.DataFrame,
        request_id: str,
        config_dict: dict,
    ) -> pyspark.sql.DataFrame:
        # 选择元素，处理None值
        select_config_dict = {
            k: (
                list(v) if v is not None else list()
            )  # optional类型 解包， None => 空列表
            for k, v in config_dict.items()
            if k
            in [
                "group_by_list",
                "merge_prodg1_list",
                "merge_product_list",
            ]
        }

        return CorrCompareQTimeByWaferAlgorithm.__run_with_kwargs__(
            base_df, source_df, request_id, **select_config_dict
        )

    @staticmethod
    def __run_with_kwargs__(
        base_df: pyspark.sql.DataFrame,
        source_df: pyspark.sql.DataFrame,
        request_id: str,
        group_by_list: List[str],
        merge_prodg1_list: List[Dict[str, List[str]]],
        merge_product_list: List[Dict[str, List[str]]],
    ) -> Optional[pyspark.sql.DataFrame]:

        # 异常数据处理
        base_df, base_col = ProcessBaseDFByWAFER.process_by_wafer(base_df)
        Q_TIME_COLUMN_NAME = "DIFF_TIME"
        # source_df 处理
        source_df = source_df.select(
            [
                "PRODG1",
                "PRODUCT_ID",
                "OPE_NO",
                "WAFER_ID",
                Q_TIME_COLUMN_NAME,
            ]
        ).dropna(how="any")
        # 处理合并选项
        merge_option_dict = {
            "PRODUCT_ID": merge_product_list,
            "PRODG1": merge_prodg1_list,
        }
        source_df = MergeOneColumnMultiValuesIntoNewOne.integrate_columns(
            source_df, **merge_option_dict
        )

        @pandas_udf(returnType=DoubleType())
        def get_corr_weight_scaler_func(x: pd.Series, y: pd.Series) -> float:
            return CorrelationDetectAlgorithm.get_corr_func(x=x, y=y)

        # 合并x, y,留下x,y两列都不为空的行
        result = source_df.join(base_df, on=["WAFER_ID"], how="left").dropna(
            subset=[Q_TIME_COLUMN_NAME, base_col], how="any"
        )
        result = result.withColumn(
            "COUNT", count("WAFER_ID").over(Window.partitionBy(*group_by_list))
        )
        source_df.unpersist()
        # print("group_by_list: \n", group_by_list)
        result = (
            result.groupby(*(group_by_list + ["COUNT"]))
            .agg(
                get_corr_weight_scaler_func(base_col, Q_TIME_COLUMN_NAME).alias(
                    "WEIGHT"
                )
            )
            .withColumn("STATS", lit("Q-TIME(分钟)"))
            .withColumn("REQUEST_ID", lit(request_id))  # 赋值request_id 信息
            .withColumn("ANALYSIS_NAME", lit("Q-TIME"))  # 赋值分析类型
        )
        result.persist(StorageLevel.MEMORY_AND_DISK)
        return result


if __name__ == "__main__":
    import time
    import findspark
    import pandas as pd
    import pyspark.pandas as ps
    import numpy as np
    import warnings

    warnings.filterwarnings("ignore")

    def get_local_spark():

        spark_home = r"D:\software\spark-3.3.0-bin-hadoop3"
        python_path = r"D:\software\Anaconda3\envs\python39\python.exe"
        findspark.init(spark_home, python_path)
        spark = (
            SparkSession.builder.appName("example")
            .config("master", "local[*]")
            .config(
                "spark.sql.execution.arrow.pyspark.enabled", "true"
            )  # 启用Arrow优化
            .getOrCreate()
        )

        return spark

    def timer(func):

        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"{func.__name__} 耗时 {end - start} seconds")
            return result

        return wrapper

    @timer
    def test_task_func(spark: SparkSession) -> pyspark.sql.DataFrame:
        df_pandas = pd.read_csv(
            r"D:\xxs_project\2024\RCA根因分析\test_data\inline_case5_label_1.csv"
        ).rename(columns={"AVERAGE": "DIFF_TIME"})
        df_spark = ps.from_pandas(df_pandas).to_spark().withColumn("PRODG1", lit("1C"))
        print(f"df_spark shape: ({df_spark.count()}, {len(df_spark.columns)})")
        df_spark.show()

        # 异常表征 df,from uva  => compare inline
        base_df = df_pandas.query("OPE_NO == '1C.CDG10' & PARAMETRIC_NAME == 'EEW0'")

        base_df_spark = ps.from_pandas(
            base_df[["WAFER_ID", "DIFF_TIME"]].rename(columns={"DIFF_TIME": "ParName"})
        ).to_spark()

        base_df_spark.show()

        json_loads_dict = {
            "request_id": "qtime",
            "qtime": {
                "group_by_list": [
                    "PRODG1",
                    "PRODUCT_ID",
                    "OPE_NO",
                ],
                "merge_prodg1_list": None,
                "merge_product_list": None,
                # "mergeProductId": None,
            },
        }
        df_spark.persist()
        final_res = CorrCompareQTimeByWaferAlgorithm.run(
            base_df=base_df_spark,
            source_df=df_spark,
            request_id="test_task_func",
            config_dict=json_loads_dict.get("qtime"),
        )
        return final_res

    # 传入spark Session
    spark = get_local_spark()
    # 测试函数耗时
    result = test_task_func(spark)
    result.orderBy("WEIGHT", ascending=False).show()
    spark.stop()
