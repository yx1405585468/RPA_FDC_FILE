import numpy as np
import pyspark
import pandas as pd

from pyspark.sql.functions import lit, col, concat_ws, split
from typing import List, Dict, Union
from src.exceptions.rca_base_exception import RCABaseException
from src.algo_config import GROUPED_ALGORITHMS

MODEL_NAME = GROUPED_ALGORITHMS.get("inline_by_wafer", "anova")


def get_remote_spark():
    import warnings
    import os
    from pyspark.sql import SparkSession

    # 本地pyspark client 链接服务器集群spark    os.environ["PYSPARK_PYTHON"] = "/usr/local/python-3.9.13/bin/python3"
    warnings.filterwarnings("ignore")

    os.environ["PYSPARK_PYTHON"] = "/usr/local/python-3.9.13/bin/python3"
    spark = (
        SparkSession.builder.appName("pandas_udf")
        .config("spark.sql.session.timeZone", "Asia/Shanghai")
        .config("spark.scheduler.mode", "FAIR")
        .config("spark.driver.memory", "8g")
        .config("spark.driver.cores", "12")
        .config("spark.executor.memory", "8g")
        .config("spark.executor.cores", "12")
        .config("spark.cores.max", "12")
        .config("spark.driver.host", "192.168.28.49")
        .master("spark://192.168.12.47:7077,192.168.12.48:7077")
        .getOrCreate()
    )
    # 添加本地依赖py文件
    spark.sparkContext.addPyFile(
        r"D:\xxs_project\2024\RCA根因分析\RCA_SPARK_JOB\src\utils\common_data_processing.py"
    )
    spark.sparkContext.addPyFile(
        r"D:\xxs_project\2024\RCA根因分析\RCA_SPARK_JOB\src\exceptions\rca_base_exception.py"
    )
    return spark


class DataPreprocessorForInline:
    def __init__(
        self,
        df: pyspark.sql.dataframe,
        grpby_list: list[str],
        columns_list: list[str],
        certain_column: str,
        key_words: list[str],
        convert_to_numeric_list: list[str],
    ):
        self.df = df
        self.grpby_list = grpby_list
        self.columns_list = columns_list
        self.certain_column = certain_column
        self.key_words = key_words
        self.convert_to_numeric_list = convert_to_numeric_list

    @staticmethod
    def select_columns(
        df: pyspark.sql.dataframe, columns_list: list[str]
    ) -> pyspark.sql.dataframe:
        return df.select(columns_list)

    @staticmethod
    def exclude_some_data(
        df: pyspark.sql.dataframe, key_words: list[str], certain_column: str
    ) -> pyspark.sql.dataframe:
        key_words_str = "|".join(key_words)
        df_filtered = df.filter(~col(certain_column).rlike(key_words_str))
        return df_filtered

    @staticmethod
    def pre_process(
        df: pyspark.sql.dataframe, convert_to_numeric_list: list[str]
    ) -> pyspark.sql.dataframe:
        for column in convert_to_numeric_list:
            df = df.withColumn(column, col(column).cast("double"))
        if "SITE_COUNT" in convert_to_numeric_list:
            convert_to_numeric_list.remove("SITE_COUNT")
        df = df.dropna(subset=convert_to_numeric_list, how="all")
        return df

    def run(self) -> pyspark.sql.dataframe:
        df_select = self.select_columns(df=self.df, columns_list=self.columns_list)
        df_esd = self.exclude_some_data(
            df=df_select, key_words=self.key_words, certain_column=self.certain_column
        )
        # df_integrate = self.integrate_columns(
        #     df=df_esd,
        #     merge_operno_list=self.merge_operno_list,
        #     merge_prodg1_list=self.merge_prodg1_list,
        #     merge_product_list=self.merge_product_list,
        # )
        # add_parametric_stats_df = self.add_feature_stats_within_groups(
        #     df_integrate=df_integrate, grpby_list=self.grpby_list
        # )
        df_preprocess = self.pre_process(
            df=df_esd, convert_to_numeric_list=self.convert_to_numeric_list
        )
        return df_preprocess


def unpivot(
    df,
    columns,
    val_type=None,
    index_name="uuid",
    feature_name="name",
    feature_value="value",
):
    """
    描述：对数据表进行反pivot操作

    :param df[DataFrame]:                 pyspark dataframe
    :param columns[List]:                 需要转换的列
    :param val_type[pyspark.sql.types]:   数据类型
    :param index_name[String]:            index column
    :param feature_name[String]:          特征列
    :param feature_value[String]:         数值列
    """
    if val_type is not None:
        df = df.select(index_name, *[col(c).cast(val_type()) for c in columns])

    stack_query = []
    for c in columns:
        stack_query.append(f"'{c}', `{c}`")

    df = df.selectExpr(
        f"`{index_name}`",
        f"stack({len(stack_query)}, {', '.join(stack_query)}) as (`{feature_name}`, `{feature_value}`)",
    ).orderBy(index_name, feature_name)
    return df


class ExertInlineByWafer:
    @staticmethod
    def fit_by_wafer_model(
        df: pyspark.sql.dataframe,
        request_id: str,
        merge_operno_list: List[Dict[str, List[str]]],
        merge_prodg1_list: List[Dict[str, List[str]]],
        merge_product_list: List[Dict[str, List[str]]],
        columns_list=None,
        key_words=None,
        convert_to_numeric_list=None,
        grpby_list=None,
        certain_column=None,
    ):
        try:
            from common_data_processing import (
                MergeOneColumnMultiValuesIntoNewOne,
                add_certain_column,
                GetScoreResultsByGroup,
            )
        except Exception:

            from src.utils.common_data_processing import (
                MergeOneColumnMultiValuesIntoNewOne,
                add_certain_column,
                GetScoreResultsByGroup,
            )

        if grpby_list is None or len(grpby_list) == 0:
            grpby_list = ["OPE_NO"]

        if columns_list is None:
            columns_list = grpby_list + [
                "WAFER_ID",
                "PARAMETRIC_NAME",
                "AVERAGE",
                "MAX_VAL",
                "MEDIAN",
                "MIN_VAL",
                "STD_DEV",
                "PERCENTILE_25",
                "PERCENTILE_75",
                "SITE_COUNT",
                "label",
            ]
        if key_words is None:
            key_words = ["CXS", "CYS", "FDS"]

        if convert_to_numeric_list is None:
            convert_to_numeric_list = [
                "AVERAGE",
                "MAX_VAL",
                "MEDIAN",
                "MIN_VAL",
                "STD_DEV",
                "PERCENTILE_25",
                "PERCENTILE_75",
                "SITE_COUNT",
            ]

        if certain_column is None:
            certain_column = "PARAMETRIC_NAME"

        # 合并功能
        df_interge = MergeOneColumnMultiValuesIntoNewOne.integrate_columns(
            df=df,
            OPE_NO=merge_operno_list,
            PRODUCT_ID=merge_product_list,
            PRODG1=merge_prodg1_list,
        )
        # 数据预处理
        df_preprocess = DataPreprocessorForInline(
            df=df_interge,
            grpby_list=grpby_list,
            columns_list=columns_list,
            certain_column=certain_column,
            key_words=key_words,
            convert_to_numeric_list=convert_to_numeric_list,
        ).run()

        # df_preprocess.show()
        # 宽表变长表
        df_unpivot = (
            df_preprocess.selectExpr(
                "WAFER_ID",
                *grpby_list,
                "PARAMETRIC_NAME",
                "label",
                """stack(5, 'MAX_VAL', `MAX_VAL`, 'MIN_VAL', `MIN_VAL`, 'AVERAGE', `AVERAGE`,
                              'MEDIAN', `MEDIAN`, 'STD_DEV', `STD_DEV` ) as (`STATS`, `RESULT`)""",
            )
            .dropna(subset="RESULT")
            .withColumn(
                "PARAMETRIC_NAME",
                concat_ws("#", col("PARAMETRIC_NAME"), lit("all_step"), col("STATS")),
            )
        )
        # df_unpivot.show()
        # 调用模型类获取得分
        result = GetScoreResultsByGroup.run(
            df_preprocessed=df_unpivot,
            grpby_list=grpby_list,
            model_name=MODEL_NAME,
            request_id=request_id,
        )

        if result is None:
            msg = f"按照{'+'.join(grpby_list)}分组后的数据无对照组!"
            raise RCABaseException(msg)
        else:

            return result.withColumn(
                "PARAMETRIC_NAME", split(col("PARAMETRIC_NAME"), "#", 0)[0]
            )


if __name__ == "__main__":
    import os
    import json
    from pyspark.sql import SparkSession
    import pyspark.pandas as ps
    import findspark
    import sys

    # sys.path.append("../../src")

    #
    # os.environ["PYSPARK_PYTHON"] = "/usr/local/python-3.9.13/bin/python3"
    # spark = (
    #     SparkSession.builder.appName("pandas_udf")
    #     .config("spark.sql.session.timeZone", "Asia/Shanghai")
    #     .config("spark.scheduler.mode", "FAIR")
    #     .config("spark.driver.memory", "8g")
    #     .config("spark.driver.cores", "12")
    #     .config("spark.executor.memory", "8g")
    #     .config("spark.executor.cores", "12")
    #     .config("spark.cores.max", "12")
    #     .config("spark.driver.host", "192.168.28.49")
    #     .master("spark://192.168.12.48:7077,192.168.12.47:7077")
    #     .getOrCreate()
    # )

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

    spark = get_remote_spark()

    # spark = get_local_spark()
    df_pandas = pd.read_csv(
        r"D:\xxs_project\2024\RCA根因分析\test_data\inline_case5_label_1.csv"
    )
    df_pandas = df_pandas[
        df_pandas["OPE_NO"].isin(
            # ["1V.EQW10", "1V.PQW10", "1F.FQE10", "1C.CDG10", "1U.EQW10", "1U.PQW10"]
            ["1U.EQW10"]
        )
    ]

    # df_pandas = df_pandas[df_pandas["PARAMETRIC_NAME"].isin([""])]
    df_pandas["label"] = 1
    df_spark = ps.from_pandas(df_pandas).to_spark()
    print(f"df_spark shape: ({df_spark.count()}, {len(df_spark.columns)})")
    df_spark.show()

    json_loads_dict = {
        "requestId": "269",
        "algorithm": "inline_by_wafer",
        "requestParam": {
            "dateRange": {"start": "2021-12-06 19:50:49", "end": "2024-03-06 19:50:49"},
            "operNo": [],
            "uploadId": "84f6a2b46a5443ec9797347424402058",
            "flagMergeAllProdg1": "0",
            "flagMergeAllProductId": "0",
            "flagMergeAllChamber": "0",
            "mergeProdg1": [],
            "mergeProductId": [],
            "mergeEqp": [],
            "mergeChamber": [],
            "mergeOperno": [
                {"xx1": ["1C.CDG10", "1V.EQW10", "1U.PQW10"]},
                {"xx2": ["1V.PQW10", "1F.FQE10"]},
            ],
            # "mergeOperno": None
        },
    }

    spark = get_remote_spark()
    df_ = pd.DataFrame(
        {
            "requestId": [json_loads_dict["requestId"]],
            "requestParam": [json.dumps(json_loads_dict["requestParam"])],
        }
    )

    request_id_ = df_["requestId"].values[0]
    request_params = df_["requestParam"].values[0]
    parse_dict = json.loads(request_params)

    merge_operno = (
        list(parse_dict.get("mergeOperno")) if parse_dict.get("mergeOperno") else None
    )
    merge_prodg1 = (
        list(parse_dict.get("mergeProdg1")) if parse_dict.get("mergeProdg1") else None
    )
    merge_product = (
        list(parse_dict.get("mergeProductId"))
        if parse_dict.get("mergeProductId")
        else None
    )
    grpby_list_ = ["OPE_NO"]
    # grpby_list_ = ['OPE_NO', 'PRODUCT_ID']

    from datetime import datetime

    time1 = datetime.now()
    print(time1)
    final_res_ = ExertInlineByWafer.fit_by_wafer_model(
        df=df_spark,
        request_id=request_id_,
        grpby_list=grpby_list_,
        merge_operno_list=merge_operno,
        merge_prodg1_list=merge_prodg1,
        merge_product_list=merge_product,
    )
    # final_res_.toPandas().to_csv("result.csv")
    time2 = datetime.now()
    print("算法运行得到最终结果消耗的时间是：", time2 - time1)
    print(f"算法结果一共有{final_res_.count()}条")
    final_res_.show(30)
    # final_res_pandas = final_res_.toPandas()
    # final_res_pandas.to_csv("final_res_pandas.csv")
