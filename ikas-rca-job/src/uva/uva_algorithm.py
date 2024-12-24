# 单变量uva 分析算法重构

import numpy as np
import pandas as pd
import pyspark.sql.dataframe
from pca import pca
from pyspark.sql.functions import (
    max,
)


from typing import List, Dict
from pyspark.sql.functions import col
from src.algo_config import GROUPED_ALGORITHMS


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


# 获取模型配置

MODEL_NAME = GROUPED_ALGORITHMS.get("uva_by_wafer", "anova")

# print("model name", MODEL_NAME)


class PreprocessForUvaData:
    def __init__(
        self,
        df: pyspark.sql.DataFrame,
        grpby_list: list[str],
        merge_operno_list: List[Dict[str, List[str]]],
        merge_prodg1_list: List[Dict[str, List[str]]],
        merge_product_list: List[Dict[str, List[str]]],
        merge_eqp_list: List[Dict[str, List[str]]],
        merge_chamber_list: List[Dict[str, List[str]]],
    ):
        self.df = df
        self.grpby_list = grpby_list
        self.merge_operno_list = merge_operno_list
        self.merge_prodg1_list = merge_prodg1_list
        self.merge_product_list = merge_product_list
        self.merge_eqp_list = merge_eqp_list
        self.merge_chamber_list = merge_chamber_list

    @staticmethod
    def pre_process(df: pyspark.sql.dataframe) -> pyspark.sql.DataFrame:
        """
        Preprocess the data extracted from the database for a specific CASE.
        :param df: Data for a specific CASE retrieved from the database.
        :return: Preprocessed data with relevant columns and filters applied.
        """
        # Select only the columns that will be used
        df = df.select(
            "WAFER_ID",
            "CHAMBER_ID",
            "RUN_ID",
            "EQP_NAME",
            "PRODUCT_ID",
            "PRODG1",
            "CHAMBER_NAME",
            "OPE_NO",
            "PARAMETRIC_NAME",
            "RESULT",
            "label",
        )
        # Remove rows with missing values in 'RESULT' column
        df = df.filter(col("RESULT").isNotNull())
        # Drop duplicates based on all columns
        df1 = df.dropDuplicates()
        # Select the rows with the latest 'RUN_ID' for each combination of 'WAFER_ID', 'OPER_NO', 'TOOL_ID'
        df2 = df1.groupBy("WAFER_ID", "OPE_NO", "CHAMBER_ID").agg(
            max("RUN_ID").alias("RUN_ID")
        )
        df_run = df1.join(
            df2.dropDuplicates(subset=["WAFER_ID", "OPE_NO", "CHAMBER_ID", "RUN_ID"]),
            on=["WAFER_ID", "OPE_NO", "CHAMBER_ID", "RUN_ID"],
            how="inner",
        )
        return df_run

    # def run(self):
    #     try:
    #         from src.utils.common_data_processing import (
    #             MergeOneColumnMultiValuesIntoNewOne,
    #             add_certain_column,
    #         )
    #     except Exception:
    #         from common_data_processing import (
    #             MergeOneColumnMultiValuesIntoNewOne,
    #             add_certain_column,
    #         )
    #
    #     # 合并按钮功能对应实现
    #     df_integrate_columns = MergeOneColumnMultiValuesIntoNewOne.integrate_columns(
    #         self.df,
    #         OPE_NO=self.merge_operno_list,
    #         PRODG1=self.merge_prodg1_list,
    #         PRODUCT_ID=self.merge_product_list,
    #         EQP_NAME=self.merge_eqp_list,
    #         CHAMBER_NAME=self.merge_chamber_list,
    #     )
    #
    #     # 使用# 拼接数据类型 (比如RANGE#MEAN)
    #     add_parametric_stats_df = self.add_feature_stats_within_groups(
    #         df_integrate=df_integrate_columns, grpby_list=self.grpby_list
    #     )
    #     # add_parametric_stats_df.show()
    #     # 数据预处理和共性分析
    #     df_run = self.pre_process(df_integrate_columns)
    #     df_run.persist()
    #     common_res = self.commonality_analysis(df_run, grpby_list=self.grpby_list)
    #
    #     return df_run, common_res, add_parametric_stats_df


class ExertUvaAlgorithm:
    @staticmethod
    def fit_uva_model(
        df: pyspark.sql.dataframe,
        grpby_list: List[str],
        request_id: str,
        merge_operno_list: List[Dict[str, List[str]]],
        merge_prodg1_list: List[Dict[str, List[str]]],
        merge_product_list: List[Dict[str, List[str]]],
        merge_eqp_list: List[Dict[str, List[str]]],
        merge_chamber_list: List[Dict[str, List[str]]],
    ):

        try:
            from rca_base_exception import RCABaseException
            from common_data_processing import (
                MergeOneColumnMultiValuesIntoNewOne,
                add_certain_column,
                GetScoreResultsByGroup,
            )

        except Exception:
            from src.exceptions.rca_base_exception import RCABaseException
            from src.utils.common_data_processing import (
                MergeOneColumnMultiValuesIntoNewOne,
                add_certain_column,
                GetScoreResultsByGroup,
            )

        # 数据预处理
        df_interger = MergeOneColumnMultiValuesIntoNewOne.integrate_columns(
            df=df,
            OPE_NO=merge_operno_list,
            PRODG1=merge_prodg1_list,
            PRODUCT_ID=merge_product_list,
            EQP_NAME=merge_eqp_list,
            CHAMBER_NAME=merge_chamber_list,
        )

        df_preprocessed = PreprocessForUvaData.pre_process(df_interger)
        result = GetScoreResultsByGroup.run(
            df_preprocessed=df_preprocessed,
            grpby_list=grpby_list,
            model_name=MODEL_NAME,
            request_id=request_id,
        )
        if result is None:
            msg = f"按照{'+'.join(grpby_list)}分组后的数据无对照组!"
            raise RCABaseException(msg)
        return result


if __name__ == "__main__":
    import os
    import json
    import warnings
    import pandas as pd
    import pyspark.pandas as ps
    from pyspark.sql import SparkSession

    def get_local_spark():
        import findspark

        spark_home = r"D:\software\spark-3.3.0-bin-hadoop3"

        python_path = r"D:\software\Anaconda3\envs\python39\python.exe"
        findspark.init(spark_home, python_path)
        # findspark.init()
        # os.environ['PYSPARK_PYTHON'] = "/usr/local/python-3.9.13/bin/python3"
        spark = (
            SparkSession.builder.appName("example")
            .master("local[4]")
            .config("spark.shuffle.file.buffer", "1000k")
            .config("spark.driver.memory", "8g")
            .config(
                "spark.sql.shuffle.partitions", "200"
            )  # 默认是 200，可以根据需要调整)
            .getOrCreate()
        )

        return spark

    spark = get_remote_spark()
    df_pandas = pd.read_csv(
        r"D:\xxs_project\2024\RCA根因分析\test_data\DWD_POC_CASE_FD_UVA_DATA_CASE1_PROCESSED1_1.csv"
    )
    # 测试小样本
    # df_pandas = df_pandas[
    #     df_pandas["PRODUCT_ID"].isin(
    #         ["AMKNXY01N.0A01", "AFGN4201N.0B01", "AFGN2T01N.0G01"]
    #     )
    # ]
    df_pandas["label"] = 1
    df_spark = ps.from_pandas(df_pandas).to_spark().repartition(200, "WAFER_ID")
    print(f"df_spark shape: ({df_spark.count()}, {len(df_spark.columns)})")
    df_spark.show()

    json_loads_dict = {
        "requestId": "uva",
        "requestParam": {
            "dateRange": [
                {"start": "2023-12-01 00:00:00", "end": "2024-01-15 00:00:00"}
            ],
            "lot": [],
            "operNo": [],
            "prodg1": [],
            "productId": [],
            "eqp": [],
            "tool": [],
            "recipeName": [],
            "waferId": {"good": [], "bad": []},
            "uploadId": "20240110170016023",
            "flagMergeAllProdg1": "0",
            "flagMergeAllProductId": "0",
            "flagMergeAllChamber": "0",
            "mergeProdg1": [],
            # "mergeProductId": [{"xx_cc": ["AFGNK401N.0A01", "AFGN1501N.0C02"]}],
            "mergeProductId": [],
            "mergeEqp": [],
            "mergeChamber": [],
            "mergeOperno": [],
            # 'mergeOperno': [{"2F.CDS10_XX.TDS01": ["2F.CDS10", "XX.TDS01"]},
            #                 {"2F.CDS20_XX.CDS20": ["2F.CDS20", "XX.CDS20"]}]
        },
    }
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
    merge_eqp = list(parse_dict.get("mergeEqp")) if parse_dict.get("mergeEqp") else None
    merge_chamber = (
        list(parse_dict.get("mergeChamber")) if parse_dict.get("mergeChamber") else None
    )
    grpby_list_ = ["OPE_NO", "PRODUCT_ID", "EQP_NAME"]
    # grpby_list_ = ['OPE_NO', 'CHAMBER_NAME']
    # grpby_list_ = ['PRODUCT_ID']

    from datetime import datetime

    time1 = datetime.now()
    print(time1)
    final_res_ = ExertUvaAlgorithm.fit_uva_model(
        df=df_spark,
        grpby_list=grpby_list_,
        request_id=request_id_,
        merge_operno_list=merge_operno,
        merge_prodg1_list=merge_prodg1,
        merge_product_list=merge_product,
        merge_eqp_list=merge_eqp,
        merge_chamber_list=merge_chamber,
    )
    time2 = datetime.now()
    if final_res_:
        print("算法结果写回数据库成功")
        print(f"算法结果一共有{final_res_.count()}条")
        print("算法结果写回数据库消耗的时间是：", time2 - time1)
        final_res_.show()
        # final_res_.toPandas().to_csv("final_res.csv")
        # train_data.toPandas().to_csv("train_data.csv")
        # print(final_res_.select("PARAMETRIC_NAME").head(1))
        # final_res_pandas = final_res_.toPandas()
        # final_res_pandas.to_csv("final_res_pandas_big1.csv")
