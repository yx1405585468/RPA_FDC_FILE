"""
异常表征数据 与 WAT by site 数据关系检测
"""
from pyspark import StorageLevel

from src.correlation.common_process.corr_base_alogrithm import (
    CorrelationDetectAlgorithm,
)

from src.correlation.common_process.data_preprocessing import (
    PrePareResultTableToCommonSchema,
    ProcessBaseDFBySite,
    MergeAllDataSourceToResultTable,
)

import numpy as np
from typing import Callable
from sklearn.metrics import r2_score, mean_squared_error
import pyspark
from typing import List, Dict, Optional
from pyspark.sql.functions import (
    max,
    countDistinct,
    when,
    lit,
    pandas_udf,
    PandasUDFType,
    monotonically_increasing_id,
    split,
    collect_set,
    concat_ws,
    col,
    count,
    countDistinct,
    udf,
    mean,
    stddev,
    min,
    percentile_approx,
)
from pyspark.sql.types import (
    StringType,
    IntegerType,
    DoubleType,
    StructType,
    StructField,
    FloatType,
)
from src.exceptions.rca_base_exception import RCABaseException
import pandas as pd
from functools import partial, reduce

from pyspark.sql import SparkSession

from src.correlation.by_wafer_algorithms.compare_wat import PreprocessForWatData
import logging

# 设置日志级别和输出
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorrCompareWatBySiteAlgorithm(PreprocessForWatData, CorrelationDetectAlgorithm):

    stat_func_dict = {
        "AVERAGE": mean,
        # "STD_DEV": stddev,
        # "MIN_VAL": min,
        # "MAX_VAL": max,
        # "MEDIAN": partial(percentile_approx, percentage=0.5),
    }

    @staticmethod
    def run(
        base_df: pyspark.sql.DataFrame,
        source_df: pyspark.sql.DataFrame,
        request_id: str,
        config_dict: dict,
    ) -> pyspark.sql.DataFrame:
        # 选择元素，处理None值
        select_config_dict = {
            k: list(v) if v is not None else list()  # optional类型 解包， None => 空列表
            for k, v in config_dict.items()
            if k
            in [
                "group_by_list",
                "merge_operno_list",
                "merge_prodg1_list",
                "merge_product_list",
                "merge_parametric_name_list"
            ]
        }

        # if len(select_config_dict['group_by_list']) == 0:
        #     raise ValueError("Inline 数据分组参数为空,请检查")

        return CorrCompareWatBySiteAlgorithm.__run_with_kwargs__(
            base_df, source_df, request_id, **select_config_dict
        )

    @staticmethod
    def __run_with_kwargs__(
        base_df: pyspark.sql.DataFrame,
        source_df: pyspark.sql.DataFrame,
        request_id: str,
        group_by_list: List[str],
        merge_operno_list: List[Dict[str, List[str]]],
        merge_prodg1_list: List[Dict[str, List[str]]],
        merge_product_list: List[Dict[str, List[str]]],
        merge_parametric_name_list: List[Dict[str, List[str]]],
    ) -> Optional[pyspark.sql.DataFrame]:
        """
        比较UVA数据by wafer的相关性分析算法

        处理步骤：
        输入:
            1. 异常表征数据，
                wafer, outerlier_parameterName
                w01,     3.1
                w02,     3.2,
                w03,     3.3,

            2. UVA 数据：
                分组字段,wafer, parameterName,result

            3. 关于wafer 拼接 异常表征数据和UVA 数据， 基于UVA数据分组字段，检测异常表征与UVA parameterName 的关系检测

                分组字段，parmeterName, count, weight

            4. rename 列名， 按照多模块分析结果总表字段汇总，没有的列，值为None，
        """
        # 被比较的Inline数据预处理

        if group_by_list is None or len(group_by_list) == 0:
            source_df = source_df.withColumn("OPE_NO", lit("FAKE_OPE_NO"))
            group_by_list = ["OPE_NO"]
            is_fake_openo = True
        else:
            is_fake_openo = False

        schema = StructType(
            [
                StructField(col_, StringType(), True)
                for col_ in group_by_list + ["PARAMETRIC_NAME"]
            ]
            + [
                StructField("COUNT", FloatType(), True),
                StructField("WEIGHT", DoubleType(), True),
            ]
        )

        def get_corr_weight_func_with_col(
            stat_col: str, base_site_columns: List[str], source_site_columns: List[str]
        ):
            @pandas_udf(returnType=schema, functionType=PandasUDFType.GROUPED_MAP)
            def get_corr_weight(data: pd.DataFrame) -> pd.DataFrame:
                # logger.info(f"get_corr_weight_func_with_col {data.to_json()}")
                if len(data) == 1:
                    x = (
                        data[
                            group_by_list
                            + ["PARAMETRIC_NAME", "COUNT"]
                            + base_site_columns
                        ]
                        .rename(
                            columns={
                                col: f"{stat_col}_{col.replace('BASE_', '')}"
                                for col in base_site_columns
                            }
                        )
                        .copy()
                    )

                    y = data[
                        group_by_list
                        + ["PARAMETRIC_NAME", "COUNT"]
                        + source_site_columns
                    ].copy()
                    # 一行分成两段，两段纵向拼成两行
                    data = pd.concat([x, y], ignore_index=True)
                    # 删除存在空值的列
                    data = data.dropna(subset=source_site_columns, how="any")
                    common_not_null_cols = list(
                        data.columns.difference(
                            group_by_list + ["PARAMETRIC_NAME", "COUNT"]
                        )
                    )

                else:
                    raise RCABaseException("pandas udf 传入的一个分组中还有一个分组,请检查")

                if len(common_not_null_cols) <= 1:
                    weight = 0.0
                else:
                    # 计算相关性
                    x = pd.Series(x[common_not_null_cols].values.ravel().tolist())
                    y = pd.Series(y[common_not_null_cols].values.ravel().tolist())

                    # temp_data = pd.DataFrame({"x": x, "y": y})
                    # temp_data.dropna(inplace=True)
                    # logger.info(f"json(temp_data):{temp_data.to_json()}")
                    # 计算相关性系数

                    # logger.info(f"x:{x},  y:{y}")

                    weight = CorrelationDetectAlgorithm.get_corr_func(x=x, y=y)
                # 返回一行分组字段 + parameter_name
                return (
                    data.head(1)
                    .filter(items=group_by_list + ["PARAMETRIC_NAME", "COUNT"])
                    .assign(WEIGHT=weight)
                    .filter(
                        items=group_by_list
                        + [
                            "PARAMETRIC_NAME",
                            "COUNT",
                            "WEIGHT",
                        ]
                    )
                )

            return get_corr_weight

        def deal_each_column(
            result: pyspark.sql.DataFrame,
            stat_col: str,
            base_site_columns: List[str],
            source_site_columns: List[str],
        ) -> pyspark.sql.DataFrame:

            # 计算x, y都不为空的行数 类似共性分析
            # 根据grblist 分组计算parmaterName 的非空值个数
            # by site 一种统计方法的提取
            stat_source_site_columns = [
                f"{stat_col}_{col_name}" for col_name in source_site_columns
            ]
            result = (
                result.select(
                    group_by_list
                    + ["PARAMETRIC_NAME", "COUNT"]
                    + base_site_columns
                    + stat_source_site_columns
                )
                # .dropna(subset=base_site_columns, how='all')
                .dropna(subset=stat_source_site_columns, how="all")
            )

            # print("stat rusult show ")
            # result.show(10, False)
            if result.count() == 0:
                return None

            # 基于分组和参数列做关系检测
            # todo:调整pandas udf
            result = result.groupby(*(group_by_list + ["PARAMETRIC_NAME"])).apply(
                get_corr_weight_func_with_col(
                    stat_col,
                    base_site_columns=base_site_columns,
                    source_site_columns=stat_source_site_columns,
                )
            )

            result = (
                result.withColumn("STATS", lit(stat_col)).withColumn(
                    "REQUEST_ID", lit(request_id)
                )  # 赋值request_id 信息
                # .withColumn('INDEX_NO', monotonically_increasing_id() + 1)  #
                .withColumn("ANALYSIS_NAME", lit("WAT"))  # 赋值分析类型
                # PrePareResultTableToCommonSchema.run(result) 拼成总表格式，缺失的列，根据schema列类型赋值空值,
                # .transform(PrePareResultTableToCommonSchema.run)
                # .orderBy(col("WEIGHT").desc()) # 放在各模块合并后的结果处理中
            )
            result.persist(StorageLevel.MEMORY_AND_DISK)
            return result

        # 异常表征预处理,获取异常表征参数列

        base_df, base_site_columns = ProcessBaseDFBySite.process_by_site(base_df)
        # print("base_df show")
        # base_df.show(10)
        # 寻找共同的site columns
        common_site_columns = list(
            set(base_site_columns) & set(list(source_df.columns))
        )

        columns_list = (
            group_by_list + ["WAFER_ID", "PARAMETRIC_NAME"] + common_site_columns
        )

        df_processed = PreprocessForWatData(
            source_df,
            columns_list=columns_list,
            group_by_list=group_by_list,
            convert_to_numeric_list=common_site_columns,
            merge_operno_list=merge_operno_list,
            merge_prodg1_list=merge_prodg1_list,
            merge_product_list=merge_product_list,
            merge_parametric_name_list=merge_parametric_name_list
        ).run()

        # by eqp: 一个parameter 会存在重复的parameter name , RESULT
        df_processed = df_processed.groupby(
            *(group_by_list + ["PARAMETRIC_NAME", "WAFER_ID"])
        ).agg(*(mean(col_name).alias(col_name) for col_name in common_site_columns))

        # print("df_processed show")
        # df_processed.show(10)
        df_processed.persist(StorageLevel.MEMORY_AND_DISK)
        source_df.unpersist()
        # # todo:from here
        # # 基于wafer 拼接 inline 和 异常表征， 因为都是聚合后的数据，每个分组的数据如下所示：
        """
        +----+-----------------+---------+----------+
        group, SITE1_VAL, SITE2_VAL, ..., BASE_SITE1_VAL, BASE_SITE2_VAL, ..., 
        1,    10.0,        20.0, ...,      50.0,       60.0, ... 
        2,    10.0,        20.0, ...,      50.0,       60.0,  ... 
        """
        # 大表
        result_all_stat_cols = df_processed.groupby(
            *(group_by_list + ["PARAMETRIC_NAME"])
        ).agg(
            # wafer count
            count("WAFER_ID").alias("COUNT"),
            # 分别计算 sitecolumns的聚合值： ['AVERAGE', 'MAX_VAL', 'MEDIAN', 'MIN_VAL', 'STD_DEV']
            *(
                stat_func(col_name).alias(f"{stat_func_name}_{col_name}")
                for col_name in common_site_columns
                for stat_func_name, stat_func in CorrCompareWatBySiteAlgorithm.stat_func_dict.items()
            ),
        )
        result_all_stat_cols.persist(StorageLevel.MEMORY_AND_DISK)  # 缓存数据集
        if result_all_stat_cols.count() == 0:
            df_processed.unpersist()  # 释放不再使用的数据集
            return None
        # 小表
        base_df_pandas = base_df.toPandas()
        base_site_cols = []

        # 异常表征的列赋值到source_df
        for col_name in common_site_columns:
            result_all_stat_cols = result_all_stat_cols.withColumn(
                f"BASE_{col_name}",
                lit(base_df_pandas[col_name].values[0]).cast("float"),
            )
            base_site_cols.append(f"BASE_{col_name}")

        # 计算bysite 的指标
        # 对bywafer 多个聚合值，一个一个处理，最后拼接
        result_list = [
            deal_each_column(
                result_all_stat_cols,
                stat_col,
                base_site_columns=base_site_cols,
                source_site_columns=common_site_columns,
            )
            for stat_col in CorrCompareWatBySiteAlgorithm.stat_func_dict.keys()
        ]
        # result_list 中pyspark dataframe 元素纵向合并
        result_list = [
            res for res in result_list if res is not None and res.count() > 0
        ]

        # 纵向拼接多列关系检测的结果表
        if len(result_list) == 0:
            df_processed.unpersist()  # 释放不再使用的数据集
            result_all_stat_cols.unpersist()  # 释放不再使用的数据集
            return None
        elif len(result_list) == 1:
            result = result_list[0]
        else:
            result = reduce(lambda x, y: x.union(y), result_list)

        # logger.info("is_fake_openo: {}".format(is_fake_openo))
        if is_fake_openo is True:
            result = result.drop("OPE_NO")
        df_processed.unpersist()  # 释放不再使用的数据集
        result_all_stat_cols.unpersist()  # 释放不再使用的数据集
        return result


if __name__ == "__main__":
    import pyspark.pandas as ps
    import os
    import warnings
    from pprint import pprint

    warnings.filterwarnings("ignore")

    def get_local_spark():
        import findspark
        import warnings

        warnings.filterwarnings("ignore")
        spark_home = r"D:\software\spark-3.3.0-bin-hadoop3"
        python_path = r"D:\software\Anaconda3\envs\python39\python.exe"
        findspark.init(spark_home, python_path)
        # findspark.init()
        # os.environ['PYSPARK_PYTHON'] = "/usr/local/python-3.9.13/bin/python3"
        spark = (
            SparkSession.builder.appName("example")
            .config("master", "local[4]")
            .getOrCreate()
        )
        return spark

    def test_single_wat(spark):
        import time

        start = time.time()
        df_pandas = pd.read_csv(r"D:\xxs_project\2024\RCA根因分析\test_data\wat_select.csv")
        # df_pandas = df_pandas[df_pandas['PRODUCT_ID'].isin(
        #     ["AEMNRM01N.0B01", "AEMNE801N.0B01", "AFXNE001N.0C01", "AGKNCE01N.0A01", "AFXNJ701N.0B01"])]
        # df_pandas = df_pandas[df_pandas['PRODUCT_ID'].isin(["AEMNRM01N.0B01", "AEMNE801N.0B01"])]
        base_df = df_pandas.query(
            """
            OPE_NO == 'ST.TTS10' & PARAMETRIC_NAME == 'BVjLN+LN+_I_p14_10nA' & PRODUCT_ID == 'AEMNRM01N.0B01'
            """
        )

        # 根据wafer id 过滤
        df_spark = ps.from_pandas(
            df_pandas[df_pandas["WAFER_ID"].isin(base_df["WAFER_ID"].unique())]
        ).to_spark()
        print(f"df_spark shape: ({df_spark.count()}, {len(df_spark.columns)})")
        df_spark.show()
        print(base_df)
        select_site_cols = []
        for col_name in base_df.columns:
            if col_name.startswith("SITE") and col_name.endswith("VAL"):
                select_site_cols.append(col_name)

        from pprint import pprint

        pprint(select_site_cols)

        base_df_spark = ps.from_pandas(
            (
                pd.DataFrame(
                    base_df.filter(items=select_site_cols).apply(
                        lambda x: x.mean(), axis=0
                    )
                )
                .rename(columns={0: "value"})
                .reset_index(drop=False)
                .rename(columns={"index": "NAME"})
            )
        ).to_spark()

        base_df_spark.show()

        json_loads_dict = {
            "requestId": "269",
            "wat": {
                "group_by_list": None,
                "merge_product_list": None,
                # "mergeProductId": [
                #     {"xx1": ["AEMNRM01N.0B01", "AEMNE801N.0B01"]},
                #     {"xx2": ["AGKNCE01N.0A01", "AFXNJ701N.0B01"]}],
                "merge_prodg1_list": None,
                "merge_operno_list": None,
            },
        }

        final_result = CorrCompareWatBySiteAlgorithm.run(
            base_df=base_df_spark,
            source_df=df_spark,
            config_dict=json_loads_dict.get("wat"),
            request_id="269",
        )

        final_result.show()
        print("映射为all data schema:列数为{}".format(len(final_result.columns)))
        final_res = MergeAllDataSourceToResultTable.run([final_result])
        print("映射为all data schema:列数为{}".format(len(final_res.columns)))
        final_res.show()

        end = time.time()

        # remote spark time cost 65 s
        # no dropletions all 耗时
        print(f"remote spark  time cost: {end - start}s")

        return final_result

    # def test_single_wat_new(spark):
    #     import time
    #
    #     start = time.time()
    #     df_pandas = pd.read_csv(r"D:\xxs_project\2024\RCA根因分析\test_data\admin_P2P_FILTER_20240708151657\WAT.csv"
    #                             )
    #
    #     df_spark = ps.from_pandas(df_pandas).to_spark()
    #     df_spark.show()
    #     base_df = (
    #         pd.read_csv(
    #             r"D:\xxs_project\2024\RCA根因分析\test_data\admin_P2P_FILTER_20240708151657\anomaly.csv"
    #         )
    #     )
    #     base_df_spark = ps.from_pandas(base_df).to_spark()
    #     base_df_spark.show()
    #     json_loads_dict = {"requestId": "269",
    #
    #                        "wat": {
    #                            'group_by_list': ["PRODG1", "PRODUCT_ID", "OPE_NO"],
    #                            "merge_product_list": None,
    #                            # "mergeProductId": [
    #                            #     {"xx1": ["AEMNRM01N.0B01", "AEMNE801N.0B01"]},
    #                            #     {"xx2": ["AGKNCE01N.0A01", "AFXNJ701N.0B01"]}],
    #
    #                            "merge_prodg1_list": None,
    #                            "merge_operno_list": None
    #                        }
    #                        }
    #
    #     final_result = CorrCompareWatBySiteAlgorithm.run(base_df=base_df_spark, source_df=df_spark,
    #                                                       config_dict=json_loads_dict.get("wat"), request_id="269")
    #     print("映射为all data schema:列数为{}".format(len(final_result.columns)))
    #     final_res = MergeAllDataSourceToResultTable.run([final_result])
    #     print("映射为all data schema:列数为{}".format(len(final_res.columns)))
    #     final_res.show()
    #
    #     end = time.time()
    #
    #     # remote spark time cost 65 s
    #     # no dropletions all 耗时
    #     print(f"remote spark  time cost: {end - start}s")
    #
    #     return final_result
    # # result = test_single_wat(spark)
    spark = get_local_spark()
    spark.read.text(r"D:\xxs_project\2024\RCA根因分析\test_data\test.txt").show()
    result1 = test_single_wat(spark)
    # print(result1.count())
