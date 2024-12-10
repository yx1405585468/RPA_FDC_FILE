"""
UVA 相关性分析

"""
import pyspark
from typing import List, Dict, Optional
from pyspark.sql.functions import max, countDistinct, when, lit, pandas_udf, PandasUDFType, monotonically_increasing_id, \
    split, collect_set, concat_ws, col, count, countDistinct, udf, mean
from pyspark.sql.types import StringType, IntegerType, DoubleType, StructType, StructField, FloatType

import pandas as pd
from src.correlation.common_process.corr_base_alogrithm import  CorrelationDetectAlgorithm
from src.correlation.common_process.data_preprocessing import ProcessBaseDFByWAFER


############################################UVA关系检测算法 look down ################################################
# 处理UVA数据接口类
class PreprocessForUvaData(object):
    @staticmethod
    def integrate_columns(df: pyspark.sql.DataFrame,
                          merge_operno_list: List[Dict[str, List[str]]],
                          merge_prodg1_list: List[Dict[str, List[str]]],
                          merge_product_list: List[Dict[str, List[str]]],
                          merge_eqp_list: List[Dict[str, List[str]]],
                          merge_chamber_list: List[Dict[str, List[str]]]) -> pyspark.sql.DataFrame:
        """
        Integrate columns in the DataFrame based on the provided list.

        :param df: The input DataFrame.
        :param merge_operno_list: A list of dictionaries where each dictionary contains values to be merged.
               Example: [{'2F.CDS10_XX.TDS01': ['2F.CDS10', 'XX.TDS01']},
                         {'2F.CDS20_XX.CDS20': ['2F.CDS20', 'XX.CDS20']}]
        :param merge_prodg1_list: A list of dictionaries for merging 'PRODG1' column in a similar fashion.
        :param merge_product_list: A list of dictionaries for merging 'PRODUCT_ID' column in a similar fashion.
        :param merge_eqp_list: A list of dictionaries for merging 'EQP_NAME' column in a similar fashion.
        :param merge_chamber_list: A list of dictionaries for merging 'TOOL_NAME' column in a similar fashion.

        :return: DataFrame with 'OPER_NO' and other specified columns integrated according to the merge rules.
        """
        df_merged = PreprocessForUvaData.integrate_single_column(df, merge_operno_list, 'OPE_NO')
        df_merged = PreprocessForUvaData.integrate_single_column(df_merged, merge_prodg1_list, 'PRODG1')
        df_merged = PreprocessForUvaData.integrate_single_column(df_merged, merge_product_list, 'PRODUCT_ID')
        df_merged = PreprocessForUvaData.integrate_single_column(df_merged, merge_eqp_list, 'EQP_NAME')
        df_merged = PreprocessForUvaData.integrate_single_column(df_merged, merge_chamber_list, 'CHAMBER_NAME')
        return df_merged

    @staticmethod
    def pre_process(df: pyspark.sql.dataframe) -> pyspark.sql.DataFrame:
        """
        Preprocess the data extracted from the database for a specific CASE.
        :param df: Data for a specific CASE retrieved from the database.
        :return: Preprocessed data with relevant columns and filters applied.
        """
        # Select only the columns that will be used
        df = df.select('WAFER_ID', 'CHAMBER_ID', 'RUN_ID', 'EQP_NAME', 'PRODUCT_ID', 'PRODG1', 'CHAMBER_NAME',
                       'OPE_NO', 'PARAMETRIC_NAME', 'RESULT')
        # Remove rows with missing values in 'STATISTIC_RESULT' column
        df = df.filter(col('RESULT').isNotNull())
        # Drop duplicates based on all columns
        df1 = df.dropDuplicates()
        # Select the rows with the latest 'RUN_ID' for each combination of 'WAFER_ID', 'OPER_NO', 'TOOL_ID'
        df2 = df1.groupBy('WAFER_ID', 'OPE_NO', 'CHAMBER_ID').agg(max('RUN_ID').alias('RUN_ID'))
        df_run = df1.join(df2.dropDuplicates(subset=['WAFER_ID', 'OPE_NO', 'CHAMBER_ID', 'RUN_ID']),
                          on=['WAFER_ID', 'OPE_NO', 'CHAMBER_ID', 'RUN_ID'], how='inner')
        return df_run

    @staticmethod
    def integrate_single_column(df: pyspark.sql.dataframe,
                                merge_list: List[Dict[str, List[str]]],
                                column_name: str) -> pyspark.sql.dataframe:
        """
        Integrate columns in the DataFrame based on the provided list.

        :param df: The input DataFrame.
        :param merge_list: A list of dictionaries where each dictionary contains values to be merged.
        :param column_name: The name of the column to be merged.

        :return: DataFrame with specified column integrated according to the merge rules.
        """
        splitter_comma = ","
        if merge_list is not None and len(merge_list) > 0:
            # Extract values from each dictionary in merge_operno_list and create a list
            values_to_replace = [list(rule.values())[0] for rule in merge_list]
            # Concatenate values from each dictionary
            merged_values = [splitter_comma.join(list(rule.values())[0]) for rule in merge_list]

            # Replace values in 'OPER_NO' column based on the rules defined in merge_operno_list
            for values, replacement_value in zip(values_to_replace, merged_values):
                df = df.withColumn(column_name,
                                   when(col(column_name).isin(values), replacement_value).otherwise(col(column_name)))
        return df

    @staticmethod
    def run(df: pyspark.sql.DataFrame,
            merge_operno_list: List[Dict[str, List[str]]],
            merge_prodg1_list: List[Dict[str, List[str]]],
            merge_product_list: List[Dict[str, List[str]]],
            merge_eqp_list: List[Dict[str, List[str]]],
            merge_chamber_list: List[Dict[str, List[str]]]):
        # 处理合并功能
        df_integrate_columns = PreprocessForUvaData.integrate_columns(df=df,
                                                                      merge_operno_list=merge_operno_list,
                                                                      merge_prodg1_list=merge_prodg1_list,
                                                                      merge_product_list=merge_product_list,
                                                                      merge_eqp_list=merge_eqp_list,
                                                                      merge_chamber_list=merge_chamber_list)
        # 选择列，去除parameterName 空值
        df_processed_df = PreprocessForUvaData.pre_process(df_integrate_columns)
        return df_processed_df


# UVA数据关系检测算法
class CorrCompareUvaByWaferAlgorithm(PreprocessForUvaData, ProcessBaseDFByWAFER, CorrelationDetectAlgorithm):

    @staticmethod
    def run(base_df: pyspark.sql.DataFrame, source_df: pyspark.sql.DataFrame, request_id: str,
            config_dict: dict) -> pyspark.sql.DataFrame:
        # 选择元素，处理None值
        select_config_dict = {
            k: list(v) if v is not None else list()  # optional类型 解包， None => 空列表
            for k, v in config_dict.items()
            if k in ['group_by_list', 'merge_operno_list', 'merge_prodg1_list', 'merge_product_list',
                     'merge_eqp_list', 'merge_chamber_list']
        }

        if len(select_config_dict['group_by_list']) == 0:
            raise ValueError("uva 数据分组参数为空,请检查")

        return CorrCompareUvaByWaferAlgorithm.__run_with_kwagrs__(base_df, source_df, request_id, **select_config_dict)

    @staticmethod
    def __run_with_kwagrs__(base_df: pyspark.sql.DataFrame,
                            source_df: pyspark.sql.DataFrame,
                            request_id: str,
                            group_by_list: List[str],
                            merge_operno_list: List[Dict[str, List[str]]],
                            merge_prodg1_list: List[Dict[str, List[str]]],
                            merge_product_list: List[Dict[str, List[str]]],
                            merge_eqp_list: List[Dict[str, List[str]]],
                            merge_chamber_list: List[Dict[str, List[str]]]
                            ):
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

        schema = StructType(
            [StructField(col_, StringType(), True) for col_ in group_by_list + ['PARAMETRIC_NAME']] +
            [
                StructField("WEIGHT", DoubleType(), True),
                StructField("COUNT", FloatType(), True),

            ])

        @pandas_udf(returnType=schema, functionType=PandasUDFType.GROUPED_MAP)
        def get_corr_weight(data: pd.DataFrame) -> pd.DataFrame:
            if data["COUNT"].values[0] == 1:
                weight = 0.0
            else:
                weight = CorrelationDetectAlgorithm.get_corr_func(x=data[outlier_parametric_name], y=data["RESULT"])
            # 返回一行分组字段 + parameter_name
            return data.head(1).assign(WEIGHT=weight).filter(
                items=group_by_list + ['PARAMETRIC_NAME', "WEIGHT", "COUNT"])

        # 异常表征预处理,获取异常表征参数列
        base_df, outlier_parametric_name = CorrCompareUvaByWaferAlgorithm.process_by_wafer(base_df)

        # 被比较的uva数据预处理
        df_processed = PreprocessForUvaData.run(source_df,
                                                merge_operno_list=merge_operno_list,
                                                merge_prodg1_list=merge_prodg1_list,
                                                merge_product_list=merge_product_list,
                                                merge_eqp_list=merge_eqp_list,
                                                merge_chamber_list=merge_chamber_list
                                                )

        # by eqp: 一个parameter 会存在重复的parameter name , RESULT
        df_processed = df_processed.groupby(*(group_by_list + ['PARAMETRIC_NAME', "WAFER_ID"])).agg(
            mean("RESULT").alias("RESULT")
        )
        # 基于wafer 拼接 uva 和 outlier
        result = df_processed.join(base_df, on='WAFER_ID', how='left')
        result = result.filter(col(outlier_parametric_name).isNotNull())

        # 计算x, y都不为空的行数 类似共性分析
        # 根据grblist 分组计算parmaterName 的非空值个数
        count_df = result.groupby(*(group_by_list + ["PARAMETRIC_NAME"])).agg(
            count(when((col('RESULT').isNotNull()) & (col(outlier_parametric_name).isNotNull()), 1)).alias("COUNT"),
        )
        count_df = count_df.filter(col("COUNT") > 0)

        if count_df.count() == 0:
            # 过滤后数据，为空
            return None

        result = result.join(count_df, on=group_by_list + ["PARAMETRIC_NAME"], how='left')

        # finish : 验证Count 数目，验证通过
        # result_pandas = result.toPandas()
        # for name, group in result_pandas.groupby((group_by_list + ["PARAMETRIC_NAME"])):
        #     print(name)
        #     print(group['COUNT'])
        #     print(len(group))
        #     break
        # 基于分组和参数列做关系检测
        result = result.groupby(
            *(group_by_list + ['PARAMETRIC_NAME'])
            # 'PRODG1', 'PRODUCT_ID', 'OPE_NO', 'EQP_NAME', 'CHAMBER_NAME', 'PARAMETRIC_NAME'
        ).apply(get_corr_weight)

        # 结果表处理:

        result = result.withColumn("split_text", split(result["PARAMETRIC_NAME"], "#"))
        result = (result
                  .withColumn("PARAMETRIC_NAME", concat_ws("#", result["split_text"][0], result["split_text"][1]))
                  .withColumn("STATS", result["split_text"][2])
                  .drop("split_text")
                  .withColumn("REQUEST_ID", lit(request_id))  # 赋值request_id 信息
                  # .withColumn('INDEX_NO', monotonically_increasing_id() + 1) #
                  .withColumn("ANALYSIS_NAME", lit("FDC"))  # 赋值分析类型

                  # PrePareResultTableToCommonSchema.run(result) 拼成总表格式，缺失的列，根据schema列类型赋值空值,
                  # .transform(PrePareResultTableToCommonSchema.run) # 放到MergeAllDataSourceToResultTable 统一处理
                  # .orderBy(col("WEIGHT").desc()) # 放在各模块合并后的结果处理中
                  )

        return result

if __name__ == "__main__":
    from src.correlation.common_process.data_preprocessing import MergeAllDataSourceToResultTable
    import pyspark.pandas as ps
    import pandas as pd
    from pyspark.sql import SparkSession
    import numpy as np


    def get_local_spark():
        import findspark
        spark_home = r"D:\software\spark-3.3.0-bin-hadoop3"

        python_path = r"D:\software\Anaconda3\envs\python39\python.exe"
        findspark.init(spark_home, python_path)
        # findspark.init()
        # os.environ['PYSPARK_PYTHON'] = "/usr/local/python-3.9.13/bin/python3"
        spark = SparkSession.builder.appName("example").getOrCreate()
        return spark

    def test_single_uva(spark):
        from pprint import pprint
        df_pandas = pd.read_csv(
            r"D:\xxs_project\2024\RCA根因分析\test_data\DWD_POC_CASE_FD_UVA_DATA_CASE1_PROCESSED1_1.csv")

        pprint(df_pandas.columns)

        # 异常表征 df
        base_df = df_pandas.query(
            "PRODG1 == 'L11CB14A' & PRODUCT_ID == 'AMKNS301N.0A01'"
            " & OPE_NO == '1F.EEK10' & EQP_NAME =='EKT72' & CHAMBER_NAME =='EKT72_PM1'"
            " & PARAMETRIC_NAME =='APC_POSITION#AOTU_STEP_2#MEAN'"

        )

        # df_spark = ps.from_pandas(base_df).to_spark()

        base_df['ParName'] = np.random.normal(size=len(base_df))
        base_df_spark = ps.from_pandas(base_df).to_spark().select("WAFER_ID", col("RESULT").alias("ParName"))
        # base_df_spark.show()

        df_spark = ps.from_pandas(df_pandas).to_spark()
        # print(f"df_spark shape: ({df_spark.count()}, {len(df_spark.columns)})")
        # df_spark.show()

        json_loads_dict = {
            "request_id": "uva",
            "uva": {
                "group_by_list": ['PRODG1', 'PRODUCT_ID', 'OPE_NO', 'EQP_NAME', "CHAMBER_NAME"],
                "merge_prodg1_list": None,
                "merge_product_list": [{"xx_cc": ["AFGNK401N.0A01", "AFGN1501N.0C02"]}],
                # "mergeProductId": None,
                "merge_eqp_list": [],
                "merge_chamber_list": [],
                "merge_operno_list": None,
                # 'mergeOperno': [{"2F.CDS10_XX.TDS01": ["2F.CDS10", "XX.TDS01"]},
                #                 {"2F.CDS20_XX.CDS20": ["2F.CDS20", "XX.CDS20"]}]
            }
        }

        final_res_ = CorrCompareUvaByWaferAlgorithm.run(base_df=base_df_spark,
                                                 source_df=df_spark,
                                                 config_dict=json_loads_dict.get("uva"),request_id="uva")

        final_res_.show()
        return final_res_

    spark = get_local_spark()
    final_res_ = test_single_uva(spark)
    final_res_.show()
