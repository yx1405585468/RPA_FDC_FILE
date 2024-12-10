"""
Inline  相关性分析

"""

import pyspark
from typing import List, Dict, Optional
from pyspark.sql.functions import max, countDistinct, when, lit, pandas_udf, PandasUDFType, monotonically_increasing_id, \
    split, collect_set, concat_ws, col, count, countDistinct, udf, mean
from pyspark.sql.types import StringType, IntegerType, DoubleType, StructType, StructField, FloatType

import pandas as pd
from functools import partial, reduce


from src.correlation.common_process.corr_base_alogrithm import CorrelationDetectAlgorithm

from src.correlation.common_process.data_preprocessing import ProcessBaseDFByWAFER



class PreprocessForInlineData(object):
    def __init__(self,
                 df: pyspark.sql.dataframe,
                 group_by_list: list[str],
                 columns_list: list[str],
                 convert_to_numeric_list: list[str],
                 merge_operno_list: List[Dict[str, List[str]]],
                 merge_prodg1_list: List[Dict[str, List[str]]],
                 merge_product_list: List[Dict[str, List[str]]]
                 ):
        self.df = df
        self.groupby_list = group_by_list
        self.columns_list = columns_list
        # self.certain_column = certain_column
        # self.key_words = key_words
        self.convert_to_numeric_list = convert_to_numeric_list
        self.merge_operno_list = merge_operno_list
        self.merge_prodg1_list = merge_prodg1_list
        self.merge_product_list = merge_product_list


    @staticmethod
    def pre_process(df: pyspark.sql.dataframe, convert_to_numeric_list: list[str]) -> pyspark.sql.dataframe:
        for column in convert_to_numeric_list:
            df = df.withColumn(column, col(column).cast('double'))
        if 'SITE_COUNT' in convert_to_numeric_list:
            convert_to_numeric_list.remove('SITE_COUNT')
        df = df.dropna(subset=convert_to_numeric_list, how='all')
        return df

    @staticmethod
    def integrate_columns(df: pyspark.sql.dataframe,
                          merge_operno_list: List[Dict[str, List[str]]],
                          merge_prodg1_list: List[Dict[str, List[str]]],
                          merge_product_list: List[Dict[str, List[str]]]) -> pyspark.sql.dataframe:
        """
        Integrate columns in the DataFrame based on the provided list.

        :param df: The input DataFrame.
        :param merge_operno_list: A list of dictionaries where each dictionary contains values to be merged.
               Example: [{'2F.CDS10_XX.TDS01': ['2F.CDS10', 'XX.TDS01']},
                         {'2F.CDS20_XX.CDS20': ['2F.CDS20', 'XX.CDS20']}]
        :param merge_prodg1_list: A list of dictionaries for merging 'PRODG1' column in a similar fashion.
        :param merge_product_list: A list of dictionaries for merging 'PRODUCT_ID' column in a similar fashion.

        :return: DataFrame with 'OPER_NO' and other specified columns integrated according to the merge rules.
        """
        df_merged = PreprocessForInlineData.integrate_single_column(df, merge_operno_list, 'OPE_NO')
        df_merged = PreprocessForInlineData.integrate_single_column(df_merged, merge_prodg1_list, 'PRODG1')
        df_merged = PreprocessForInlineData.integrate_single_column(df_merged, merge_product_list, 'PRODUCT_ID')
        return df_merged

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
            values_to_replace = [list(rule.values())[0] for rule in merge_list]
            merged_values = [splitter_comma.join(list(rule.values())[0]) for rule in merge_list]

            for values, replacement_value in zip(values_to_replace, merged_values):
                df = df.withColumn(column_name,
                                   when(col(column_name).isin(values), replacement_value).otherwise(col(column_name)))
        return df

    @staticmethod
    def add_feature_stats_within_groups(df_integrate: pyspark.sql.dataframe, grpby_list) -> pyspark.sql.dataframe:
        unique_params_within_groups = (df_integrate.groupBy(grpby_list + ['PARAMETRIC_NAME'])
                                       .agg(
            countDistinct('WAFER_ID', when(df_integrate['label'] == 0, 1)).alias('GOOD_NUM'),
            countDistinct('WAFER_ID', when(df_integrate['label'] == 1, 1)).alias('BAD_NUM'))
                                       .na.fill(0))
        return unique_params_within_groups

    def run(self) -> pyspark.sql.dataframe:
        df_select = self.df.select(self.columns_list)
        # df_esd = self.exclude_some_data(df=df_select, key_words=self.key_words, certain_column=self.certain_column)
        df_integrate = self.integrate_columns(df=df_select,
                                              merge_operno_list=self.merge_operno_list,
                                              merge_prodg1_list=self.merge_prodg1_list,
                                              merge_product_list=self.merge_product_list)

        df_preprocess = self.pre_process(df=df_integrate, convert_to_numeric_list=self.convert_to_numeric_list)
        return df_preprocess


class CorrCompareInlineByWaferAlgorithm(PreprocessForInlineData, ProcessBaseDFByWAFER, CorrelationDetectAlgorithm):

    @staticmethod
    def run(base_df: pyspark.sql.DataFrame, source_df: pyspark.sql.DataFrame, request_id: str,
            config_dict: dict) -> pyspark.sql.DataFrame:
        # 选择元素，处理None值
        select_config_dict = {
            k: list(v) if v is not None else list()  # optional类型 解包， None => 空列表
            for k, v in config_dict.items()
            if k in ['group_by_list', 'merge_operno_list', 'merge_prodg1_list', 'merge_product_list',
                     ]
        }

        if len(select_config_dict['group_by_list']) == 0:
            raise ValueError("Inline 数据分组参数为空,请检查")

        return CorrCompareInlineByWaferAlgorithm.__run_with_kwargs__(base_df, source_df, request_id,
                                                                     **select_config_dict)

    @staticmethod
    def __run_with_kwargs__(base_df: pyspark.sql.DataFrame,
                            source_df: pyspark.sql.DataFrame,
                            request_id: str,
                            group_by_list: List[str],
                            merge_operno_list: List[Dict[str, List[str]]],
                            merge_prodg1_list: List[Dict[str, List[str]]],
                            merge_product_list: List[Dict[str, List[str]]],

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

        schema = StructType(
            [StructField(col_, StringType(), True) for col_ in group_by_list + ['PARAMETRIC_NAME']] +
            [
                StructField("WEIGHT", DoubleType(), True),
                StructField("COUNT", FloatType(), True),

            ])

        def get_corr_weight_func_with_col(stat_col: str):
            @pandas_udf(returnType=schema, functionType=PandasUDFType.GROUPED_MAP)
            def get_corr_weight(data: pd.DataFrame) -> pd.DataFrame:
                if data["COUNT"].values[0] == 1:
                    weight = 0.0
                else:
                    weight = CorrelationDetectAlgorithm.get_corr_func(x=data[outlier_parametric_name], y=data[stat_col])
                # 返回一行分组字段 + parameter_name
                return data.head(1).assign(WEIGHT=weight).filter(
                    items=group_by_list + ['PARAMETRIC_NAME', "WEIGHT", "COUNT"])

            return get_corr_weight

        def deal_each_column(result: pyspark.sql.DataFrame, stat_col: str) -> pyspark.sql.DataFrame:

            # 计算x, y都不为空的行数 类似共性分析
            # 根据grblist 分组计算parmaterName 的非空值个数
            result = (
                result.select(group_by_list + ["PARAMETRIC_NAME", 'WAFER_ID', stat_col] + [outlier_parametric_name])
                .dropna(subset=group_by_list + ["PARAMETRIC_NAME", stat_col, outlier_parametric_name], how='any')
                )

            if result.count() == 0:
                return None

            count_df = result.groupby(*(group_by_list + ["PARAMETRIC_NAME"])).agg(
                count(when((col(stat_col).isNotNull()) & (col(outlier_parametric_name).isNotNull()), 1)).alias("COUNT"),
            )

            # 分组比较的两列只有一行都不为空，weight 为0
            # count_df_len_one = count_df.filter(col("COUNT") == 1).withColumn("WEIGHT", lit(0.0))
            #
            # print("---------------------count df len zero show--------------------")
            # count_df.filter(col("COUNT") == 0).show()
            #
            #
            # count_df = count_df.filter(col("COUNT") > 1)

            if result.count() == 0:
                # 过滤后数据，为空
                return None

            result = result.join(count_df, on=group_by_list + ["PARAMETRIC_NAME"], how='left')

            # 基于分组和参数列做关系检测
            result = result.groupby(
                *(group_by_list + ['PARAMETRIC_NAME'])

            ).apply(get_corr_weight_func_with_col(stat_col))

            # if count_df_len_one.count() > 0:
            #     result = result.union(count_df_len_one)

            # 结果表处理:
            # print("count >1 or count =1 ")
            # result.filter(col("COUNT").isNull()).show()

            result = (result
                      .withColumn("STATS", lit(stat_col))
                      .withColumn("REQUEST_ID", lit(request_id))  # 赋值request_id 信息
                      # .withColumn('INDEX_NO', monotonically_increasing_id() + 1)  #
                      .withColumn("ANALYSIS_NAME", lit("INLINE"))  # 赋值分析类型
                      # PrePareResultTableToCommonSchema.run(result) 拼成总表格式，缺失的列，根据schema列类型赋值空值,
                      # .transform(PrePareResultTableToCommonSchema.run)
                      # .orderBy(col("WEIGHT").desc()) # 放在各模块合并后的结果处理中
                      )

            return result

        # 异常表征预处理,获取异常表征参数列
        base_df, outlier_parametric_name = ProcessBaseDFByWAFER.process_by_wafer(base_df)

        # 被比较的Inline数据预处理
        if len(group_by_list) == 0:
            group_by_list = ['OPE_NO']

        columns_list = group_by_list + ['WAFER_ID', 'PARAMETRIC_NAME', 'AVERAGE', 'MAX_VAL', 'MEDIAN', 'MIN_VAL',
                                        'STD_DEV']
        convert_to_numeric_list = ['AVERAGE', 'MAX_VAL', 'MEDIAN', 'MIN_VAL', 'STD_DEV']

        df_processed = PreprocessForInlineData(source_df,
                                               columns_list=columns_list,
                                               group_by_list=group_by_list,
                                               convert_to_numeric_list=convert_to_numeric_list,
                                               merge_operno_list=merge_operno_list,
                                               merge_prodg1_list=merge_prodg1_list,
                                               merge_product_list=merge_product_list).run()

        # by eqp: 一个parameter 会存在重复的parameter name , RESULT
        df_processed = df_processed.groupby(*(group_by_list + ['PARAMETRIC_NAME', "WAFER_ID"])).agg(
            *(
                mean(col_name).alias(col_name) for col_name in convert_to_numeric_list
            )
        )
        # 基于wafer 拼接 uva 和 outlier
        result_all_stat_cols = df_processed.join(base_df, on='WAFER_ID', how='left')


        # 对bywafer 多个聚合值，一个一个处理，最后拼接
        result_list = [deal_each_column(result_all_stat_cols, stat_col) for stat_col in convert_to_numeric_list]

        # result_list 中pyspark dataframe 元素纵向合并
        result_list = [res for res in result_list if res is not None and res.count() > 0]

        # 纵向拼接多列关系检测的结果表
        if len(result_list) == 0:
            return None
        elif len(result_list) == 1:
            return result_list[0]
        else:
            result = reduce(lambda x, y: x.union(y), result_list)

        return result


########################################################Inline关系检测算法 lookup #######################################
if __name__ == "__main__":
    from src.correlation.common_process.data_preprocessing import MergeAllDataSourceToResultTable
    import pyspark.pandas as ps
    import pandas as pd
    from pyspark.sql import SparkSession


    def get_local_spark():
        import findspark
        spark_home = r"D:\software\spark-3.3.0-bin-hadoop3"

        python_path = r"D:\software\Anaconda3\envs\python39\python.exe"
        findspark.init(spark_home, python_path)
        # findspark.init()
        # os.environ['PYSPARK_PYTHON'] = "/usr/local/python-3.9.13/bin/python3"
        spark = SparkSession.builder.appName("example").getOrCreate()
        return spark

    def test_single_inline(spark):
        df_pandas = pd.read_csv(
            r"D:\xxs_project\2024\RCA根因分析\test_data\inline_case5_label_1.csv")

        df_spark = ps.from_pandas(df_pandas).to_spark()
        print(f"df_spark shape: ({df_spark.count()}, {len(df_spark.columns)})")
        df_spark.show()

        uva_pandas = pd.read_csv(
            r"D:\xxs_project\2024\RCA根因分析\test_data\DWD_POC_CASE_FD_UVA_DATA_CASE1_PROCESSED1_1.csv")

        # 异常表征 df,from uva  => compare inline
        base_df = df_pandas.query(
            "OPE_NO == '1C.CDG10' & PARAMETRIC_NAME == 'EEW0'"
        )

        base_df_spark = ps.from_pandas(
            base_df[['WAFER_ID', 'AVERAGE']].rename(columns={'AVERAGE': 'ParName'})).to_spark()
        base_df_spark.show()
        #
        # join_result = df_pandas.merge(base_df[['WAFER_ID', 'AVERAGE']].rename(columns={'AVERAGE': 'ParName'}), on=['WAFER_ID'], how='left')
        #
        # # print("join 结果\n", join_result.groupby(['OPE_NO', 'PARAMETRIC_NAME_x'])['AVERAGE'].agg(lambda x: x.isna().sum()))
        # pprint(join_result[["AVERAGE", 'ParName', "WAFER_ID"]])

        # exit(0)

        json_loads_dict = {"requestId": "269",

                           "inline": {
                               "group_by_list": ["PRODUCT_ID", "OPE_NO"],
                               "merge_prodg1_list": None,
                               "merge_product_list": None,
                               # "merge_eqp_list": [],
                               # "merge_chamber": [],

                               # "merge_operno_list": [{"xx1": ["1C.CDG10", "1V.EQW10", "1U.PQW10"]},
                               #                 {"xx2": ["1V.PQW10", "1F.FQE10"]}],
                               "merge_operno_list": [],
                               # "mergeOperno": None
                           }
                           }

        final_result = CorrCompareInlineByWaferAlgorithm.run(base_df=base_df_spark, source_df=df_spark,
                                                             config_dict=json_loads_dict.get("inline"),
                                                             request_id="269")
        #
        # pprint(df_spark.dtypes)


        return final_result


    spark = get_local_spark()
    final_result = test_single_inline(spark)
    final_result = MergeAllDataSourceToResultTable.run( [final_result])
    final_result.show()