# encoding: utf-8
# 存放可复用的数据函数和类
import pyspark
from typing import List, Dict, Optional
from pyspark.sql.functions import (
    when,
    lit,
    col,
    first,
    row_number,
)

from src.exceptions.rca_base_exception import RCABaseException
from functools import reduce
from pyspark.sql.window import Window
import operator
import pyspark.sql.functions as F


#  处理by wafer 算法异常表征处理的接口类
class ProcessBaseDFByWAFER(object):
    """
    check处理by wafer 算法的异常表征dataframe
    """

    @staticmethod
    def process_by_wafer(
        df: pyspark.sql.DataFrame, first_column_name: str = "WAFER_ID"
    ):
        """
        df 字段
        -----------------
        WAFER_ID, | value

        """
        if df.isEmpty():
            raise RCABaseException("异常表征数据为空，请检查！")

        # 获取异常表征参数列
        base_cols = list(set(df.columns) - {first_column_name})

        if len(base_cols) == 1:
            base_col = base_cols[0]
        elif len(base_cols) == 0:
            raise RCABaseException("异常表征列只有一列WAFER_ID,没有参数列，请检查！")
        else:
            raise RCABaseException("异常表征参数列数目大于1，请检查！")
        return df.dropDuplicates([first_column_name]), base_col


class ProcessBaseDFBySite(object):
    """
    check处理by site 算法的异常表征dataframe
    df 字段：
    -----------------
    NAME, | value
    """

    @staticmethod
    def process_by_site(df: pyspark.sql.DataFrame):
        if df.isEmpty():
            raise RCABaseException("异常表征数据为空，请检查！")
        value_cols = list(set(df.columns) - {"NAME"})

        # print("base _df process show")
        df = df.withColumn("id", lit(1))
        # 去除缺失值，行转列
        df = (
            df.dropna().groupby("id").pivot("NAME").agg(first(value_cols[0])).drop("id")
        )
        # df.show()
        # base_cols:取出NAME的取值，转换为list
        base_cols = list(set(df.columns))
        return df, base_cols


# class ProcessBaseDFByZone(ProcessBaseDFBySite):
#     """
#     check处理by zone 算法的异常表征dataframe
#     df 字段：
#     -----------------
#     NAME, | value
#     zone1, 10.1
#     zone2, 10.2,
#     zone3, 10.5
#     """
#
#     @staticmethod
#     def process_by_zone(df: pyspark.sql.DataFrame):
#         return super(ProcessBaseDFByZone, ProcessBaseDFByZone).process_by_site(df)


# 结果表处理接口类: 基于约定的混合数据分析的总表字段， 补充处理各个模块分析结果缺失的列,赋值为空
class PrePareResultTableToCommonSchema(object):
    # 总表结构字段，和对应的类型
    schema: Dict[str, str] = {
        "REQUEST_ID": "string",
        "INDEX_NO": "int",
        "ANALYSIS_NAME": "string",
        "PRODG1": "string",
        "PRODUCT_ID": "string",
        "OPE_NO": "string",
        "EQP_NAME": "string",
        "CHAMBER_NAME": "string",
        "PARAMETRIC_NAME": "string",
        "STATS": "string",
        "COUNT": "float",
        "WEIGHT": "double",
        "MARK": "string",
    }

    @staticmethod
    def run(df: pyspark.sql.DataFrame) -> Optional[pyspark.sql.DataFrame]:

        if df is None:  # 去除不满足要求的数据，如果为空，返回None
            return None

        if df.isEmpty():  # 空表，返回None.
            return None

        df_columns = df.columns
        for col, col_type in PrePareResultTableToCommonSchema.schema.items():
            if col not in df_columns:
                df = df.withColumn(
                    col, lit(None).cast(col_type)
                )  # 按照schmea 指定类型赋值空列
        result = df.select(*(PrePareResultTableToCommonSchema.schema.keys()))
        return result


# 后处理接口类, 处理各个数据源拼成总表的形式，
class MergeAllDataSourceToResultTable(object):

    @staticmethod
    def run(df_list: List[Optional[pyspark.sql.DataFrame]]):

        empty_result_count = 0
        not_empty_df_list = []
        # 判断是否全为None 或者全为空表。
        for i, df in enumerate(df_list):
            if df is None:
                empty_result_count += 1
                continue

            if df.isEmpty():
                empty_result_count += 1
                continue

            # 处理成共同表结构的形式
            df = PrePareResultTableToCommonSchema.run(df)
            not_empty_df_list.append(df)

        if len(not_empty_df_list) == 0:
            raise RCABaseException(
                "所有数据源结果表都为空！,请检查每个分组相关性两列都非空的数目是不是为0"
            )

        if len(not_empty_df_list) == 1:
            result = not_empty_df_list[0]
        else:
            result = reduce(lambda x, y: x.union(y), not_empty_df_list)

        # 无须每个分量去除他们的和，归一化，每一个元素的含义为相似度取值 0-1
        result = (
            result.orderBy(col("WEIGHT").desc())
            # .withColumn("INDEX_NO", monotonically_increasing_id() + 1)
            .withColumn(
                "INDEX_NO",
                row_number().over(
                    Window.orderBy(col("WEIGHT").desc(), col("COUNT").desc())
                ),
            )
        )

        return result


class MergeOneColumnMultiValuesIntoNewOne(object):
    """
    实现合并按钮的功能：将一列中的多个取值，合并为一个值， 用来合并具体业务意义上的（分批）比如：产品，产品组.
    """

    # def integrate_columns(df: pyspark.sql.dataframe,
    #                       merge_operno_list: List[Dict[str, List[str]]],
    #                       merge_prodg1_list: List[Dict[str, List[str]]],
    #                       merge_product_list: List[Dict[str, List[str]]]) -> pyspark.sql.dataframe:
    #     """
    #     Integrate columns in the DataFrame based on the provided list.
    #
    #     :param df: The input DataFrame.
    #     :param merge_operno_list: A list of dictionaries where each dictionary contains values to be merged.
    #            Example: [{'2F.CDS10_XX.TDS01': ['2F.CDS10', 'XX.TDS01']},
    #                      {'2F.CDS20_XX.CDS20': ['2F.CDS20', 'XX.CDS20']}]
    #     :param merge_prodg1_list: A list of dictionaries for merging 'PRODG1' column in a similar fashion.
    #     :param merge_product_list: A list of dictionaries for merging 'PRODUCT_ID' column in a similar fashion.
    #
    #     :return: DataFrame with 'OPER_NO' and other specified columns integrated according to the merge rules.
    #     """
    #     df_merged = PreprocessForInlineData.integrate_single_column(df, merge_operno_list, 'OPE_NO')
    #     df_merged = PreprocessForInlineData.integrate_single_column(df_merged, merge_prodg1_list, 'PRODG1')
    #     df_merged = PreprocessForInlineData.integrate_single_column(df_merged, merge_product_list, 'PRODUCT_ID')
    #     return df_merged
    @staticmethod
    def integrate_columns(df: pyspark.sql.dataframe, **kwargs) -> pyspark.sql.dataframe:
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
        df_columns = df.columns
        for key, value in kwargs.items():
            if key in df_columns:
                df = MergeOneColumnMultiValuesIntoNewOne.integrate_single_column(
                    df, value, key
                )
            else:
                raise RCABaseException(
                    "The column: {} does not exist in the DataFrame".format(key)
                )
        return df

    @staticmethod
    def integrate_single_column(
        df: pyspark.sql.dataframe,
        merge_list: List[Dict[str, List[str]]],
        column_name: str,
    ) -> pyspark.sql.dataframe:
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
            merged_values = [
                splitter_comma.join(list(rule.values())[0]) for rule in merge_list
            ]

            for values, replacement_value in zip(values_to_replace, merged_values):
                df = df.withColumn(
                    column_name,
                    when(col(column_name).isin(values), replacement_value).otherwise(
                        col(column_name)
                    ),
                )
        return df


def compute_not_na_mean_by_row_func(columns: List[str]):
    """
    按照列方向，逐行计算指定列的非空，非NaN平均值。
    :param columns: 指定列的名称列表。
    :return: 返回一个pyspark 表达式

    等价于pandas df.apply(lambda x: x[columns].mean(),axis="columns")
    """

    count_value_dict = {
        "count_sum": reduce(
            operator.add,
            [
                F.when((~F.isnan(c)) & (F.col(c).isNotNull()), 1).otherwise(0)
                for c in columns
            ],
        ),
        "age_weight_sum": reduce(
            operator.add,
            [
                F.when((~F.isnan(c)) & (F.col(c).isNotNull()), F.col(c)).otherwise(0)
                for c in columns
            ],
        ),
    }

    return F.when(
        count_value_dict["count_sum"] > 0,
        count_value_dict["age_weight_sum"] / count_value_dict["count_sum"],
    ).otherwise(None)
