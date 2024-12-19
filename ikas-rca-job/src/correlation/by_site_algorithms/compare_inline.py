"""
异常表征数据 与 inline 数据 by site 关系检测算法
"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
# from src.correlation.common_process.data_preprocessing import (PrePareResultTableToCommonSchema,
#                                                                ProcessBaseDFBySite)

# from src.correlation.common_process.corr_base_alogrithm import CorrelationDetectAlgorithm

import numpy as np
from typing import Callable
from sklearn.metrics import r2_score, mean_squared_error
import pyspark
from typing import List, Dict, Optional
from pyspark.sql.functions import max, countDistinct, when, lit, pandas_udf, PandasUDFType, monotonically_increasing_id, \
    split, collect_set, concat_ws, col, count, countDistinct, udf, mean, stddev, min, percentile_approx
from pyspark.sql.types import StringType, IntegerType, DoubleType, StructType, StructField, FloatType
# from src.exceptions.rca_base_exception import RCABaseException
import pandas as pd
from functools import partial, reduce

from pyspark.sql import SparkSession

# from src.correlation.by_wafer_algorithms.compare_inline import PreprocessForInlineData
import logging

R2_THRESHOLD = 0.0  # R2 阈值，超过阈值，返回r2,没超过返回0.
# 设置日志级别和输出
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def algo_detect_relation(x: pd.Series, y: pd.Series, r2_threshold: float, is_cv=True) -> float:
    """
    检测两组异源数据是否存在关系,使用机器学习模型。
    """
    try:
        # 传进来的数据类型为Decimal 需要转为float, 不然，不能直接相加
        x = x.astype(float)
        y = y.astype(float)

    except ValueError:
        # 存在非数字类型字符串‘123ab‘，需要转为float,'123abc'不能转为数字类型的需要置为Nan
        x = pd.to_numeric(x, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')

    # 拼接去除缺失值
    small_df = pd.DataFrame({"x": x, "y": y}).dropna()

    #  判断x, y 两列 是否存在一列全是一个值，方差为 0.
    if small_df['x'].nunique() == 1 or small_df['y'].nunique() == 1:
        return 0.0

    count = len(small_df)
    if count < 3:
        # count 取 0 x,y 同时不为空的一行也没有
        # count 取1 x,y同时不为空的行只有1，结果不可信
        return 0.0
    else:
        # 计算所有关系中的最大r2
        r2_max = 0.0

        for func_name, func in DETECT_CORRELATION_ML_FUNC_MAP.items():
            r2 = func(small_df['x'].values, small_df['y'].values, is_cv=is_cv)
            # print(f"func_name: {func_name}, r2: {r2}")
            # 选出最大的r2
            if r2 > r2_max:
                r2_max = r2

        # logger.info(f"r2_max: {r2_max}")
        # 确定r2 在 0-1 范围内
        if r2_max < 0 or r2_max > 1:
            return 0.0
        # 根据r2判段关系是否在设定关系中
        if r2_max >= r2_threshold:
            # 关系在DETECT_CORRELATION_FUNC_MAP之中
            return r2_max
        else:
            # 关系不在DETECT_CORRELATION_FUNC_MAP之中， 置为0.
            return 0.0


def get_cv_fold(num_samples: int) -> int:
    """
    根据样本数量，计算交叉验证的折数
    """
    if num_samples == 3:
        return 1
    elif num_samples == 4 or num_samples == 5:
        # 3 个样本，但是，验证集r2计算需要两个样本
        return 2
    elif num_samples <= 21:
        return 3
    else:
        return 5


def check_linear_relationship_with_ml(X, y, fit_intercept=True, is_cv=True):
    """
    测试X和y之间是否存在线性关系
    """

    model = LinearRegression(fit_intercept=fit_intercept)
    if is_cv:
        cv = get_cv_fold(num_samples=len(y))
        model = LinearRegression(fit_intercept=fit_intercept)
        if cv > 1:
            # logger.info(f'cv fold: {cv}')

            scores = cross_val_score(model, X.reshape(-1, 1), y, cv=cv, scoring='r2')  # 假设X是一维数组
            mean_score = np.mean(scores)
        else:
            # cv ==1,只有三个样本，
            model.fit(X.reshape(-1, 1)[:1], y[:1])
            mean_score = r2_score(y[1:], model.predict(X.reshape(-1, 1)[1:]))
    else:
        model.fit(X.reshape(-1, 1), y)
        mean_score = r2_score(y, model.predict(X.reshape(-1, 1)))
    return mean_score  # 假设R^2大于0.8表示存在线性关系


def check_quadratic_relationship_with_ml(X, y, fit_intercept=True, is_cv=True):
    """
    测试X和y之间是否存在平方关系
    """

    # logger.info(f"num_samples : {len(y)} cv fold:{cv}")
    model = make_pipeline(PolynomialFeatures(2), LinearRegression(fit_intercept=fit_intercept))  # 平方特征
    if is_cv:
        cv = get_cv_fold(num_samples=len(y))
        if cv > 1:
            scores = cross_val_score(model, X.reshape(-1, 1), y, cv=cv, scoring='r2')
            mean_score = np.mean(scores)
        else:
            # cv ==1,只有三个样本，
            model.fit(X.reshape(-1, 1)[:1], y[:1])
            mean_score = r2_score(y[1:], model.predict(X.reshape(-1, 1)[1:]))
    else:
        model.fit(X.reshape(-1, 1), y)
        mean_score = r2_score(y, model.predict(X.reshape(-1, 1)))
    return mean_score  # 假设R^2大于0.8表示存在平方关系


def check_sqrt_relationship_with_ml(X, y, fit_intercept=True, is_cv=True):
    """
    测试X和y之间是否存在根号关系
    注意：在实际情况下，根号关系可能不是直接测试，而是通过变换数据来间接测试。

    y = sqrt(kx+b  ) => sqrt(y) =kx+b
    """
    if y.min() < 0:
        # 存在负数，无法做根号运算
        return 0.0
    return check_linear_relationship_with_ml(X, np.sqrt(y), is_cv=is_cv, fit_intercept=fit_intercept)


DETECT_CORRELATION_ML_FUNC_MAP: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "linear": check_linear_relationship_with_ml,  # 线性关系
    "quadratic": check_quadratic_relationship_with_ml,  # 二次项关系
    "sqrt": check_sqrt_relationship_with_ml,  # # 根号关系
    "linear_no_intercept": partial(check_linear_relationship_with_ml, fit_intercept=False),  # 线性关系
    "quadratic_no_intercept": partial(check_quadratic_relationship_with_ml, fit_intercept=False),  # 二次项关系
    "sqrt_no_intercept": partial(check_sqrt_relationship_with_ml, fit_intercept=False),  # # 根号关系
}


class RCABaseException(Exception):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return f"Task failed.Message: {self.message}"


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


class CorrelationDetectAlgorithm(object):

    @staticmethod
    def get_corr_func(x: pd.Series, y: pd.Series) -> float:
        corr_func = partial(algo_detect_relation, r2_threshold=R2_THRESHOLD)
        return corr_func(x, y)


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
        df = df.withColumn('id', lit(1))
        # 去除缺失值，行转列
        df = df.dropna().groupby("id").pivot("NAME").agg(first(value_cols[0])).drop("id")
        # df.show()
        # base_cols:取出NAME的取值，转换为list
        base_cols = list(set(df.columns))
        return df, base_cols


class PrePareResultTableToCommonSchema(object):
    # 总表结构字段，和对应的类型
    schema: Dict[str, str] = {
        "REQUEST_ID": "string",
        "INDEX_NO": "int",
        "ANALYSIS_NAME": "string",
        "PRODG1": 'string',
        "PRODUCT_ID": "string",
        "OPE_NO": "string",
        "EQP_NAME": "string",
        "CHAMBER_NAME": "string",
        "PARAMETRIC_NAME": "string",
        "STATS": "string",
        "COUNT": "float",
        "WEIGHT": "double",
        "MARK": "string"
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
                df = df.withColumn(col, lit(None).cast(col_type))  # 按照schmea 指定类型赋值空列
        result = df.select(*(PrePareResultTableToCommonSchema.schema.keys()))
        return result


class CorrCompareInlineBySiteAlgorithm(PreprocessForInlineData):
    stat_func_dict = {
        "AVERAGE": mean,
        # "STD_DEV": stddev,
        # "MIN_VAL": min,
        # "MAX_VAL": max,
        # "MEDIAN": partial(percentile_approx, percentage=0.5),
    }

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

        return CorrCompareInlineBySiteAlgorithm.__run_with_kwargs__(base_df, source_df, request_id,
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
        # 被比较的Inline数据预处理
        if len(group_by_list) == 0:
            group_by_list = ['OPE_NO']

        schema = StructType(
            [StructField(col_, StringType(), True) for col_ in group_by_list + ['PARAMETRIC_NAME']] +
            [
                StructField("COUNT", FloatType(), True),
                StructField("WEIGHT", DoubleType(), True),

            ])

        def get_corr_weight_func_with_col(stat_col: str, base_site_columns: List[str], source_site_columns: List[str]):
            @pandas_udf(returnType=schema, functionType=PandasUDFType.GROUPED_MAP)
            def get_corr_weight(data: pd.DataFrame) -> pd.DataFrame:
                # logger.info(f"get_corr_weight_func_with_col {data.to_json()}")
                if len(data) == 1:
                    x = data[group_by_list + ['PARAMETRIC_NAME', 'COUNT', ] + base_site_columns].rename(
                        columns={

                            col: f"{stat_col}_{col.replace('BASE_', '')}"
                            for col in base_site_columns
                        }
                    ).copy()

                    y = data[group_by_list + ['PARAMETRIC_NAME', 'COUNT', ] + source_site_columns].copy()
                    # 一行分成两段，两段纵向拼成两行
                    data = pd.concat([x, y], ignore_index=True)
                    # 删除存在空值的列
                    data = data.dropna(subset=source_site_columns, how='any')
                    common_not_null_cols = list(data.columns.difference(group_by_list + ['PARAMETRIC_NAME', 'COUNT']))


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
                    # logger.info(f"group_by_list:{group_by_list}")
                    weight = CorrelationDetectAlgorithm.get_corr_func(x=x,
                                                                      y=y)
                # 返回一行分组字段 + parameter_name
                return (data.head(1)
                .filter(items=group_by_list + ['PARAMETRIC_NAME', "COUNT"])
                .assign(WEIGHT=weight)
                .filter(
                    items=group_by_list + ['PARAMETRIC_NAME', "COUNT", "WEIGHT", ])
                )

            return get_corr_weight

        def deal_each_column(result: pyspark.sql.DataFrame, stat_col: str, base_site_columns: List[str],
                             source_site_columns: List[str]) -> pyspark.sql.DataFrame:

            # 计算x, y都不为空的行数 类似共性分析
            # 根据grblist 分组计算parmaterName 的非空值个数
            # by site 一种统计方法的提取
            stat_source_site_columns = [f'{stat_col}_{col_name}' for col_name in source_site_columns]
            result = (
                result.select(
                    group_by_list + ["PARAMETRIC_NAME", "COUNT"] + base_site_columns + stat_source_site_columns)
                # .dropna(subset=base_site_columns, how='all')
                .dropna(subset=stat_source_site_columns, how='all')
            )

            # print("stat rusult show ")
            # result.show(10, False)
            if result.count() == 0:
                return None

            # 基于分组和参数列做关系检测
            # todo:调整pandas udf
            result = result.groupby(
                *(group_by_list + ['PARAMETRIC_NAME'])

            ).apply(get_corr_weight_func_with_col(stat_col, base_site_columns=base_site_columns,
                                                  source_site_columns=stat_source_site_columns))

            result = (result
                      .withColumn("STATS", lit(stat_col))
                      .withColumn("REQUEST_ID", lit(request_id))  # 赋值request_id 信息
                      # .withColumn('INDEX_NO', monotonically_increasing_id() + 1)  #
                      .withColumn("ANALYSIS_NAME", lit("INLINE"))  # 赋值分析类型
                      # PrePareResultTableToCommonSchema.run(result) 拼成总表格式，缺失的列，根据schema列类型赋值空值,
                      .transform(PrePareResultTableToCommonSchema.run)
                      # .orderBy(col("WEIGHT").desc()) # 放在各模块合并后的结果处理中
                      )

            return result

        # 异常表征预处理,获取异常表征参数列

        base_df, base_site_columns = ProcessBaseDFBySite.process_by_site(base_df)
        # print("base_df show")
        # base_df.show(10)
        # 寻找共同的site columns
        common_site_columns = list(set(base_site_columns) & set(list(source_df.columns)))

        columns_list = group_by_list + ['WAFER_ID', 'PARAMETRIC_NAME'] + common_site_columns

        df_processed = PreprocessForInlineData(source_df,
                                               columns_list=columns_list,
                                               group_by_list=group_by_list,
                                               convert_to_numeric_list=common_site_columns,
                                               merge_operno_list=merge_operno_list,
                                               merge_prodg1_list=merge_prodg1_list,
                                               merge_product_list=merge_product_list).run()

        # by eqp: 一个parameter 会存在重复的parameter name , RESULT
        df_processed = df_processed.groupby(*(group_by_list + ['PARAMETRIC_NAME', 'WAFER_ID'])).agg(
            *(
                mean(col_name).alias(col_name) for col_name in common_site_columns
            )
        )

        # print("df_processed show")
        # df_processed.show(10)

        # # todo:from here
        # # 基于wafer 拼接 inline 和 异常表征， 因为都是聚合后的数据，每个分组的数据如下所示：
        """
        +----+-----------------+---------+----------+
        group, SITE1_VAL, SITE2_VAL, ..., BASE_SITE1_VAL, BASE_SITE2_VAL, ..., 
        1,    10.0,        20.0, ...,      50.0,       60.0, ... 
        2,    10.0,        20.0, ...,      50.0,       60.0,  ... 
        """
        # 大表
        result_all_stat_cols = (df_processed
        .groupby(*(group_by_list + ['PARAMETRIC_NAME']))
        .agg(
            # count wafer_ID
            count('WAFER_ID').alias("COUNT"),

            # 分别计算 sitecolumns的聚合值： ['AVERAGE', 'MAX_VAL', 'MEDIAN', 'MIN_VAL', 'STD_DEV']
            *(
                stat_func(col_name).alias(f'{stat_func_name}_{col_name}') for col_name in common_site_columns
                for stat_func_name, stat_func in CorrCompareInlineBySiteAlgorithm.stat_func_dict.items()
            )
        )
        )

        if result_all_stat_cols.count() == 0:
            return None
        # 小表
        base_df_pandas = base_df.toPandas()
        base_site_cols = []

        # 异常表征的列赋值到source_df
        for col_name in common_site_columns:
            result_all_stat_cols = result_all_stat_cols.withColumn(f'BASE_{col_name}',
                                                                   lit(base_df_pandas[col_name].values[0]).cast(
                                                                       'float'))
            base_site_cols.append(f'BASE_{col_name}')

        # print("result_all_stat_cols show")
        # result_all_stat_cols.show(10)

        # 计算bysite 的指标

        # 对bywafer 多个聚合值，一个一个处理，最后拼接
        result_list = [deal_each_column(result_all_stat_cols, stat_col, base_site_columns=base_site_cols,
                                        source_site_columns=common_site_columns)
                       for stat_col in CorrCompareInlineBySiteAlgorithm.stat_func_dict.keys()
                       ]

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


if __name__ == "__main__":
    import pyspark.pandas as ps
    import os
    import warnings
    from pprint import pprint

    warnings.filterwarnings("ignore")


    def get_local_spark():
        import findspark
        spark_home = r"D:\software\spark-3.3.0-bin-hadoop3"

        python_path = r"D:\software\Anaconda3\envs\python39\python.exe"
        findspark.init(spark_home, python_path)
        # findspark.init()
        # os.environ['PYSPARK_PYTHON'] = "/usr/local/python-3.9.13/bin/python3"
        spark = SparkSession.builder.appName("example").config("master", "local[4]").getOrCreate()

        return spark


    def test_single_inline(spark):
        df_pandas = pd.read_csv(
            r"D:\xxs_project\2024\RCA根因分析\test_data\inline_case5_label_1.csv")

        # 异常表征 df,from uva  => compare inline
        base_df = df_pandas.query("OPE_NO == '1F.FQE10' and PARAMETRIC_NAME == 'FDS1'")
        df_spark = ps.from_pandas(df_pandas[df_pandas['WAFER_ID'].isin(base_df['WAFER_ID'].values.tolist())]).to_spark()
        print(f"df_spark shape: ({df_spark.count()}, {len(df_spark.columns)})")
        df_spark.show()

        select_site_cols = []
        for col_name in base_df.columns:
            if col_name.startswith("SITE") and col_name.endswith("VAL"):
                select_site_cols.append(col_name)

        print('------------------------------')
        print(pd.DataFrame(base_df
                           .filter(items=select_site_cols)

                           .apply(lambda x: x.mean(), axis=0)
                           ).rename(columns={0: 'value'})
              .reset_index(drop=False).rename(columns={'index': 'NAME'})

              )

        base_df_spark = ps.from_pandas(
            pd.DataFrame(base_df
                         .filter(items=select_site_cols)

                         .apply(lambda x: x.mean(), axis=0)
                         ).rename(columns={0: 'value'})
            .reset_index(drop=False).rename(columns={'index': 'NAME'})

        ).to_spark()
        base_site_columns = select_site_cols

        base_df_spark.show()

        #
        # base_df_spark = base_df_spark.agg(
        #     *(
        #         mean(col_name).alias(col_name) for col_name in base_site_columns
        #     )
        # )
        # base_df_spark.show()

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
                               "base_site_columns": base_site_columns
                           }
                           }

        final_result = CorrCompareInlineBySiteAlgorithm.run(base_df=base_df_spark, source_df=df_spark,
                                                            config_dict=json_loads_dict.get("inline"),
                                                            request_id="269")

        final_result = final_result.orderBy(col("WEIGHT").desc(), col("COUNT").desc())
        # #
        # # pprint(df_spark.dtypes)
        final_result.show()
        #
        return final_result


    def test_single_inline_new(spark):
        df_pandas = pd.read_csv(r"D:\xxs_project\2024\RCA根因分析\test_data\data\INLINE.csv"
                                )

        df_spark = ps.from_pandas(df_pandas).to_spark()
        df_spark.show()
        base_df = (
            pd.read_csv(
                r"D:\xxs_project\2024\RCA根因分析\test_data\data\异常表征.csv"
            )
        )
        base_df_spark = ps.from_pandas(base_df).to_spark()
        base_df_spark.show()

        #
        # base_df_spark = base_df_spark.agg(
        #     *(
        #         mean(col_name).alias(col_name) for col_name in base_site_columns
        #     )
        # )
        # base_df_spark.show()

        json_loads_dict = {"requestId": "269",

                           "inline": {
                               "group_by_list": ["PRODG1", "PRODUCT_ID", "OPE_NO"],
                               "merge_prodg1_list": None,
                               "merge_product_list": None,
                               # "merge_eqp_list": [],
                               # "merge_chamber": [],

                               # "merge_operno_list": [{"xx1": ["1C.CDG10", "1V.EQW10", "1U.PQW10"]},
                               #                 {"xx2": ["1V.PQW10", "1F.FQE10"]}],
                               "merge_operno_list": [],
                               # "mergeOperno": None
                               # "base_site_columns":base_df.NAME.tolist()
                           }
                           }

        final_result = CorrCompareInlineBySiteAlgorithm.run(base_df=base_df_spark, source_df=df_spark,
                                                            config_dict=json_loads_dict.get("inline"),
                                                            request_id="269")

        final_result = final_result.orderBy(col("WEIGHT").desc(), col("COUNT").desc())
        # #
        # # pprint(df_spark.dtypes)
        final_result.show()
        #
        return final_result


    spark = get_local_spark()
    res = test_single_inline(spark)
    # res.toPandas().to_csv("result.csv")
