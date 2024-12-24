# encoding: utf-8
# 存放可复用的数据函数和类
import pyspark
from typing import List, Dict, Optional
from pyspark.sql.functions import (
    col,
    countDistinct,
    lit,
    pandas_udf,
    PandasUDFType,
    monotonically_increasing_id,
    split,
    collect_set,
    concat_ws,
    mean,
    when,
)
from pyspark.sql.types import (
    StringType,
    IntegerType,
    FloatType,
    StructType,
    StructField,
)
from pca import pca
import numpy as np

from functools import reduce
import pandas as pd

from pyspark.sql.types import FloatType, StringType
from pyspark.sql.window import Window
import operator
import pyspark.sql.functions as F
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.under_sampling import ClusterCentroids
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import (
    SelectPercentile,
    f_classif,
    chi2,
    mutual_info_classif,
)

#
# from src.exceptions.rca_base_exception import RCABaseException


class MergeOneColumnMultiValuesIntoNewOne(object):
    """
    实现合并按钮的功能：将一列中的多个取值，合并为一个值， 用来合并具体业务意义上的（分批）比如：产品，产品组.
    """

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
            # else:
            #     raise RCABaseException(
            #         "The column: {} does not exist in the DataFrame".format(key)
            #     )
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


def get_pipe_params(model):
    common_steps = [
        ("imputer", SimpleImputer(strategy="constant", fill_value=-999)),
        ("scaler", StandardScaler()) if model != "chi2" else ("scaler", MinMaxScaler()),
    ]
    models = {
        "rf": (
            RandomForestClassifier(random_state=2024),
            {
                "model__n_estimators": [*range(10, 60, 10)],
                "model__max_depth": [*range(5, 50, 10)],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 3],
            },
        ),
        "decisionTree": (
            DecisionTreeClassifier(random_state=2024),
            {
                "model__max_depth": [None, 5, 10, 15],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        "svc": (
            LinearSVC(random_state=2024, fit_intercept=False),
            {
                "model__loss": ["hinge", "squared_hinge"],
                "model__C": [0.1, 0.5, 1, 10, 50],
            },
        ),
        "logistic": (
            LogisticRegression(
                random_state=2024, fit_intercept=False, solver="liblinear"
            ),
            {"model__penalty": ["l1", "l2"], "model__C": [0.1, 0.5, 1, 10, 50]},
        ),
        "sgd": (
            SGDClassifier(random_state=2024, fit_intercept=False),
            {
                "model__loss": ["hinge", "log_loss", "perceptron", "huber"],
                "model__penalty": ["l1", "l2", "elasticnet", None],
                "model__alpha": [0.0001, 0.001, 0.01, 0.1],
                "model__max_iter": [100, 500, 1000],
            },
        ),
        # 非参检验 - 卡方检验
        "chi2": (
            SelectPercentile(chi2, percentile=100),
            None,
        ),
        # 方差分析
        "anova": (
            SelectPercentile(f_classif, percentile=100),
            None,
        ),
        # 互信息
        "mutual_info": (
            SelectPercentile(mutual_info_classif, percentile=100),
            None,
        ),
    }

    if model in models:
        model_class, param_grid = models[model]
        steps = common_steps + [("model", model_class)]
        pipe = Pipeline(steps)
        return pipe, param_grid
    else:
        return None, None


def add_certain_column(
    df: pyspark.sql.dataframe, request_id: str
) -> pyspark.sql.dataframe:
    # 添加新列- 权重百分比
    df = (
        df.withColumn("WEIGHT_PERCENT", col("weight") * 100)
        .withColumn("GOOD_NUM", df["good_num"].cast(FloatType()))
        .withColumn("BAD_NUM", df["bad_num"].cast(FloatType()))
        .withColumn("REQUEST_ID", lit(request_id))
    )
    df = df.orderBy(col("WEIGHT").desc())
    df = df.withColumn("INDEX_NO", F.monotonically_increasing_id() + 1)
    info_list = ["PRODUCT_ID", "OPE_NO", "EQP_NAME", "PRODG1", "CHAMBER_NAME"]
    for column in info_list:
        if column not in df.columns:
            df = df.withColumn(column, lit(None).cast(StringType()))
    return df


class GetGroupInfo(object):
    # 获取wafer bood/bad 等多分组消息
    @staticmethod
    def commonality_analysis(
        df_run: pyspark.sql.dataframe, grpby_list: List[str]
    ) -> pyspark.sql.dataframe:
        """
        Perform commonality analysis on preprocessed data.
        :param df_run: Preprocessed data after data preprocessing.
        :param grpby_list: List of columns ['PRODG1', 'EQP_NAME', 'OPER_NO', 'PRODUCT_ID', 'TOOL_NAME'] for grouping.
                Example: grpby_list = ['PRODG1', 'TOOL_NAME', 'OPER_NO'], grpby_list = ['PRODUCT_ID', 'OPER_NO']
        :return: Results of commonality analysis, showing the top ten combinations with the highest number of bad wafers.
        """
        common_res = (
            df_run.groupBy(grpby_list)
            .agg(
                countDistinct("WAFER_ID", when(df_run["label"] == 0, 1)).alias(
                    "GOOD_NUM"
                ),
                countDistinct("WAFER_ID", when(df_run["label"] == 1, 1)).alias(
                    "BAD_NUM"
                ),
            )
            .na.fill(0)
        )
        return common_res

    @staticmethod
    def add_feature_stats_within_groups(
        df_integrate: pyspark.sql.DataFrame, grpby_list: List[str]
    ) -> pyspark.sql.DataFrame:
        # 将df_integrate中的PARAMETRIC_NAME按照PARAM#STEP#STATS提取出来
        split_params_for_df_integrate = (
            df_integrate
            # .withColumn(
            #     "PARAM", split(col("PARAMETRIC_NAME"), "#").getItem(0)
            # )
            # .withColumn("STEP", split(col("PARAMETRIC_NAME"), "#").getItem(1))
            .withColumn("STATS", split(col("PARAMETRIC_NAME"), "#").getItem(2))
        )

        # 按照grpby_list+PARAM+STEP统计每个分组中的G/B比例和STATS(用#连接)
        unique_params_within_groups = (
            split_params_for_df_integrate.groupBy(
                grpby_list + ["PARAMETRIC_NAME", "STATS"]
            )
            .agg(
                # collect_set("STATS").alias("STATS"),
                countDistinct(
                    "WAFER_ID",
                    when(
                        (
                            (split_params_for_df_integrate["label"] == 0)
                            & (split_params_for_df_integrate["RESULT"].isNotNull())
                        ),
                        1,
                    ),
                ).alias("GOOD_NUM"),
                countDistinct(
                    "WAFER_ID",
                    when(
                        (split_params_for_df_integrate["label"] == 1)
                        & (split_params_for_df_integrate["RESULT"].isNotNull()),
                        1,
                    ),
                ).alias("BAD_NUM"),
            )
            .na.fill(0)
        )

        return unique_params_within_groups


class ModelScorer:
    """对每个分组建立模型分析，获取得分"""

    @staticmethod
    def process_missing_values(
        df: pd.DataFrame, columns_to_process, missing_value_threshold
    ) -> pd.DataFrame:
        for column in columns_to_process:
            missing_percentage = df[column].isna().sum() / len(df)
            if missing_percentage > missing_value_threshold:
                df = df.drop(columns=[column])
            else:
                df[column] = df[column].fillna(df[column].mean())

        return df

    @staticmethod
    def get_pivot_table(df: pd.DataFrame, grpby_list: List[str]) -> pd.DataFrame:
        """
        Pivot the DataFrame based on specified grouping columns.

        Parameters:
        - df: Data for modeling.
        - grpby_list: List of grouping columns.

        Returns:
        - DataFrame: Result of pivoting the table.
        """
        index_cols = ["WAFER_ID", "label"]
        columns_cols = grpby_list + ["PARAMETRIC_NAME"]
        df_pivot = df.dropna(axis=0).pivot_table(
            index=index_cols, columns=columns_cols, values=["RESULT"]
        )
        df_pivot.columns = df_pivot.columns.map("#".join)
        df_pivot = df_pivot.reset_index(drop=False)
        # df_pivot = ModelScorer.process_missing_values(
        #     df_pivot, df_pivot.columns, missing_value_threshold=0.9
        # ).reset_index(drop=False)
        # df_pivot = df_pivot.fillna(df_pivot.mean()).reset_index(drop=False)
        # Remove completely identical columns
        # 删除同一值的列,会让parameter name 减少
        for column in df_pivot.columns.difference(index_cols):
            if df_pivot[column].nunique() == 1:
                df_pivot = df_pivot.drop(column, axis=1)

        # 确保label 分组，每一列特征都至少有一个值不为空，否则删除该列，如果成立，基于分组均值填充缺失值
        for column in df_pivot.columns.difference(["WAFER_ID", "label"]).tolist():
            if df_pivot[column].isna().sum() != 0:
                # 存在缺失值, 分label 填充
                fill_series = df_pivot.groupby(["label"])[column].transform(
                    lambda x: x.fillna(x.mean())
                )
                if fill_series.isna().sum() != 0:
                    # 说存在一个类别分组，该列全是None但是填充后仍然存在缺失值，则删除该列
                    df_pivot = df_pivot.drop(column, axis=1)
                else:
                    df_pivot[column] = fill_series
        return df_pivot

    @staticmethod
    def get_pipe_and_params(model):
        # from common_data_processing import get_pipe_params

        return get_pipe_params(model)

    @staticmethod
    def construct_features_when_not_satisfied(
        df_run, df_pivot, x_train, grpby_list, model_condition
    ):
        x_len = len(x_train.columns)

        if model_condition == "classification":

            small_importance_res = pd.DataFrame(
                {"features": x_train.columns, "importance": [0.0] * x_len}
            )

            sample_res_dict = {
                "bad_wafer": sum(df_pivot["label"]),
                "roc_auc_score": 0.0,
                "algorithm_satisfied": "FALSE",
                "x_train_shape": str(x_train.shape),
            }
            sample_res_dict.update(
                {col_: df_run[col_].values[0] for col_ in grpby_list}
            )
            # small_sample_res = pd.DataFrame(sample_res_dict, index=[0])
            for col_ in sample_res_dict.keys():
                small_importance_res[col_] = sample_res_dict[col_]
            return small_importance_res

        elif model_condition == "pca":

            res_top_select = pd.DataFrame(
                {
                    "features": x_train.columns,
                    "importance": [0.0] * x_len,
                    "bad_wafer": sum(df_pivot["label"]),
                    "algorithm_satisfied": ["FALSE"] * x_len,
                    "x_train_shape": [str(x_train.shape)] * x_len,
                }
            )
            for col_ in grpby_list:
                res_top_select[col_] = df_run[col_].values[0]
            return res_top_select

    @staticmethod
    def fit_classification_model(
        df: pyspark.sql.DataFrame, grpby_list: List[str], model
    ) -> pyspark.sql.DataFrame:
        struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
        struct_fields.extend(
            [
                StructField("bad_wafer", IntegerType(), True),
                StructField("roc_auc_score", FloatType(), True),
                StructField("features", StringType(), True),
                StructField("importance", FloatType(), True),
                StructField("algorithm_satisfied", StringType(), True),
                StructField("x_train_shape", StringType(), True),
            ]
        )
        schema_all = StructType(struct_fields)

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run: pd.DataFrame) -> pd.DataFrame:
            # Pivot the table
            df_pivot = ModelScorer.get_pivot_table(df=df_run, grpby_list=grpby_list)

            # Define independent and dependent variables
            x_train = df_pivot[
                df_pivot.columns.difference(["WAFER_ID", "label"]).tolist()
            ]
            y_train = df_pivot[["label"]]

            # 必须有两个特征，且类别数目大于1
            if x_train.shape[1] > 1 and y_train["label"].nunique() > 1:
                z_ratio = y_train.value_counts(normalize=True)
                good_ratio = z_ratio[0]
                bad_ratio = z_ratio[1]
                if abs(good_ratio - bad_ratio) > 0.7:
                    undersampler = ClusterCentroids(random_state=1024)
                    x_train, y_train = undersampler.fit_resample(x_train, y_train)

                # 加载模型
                pipe, param_grid = ModelScorer.get_pipe_and_params(model=model)

                try:
                    if param_grid is not None:
                        # 多变量分类模型
                        grid = GridSearchCV(
                            estimator=pipe,
                            scoring="roc_auc",
                            param_grid=param_grid,
                            cv=3,
                            n_jobs=-1,
                        )
                        grid.fit(x_train.values, y_train.values.ravel())
                    else:
                        # 单变量特征选择模型
                        grid = pipe.fit(x_train, y_train)

                except ValueError:
                    return ModelScorer.construct_features_when_not_satisfied(
                        df_run, df_pivot, x_train, grpby_list, "classification"
                    )

                if hasattr(grid, "best_estimator_"):
                    best_est = grid.best_estimator_.steps[-1][-1]

                    if hasattr(best_est, "feature_importances_"):
                        small_importance_res = pd.DataFrame(
                            {
                                "features": x_train.columns,
                                "importance": best_est.feature_importances_,
                            }
                        )
                    else:
                        small_importance_res = pd.DataFrame(
                            {
                                "features": x_train.columns,
                                "importance": abs(best_est.coef_.ravel()),
                            }
                        )
                else:
                    small_importance_res = pd.DataFrame(
                        {
                            "features": x_train.columns,
                            "importance": grid.steps[-1][-1].scores_
                            / (grid.steps[-1][-1].scores_).sum(),
                        }
                    )

                    grid.best_score_ = 1.0

                sample_res_dict = {
                    "bad_wafer": sum(df_pivot["label"]),
                    "roc_auc_score": (
                        0.0 if np.isnan(grid.best_score_) else grid.best_score_
                    ),
                    "algorithm_satisfied": "TRUE",
                    "x_train_shape": str(x_train.shape),
                }
                sample_res_dict.update(
                    {col_: df_run[col_].values[0] for col_ in grpby_list}
                )
                # rf，分组算法这种合并是不对的，容易造成空值出现
                # small_sample_res = pd.DataFrame(sample_res_dict, index=[0])
                # res_top_select = pd.concat([small_importance_res, small_sample_res])

                for k, v in sample_res_dict.items():
                    if k not in small_importance_res.columns:
                        small_importance_res[k] = v
                return small_importance_res
            else:
                res_top_select = ModelScorer.construct_features_when_not_satisfied(
                    df_run, df_pivot, x_train, grpby_list, "classification"
                )
                return res_top_select

        result = df.groupby(grpby_list).apply(get_model_result)
        # 在 groupby 和 apply 操作之后取消缓存
        df.unpersist()

        return result

    @staticmethod
    def construct_features_when_satisfy_pca(
        df_run, df_pivot, x_train, grpby_list
    ) -> pd.DataFrame:
        # 得到PCA算法结果res_top_select
        n_components = min(min(x_train.shape) - 2, 20)
        model = pca(n_components=n_components, verbose=None, random_state=2024)
        results = model.fit_transform(x_train)
        res_top = results["topfeat"]
        res_top_select = res_top[res_top["type"] == "best"][["feature", "loading"]]
        # 取绝对值, 归一化
        res_top_select["importance"] = abs(res_top_select["loading"]) / (
            np.abs(res_top_select["loading"]).sum()
        )
        res_top_select = (
            res_top_select.rename(columns={"feature": "features"})
            .drop("loading", axis=1)
            .drop_duplicates()
        )

        res_top_select["bad_wafer"] = sum(df_pivot["label"])
        for col_ in grpby_list:
            res_top_select[col_] = df_run[col_].values[0]
        res_top_select["x_train_shape"] = str(x_train.shape)
        res_top_select["algorithm_satisfied"] = "TRUE"
        return res_top_select

    @staticmethod
    def fit_pca_model(
        df: pyspark.sql.DataFrame, grpby_list: List[str]
    ) -> pyspark.sql.DataFrame:
        # Dynamically build schema according to the grpby_list
        struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
        struct_fields.extend(
            [
                StructField("features", StringType(), True),
                StructField("importance", FloatType(), True),
                StructField("bad_wafer", IntegerType(), True),
                StructField("algorithm_satisfied", StringType(), True),
                StructField("x_train_shape", StringType(), True),
            ]
        )
        schema_all = StructType(struct_fields)

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run: pd.DataFrame) -> pd.DataFrame:
            df_pivot = ModelScorer.get_pivot_table(df=df_run, grpby_list=grpby_list)
            df_pivot_copy = df_pivot.copy()
            df_pivot_all = pd.concat([df_pivot, df_pivot_copy], axis=0)

            x_train = df_pivot_all[
                df_pivot_all.columns.difference(["WAFER_ID", "label"]).tolist()
            ]

            if min(x_train.shape) > 2:
                res_top_select = ModelScorer.construct_features_when_satisfy_pca(
                    df_run, df_pivot, x_train, grpby_list
                )
                return res_top_select
            else:
                res_top_select = ModelScorer.construct_features_when_not_satisfied(
                    df_run, df_pivot, x_train, grpby_list, "pca"
                )
                return res_top_select

        result = df.groupby(grpby_list).apply(get_model_result)
        df.unpersist()
        return result


class GetScoreResultsByGroup(object):
    """获取每个分组的得分汇总结构"""

    @staticmethod
    def run(
        df_preprocessed: pyspark.sql.DataFrame,
        grpby_list: List[str],
        model_name: str,
        request_id: str,
    ):
        """
        df_preprocessed : 处理过后的长表结构， 带有STATS列
        grpby_list : List[str] : 分组列表
        """
        # if "STATS" not in df_preprocessed.columns:
        #     raise KeyError("df_preprocessed must have STATS column")
        # 获取分组共性分析结果
        common_res = GetGroupInfo.commonality_analysis(df_preprocessed, grpby_list)
        # 获取分组+ parameter + stats 的共性分析结果
        add_parametric_stats_df = GetGroupInfo.add_feature_stats_within_groups(
            df_preprocessed, grpby_list
        )
        # print("parameter stats count: {}".format(add_parametric_stats_df.count()))

        df_preprocessed = df_preprocessed.join(common_res, on=grpby_list, how="inner")

        if df_preprocessed.isEmpty():

            return None
            # 1. 分组只有一种类别
            # 2. 分组有多种类别(需要考虑退化的问题)
            # 2.1 每种类别都大于等于3， 可以进行三折交叉验证，调用大样本算法
            # 2.2 至少有一种类别小于3， 无法进行三折交叉验证 ，需要调用小样本算法

            # todo :good_num bad_num 放入pandas udf 计算
            # todo:核验输入参数和处理后的参数必须个数一致
        bad_wafer_num = (
            df_preprocessed.filter("label == 1").select("WAFER_ID").distinct().count()
        )

        # is_only_one_class_cond = common_res.filter("GOOD_NUM == 0 OR BAD_NUM == 0")
        #
        # if is_only_one_class_cond.count() > 0:
        #     # result_one_class
        #
        #     result_one_class = add_parametric_stats_df.filter(
        #         "GOOD_NUM == 0 OR BAD_NUM == 0"
        #     )
        #     result_one_class = result_one_class.withColumn(
        #         "WEIGHT",
        #         lit(0.0),
        #     )
        #
        #     print(
        #         "-----------------检查到存在只有一种类别的分组:result_one_class----------------------"
        #     )
        #     result_one_class = add_certain_column(
        #         df=result_one_class, request_id=request_id
        #     )
        #     # 存在只有一种类别的，无法进行算法分析，算法自己赋予权重0a
        #     # result_one_class.show()
        #
        # else:
        #
        #     result_one_class = None

        # print("result one class is ", result_one_class)

        is_big_algo_cond = common_res.filter("GOOD_NUM >= 3 AND BAD_NUM >= 3")

        # add_parametric_stats_df = add_parametric_stats_df.filter(
        #     "GOOD_NUM > 0 AND BAD_NUM > 0"
        # )
        # print("len(bo) ", add_parametric_stats_df.count())

        if is_big_algo_cond.count() > 0:

            # 使用三折交叉验证算法，调用大样本算法

            big_train_data = df_preprocessed.filter("GOOD_NUM >= 3 AND BAD_NUM >= 3")
            print(
                f"--------------------调用 大样本算法 train_data :{big_train_data.count()}条数------------------"
            )
            # is_big_algo_cond.show()
            result_big = ModelScorer.fit_classification_model(
                df=big_train_data.drop("GOOD_NUM", "BAD_NUM"),
                grpby_list=grpby_list,
                model=model_name,
            )

        else:
            result_big = None

        is_small_algo_cond = common_res.filter("GOOD_NUM> 0 and BAD_NUM > 0").filter(
            "(GOOD_NUM < 3) OR (BAD_NUM< 3) "
        )
        if is_small_algo_cond.count() > 0:
            # 使用小样本算法，(三种及三种以上类别)考虑退化问题，一个类别为0，

            # small_train_data = train_data.join(
            #     is_small_algo_cond, on=grpby_list, how="inner"
            # )
            small_train_data = df_preprocessed.filter(
                "GOOD_NUM> 0 and BAD_NUM > 0"
            ).filter("!((GOOD_NUM >= 3) AND (BAD_NUM>= 3)) ")

            # print(
            #     f"--------------------调用小样本算法 train_data :{small_train_data.count()}条数------------------"
            # )
            # is_small_algo_cond.show()
            # result_small = FitModelForUvaData.fit_pca_model(
            #     df=small_train_data, grpby_list=grpby_list
            # )
            if model_name in ["chi2", "mutual_info", "anova"]:
                result_small = ModelScorer.fit_classification_model(
                    df=small_train_data.drop("GOOD_NUM", "BAD_NUM"),
                    grpby_list=grpby_list,
                    model=model_name,
                )
            else:
                # result_small = ModelScorer.fit_pca_model(
                #     df=small_train_data, grpby_list=grpby_list
                # )
                result_small = ModelScorer.fit_classification_model(
                    df=small_train_data.drop("GOOD_NUM", "BAD_NUM"),
                    grpby_list=grpby_list,
                    model="anova",
                )
            # 复用大样本算法的处理办法
            if "roc_auc_score" not in result_small.columns:
                result_small = result_small.withColumn("roc_auc_score", lit(1.0))

            print(
                f"--------------------调用小样本算法 train_data :{small_train_data.count()}条数------------------"
            )

        else:
            result_small = None

        # print(result_big, result_small)
        # 合并 大样本和小样本的结果
        result_big_small = [i for i in [result_big, result_small] if i is not None]

        if len(result_big_small) > 1:
            result_big_small = reduce(lambda x, y: x.unionByName(y), result_big_small)
        elif len(result_big_small) == 1:
            result_big_small = result_big_small[0]
        else:
            result_big_small = None

        if result_big_small is not None:
            n_feats = len(grpby_list)
            # 第一个是RESULT 中间是groupy_list
            for i in range(n_feats):
                result_big_small = result_big_small.withColumn(
                    grpby_list[i], split(col("features"), "#").getItem(i + 1)
                )

            result_big_small = (
                result_big_small.withColumn(
                    "PARAMETRIC_NAME", split(col("features"), "#").getItem(n_feats + 1)
                )
                .withColumn("STEP", split(col("features"), "#").getItem(n_feats + 2))
                .withColumn("STATS", split(col("features"), "#").getItem(n_feats + 3))
                .drop("features")
            )
            # parameter 拼接sensor_name 和step
            result_big_small = result_big_small.withColumn(
                "PARAMETRIC_NAME",
                concat_ws(
                    "#",
                    col("PARAMETRIC_NAME"),
                    col("STEP"),
                    col("STATS"),
                ),
            )
            # result_big_small.show()

            # 计算每个分组roc_auc 的得分和 ，pca(默认为1， 单变量默认也为1)
            roc_auc_score_all = result_big_small.groupBy(grpby_list).agg(
                mean("roc_auc_score").alias("roc_auc_score")
            )  # 提取每个分组模型的auc
            roc_auc_score_all = (
                roc_auc_score_all.select("roc_auc_score")
                .toPandas()["roc_auc_score"]
                .sum()
            )  # 计算所有模型的roc_auc 求和得分

            result_big_small = result_big_small.withColumn(
                "WEIGHT",
                col("bad_wafer")
                / lit(bad_wafer_num)  # bad wafer ratio
                * (col("roc_auc_score") / roc_auc_score_all)  # roc auc ratio
                * col("importance"),  # score
            )
            weight_sum = result_big_small.select("WEIGHT").toPandas()["WEIGHT"].sum()
            result_big_small = (
                result_big_small.withColumn("WEIGHT", col("WEIGHT") / weight_sum)
                .withColumn("WEIGHT_PERCENT", col("WEIGHT") * lit(100))
                .orderBy(col("WEIGHT_PERCENT").desc(), col("bad_wafer").desc())
            ).drop(
                # 删除多余的列
                "bad_wafer",
                "roc_auc_score",
                "importance",
                "x_train_shape",
                "algorithm_satisfied",
                "STEP",
            )

            # result_big_small.show()
            # print("before join ")
            # print(result_big_small.count())
            # print("before join , result show ")
            # result_big_small.show()
            #  算法分析的时候，删除了唯一值个数为1的变量名,或者删除缺失值比率过高的参数
            res_all = result_big_small.join(
                add_parametric_stats_df,
                on=grpby_list + ["PARAMETRIC_NAME", "STATS"],
                how="right",
            ).fillna(0.0, subset=["WEIGHT"])
            # result_big_small = result_big_small.join(
            #     add_parametric_stats_df,
            #     on=grpby_list + ["PARAMETRIC_NAME", "STATS"],
            #     how="left",
            # )

            # # 算法分析的时候，删除了唯一值个数为1的变量名
            # missing_rows = add_parametric_stats_df.join(
            #     result_big_small,
            #     on=grpby_list + ["PARAMETRIC_NAME", "STATS"],
            #     how="left_anti",
            # )
            # print("missing_rows ", missing_rows.count())
            #
            # missing_rows = missing_rows.withColumn("WEIGHT", lit(0))
            # res_all = result_big_small.unionByName(
            #     missing_rows, allowMissingColumns=True
            # ).fillna(0.0, subset=["WEIGHT"])

            # print("after join res_all")
            # print(res_all.count())

            # result_big_small.show()

            result_big_small = add_certain_column(df=res_all, request_id=request_id)

        # 汇总分析结果
        # result = [i for i in [result_big_small, result_one_class] if i is not None]
        result = [result_big_small] if result_big_small is not None else []
        # if len(result) > 1:
        #     result = reduce(lambda x, y: x.unionByName(y), result)
        if len(result) == 1:
            result = result[0]
        else:
            return None

        # 拼接sensor_name 和step 为新的parametric_name
        result = result.withColumn(
            "PARAMETRIC_NAME",
            concat_ws(
                "#",
                split(col("PARAMETRIC_NAME"), "#").getItem(0),
                split(col("PARAMETRIC_NAME"), "#").getItem(1),
            ),
        )

        # left-anti 只存在左侧，不存在右侧

        return result


"""
各个数据源分组算法分析流程:
1. 合并功能。
2. 数据预处理
3. 分析流程：基于类别个数调整，调用分组分析计算得分 模块，
4. 得分汇总分析
5. 后处理输出结果排名，没有计算的得分为0  

"""

if __name__ == "__main__":
    get_pipe_params("rf")
