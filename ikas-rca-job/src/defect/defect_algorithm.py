import pyspark
import numpy as np
import pandas as pd
from pca import pca
from typing import List, Dict
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import ClusterCentroids
from src.exceptions.rca_base_exception import RCABaseException
from sklearn.linear_model import SGDClassifier, LogisticRegression
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql.functions import pandas_udf, PandasUDFType, lit, col, countDistinct, when, monotonically_increasing_id, split, explode, sum as spark_sum


class DataPreprocessorForDefect:
    def __init__(self,
                 df: pyspark.sql.dataframe,
                 grpby_list: list[str],
                 features_columns: list[str],
                 columns_list: list[str],
                 merge_layerid_list: List[Dict[str, List[str]]],
                 merge_prodg1_list: List[Dict[str, List[str]]],
                 merge_product_list: List[Dict[str, List[str]]]
                 ):
        self.df = df
        self.grpby_list = grpby_list
        self.features_columns = features_columns
        self.columns_list = columns_list
        self.merge_layerid_list = merge_layerid_list
        self.merge_prodg1_list = merge_prodg1_list
        self.merge_product_list = merge_product_list

    @staticmethod
    def select_columns(df: pyspark.sql.dataframe, columns_list: list[str]) -> pyspark.sql.dataframe:
        return df.select(columns_list)

    @staticmethod
    def integrate_columns(df: pyspark.sql.dataframe,
                          merge_layerid_list: List[Dict[str, List[str]]],
                          merge_prodg1_list: List[Dict[str, List[str]]],
                          merge_product_list: List[Dict[str, List[str]]]) -> pyspark.sql.dataframe:
        df_merged = DataPreprocessorForDefect.integrate_single_column(df, merge_layerid_list, 'LAYER_ID')
        df_merged = DataPreprocessorForDefect.integrate_single_column(df_merged, merge_prodg1_list, 'PRODG1')
        df_merged = DataPreprocessorForDefect.integrate_single_column(df_merged, merge_product_list, 'PRODUCT_ID')
        return df_merged

    @staticmethod
    def integrate_single_column(df: pyspark.sql.dataframe,
                                merge_list: List[Dict[str, List[str]]],
                                column_name: str) -> pyspark.sql.dataframe:
        splitter_comma = ","
        if merge_list is not None and len(merge_list) > 0:
            values_to_replace = [list(rule.values())[0] for rule in merge_list]
            merged_values = [splitter_comma.join(list(rule.values())[0]) for rule in merge_list]

            for values, replacement_value in zip(values_to_replace, merged_values):
                df = df.withColumn(column_name,
                                   when(col(column_name).isin(values), replacement_value).otherwise(col(column_name)))
        return df

    @staticmethod
    def pre_process(df: pyspark.sql.dataframe, features_columns: list[str]) -> pyspark.sql.dataframe:
        df_run = df.dropDuplicates()
        df_run = df_run.dropna(subset=features_columns, how='any')
        return df_run

    @staticmethod
    def add_feature_stats_within_groups(df_integrate: pyspark.sql.dataframe,
                                        grpby_list: list[str], ) -> pyspark.sql.dataframe:
        df = (df_integrate.groupBy(grpby_list)
              .agg(
            countDistinct('WAFER_ID', when(df_integrate['label'] == 0, 1)).alias('GOOD_NUM'),
            countDistinct('WAFER_ID', when(df_integrate['label'] == 1, 1)).alias('BAD_NUM'))
              .na.fill(0))
        df_with_list = df.withColumn("PARAMETRIC_NAME",
                                     lit("RANDOM_DEFECTS,DEFECTS,ADDER_DEFECTS,CLUSTERS,ADDER_RANDOM_DEFECTS,ADDER_CLUSTERS"))
        df_with_array = df_with_list.withColumn("PARAMETRIC_NAME", split("PARAMETRIC_NAME", ","))
        add_parametric_stats_df = df_with_array.select(*df.columns, explode("PARAMETRIC_NAME").alias("PARAMETRIC_NAME"))
        return add_parametric_stats_df

    def run(self) -> pyspark.sql.dataframe:
        df_select = self.select_columns(df=self.df, columns_list=self.columns_list)
        df_integrate = self.integrate_columns(df=df_select,
                                              merge_layerid_list=self.merge_layerid_list,
                                              merge_prodg1_list=self.merge_prodg1_list,
                                              merge_product_list=self.merge_product_list)
        add_parametric_stats_df = self.add_feature_stats_within_groups(df_integrate=df_integrate,
                                                                       grpby_list=self.grpby_list)
        df_preprocess = self.pre_process(df=df_integrate, features_columns=self.features_columns)
        return add_parametric_stats_df, df_preprocess


class GetTrainDataForDefect:
    def __init__(self, df: pyspark.sql.dataframe, grpby_list: list[str]):
        self.df_run = df
        self.grpby_list = grpby_list

    @staticmethod
    def commonality_analysis(df_run: pyspark.sql.dataframe, grpby_list: List[str]) -> pyspark.sql.dataframe:
        common_res = (df_run.groupBy(grpby_list)
                      .agg(countDistinct('WAFER_ID', when(df_run['label'] == 0, 1)).alias('GOOD_NUM'),
                           countDistinct('WAFER_ID', when(df_run['label'] == 1, 1)).alias('BAD_NUM'))
                      .na.fill(0))
        return common_res

    @staticmethod
    def get_data_list(common_res: pyspark.sql.dataframe, grpby_list: list[str]) -> List[Dict[str, str]]:
        data_list = common_res.select(grpby_list).collect()
        data_dict_list = [row.asDict() for row in data_list]
        return data_dict_list

    @staticmethod
    def get_train_data(df_run: pyspark.sql.dataframe, data_dict_list: List[Dict[str, str]]) -> pyspark.sql.dataframe:
        first_data_dict = data_dict_list[0]
        conditions = " AND ".join(["{} == '{}'".format(col_, first_data_dict[col_]) for col_ in first_data_dict])
        df_s = df_run.filter(conditions)
        for i in range(1, len(data_dict_list)):
            data_dict = data_dict_list[i]
            conditions = " AND ".join(["{} == '{}'".format(col_, data_dict[col_]) for col_ in data_dict])
            df_m = df_run.filter(conditions)
            df_s = df_s.union(df_m)
        return df_s

    def run(self) -> pyspark.sql.dataframe:
        common_res = self.commonality_analysis(df_run=self.df_run, grpby_list=self.grpby_list)
        common_res = common_res.withColumn("conditions_satisfied",
                                           when((col('GOOD_NUM') >= 1) & (col('BAD_NUM') >= 1), True).otherwise(False))
        grps_large = common_res.filter("GOOD_NUM > 3 AND BAD_NUM > 3")

        if grps_large.isEmpty():
            grps_less = common_res.filter("GOOD_NUM >= 1 AND BAD_NUM >= 1")
            if grps_less.isEmpty():
                print(
                    f"按照{'+'.join(self.grpby_list)}分组后的数据, 没有组合满足条件good >= 1且bad >= 1, 无法给出算法权重.")
                train_data = common_res
                big_or_small = 'no'
                return common_res, train_data, big_or_small
            else:
                print(f"按照{'+'.join(self.grpby_list)}分组后的数据, 一共有{grps_less.count()}种不同的分组.")
                data_dict_list = self.get_data_list(common_res=grps_less, grpby_list=self.grpby_list)
                train_data = self.get_train_data(df_run=self.df_run, data_dict_list=data_dict_list)
                big_or_small = 'small'
                return common_res, train_data, big_or_small
        else:
            print(f"按照{'+'.join(self.grpby_list)}分组后的数据, 一共有{grps_large.count()}种不同的分组.")
            data_dict_list = self.get_data_list(common_res=grps_large, grpby_list=self.grpby_list)
            train_data = self.get_train_data(df_run=self.df_run, data_dict_list=data_dict_list)
            big_or_small = 'big'
            return common_res, train_data, big_or_small


class FitModelForDefect:

    @staticmethod
    def select_max_inspection_time_records(df: pd.DataFrame, grpby_list: list[str]) -> pd.DataFrame:
        df['INSPECTION_TIME'] = pd.to_datetime(df['INSPECTION_TIME'])
        idx = df.groupby(['WAFER_ID', 'label'] + grpby_list)['INSPECTION_TIME'].idxmax()
        df_sort = df.loc[idx, :]
        return df_sort

    @staticmethod
    def get_pipe_params(model):
        common_steps = [
            ('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
            ('scaler', StandardScaler())
        ]
        models = {
            'rf': (RandomForestClassifier(random_state=2024), {
                'model__n_estimators': [*range(10, 60, 10)],
                'model__max_depth': [*range(5, 50, 10)],
                'model__min_samples_split': [2, 5],
                'model__min_samples_leaf': [1, 3]
            }),

            'decisionTree': (DecisionTreeClassifier(random_state=2024), {
                'model__max_depth': [None, 5, 10, 15],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            }),

            'svc': (LinearSVC(random_state=2024, fit_intercept=False), {
                'model__loss': ['hinge', 'squared_hinge'],
                'model__C': [0.1, 0.5, 1, 10, 50]
            }),

            'logistic': (LogisticRegression(random_state=2024, fit_intercept=False, solver='liblinear'), {
                'model__penalty': ['l1', 'l2'],
                'model__C': [0.1, 0.5, 1, 10, 50]
            }),

            'sgd': (SGDClassifier(random_state=2024, fit_intercept=False), {
                'model__loss': ['hinge', 'log_loss', 'perceptron', 'huber'],
                'model__penalty': ['l1', 'l2', 'elasticnet', None],
                'model__alpha': [0.0001, 0.001, 0.01, 0.1],
                'model__max_iter': [100, 500, 1000]
            })
        }

        if model in models:
            model_class, param_grid = models[model]
            steps = common_steps + [('model', model_class)]
            pipe = Pipeline(steps)
        else:
            raise Exception('Wrong Model Selection. Supported models are: pca, rf, decisionTree, svc, logistic, sgd.')
        return pipe, param_grid

    @staticmethod
    def fit_classification_model(df, grpby_list, features_columns, model):
        struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
        struct_fields.extend([StructField("PARAMETRIC_NAME", StringType(), True),
                              StructField("importance", FloatType(), True),
                              StructField("roc_auc_score", FloatType(), True),
                              StructField("algorithm_satisfied", StringType(), True),
                              StructField("x_train_shape", StringType(), True)])
        schema_all = StructType(struct_fields)

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run):
            df_run = FitModelForDefect.select_max_inspection_time_records(df=df_run, grpby_list=grpby_list)
            x_train = df_run[features_columns]
            y_train = df_run[['label']]

            # defect数据每个组合下只会有6个参数, 所以x_train.shape[1]一定是6,
            # 这里用x_train.shape[0] > 6卡控每个组合下必须有6片以上的wafer
            if x_train.shape[0] > 6 and y_train['label'].nunique() > 1:
                z_ratio = y_train.value_counts(normalize=True)
                good_ratio = z_ratio[0]
                bad_ratio = z_ratio[1]
                if abs(good_ratio - bad_ratio) > 0.7:
                    undersampler = ClusterCentroids(random_state=1024)
                    x_train, y_train = undersampler.fit_resample(x_train, y_train)

                pipe, param_grid = FitModelForDefect.get_pipe_params(model=model)
                try:
                    grid = GridSearchCV(estimator=pipe, scoring='roc_auc', param_grid=param_grid, cv=3, n_jobs=-1)
                    grid.fit(x_train.values, y_train.values.ravel())
                except ValueError:
                    return pd.DataFrame()

                best_est = grid.best_estimator_.steps[-1][-1]
                feature_importances = best_est.feature_importances_ if hasattr(best_est,
                                                                               'feature_importances_') else abs(
                    best_est.coef_.ravel())
                small_importance_res = pd.DataFrame(
                    {'PARAMETRIC_NAME': x_train.columns, 'importance': feature_importances})

                info_cols_list = {col_: df_run[col_].values[0] for col_ in grpby_list}
                res_top_select = small_importance_res.assign(**info_cols_list,
                                                             roc_auc_score=0.0 if np.isnan(
                                                                 grid.best_score_) else grid.best_score_,
                                                             algorithm_satisfied='TRUE',
                                                             x_train_shape=str(x_train.shape))
                return res_top_select

            else:
                res_top_select = FitModelForDefect.construct_features_when_not_satisfied(df_run=df_run,
                                                                                         x_train=x_train,
                                                                                         grpby_list=grpby_list,
                                                                                         model_condition='classification')
                return res_top_select

        return df.groupby(grpby_list).apply(get_model_result)

    @staticmethod
    def fit_pca_model(df: pyspark.sql.dataframe, grpby_list, features_columns) -> pyspark.sql.dataframe:
        struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
        struct_fields.extend([StructField("PARAMETRIC_NAME", StringType(), True),
                              StructField("importance", FloatType(), True),
                              StructField("algorithm_satisfied", StringType(), True),
                              StructField("x_train_shape", StringType(), True)])
        schema_all = StructType(struct_fields)

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run):
            df_run = FitModelForDefect.select_max_inspection_time_records(df=df_run, grpby_list=grpby_list)
            x_train = df_run[features_columns]

            if min(x_train.shape) > 2:
                res_top_select = FitModelForDefect.construct_features_when_satisfy_pca(df_run=df_run,
                                                                                       x_train=x_train,
                                                                                       grpby_list=grpby_list)
                return res_top_select
            else:
                res_top_select = FitModelForDefect.construct_features_when_not_satisfied(df_run=df_run,
                                                                                         x_train=x_train,
                                                                                         grpby_list=grpby_list,
                                                                                         model_condition='pca')
                return res_top_select

        return df.groupby(grpby_list).apply(get_model_result)

    @staticmethod
    def construct_features_when_satisfy_pca(df_run, x_train, grpby_list) -> pd.DataFrame:
        # 得到PCA算法结果res_top_select
        n_components = min(min(x_train.shape) - 2, 20)
        model = pca(n_components=n_components, verbose=None, random_state=2024)
        results = model.fit_transform(x_train)
        res_top = results['topfeat']
        res_top_select = res_top[res_top['type'] == 'best'][['feature', 'loading']]
        res_top_select['importance'] = abs(res_top_select['loading'])
        res_top_select = res_top_select.rename(columns={'feature': 'PARAMETRIC_NAME'}).drop("loading",
                                                                                            axis=1).drop_duplicates()
        res_top_select['x_train_shape'] = str(x_train.shape)
        res_top_select['algorithm_satisfied'] = 'TRUE'
        for col_ in grpby_list:
            res_top_select[col_] = df_run[col_].values[0]
        return res_top_select

    @staticmethod
    def construct_features_when_not_satisfied(df_run, x_train, grpby_list, model_condition) -> pd.DataFrame:
        x_len = len(x_train.columns)
        info_cols_list = {col_: df_run[col_].values[0] for col_ in grpby_list}
        if model_condition == 'classification':
            small_importance_res = pd.DataFrame({"PARAMETRIC_NAME": x_train.columns,
                                                 "importance": [0.0] * x_len,
                                                 "roc_auc_score": [0.0] * x_len})
        else:  # model_condition == 'pca':
            small_importance_res = pd.DataFrame({"PARAMETRIC_NAME": x_train.columns,
                                                 "importance": [0.0] * x_len})

        res_top_select = small_importance_res.assign(**info_cols_list,
                                                     algorithm_satisfied='FALSE',
                                                     x_train_shape=str(x_train.shape))
        return res_top_select


class SplitDefectModelResults:
    def __init__(self, df: pyspark.sql.dataframe, grpby_list: List[str], request_id: str,
                 grps_all: pyspark.sql.dataframe, big_or_small: str,
                 add_parametric_stats_df: pyspark.sql.dataframe) -> pyspark.sql.dataframe:
        self.df = df
        self.grpby_list = grpby_list
        self.request_id = request_id
        self.grps_all = grps_all
        self.big_or_small = big_or_small
        self.add_parametric_stats_df = add_parametric_stats_df

    @staticmethod
    def get_final_results_big_sample(df: pyspark.sql.dataframe,
                                     grpby_list: List[str]) -> pyspark.sql.dataframe:

        df_grpby = df.groupBy(grpby_list + ['PARAMETRIC_NAME', 'algorithm_satisfied',
                                            'roc_auc_score', 'x_train_shape']).agg(
            spark_sum('importance').alias('importance'))
        roc_auc_score_all = df_grpby.agg({"roc_auc_score": "sum"}).collect()[0][0]

        if roc_auc_score_all is not None and roc_auc_score_all > 0:
            df_merge = (df_grpby.withColumn("roc_auc_score_ratio", col("roc_auc_score") / roc_auc_score_all)
                        .withColumn('weight_original', col('roc_auc_score_ratio') * col('importance')))
            # Normalize again
            weight_all = df_merge.agg({"weight_original": "sum"}).collect()[0][0]
            df_merge = df_merge.withColumn("WEIGHT", col("weight_original") / weight_all)
        else:
            weight_all = df_grpby.agg({"importance": "sum"}).collect()[0][0]
            df_merge = df.withColumn('WEIGHT', col('importance') / weight_all)

        df_merge = df_merge.select(grpby_list + ['PARAMETRIC_NAME', 'WEIGHT'])
        return df_merge

    @staticmethod
    def get_final_results_small_sample(df: pyspark.sql.dataframe,
                                       grpby_list: List[str]) -> pyspark.sql.dataframe:
        df_grpby = df.groupBy(grpby_list + ['PARAMETRIC_NAME',
                                            'algorithm_satisfied', 'x_train_shape']).agg(
            spark_sum('importance').alias('importance'))
        # Normalize weights again
        weight_all = df_grpby.agg({"importance": "sum"}).collect()[0][0]
        df_merge = df_grpby.withColumn("WEIGHT", col("importance") / weight_all)

        df_merge = df_merge.select(grpby_list + ['PARAMETRIC_NAME', 'WEIGHT'])
        return df_merge

    @staticmethod
    def add_certain_column(df: pyspark.sql.dataframe, request_id: str) -> pyspark.sql.dataframe:
        df = (df.withColumn("WEIGHT_PERCENT", col("WEIGHT") * 100)
              .withColumn("GOOD_NUM", df["GOOD_NUM"].cast(FloatType()))
              .withColumn("BAD_NUM", df["BAD_NUM"].cast(FloatType()))
              .withColumn("REQUEST_ID", lit(request_id)))
        df = df.orderBy(col("WEIGHT").desc())
        df = df.withColumn('INDEX_NO', monotonically_increasing_id() + 1)

        info_list = ['PRODUCT_ID', 'LAYER_ID', 'PRODG1']
        for column in info_list:
            if column not in df.columns:
                df = df.withColumn(column, lit(None).cast(StringType()))
        return df

    def run(self) -> pyspark.sql.dataframe:
        m = self.df.filter("algorithm_satisfied==TRUE").count()
        if m == 0:
            final_res = self.add_parametric_stats_df.withColumn('WEIGHT', lit(0))
            final_res = self.add_certain_column(df=final_res, request_id=self.request_id)
            return final_res

        if self.big_or_small == 'big':
            res_all = self.get_final_results_big_sample(df=self.df, grpby_list=self.grpby_list)
        else:  # self.big_or_small == 'small':
            res_all = self.get_final_results_small_sample(df=self.df, grpby_list=self.grpby_list)

        res_all = res_all.join(self.add_parametric_stats_df, on=self.grpby_list + ['PARAMETRIC_NAME'], how='left')
        missing_rows = self.add_parametric_stats_df.join(res_all, on=self.grpby_list + ['PARAMETRIC_NAME'],
                                                         how='left_anti')
        missing_rows = missing_rows.withColumn('WEIGHT', lit(0))
        res_all_update_missing_features = res_all.unionByName(missing_rows, allowMissingColumns=True)
        final_res = self.add_certain_column(df=res_all_update_missing_features, request_id=self.request_id)
        return final_res


class ExertDefectAlgorithm:
    @staticmethod
    def fit_defect_model(df: pyspark.sql.dataframe,
                         request_id: str,
                         merge_layerid_list: List[Dict[str, List[str]]],
                         merge_prodg1_list: List[Dict[str, List[str]]],
                         merge_product_list: List[Dict[str, List[str]]],
                         features_columns=None,
                         columns_list=None,
                         grpby_list=None):

        if grpby_list is None or len(grpby_list) == 0:
            grpby_list = ['LAYER_ID']

        if features_columns is None:
            features_columns = ["RANDOM_DEFECTS", "DEFECTS", "ADDER_DEFECTS", "CLUSTERS", "ADDER_RANDOM_DEFECTS",
                                "ADDER_CLUSTERS"]

        if columns_list is None:
            columns_list = list(set(grpby_list + features_columns + ['WAFER_ID', 'INSPECTION_TIME', 'label',
                                                                     'PRODUCT_ID', 'PRODG1', 'LAYER_ID']))

        add_parametric_stats_df, df_preprocess = DataPreprocessorForDefect(df=df,
                                                                           grpby_list=grpby_list,
                                                                           features_columns=features_columns,
                                                                           columns_list=columns_list,
                                                                           merge_layerid_list=merge_layerid_list,
                                                                           merge_prodg1_list=merge_prodg1_list,
                                                                           merge_product_list=merge_product_list).run()
        if df_preprocess.isEmpty():
            msg = '数据库中暂无数据.'
            raise RCABaseException(msg)

        common_res, train_data, big_or_small = GetTrainDataForDefect(df=df_preprocess, grpby_list=grpby_list).run()

        if train_data.isEmpty():
            msg = f"按照{'+'.join(grpby_list)}分组后的训练数据暂时为空."
            raise RCABaseException(msg)

        if big_or_small == 'big':
            print("****************Call Big Sample Algorithm****************")
            res = FitModelForDefect.fit_classification_model(df=train_data, grpby_list=grpby_list,
                                                             features_columns=features_columns, model='rf')
        elif big_or_small == 'small':
            print("****************Call Small Sample Algorithm****************")
            res = FitModelForDefect.fit_pca_model(df=train_data, grpby_list=grpby_list,
                                                  features_columns=features_columns)
        else:  # big_or_small == 'no'
            final_res = add_parametric_stats_df.withColumn('WEIGHT', lit(0))
            final_res = SplitDefectModelResults.add_certain_column(df=final_res, request_id=request_id)
            return final_res

        final_res = SplitDefectModelResults(df=res, grpby_list=grpby_list, request_id=request_id,
                                            grps_all=common_res, big_or_small=big_or_small,
                                            add_parametric_stats_df=add_parametric_stats_df).run()
        return final_res


if __name__ == "__main__":
    import os
    import json
    from pyspark.sql import SparkSession
    import pyspark.pandas as ps

    os.environ['PYSPARK_PYTHON'] = '/usr/local/python-3.9.13/bin/python3'
    spark = SparkSession.builder \
        .appName("pandas_udf") \
        .config('spark.sql.session.timeZone', 'Asia/Shanghai') \
        .config("spark.scheduler.mode", "FAIR") \
        .config('spark.driver.memory', '8g') \
        .config('spark.driver.cores', '12') \
        .config('spark.executor.memory', '8g') \
        .config('spark.executor.cores', '12') \
        .config('spark.cores.max', '12') \
        .config('spark.driver.host', '192.168.22.28') \
        .master("spark://192.168.12.47:7077,192.168.12.48:7077") \
        .getOrCreate()

    df_run_pandas = pd.read_csv(
        'D:/Jupyterfiles/晶合MVAFDC_general开发/MVAanlysisDevelop/defect_algorithm/defect_by_wafer_labeled_6.csv')

    # print(df_run_pandas.dtypes)
    # df_run_pandas['INSPECTION_TIME'] = pd.to_datetime(df_run_pandas['INSPECTION_TIME'])
    # prodg = 'KLKL'
    # product_id = 'Ab.KMM01'
    # df_run_select1 = df_run_pandas.query(f"PRODG1 == '{prodg}' & PRODUCT_ID == '{product_id}'")
    # print(df_run_select1.shape)
    # print(df_run_select1['label'].value_counts())
    # print(df_run_select1['WAFER_ID'].nunique())

    df_spark = ps.from_pandas(df_run_pandas).to_spark()
    print(f"df_spark shape: ({df_spark.count()}, {len(df_spark.columns)})")
    df_spark.show()

    json_loads_dict = {"requestId": "269",
                       "algorithm": "defect",
                       "requestParam": {"dateRange": {"start": "2021-12-06 19:50:49", "end": "2024-03-06 19:50:49"},
                                        "operNo": [],
                                        "uploadId": "84f6a2b46a5443ec9797347424402058",
                                        "flagMergeAllProdg1": "0",
                                        "flagMergeAllProductId": "0",
                                        "flagMergeAllChamber": "0",
                                        "mergeProdg1": [],
                                        "mergeProductId": [],
                                        "mergeEqp": [],
                                        "mergeChamber": [],
                                        "mergeLayerId": [{"xx1": ["EOL", "5FP10", "HEFEI"]},
                                                         {"xx2": ["BH90WA12S1", "1CG10", "1FG10"]}],
                                        # "mergeOperno": None
                                        }
                       }

    df_ = pd.DataFrame({"requestId": [json_loads_dict["requestId"]],
                        "requestParam": [json.dumps(json_loads_dict["requestParam"])]})

    request_id_ = df_["requestId"].values[0]
    request_params = df_["requestParam"].values[0]
    parse_dict = json.loads(request_params)

    merge_layer = list(parse_dict.get('mergeLayerId')) if parse_dict.get('mergeLayerId') else None
    merge_prodg1 = list(parse_dict.get('mergeProdg1')) if parse_dict.get('mergeProdg1') else None
    merge_product = list(parse_dict.get('mergeProductId')) if parse_dict.get('mergeProductId') else None
    # grpby_list_ = ['LAYER_ID', 'PRODUCT_ID', 'PRODG1']
    grpby_list_ = ['LAYER_ID']

    from datetime import datetime

    time1 = datetime.now()
    print(time1)
    final_res_ = ExertDefectAlgorithm.fit_defect_model(df=df_spark,
                                                       request_id=request_id_,
                                                       grpby_list=grpby_list_,
                                                       merge_layerid_list=merge_layer,
                                                       merge_prodg1_list=merge_prodg1,
                                                       merge_product_list=merge_product)
    print(f"final_res_: ({final_res_.count()}, {len(final_res_.columns)})")
    final_res_.show(30)
    print(final_res_.dtypes)
    time2 = datetime.now()
    print(time2, time2 - time1)

    final_res_pandas = final_res_.toPandas()
    final_res_pandas.to_csv("final_res_pandas.csv")
