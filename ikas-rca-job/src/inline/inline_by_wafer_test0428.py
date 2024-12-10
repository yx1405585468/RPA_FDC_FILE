import pyspark
import pandas as pd
from pca import pca
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql.functions import pandas_udf, PandasUDFType, lit, col, collect_set, explode, countDistinct, when, \
    monotonically_increasing_id, sum as spark_sum
from typing import List, Dict, Union
from src.exceptions.rca_base_exception import RCABaseException


class DataPreprocessorForInline:
    def __init__(self,
                 df: pyspark.sql.dataframe,
                 grpby_list: list[str],
                 columns_list: list[str],
                 certain_column: str,
                 key_words: list[str],
                 convert_to_numeric_list: list[str],
                 merge_operno_list: List[Dict[str, List[str]]],
                 merge_prodg1_list: List[Dict[str, List[str]]],
                 merge_product_list: List[Dict[str, List[str]]]
                 ):
        self.df = df
        self.grpby_list = grpby_list
        self.columns_list = columns_list
        self.certain_column = certain_column
        self.key_words = key_words
        self.convert_to_numeric_list = convert_to_numeric_list
        self.merge_operno_list = merge_operno_list
        self.merge_prodg1_list = merge_prodg1_list
        self.merge_product_list = merge_product_list

    @staticmethod
    def select_columns(df: pyspark.sql.DataFrame, columns_list: list[str]) -> pyspark.sql.DataFrame:
        return df.select(columns_list)

    @staticmethod
    def exclude_some_data(df: pyspark.sql.DataFrame, key_words: list[str],
                          certain_column: str) -> pyspark.sql.DataFrame:
        key_words_str = '|'.join(key_words)
        df_filtered = df.filter(~col(certain_column).rlike(key_words_str))
        return df_filtered

    @staticmethod
    def pre_process(df: pyspark.sql.DataFrame, convert_to_numeric_list: list[str]) -> pyspark.sql.DataFrame:
        for column in convert_to_numeric_list:
            df = df.withColumn(column, col(column).cast('double'))
        if 'SITE_COUNT' in convert_to_numeric_list:
            convert_to_numeric_list.remove('SITE_COUNT')
        df = df.dropna(subset=convert_to_numeric_list, how='all')
        return df

    @staticmethod
    def integrate_columns(df: pyspark.sql.DataFrame,
                          merge_operno_list: List[Dict[str, List[str]]],
                          merge_prodg1_list: List[Dict[str, List[str]]],
                          merge_product_list: List[Dict[str, List[str]]]) -> pyspark.sql.DataFrame:
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
        df_merged = DataPreprocessorForInline.integrate_single_column(df, merge_operno_list, 'OPE_NO')
        df_merged = DataPreprocessorForInline.integrate_single_column(df_merged, merge_prodg1_list, 'PRODG1')
        df_merged = DataPreprocessorForInline.integrate_single_column(df_merged, merge_product_list, 'PRODUCT_ID')
        return df_merged

    @staticmethod
    def integrate_single_column(df: pyspark.sql.DataFrame,
                                merge_list: List[Dict[str, List[str]]],
                                column_name: str) -> pyspark.sql.DataFrame:
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
    def extract_unique_params_within_groups(df: pyspark.sql.DataFrame, grpby_list) -> pyspark.sql.DataFrame:
        grouped = df.groupby(*grpby_list).agg(collect_set('PARAMETRIC_NAME').alias('unique_values'))
        exploded = grouped.select(*grpby_list, explode(col('unique_values')).alias('PARAMETRIC_NAME'))
        unique_params_within_groups = exploded.dropDuplicates()
        return unique_params_within_groups

    def run(self) -> pyspark.sql.DataFrame:
        df_select = self.select_columns(df=self.df, columns_list=self.columns_list)
        df_esd = self.exclude_some_data(df=df_select, key_words=self.key_words, certain_column=self.certain_column)
        df_integrate = self.integrate_columns(df=df_esd,
                                              merge_operno_list=self.merge_operno_list,
                                              merge_prodg1_list=self.merge_prodg1_list,
                                              merge_product_list=self.merge_product_list)
        unique_params_within_groups = self.extract_unique_params_within_groups(df=df_integrate,
                                                                               grpby_list=self.grpby_list)
        df_preprocess = self.pre_process(df=df_integrate, convert_to_numeric_list=self.convert_to_numeric_list)
        unique_params_within_groups.persist()
        df_preprocess.persist()
        return unique_params_within_groups, df_preprocess


class GetTrainDataForInline:
    def __init__(self, df: pyspark.sql.DataFrame, grpby_list: list[str]):
        """
        Initialize the GetTrainDataForInline class.

        Parameters:
        - df (pyspark.sql.dataframe): Input DataFrame.
        - grpby_list (list): List of grouping columns.

        This class is designed to perform commonality analysis and retrieve training data based on the
        condition "bad_num >= 1 AND good_num >= 1" in each grpby_list, i.e. each OPE_NO for inline data.
        """
        self.df_run = df
        self.grpby_list = grpby_list

    @staticmethod
    def commonality_analysis(df_run: pyspark.sql.DataFrame, grpby_list: list[str]) -> pyspark.sql.DataFrame:
        grps_all = (df_run.groupBy(grpby_list)
                    .agg(countDistinct('WAFER_ID', when(df_run['label'] == 0, 1)).alias('GOOD_NUM'),
                         countDistinct('WAFER_ID', when(df_run['label'] == 1, 1)).alias('BAD_NUM'))
                    .na.fill(0))
        grps_all = grps_all.withColumn("conditions_satisfied",
                                       when((col('GOOD_NUM') >= 1) & (col('BAD_NUM') >= 1), True).otherwise(False))
        grps = grps_all.filter("GOOD_NUM >= 1 AND BAD_NUM >= 1")
        return grps_all, grps

    @staticmethod
    def get_data_list(common_res: pyspark.sql.DataFrame, grpby_list: list[str]) -> List[Dict[str, str]]:
        data_list = common_res.select(grpby_list).collect()
        data_dict_list = [row.asDict() for row in data_list]
        return data_dict_list

    @staticmethod
    def get_train_data(df_run: pyspark.sql.DataFrame, data_dict_list: List[Dict[str, str]]) -> pyspark.sql.DataFrame:
        first_data_dict = data_dict_list[0]
        conditions = " AND ".join(["{} == '{}'".format(col_, first_data_dict[col_]) for col_ in first_data_dict])
        df_s = df_run.filter(conditions)
        for i in range(1, len(data_dict_list)):
            data_dict = data_dict_list[i]
            conditions = " AND ".join(["{} == '{}'".format(col_, data_dict[col_]) for col_ in data_dict])
            df_m = df_run.filter(conditions)
            df_s = df_s.union(df_m)
        return df_s

    def run(self) -> pyspark.sql.DataFrame:
        grps_all, grps = self.commonality_analysis(df_run=self.df_run, grpby_list=self.grpby_list)
        # conditions_not_satisfied = grps_all.filter("conditions_satisfied==False").select(self.grpby_list + ["good_num", 'bad_num'])
        # parametric_for_unsatisfied_conditions = conditions_not_satisfied.join(self.df_run.select(self.grpby_list + ['INLINE_PARAMETER_ID']), self.grpby_list, "left").dropDuplicates()

        if grps.isEmpty():
            msg = f"按照{'+'.join(self.grpby_list)}分组后的数据, 没有组合满足条件good >= 1且bad >= 1, 无法进行分析."
            raise RCABaseException(msg)

        if grps_all.count() == grps.count():
            return grps_all, self.df_run
        else:
            data_dict_list = self.get_data_list(common_res=grps, grpby_list=self.grpby_list)
            train_data = self.get_train_data(df_run=self.df_run, data_dict_list=data_dict_list)
            return grps_all, train_data


class FitInlineModelByWafer:
    def __init__(self,
                 df: pyspark.sql.DataFrame,
                 grpby_list: list[str],
                 columns_to_process: list[str],
                 missing_value_threshold: float,
                 model: str = 'pca'):
        self.df = df
        self.grpby_list = grpby_list
        self.columns_to_process = columns_to_process
        self.missing_value_threshold = missing_value_threshold
        self.model = model

    @staticmethod
    def process_missing_values(df: pd.DataFrame, columns_to_process, missing_value_threshold) -> pd.DataFrame:
        for column in columns_to_process:
            missing_percentage = df[column].isnull().mean()
            if missing_percentage > missing_value_threshold:
                df = df.drop(columns=[column])
            else:
                df[column] = df[column].fillna(df[column].mean())
        return df

    @staticmethod
    def get_pivot_table(df: pd.DataFrame, grpby_list, columns_to_process, missing_value_threshold) -> pd.DataFrame:
        df_specific = FitInlineModelByWafer.process_missing_values(df, columns_to_process, missing_value_threshold)
        index_list = ['WAFER_ID', 'label']
        columns_list = grpby_list + ['PARAMETRIC_NAME']
        values_list = df_specific.columns.difference(
            ['WAFER_ID', 'PARAMETRIC_NAME', 'SITE_COUNT', 'label'] + grpby_list)
        pivot_result = df_specific.pivot_table(index=index_list,
                                               columns=columns_list,
                                               values=values_list)
        pivot_result.columns = pivot_result.columns.map('#'.join)
        pivot_result = FitInlineModelByWafer.process_missing_values(pivot_result, pivot_result.columns,
                                                                    missing_value_threshold)
        pivot_result = pivot_result.reset_index(drop=False)
        # Remove completely identical columns
        for column in pivot_result.columns.difference(index_list):
            if pivot_result[column].nunique() == 1:
                pivot_result = pivot_result.drop(column, axis=1)
        return pivot_result

    @staticmethod
    def fit_pca_model(df: pyspark.sql.DataFrame, grpby_list, columns_to_process,
                      missing_value_threshold) -> pyspark.sql.DataFrame:
        schema_all = StructType([StructField("features", StringType(), True),
                                 StructField("importance", FloatType(), True),
                                 StructField("algorithm_satisfied", StringType(), True),
                                 StructField("x_train_shape", StringType(), True)])

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run):
            pivot_result = FitInlineModelByWafer.get_pivot_table(df=df_run,
                                                                 grpby_list=grpby_list,
                                                                 columns_to_process=columns_to_process,
                                                                 missing_value_threshold=missing_value_threshold)

            x_train = pivot_result[pivot_result.columns.difference(['WAFER_ID', 'label']).tolist()]

            if min(x_train.shape) > 2:
                res_top_select = FitInlineModelByWafer.construct_features_when_satisfy_pca(x_train=x_train)
                return res_top_select
            else:
                res_top_select = FitInlineModelByWafer.construct_features_when_not_satisfied(x_train=x_train)
                return res_top_select

        return df.groupby(grpby_list).apply(get_model_result)

    @staticmethod
    def construct_features_when_not_satisfied(x_train) -> pd.DataFrame:
        x_len = len(x_train.columns)
        res_top_select = pd.DataFrame({"features": x_train.columns,
                                       "importance": [0.0] * x_len,
                                       "algorithm_satisfied": ['FALSE'] * x_len,
                                       "x_train_shape": [str(x_train.shape)] * x_len})
        return res_top_select

    @staticmethod
    def construct_features_when_satisfy_pca(x_train) -> pd.DataFrame:
        # 得到PCA算法结果res_top_select
        n_components = min(min(x_train.shape) - 2, 20)
        model = pca(n_components=n_components, verbose=None)
        results = model.fit_transform(x_train)
        res_top = results['topfeat']
        res_top_select = res_top[res_top['type'] == 'best'][['feature', 'loading']]
        res_top_select['importance'] = abs(res_top_select['loading'])
        res_top_select = res_top_select.rename(columns={'feature': 'features'}).drop("loading",
                                                                                     axis=1).drop_duplicates()
        # 增加算法为0的features
        has_importance_features = res_top_select['features'].values
        zero_importance_features = x_train.columns.difference(has_importance_features).to_list()
        len_f = len(zero_importance_features)
        zero_df = pd.DataFrame({'features': zero_importance_features, 'importance': [0.0] * len_f})
        res_top_select = res_top_select.append(zero_df, ignore_index=True)

        # 合并二者的结果
        res_top_select['x_train_shape'] = str(x_train.shape)
        res_top_select['algorithm_satisfied'] = 'TRUE'
        return res_top_select

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
    def fit_classification_model(df, grpby_list, columns_to_process, missing_value_threshold, model):
        schema_all = StructType([StructField("features", StringType(), True),
                                 StructField("importance", FloatType(), True)])

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run):
            pivot_result = FitInlineModelByWafer.get_pivot_table(df=df_run,
                                                                 grpby_list=grpby_list,
                                                                 columns_to_process=columns_to_process,
                                                                 missing_value_threshold=missing_value_threshold)
            x_train = pivot_result[pivot_result.columns.difference(['WAFER_ID', 'label']).tolist()]
            y_train = pivot_result[['label']]

            if x_train.shape[1] > 1 and y_train['label'].nunique() > 1:
                pipe, param_grid = FitInlineModelByWafer.get_pipe_params(model=model)
                try:  # cv=3 may be large
                    grid = GridSearchCV(estimator=pipe, scoring='roc_auc', param_grid=param_grid, cv=3, n_jobs=-1)
                    grid.fit(x_train.values, y_train.values.ravel())
                    # roc_auc_score_ = grid.best_score_
                except ValueError:
                    return pd.DataFrame()

                best_est = grid.best_estimator_.steps[-1][-1]
                if hasattr(best_est, 'feature_importances_'):
                    small_importance_res = pd.DataFrame({'features': x_train.columns,
                                                         'importance': best_est.feature_importances_})
                else:
                    small_importance_res = pd.DataFrame({'features': x_train.columns,
                                                         'importance': abs(best_est.coef_.ravel())})
                return small_importance_res

            else:
                small_importance_res = pd.DataFrame()
                return small_importance_res

        return df.groupby(grpby_list).apply(get_model_result)

    def run(self):
        if self.model == 'pca':
            res = self.fit_pca_model(df=self.df, grpby_list=self.grpby_list,
                                     columns_to_process=self.columns_to_process,
                                     missing_value_threshold=self.missing_value_threshold)

        else:
            res = self.fit_classification_model(df=self.df, grpby_list=self.grpby_list,
                                                columns_to_process=self.columns_to_process,
                                                missing_value_threshold=self.missing_value_threshold,
                                                model=self.model)
        return res


class SplitInlineModelResults:
    def __init__(self, df: pyspark.sql.DataFrame, grpby_list: List[str], request_id: str,
                 grps_all: pyspark.sql.DataFrame,
                 unique_params_within_groups: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        self.df = df
        self.grpby_list = grpby_list
        self.request_id = request_id
        self.grps_all = grps_all
        self.unique_params_within_groups = unique_params_within_groups

    @staticmethod
    def split_features(df: pd.DataFrame, index: int) -> str:
        return df['features'].apply(lambda x: x.split('#')[index])

    @staticmethod
    def get_split_features(df: pd.DataFrame, grpby_list: List[str]) -> pd.DataFrame:
        n_feats = len(grpby_list)
        for i in range(n_feats):
            df[grpby_list[i]] = SplitInlineModelResults.split_features(df, i + 1)

        df['PARAMETRIC_NAME'] = SplitInlineModelResults.split_features(df, n_feats + 1)
        df = df.drop(['features'], axis=1).reset_index(drop=True)
        return df

    @staticmethod
    def split_calculate_features(df: pyspark.sql.DataFrame, grpby_list: List[str], by: str) -> pyspark.sql.DataFrame:
        struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
        struct_fields.extend([StructField("PARAMETRIC_NAME", StringType(), True),
                              StructField("importance", FloatType(), True),
                              StructField("algorithm_satisfied", StringType(), True)])
        schema_all = StructType(struct_fields)

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run):
            split_table = SplitInlineModelResults.get_split_features(df=df_run, grpby_list=grpby_list)
            split_table_grpby = split_table.groupby(grpby_list + ['PARAMETRIC_NAME', 'algorithm_satisfied'])[
                'importance'].sum().reset_index(drop=False)
            return split_table_grpby

        return df.groupby(by).apply(get_model_result)

    @staticmethod
    def add_certain_column(df: pyspark.sql.DataFrame, request_id: str) -> pyspark.sql.DataFrame:
        total_importance = df.select(spark_sum("importance")).collect()[0][0]
        df = (df.withColumn("WEIGHT", col("importance") / total_importance)
              .withColumn("WEIGHT_PERCENT", col("WEIGHT") * 100))

        df = df.drop('importance')
        df = df.orderBy(col("WEIGHT").desc())
        df = (df.withColumn("GOOD_NUM", df["GOOD_NUM"].cast(FloatType()))
              .withColumn("BAD_NUM", df["BAD_NUM"].cast(FloatType()))
              .withColumn('INDEX_NO', monotonically_increasing_id() + 1)
              .withColumn("REQUEST_ID", lit(request_id)))

        info_list = ['PRODUCT_ID', 'OPE_NO', 'PRODG1']
        for column in info_list:
            if column not in df.columns:
                df = df.withColumn(column, lit(None).cast(StringType()))
        return df

    def run(self) -> pyspark.sql.DataFrame:
        df = self.df.withColumn('temp', lit(0))
        split_res = self.split_calculate_features(df=df, grpby_list=self.grpby_list, by='temp')

        split_res = split_res.drop('algorithm_satisfied')
        missing_rows = self.unique_params_within_groups.join(split_res,
                                                             on=self.grpby_list + ['PARAMETRIC_NAME'],
                                                             how='left_anti')
        missing_rows = missing_rows.withColumn('importance', lit(0))
        split_res_update_missing_features = split_res.unionByName(missing_rows, allowMissingColumns=True)
        print("split_res_update_missing_features:", split_res_update_missing_features.count())
        split_res_update_missing_features.show(10)

        split_res_update_wafer_num = split_res_update_missing_features.join(
            self.grps_all.select(self.grpby_list + ["GOOD_NUM", 'BAD_NUM']),
            on=self.grpby_list, how='left')
        print("split_res_update_wafer_num:", split_res_update_wafer_num.count())
        split_res_update_wafer_num.show(10)

        final_res = self.add_certain_column(df=split_res_update_wafer_num,
                                            request_id=self.request_id)
        final_res.unpersist()
        return final_res


class ExertInlineByWafer:
    @staticmethod
    def fit_by_wafer_model(df: pyspark.sql.DataFrame,
                           request_id: str,
                           merge_operno_list: List[Dict[str, List[str]]],
                           merge_prodg1_list: List[Dict[str, List[str]]],
                           merge_product_list: List[Dict[str, List[str]]],
                           columns_list=None,
                           key_words=None,
                           convert_to_numeric_list=None,
                           grpby_list=None,
                           certain_column=None,
                           model='pca') -> Union[str, pyspark.sql.DataFrame]:
        if grpby_list is None or len(grpby_list) == 0:
            grpby_list = ['OPE_NO']

        if columns_list is None:
            columns_list = grpby_list + ['WAFER_ID', 'PARAMETRIC_NAME', 'AVERAGE', 'MAX_VAL', 'MEDIAN', 'MIN_VAL',
                                         'STD_DEV', 'PERCENTILE_25', 'PERCENTILE_75', 'SITE_COUNT', 'label']
        if key_words is None:
            key_words = ['CXS', 'CYS', 'FDS']

        if convert_to_numeric_list is None:
            convert_to_numeric_list = ['AVERAGE', 'MAX_VAL', 'MEDIAN', 'MIN_VAL', 'STD_DEV', 'PERCENTILE_25',
                                       'PERCENTILE_75', 'SITE_COUNT']

        if certain_column is None:
            certain_column = 'PARAMETRIC_NAME'

        unique_params_within_groups, df_preprocess = DataPreprocessorForInline(df=df,
                                                                               grpby_list=grpby_list,
                                                                               columns_list=columns_list,
                                                                               certain_column=certain_column,
                                                                               key_words=key_words,
                                                                               convert_to_numeric_list=convert_to_numeric_list,
                                                                               merge_operno_list=merge_operno_list,
                                                                               merge_prodg1_list=merge_prodg1_list,
                                                                               merge_product_list=merge_product_list).run()
        if df_preprocess.isEmpty():
            msg = '数据库中暂无数据.'
            raise RCABaseException(msg)
        df_preprocess.persist()
        grps_all, df_train = GetTrainDataForInline(df=df_preprocess, grpby_list=grpby_list).run()
        if df_train.isEmpty():
            msg = f"按照{'+'.join(grpby_list)}分组后的训练数据暂时为空."
            raise RCABaseException(msg)

        res = FitInlineModelByWafer(df=df_train,
                                    grpby_list=grpby_list,
                                    columns_to_process=['AVERAGE', 'MAX_VAL', 'MEDIAN', 'MIN_VAL', 'STD_DEV',
                                                        'PERCENTILE_25', 'PERCENTILE_75'],
                                    missing_value_threshold=0.6,
                                    model=model).run()

        m = res.filter('algorithm_satisfied==True').count()
        res.unpersist()
        res.persist()
        if m == 0:
            msg = f"按照{'+'.join(grpby_list)}分组后的数据，没有具有差异性的参数"
            raise RCABaseException(msg)

        final_res = SplitInlineModelResults(df=res, grpby_list=grpby_list, request_id=request_id,
                                            grps_all=grps_all,
                                            unique_params_within_groups=unique_params_within_groups).run()
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
    # import findspark
    # findspark.init()
    # spark = SparkSession \
    #     .builder \
    #     .appName("ywj") \
    #     .config('spark.sql.session.timeZone', 'Asia/Shanghai') \
    #     .master("local[*]") \
    #     .getOrCreate()

    df_pandas = pd.read_csv(
        "D:/Jupyterfiles/晶合MVAFDC_general开发/MVAanlysisDevelop/inline_algorithm/inline_case5_label_1.csv")
    df_pandas = df_pandas[
        df_pandas['OPE_NO'].isin(['1V.EQW10', '1V.PQW10', '1F.FQE10', '1C.CDG10', '1U.EQW10', '1U.PQW10'])]
    df_spark = ps.from_pandas(df_pandas).to_spark()
    print(f"df_spark shape: ({df_spark.count()}, {len(df_spark.columns)})")
    df_spark.show()

    json_loads_dict = {"requestId": "269",
                       "algorithm": "inline_by_wafer",
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

                                        # "mergeOperno": [{"xx1": ["1C.CDG10", "1V.EQW10", "1U.PQW10"]},
                                        #                 {"xx2": ["1V.PQW10", "1F.FQE10"]}],

                                        "mergeOperno": None
                                        }
                       }

    df_ = pd.DataFrame({"requestId": [json_loads_dict["requestId"]],
                        "requestParam": [json.dumps(json_loads_dict["requestParam"])]})

    request_id_ = df_["requestId"].values[0]
    request_params = df_["requestParam"].values[0]
    parse_dict = json.loads(request_params)

    merge_operno = list(parse_dict.get('mergeOperno')) if parse_dict.get('mergeOperno') else None
    merge_prodg1 = list(parse_dict.get('mergeProdg1')) if parse_dict.get('mergeProdg1') else None
    merge_product = list(parse_dict.get('mergeProductId')) if parse_dict.get('mergeProductId') else None
    grpby_list_ = ['OPE_NO']
    # grpby_list_ = ['OPE_NO', 'PRODUCT_ID']

    from datetime import datetime

    time1 = datetime.now()
    print(time1)
    final_res_ = ExertInlineByWafer.fit_by_wafer_model(df=df_spark, request_id=request_id_,
                                                       grpby_list=grpby_list_,
                                                       merge_operno_list=merge_operno,
                                                       merge_prodg1_list=merge_prodg1,
                                                       merge_product_list=merge_product)
    print("final_res_:", final_res_.count())
    final_res_pandas = final_res_.toPandas()
    final_res_pandas.to_csv("final_res_pandas1.csv")
    time2 = datetime.now()
    print(time2, time2 - time1)
