import pyspark
import pandas as pd
from pca import pca
from typing import Union, List, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from pyspark.sql.types import StructType, StructField, StringType, FloatType, BooleanType
from pyspark.sql.functions import pandas_udf, PandasUDFType, lit, col, when, sum as spark_sum, monotonically_increasing_id
from src.exceptions.rca_base_exception import RCABaseException


class DataPreprocessorForInline:
    def __init__(self,
                 df: pyspark.sql.dataframe,
                 columns_list: list[str],
                 certain_column: str,
                 key_words: list[str],
                 convert_to_numeric_list: list[str],
                 merge_operno_list: List[Dict[str, List[str]]],
                 merge_prodg1_list: List[Dict[str, List[str]]],
                 merge_product_list: List[Dict[str, List[str]]]
                 ):
        self.df = df
        self.columns_list = columns_list
        self.certain_column = certain_column
        self.key_words = key_words
        self.convert_to_numeric_list = convert_to_numeric_list
        self.merge_operno_list = merge_operno_list
        self.merge_prodg1_list = merge_prodg1_list
        self.merge_product_list = merge_product_list

    @staticmethod
    def select_columns(df: pyspark.sql.dataframe, columns_list: list[str]) -> pyspark.sql.dataframe:
        return df.select(columns_list)

    @staticmethod
    def exclude_some_data(df: pyspark.sql.dataframe, key_words: list[str],
                          certain_column: str) -> pyspark.sql.dataframe:
        key_words_str = '|'.join(key_words)
        df_filtered = df.filter(~col(certain_column).rlike(key_words_str))
        return df_filtered

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
        df_merged = DataPreprocessorForInline.integrate_single_column(df, merge_operno_list, 'OPER_NO')
        df_merged = DataPreprocessorForInline.integrate_single_column(df_merged, merge_prodg1_list, 'PRODG1')
        df_merged = DataPreprocessorForInline.integrate_single_column(df_merged, merge_product_list, 'PRODUCT_ID')
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

    def run(self) -> pyspark.sql.dataframe:
        df_select = self.select_columns(df=self.df, columns_list=self.columns_list)
        df_esd = self.exclude_some_data(df=df_select, key_words=self.key_words, certain_column=self.certain_column)
        df_pp = self.pre_process(df=df_esd, convert_to_numeric_list=self.convert_to_numeric_list)
        df_integrate = self.integrate_columns(df=df_pp,
                                              merge_operno_list=self.merge_operno_list,
                                              merge_prodg1_list=self.merge_prodg1_list,
                                              merge_product_list=self.merge_product_list)
        return df_integrate


class ExtractFeaturesBySite:
    @staticmethod
    def process_missing_values_for_site(df: pd.DataFrame,
                                        good_site_columns: list[str],
                                        bad_site_columns: list[str],
                                        missing_value_threshold: Union[int, float] = 0.7,
                                        process_miss_site_mode: str = 'drop') -> pd.DataFrame:
        assert process_miss_site_mode in ['drop', 'fill']
        site_columns = good_site_columns + bad_site_columns
        if process_miss_site_mode == 'drop':
            # drop rows based on the missing value threshold
            df = df.dropna(subset=site_columns, thresh=missing_value_threshold)
        else:
            # fill missing values in the corresponding site rows using the AVERAGE of that row
            df[site_columns] = df[site_columns].apply(lambda column: column.fillna(df['AVERAGE']))
        return df

    @staticmethod
    def calculate_statistics(row):
        return pd.Series({
            'MAX_VAL': row.max(),
            'MIN_VAL': row.min(),
            'MEDIAN': row.median(),
            'AVERAGE': row.mean(),
            'STD_DEV': row.std(),
            'PERCENTILE_25': row.quantile(0.25),
            'PERCENTILE_75': row.quantile(0.75)})

    @staticmethod
    def calculate_site_stats(df: pd.DataFrame, grpby_list: list[str], site_columns: list[str],
                             good_or_bad: str) -> pd.DataFrame:
        assert good_or_bad in ['good', 'bad'], "Label could only be 'good' or 'bad'"
        selected_df = df[grpby_list + ['WAFER_ID', 'INLINE_PARAMETER_ID'] + site_columns].reset_index(drop=True)
        # Perform statistical calculations for each row
        side_features = selected_df.apply(lambda row: ExtractFeaturesBySite.calculate_statistics(row[site_columns]),
                                          axis=1)
        side_features = side_features.fillna(0)
        df_with_features = pd.concat([selected_df, side_features], axis=1)
        if good_or_bad == 'good':
            df_with_features['label'] = 0
        else:
            df_with_features['label'] = 1
        return df_with_features

    @staticmethod
    def extract_features_by_site(df: pd.DataFrame,
                                 grpby_list: list[str],
                                 good_site_columns: list[str],
                                 bad_site_columns: list[str],
                                 missing_value_threshold: Union[int, float] = 0.7,
                                 process_miss_site_mode: str = 'drop') -> Union[pd.DataFrame, None]:
        """
        Extracts features from a DataFrame based on good and bad site columns.
        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - grp_list: ['OPE_NO'] for most the case.
        - good_site_columns (list): List of columns representing good sites.
        - bad_site_columns (list): List of columns representing bad sites.
        - missing_value_threshold (Union[int, float]): Threshold for missing values.
        - process_miss_site_mode (str): Mode for handling missing values in site columns, e.g. drop or fill
        Returns:
        - Union[pd.DataFrame, None]: DataFrame with extracted features or None if no data is available.
        """
        df_pandas_specific_ = ExtractFeaturesBySite.process_missing_values_for_site(df=df,
                                                                                    good_site_columns=good_site_columns,
                                                                                    bad_site_columns=bad_site_columns,
                                                                                    missing_value_threshold=missing_value_threshold,
                                                                                    process_miss_site_mode=process_miss_site_mode)
        if df_pandas_specific_.shape[0] != 0:
            side_with_features1 = ExtractFeaturesBySite.calculate_site_stats(df_pandas_specific_, grpby_list,
                                                                             good_site_columns,
                                                                             good_or_bad='good')
            side_with_features2 = ExtractFeaturesBySite.calculate_site_stats(df_pandas_specific_, grpby_list,
                                                                             bad_site_columns,
                                                                             good_or_bad='bad')
            side_with_features1_select = side_with_features1[
                grpby_list + ['WAFER_ID', 'INLINE_PARAMETER_ID', 'MAX_VAL', 'MIN_VAL', 'MEDIAN',
                              'AVERAGE', 'STD_DEV', 'PERCENTILE_25', 'PERCENTILE_75', 'label']]
            side_with_features2_select = side_with_features2[
                grpby_list + ['WAFER_ID', 'INLINE_PARAMETER_ID', 'MAX_VAL', 'MIN_VAL', 'MEDIAN',
                              'AVERAGE', 'STD_DEV', 'PERCENTILE_25', 'PERCENTILE_75', 'label']]
            side_with_features_all = pd.concat([side_with_features1_select, side_with_features2_select], axis=0)
            return side_with_features_all


class FitInlineModelBySite:
    def __init__(self,
                 df: pyspark.sql.dataframe,
                 grpby_list: list[str],
                 good_site_columns: list[str],
                 bad_site_columns: list[str],
                 process_miss_site_mode: str,
                 columns_to_process: list[str],
                 missing_value_threshold: Union[int, float],
                 model: str = 'pca'):
        """
        Initialize the FitInlineModelBySite object.

        Parameters:
        - df: pyspark.sql.dataframe, the input data
        - grpby_list: list[str], the grouping variable, inline data should be ["OPE_NO"] mostly
        - good_site_columns: List of str, column names for good sites
        - bad_site_columns: List of str, column names for bad sites
        - process_miss_site_mode: str, mode for handling missing values in site data, e.g. drop or fill
        - columns_to_process: List of str, columns to process in missing value functions
        - missing_value_threshold: Union[int, float], threshold for missing values
        - model: str, default is 'pca', other options include 'rf' for random forest, 'decisionTree' for decision tree,
                 svc, logistic and sgd.
        """
        self.df = df
        self.grpby_list = grpby_list
        self.good_site_columns = good_site_columns
        self.bad_site_columns = bad_site_columns
        self.process_miss_site_mode = process_miss_site_mode
        self.columns_to_process = columns_to_process
        self.missing_value_threshold = missing_value_threshold
        self.model = model

    @staticmethod
    def process_missing_values(df, columns_to_process, missing_value_threshold):
        for column in columns_to_process:
            missing_percentage = df[column].isnull().mean()
            if missing_percentage > missing_value_threshold:
                df = df.drop(columns=[column])
            else:
                df[column] = df[column].fillna(df[column].mean())
        return df

    @staticmethod
    def get_pivot_table(df, grpby_list, columns_to_process, missing_value_threshold):
        df = FitInlineModelBySite.process_missing_values(df, columns_to_process, missing_value_threshold)
        index_list = ['WAFER_ID', 'label']
        columns_list = grpby_list + ['INLINE_PARAMETER_ID']
        values_list = df.columns.difference(['WAFER_ID', 'INLINE_PARAMETER_ID', 'label'] + grpby_list)
        pivot_result = df.pivot_table(index=index_list,
                                      columns=columns_list,
                                      values=values_list)
        pivot_result.columns = pivot_result.columns.map('#'.join)
        pivot_result = FitInlineModelBySite.process_missing_values(pivot_result, pivot_result.columns,
                                                                   missing_value_threshold)
        pivot_result = pivot_result.reset_index(drop=False)
        # Remove completely identical columns
        for column in pivot_result.columns.difference(index_list):
            if pivot_result[column].nunique() == 1:
                pivot_result = pivot_result.drop(column, axis=1)
        return pivot_result

    @staticmethod
    def construct_features_when_not_satisfied(grpby_list, df_run, x_train_shape):
        grpby_values = [df_run[item].iloc[0] for item in grpby_list]
        features_value = f"STATS#{'#'.join(map(str, grpby_values))}#PARAM"
        res_top_select = pd.DataFrame({"features": features_value,
                                       "importance": -1,
                                       "algorithm_satisfied": False,
                                       "x_train_shape": str(x_train_shape),
                                       "valid_wafer_num": x_train_shape[0]/2}, index=[0])
        return res_top_select

    @staticmethod
    def construct_features_when_satisfy_pca(x_train):
        n_components = min(min(x_train.shape) - 2, 20)
        model = pca(n_components=n_components, verbose=None)
        results = model.fit_transform(x_train)
        res_top = results['topfeat']
        res_top_select = res_top[res_top['type'] == 'best'][['feature', 'loading']]
        res_top_select['importance'] = abs(res_top_select['loading'])
        res_top_select = res_top_select.rename(columns={'feature': 'features'}).drop("loading",
                                                                                     axis=1).drop_duplicates()
        res_top_select['x_train_shape'] = str(x_train.shape)
        res_top_select['valid_wafer_num'] = x_train.shape[0]/2
        res_top_select['algorithm_satisfied'] = True
        return res_top_select

    @staticmethod
    def fit_pca_model(df, grpby_list, good_site_columns, bad_site_columns, columns_to_process, process_miss_site_mode,
                      missing_value_threshold):
        schema_all = StructType([StructField("features", StringType(), True),
                                 StructField("importance", FloatType(), True),
                                 StructField("algorithm_satisfied", BooleanType(), True),
                                 StructField("x_train_shape", StringType(), True),
                                 StructField("valid_wafer_num", FloatType(), True)])

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run):
            side_with_features_all = ExtractFeaturesBySite.extract_features_by_site(df=df_run,
                                                                                    grpby_list=grpby_list,
                                                                                    good_site_columns=good_site_columns,
                                                                                    bad_site_columns=bad_site_columns,
                                                                                    missing_value_threshold=missing_value_threshold,
                                                                                    process_miss_site_mode=process_miss_site_mode)
            if side_with_features_all is None:
                grpby_values = [df_run[item].iloc[0] for item in grpby_list]
                features_value = f"STATS#{'#'.join(map(str, grpby_values))}#PARAM"
                res_top_select = pd.DataFrame({"features": features_value,
                                               "importance": -2,
                                               "algorithm_satisfied": False,
                                               "x_train_shape": str(0),
                                               "valid_wafer_num": 0.0}, index=[0])
                return res_top_select

            pivot_result = FitInlineModelBySite.get_pivot_table(df=side_with_features_all,
                                                                grpby_list=grpby_list,
                                                                columns_to_process=columns_to_process,
                                                                missing_value_threshold=missing_value_threshold)
            x_train = pivot_result[pivot_result.columns.difference(['WAFER_ID', 'label']).tolist()]

            if min(x_train.shape) > 2:
                res_top_select = FitInlineModelBySite.construct_features_when_satisfy_pca(x_train=x_train)
                return res_top_select
            else:
                res_top_select = FitInlineModelBySite.construct_features_when_not_satisfied(grpby_list, df_run, x_train.shape)
                return res_top_select
        return df.groupby(grpby_list).apply(get_model_result)

    # @staticmethod
    # def fit_pca_model(df, grpby_list, good_site_columns, bad_site_columns, columns_to_process, process_miss_site_mode,
    #                   missing_value_threshold):
    #     schema_all = StructType([StructField("features", StringType(), True),
    #                              StructField("importance", FloatType(), True)])
    #
    #     @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
    #     def get_model_result(df_run):
    #         side_with_features_all = ExtractFeaturesBySite.extract_features_by_site(df=df_run,
    #                                                                                 grpby_list=grpby_list,
    #                                                                                 good_site_columns=good_site_columns,
    #                                                                                 bad_site_columns=bad_site_columns,
    #                                                                                 missing_value_threshold=missing_value_threshold,
    #                                                                                 process_miss_site_mode=process_miss_site_mode)
    #         if side_with_features_all is None:
    #             return pd.DataFrame()
    #
    #         pivot_result = FitInlineModelBySite.get_pivot_table(df=side_with_features_all,
    #                                                             grpby_list=grpby_list,
    #                                                             columns_to_process=columns_to_process,
    #                                                             missing_value_threshold=missing_value_threshold)
    #         x_train = pivot_result[pivot_result.columns.difference(['WAFER_ID', 'label']).tolist()]
    #
    #         if x_train.shape[1] > 1:
    #             n_components = min(min(x_train.shape) - 2, 20)
    #             model = pca(n_components=n_components, verbose=None)
    #             results = model.fit_transform(x_train)
    #             res_top = results['topfeat']
    #             res_top_select = res_top[res_top['type'] == 'best'][['feature', 'loading']]
    #             res_top_select['importance'] = abs(res_top_select['loading'])
    #             res_top_select = res_top_select.rename(columns={'feature': 'features'}).drop("loading",
    #                                                                                          axis=1).drop_duplicates()
    #             return res_top_select
    #         else:
    #             res_top_select = pd.DataFrame()
    #             return res_top_select
    #
    #     return df.groupby(grpby_list).apply(get_model_result)

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
    def fit_classification_model(df, grpby_list, good_site_columns, bad_site_columns,
                                 columns_to_process, process_miss_site_mode, missing_value_threshold, model):
        schema_all = StructType([StructField("features", StringType(), True),
                                 StructField("importance", FloatType(), True)])

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run):
            side_with_features_all = ExtractFeaturesBySite.extract_features_by_site(df=df_run,
                                                                                    grpby_list=grpby_list,
                                                                                    good_site_columns=good_site_columns,
                                                                                    bad_site_columns=bad_site_columns,
                                                                                    missing_value_threshold=missing_value_threshold,
                                                                                    process_miss_site_mode=process_miss_site_mode)
            if side_with_features_all is None:
                return pd.DataFrame()

            pivot_result = FitInlineModelBySite.get_pivot_table(df=side_with_features_all,
                                                                grpby_list=grpby_list,
                                                                columns_to_process=columns_to_process,
                                                                missing_value_threshold=missing_value_threshold)
            x_train = pivot_result[pivot_result.columns.difference(['WAFER_ID', 'label']).tolist()]
            y_train = pivot_result[['label']]

            if min(x_train.shape) > 4 and y_train['label'].nunique() > 1:
                pipe, param_grid = FitInlineModelBySite.get_pipe_params(model=model)
                try:
                    grid = GridSearchCV(estimator=pipe, scoring='roc_auc', param_grid=param_grid, cv=3, n_jobs=-1)
                    grid.fit(x_train.values, y_train.values.ravel())

                except ValueError:
                    small_importance_res = pd.DataFrame()
                    return small_importance_res

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
                                     good_site_columns=self.good_site_columns,
                                     bad_site_columns=self.bad_site_columns,
                                     columns_to_process=self.columns_to_process,
                                     process_miss_site_mode=self.process_miss_site_mode,
                                     missing_value_threshold=self.missing_value_threshold)
        else:
            res = self.fit_classification_model(df=self.df, grpby_list=self.grpby_list,
                                                good_site_columns=self.good_site_columns,
                                                bad_site_columns=self.bad_site_columns,
                                                columns_to_process=self.columns_to_process,
                                                process_miss_site_mode=self.process_miss_site_mode,
                                                missing_value_threshold=self.missing_value_threshold,
                                                model=self.model)
        return res


class SplitInlineModelResults:
    def __init__(self, df: pyspark.sql.dataframe, grpby_list: List[str], request_id: str):
        self.df = df
        self.grpby_list = grpby_list
        self.request_id = request_id

    @staticmethod
    def split_features(df: pd.DataFrame, index: int) -> str:
        return df['features'].apply(lambda x: x.split('#')[index])

    @staticmethod
    def get_split_features(df: pd.DataFrame, grpby_list: List[str]) -> pd.DataFrame:
        n_feats = len(grpby_list)
        for i in range(n_feats):
            df[grpby_list[i]] = SplitInlineModelResults.split_features(df, i + 1)

        df['INLINE_PARAMETER_ID'] = SplitInlineModelResults.split_features(df, n_feats + 1)
        df = df.drop(['features'], axis=1).reset_index(drop=True)
        return df

    @staticmethod
    def split_calculate_features(df: pyspark.sql.dataframe, grpby_list: List[str], by: str) -> pyspark.sql.dataframe:
        struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
        struct_fields.extend([StructField("INLINE_PARAMETER_ID", StringType(), True),
                              StructField("importance", FloatType(), True),
                              StructField("valid_wafer_num", FloatType(), True)])
        schema_all = StructType(struct_fields)

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run):
            split_table = SplitInlineModelResults.get_split_features(df=df_run, grpby_list=grpby_list)
            split_table_grpby = split_table.groupby(grpby_list + ['INLINE_PARAMETER_ID', 'valid_wafer_num'])[
                'importance'].sum().reset_index(drop=False)
            return split_table_grpby

        return df.groupby(by).apply(get_model_result)

    # @staticmethod
    # def add_certain_column(df: pyspark.sql.dataframe, grpby_list: List[str], request_id: str,
    #                        by: str) -> pyspark.sql.dataframe:
    #     struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
    #     struct_fields.extend([StructField("INLINE_PARAMETER_ID", StringType(), True),
    #                           StructField("AVG_SPEC_CHK_RESULT_COUNT", FloatType(), True),
    #                           StructField("request_id", StringType(), True),
    #                           StructField("weight", FloatType(), True),
    #                           StructField("weight_percent", FloatType(), True),
    #                           StructField("index_no", IntegerType(), True)])
    #     schema_all = StructType(struct_fields)
    #
    #     @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
    #     def get_result(final_res):
    #         # Calculate weights and normalize
    #         final_res['importance'] = final_res['importance'].astype(float)
    #         final_res = final_res.query("importance > 0")
    #         final_res['weight'] = final_res['importance'] / final_res['importance'].sum()
    #         final_res['weight_percent'] = final_res['weight'] * 100
    #         final_res = final_res.sort_values('weight', ascending=False)
    #
    #         final_res['index_no'] = [i + 1 for i in range(len(final_res))]
    #         final_res['AVG_SPEC_CHK_RESULT_COUNT'] = 0.0
    #         final_res['request_id'] = request_id
    #         final_res = final_res.drop(['importance', 'temp'], axis=1)
    #         return final_res
    #
    #     return df.groupby(by).apply(get_result)
    @staticmethod
    def add_certain_column(df: pyspark.sql.dataframe, request_id: str):
        df = df.filter("importance > 0")
        total_importance = df.select(spark_sum("importance")).collect()[0][0]
        df = (df.withColumn("weight", col("importance") / total_importance)
              .withColumn("weight_percent", col("weight") * 100)
              .withColumn("request_id", lit(request_id))
              .withColumn("AVG_SPEC_CHK_RESULT_COUNT", lit(0.0))
              .withColumn("bad_num", lit(None).cast(FloatType()))
              .withColumnRenamed("valid_wafer_num", "good_num"))

        df = df.drop('importance')
        df = df.orderBy(col("weight").desc())
        df = df.withColumn('index_no', monotonically_increasing_id() + 1)

        info_list = ['PRODUCT_ID', 'OPER_NO', 'PRODG1']
        for column in info_list:
            if column not in df.columns:
                df = df.withColumn(column, lit(None).cast(StringType()))
        return df

    def run(self):
        df = self.df.withColumn('temp', lit(0))
        split_res = self.split_calculate_features(df=df, grpby_list=self.grpby_list, by='temp')
        print("split_res:")
        split_res.show()

        final_res = self.add_certain_column(df=split_res, request_id=self.request_id)
        print("final_res:")
        final_res.show()
        print(final_res.dtypes)
        return final_res


class ExertInlineBySite:
    @staticmethod
    def fit_by_site_model(df: pyspark.sql.dataframe,
                          request_id: str,
                          merge_operno_list: List[Dict[str, List[str]]],
                          merge_prodg1_list: List[Dict[str, List[str]]],
                          merge_product_list: List[Dict[str, List[str]]],
                          good_site_columns: List[str],
                          bad_site_columns: List[str],
                          columns_list=None,
                          key_words=None,
                          convert_to_numeric_list=None,
                          grpby_list=None,
                          certain_column=None,
                          model='pca') -> Union[str, pyspark.sql.dataframe.DataFrame]:
        if good_site_columns is None or len(good_site_columns) == 0:
            msg = "good sites为空, 请指定good sites."
            raise RCABaseException(msg)
        if bad_site_columns is None or len(bad_site_columns) == 0:
            msg = "bad sites为空, 请指定bad sites."
            raise RCABaseException(msg)

        # drop duplicates
        good_site_columns = list(set(good_site_columns))
        bad_site_columns = list(set(bad_site_columns))
        site_columns = good_site_columns + bad_site_columns

        if grpby_list is None or len(grpby_list) == 0:
            grpby_list = ['OPER_NO', 'PRODUCT_ID']

        if columns_list is None:
            columns_list = grpby_list + ['WAFER_ID', 'INLINE_PARAMETER_ID', 'SITE_COUNT', 'AVERAGE'] + site_columns

        if key_words is None:
            key_words = ['CXS', 'CYS', 'FDS']

        if convert_to_numeric_list is None:
            convert_to_numeric_list = ['SITE_COUNT', 'AVERAGE'] + site_columns

        if certain_column is None:
            certain_column = 'INLINE_PARAMETER_ID'

        df = df.withColumnRenamed('OPE_NO', 'OPER_NO')
        df_preprocess = DataPreprocessorForInline(df=df,
                                                  columns_list=columns_list,
                                                  certain_column=certain_column,
                                                  key_words=key_words,
                                                  convert_to_numeric_list=convert_to_numeric_list,
                                                  merge_operno_list=merge_operno_list,
                                                  merge_prodg1_list=merge_prodg1_list,
                                                  merge_product_list=merge_product_list).run()
        print(f"df_preprocess shape: ({df_preprocess.count()}, {len(df_preprocess.columns)})")
        df_preprocess.show()
        if df_preprocess.isEmpty():
            msg = '数据库中暂无数据.'
            raise RCABaseException(msg)

        res = FitInlineModelBySite(df=df_preprocess,
                                   grpby_list=grpby_list,
                                   good_site_columns=good_site_columns,
                                   bad_site_columns=bad_site_columns,
                                   process_miss_site_mode='drop',
                                   columns_to_process=['AVERAGE', 'MAX_VAL', 'MEDIAN', 'MIN_VAL', 'STD_DEV',
                                                       'PERCENTILE_25', 'PERCENTILE_75'],
                                   missing_value_threshold=0.7,
                                   model=model).run()
        print(f"res shape: ({res.count()}, {len(res.columns)})")
        res.show(50)
        print(res.select('features').collect())

        m = res.filter('algorithm_satisfied==True').count()
        if m == 0:
            unique_importance_values = res.select('importance').distinct().collect()
            unique_importance_values_list = [row.importance for row in unique_importance_values]
            if -1 in unique_importance_values_list and len(unique_importance_values_list) == 1:
                msg = f"按照{'+'.join(grpby_list)}分组后的数据，没有具有差异性的参数"
                raise RCABaseException(msg)
            elif -2 in unique_importance_values_list and len(unique_importance_values_list) == 1:
                msg = f"按照{'+'.join(grpby_list)}分组后的数据，每个组合中的site_values的缺失值超过了70%，无法进行分析"
                raise RCABaseException(msg)
            else:
                msg = f"按照{'+'.join(grpby_list)}分组后的数据，site_values的缺失值超过了70%，且没有具有差异性的参数"
                raise RCABaseException(msg)

        final_res = SplitInlineModelResults(df=res, grpby_list=grpby_list, request_id=request_id).run()
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

    df_pandas = pd.read_csv(
        "D:/Jupyterfiles/晶合MVAFDC_general开发/MVAanlysisDevelop/inline_algorithm/inline_case5_label.csv")
    # df_pandas = df_pandas[
    #     df_pandas['OPE_NO'].isin(['1V.EQW10', '1V.PQW10', '1F.FQE10', '1C.CDG10', '1U.EQW10', '1U.PQW10'])]
    df_spark = ps.from_pandas(df_pandas).to_spark()

    json_loads_dict = {"requestId": "269",
                       "algorithm": "inline_by_site",
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
                                        "goodSite": ["SITE4_VAL", "SITE8_VAL", "SITE9_VAL", "SITE12_VAL", "SITE13_VAL"],
                                        "badSite": ["SITE2_VAL", "SITE6_VAL", "SITE7_VAL", "SITE10_VAL", "SITE11_VAL"],
                                        # "mergeOperno": []
                                        "mergeOperno": [{"1F.FQE10,1C.CDG10": ["1F.FQE10", "1C.CDG10"]},
                                                        {"1U.EQW10_1U.PQW10": ["1U.EQW10", "1U.PQW10"]}]
                                        }
                       }

    df_ = pd.DataFrame({"requestId": [json_loads_dict["requestId"]],
                        "requestParam": [json.dumps(json_loads_dict["requestParam"])]})

    request_id_ = df_["requestId"].values[0]
    request_params = df_["requestParam"].values[0]
    parse_dict = json.loads(request_params)

    merge_operno = list(parse_dict.get('mergeOperno')) if parse_dict.get('mergeOperno') else None
    good_site_columns_ = list(parse_dict.get('goodSite')) if parse_dict.get('goodSite') else None
    bad_site_columns_ = list(parse_dict.get('badSite')) if parse_dict.get('badSite') else None
    merge_prodg1 = list(parse_dict.get('mergeProdg1')) if parse_dict.get('mergeProdg1') else None
    merge_product = list(parse_dict.get('mergeProductId')) if parse_dict.get('mergeProductId') else None

    from datetime import datetime
    time1 = datetime.now()
    print(time1)
    final_res_ = ExertInlineBySite.fit_by_site_model(df=df_spark,
                                                     request_id=request_id_,
                                                     merge_operno_list=merge_operno,
                                                     good_site_columns=good_site_columns_,
                                                     bad_site_columns=bad_site_columns_,
                                                     merge_prodg1_list=merge_prodg1,
                                                     merge_product_list=merge_product)
    time2 = datetime.now()
    print(time2, time2-time1)
