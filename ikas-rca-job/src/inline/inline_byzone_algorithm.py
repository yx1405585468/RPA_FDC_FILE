import pyspark
import pandas as pd
from pca import pca
from typing import Union, List, Dict
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql.functions import pandas_udf, PandasUDFType, lit, col, when, sum as spark_sum, monotonically_increasing_id, countDistinct
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
        df_merged = DataPreprocessorForInline.integrate_single_column(df, merge_operno_list, 'OPE_NO')
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

    @staticmethod
    def commonality_analysis(df_run: pyspark.sql.dataframe, grpby_list: list[str]) -> pyspark.sql.dataframe:
        grps_all = df_run.groupBy(grpby_list).agg(countDistinct('WAFER_ID').alias('GOOD_NUM'))
        return grps_all

    @staticmethod
    def add_feature_stats_within_groups(df_integrate: pyspark.sql.dataframe, grpby_list) -> pyspark.sql.dataframe:
        unique_params_within_groups = (df_integrate.groupBy(grpby_list + ['PARAMETRIC_NAME'])
                                       .agg(countDistinct('WAFER_ID').alias('GOOD_NUM'))
                                       .na.fill(0))
        return unique_params_within_groups

    def run(self) -> pyspark.sql.dataframe:
        df_select = self.select_columns(df=self.df, columns_list=self.columns_list)
        df_esd = self.exclude_some_data(df=df_select, key_words=self.key_words, certain_column=self.certain_column)
        df_integrate = self.integrate_columns(df=df_esd,
                                              merge_operno_list=self.merge_operno_list,
                                              merge_prodg1_list=self.merge_prodg1_list,
                                              merge_product_list=self.merge_product_list)
        grps_all = self.commonality_analysis(df_run=df_integrate, grpby_list=self.grpby_list)
        add_parametric_stats_df = self.add_feature_stats_within_groups(df_integrate=df_integrate,
                                                                       grpby_list=self.grpby_list)
        df_preprocess = self.pre_process(df=df_integrate, convert_to_numeric_list=self.convert_to_numeric_list)
        return grps_all, add_parametric_stats_df, df_preprocess


class ExtractFeaturesByZone:
    @staticmethod
    def process_missing_values_for_zone(df: pd.DataFrame,
                                        zone_columns: List[str],
                                        missing_value_threshold: Union[int, float] = 0.7,
                                        process_miss_zone_mode: str = 'drop') -> pd.DataFrame:
        assert process_miss_zone_mode in ['drop', 'fill'], "process_miss_zone_mode should be either 'drop' or 'fill'"
        if process_miss_zone_mode == 'drop':
            # drop rows based on the missing value threshold
            df = df.dropna(thresh=int(len(zone_columns) * missing_value_threshold), subset=zone_columns)
        else:
            # fill missing values in the corresponding zone rows using the AVERAGE of that row
            df[zone_columns] = df[zone_columns].apply(lambda column: column.fillna(df['AVERAGE']))

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
            'PERCENTILE_75': row.quantile(0.75)
        })

    @staticmethod
    def calculate_zone_stats(df: pd.DataFrame, grpby_list: List[str], zone_columns: List[str]) -> pd.DataFrame:
        selected_df = df[grpby_list + ['WAFER_ID', 'PARAMETRIC_NAME'] + zone_columns].reset_index(drop=True)
        # Perform statistical calculations for each row
        zone_features = selected_df.apply(lambda row: ExtractFeaturesByZone.calculate_statistics(row[zone_columns]),
                                          axis=1)
        zone_features = zone_features.fillna(0)

        df_with_features = pd.concat([selected_df, zone_features], axis=1)
        return df_with_features

    @staticmethod
    def extract_features_by_zone(df: pd.DataFrame,
                                 grpby_list: List[str],
                                 zone_columns: List[str],
                                 missing_value_threshold: Union[int, float] = 0.7,
                                 process_miss_zone_mode: str = 'drop') -> Union[pd.DataFrame, None]:
        """
        Extracts features from a DataFrame based on zone columns.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - grpby_list (List[str]): Group by list, e.g. ['OPE_NO'].
        - zone_columns (List[str]): List of columns representing a zone.
        - missing_value_threshold (Union[int, float]): Threshold for missing values.
        - process_miss_zone_mode (str): Mode for handling missing values in zone columns, 'drop' or 'fill'.

        Returns:
        - Union[pd.DataFrame, None]: DataFrame with extracted features or None if no data is available.
        """
        df_processed = ExtractFeaturesByZone.process_missing_values_for_zone(
            df=df,
            zone_columns=zone_columns,
            missing_value_threshold=missing_value_threshold,
            process_miss_zone_mode=process_miss_zone_mode
        )

        if df_processed.shape[0] == 0:
            return None

        zone_with_features = ExtractFeaturesByZone.calculate_zone_stats(df_processed, grpby_list, zone_columns)

        features_columns = grpby_list + ['WAFER_ID', 'PARAMETRIC_NAME', 'MAX_VAL', 'MIN_VAL', 'MEDIAN', 'AVERAGE',
                                         'STD_DEV', 'PERCENTILE_25', 'PERCENTILE_75']

        zone_with_features_select = zone_with_features[features_columns]

        return zone_with_features_select

    @staticmethod
    def extract_features_from_info(df: pd.DataFrame,
                                   info: Dict[str, Dict[str, List[str]]],
                                   grpby_list: List[str]) -> pd.DataFrame:
        all_features = []

        for label, zones in info.items():
            for zone, site_columns in zones.items():
                zone_features = ExtractFeaturesByZone.extract_features_by_zone(
                    df=df,
                    grpby_list=grpby_list,
                    zone_columns=site_columns,
                    missing_value_threshold=0.7,
                    process_miss_zone_mode='drop'
                )

                if zone_features is not None:
                    zone_features['label'] = label  # 动态设置标签
                    all_features.append(zone_features)

        if all_features:
            return pd.concat(all_features, axis=0).reset_index(drop=True)
        else:
            return None


class FitInlineModelByZone:
    def __init__(self,
                 df: pyspark.sql.dataframe,
                 grpby_list: list[str],
                 inline_info: Dict[str, Dict[str, List[str]]],
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
        self.inline_info = inline_info
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
        df = FitInlineModelByZone.process_missing_values(df, columns_to_process, missing_value_threshold)
        index_list = ['WAFER_ID', 'label']
        columns_list = grpby_list + ['PARAMETRIC_NAME']
        values_list = df.columns.difference(['WAFER_ID', 'PARAMETRIC_NAME', 'label'] + grpby_list)
        pivot_result = df.pivot_table(index=index_list,
                                      columns=columns_list,
                                      values=values_list)
        pivot_result.columns = pivot_result.columns.map('#'.join)
        pivot_result = FitInlineModelByZone.process_missing_values(pivot_result, pivot_result.columns,
                                                                   missing_value_threshold)
        pivot_result = pivot_result.reset_index(drop=False)
        # Remove completely identical columns
        for column in pivot_result.columns.difference(index_list):
            if pivot_result[column].nunique() == 1:
                pivot_result = pivot_result.drop(column, axis=1)
        return pivot_result

    @staticmethod
    def construct_features_when_not_satisfied(x_train) -> pd.DataFrame:
        x_len = len(x_train.columns)
        res_top_select = pd.DataFrame({"features": x_train.columns,
                                       "importance": [-1.0] * x_len,
                                       "algorithm_satisfied": ['FALSE'] * x_len,
                                       "x_train_shape": [str(x_train.shape)] * x_len})
        return res_top_select

    @staticmethod
    def construct_features_when_satisfy_pca(x_train) -> pd.DataFrame:
        # 得到PCA算法结果res_top_select
        n_components = min(min(x_train.shape) - 2, 20)
        model = pca(n_components=n_components, verbose=None, random_state=2024)
        results = model.fit_transform(x_train)
        res_top = results['topfeat']
        res_top_select = res_top[res_top['type'] == 'best'][['feature', 'loading']]
        res_top_select['importance'] = abs(res_top_select['loading'])
        res_top_select = res_top_select.rename(columns={'feature': 'features'}).drop("loading",
                                                                                     axis=1).drop_duplicates()
        res_top_select['x_train_shape'] = str(x_train.shape)
        res_top_select['algorithm_satisfied'] = 'TRUE'
        return res_top_select

    @staticmethod
    def fit_pca_model(df: pyspark.sql.dataframe, grpby_list, inline_info,
                      columns_to_process, missing_value_threshold) -> pyspark.sql.dataframe:
        schema_all = StructType([StructField("features", StringType(), True),
                                 StructField("importance", FloatType(), True),
                                 StructField("algorithm_satisfied", StringType(), True),
                                 StructField("x_train_shape", StringType(), True)])

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run):
            side_with_features_all = ExtractFeaturesByZone.extract_features_from_info(df=df_run,
                                                                                      info=inline_info,
                                                                                      grpby_list=grpby_list)
            # 如果df_run中的good_site_columns和bad_site_columns的每列缺失比例都大于70%, 则无法提取特征, side_with_features_all就是None
            if side_with_features_all is None:
                grpby_values = [df_run[item].iloc[0] for item in grpby_list]
                features_value = f"STATS#{'#'.join(map(str, grpby_values))}#PARAM"
                res_top_select = pd.DataFrame({"features": features_value,
                                               "importance": -2.0,
                                               "algorithm_satisfied": 'FALSE',
                                               "x_train_shape": str(0)}, index=[0])
                return res_top_select

            pivot_result = FitInlineModelByZone.get_pivot_table(df=side_with_features_all,
                                                                grpby_list=grpby_list,
                                                                columns_to_process=columns_to_process,
                                                                missing_value_threshold=missing_value_threshold)
            x_train = pivot_result[pivot_result.columns.difference(['WAFER_ID', 'label']).tolist()]

            if min(x_train.shape) > 2:
                res_top_select = FitInlineModelByZone.construct_features_when_satisfy_pca(x_train=x_train)
                return res_top_select
            else:
                res_top_select = FitInlineModelByZone.construct_features_when_not_satisfied(x_train=x_train)
                return res_top_select

        return df.groupby(grpby_list).apply(get_model_result)

    # 暂时没用到get_pipe_params
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

    def run(self):
        if self.model == 'pca':
            res = self.fit_pca_model(df=self.df,
                                     grpby_list=self.grpby_list,
                                     inline_info=self.inline_info,
                                     columns_to_process=self.columns_to_process,
                                     missing_value_threshold=self.missing_value_threshold)
        else:
            res = None
        return res


class SplitInlineModelResults:
    def __init__(self, df: pyspark.sql.dataframe, grpby_list: List[str], request_id: str,
                 add_parametric_stats_df: pyspark.sql.dataframe) -> pyspark.sql.dataframe:
        self.df = df
        self.grpby_list = grpby_list
        self.request_id = request_id
        self.add_parametric_stats_df = add_parametric_stats_df

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
    def split_calculate_features(df: pyspark.sql.dataframe, grpby_list: List[str], by: str) -> pyspark.sql.dataframe:
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
    def add_certain_column(df: pyspark.sql.dataframe, request_id: str, total_algorithm_satisfied=False):
        if not total_algorithm_satisfied:
            total_importance = df.select(spark_sum("importance")).collect()[0][0]
            df = df.withColumn("WEIGHT", col("importance") / total_importance)
            df = df.drop('importance')
            df = df.orderBy(col("WEIGHT").desc())

        df = (df.withColumn("GOOD_NUM", df["GOOD_NUM"].cast(FloatType()))
              .withColumn("BAD_NUM", lit(None).cast(FloatType()))
              .withColumn("REQUEST_ID", lit(request_id))
              .withColumn("WEIGHT_PERCENT", col("WEIGHT") * 100))

        df = df.withColumn("INDEX_NO", monotonically_increasing_id() + 1)
        info_list = ['PRODUCT_ID', 'OPE_NO', 'PRODG1']
        for column in info_list:
            if column not in df.columns:
                df = df.withColumn(column, lit(None).cast(StringType()))
        return df

    def run(self):
        # BySite算法结果importance为-2的代表原始数据缺失比例大于70%无法提取特征,
        # importance为-1的代表不满足PCA算法要求(该组合下至少要有3个以上的不同sensor)
        # res过滤掉这些组合
        df = self.df.filter("algorithm_satisfied == TRUE")
        m = df.count()
        if m == 0:
            final_res = self.add_parametric_stats_df.withColumn('WEIGHT', lit(0))
            final_res = self.add_certain_column(df=final_res, request_id=self.request_id,
                                                total_algorithm_satisfied=True)
            return final_res

        df = df.withColumn('temp', lit(0))
        split_res = self.split_calculate_features(df=df, grpby_list=self.grpby_list, by='temp')
        split_res = split_res.drop('algorithm_satisfied')

        res_all = split_res.join(self.add_parametric_stats_df, on=self.grpby_list + ['PARAMETRIC_NAME'], how='left')
        missing_rows = self.add_parametric_stats_df.join(res_all, on=self.grpby_list + ['PARAMETRIC_NAME'],
                                                         how='left_anti')
        missing_rows = missing_rows.withColumn('importance', lit(0))
        res_all_update_missing_features = res_all.unionByName(missing_rows, allowMissingColumns=True)
        final_res = self.add_certain_column(df=res_all_update_missing_features, request_id=self.request_id)
        return final_res


class ExertInlineByZone:
    @staticmethod
    def get_site_vals(inline_info):
        all_columns = set()
        for label, zones in inline_info.items():
            for zone, site_columns in zones.items():
                if not site_columns:
                    msg = f"'{zone}'为空, 请指定对应的site"
                    raise RCABaseException(msg)

                if any(col_ in all_columns for col_ in site_columns):
                    msg = f"'{zone}'中的site在其他zone下有重复定义, 请检查."
                    raise RCABaseException(msg)
                all_columns.update(site_columns)
        return list(all_columns)

    @staticmethod
    def fit_by_zone_model(df: pyspark.sql.dataframe,
                          request_id: str,
                          merge_operno_list: List[Dict[str, List[str]]],
                          merge_prodg1_list: List[Dict[str, List[str]]],
                          merge_product_list: List[Dict[str, List[str]]],
                          mode_info: Dict,
                          columns_list=None,
                          key_words=None,
                          convert_to_numeric_list=None,
                          grpby_list=None,
                          certain_column=None,
                          model='pca'):

        inline_info = mode_info.get("INLINE")
        length = len(inline_info)
        if length < 2:
            msg = "请至少指定两类不同的Zone"
            raise RCABaseException(msg)

        site_columns = ExertInlineByZone.get_site_vals(inline_info)

        if grpby_list is None or len(grpby_list) == 0:
            grpby_list = ['OPE_NO']

        if columns_list is None:
            columns_list = grpby_list + ['WAFER_ID', 'PARAMETRIC_NAME', 'SITE_COUNT', 'AVERAGE'] + site_columns

        if key_words is None:
            key_words = ['CXS', 'CYS', 'FDS']

        if convert_to_numeric_list is None:
            convert_to_numeric_list = ['SITE_COUNT', 'AVERAGE'] + site_columns

        if certain_column is None:
            certain_column = 'PARAMETRIC_NAME'

        grps_all, add_parametric_stats_df, df_preprocess = DataPreprocessorForInline(df=df,
                                                                                     grpby_list=grpby_list,
                                                                                     columns_list=columns_list,
                                                                                     certain_column=certain_column,
                                                                                     key_words=key_words,
                                                                                     convert_to_numeric_list=convert_to_numeric_list,
                                                                                     merge_operno_list=merge_operno_list,
                                                                                     merge_prodg1_list=merge_prodg1_list,
                                                                                     merge_product_list=merge_product_list).run()
        print(f"按照{'+'.join(grpby_list)}分组后的数据, 一共有{grps_all.count()}种不同的分组.")

        if df_preprocess.isEmpty():
            msg = '数据库中暂无数据.'
            raise RCABaseException(msg)

        res = FitInlineModelByZone(df=df_preprocess,
                                   grpby_list=grpby_list,
                                   inline_info=inline_info,
                                   columns_to_process=['AVERAGE', 'MAX_VAL', 'MEDIAN', 'MIN_VAL', 'STD_DEV',
                                                       'PERCENTILE_25', 'PERCENTILE_75'],
                                   missing_value_threshold=0.7,
                                   model=model).run()

        final_res = SplitInlineModelResults(df=res, grpby_list=grpby_list, request_id=request_id,
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
    # df_pandas = df_pandas[
    #     df_pandas['OPE_NO'].isin(['1V.EQW10', '1V.PQW10', '1F.FQE10', '1C.CDG10', '1U.EQW10', '1U.PQW10'])]
    df_pandas = df_pandas[
        df_pandas['OPE_NO'].isin(
            ['1V.EQW10', '1V.PQW10', '1F.FQE10', '1C.CDG10', '1U.EQW10', '1U.PQW10', '1C.PQX10', '1V.PQX10'])]
    # df_params = df_pandas.groupby(["OPE_NO", "PRODUCT_ID"])['PARAMETRIC_NAME'].nunique()
    # print("df_params:", sum(df_params))
    # print(df_params)
    # df_params1 = df_pandas.groupby(["OPE_NO", "PRODUCT_ID"])['PARAMETRIC_NAME'].value_counts()
    # df_params1.to_csv("df_params1.csv")
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
                                        "goodSite": ["SITE1_VAL", "SITE3_VAL", "SITE5_VAL", "SITE7_VAL", "SITE9_VAL"],
                                        "badSite": ["SITE2_VAL", "SITE4_VAL", "SITE6_VAL", "SITE8_VAL"],
                                        # "mergeOperno": []
                                        "mergeOperno": [{"1F.FQE10,1C.CDG10": ["1F.FQE10", "1C.CDG10"]},
                                                        {"1U.EQW10_1U.PQW10": ["1U.EQW10", "1U.PQW10"]}],

                                        "mode_info": {
                                            "inline": {
                                                "label1": {"zone1": ["SITE5_VAL", "SITE7_VAL"],
                                                           "zone2": ["SITE1_VAL", "SITE3_VAL"],
                                                           "zob": ["SITE11_VAL"]},

                                                "label2": {"zone3": ["SITE8_VAL", "SITE2_VAL"],
                                                           "zone4": ["SITE4_VAL", "SITE9_VAL"]},

                                                "norm": {"zone5": ["SITE18_VAL", "SITE12_VAL"],
                                                         "zone46": ["SITE14_VAL"]}
                                            }
                                        }
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

    mode_info_ = parse_dict.get('mode_info') if parse_dict.get('mode_info') else None
    print(type(mode_info_))
    print(mode_info_)

    # grpby_list_ = ['OPE_NO']
    grpby_list_ = ['OPE_NO', 'PRODUCT_ID']

    from datetime import datetime

    time1 = datetime.now()
    final_res_ = ExertInlineByZone.fit_by_zone_model(df=df_spark,
                                                     request_id=request_id_,
                                                     grpby_list=grpby_list_,
                                                     merge_operno_list=merge_operno,
                                                     merge_prodg1_list=merge_prodg1,
                                                     merge_product_list=merge_product,
                                                     mode_info=mode_info_)
    time2 = datetime.now()
    final_res_.show(50)
    # final_res_pandas = final_res_.toPandas()
    # final_res_pandas.to_csv("final_res_pandas_by_site.csv")
    print(f"算法结果一共有{final_res_.count()}条")
    print("算法结果写回数据库消耗的时间是：", time2 - time1)
