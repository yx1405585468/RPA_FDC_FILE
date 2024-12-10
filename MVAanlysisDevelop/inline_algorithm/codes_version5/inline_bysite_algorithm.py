import pyspark
import pandas as pd
import pyspark.pandas as ps
from pca import pca
from typing import Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.sql.functions import pandas_udf, PandasUDFType, lit, col


class DataPreprocessorForInline:
    def __init__(self,
                 df: pyspark.sql.dataframe,
                 columns_list: list[str],
                 certain_column: str,
                 key_words: list[str],
                 convert_to_numeric_list: list[str]):
        self.df = df
        self.columns_list = columns_list
        self.certain_column = certain_column
        self.key_words = key_words
        self.convert_to_numeric_list = convert_to_numeric_list

    def select_columns(self, df):
        return df.select(self.columns_list)

    def exclude_some_data(self, df):
        key_words_str = '|'.join(self.key_words)
        df_filtered = df.filter(~col(self.certain_column).rlike(key_words_str))
        return df_filtered

    def pre_process(self, df):
        for column in self.convert_to_numeric_list:
            df = df.withColumn(column, col(column).cast('double'))
        if 'SITE_COUNT' in self.convert_to_numeric_list:
            self.convert_to_numeric_list.remove('SITE_COUNT')
        df = df.dropna(subset=self.convert_to_numeric_list, how='all')
        return df

    def run(self):
        df_select = self.select_columns(df=self.df)
        df_esd = self.exclude_some_data(df=df_select)
        df_pp = self.pre_process(df=df_esd)
        return df_pp


class ExtractFeaturesBySite:
    @staticmethod
    def process_missing_values_for_site(df: pd.DataFrame,
                                        good_site_columns: list[str],
                                        bad_site_columns: list[str],
                                        missing_value_threshold: Union[int, float] = 0.6,
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
    def calculate_site_stats(df: pd.DataFrame, site_columns: list[str], good_or_bad: str) -> pd.DataFrame:
        assert good_or_bad in ['good', 'bad'], "Label could only be 'good' or 'bad'"
        selected_df = df[['WAFER_ID', 'OPE_NO', 'INLINE_PARAMETER_ID'] + site_columns].reset_index(drop=True)
        # Perform statistical calculations for each row
        side_features = selected_df.apply(lambda row: ExtractFeaturesBySite.calculate_statistics(row[site_columns]), axis=1)
        side_features = side_features.fillna(0)
        df_with_features = pd.concat([selected_df, side_features], axis=1)
        if good_or_bad == 'good':
            df_with_features['label'] = 0
        else:
            df_with_features['label'] = 1
        return df_with_features

    @staticmethod
    def extract_features_by_site(df: pd.DataFrame,
                                 good_site_columns: list[str],
                                 bad_site_columns: list[str],
                                 missing_value_threshold: Union[int, float] = 0.6,
                                 process_miss_site_mode: str = 'drop') -> Union[pd.DataFrame, None]:
        """
        Extracts features from a DataFrame based on good and bad site columns.
        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - good_site_columns (list): List of columns representing good sites.
        - bad_site_columns (list): List of columns representing bad sites.
        - missing_value_threshold (Union[int, float]): Threshold for missing values.
        - process_miss_site_mode (str): Mode for handling missing values in site columns, e.g. drop or fill
        Returns:
        - Union[pd.DataFrame, None]: DataFrame with extracted features or None if no data is available.
        """
        df_pandas_specific_oper = ExtractFeaturesBySite.process_missing_values_for_site(df=df,
                                                                                        good_site_columns=good_site_columns,
                                                                                        bad_site_columns=bad_site_columns,
                                                                                        missing_value_threshold=missing_value_threshold,
                                                                                        process_miss_site_mode=process_miss_site_mode)
        if df_pandas_specific_oper.shape[0] != 0:
            side_with_features1 = ExtractFeaturesBySite.calculate_site_stats(df_pandas_specific_oper, good_site_columns,
                                                                             good_or_bad='good')
            side_with_features2 = ExtractFeaturesBySite.calculate_site_stats(df_pandas_specific_oper, bad_site_columns,
                                                                             good_or_bad='bad')
            side_with_features1_select = side_with_features1[
                ['WAFER_ID', 'OPE_NO', 'INLINE_PARAMETER_ID', 'MAX_VAL', 'MIN_VAL', 'MEDIAN',
                 'AVERAGE', 'STD_DEV', 'PERCENTILE_25', 'PERCENTILE_75', 'label']]
            side_with_features2_select = side_with_features2[
                ['WAFER_ID', 'OPE_NO', 'INLINE_PARAMETER_ID', 'MAX_VAL', 'MIN_VAL', 'MEDIAN',
                 'AVERAGE', 'STD_DEV', 'PERCENTILE_25', 'PERCENTILE_75', 'label']]
            side_with_features_all = pd.concat([side_with_features1_select, side_with_features2_select], axis=0)
            return side_with_features_all


class FitInlineModelBySite:
    def __init__(self,
                 df: pyspark.sql.dataframe,
                 by: list[str],
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
        - by: list[str], the grouping variable, inline data should be ["OPE_NO"]
        - good_site_columns: List of str, column names for good sites
        - bad_site_columns: List of str, column names for bad sites
        - process_miss_site_mode: str, mode for handling missing values in site data, e.g. drop or fill
        - columns_to_process: List of str, columns to process in missing value functions
        - missing_value_threshold: Union[int, float], threshold for missing values
        - model: str, default is 'pca', other options include 'rf' for random forest
        """
        self.df = df
        self.by = by
        self.model = model
        self.good_site_columns = good_site_columns
        self.bad_site_columns = bad_site_columns
        self.process_miss_site_mode = process_miss_site_mode
        self.columns_to_process = columns_to_process
        self.missing_value_threshold = missing_value_threshold

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
    def get_pivot_table(df, columns_to_process, missing_value_threshold):
        df = FitInlineModelBySite.process_missing_values(df, columns_to_process, missing_value_threshold)
        index_list = ['WAFER_ID', 'label']
        values_list = df.columns.difference(['WAFER_ID', 'OPE_NO', 'INLINE_PARAMETER_ID', 'label'])
        pivot_result = df.pivot_table(index=index_list,
                                      columns=['OPE_NO', 'INLINE_PARAMETER_ID'],
                                      values=values_list)
        pivot_result.columns = pivot_result.columns.map('#'.join)
        pivot_result = FitInlineModelBySite.process_missing_values(pivot_result, pivot_result.columns, missing_value_threshold)
        pivot_result = pivot_result.reset_index(drop=False)
        # Remove completely identical columns
        for column in pivot_result.columns.difference(index_list):
            if pivot_result[column].nunique() == 1:
                pivot_result = pivot_result.drop(column, axis=1)
        return pivot_result

    def fit_pca_model(self):
        schema_all = StructType([StructField("features", StringType(), True),
                                 StructField("importance", FloatType(), True)])
        good_site_columns = self.good_site_columns
        bad_site_columns = self.bad_site_columns
        missing_value_threshold = self.missing_value_threshold
        process_miss_site_mode = self.process_miss_site_mode
        columns_to_process = self.columns_to_process

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run):
            side_with_features_all = ExtractFeaturesBySite.extract_features_by_site(df=df_run,
                                                                                    good_site_columns=good_site_columns,
                                                                                    bad_site_columns=bad_site_columns,
                                                                                    missing_value_threshold=missing_value_threshold,
                                                                                    process_miss_site_mode=process_miss_site_mode)
            if side_with_features_all is None:
                return pd.DataFrame({"features": "STATS#OPE#PARAM", "importance": -100}, index=[0])

            pivot_result = FitInlineModelBySite.get_pivot_table(df=side_with_features_all,
                                                                columns_to_process=columns_to_process,
                                                                missing_value_threshold=missing_value_threshold)
            x_train = pivot_result[pivot_result.columns.difference(['WAFER_ID', 'label']).tolist()]

            if x_train.shape[1] > 1:
                n_components = min(min(x_train.shape) - 2, 20)
                model = pca(n_components=n_components, verbose=None)
                results = model.fit_transform(x_train)
                res_top = results['topfeat']
                res_top_select = res_top[res_top['type'] == 'best'][['feature', 'loading']]
                res_top_select['importance'] = abs(res_top_select['loading'])
                res_top_select = res_top_select.rename(columns={'feature': 'features'}).drop("loading", axis=1).drop_duplicates()
                return res_top_select
            else:
                res_top_select = pd.DataFrame({"features": "STATS#OPE#PARAM", "importance": -101}, index=[0])
                return res_top_select
        return self.df.groupby(self.by).apply(get_model_result)

    def fit_rf_model(self):
        schema_all = StructType([StructField("features", StringType(), True),
                                 StructField("importance", FloatType(), True)])
        good_site_columns = self.good_site_columns
        bad_site_columns = self.bad_site_columns
        missing_value_threshold = self.missing_value_threshold
        process_miss_site_mode = self.process_miss_site_mode
        columns_to_process = self.columns_to_process

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run):
            side_with_features_all = ExtractFeaturesBySite.extract_features_by_site(df=df_run,
                                                                                    good_site_columns=good_site_columns,
                                                                                    bad_site_columns=bad_site_columns,
                                                                                    missing_value_threshold=missing_value_threshold,
                                                                                    process_miss_site_mode=process_miss_site_mode)
            if side_with_features_all is None:
                return pd.DataFrame({"features": "STATS#OPE#PARAM", "importance": -100}, index=[0])

            pivot_result = FitInlineModelBySite.get_pivot_table(df=side_with_features_all,
                                                                columns_to_process=columns_to_process,
                                                                missing_value_threshold=missing_value_threshold)
            x_train = pivot_result[pivot_result.columns.difference(['WAFER_ID', 'label']).tolist()]
            y_train = pivot_result[['label']]
            if min(x_train.shape) > 4 and y_train['label'].nunique() > 1:
                pipe = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
                    ('scaler', StandardScaler()),
                    ('model', RandomForestClassifier(random_state=2024))])
                param_grid = {'model__n_estimators': [*range(10, 60, 10)],
                              'model__max_depth': [*range(5, 50, 10)],
                              'model__min_samples_split': [2, 5],
                              'model__min_samples_leaf': [1, 3]}
                grid = GridSearchCV(estimator=pipe, scoring='roc_auc', param_grid=param_grid, cv=3, n_jobs=-1)
                grid.fit(x_train.values, y_train.values.ravel())
                roc_auc_score_ = grid.best_score_
                if roc_auc_score_ >= 0.6:
                    small_importance_res = pd.DataFrame({'features': x_train.columns,
                                                         'importance': grid.best_estimator_.steps[2][1].feature_importances_})
                    return small_importance_res
                else:
                    small_importance_res = pd.DataFrame({"features": "STATS#OPE#PARAM", "importance": -101}, index=[0])
                    return small_importance_res
            else:
                small_importance_res = pd.DataFrame({"features": "STATS#OPE#PARAM", "importance": -102}, index=[0])
                return small_importance_res
        return self.df.groupby(self.by).apply(get_model_result)

    def run(self):
        if self.model == 'pca':
            res = self.fit_pca_model()
        elif self.model == 'rf':
            res = self.fit_rf_model()
        else:
            res = None
        return res


class SplitInlineModelResults:
    def __init__(self, df: pyspark.sql.dataframe, request_id: str):
        self.df = df
        self.request_id = request_id

    @staticmethod
    def split_features(df: pd.DataFrame, index: int) -> str:
        return df['features'].apply(lambda x: x.split('#')[index])

    @staticmethod
    def get_split_features(df: pd.DataFrame) -> pd.DataFrame:
        df['STATISTIC_RESULT'] = SplitInlineModelResults.split_features(df, 0)
        df['OPE_NO'] = SplitInlineModelResults.split_features(df, 1)
        df['INLINE_PARAMETER_ID'] = SplitInlineModelResults.split_features(df, 2)
        df = df.drop(['features', 'STATISTIC_RESULT'], axis=1).reset_index(drop=True)
        return df

    @staticmethod
    def split_calculate_features(df: pyspark.sql.dataframe, by: str) -> pyspark.sql.dataframe:
        schema_all = StructType([StructField("OPE_NO", StringType(), True),
                                 StructField("INLINE_PARAMETER_ID", StringType(), True),
                                 StructField("importance", FloatType(), True)])

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run):
            split_table = SplitInlineModelResults.get_split_features(df_run)
            split_table_grpby = split_table.groupby(['OPE_NO', 'INLINE_PARAMETER_ID'])['importance'].sum().reset_index(drop=False)
            return split_table_grpby
        return df.groupby(by).apply(get_model_result)

    @staticmethod
    def add_certain_column(df: pyspark.sql.dataframe, by: str) -> pyspark.sql.dataframe:
        schema_all = StructType([StructField("OPER_NO", StringType(), True),
                                StructField("INLINE_PARAMETER_ID", StringType(), True),
                                StructField("AVG_SPEC_CHK_RESULT_COUNT", FloatType(), True),
                                StructField("weight", FloatType(), True),
                                StructField("weight_percent", FloatType(), True),
                                StructField("index_no", IntegerType(), True)])

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_result(final_res):
            # Calculate weights and normalize
            final_res['importance'] = final_res['importance'].astype(float)
            final_res = final_res.query("importance > 0")
            final_res['weight'] = final_res['importance'] / final_res['importance'].sum()
            final_res['weight_percent'] = final_res['weight'] * 100
            final_res = final_res.sort_values('weight', ascending=False)
            # add some columns like index_no, request_id
            final_res['index_no'] = [i + 1 for i in range(len(final_res))]
            final_res['AVG_SPEC_CHK_RESULT_COUNT'] = 0.0
            final_res = final_res.rename(columns={'OPE_NO': 'OPER_NO'})
            final_res = final_res.drop(['importance', 'temp'], axis=1)
            return final_res
        return df.groupby(by).apply(get_result)

    def run(self):
        df = self.df.withColumn('temp', lit(0))
        res = self.split_calculate_features(df=df, by='temp')
        res = res.withColumn('temp', lit(1))
        final_res = self.add_certain_column(df=res, by='temp')
        final_res = final_res.withColumn('request_id', lit(self.request_id))
        return final_res


if __name__ == "__main__":
    import os
    from pyspark.sql import SparkSession

    os.environ['PYSPARK_PYTHON'] = '/usr/local/python-3.9.13/bin/python3'
    spark = SparkSession.builder \
        .appName("pandas_udf") \
        .config('spark.sql.session.timeZone', 'Asia/Shanghai') \
        .config("spark.scheduler.mode", "FAIR") \
        .config('spark.driver.memory', '1024m') \
        .config('spark.driver.cores', '3') \
        .config('spark.executor.memory', '1024m') \
        .config('spark.executor.cores', '1') \
        .config('spark.cores.max', '2') \
        .config('spark.driver.host', '192.168.22.28') \
        .master("spark://192.168.12.47:7077,192.168.12.48:7077") \
        .getOrCreate()

    # 1. 读取数据
    df_pandas = pd.read_csv(
        "D:/Jupyterfiles/晶合MVAFDC_general开发/MVAanlysisDevelop/inline_algorithm/inline_case5_label.csv")
    df_spark = ps.from_pandas(df_pandas).to_spark()
    num_rows = df_spark.count()
    num_columns = len(df_spark.columns)
    print(f"df_spark shape: ({num_rows}, {num_columns})")

    # 2. 数据预处理
    dp = DataPreprocessorForInline(df=df_spark,
                                   columns_list=['WAFER_ID', 'OPE_NO', 'INLINE_PARAMETER_ID', 'SITE_COUNT', 'AVERAGE',
                                                 'SITE1_VAL', 'SITE2_VAL', 'SITE3_VAL', 'SITE4_VAL', 'SITE5_VAL',
                                                 'SITE6_VAL',
                                                 'SITE7_VAL', 'SITE8_VAL', 'SITE9_VAL', 'SITE10_VAL', 'SITE11_VAL',
                                                 'SITE12_VAL',
                                                 'SITE13_VAL', 'SITE14_VAL', 'SITE15_VAL', 'SITE16_VAL', 'SITE17_VAL'],
                                   certain_column='INLINE_PARAMETER_ID',
                                   key_words=['CXS', 'CYS', 'FDS'],
                                   convert_to_numeric_list=['SITE_COUNT', 'AVERAGE', 'SITE1_VAL', 'SITE2_VAL',
                                                            'SITE3_VAL',
                                                            'SITE4_VAL', 'SITE5_VAL', 'SITE6_VAL', 'SITE7_VAL',
                                                            'SITE8_VAL',
                                                            'SITE9_VAL', 'SITE10_VAL', 'SITE11_VAL', 'SITE12_VAL',
                                                            'SITE13_VAL',
                                                            'SITE14_VAL', 'SITE15_VAL', 'SITE16_VAL', 'SITE17_VAL'])
    df_pp_ = dp.run()
    num_rows = df_pp_.count()
    num_columns = len(df_pp_.columns)
    print(f"df_pp_ shape: ({num_rows}, {num_columns})")

    # 3. BySite特征提取, 拟合模型
    by_ = ['OPE_NO']
    model_ = 'pca'
    good_site_columns_ = ['SITE4_VAL', 'SITE8_VAL', 'SITE9_VAL', 'SITE12_VAL', 'SITE13_VAL']
    bad_site_columns_ = ['SITE2_VAL', 'SITE6_VAL', 'SITE7_VAL', 'SITE10_VAL', 'SITE11_VAL']
    process_miss_site_mode_ = 'drop'
    columns_to_process_ = ['AVERAGE', 'MAX_VAL', 'MEDIAN', 'MIN_VAL', 'STD_DEV', 'PERCENTILE_25', 'PERCENTILE_75']
    missing_value_threshold_ = 0.6

    BySiteModel = FitInlineModelBySite(df=df_pp_,
                                       by=by_,
                                       model=model_,
                                       good_site_columns=good_site_columns_,
                                       bad_site_columns=bad_site_columns_,
                                       process_miss_site_mode=process_miss_site_mode_,
                                       columns_to_process=columns_to_process_,
                                       missing_value_threshold=missing_value_threshold_)

    res_pca = BySiteModel.run()
    print(res_pca.count())
    res_pca.show()

    # res_p1 = res_pca.toPandas()
    # print(res_p1.query("importance < 0")['importance'].value_counts())
    # res_p1 = res_p1.query("importance > 0")
    # print(res_p1.sort_values('importance').reset_index(drop=True))

    # 4. 结果整理
    final_res_pca = SplitInlineModelResults(df=res_pca, request_id='855s').run()
    print(final_res_pca.count())
    final_res_pca.show()
