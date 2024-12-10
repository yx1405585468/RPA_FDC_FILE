import json
from typing import List, Dict, Union

import pyspark
import pandas as pd
import pyspark.pandas as ps
from pca import pca
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.sql.functions import pandas_udf, PandasUDFType, lit, col, countDistinct, when


class DataPreprocessorForInline:
    def __init__(self,
                 df: pyspark.sql.dataframe,
                 columns_list: list[str],
                 certain_column: str,
                 key_words: list[str],
                 convert_to_numeric_list: list[str],
                 merge_operno_list: List[Dict[str, List[str]]]):
        self.df = df
        self.columns_list = columns_list
        self.certain_column = certain_column
        self.key_words = key_words
        self.convert_to_numeric_list = convert_to_numeric_list
        self.merge_operno_list = merge_operno_list

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

    @staticmethod
    def integrate_columns(df, merge_operno_list: List[Dict[str, List[str]]]) -> pyspark.sql.dataframe:
        """
        Integrate columns in the DataFrame based on the provided list.

        :param df: The input DataFrame.
        :param merge_operno_list: A list of dictionaries where each dictionary contains values to be merged.
               Example: [{'2F.CDS10_XX.TDS01': ['2F.CDS10', 'XX.TDS01']},
                         {'2F.CDS20_XX.CDS20': ['2F.CDS20', 'XX.CDS20']}]
        :return: DataFrame with 'OPER_NO' and other specified columns integrated according to the merge rules.
        """
        # split using comma
        splitter_comma = ","
        if merge_operno_list is not None and len(merge_operno_list) > 0:
            # Extract values from each dictionary in merge_operno_list and create a list
            values_to_replace = [list(rule.values())[0] for rule in merge_operno_list]
            # Concatenate values from each dictionary
            merged_values = [splitter_comma.join(list(rule.values())[0]) for rule in merge_operno_list]

            # Replace values in 'OPER_NO' column based on the rules defined in merge_operno_list
            for values, replacement_value in zip(values_to_replace, merged_values):
                df = df.withColumn("OPE_NO", when(col("OPE_NO").isin(values), replacement_value).otherwise(col("OPE_NO")))
        return df

    def run(self):
        df_select = self.select_columns(df=self.df)
        df_esd = self.exclude_some_data(df=df_select)
        df_pp = self.pre_process(df=df_esd)
        df_integrate = self.integrate_columns(df=df_pp, merge_operno_list=self.merge_operno_list)
        return df_integrate


class GetTrainDataForInline:
    def __init__(self, df: pyspark.sql.dataframe, grpby_list: list[str]):
        """
        Initialize the GetTrainDataForInline class.

        Parameters:
        - df (pyspark.sql.dataframe): Input DataFrame.
        - grpby_list (list): List of grouping columns, inline data should be ["OPE_NO"]

        This class is designed to perform commonality analysis and retrieve training data based on the
        condition "bad_num > 1 AND wafer_count > 2" in each grpby_list, i.e. each OPE_NO for inline data.
        """
        self.df_run = df
        self.grpby_list = grpby_list

    def commonality_analysis(self):
        grps = (self.df_run.groupBy(self.grpby_list)
                .agg(countDistinct('WAFER_ID').alias('wafer_count'),
                     countDistinct('WAFER_ID', when(self.df_run['label'] == 0, 1)).alias('good_num'),
                     countDistinct('WAFER_ID', when(self.df_run['label'] == 1, 1)).alias('bad_num'))
                .na.fill(0)
                .orderBy(['bad_num', 'good_num'], ascending=False))
        if grps.count() == 1:
            return grps
        else:
            grps = grps.filter("bad_num > 1 AND wafer_count > 2")
            return grps

    def get_data_list(self, common_res):
        data_list = common_res.select(self.grpby_list).collect()
        data_dict_list = [row.asDict() for row in data_list]
        return data_dict_list

    def get_train_data(self, data_dict_list):
        oper = data_dict_list[0]['OPE_NO']
        df_s = self.df_run.filter("OPE_NO == '{}'".format(oper))
        for i in range(1, len(data_dict_list)):
            oper = data_dict_list[i]['OPE_NO']
            df_m = self.df_run.filter("OPE_NO == '{}'".format(oper))
            df_s = df_s.union(df_m)
        return df_s

    def run(self):
        common_res = self.commonality_analysis()
        data_dict_list = self.get_data_list(common_res)
        train_data = self.get_train_data(data_dict_list)
        return train_data


class FitInlineModelByWafer:
    def __init__(self,
                 df: pyspark.sql.dataframe,
                 by: list[str],
                 columns_to_process: list[str],
                 missing_value_threshold: float,
                 model: str = 'pca'):
        """
        Initialize the FitInlineModelByWafer object.

        Parameters:
        - df: pyspark.sql.dataframe, the input data
        - by: list[str], the grouping variable, inline data should be ["OPE_NO"]
        - columns_to_process: List of str, columns to process in missing value functions
        - missing_value_threshold: Union[int, float], threshold for missing values
        - model: str, default is 'pca', other options include 'rf' for random forest
        """
        self.df = df
        self.by = by
        self.model = model
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
        df_specific = FitInlineModelByWafer.process_missing_values(df, columns_to_process, missing_value_threshold)
        index_list = ['WAFER_ID', 'label']
        values_list = df_specific.columns.difference(
            ['WAFER_ID', 'OPE_NO', 'INLINE_PARAMETER_ID', 'SITE_COUNT', 'label'])
        pivot_result = df_specific.pivot_table(index=index_list,
                                               columns=['OPE_NO', 'INLINE_PARAMETER_ID'],
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

    def fit_pca_model(self):
        schema_all = StructType([StructField("features", StringType(), True),
                                 StructField("importance", FloatType(), True)])
        columns_to_process_ = self.columns_to_process
        missing_value_threshold_ = self.missing_value_threshold

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run):
            pivot_result = FitInlineModelByWafer.get_pivot_table(df=df_run,
                                                                 columns_to_process=columns_to_process_,
                                                                 missing_value_threshold=missing_value_threshold_)
            # 定义自变量
            x_train = pivot_result[pivot_result.columns.difference(['WAFER_ID', 'label']).tolist()]
            if x_train.shape[1] > 1:
                n_components = min(min(x_train.shape) - 2, 20)
                model = pca(n_components=n_components, verbose=None)
                results = model.fit_transform(x_train)
                res_top = results['topfeat']
                res_top_select = res_top[res_top['type'] == 'best'][['feature', 'loading']]
                res_top_select['importance'] = abs(res_top_select['loading'])
                res_top_select = res_top_select.rename(columns={'feature': 'features'}).drop("loading",
                                                                                             axis=1).drop_duplicates()
                return res_top_select
            else:
                res_top_select = pd.DataFrame({"features": "STATS#OPE#PARAM", "importance": -100}, index=[0])
                return res_top_select

        return self.df.groupby(self.by).apply(get_model_result)

    def fit_rf_model(self):
        schema_all = StructType([StructField("features", StringType(), True),
                                 StructField("importance", FloatType(), True)])
        columns_to_process_ = self.columns_to_process
        missing_value_threshold_ = self.missing_value_threshold

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run):
            pivot_result = FitInlineModelByWafer.get_pivot_table(df=df_run,
                                                                 columns_to_process=columns_to_process_,
                                                                 missing_value_threshold=missing_value_threshold_)
            x_train = pivot_result[pivot_result.columns.difference(['WAFER_ID', 'label']).tolist()]
            y_train = pivot_result[['label']]

            if x_train.shape[1] > 1 and y_train['label'].nunique() > 1:
                pipe = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
                    ('scaler', StandardScaler()),
                    ('model', RandomForestClassifier(random_state=2024))])
                param_grid = {'model__n_estimators': [*range(10, 60, 10)],
                              'model__max_depth': [*range(5, 50, 10)],
                              'model__min_samples_split': [2, 5],
                              'model__min_samples_leaf': [1, 3]}
                grid = GridSearchCV(estimator=pipe, scoring='roc_auc', param_grid=param_grid, cv=2, n_jobs=-1)
                grid.fit(x_train.values, y_train.values.ravel())
                roc_auc_score_ = grid.best_score_
                if roc_auc_score_ >= 0.6:
                    small_importance_res = pd.DataFrame({'features': x_train.columns,
                                                         'importance': grid.best_estimator_.steps[2][
                                                             1].feature_importances_})
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
            split_table_grpby = split_table.groupby(['OPE_NO', 'INLINE_PARAMETER_ID'])['importance'].sum().reset_index(
                drop=False)
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


class ExertInlineByWafer:
    @staticmethod
    def fit_by_wafer_model(df: pyspark.sql.dataframe,
                           request_id: str,
                           merge_operno_list: List[Dict[str, List[str]]],
                           columns_list=None,
                           key_words=None,
                           convert_to_numeric_list=None,
                           grpby_list=None,
                           certain_column=None) -> Union[str, pyspark.sql.dataframe.DataFrame]:
        if columns_list is None:
            columns_list = ['WAFER_ID', 'OPE_NO', 'INLINE_PARAMETER_ID', 'AVERAGE', 'MAX_VAL', 'MEDIAN', 'MIN_VAL',
                            'STD_DEV', 'PERCENTILE_25', 'PERCENTILE_75', 'SITE_COUNT', 'label']
        if key_words is None:
            key_words = ['CXS', 'CYS', 'FDS']
        if convert_to_numeric_list is None:
            convert_to_numeric_list = ['AVERAGE', 'MAX_VAL', 'MEDIAN', 'MIN_VAL', 'STD_DEV', 'PERCENTILE_25',
                                       'PERCENTILE_75', 'SITE_COUNT']
        if grpby_list is None:
            grpby_list = ['OPE_NO']
        if certain_column is None:
            certain_column = 'INLINE_PARAMETER_ID'

        df_preprocess = DataPreprocessorForInline(df=df,
                                                  columns_list=columns_list,
                                                  certain_column=certain_column,
                                                  key_words=key_words,
                                                  convert_to_numeric_list=convert_to_numeric_list,
                                                  merge_operno_list=merge_operno_list).run()
        df_preprocess.show()
        if df_preprocess.isEmpty():
            msg = 'No data of this type in the database!'
            return msg

        df_train = GetTrainDataForInline(df=df_preprocess, grpby_list=grpby_list).run()
        if df_train.isEmpty():
            msg = 'Get train data Exception!'
            return msg

        res = FitInlineModelByWafer(df=df_train,
                                    by=grpby_list,
                                    model='pca',
                                    columns_to_process=['AVERAGE', 'MAX_VAL', 'MEDIAN', 'MIN_VAL', 'STD_DEV',
                                                        'PERCENTILE_25', 'PERCENTILE_75'],
                                    missing_value_threshold=0.6).run()
        if res.isEmpty():
            msg = 'No difference in this data. The output of the algorithm is 0.'
            return msg

        final_res = SplitInlineModelResults(df=res, request_id=request_id).run()
        if final_res.isEmpty():
            msg = 'Temporary exception in adding columns to algorithm results'
            return msg
        else:
            return final_res


if __name__ == "__main__":
    import os
    from pyspark.sql import SparkSession

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
    # from pyspark.sql import SparkSession
    # findspark.init()
    # spark = SparkSession \
    #     .builder \
    #     .appName("ywj") \
    #     .config('spark.sql.session.timeZone', 'Asia/Shanghai') \
    #     .master("local[*]") \
    #     .getOrCreate()

    df_pandas = pd.read_csv(
        "D:/Jupyterfiles/晶合MVAFDC_general开发/MVAanlysisDevelop/inline_algorithm/inline_case5_label.csv")
    df_pandas = df_pandas[df_pandas['OPE_NO'].isin(['1V.EQW10', '1V.PQW10', '1F.FQE10', '1C.CDG10', '1U.EQW10', '1U.PQW10'])]
    df_spark = ps.from_pandas(df_pandas).to_spark()
    num_rows = df_spark.count()
    num_columns = len(df_spark.columns)
    print(f"df_spark shape: ({num_rows}, {num_columns})")

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
    print(merge_operno)

    final_res_ = ExertInlineByWafer.fit_by_wafer_model(df=df_spark, request_id=request_id_, merge_operno_list=merge_operno)
    if isinstance(final_res_, str):
        print(final_res_)
    else:
        num_rows = final_res_.count()
        num_columns = len(final_res_.columns)
        print(f"final_res shape: ({num_rows}, {num_columns})")
        final_res_.show()
