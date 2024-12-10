import pandas as pd
import pyspark.sql.dataframe
from pca import pca
from pyspark.sql.functions import max, countDistinct, when, lit, pandas_udf, PandasUDFType, monotonically_increasing_id, split
from typing import List, Dict
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, StructType, StructField, BooleanType
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import ClusterCentroids
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from src.exceptions.rca_base_exception import RCABaseException


class PreprocessForUvaData:
    def __init__(self,
                 df: pyspark.sql.dataframe,
                 grpby_list: list[str],
                 merge_operno_list: List[Dict[str, List[str]]],
                 merge_prodg1_list: List[Dict[str, List[str]]],
                 merge_product_list: List[Dict[str, List[str]]],
                 merge_eqp_list: List[Dict[str, List[str]]],
                 merge_chamber_list: List[Dict[str, List[str]]]):
        self.df = df
        self.grpby_list = grpby_list
        self.merge_operno_list = merge_operno_list
        self.merge_prodg1_list = merge_prodg1_list
        self.merge_product_list = merge_product_list
        self.merge_eqp_list = merge_eqp_list
        self.merge_chamber_list = merge_chamber_list

    @staticmethod
    def integrate_columns(df: pyspark.sql.dataframe,
                          merge_operno_list: List[Dict[str, List[str]]],
                          merge_prodg1_list: List[Dict[str, List[str]]],
                          merge_product_list: List[Dict[str, List[str]]],
                          merge_eqp_list: List[Dict[str, List[str]]],
                          merge_chamber_list: List[Dict[str, List[str]]]) -> pyspark.sql.dataframe:
        """
        Integrate columns in the DataFrame based on the provided list.

        :param df: The input DataFrame.
        :param merge_operno_list: A list of dictionaries where each dictionary contains values to be merged.
               Example: [{'2F.CDS10_XX.TDS01': ['2F.CDS10', 'XX.TDS01']},
                         {'2F.CDS20_XX.CDS20': ['2F.CDS20', 'XX.CDS20']}]
        :param merge_prodg1_list: A list of dictionaries for merging 'PRODG1' column in a similar fashion.
        :param merge_product_list: A list of dictionaries for merging 'PRODUCT_ID' column in a similar fashion.
        :param merge_eqp_list: A list of dictionaries for merging 'EQP_NAME' column in a similar fashion.
        :param merge_chamber_list: A list of dictionaries for merging 'TOOL_NAME' column in a similar fashion.

        :return: DataFrame with 'OPER_NO' and other specified columns integrated according to the merge rules.
        """
        df_merged = PreprocessForUvaData.integrate_single_column(df, merge_operno_list, 'OPER_NO')
        df_merged = PreprocessForUvaData.integrate_single_column(df_merged, merge_prodg1_list, 'PRODG1')
        df_merged = PreprocessForUvaData.integrate_single_column(df_merged, merge_product_list, 'PRODUCT_ID')
        df_merged = PreprocessForUvaData.integrate_single_column(df_merged, merge_eqp_list, 'EQP_NAME')
        df_merged = PreprocessForUvaData.integrate_single_column(df_merged, merge_chamber_list, 'TOOL_NAME')
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
            # Extract values from each dictionary in merge_operno_list and create a list
            values_to_replace = [list(rule.values())[0] for rule in merge_list]
            # Concatenate values from each dictionary
            merged_values = [splitter_comma.join(list(rule.values())[0]) for rule in merge_list]

            # Replace values in 'OPER_NO' column based on the rules defined in merge_operno_list
            for values, replacement_value in zip(values_to_replace, merged_values):
                df = df.withColumn(column_name,
                                   when(col(column_name).isin(values), replacement_value).otherwise(col(column_name)))
        return df

    @staticmethod
    def pre_process(df: pyspark.sql.dataframe) -> pyspark.sql.dataframe:
        """
        Preprocess the data extracted from the database for a specific CASE.
        :param df: Data for a specific CASE retrieved from the database.
        :return: Preprocessed data with relevant columns and filters applied.
        """
        # Select only the columns that will be used
        df = df.select('WAFER_ID', 'TOOL_ID', 'RUN_ID', 'EQP_NAME', 'PRODUCT_ID', 'PRODG1', 'TOOL_NAME',
                       'OPER_NO', 'parametric_name', 'STATISTIC_RESULT', 'label')
        # Remove rows with missing values in 'STATISTIC_RESULT' column
        df = df.filter(col('STATISTIC_RESULT').isNotNull())
        # Drop duplicates based on all columns
        df1 = df.dropDuplicates()
        # Select the rows with the latest 'RUN_ID' for each combination of 'WAFER_ID', 'OPER_NO', 'TOOL_ID'
        df2 = df1.groupBy('WAFER_ID', 'OPER_NO', 'TOOL_ID').agg(max('RUN_ID').alias('RUN_ID'))
        df_run = df1.join(df2.dropDuplicates(subset=['WAFER_ID', 'OPER_NO', 'TOOL_ID', 'RUN_ID']),
                          on=['WAFER_ID', 'OPER_NO', 'TOOL_ID', 'RUN_ID'], how='inner')
        return df_run

    @staticmethod
    def commonality_analysis(df_run: pyspark.sql.dataframe, grpby_list: List[str]) -> pyspark.sql.dataframe:
        """
        Perform commonality analysis on preprocessed data.
        :param df_run: Preprocessed data after data preprocessing.
        :param grpby_list: List of columns ['PRODG1', 'EQP_NAME', 'OPER_NO', 'PRODUCT_ID', 'TOOL_NAME'] for grouping.
                Example: grpby_list = ['PRODG1', 'TOOL_NAME', 'OPER_NO'], grpby_list = ['PRODUCT_ID', 'OPER_NO']
        :return: Results of commonality analysis, showing the top ten combinations with the highest number of bad wafers.
        """
        common_res = (df_run.groupBy(grpby_list)
                      .agg(countDistinct('WAFER_ID', when(df_run['label'] == 0, 1)).alias('good_num'),
                           countDistinct('WAFER_ID', when(df_run['label'] == 1, 1)).alias('bad_num'))
                      .na.fill(0))
        return common_res

    @staticmethod
    def get_data_list(common_res: pyspark.sql.dataframe,
                      grpby_list: List[str]) -> List[Dict[str, str]]:
        """
        Get a list of dictionaries for corresponding groups based on commonality analysis.

        :param common_res: Result of commonality analysis.
        :param grpby_list:  List of columns ['PRODG1', 'EQP_NAME', 'OPER_NO', 'PRODUCT_ID', 'TOOL_NAME'] for grouping.
        :return: List of dictionaries for corresponding groups.
                Example: [{'OPER_NO': '1F.EEK10', 'PRODUCT_ID': 'AFKN2J01N.0U01'},
                          {'OPER_NO': '1F.EEK10', 'PRODUCT_ID': 'AFKN4X01N.0B01'},
                          {'OPER_NO': '1F.EEK10', 'PRODUCT_ID': 'AFGN1501N.0C02'}]
        """
        data_list = common_res.select(grpby_list).collect()
        data_dict_list = [row.asDict() for row in data_list]
        return data_dict_list

    @staticmethod
    def get_train_data(df_run: pyspark.sql.dataframe, data_dict_list: List[Dict[str, str]]) -> pyspark.sql.dataframe:
        """
        Get the actual combination data for modeling from the original data.

        :param df_run: Preprocessed data after data preprocessing.
        :param data_dict_list: List of dictionaries with filtering conditions.
               Example: [{'OPER_NO': '1F.EEK10', 'PRODUCT_ID': 'AFKN2J01N.0U01'},
                          {'OPER_NO': '1F.EEK10', 'PRODUCT_ID': 'AFKN4X01N.0B01'},
                          {'OPER_NO': '1F.EEK10', 'PRODUCT_ID': 'AFGN1501N.0C02'}]
        :return: Filtered data for modeling.
        """
        first_data_dict = data_dict_list[0]
        conditions = " AND ".join(["{} == '{}'".format(col_, first_data_dict[col_]) for col_ in first_data_dict])
        df_s = df_run.filter(conditions)

        for i in range(1, len(data_dict_list)):
            data_dict = data_dict_list[i]
            conditions = " AND ".join(["{} == '{}'".format(col_, data_dict[col_]) for col_ in data_dict])
            df_m = df_run.filter(conditions)
            df_s = df_s.union(df_m)
        return df_s

    @staticmethod
    def get_all_bad_wafer_num(df: pyspark.sql.dataframe) -> int:
        """
        Get the number of distinct bad WAFER in the DataFrame.
        """
        return df.filter("label == 1").select('WAFER_ID').distinct().count()

    @staticmethod
    def add_feature_stats_for_grp_less(add_parametric_name_df, grpby_list):
        struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
        struct_fields.extend([StructField("bad_num", FloatType(), True),
                              StructField("good_num", FloatType(), True),
                              StructField("parametric_name", StringType(), True),
                              StructField("stats", StringType(), True)])
        schema_all = StructType(struct_fields)

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_result(df) -> pd.DataFrame:
            grpby_list_extend = grpby_list + ['good_num', 'bad_num', 'parametric_name', 'step']
            feature_stats = df.groupby(grpby_list_extend)['stats'].unique().reset_index()
            feature_stats['stats'] = [feature_stats['stats'].iloc[i].tolist() for i in range(len(feature_stats))]
            feature_stats['stats'] = feature_stats['stats'].apply(lambda x: "#".join(x))
            feature_stats = feature_stats.assign(
                parametric_name=lambda x: x['parametric_name'] + str('#') + x['step']).drop(
                'step', axis=1)
            return feature_stats

        return add_parametric_name_df.groupby(grpby_list).apply(get_result)

    def run(self):
        df_integrate_columns = self.integrate_columns(df=self.df,
                                                      merge_operno_list=self.merge_operno_list,
                                                      merge_prodg1_list=self.merge_prodg1_list,
                                                      merge_product_list=self.merge_product_list,
                                                      merge_eqp_list=self.merge_eqp_list,
                                                      merge_chamber_list=self.merge_chamber_list)

        df_run = self.pre_process(df_integrate_columns)
        # df_run_pandas = df_run.toPandas()
        # df_run_pandas.to_csv("df_run_new.csv")

        common_res = self.commonality_analysis(df_run=df_run, grpby_list=self.grpby_list)
        common_res = common_res.withColumn("conditions_satisfied",
                                           when((col('good_num') >= 1) & (col('bad_num') >= 1), True).otherwise(False))
        print("最原始的common_res:")
        print(common_res.count())
        common_res.show()

        grps_large = common_res.filter("good_num > 3000 AND bad_num > 3000")
        print("grps_large:")
        grps_large.show(50)
        if grps_large.isEmpty():
            grps_less = common_res.filter("good_num >= 1 AND bad_num >= 1")
            print("grps_less:")
            grps_less.show(50)
            if grps_less.isEmpty():
                msg = f"按照{'+'.join(self.grpby_list)}分组后的数据, 没有组合满足条件good >= 1且bad >= 1, 无法进行分析."
                raise RCABaseException(msg)
            else:
                conditions_not_satisfied = common_res.filter("conditions_satisfied==False").select(
                    self.grpby_list + ["good_num", 'bad_num'])
                add_parametric_name_df = conditions_not_satisfied.join(
                    df_run.select(self.grpby_list + ['parametric_name']), self.grpby_list, "left").dropDuplicates()
                add_parametric_name_df = (
                    add_parametric_name_df.withColumn('step', split(col('parametric_name'), '#').getItem(1))
                    .withColumn('stats', split(col('parametric_name'), '#').getItem(2))
                    .withColumn('parametric_name', split(col('parametric_name'), '#').getItem(0)))
                add_parametric_name_useless = self.add_feature_stats_for_grp_less(add_parametric_name_df,
                                                                                  self.grpby_list)

                data_dict_list = self.get_data_list(common_res=grps_less, grpby_list=self.grpby_list)
                train_data = self.get_train_data(df_run=df_run, data_dict_list=data_dict_list)
                big_or_small = 'small'
                bad_wafer_num = self.get_all_bad_wafer_num(train_data)
                return grps_less, train_data, bad_wafer_num, big_or_small, add_parametric_name_useless

        else:
            data_dict_list = self.get_data_list(common_res=grps_large, grpby_list=self.grpby_list)
            train_data = self.get_train_data(df_run=df_run, data_dict_list=data_dict_list)
            big_or_small = 'big'
            bad_wafer_num = self.get_all_bad_wafer_num(train_data)
            add_parametric_name_useless = None
            return grps_large, train_data, bad_wafer_num, big_or_small, add_parametric_name_useless


class FitModelForUvaData:
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
        index_cols = ['WAFER_ID', 'label']
        columns_cols = grpby_list + ['parametric_name']
        df_pivot = df.dropna(axis=0).pivot_table(index=index_cols,
                                                 columns=columns_cols,
                                                 values=['STATISTIC_RESULT'])
        df_pivot.columns = df_pivot.columns.map('#'.join)
        df_pivot = df_pivot.fillna(df_pivot.mean()).reset_index(drop=False)
        # Remove completely identical columns
        for column in df_pivot.columns.difference(index_cols):
            if df_pivot[column].nunique() == 1:
                df_pivot = df_pivot.drop(column, axis=1)
        return df_pivot

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
            return pipe, param_grid
        else:
            return None, None

    @staticmethod
    def construct_features_when_not_satisfied(df_run, df_pivot, grpby_list, x_train_shape, model_condition):
        grpby_values = [df_run[item].iloc[0] for item in grpby_list]
        features_value = f"STATISTIC_RESULT#{'#'.join(map(str, grpby_values))}#PARAM#STEPS#STATS"

        if model_condition == 'classification':
            sample_res_dict = {'bad_wafer': sum(df_pivot['label']), 'roc_auc_score': -1,
                               'algorithm_satisfied': False, 'x_train_shape': str(x_train_shape)}
            sample_res_dict.update({col_: df_run[col_].values[0] for col_ in grpby_list})
            small_sample_res = pd.DataFrame(sample_res_dict, index=[0])

            small_importance_res = pd.DataFrame({"features": features_value, "importance": -1}, index=[0])
            res_top_select = pd.concat([small_importance_res, small_sample_res])
            return res_top_select
        elif model_condition == 'pca':
            res_top_select = pd.DataFrame({"features": features_value,
                                           "importance": -1,
                                           "bad_wafer": sum(df_pivot['label']),
                                           "algorithm_satisfied": False,
                                           "x_train_shape": str(x_train_shape)}, index=[0])
            for col_ in grpby_list:
                res_top_select[col_] = df_run[col_].values[0]
            return res_top_select

    @staticmethod
    def fit_classification_model(df: pyspark.sql.dataframe, grpby_list: List[str], model) -> pyspark.sql.dataframe:
        struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
        struct_fields.extend([StructField("bad_wafer", IntegerType(), True),
                              StructField("roc_auc_score", FloatType(), True),
                              StructField("features", StringType(), True),
                              StructField("importance", FloatType(), True),
                              StructField("algorithm_satisfied", BooleanType(), True),
                              StructField("x_train_shape", StringType(), True)])
        schema_all = StructType(struct_fields)

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run: pd.DataFrame) -> pd.DataFrame:
            # Pivot the table
            df_pivot = FitModelForUvaData.get_pivot_table(df=df_run, grpby_list=grpby_list)

            # Define independent and dependent variables
            x_train = df_pivot[df_pivot.columns.difference(['WAFER_ID', 'label']).tolist()]
            y_train = df_pivot[['label']]

            if x_train.shape[1] > 1 and y_train['label'].nunique() > 1:
                z_ratio = y_train.value_counts(normalize=True)
                good_ratio = z_ratio[0]
                bad_ratio = z_ratio[1]
                if abs(good_ratio - bad_ratio) > 0.7:
                    undersampler = ClusterCentroids(random_state=1024)
                    x_train, y_train = undersampler.fit_resample(x_train, y_train)

                pipe, param_grid = FitModelForUvaData.get_pipe_params(model=model)
                try:
                    grid = GridSearchCV(estimator=pipe, scoring='roc_auc', param_grid=param_grid, cv=3, n_jobs=-1)
                    grid.fit(x_train.values, y_train.values.ravel())
                except ValueError:
                    return pd.DataFrame()

                best_est = grid.best_estimator_.steps[-1][-1]
                if hasattr(best_est, 'feature_importances_'):
                    small_importance_res = pd.DataFrame(
                        {'features': x_train.columns, 'importance': best_est.feature_importances_})
                else:
                    small_importance_res = pd.DataFrame(
                        {'features': x_train.columns, 'importance': abs(best_est.coef_.ravel())})

                sample_res_dict = {'bad_wafer': sum(df_pivot['label']), 'roc_auc_score': grid.best_score_,
                                   'algorithm_satisfied': True, 'x_train_shape': str(x_train.shape)}
                sample_res_dict.update({col_: df_run[col_].values[0] for col_ in grpby_list})
                small_sample_res = pd.DataFrame(sample_res_dict, index=[0])
                res_top_select = pd.concat([small_importance_res, small_sample_res])
                return res_top_select
            else:
                res_top_select = FitModelForUvaData.construct_features_when_not_satisfied(df_run, df_pivot, grpby_list,
                                                                                          x_train.shape,
                                                                                          'classification')
                return res_top_select

        return df.groupby(grpby_list).apply(get_model_result)

    @staticmethod
    def construct_features_when_satisfy_pca(df_run, df_pivot, x_train, grpby_list):
        n_components = min(min(x_train.shape) - 2, 20)
        model = pca(n_components=n_components, verbose=None)
        results = model.fit_transform(x_train)
        res_top = results['topfeat']
        res_top_select = res_top[res_top['type'] == 'best'][['feature', 'loading']]
        res_top_select['importance'] = abs(res_top_select['loading'])
        res_top_select = res_top_select.rename(columns={'feature': 'features'}).drop("loading",
                                                                                     axis=1).drop_duplicates()
        res_top_select['bad_wafer'] = sum(df_pivot['label'])
        for col_ in grpby_list:
            res_top_select[col_] = df_run[col_].values[0]
        res_top_select['x_train_shape'] = str(x_train.shape)
        res_top_select['algorithm_satisfied'] = True
        return res_top_select

    @staticmethod
    def fit_pca_model(df: pyspark.sql.dataframe, grpby_list: List[str]) -> pyspark.sql.dataframe:
        # Dynamically build schema according to the grpby_list
        struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
        struct_fields.extend([StructField("features", StringType(), True),
                              StructField("importance", FloatType(), True),
                              StructField("bad_wafer", IntegerType(), True),
                              StructField("algorithm_satisfied", BooleanType(), True),
                              StructField("x_train_shape", StringType(), True)])
        schema_all = StructType(struct_fields)

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run: pd.DataFrame) -> pd.DataFrame:
            df_pivot = FitModelForUvaData.get_pivot_table(df=df_run, grpby_list=grpby_list)
            # Since it is a small sample, make a copy to generate more data for the PCA model
            df_pivot_copy = df_pivot.copy()
            df_pivot_all = pd.concat([df_pivot, df_pivot_copy], axis=0)

            x_train = df_pivot_all[df_pivot_all.columns.difference(['WAFER_ID', 'label']).tolist()]

            if min(x_train.shape) > 100:
                res_top_select = FitModelForUvaData.construct_features_when_satisfy_pca(df_run, df_pivot, x_train,
                                                                                        grpby_list)
                return res_top_select
            else:
                res_top_select = FitModelForUvaData.construct_features_when_not_satisfied(df_run, df_pivot, grpby_list,
                                                                                          x_train.shape, 'pca')
                return res_top_select

        return df.groupby(grpby_list).apply(get_model_result)


class GetFinalResultsForUvaData:
    def __init__(self, df: pyspark.sql.dataframe, grpby_list: List[str], request_id: str,
                 grps_all: pyspark.sql.dataframe, bad_wafer_num, big_or_small,
                 add_parametric_name_useless):
        self.df = df
        self.grpby_list = grpby_list
        self.request_id = request_id
        self.grps_all = grps_all
        self.bad_wafer_num = bad_wafer_num
        self.big_or_small = big_or_small
        self.add_parametric_name_useless = add_parametric_name_useless

    @staticmethod
    def split_score_big_sample(df: pyspark.sql.dataframe, grpby_list: List[str]) -> pyspark.sql.dataframe:
        select_expr = grpby_list + ['bad_wafer', 'roc_auc_score', 'algorithm_satisfied', 'x_train_shape']
        selected_df = df.select(*select_expr)
        sample_res = selected_df.dropna()
        return sample_res

    @staticmethod
    def split_features(df: pd.DataFrame, index: int) -> str:
        """
        Split the 'features' column based on the specified index.

        Parameters:
        - df: RandomForest modeling results with 'features' column.
        - index: Order value.

        Returns:
        - str: Field attribute value.
        """
        return df['features'].apply(lambda x: x.split('#')[index])

    @staticmethod
    def get_split_feature_importance_table(df: pd.DataFrame, grpby_list: List[str]) -> pd.DataFrame:
        """
        Get the table after splitting the 'features' column based on the specified grouping columns.

        Parameters:
        - df: RandomForest modeling results with 'features' column.
        - grpby_list: List of grouping columns.

        Returns:
        - DataFrame: Table after splitting features.
        """
        n_feats = len(grpby_list)
        for i in range(n_feats):
            df[grpby_list[i]] = GetFinalResultsForUvaData.split_features(df, i + 1)

        df['parametric_name'] = GetFinalResultsForUvaData.split_features(df, n_feats + 1)
        df['step'] = GetFinalResultsForUvaData.split_features(df, n_feats + 2)
        df['stats'] = GetFinalResultsForUvaData.split_features(df, n_feats + 3)
        df = df.drop(['features'], axis=1).reset_index(drop=True)
        return df

    @staticmethod
    def add_feature_stats(df: pd.DataFrame, grpby_list: List[str]) -> pd.DataFrame:
        """
        Add a column with all statistical features of parameters.

        Parameters:
        - df: Feature importance table after processing.
        - grpby_list: List of grouping columns.

        Returns:
        - DataFrame: New column containing all statistical features: 'feature_stats'.
        """
        grpby_list_extend = grpby_list + ['parametric_name', 'step']
        feature_stats = df.groupby(grpby_list_extend)['stats'].unique().reset_index()
        feature_stats['stats'] = [feature_stats['stats'].iloc[i].tolist() for i in range(len(feature_stats))]
        feature_stats['stats'] = feature_stats['stats'].apply(lambda x: "#".join(x))
        feature_stats = feature_stats.assign(
            parametric_name=lambda x: x['parametric_name'] + str('#') + x['step']).drop(
            'step', axis=1)
        return feature_stats

    @staticmethod
    def split_calculate_features_big_sample(df: pyspark.sql.dataframe, grpby_list: List[str]) -> pyspark.sql.dataframe:
        """
        Split and calculate features based on the specified grouping columns.

        Parameters:
        - df: Results after RandomForest modeling.
        - grpby_list: List of grouping columns.

        Returns:
        - DataFrame: Features importance results.
        """
        # Dynamically build schema
        struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
        struct_fields.extend([StructField("parametric_name", StringType(), True),
                              StructField("importance", FloatType(), True),
                              StructField("stats", StringType(), True)])
        schema_all = StructType(struct_fields)

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_result(model_results: pd.DataFrame) -> pd.DataFrame:
            # Extract 'features' and 'importance' from the RandomForest model results
            feature_importance_table = model_results[['features', 'importance']].dropna(axis=0)

            # Split features
            feature_importance_res_split = GetFinalResultsForUvaData.get_split_feature_importance_table(
                df=feature_importance_table,
                grpby_list=grpby_list)

            # Remove combinations with importance equal to 0
            feature_importance_res_split_drop = feature_importance_res_split.query("importance > 0.0001").reset_index(
                drop=True)

            # Take the top 60% or 100% of each combination result
            feature_importance_res_split_nlargest = (feature_importance_res_split_drop.groupby(by=grpby_list)
                                                     .apply(
                lambda x: x.nlargest(int(x.shape[0] * 0.6), 'importance') if x.shape[0] > 1 else x.nlargest(
                    int(x.shape[0] * 1), 'importance'))
                                                     .reset_index(drop=True))

            # Add a column with all statistical features: 'feature_stats'
            feature_stats = GetFinalResultsForUvaData.add_feature_stats(df=feature_importance_res_split_drop,
                                                                        grpby_list=grpby_list)

            # Sum the importance for the same combination and parameter: 'feature_importance_groupby'
            feature_importance_groupby = (
                feature_importance_res_split_nlargest.groupby(grpby_list + ['parametric_name', 'step'])['importance']
                .sum().reset_index())
            feature_importance_groupby = (
                feature_importance_groupby.assign(parametric_name=lambda x: x['parametric_name'] + str('#') + x['step'])
                .drop('step', axis=1))

            # Connect 'feature_stats' and 'feature_importance_groupby'
            grpby_stats = pd.merge(feature_stats, feature_importance_groupby,
                                   on=grpby_list + ['parametric_name']).dropna().reset_index(drop=True)
            return grpby_stats

        return df.groupby(grpby_list).apply(get_result)

    @staticmethod
    def get_final_results_big_sample(s_res: pyspark.sql.dataframe, f_res: pyspark.sql.dataframe, grpby_list: List[str],
                                     bad_wafer_num: int) -> pyspark.sql.dataframe:
        """
        Get the final modeling results.

        Parameters:
        - s_res: ROC AUC scores result.
        - f_res: Features importance result.
        - grpby_list: List of grouping columns.
        - bad_wafer_num: Number of bad wafers in the data.

        Returns:
        - DataFrame: Final modeling result.
        """
        roc_auc_score_all = s_res.agg({"roc_auc_score": "sum"}).collect()[0][0]
        s_res = s_res.withColumn("roc_auc_score_ratio", col("roc_auc_score") / roc_auc_score_all)
        s_res = s_res.withColumn("bad_ratio", col("bad_wafer") / bad_wafer_num)

        df_merge = s_res.join(f_res, on=grpby_list, how='left')
        df_merge = df_merge.withColumn('weight_original',
                                       col('roc_auc_score_ratio') * col('bad_ratio') * col('importance'))

        # Normalize again
        weight_all = df_merge.agg({"weight_original": "sum"}).collect()[0][0]
        df_merge = df_merge.withColumn("weight", col("weight_original") / weight_all)
        df_merge = df_merge.select(grpby_list + ['parametric_name', 'weight', 'stats'])
        return df_merge

    @staticmethod
    def split_calculate_features_small_sample(df: pyspark.sql.dataframe,
                                              grpby_list: List[str]) -> pyspark.sql.dataframe:
        """
        Calculate features and importance after PCA modeling on a small sample.

        Parameters:
        - df: PCA modeling results (pyspark.sql.dataframe).
        - grpby_list: List of grouping columns (List[str]).

        Returns:
        - DataFrame: Dataframe containing features, importance and other information after PCA modeling.
        """
        # Dynamically build schema
        struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
        struct_fields.extend([StructField("parametric_name", StringType(), True),
                              StructField("importance", FloatType(), True),
                              # StructField("bad_wafer", IntegerType(), True),
                              StructField("stats", StringType(), True),
                              StructField("algorithm_satisfied", BooleanType(), True),
                              StructField("x_train_shape", StringType(), True)])
        schema_all = StructType(struct_fields)

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_result(model_results: pd.DataFrame) -> pd.DataFrame:
            feature_importance_table = model_results[
                ['features', 'importance', 'algorithm_satisfied', 'x_train_shape']].dropna(axis=0)
            # Split features
            feature_importance_res_split = GetFinalResultsForUvaData.get_split_feature_importance_table(
                df=feature_importance_table,
                grpby_list=grpby_list)

            # Add a column with all statistical features containing parameters: feature_stats
            feature_stats = GetFinalResultsForUvaData.add_feature_stats(df=feature_importance_res_split,
                                                                        grpby_list=grpby_list)

            # Sum the same parameter in the same combination: feature_importance_groupby
            feature_importance_groupby = (
                feature_importance_res_split.groupby(
                    grpby_list + ['parametric_name', 'step', 'algorithm_satisfied', 'x_train_shape'])[
                    'importance'].sum().reset_index())
            feature_importance_groupby = feature_importance_groupby.assign(
                parametric_name=lambda x: x['parametric_name'] + str('#') + x['step']).drop('step', axis=1)

            # Connect feature_stats and feature_importance_groupby
            grpby_stats = pd.merge(feature_stats, feature_importance_groupby,
                                   on=grpby_list + ['parametric_name']).dropna().reset_index(drop=True)
            return grpby_stats

        return df.groupby(grpby_list).apply(get_result)

    @staticmethod
    def get_final_results_small_sample(f_res: pyspark.sql.dataframe,
                                       grpby_list: List[str]) -> pyspark.sql.dataframe:
        """
        Get the final modeling results for a small sample.

        Parameters:
        - f_res: Features and importance results (pyspark.sql.dataframe).
        - grpby_list: List of grouping columns (List[str]).

        Returns:
        - DataFrame: Final modeling results with weights and statistics.
        """
        # f_res = f_res.withColumn("bad_ratio", col("bad_wafer") / bad_wafer_num)
        # df_merge = f_res.withColumn('weight_original', col('importance') * col('bad_ratio'))

        # Normalize weights again
        weight_all = f_res.agg({"importance": "sum"}).collect()[0][0]
        df_merge = f_res.withColumn("weight", col("importance") / weight_all)

        # Select columns
        df_merge = df_merge.select(grpby_list + ['parametric_name', 'weight', 'stats'])
        return df_merge

    @staticmethod
    def add_certain_column(df: pyspark.sql.dataframe, request_id: str, grps_all: pyspark.sql.dataframe,
                           grpby_list, add_parametric_name_useless) -> pyspark.sql.dataframe:
        df = df.filter("weight > 0.0001")
        df = df.join(grps_all.select(grpby_list + ['good_num', 'bad_num']), on=grpby_list, how='right')
        if add_parametric_name_useless is not None:
            df = df.unionByName(add_parametric_name_useless, allowMissingColumns=True)

        df = (df.withColumn("weight_percent", col("weight") * 100)
              .withColumn("good_num", df["good_num"].cast(FloatType()))
              .withColumn("bad_num", df["bad_num"].cast(FloatType()))
              .withColumn("request_id", lit(request_id)))
        df = df.orderBy(col("weight").desc())
        df = df.withColumn('index_no', monotonically_increasing_id() + 1)
        info_list = ['PRODUCT_ID', 'OPER_NO', 'EQP_NAME', 'PRODG1', 'TOOL_NAME']
        for column in info_list:
            if column not in df.columns:
                df = df.withColumn(column, lit(None).cast(StringType()))
        return df

    def run(self):
        if self.big_or_small == 'big':
            s_res = self.split_score_big_sample(df=self.df, grpby_list=self.grpby_list)
            m = s_res.filter('algorithm_satisfied==True').count()
            if m == 0:
                from src.exceptions.rca_base_exception import RCABaseException
                msg = f"按照{'+'.join(self.grpby_list)}分组后的数据, 每个组合中只有一种sensor_name, 无法进行差异性分析."
                raise RCABaseException(msg)
            print("s_res:")
            s_res.show()

            # self.grps_all是commonality_analysis后的结果, 需要剔除不满足算法条件的那些组合, 得到grps_all_new
            algorithm_not_satisfied = s_res.filter("algorithm_satisfied==False").select(self.grpby_list)
            grps_all_new = self.grps_all.join(algorithm_not_satisfied, self.grpby_list, "left_anti")
            print("grps_all_new:")
            print(grps_all_new.count())
            grps_all_new.show()

            f_res = self.split_calculate_features_big_sample(df=self.df, grpby_list=self.grpby_list)
            print("f_res:")
            f_res.show()

            res_all = self.get_final_results_big_sample(s_res=s_res, f_res=f_res, grpby_list=self.grpby_list,
                                                        bad_wafer_num=self.bad_wafer_num)
            print("res_all:")
            res_all.show()

            final_res = self.add_certain_column(df=res_all, request_id=self.request_id, grps_all=grps_all_new,
                                                grpby_list=self.grpby_list,
                                                add_parametric_name_useless=self.add_parametric_name_useless)
            return final_res

        elif self.big_or_small == 'small':
            df = self.df.filter('algorithm_satisfied==True')
            if df.count() == 0:
                from src.exceptions.rca_base_exception import RCABaseException
                msg = f"按照{'+'.join(self.grpby_list)}分组后的数据, 每个组合中只有一种sensor_name, 无法进行差异性分析."
                raise RCABaseException(msg)

            algorithm_not_satisfied = self.df.filter("algorithm_satisfied==False").select(self.grpby_list)
            print("algorithm_not_satisfied:")
            algorithm_not_satisfied.show()
            grps_all_new = self.grps_all.join(algorithm_not_satisfied, self.grpby_list, "left_anti")
            print("grps_all_new:")
            print(grps_all_new.count())
            grps_all_new.show()

            f_res = self.split_calculate_features_small_sample(df=df, grpby_list=self.grpby_list)
            print("f_res:")
            f_res.show()
            f_res = f_res.withColumn("algorithm_satisfied", f_res["algorithm_satisfied"].cast(StringType()))
            f_res_pandas = f_res.toPandas()
            f_res_pandas.to_csv("f_res_pandas5.csv")

            res_all = self.get_final_results_small_sample(f_res=f_res, grpby_list=self.grpby_list)
            print("res_all:")
            res_all.show()

            final_res = self.add_certain_column(df=res_all, request_id=self.request_id, grps_all=grps_all_new,
                                                grpby_list=self.grpby_list,
                                                add_parametric_name_useless=self.add_parametric_name_useless)
            return final_res
        else:
            raise ValueError("Invalid value for self.big_or_small: {}".format(self.big_or_small))


class ExertUvaAlgorithm:
    @staticmethod
    def fit_uva_model(df: pyspark.sql.dataframe,
                      grpby_list: List[str],
                      request_id: str,
                      merge_operno_list: List[Dict[str, List[str]]],
                      merge_prodg1_list: List[Dict[str, List[str]]],
                      merge_product_list: List[Dict[str, List[str]]],
                      merge_eqp_list: List[Dict[str, List[str]]],
                      merge_chamber_list: List[Dict[str, List[str]]]):

        common_res, train_data, bad_wafer_num, big_or_small, add_parametric_name_useless = PreprocessForUvaData(df=df,
                                                                                                                grpby_list=grpby_list,
                                                                                                                merge_operno_list=merge_operno_list,
                                                                                                                merge_prodg1_list=merge_prodg1_list,
                                                                                                                merge_product_list=merge_product_list,
                                                                                                                merge_eqp_list=merge_eqp_list,
                                                                                                                merge_chamber_list=merge_chamber_list).run()
        print("bad_wafer_num: ", bad_wafer_num)
        print("过滤后的common_res是：")
        # common_res = common_res.withColumn("conditions_satisfied", common_res["conditions_satisfied"].cast(StringType()))
        # common_res_pandas = common_res.toPandas()
        # common_res_pandas.to_csv("common_res_pandas.csv")

        print(common_res.count())
        common_res.show()

        if big_or_small == 'big':
            print("****************Call Big Sample Algorithm****************")
            result = FitModelForUvaData.fit_classification_model(df=train_data, grpby_list=grpby_list, model='rf')

        else:
            print("****************Call Small Sample Algorithm****************")
            result = FitModelForUvaData.fit_pca_model(df=train_data, grpby_list=grpby_list)

        print("最后算法结果是：")
        result.show(30)

        final_res = GetFinalResultsForUvaData(df=result, grpby_list=grpby_list, request_id=request_id,
                                              grps_all=common_res,
                                              bad_wafer_num=bad_wafer_num, big_or_small=big_or_small,
                                              add_parametric_name_useless=add_parametric_name_useless).run()
        print("写回数据库的结果是：")
        final_res.show(30)

        final_res_pandas = final_res.toPandas()
        final_res_pandas.to_csv("final_res_pandas1_5.csv")
        return final_res


if __name__ == '__main__':
    import os
    import json
    import warnings
    import pandas as pd
    import pyspark.pandas as ps
    from pyspark.sql import SparkSession

    os.environ['PYSPARK_PYTHON'] = '/usr/local/python-3.9.13/bin/python3'
    warnings.filterwarnings('ignore')

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
        "D:/Jupyterfiles/晶合MVAFDC_general开发/MVAanlysisDevelop/uva_algorithm/CASE1_DATA/DWD_POC_CASE_FD_UVA_DATA_CASE1_PROCESSED1.csv")
    # df_pandas = pd.read_csv(
    #     "D:/Jupyterfiles/晶合MVAFDC_general开发/MVAanlysisDevelop/uva_algorithm/small_samples_data/small2_labeled.csv")
    df_spark = ps.from_pandas(df_pandas).to_spark()
    print(f"df_spark shape: ({df_spark.count()}, {len(df_spark.columns)})")
    df_spark.show()

    json_loads_dict = {
        "requestId": "uva",
        "requestParam": {'dateRange': [{'start': "2023-12-01 00:00:00", 'end': "2024-01-15 00:00:00"}],
                         'lot': [],
                         'operNo': [],
                         'prodg1': [],
                         'productId': [],
                         'eqp': [],
                         'tool': [],
                         'recipeName': [],
                         'waferId': {'good': [],
                                     'bad': []},
                         'uploadId': '20240110170016023',
                         "flagMergeAllProdg1": "0",
                         "flagMergeAllProductId": "0",
                         "flagMergeAllChamber": "0",
                         "mergeProdg1": [],
                         "mergeProductId": [{"xx_cc": ["AFGN1501N.0C02", "AFKN2J01N.0U01"]}],
                         # "mergeProductId": [],
                         "mergeEqp": [],
                         "mergeChamber": [],
                         "mergeOperno": [],
                         # 'mergeOperno': [{"2F.CDS10_XX.TDS01": ["2F.CDS10", "XX.TDS01"]},
                         #                 {"2F.CDS20_XX.CDS20": ["2F.CDS20", "XX.CDS20"]}]
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
    merge_eqp = list(parse_dict.get('mergeEqp')) if parse_dict.get('mergeEqp') else None
    merge_chamber = list(parse_dict.get('mergeChamber')) if parse_dict.get('mergeChamber') else None
    grpby_list_ = ['OPER_NO', 'TOOL_NAME', 'PRODUCT_ID']
    # grpby_list_ = ['OPER_NO', 'TOOL_NAME']
    # grpby_list_ = ['PRODUCT_ID']

    final_res_ = ExertUvaAlgorithm.fit_uva_model(df=df_spark,
                                                 grpby_list=grpby_list_,
                                                 request_id=request_id_,
                                                 merge_operno_list=merge_operno,
                                                 merge_prodg1_list=merge_prodg1,
                                                 merge_product_list=merge_product,
                                                 merge_eqp_list=merge_eqp,
                                                 merge_chamber_list=merge_chamber)
