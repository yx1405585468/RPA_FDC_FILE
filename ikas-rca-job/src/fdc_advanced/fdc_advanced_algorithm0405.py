import pandas as pd
import pyspark.sql.dataframe
from pca import pca
from pyspark.sql.functions import max, countDistinct, when, lit, pandas_udf, PandasUDFType, monotonically_increasing_id
from typing import List, Dict, Union
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, StructType, StructField
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import ClusterCentroids
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from src.exceptions.rca_base_exception import RCABaseException


class PreprocessForRunData:
    @staticmethod
    def integrate_columns(df: pyspark.sql.DataFrame,
                          merge_operno_list: List[Dict[str, List[str]]],
                          merge_prodg1_list: List[Dict[str, List[str]]],
                          merge_product_list: List[Dict[str, List[str]]],
                          merge_eqp_list: List[Dict[str, List[str]]],
                          merge_chamber_list: List[Dict[str, List[str]]]) -> pyspark.sql.DataFrame:
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
        df_merged = PreprocessForRunData.integrate_single_column(df, merge_operno_list, 'OPER_NO')
        df_merged = PreprocessForRunData.integrate_single_column(df_merged, merge_prodg1_list, 'PRODG1')
        df_merged = PreprocessForRunData.integrate_single_column(df_merged, merge_product_list, 'PRODUCT_ID')
        df_merged = PreprocessForRunData.integrate_single_column(df_merged, merge_eqp_list, 'EQP_NAME')
        df_merged = PreprocessForRunData.integrate_single_column(df_merged, merge_chamber_list, 'TOOL_NAME')
        df_merged.persist()
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
    def pre_process(df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """
        Preprocess the data extracted from the database for a specific CASE.
        :param df: Data for a specific CASE retrieved from the database.
        :return: Preprocessed data with relevant columns and filters applied.
        """
        try:
            delete_df = df.dropDuplicates(
                subset=['WAFER_ID', 'TOOL_ID', 'RUN_ID', 'EQP_NAME', 'PRODUCT_ID', 'PRODG1', 'TOOL_NAME',
                        'LOT_ID', 'RECIPE_NAME', 'OPER_NO', 'parametric_name', 'mean', 'std', 'min',
                        '25percentpoint', 'median', '75percentpoint', 'max', 'range1', 'label'])
            # Select only the columns that will be used
            df1 = delete_df.select('WAFER_ID', 'TOOL_ID', 'RUN_ID', 'EQP_NAME', 'PRODUCT_ID', 'PRODG1', 'TOOL_NAME',
                                   'LOT_ID', 'RECIPE_NAME', 'OPER_NO', 'parametric_name', 'mean', 'std', 'min',
                                   '25percentpoint',
                                   'median', '75percentpoint', 'max', 'range1', 'label')
            # Drop duplicates based on all columns
            # Select the rows with the latest 'RUN_ID' for each combination of 'WAFER_ID', 'OPER_NO', 'TOOL_ID'
            df2 = df1.groupBy('WAFER_ID', 'OPER_NO', 'TOOL_ID').agg(max('RUN_ID').alias('RUN_ID'))
            df_run = df1.join(df2.dropDuplicates(subset=['WAFER_ID', 'OPER_NO', 'TOOL_ID', 'RUN_ID']),
                              on=['WAFER_ID', 'OPER_NO', 'TOOL_ID', 'RUN_ID'], how='inner')
            df_run.persist()
            return df_run
        except Exception as e:
            raise RCABaseException(e)
            # return df

    @staticmethod
    def commonality_analysis(df_run: pyspark.sql.dataframe, grpby_list: List[str]) -> pyspark.sql.dataframe:
        """
        Perform commonality analysis on preprocessed data.
        :param df_run: Preprocessed data after data preprocessing.
        :param grpby_list: List of columns ['PRODG1', 'EQP_NAME', 'OPER_NO', 'PRODUCT_ID', 'TOOL_NAME'] for grouping.
                Example: grpby_list = ['PRODG1', 'TOOL_NAME', 'OPER_NO'], grpby_list = ['PRODUCT_ID', 'OPER_NO']
        :return: Results of commonality analysis.
        """
        grps = (df_run.groupBy(grpby_list)
                .agg(countDistinct('WAFER_ID').alias('wafer_count'),
                     countDistinct('WAFER_ID', when(df_run['label'] == 0, 1)).alias('good_num'),
                     countDistinct('WAFER_ID', when(df_run['label'] == 1, 1)).alias('bad_num'))
                .orderBy('bad_num', ascending=False))

        if grps.count() == 1:
            return grps
        else:
            grps = grps.filter("bad_num > 1 AND wafer_count > 2")
            return grps

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
        # Order the results and limit to the top 10 groups
        good_bad_grps = common_res.orderBy(col("bad_num").desc(), col("wafer_count").desc(),
                                           col("good_num").desc()).limit(10)

        # Collect the data and convert it into a list of dictionaries
        data_list = good_bad_grps[grpby_list].collect()
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
        # Get the filtering conditions for the first data dictionary
        first_data_dict = data_dict_list[0]
        conditions = " AND ".join(["{} == '{}'".format(col_, first_data_dict[col_]) for col_ in first_data_dict])
        # Filter the data for the first condition
        df_s = df_run.filter(conditions)

        # Loop through the remaining data dictionaries and filter the data accordingly
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


class FitModelForRunData:

    def __init__(self, df, grpby_list, model):
        self.df = df
        self.grpby_list = grpby_list
        self.model = model

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
        values_cols = ['mean', 'std', 'min', '25percentpoint', 'median', '75percentpoint', 'max', 'range1']
        df_pivot = df.dropna(axis=0).pivot_table(index=index_cols,
                                                 columns=columns_cols,
                                                 values=values_cols)
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
    def fit_classification_model(df: pyspark.sql.DataFrame, grpby_list: List[str], model) -> pyspark.sql.DataFrame:
        struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
        struct_fields.extend([StructField("bad_wafer", IntegerType(), True),
                              StructField("roc_auc_score", FloatType(), True),
                              StructField("features", StringType(), True),
                              StructField("importance", FloatType(), True)])
        schema_all = StructType(struct_fields)

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run: pd.DataFrame) -> pd.DataFrame:
            # Pivot the table
            df_pivot = FitModelForRunData.get_pivot_table(df=df_run, grpby_list=grpby_list)

            # Define independent and dependent variables
            x_train = df_pivot[df_pivot.columns.difference(['WAFER_ID', 'label']).tolist()]
            y_train = df_pivot[['label']]

            if x_train.shape[1] > 6 and y_train['label'].nunique() > 1:
                z_ratio = y_train.value_counts(normalize=True)
                good_ratio = z_ratio[0]
                bad_ratio = z_ratio[1]
                if abs(good_ratio - bad_ratio) > 0.7:
                    undersampler = ClusterCentroids(random_state=1024)
                    x_train, y_train = undersampler.fit_resample(x_train, y_train)

                pipe, param_grid = FitModelForRunData.get_pipe_params(model=model)
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

                sample_res_dict = {'bad_wafer': sum(df_pivot['label']), 'roc_auc_score': grid.best_score_}
                sample_res_dict.update({col_: df_run[col_].values[0] for col_ in grpby_list})
                small_sample_res = pd.DataFrame(sample_res_dict, index=[0])
                return pd.concat([small_importance_res, small_sample_res])
            else:
                return pd.DataFrame()

        res_df = df.groupby(grpby_list).apply(get_model_result)
        res_df.persist()
        return res_df

    @staticmethod
    def fit_pca_model(df, grpby_list):
        struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
        struct_fields.extend([StructField("bad_wafer", IntegerType(), True),
                              StructField("roc_auc_score", FloatType(), True),
                              StructField("features", StringType(), True),
                              StructField("importance", FloatType(), True)])
        schema_all = StructType(struct_fields)

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run):
            df_pivot = FitModelForRunData.get_pivot_table(df=df_run, grpby_list=grpby_list)
            df_pivot_copy = df_pivot.copy()
            df_pivot_all = pd.concat([df_pivot, df_pivot_copy], axis=0)

            x_train = df_pivot_all[df_pivot_all.columns.difference(['WAFER_ID', 'label']).tolist()]
            if min(x_train.shape) > 2:
                n_components = min(min(x_train.shape) - 2, 20)
                model = pca(n_components=n_components, verbose=None)
                results = model.fit_transform(x_train)
                res_top = results['topfeat']
                res_top_select = res_top[res_top['type'] == 'best'][['feature', 'loading']]
                res_top_select['importance'] = abs(res_top_select['loading'])
                res_top_select = res_top_select.rename(columns={'feature': 'features'}).drop("loading",
                                                                                             axis=1).drop_duplicates()

                sample_res_dict = {'bad_wafer': sum(df_pivot['label']), 'roc_auc_score': -100}
                sample_res_dict.update({col_: df_run[col_].values[0] for col_ in grpby_list})
                small_sample_res = pd.DataFrame(sample_res_dict, index=[0])

                return pd.concat([res_top_select, small_sample_res])
            else:
                return pd.DataFrame()

        return df.groupby(grpby_list).apply(get_model_result)

    def run(self):
        if self.model in ['rf', 'decisionTree', 'svc', 'logistic', 'sgd']:
            print(f"**************fit {self.model} model**************")
            res_classification = self.fit_classification_model(df=self.df, grpby_list=self.grpby_list, model=self.model)
            if res_classification.isEmpty():
                print(f"**************Results of {self.model} is empty, now fitting pca model**************")
                res_pca = self.fit_pca_model(df=self.df, grpby_list=self.grpby_list)
                return res_pca
            else:
                return res_classification
        elif self.model == 'pca':
            print("**************fit pca model**************")
            res_pca = self.fit_pca_model(df=self.df, grpby_list=self.grpby_list)
            return res_pca
        else:
            raise Exception('Wrong Model Selection. Supported models are: pca, rf, decisionTree, svc, logistic, sgd.')


class GetFinalResultsForRunData:

    def __init__(self, df, grpby_list, request_id, bad_wafer_num):
        self.df = df
        self.grpby_list = grpby_list
        self.request_id = request_id
        self.bad_wafer_num = bad_wafer_num

    @staticmethod
    def split_score(df: pyspark.sql.dataframe, grpby_list: List[str]) -> pyspark.sql.dataframe:
        select_expr = grpby_list + ['bad_wafer', 'roc_auc_score']
        sample_res = df.select(select_expr)
        sample_res = sample_res.dropna().filter(col('roc_auc_score') > 0)
        return sample_res

    @staticmethod
    def split_features(df: pd.DataFrame, index: int) -> str:
        """
        Split the 'features' column based on the specified index.

        Parameters:
        - df: Modeling results with 'features' column.
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
        - df: Modeling results with 'features' column.
        - grpby_list: List of grouping columns.

        Returns:
        - DataFrame: Table after splitting features.
        """
        n_feats = len(grpby_list)
        for i in range(n_feats):
            df[grpby_list[i]] = GetFinalResultsForRunData.split_features(df, i + 1)

        df['parametric_name'] = GetFinalResultsForRunData.split_features(df, n_feats + 1)
        df = df.drop(['features'], axis=1).reset_index(drop=True)
        return df

    @staticmethod
    def split_calculate_features(df: pyspark.sql.dataframe, grpby_list: List[str]) -> pyspark.sql.dataframe:
        """
        Split and calculate features based on the specified grouping columns.

        Parameters:
        - df: Results after modeling.
        - grpby_list: List of grouping columns.

        Returns:
        - DataFrame: Features importance results.
        """
        # Dynamically build schema
        struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
        struct_fields.extend([StructField("parametric_name", StringType(), True),
                              StructField("importance", FloatType(), True)])
        schema_all = StructType(struct_fields)

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_result(model_results: pd.DataFrame) -> pd.DataFrame:
            feature_importance_table = model_results[['features', 'importance']].dropna(axis=0)

            # Split features
            feature_importance_res_split = GetFinalResultsForRunData.get_split_feature_importance_table(
                df=feature_importance_table,
                grpby_list=grpby_list)

            feature_importance_res_split_drop = feature_importance_res_split.query("importance > 0").reset_index(
                drop=True)

            # Take the top 60% or 100% of each combination result
            feature_importance_res_split_nlargest = (feature_importance_res_split_drop.groupby(by=grpby_list)
                                                     .apply(
                lambda x: x.nlargest(int(x.shape[0] * 0.6), 'importance') if x.shape[0] > 1 else x.nlargest(
                    int(x.shape[0] * 1), 'importance')).reset_index(drop=True))

            # Sum the importance for the same combination and parameter: 'feature_importance_groupby'
            feature_importance_groupby = (
                feature_importance_res_split_nlargest.groupby(grpby_list + ['parametric_name'])[
                    'importance'].sum().reset_index())

            return feature_importance_groupby

        return df.groupby(grpby_list).apply(get_result)

    @staticmethod
    def get_final_results(s_res: pyspark.sql.dataframe,
                          f_res: pyspark.sql.dataframe,
                          grpby_list: List[str],
                          bad_wafer_num: int) -> pyspark.sql.dataframe:
        if not s_res.isEmpty():
            roc_auc_score_all = s_res.agg({"roc_auc_score": "sum"}).collect()[0][0]
            s_res = s_res.withColumn("roc_auc_score_ratio", col("roc_auc_score") / roc_auc_score_all).withColumn(
                "bad_ratio", col("bad_wafer") / bad_wafer_num)
            df_merge = s_res.join(f_res, on=grpby_list, how='left')
            df_merge = df_merge.withColumn('weight_original',
                                           col('roc_auc_score_ratio') * col('bad_ratio') * col('importance'))
        else:
            df_merge = f_res.withColumnRenamed('importance', 'weight_original')

        # Normalize again
        weight_all = df_merge.agg({"weight_original": "sum"}).collect()[0][0]
        df_merge = df_merge.withColumn("weight", col("weight_original") / weight_all)
        df_merge = df_merge.select(grpby_list + ['parametric_name', 'weight']).orderBy('weight', ascending=False)
        return df_merge

    @staticmethod
    def add_certain_column(df: pyspark.sql.DataFrame, request_id: str) -> pyspark.sql.DataFrame:
        df = df.withColumn('weight_percent', col('weight') * 100)
        df = df.withColumn('request_id', lit(request_id))
        df = df.withColumn('index_no', monotonically_increasing_id() + 1)

        info_list = ['PRODUCT_ID', 'OPER_NO', 'EQP_NAME', 'PRODG1', 'TOOL_NAME']
        for column in info_list:
            if column not in df.columns:
                df = df.withColumn(column, lit(None).cast(StringType()))
        return df

    def run(self):
        s_res = GetFinalResultsForRunData.split_score(df=self.df, grpby_list=self.grpby_list)
        f_res = GetFinalResultsForRunData.split_calculate_features(df=self.df, grpby_list=self.grpby_list)
        final_res = GetFinalResultsForRunData.get_final_results(s_res=s_res, f_res=f_res,
                                                                grpby_list=self.grpby_list,
                                                                bad_wafer_num=self.bad_wafer_num)
        final_res_add_columns = GetFinalResultsForRunData.add_certain_column(df=final_res, request_id=self.request_id)
        final_res_add_columns.unpersist()
        return final_res_add_columns


class ExertAdvancedAlgorithm:
    @staticmethod
    def fit_advanced_model(df: pyspark.sql.DataFrame,
                           grpby_list: List[str],
                           merge_operno_list: List[Dict[str, List[str]]],
                           merge_prodg1_list: List[Dict[str, List[str]]],
                           merge_product_list: List[Dict[str, List[str]]],
                           merge_eqp_list: List[Dict[str, List[str]]],
                           merge_chamber_list: List[Dict[str, List[str]]],
                           request_id: str,
                           model: str = 'rf') -> Union[str, pyspark.sql.DataFrame]:

        df_integrate_columns = PreprocessForRunData.integrate_columns(df=df,
                                                                      merge_operno_list=merge_operno_list,
                                                                      merge_prodg1_list=merge_prodg1_list,
                                                                      merge_product_list=merge_product_list,
                                                                      merge_eqp_list=merge_eqp_list,
                                                                      merge_chamber_list=merge_chamber_list)
        if df_integrate_columns.isEmpty():
            msg = 'Merge columns exception!'
            raise RCABaseException(msg)

        df_run = PreprocessForRunData.pre_process(df_integrate_columns)
        if df_run.isEmpty():
            msg = 'No data in the database under this condition!'
            raise RCABaseException(msg)

        common_res = PreprocessForRunData.commonality_analysis(df_run=df_run, grpby_list=grpby_list)
        if common_res.isEmpty():
            msg = 'No bad or good wafer in this data. Unable to do the commonality analysis.'
            raise RCABaseException(msg)

        data_dict_list = PreprocessForRunData.get_data_list(common_res=common_res, grpby_list=grpby_list)
        train_data = PreprocessForRunData.get_train_data(df_run=df_run, data_dict_list=data_dict_list)
        all_bad_wafer_num = PreprocessForRunData.get_all_bad_wafer_num(df=train_data)
        if train_data.isEmpty():
            msg = 'No data of this type in the database!'
            raise RCABaseException(msg)

        res = FitModelForRunData(df=train_data, grpby_list=grpby_list, model=model).run()
        if res.isEmpty():
            msg = 'No difference in this data. The output of the algorithm is 0.'
            raise RCABaseException(msg)

        final_res_add_columns = GetFinalResultsForRunData(df=res, grpby_list=grpby_list, request_id=request_id,
                                                          bad_wafer_num=all_bad_wafer_num).run()
        if final_res_add_columns.isEmpty():
            msg = 'Results are empty.'
            raise RCABaseException(msg)

        return final_res_add_columns
