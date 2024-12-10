import numpy as np
import pandas as pd
import pyspark.sql.dataframe
from pca import pca
from pyspark.sql.functions import max, countDistinct, when, rank, lit, pandas_udf, PandasUDFType
from pyspark.sql.window import Window
from typing import List, Dict, Union
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, StructType, StructField
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import ClusterCentroids
from src.exceptions.rca_base_exception import RCABaseException


class PreprocessForUvaData:
    @staticmethod
    def integrate_columns(df: pyspark.sql.dataframe,
                          merge_operno_list: List[Dict[str, List[str]]],
                          merge_prodg1_list: List[Dict[str, List[str]]],
                          merge_product_list: List[Dict[str, List[str]]],
                          merge_eqp_list: List[Dict[str, List[str]]],
                          merge_chamber_list: List[Dict[str, List[str]]], ) -> pyspark.sql.dataframe:
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
        # split using comma
        splitter_comma = ","
        if merge_operno_list is not None and len(merge_operno_list) > 0:
            # Extract values from each dictionary in merge_operno_list and create a list
            values_to_replace = [list(rule.values())[0] for rule in merge_operno_list]
            # Concatenate values from each dictionary
            merged_values = [splitter_comma.join(list(rule.values())[0]) for rule in merge_operno_list]

            # Replace values in 'OPER_NO' column based on the rules defined in merge_operno_list
            for values, replacement_value in zip(values_to_replace, merged_values):
                df = df.withColumn("OPER_NO",
                                   when(col("OPER_NO").isin(values), replacement_value).otherwise(col("OPER_NO")))

        if merge_prodg1_list is not None and len(merge_prodg1_list) > 0:
            values_to_replace = [list(rule.values())[0] for rule in merge_prodg1_list]
            merged_values = [splitter_comma.join(list(rule.values())[0]) for rule in merge_prodg1_list]

            for values, replacement_value in zip(values_to_replace, merged_values):
                df = df.withColumn("PRODG1",
                                   when(col("PRODG1").isin(values), replacement_value).otherwise(col("PRODG1")))

        if merge_product_list is not None and len(merge_product_list) > 0:
            values_to_replace = [list(rule.values())[0] for rule in merge_product_list]
            merged_values = [splitter_comma.join(list(rule.values())[0]) for rule in merge_product_list]

            for values, replacement_value in zip(values_to_replace, merged_values):
                df = df.withColumn("PRODUCT_ID",
                                   when(col("PRODUCT_ID").isin(values), replacement_value).otherwise(col("PRODUCT_ID")))

        if merge_eqp_list is not None and len(merge_eqp_list) > 0:
            values_to_replace = [list(rule.values())[0] for rule in merge_eqp_list]
            merged_values = [splitter_comma.join(list(rule.values())[0]) for rule in merge_eqp_list]

            for values, replacement_value in zip(values_to_replace, merged_values):
                df = df.withColumn("EQP_NAME",
                                   when(col("EQP_NAME").isin(values), replacement_value).otherwise(col("EQP_NAME")))

        if merge_chamber_list is not None and len(merge_chamber_list) > 0:
            values_to_replace = [list(rule.values())[0] for rule in merge_chamber_list]
            merged_values = [splitter_comma.join(list(rule.values())[0]) for rule in merge_chamber_list]

            for values, replacement_value in zip(values_to_replace, merged_values):
                df = df.withColumn("TOOL_NAME",
                                   when(col("TOOL_NAME").isin(values), replacement_value).otherwise(col("TOOL_NAME")))
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
        grps = (df_run.groupBy(grpby_list)
                .agg(countDistinct('WAFER_ID').alias('wafer_count'),
                     countDistinct('WAFER_ID', when(df_run['label'] == 0, 1)).alias('good_num'),
                     countDistinct('WAFER_ID', when(df_run['label'] == 1, 1)).alias('bad_num'))
                .orderBy('bad_num', ascending=False))

        # Handle the case of a single OPER_NO or single TOOL_NAME
        if grps.count() == 1:
            return grps
        else:
            # Filter out groups with no bad wafers
            grps = grps.filter(grps['bad_num'] > 0)
            # Rank the groups based on the number of bad wafers
            window_sep = Window().orderBy(col("bad_num").desc())
            ranked_df = grps.withColumn("rank", rank().over(window_sep))
            # Select the top ten groups and remove the 'rank' column
            grpss = ranked_df.filter(col("rank") <= 10).drop("rank")
            return grpss

    @staticmethod
    def get_data_list(common_res: pyspark.sql.dataframe,
                      grpby_list: List[str],
                      big_or_small: str = 'big') -> List[Dict[str, str]]:
        """
        Get a list of dictionaries for corresponding groups based on commonality analysis.

        :param common_res: Result of commonality analysis.
        :param grpby_list:  List of columns ['PRODG1', 'EQP_NAME', 'OPER_NO', 'PRODUCT_ID', 'TOOL_NAME'] for grouping.
        :param big_or_small: 'big' or 'small'.
        :return: List of dictionaries for corresponding groups.
                Example: [{'OPER_NO': '1F.EEK10', 'PRODUCT_ID': 'AFKN2J01N.0U01'},
                          {'OPER_NO': '1F.EEK10', 'PRODUCT_ID': 'AFKN4X01N.0B01'},
                          {'OPER_NO': '1F.EEK10', 'PRODUCT_ID': 'AFGN1501N.0C02'}]
        """
        assert big_or_small in ['big', 'small'], "Choose only 'big' or 'small'. Please check the spelling."

        # Filter groups based on big or small sample conditions
        if big_or_small == 'big':
            good_bad_grps = common_res.filter("good_num >= 3 AND bad_num >= 3")
        else:
            good_bad_grps = common_res.filter("bad_num >= 1 AND wafer_count >= 2")

        # Order the results and limit to the top 10 groups
        good_bad_grps = good_bad_grps.orderBy(col("bad_num").desc(), col("wafer_count").desc(),
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
    def fit_rf_big_sample(df: pyspark.sql.dataframe, grpby_list: List[str]) -> pyspark.sql.dataframe:
        """
        Fit a RandomForest model on the train data. It is for large sample method(good_wafer_num >= 3 AND bad_wafer_num >= 3)

        Parameters:
        - df: Data for modeling.
        - grpby_list: List of grouping columns.

        Returns:
        - DataFrame: Combined dataframe of roc_auc_score result and feature importance after RandomForest modeling.
        """
        # Dynamically build schema according to the grpby_list
        struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
        struct_fields.extend([StructField("bad_wafer", IntegerType(), True),
                              StructField("roc_auc_score", FloatType(), True),
                              StructField("features", StringType(), True),
                              StructField("importance", FloatType(), True)])
        schema_all = StructType(struct_fields)

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run: pd.DataFrame) -> pd.DataFrame:
            # Pivot the table
            df_pivot = FitModelForUvaData.get_pivot_table(df=df_run, grpby_list=grpby_list)

            # Define independent and dependent variables
            x_train = df_pivot[df_pivot.columns.difference(['WAFER_ID', 'label']).tolist()]
            y_train = df_pivot[['label']]
            if min(x_train.shape) <= 0:
                return pd.DataFrame()

            z_ratio = y_train.value_counts(normalize=True)
            good_ratio = z_ratio[0]
            bad_ratio = z_ratio[1]
            if abs(good_ratio - bad_ratio) > 0.7:
                undersampler = ClusterCentroids(random_state=101)
                x_train, y_train = undersampler.fit_resample(x_train, y_train)

            # Grid search
            pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(random_state=2024))])
            param_grid = {'model__n_estimators': [*range(50, 100, 10)],
                          'model__max_depth': [*range(10, 50, 10)]}
            grid = GridSearchCV(estimator=pipe, scoring='roc_auc', param_grid=param_grid, cv=3, n_jobs=-1)
            grid.fit(x_train.values, y_train.values.ravel())
            roc_auc_score_ = grid.best_score_

            # Feature importance and result summary
            small_importance_res = pd.DataFrame({'features': x_train.columns,
                                                 'importance': grid.best_estimator_.steps[2][1].feature_importances_})

            sample_res_dict = {'bad_wafer': sum(df_pivot['label']),
                               'roc_auc_score': roc_auc_score_}
            sample_res_dict.update({col_: df_run[col_].unique() for col_ in grpby_list})
            small_sample_res = pd.DataFrame(sample_res_dict)
            return pd.concat([small_importance_res, small_sample_res])

        return df.groupby(grpby_list).apply(get_model_result)

    @staticmethod
    def fit_pca_small_sample(df: pyspark.sql.dataframe, grpby_list: List[str]) -> pyspark.sql.dataframe:
        """
        Fit a PCA model on the train data. It is for small sample method (bad_wafer_num >= 1 AND wafer_count >= 2).

        Parameters:
        - df: Data for modeling.
        - grpby_list: List of grouping columns.

        Returns:
        - DataFrame: Combined dataframe of every feature and its importance in each combination of grpby_list after PCA modeling.
        """
        # Dynamically build schema according to the grpby_list
        struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
        struct_fields.extend([StructField("features", StringType(), True),
                              StructField("importance", FloatType(), True),
                              StructField("bad_wafer", IntegerType(), True)])
        schema_all = StructType(struct_fields)

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run: pd.DataFrame) -> pd.DataFrame:
            """
            Perform PCA modeling on the small sample data.

            Parameters:
            - df_run: Subset of data for modeling (pandas.DataFrame).

            Returns:
            - pd.DataFrame: Combined dataframe of every feature and its importance in each combination of grpby_list after PCA modeling.
            """
            df_pivot = FitModelForUvaData.get_pivot_table(df=df_run, grpby_list=grpby_list)
            # Since it is a small sample, make a copy to generate more data for the PCA model
            df_pivot_copy = df_pivot.copy()
            df_pivot_all = pd.concat([df_pivot, df_pivot_copy], axis=0)

            # Define independent variables
            x_train = df_pivot_all[df_pivot_all.columns.difference(['WAFER_ID', 'label']).tolist()]
            if min(x_train.shape) <= 0:
                return pd.DataFrame()

            n_components = min(min(x_train.shape) - 2, 20)
            model = pca(n_components=n_components, verbose=None)
            results = model.fit_transform(x_train)
            res_top = results['topfeat']
            res_top_select = res_top[res_top['type'] == 'best'][['feature', 'loading']]
            res_top_select['importance'] = abs(res_top_select['loading'])
            res_top_select = res_top_select.rename(columns={'feature': 'features'}).drop("loading",
                                                                                         axis=1).drop_duplicates()

            # Add some field information
            res_top_select['bad_wafer'] = sum(df_pivot['label'])
            for col_ in grpby_list:
                res_top_select[col_] = df_run[col_].values[0]
            return res_top_select

        return df.groupby(grpby_list).apply(get_model_result)


class GetFinalResultsForUvaData:
    @staticmethod
    def split_score_big_sample(df: pyspark.sql.dataframe, grpby_list: List[str]) -> pyspark.sql.dataframe:
        """
        Split the ROC AUC scores based on the specified grouping columns.

        Parameters:
        - df: Results after RandomForest modeling.
        - grpby_list: List of grouping columns.

        Returns:
        - DataFrame: ROC AUC scores result with each element in grpby_list as columns.
        """
        struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
        struct_fields.extend([StructField("bad_wafer", IntegerType(), True),
                              StructField("roc_auc_score", FloatType(), True)])
        schema_all = StructType(struct_fields)

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_result(model_results: pd.DataFrame) -> pd.DataFrame:
            select_expr = grpby_list + ['bad_wafer', 'roc_auc_score']
            sample_res = model_results[select_expr].dropna(axis=0)
            sample_res = sample_res[sample_res['roc_auc_score'] > 0.1]
            return sample_res

        return df.groupby(grpby_list).apply(get_result)

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
            feature_importance_res_split_drop = feature_importance_res_split.query("importance > 0").reset_index(
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
        df_merge = df_merge.select(grpby_list + ['parametric_name', 'weight', 'stats']).orderBy('weight',
                                                                                                ascending=False)
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
                              StructField("bad_wafer", IntegerType(), True),
                              StructField("stats", StringType(), True)])
        schema_all = StructType(struct_fields)

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_result(model_results: pd.DataFrame) -> pd.DataFrame:
            feature_importance_table = model_results[['features', 'importance', 'bad_wafer']].dropna(axis=0)
            # Split features
            feature_importance_res_split = GetFinalResultsForUvaData.get_split_feature_importance_table(
                df=feature_importance_table,
                grpby_list=grpby_list)

            # Add a column with all statistical features containing parameters: feature_stats
            feature_stats = GetFinalResultsForUvaData.add_feature_stats(df=feature_importance_res_split,
                                                                        grpby_list=grpby_list)

            # Sum the same parameter in the same combination: feature_importance_groupby
            feature_importance_groupby = (
                feature_importance_res_split.groupby(grpby_list + ['bad_wafer', 'parametric_name', 'step'])[
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
                                       bad_wafer_num: int,
                                       grpby_list: List[str]) -> pyspark.sql.dataframe:
        """
        Get the final modeling results for a small sample.

        Parameters:
        - f_res: Features and importance results (pyspark.sql.dataframe).
        - bad_wafer_num: Total number of bad wafers in the data (int).
        - grpby_list: List of grouping columns (List[str]).

        Returns:
        - DataFrame: Final modeling results with weights and statistics.
        """
        f_res = f_res.withColumn("bad_ratio", col("bad_wafer") / bad_wafer_num)
        df_merge = f_res.withColumn('weight_original', col('importance') * col('bad_ratio'))

        # Normalize weights again
        weight_all = df_merge.agg({"weight_original": "sum"}).collect()[0][0]
        df_merge = df_merge.withColumn("weight", col("weight_original") / weight_all)

        # Select columns and order by weight in descending order
        df_merge = df_merge.select(grpby_list + ['parametric_name', 'weight', 'stats']).orderBy('weight',
                                                                                                ascending=False)
        return df_merge

    @staticmethod
    def add_certain_column(df: pyspark.sql.dataframe, by: str, request_id: str,
                           grpby_list: List[str]) -> pyspark.sql.dataframe:
        """
        Add specific columns to the final modeling results.

        Parameters:
        - df: Final modeling result.
        - by: Grouping column, manually add a column 'add'.
        - request_id: Request ID passed in.
        - grpby_list: List of grouping columns.

        Returns:
        - DataFrame: Final modeling result with specific columns added.
        """
        # Dynamically build schema_all
        struct_fields = [StructField("PRODUCT_ID", StringType(), True),
                         StructField("OPER_NO", StringType(), True),
                         StructField("EQP_NAME", StringType(), True),
                         StructField("PRODG1", StringType(), True),
                         StructField("TOOL_NAME", StringType(), True)]
        struct_fields.extend([StructField("stats", StringType(), True),
                              StructField("parametric_name", StringType(), True),
                              StructField("weight", FloatType(), True),
                              StructField("request_id", StringType(), True),
                              StructField("weight_percent", FloatType(), True),
                              StructField("index_no", IntegerType(), True)])
        schema_all = StructType(struct_fields)

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_result(final_res: pd.DataFrame) -> pd.DataFrame:
            final_res['weight'] = final_res['weight'].astype(float)
            final_res = final_res.query("weight > 0")
            final_res['request_id'] = request_id
            final_res['weight_percent'] = final_res['weight'] * 100
            final_res = final_res.sort_values('weight', ascending=False)
            final_res['index_no'] = [i + 1 for i in range(len(final_res))]
            final_res = final_res.drop('add', axis=1)

            info_list = ['PRODUCT_ID', 'OPER_NO', 'EQP_NAME', 'PRODG1', 'TOOL_NAME']
            for column in info_list:
                if column not in final_res.columns.tolist():
                    final_res[column] = np.nan
            return final_res

        return df.groupby(by).apply(get_result)


class ExertUvaAlgorithm:
    @staticmethod
    def fit_big_data_model(df_run: pyspark.sql.dataframe,
                           data_dict_list_bs: List[Dict[str, str]],
                           grpby_list: List[str],
                           request_id: str) -> Union[str, pyspark.sql.dataframe.DataFrame]:
        """
        Perform random forest model on a large sample of data(good_wafer_num >= 3 AND bad_wafer_num >= 3).

        Parameters:
        - df_run: Preprocessed data after data preprocessing.
        - data_dict_list_bs: List of dictionaries with filtering conditions for big sample.
        - grpby_list: List of grouping columns.
        - request_id: Unique identifier for the request.

        Returns:
        - DataFrame: Dataframe containing the error message or the final modeling results.
        """
        # 1. Get data for modeling with a large sample
        df_run_bs = PreprocessForUvaData.get_train_data(df_run=df_run, data_dict_list=data_dict_list_bs)
        if df_run_bs.isEmpty():
            msg = 'No data of this type in the database!'
            raise RCABaseException(msg)

        # 2. Get the total number of bad wafers
        bad_wafer_num_big_sample = PreprocessForUvaData.get_all_bad_wafer_num(df=df_run_bs)
        if bad_wafer_num_big_sample < 3:
            msg = 'The actual number of BAD_WAFER in the database is less than 3, please provide more BAD_WAFER!'
            raise RCABaseException(msg)

        # 3. Model the selected big sample data
        res = FitModelForUvaData.fit_rf_big_sample(df=df_run_bs, grpby_list=grpby_list)
        if res.isEmpty():
            msg = 'No difference in this data. The output of the algorithm is 0.'
            raise RCABaseException(msg)

        # 4. Integrate the modeling results
        s_res = GetFinalResultsForUvaData.split_score_big_sample(df=res, grpby_list=grpby_list)
        if s_res.isEmpty():
            msg = 'The algorithm has a low running score, no output for now, it is recommended to increase the number of BAD_WAFER'
            raise RCABaseException(msg)

        f_res = GetFinalResultsForUvaData.split_calculate_features_big_sample(df=res, grpby_list=grpby_list)
        if f_res.isEmpty():
            msg = 'Temporary exception in calculating the algorithm result'
            raise RCABaseException(msg)

        model_res_bs = GetFinalResultsForUvaData.get_final_results_big_sample(s_res=s_res, f_res=f_res,
                                                                              grpby_list=grpby_list,
                                                                              bad_wafer_num=bad_wafer_num_big_sample)
        if model_res_bs.isEmpty():
            msg = 'Temporary exception in splicing algorithm results'
            raise RCABaseException(msg)

        # 7. Add specific columns
        final_res_bs = model_res_bs.withColumn('add', lit(0))
        final_res_add_columns = GetFinalResultsForUvaData.add_certain_column(df=final_res_bs, by='add',
                                                                             request_id=request_id,
                                                                             grpby_list=grpby_list)
        if final_res_add_columns.isEmpty():
            msg = 'Temporary exception in adding columns to algorithm results'
            raise RCABaseException(msg)
        else:
            return final_res_add_columns

    @staticmethod
    def fit_small_data_model(df_run: pyspark.sql.dataframe,
                             common_res: pyspark.sql.dataframe,
                             grpby_list: List[str],
                             request_id: str) -> Union[str, pyspark.sql.dataframe.DataFrame]:
        """
        Perform PCA model on a small sample of data (bad_wafer_num >= 1 AND wafer_count >= 2).

        Parameters:
        - df_run: Preprocessed data after data preprocessing.
        - common_res: Common analysis results dataframe.
        - grpby_list: List of grouping columns.
        - request_id: Unique identifier for the request.

        Returns:
        - Union[str, DataFrame]: Error message or the final modeling results dataframe.
        """
        # 1. Get data for modeling with a large sample
        data_dict_list_ss = PreprocessForUvaData.get_data_list(common_res=common_res, grpby_list=grpby_list,
                                                               big_or_small='small')
        if len(data_dict_list_ss) == 0:
            msg = 'The actual number of WAFER in the database under this query condition is 1, unable to analyze'
            raise RCABaseException(msg)

        # 2. Get the data
        df_run_ss = PreprocessForUvaData.get_train_data(df_run=df_run, data_dict_list=data_dict_list_ss)
        if df_run_ss.isEmpty():
            msg = 'No data of this type in the database!'
            raise RCABaseException(msg)

        # 3. Get the total number of bad wafers
        bad_wafer_num_small_sample = PreprocessForUvaData.get_all_bad_wafer_num(df_run_ss)
        if bad_wafer_num_small_sample <= 1:
            msg = 'The actual number of BAD_WAFER in the database under this query condition is 1, please provide more BAD_WAFER!'
            raise RCABaseException(msg)

        # 4. Model the selected big sample data
        res = FitModelForUvaData.fit_pca_small_sample(df=df_run_ss, grpby_list=grpby_list)
        if res.isEmpty():
            msg = 'No difference in this data. The output of the algorithm is 0.'
            raise RCABaseException(msg)

        f_res = GetFinalResultsForUvaData.split_calculate_features_small_sample(df=res, grpby_list=grpby_list)
        if f_res.isEmpty():
            msg = 'Temporary exception in calculating the algorithm result'
            raise RCABaseException(msg)

        model_res_ss = GetFinalResultsForUvaData.get_final_results_small_sample(f_res=f_res,
                                                                                bad_wafer_num=bad_wafer_num_small_sample,
                                                                                grpby_list=grpby_list)
        if model_res_ss.isEmpty():
            msg = 'Temporary exception in splicing algorithm results'
            raise RCABaseException(msg)

        final_res_ss = model_res_ss.withColumn('add', lit(0))
        final_res_add_columns = GetFinalResultsForUvaData.add_certain_column(df=final_res_ss, by='add',
                                                                             request_id=request_id,
                                                                             grpby_list=grpby_list)
        if final_res_add_columns.isEmpty():
            msg = 'Temporary exception in adding columns to algorithm results'
            raise RCABaseException(msg)
        else:
            return final_res_add_columns

    # @staticmethod
    # def validate_grpby_list(grpby_list: List[str]) -> bool:
    #     valid_fields = {'PRODUCT_ID', 'OPER_NO', 'EQP_NAME', 'PRODG1', 'TOOL_NAME'}
    #     for field in grpby_list:
    #         if field not in valid_fields:
    #             return False
    #     return True
    #
    # @staticmethod
    # def get_some_info_test(df: pd.DataFrame) -> Tuple[dict, str, list, Optional[list]]:
    #     """
    #     Extracts information from a DataFrame containing request information.
    #
    #     Parameters:
    #     - df: DataFrame containing request information.
    #
    #     Returns:
    #     - Tuple containing parsed information: (parse_dict, request_id, grpby_list, merge_operno)
    #     """
    #     request_id = df["requestId"].values[0]
    #     request_params = df["requestParam"].values[0]
    #     request_params = request_params.replace('\'', '\"')
    #
    #     parse_dict = json.loads(request_params)
    #     grpby_list = parse_dict['groupByList']
    #     if not ExertUvaAlgorithm.validate_grpby_list(grpby_list):
    #         raise ValueError(
    #             "Invalid grpby_list. Must include at least one of 'PRODUCT_ID', 'OPER_NO', 'EQP_NAME', 'PRODG1', 'TOOL_NAME'.")
    #
    #     try:
    #         merge_operno = list(parse_dict['mergeOperno'])
    #     except KeyError:
    #         merge_operno = None
    #
    #     return parse_dict, request_id, grpby_list, merge_operno
    #
    #
    # @staticmethod
    # def run(df_info: pd.DataFrame, properties_config, sparkSession):
    #
    #     parse_dict, request_id, grpby_list, merge_operno, merge_prodg1, merge_product, merge_eqp, merge_chamber = ExertUvaAlgorithm.get_some_info_test(df_info)
    #     print("parse_dict:")
    #     print(parse_dict)
    #     print("request_id:")
    #     print(request_id)
    #     print("grpby_list:")
    #     print(grpby_list)
    #     print("merge_operno:")
    #     print(merge_operno)
    #     # todo1: 将parse_dict传给解析SQL的函数, 拼接SQL获取数据
    #     query_sql = build_uva_query(parse_dict, properties_config)
    #     doris_spark_df = read_jdbc_executor.read(sparkSession, query_sql, properties_config)
    #
    #     # 1. Station merge and data preprocessing
    #     df_merge_operno = PreprocessForUvaData.integrate_operno(df=doris_spark_df, merge_operno_list=merge_operno)
    #     m, n = df_merge_operno.count(), len(df_merge_operno.columns)
    #     print(f"Merged data: ({m}, {n})")
    #     if df_merge_operno.count() == 0:
    #         msg = 'Station merge exception!！'
    #         raise RCABaseException(msg)
    #
    #     df_run = PreprocessForUvaData.pre_process(doris_spark_df)
    #     m, n = df_run.count(), len(df_run.columns)
    #     print(f"Preprocessed data: ({m}, {n})")
    #     if df_run.count() == 0:
    #         msg = 'No data in the database under this condition！'
    #         raise RCABaseException(msg)
    #
    #     # 2. Commonality analysis
    #     common_res = PreprocessForUvaData.commonality_analysis(df_run=df_run, grpby_list=grpby_list)
    #     common_res.show()
    #     if common_res.count() == 0:
    #         msg = 'Commonality analysis result exception!'
    #         raise RCABaseException(msg)
    #
    #     data_dict_list_bs = PreprocessForUvaData.get_data_list(common_res=common_res, grpby_list=grpby_list,
    #                                                            big_or_small='big')
    #     print("data_dict_list_bs:", data_dict_list_bs)
    #     print("len(data_dict_list_bs):", len(data_dict_list_bs))
    #
    #     if len(data_dict_list_bs) != 0:
    #         print("****************Call Big Sample Algorithm****************")
    #         result = ExertUvaAlgorithm.fit_big_data_model(df_run=df_run, data_dict_list_bs=data_dict_list_bs,
    #                                                       grpby_list=grpby_list, request_id=request_id)
    #     else:
    #         print("****************Call Small Sample Algorithm****************")
    #         result = ExertUvaAlgorithm.fit_small_data_model(df_run=df_run, common_res=common_res,
    #                                                         grpby_list=grpby_list, request_id=request_id)
    #
    #     return result


if __name__ == '__main__':
    import warnings
    import os
    import pandas as pd
    import pyspark.pandas as ps
    from pyspark.sql import SparkSession

    os.environ['PYSPARK_PYTHON'] = '/usr/local/python-3.9.13/bin/python3'
    warnings.filterwarnings('ignore')

    spark_session = SparkSession.builder \
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

    df_pandas = pd.read_csv("D:/Jupyterfiles/晶合MVAFDC_general开发/MVAanlysisDevelop/uva_algorithm/CASE1_DATA/DWD_POC_CASE_FD_UVA_DATA_CASE1_PROCESSED1.csv")
    # df_pandas = pd.read_csv("D:/Jupyterfiles/晶合MVAFDC_general开发/MVAanlysisDevelop/uva_algorithm/small_samples_data/small2_labeled.csv")
    print(df_pandas.shape)
    doris_spark_df = ps.from_pandas(df_pandas).to_spark()
    print(doris_spark_df.count())

    request_id = '267'
    grpby_list = ['PRODG1', 'OPER_NO', 'TOOL_NAME']
    merge_operno = []
    merge_prodg1 = []
    merge_product = []
    merge_eqp = []
    merge_chamber = []

    # 1. Station merge and data preprocessing
    df_integrate_columns = PreprocessForUvaData.integrate_columns(df=doris_spark_df,
                                                                  merge_operno_list=merge_operno,
                                                                  merge_prodg1_list=merge_prodg1,
                                                                  merge_product_list=merge_product,
                                                                  merge_eqp_list=merge_eqp,
                                                                  merge_chamber_list=merge_chamber)
    m, n = df_integrate_columns.count(), len(df_integrate_columns.columns)
    print(f"Merged data: ({m}, {n})")
    if df_integrate_columns.count() == 0:
        msg = 'Merge columns exception!'
        raise RCABaseException(msg)

    df_run = PreprocessForUvaData.pre_process(df_integrate_columns)
    m, n = df_run.count(), len(df_run.columns)
    print(f"Preprocessed data: ({m}, {n})")
    if df_run.count() == 0:
        msg = 'No data in the database under this condition!'
        raise RCABaseException(msg)

    # 2. Commonality analysis
    common_res = PreprocessForUvaData.commonality_analysis(df_run=df_run, grpby_list=grpby_list)
    common_res.show()
    if common_res.count() == 0:
        msg = 'No bad or good wafer in this data. Unable to do the commonality analysis.'
        raise RCABaseException(msg)

    data_dict_list_bs = PreprocessForUvaData.get_data_list(common_res=common_res, grpby_list=grpby_list,
                                                           big_or_small='big')
    print("data_dict_list_bs:", data_dict_list_bs)
    print("len(data_dict_list_bs):", len(data_dict_list_bs))

    if len(data_dict_list_bs) != 0:
        print("****************Call Big Sample Algorithm****************")
        result = ExertUvaAlgorithm.fit_big_data_model(df_run=df_run, data_dict_list_bs=data_dict_list_bs,
                                                      grpby_list=grpby_list, request_id=request_id)
    else:
        print("****************Call Small Sample Algorithm****************")
        result = ExertUvaAlgorithm.fit_small_data_model(df_run=df_run, common_res=common_res,
                                                        grpby_list=grpby_list, request_id=request_id)
    print("--result：")
    result.show()
