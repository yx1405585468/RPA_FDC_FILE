import numpy as np
import pandas as pd
import pyspark.sql.dataframe
from typing import List, Dict
from pyspark.sql.functions import col
from src.exceptions.rca_base_exception import RCABaseException
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, levene
from pyspark.sql.types import StringType, FloatType, StructType, StructField
from pyspark.sql.functions import countDistinct, when, lit, pandas_udf, PandasUDFType, monotonically_increasing_id


class PreprocessForDefectData:
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
        df_merged = PreprocessForDefectData.integrate_single_column(df, merge_operno_list, 'OPER_NO')
        df_merged = PreprocessForDefectData.integrate_single_column(df_merged, merge_prodg1_list, 'PRODG1')
        df_merged = PreprocessForDefectData.integrate_single_column(df_merged, merge_product_list, 'PRODUCT_ID')
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
    def pre_process(df: pyspark.sql.dataframe, columns_list: list[str]) -> pyspark.sql.dataframe:
        """
        Preprocess the data extracted from the database for a specific CASE.
        :param df: Data for a specific CASE retrieved from the database.
        :param columns_list: columns need to be selected.
        :return: Preprocessed data with relevant columns and filters applied.
        """
        try:
            df = df.select(columns_list)
        except Exception as e:
            print(str(e))
        finally:
            df_run = df.dropDuplicates()
            df_run = df_run.dropna(subset=['RANDOM_DEFECTS'])
            return df_run

    @staticmethod
    def commonality_analysis(df_run: pyspark.sql.dataframe, grpby_list: List[str]) -> pyspark.sql.dataframe:
        """
        暂时用不到, debug数据时可以用来查看数据情况
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
        return grps


class FitModelForDefectData:

    def __init__(self, df: pyspark.sql.dataframe, grpby_list: List[str]):
        self.df = df
        self.grpby_list = grpby_list

    @staticmethod
    def fit_stats_model(df: pyspark.sql.dataframe, grpby_list: List[str]) -> pyspark.sql.dataframe:
        """
        Fit statistical model to the defect data.

        Args:
        - df (DataFrame): Input DataFrame containing defect data.
        - grpby_list (list): List of columns to group by.

        Returns:
        - DataFrame: Result DataFrame containing statistical importance.
        """
        struct_fields = [StructField(col_, StringType(), True) for col_ in grpby_list]
        struct_fields.extend([StructField("importance", FloatType(), True)])
        schema_all = StructType(struct_fields)

        @pandas_udf(returnType=schema_all, functionType=PandasUDFType.GROUPED_MAP)
        def get_model_result(df_run):
            good_wafers = df_run.loc[df_run['label'] == 0, 'RANDOM_DEFECTS'].tolist()
            bad_wafers = df_run.loc[df_run['label'] == 1, 'RANDOM_DEFECTS'].tolist()

            if len(good_wafers) == 0 or len(bad_wafers) == 0:
                return pd.DataFrame()

            p_shapiro_good, p_shapiro_bad, p_levene = FitModelForDefectData.do_normality_tests(good_wafers, bad_wafers)
            statistic, p_value = FitModelForDefectData.get_difference_results(good_wafers, bad_wafers, p_shapiro_good,
                                                                              p_shapiro_bad, p_levene)

            importance_dict = {'importance': 1 - p_value}
            importance_dict.update({col_: df_run[col_].values[0] for col_ in grpby_list})
            importance_res = pd.DataFrame(importance_dict, index=[0])
            return importance_res

        return df.groupby(grpby_list).apply(get_model_result)

    @staticmethod
    def extend_wafers(good_wafers: list, bad_wafers: list):
        """
        Extend the lists of wafers with zeros if their length is less than 3.

        Args:
        - good_wafers (list): List of wafers classified as good.
        - bad_wafers (list): List of wafers classified as bad.

        Returns: good_wafers (list), bad_wafers (list)
        """
        if len(good_wafers) < 3:
            n = len(good_wafers)
            g_mean = np.mean(good_wafers)
            good_wafers.extend([g_mean] * (3 - n))

        if len(bad_wafers) < 3:
            n = len(bad_wafers)
            b_mean = np.mean(bad_wafers)
            bad_wafers.extend([b_mean] * (3 - n))
        return good_wafers, bad_wafers

    @staticmethod
    def do_normality_tests(good_wafers: list, bad_wafers: list):
        """
        Perform normality tests on the lists of good and bad wafers.

        Args:
        - good_wafers (list): List of wafers classified as good.
        - bad_wafers (list): List of wafers classified as bad.

        Returns: p-values from Shapiro-Wilk test for good and bad wafers, and p-value from Levene's test(All float).
        """

        good_wafers, bad_wafers = FitModelForDefectData.extend_wafers(good_wafers=good_wafers, bad_wafers=bad_wafers)

        # Shapiro-Wilk test: Normality Assumption
        _, p_shapiro_good = shapiro(good_wafers)
        _, p_shapiro_bad = shapiro(bad_wafers)

        # Levene's test: Homogeneity of Variance Assumption
        _, p_levene = levene(good_wafers, bad_wafers)
        return p_shapiro_good, p_shapiro_bad, p_levene

    @staticmethod
    def get_difference_results(good_wafers: list, bad_wafers: list,
                               p_shapiro_good: float, p_shapiro_bad: float,
                               p_levene: float, alpha: float = 0.05):
        """
        Compute statistical difference between good and bad wafers.

        Args:
        - good_wafers (list): List of wafers classified as good.
        - bad_wafers (list): List of wafers classified as bad.
        - p_shapiro_good (float): p-value from Shapiro-Wilk test for good wafers.
        - p_shapiro_bad (float): p-value from Shapiro-Wilk test for bad wafers.
        - p_levene (float): p-value from Levene's test.
        - alpha (float): Significance level.

        Returns: test statistic and p-value(All float).
        """
        good_wafers, bad_wafers = FitModelForDefectData.extend_wafers(good_wafers=good_wafers, bad_wafers=bad_wafers)

        if p_shapiro_good > alpha and p_shapiro_bad > alpha and p_levene > alpha:
            statistic, p_value = ttest_ind(good_wafers, bad_wafers, equal_var=True)

        elif p_shapiro_good > alpha > p_levene and p_shapiro_bad > alpha:
            statistic, p_value = ttest_ind(good_wafers, bad_wafers, equal_var=False)

        else:
            statistic, p_value = mannwhitneyu(good_wafers, bad_wafers)
        return statistic, p_value

    def run(self) -> pyspark.sql.dataframe:
        """
        Run the statistical model fitting.

        Returns:
        - DataFrame: Result DataFrame containing statistical importance.
        """
        res_defect = self.fit_stats_model(df=self.df, grpby_list=self.grpby_list)
        return res_defect


class GetFinalResultsForDefect:

    def __init__(self, df: pyspark.sql.dataframe, request_id: str):
        self.df = df
        self.request_id = request_id

    @staticmethod
    def get_final_results(df: pyspark.sql.dataframe) -> pyspark.sql.dataframe:
        """
        Normalize the importance scores and return the final results.

        Args:
        - df (DataFrame): Input DataFrame containing importance scores.

        Returns:
        - DataFrame: DataFrame with normalized weights.
        """
        # Normalize again
        weight_all = df.agg({"importance": "sum"}).collect()[0][0]
        df_merge = df.withColumn("weight", col("importance") / weight_all).drop("importance")
        return df_merge

    @staticmethod
    def add_certain_column(df: pyspark.sql.dataframe, request_id: str) -> pyspark.sql.dataframe:
        """
         Add certain columns like weight percentage, request ID, and index number.

         Args:
         - df (DataFrame): Input DataFrame.
         - request_id (str): Identifier for the request.

         Returns:
         - DataFrame: DataFrame with added columns.
         """
        df = df.filter('weight > 0')
        df = df.orderBy('weight', ascending=False)
        df = (df.withColumn('weight_percent', col('weight') * 100)
              .withColumn('request_id', lit(request_id))
              .withColumn('index_no', monotonically_increasing_id() + 1))

        info_list = ['PRODUCT_ID', 'OPER_NO', 'PRODG1']
        for column in info_list:
            if column not in df.columns:
                df = df.withColumn(column, lit(None).cast(StringType()))
        return df

    def run(self) -> pyspark.sql.dataframe:
        """
        Run the process to get final results.

        Returns:
        - DataFrame: Final DataFrame with added columns.
        """
        res = self.get_final_results(self.df)
        final_res = self.add_certain_column(res, self.request_id)
        return final_res


class ExertDefectAlgorithm:
    @staticmethod
    def fit_defect_model(df: pyspark.sql.dataframe,
                         grpby_list: List[str],
                         merge_operno_list: List[Dict[str, List[str]]],
                         merge_prodg1_list: List[Dict[str, List[str]]],
                         merge_product_list: List[Dict[str, List[str]]],
                         request_id: str,
                         columns_list=None):
        """
        Fit the defect model and get final results.

        Args:
        - df (DataFrame): Input DataFrame containing defect data.
        - grpby_list (List[str]): List of columns to group by.
        - merge_operno_list (List[Dict[str, List[str]]]): List of dictionaries mapping columns to merge for 'OPER_NO'.
        - merge_prodg1_list (List[Dict[str, List[str]]]): List of dictionaries mapping columns to merge for 'PRODG1'.
        - merge_product_list (List[Dict[str, List[str]]]): List of dictionaries mapping columns to merge for 'PRODUCT_ID'.
        - request_id (str): Request ID for the data.
        - columns_list (List[str]): List of columns to consider.

        Returns:
        - DataFrame: Final result DataFrame.

        Raises:
        - RCABaseException: If any exceptions occur during processing.
        """
        if grpby_list is None or len(grpby_list) == 0:
            grpby_list = ['OPER_NO']

        if columns_list is None or len(columns_list) == 0:
            columns_list = list(set(grpby_list + ['WAFER_ID', 'PRODG1', 'PRODUCT_ID', 'OPER_NO', 'LOT_ID',
                                                  'RECIPE_ID', 'RANDOM_DEFECTS', 'INSPECTION_TIME', 'label']))

        df_integrate_columns = PreprocessForDefectData.integrate_columns(df=df,
                                                                         merge_operno_list=merge_operno_list,
                                                                         merge_prodg1_list=merge_prodg1_list,
                                                                         merge_product_list=merge_product_list)
        if df_integrate_columns.isEmpty():
            msg = 'Merge columns exception!'
            raise RCABaseException(msg)

        df_run = PreprocessForDefectData.pre_process(df_integrate_columns, columns_list=columns_list)
        if df_run.isEmpty():
            msg = 'No data in the database under this condition!'
            raise RCABaseException(msg)

        res = FitModelForDefectData(df=df_run, grpby_list=grpby_list).run()
        if res.isEmpty():
            msg = 'No difference in this data. The output of the algorithm is 0.'
            raise RCABaseException(msg)

        final_res = GetFinalResultsForDefect(df=res, request_id=request_id).run()
        if final_res.isEmpty():
            msg = 'Results are empty.'
            raise RCABaseException(msg)
        return final_res



if __name__ == "__main__":
    import warnings
    import os
    import pyspark.pandas as ps
    from pyspark.sql import SparkSession

    os.environ['PYSPARK_PYTHON'] = '/usr/local/python-3.9.13/bin/python3'
    warnings.filterwarnings('ignore')

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

    df_run_pandas = pd.read_csv('C:/Users/yang.wenjun/Desktop/晶合FDC-freedemo资料/df_defect.csv')
    df1_ = ps.from_pandas(df_run_pandas).to_spark()
    print(df1_.count())

    param_dict = {
        'df': df1_,
        'grpby_list': ['OPER_NO', 'RECIPE_ID'],
        'merge_operno_list': None,
        'merge_prodg1_list': None,
        'merge_product_list': None,
        'request_id': 'defect',
    }

    # param_dict = {
    #     'df': df1_,
    #     'grpby_list': [],
    #     'merge_operno_list': [
    #         {'oper_oper': ['5FP14', '5FP15']}
    #     ],
    #     'merge_prodg1_list': [],
    #     'merge_product_list': [],
    #     'request_id': 'defect',
    # }

    final_res_ = ExertDefectAlgorithm.fit_defect_model(**param_dict)
    final_res_.show()
