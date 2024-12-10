import pandas as pd
import pyspark
import pyspark.pandas as ps
from pca import pca
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType, FloatType


class FitInlineModel:
    def __init__(self,
                 df: pyspark.sql.dataframe,
                 by: list[str],
                 model: str,
                 columns_to_process: list[str],
                 missing_value_threshold: float):
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
        df_specific = FitInlineModel.process_missing_values(df, columns_to_process, missing_value_threshold)
        index_list = ['WAFER_ID', 'label']
        values_list = df_specific.columns.difference(['WAFER_ID', 'OPE_NO', 'INLINE_PARAMETER_ID', 'SITE_COUNT', 'label'])
        pivot_result = df_specific.pivot_table(index=index_list,
                                               columns=['OPE_NO', 'INLINE_PARAMETER_ID'],
                                               values=values_list)
        pivot_result.columns = pivot_result.columns.map('#'.join)
        pivot_result = FitInlineModel.process_missing_values(pivot_result, pivot_result.columns, missing_value_threshold)
        pivot_result = pivot_result.reset_index(drop=False)
        # 删除完全相同的列
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
            pivot_result = FitInlineModel.get_pivot_table(df=df_run,
                                                          columns_to_process=columns_to_process_,
                                                          missing_value_threshold=missing_value_threshold_)
            # 定义自变量
            x_train = pivot_result[pivot_result.columns.difference(['WAFER_ID', 'label']).tolist()]
            if x_train.shape[1] > 1:
                n_components = min(min(x_train.shape) - 1, 5)

                model = pca(n_components=n_components, verbose=None)
                results = model.fit_transform(x_train)
                res_top = results['topfeat']
                res_top_select = res_top[res_top['type'] == 'best'][['feature', 'loading']]
                res_top_select['importance'] = abs(res_top_select['loading'])
                res_top_select = res_top_select.rename(columns={'feature': 'features'}).drop("loading", axis=1).drop_duplicates()
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
            pivot_result = FitInlineModel.get_pivot_table(df=df_run,
                                                          columns_to_process=columns_to_process_,
                                                          missing_value_threshold=missing_value_threshold_)
            x_train = pivot_result[pivot_result.columns.difference(['WAFER_ID', 'label']).tolist()]
            y_train = pivot_result[['label']]
            if x_train.shape[1] > 1 and y_train['label'].nunique() > 1:
                pipe = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
                    ('scaler', StandardScaler()),
                    ('model', RandomForestClassifier())])
                param_grid = {'model__n_estimators': [*range(10, 60, 10)],
                              'model__max_depth': [*range(5, 50, 10)],
                              'model__min_samples_split': [2, 5],
                              'model__min_samples_leaf': [1, 3]}
                grid = GridSearchCV(estimator=pipe, scoring='roc_auc', param_grid=param_grid, cv=2, n_jobs=-1)
                grid.fit(x_train.values, y_train.values.ravel())
                roc_auc_score_ = grid.best_score_
                if roc_auc_score_ >= 0.6:
                    small_importance_res = pd.DataFrame({'features': x_train.columns,
                                                         'importance': grid.best_estimator_.steps[2][1].feature_importances_})
                    return small_importance_res
                else:
                    small_importance_res = pd.DataFrame({"features": "STATS#OPE#PARAM", "importance": -100}, index=[0])
                    return small_importance_res
            else:
                small_importance_res = pd.DataFrame({"features": "STATS#OPE#PARAM", "importance": -100}, index=[0])
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
    df_pandas = pd.read_csv(filepath_or_buffer="df_run_.csv", index_col=0)
    df_spark = ps.from_pandas(df_pandas).to_spark()
    num_rows = df_spark.count()
    num_columns = len(df_spark.columns)
    print(f"df_spark shape: ({num_rows}, {num_columns})")

    # 2. 训练模型
    fim = FitInlineModel(df=df_spark,
                         by=['OPE_NO'],
                         model='rf',
                         columns_to_process=['AVERAGE', 'MAX_VAL', 'MEDIAN', 'MIN_VAL', 'STD_DEV', 'PERCENTILE_25',
                                             'PERCENTILE_75'],
                         missing_value_threshold=0.6)
    res_ = fim.run()
    print(res_.count())
    res_.show()
    # res_.toPandas().to_csv("inline_model_by_wafer_res1.csv", index=False)