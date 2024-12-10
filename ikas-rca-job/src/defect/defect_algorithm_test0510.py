import numpy as np
import pyspark
import pandas as pd
from imblearn.under_sampling import ClusterCentroids
from pca import pca
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
from pyspark.sql.functions import pandas_udf, PandasUDFType, lit, col, countDistinct, when, monotonically_increasing_id
from typing import List, Dict, Union


import warnings
import os
import pyspark.pandas as ps
from pyspark.sql import SparkSession

# os.environ['PYSPARK_PYTHON'] = '/usr/local/python-3.9.13/bin/python3'
# warnings.filterwarnings('ignore')
#
# spark = SparkSession.builder \
#     .appName("pandas_udf") \
#     .config('spark.sql.session.timeZone', 'Asia/Shanghai') \
#     .config("spark.scheduler.mode", "FAIR") \
#     .config('spark.driver.memory', '8g') \
#     .config('spark.driver.cores', '12') \
#     .config('spark.executor.memory', '8g') \
#     .config('spark.executor.cores', '12') \
#     .config('spark.cores.max', '12') \
#     .config('spark.driver.host', '192.168.22.28') \
#     .master("spark://192.168.12.47:7077,192.168.12.48:7077") \
#     .getOrCreate()

df_run_pandas = pd.read_csv('D:/Jupyterfiles/晶合MVAFDC_general开发/MVAanlysisDevelop/defect_algorithm/defect_by_wafer_labeled_6.csv')
df_run_pandas['INSPECTION_TIME'] = pd.to_datetime(df_run_pandas['INSPECTION_TIME'])
# print("OPE_NO:", df_run_pandas['OPE_NO'].unique())
# print("PRODG1:", df_run_pandas['PRODG1'].unique())  # ['KLKL' 'DFDF']
# print("PRODUCT_ID:", df_run_pandas['PRODUCT_ID'].unique())  # ['AJ.KLL' 'Ab.KMM01' 'VB.JJJJJJJ01' 'VB.JJJJJJJ1000']

grpby_list = ['PRODG1', 'PRODUCT_ID']
prodg = 'KLKL'
product_id = 'Ab.KMM01'
df_run_select1 = df_run_pandas.query(f"PRODG1 == '{prodg}' & PRODUCT_ID == '{product_id}'")
print(df_run_select1.shape)
print(df_run_select1['label'].value_counts())
print(df_run_select1['WAFER_ID'].nunique())

df_run_select1 = df_run_select1.drop_duplicates().dropna(subset=["RANDOM_DEFECTS",	"DEFECTS", "ADDER_DEFECTS",	"CLUSTERS",	"ADDER_RANDOM_DEFECTS",	"ADDER_CLUSTERS"],
                                                         how='any')
print("删除缺失值和重复值后：")
print(df_run_select1.shape)
print(df_run_select1['label'].value_counts())
print(df_run_select1['WAFER_ID'].nunique())

idx = df_run_select1.groupby(['WAFER_ID', 'label'] + grpby_list)['INSPECTION_TIME'].idxmax()
df_sort = df_run_select1.loc[idx, :]
print(df_sort.shape)
print(df_sort['label'].value_counts())
print(df_sort['WAFER_ID'].nunique())


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


def get_model_result(df_run, model='rf'):
    x_train = df_run[["RANDOM_DEFECTS",	"DEFECTS", "ADDER_DEFECTS",	"CLUSTERS",	"ADDER_RANDOM_DEFECTS",	"ADDER_CLUSTERS"]]
    y_train = df_run[['label']]

    z_ratio = y_train.value_counts(normalize=True)
    good_ratio = z_ratio[0]
    bad_ratio = z_ratio[1]
    if abs(good_ratio - bad_ratio) > 0.7:
        undersampler = ClusterCentroids(random_state=1024)
        x_train, y_train = undersampler.fit_resample(x_train, y_train)

    pipe, param_grid = get_pipe_params(model=model)
    try:
        grid = GridSearchCV(estimator=pipe, scoring='roc_auc', param_grid=param_grid, cv=3, n_jobs=-1)
        grid.fit(x_train.values, y_train.values.ravel())
    except ValueError:
        return pd.DataFrame()

    best_est = grid.best_estimator_.steps[-1][-1]
    if hasattr(best_est, 'feature_importances_'):
        small_importance_res = pd.DataFrame({'features': x_train.columns, 'importance': best_est.feature_importances_})
    else:
        small_importance_res = pd.DataFrame({'features': x_train.columns, 'importance': abs(best_est.coef_.ravel())})

    # star_features = ["RANDOM_DEFECTS",	"DEFECTS",	"CLUSTERS"]
    # star_features_importance = sum(small_importance_res[small_importance_res['features'].isin(star_features)]['importance'])
    # least_features_importance = sum(small_importance_res[~small_importance_res['features'].isin(star_features)]['importance'])

    sample_res_dict = {'bad_wafer': sum(df_run['label']),
                       'roc_auc_score': 0.0 if np.isnan(grid.best_score_) else grid.best_score_,
                       'algorithm_satisfied': 'TRUE',
                       'x_train_shape': str(x_train.shape)}
    sample_res_dict.update({col_: df_run[col_].values[0] for col_ in grpby_list})
    small_sample_res = pd.DataFrame(sample_res_dict, index=[0])
    # res_top_select = pd.concat([small_importance_res, small_sample_res])
    return small_importance_res, small_sample_res


small_importance_res_, small_sample_res_ = get_model_result(df_run=df_sort, model='rf')
print(small_importance_res_)
print(small_sample_res_)









