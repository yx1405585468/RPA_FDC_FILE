# -*- coding: utf-8 -*-
"""
关系检测算法：
输入的两组一维异源数据， 检测是否存在如下关系：
    1. 线性关系： y= kx +b
    2. 二次项关系： y = ax^2 +b x +c
    3. 根号关系：y = \sqrt{x}  《===》 y^2 = kx +b
    4. 不满足以上关系，则返回weight 0.
"""

from typing import Callable
from sklearn.metrics import r2_score
import pandas as pd
from functools import partial
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

def check_linear_relation(x: pd.Series, y: pd.Series) -> np.ndarray:
    """
    检查线性关系：y = kx + b

    参数:
        x (array): 数据集1（自变量）
        y (array): 数据集2（因变量）

    返回值:
        bool: 如果数据满足线性关系则返回True，否则False
    """
    fn = np.poly1d(np.polyfit(x, y, 1))

    return fn(x)


def check_quadratic_relation(x: pd.Series, y: pd.Series) -> np.ndarray:
    """
    检查二次关系：y = ax^2 + bx + c

    参数:
        x (array): 数据集1（自变量）
        y (array): 数据集2（因变量）

    返回值:
        bool: 如果数据满足二次关系则返回True，否则False
    """

    fn = np.poly1d(np.polyfit(x, y, 2))

    return fn(x)


def check_sqrt_relation(x: pd.Series, y: pd.Series) -> np.ndarray:
    """
    检查根号关系：y = sqrt(x)  <==> y^2 = kx +b

    参数:
        x (array): 数据集1（自变量）
        y (array): 数据集2（因变量）

    返回值:
        bool: 如果数据满足根号关系则返回True，否则False
    """
    fn = np.poly1d(np.polyfit(x, y ** 2, 1))

    return fn(x)


# # 列举需要检测关系的函数字典
DETECT_CORRELATION_FUNC_MAP: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "linear": check_linear_relation,  # 线性关系
    "quadratic": check_quadratic_relation,  # 二次项关系
    "sqrt": check_sqrt_relation  # 根号关系
}


def get_cv_fold(num_samples:int) -> int:
    """
    根据样本数量，计算交叉验证的折数
    """
    if num_samples == 3:
        return 1
    elif num_samples == 4 or num_samples == 5:
        # 3 个样本，但是，验证集r2计算需要两个样本
        return 2
    elif num_samples <= 21:
        return 3
    else:
        return 5


def check_linear_relationship_with_ml(X, y, fit_intercept=True,is_cv=True):
    """
    测试X和y之间是否存在线性关系
    """

    model = LinearRegression(fit_intercept=fit_intercept)
    if is_cv:
        cv = get_cv_fold(num_samples=len(y))
        model = LinearRegression(fit_intercept=fit_intercept)
        if cv > 1:
        # logger.info(f'cv fold: {cv}')

            scores = cross_val_score(model, X.reshape(-1, 1), y, cv=cv, scoring='r2')  # 假设X是一维数组
            mean_score = np.mean(scores)
        else:
            # cv ==1,只有三个样本，
            model.fit(X.reshape(-1, 1)[:1], y[:1])
            mean_score = r2_score(y[1:], model.predict(X.reshape(-1, 1)[1:]))
    else:
        model.fit(X.reshape(-1, 1), y)
        mean_score = r2_score(y, model.predict(X.reshape(-1, 1)))
    return mean_score  # 假设R^2大于0.8表示存在线性关系


def check_quadratic_relationship_with_ml(X, y, fit_intercept=True, is_cv=True):
    """
    测试X和y之间是否存在平方关系
    """

    # logger.info(f"num_samples : {len(y)} cv fold:{cv}")
    model = make_pipeline(PolynomialFeatures(2), LinearRegression(fit_intercept=fit_intercept))  # 平方特征
    if is_cv:
        cv = get_cv_fold(num_samples=len(y))
        if cv >1:
            scores = cross_val_score(model, X.reshape(-1, 1), y, cv=cv, scoring='r2')
            mean_score = np.mean(scores)
        else:
            # cv ==1,只有三个样本，
            model.fit(X.reshape(-1, 1)[:1], y[:1])
            mean_score = r2_score(y[1:], model.predict(X.reshape(-1, 1)[1:]))
    else:
        model.fit(X.reshape(-1, 1), y)
        mean_score = r2_score(y, model.predict(X.reshape(-1, 1)))
    return mean_score  # 假设R^2大于0.8表示存在平方关系


def check_sqrt_relationship_with_ml(X, y, fit_intercept=True, is_cv=True):
    """
    测试X和y之间是否存在根号关系
    注意：在实际情况下，根号关系可能不是直接测试，而是通过变换数据来间接测试。

    y = sqrt(kx+b  ) => sqrt(y) =kx+b
    """
    if y.min() < 0:
        # 存在负数，无法做根号运算
        return 0.0
    return check_linear_relationship_with_ml(X, np.sqrt(y), is_cv=is_cv, fit_intercept=fit_intercept)


DETECT_CORRELATION_ML_FUNC_MAP: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "linear": check_linear_relationship_with_ml,  # 线性关系
    "quadratic": check_quadratic_relationship_with_ml,  # 二次项关系
    "sqrt": check_sqrt_relationship_with_ml,  #  # 根号关系
    "linear_no_intercept": partial(check_linear_relationship_with_ml,fit_intercept=False),  # 线性关系
    "quadratic_no_intercept": partial(check_quadratic_relationship_with_ml, fit_intercept=False) ,# 二次项关系
    "sqrt_no_intercept": partial(check_sqrt_relationship_with_ml,fit_intercept=False),  #  # 根号关系
}

R2_THRESHOLD = 0.0  # R2 阈值，超过阈值，返回r2,没超过返回0.

# to do:后续会继续优化
"""
0. 数据预处理: X, Y 存在缺失值， 只取x,y都有的。 考虑如果xy都有值的只有1对情况下，结果可信吗。 
1. 判断关系是DETECT_CORRELATION_FUNC_MAP中几个关系中的那一个，给出得分，如果都不是，置为0.

"""
import logging

# 设置日志级别和输出
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 使用多项式拟合的方法比较两列的关系检测算法
def algo_detect_relation_old(x: pd.Series, y: pd.Series, r2_threshold: float) -> float:
    """
    检测两组异源数据是否存在关系，返回关系权重。
    """

    try:
        # 传进来的数据类型为Decimal 需要转为float, 不然，不能直接相加
        x = x.astype(float)
        y = y.astype(float)

    except ValueError:
        # 存在非数字类型字符串‘123ab‘，需要转为float,'123abc'不能转为数字类型的需要置为Nan
        x = pd.to_numeric(x, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')

    # 拼接去除缺失值
    small_df = pd.DataFrame({"x": x, "y": y}).dropna()
    #  判断x, y 两列 是否存在一列全是0， 如果存在直接为0
    if small_df['x'].abs().sum() == 0 or small_df['y'].abs().sum() == 0:
        return 0.0

    count = len(small_df)
    if count <= 1:
        # count 取 0 x,y 同时不为空的一行也没有
        # count 取1 x,y同时不为空的行只有1，结果不可信
        return 0.0
    else:
        # 计算所有关系中的最大r2
        r2_max = 0.0

        for func_name, func in DETECT_CORRELATION_FUNC_MAP.items():
            y_pred = func(small_df['x'], small_df['y'])
            # 判断拟合优度(y_pred)
            if func_name == 'sqrt':
                r2 = r2_score(small_df['y'].apply(lambda y:y**2), y_pred)
            else:
                r2 = r2_score(small_df['y'], y_pred)

            # print(f"func_name: {func_name}, r2: {r2}")
            # 选出最大的r2
            if r2 > r2_max:
                r2_max = r2

        # logger.info(f"r2_max: {r2_max}")
        # 确定r2 在 0-1 范围内
        if r2_max < 0 or r2_max > 1:
            return 0.0
        # 根据r2判段关系是否在设定关系中
        if r2_max >= r2_threshold:
            # 关系在DETECT_CORRELATION_FUNC_MAP之中
            return r2_max
        else:
            # 关系不在DETECT_CORRELATION_FUNC_MAP之中， 置为0.
            return 0.0


# 使用机器学习方法，交叉验证，计算平均r2 得分
def algo_detect_relation(x: pd.Series, y: pd.Series, r2_threshold: float, is_cv=True) -> float:
    """
    检测两组异源数据是否存在关系,使用机器学习模型。
    """
    try:
        # 传进来的数据类型为Decimal 需要转为float, 不然，不能直接相加
        x = x.astype(float)
        y = y.astype(float)

    except ValueError:
        # 存在非数字类型字符串‘123ab‘，需要转为float,'123abc'不能转为数字类型的需要置为Nan
        x = pd.to_numeric(x, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')

    # 拼接去除缺失值
    small_df = pd.DataFrame({"x": x, "y": y}).dropna()

    #  判断x, y 两列 是否存在一列全是一个值，方差为 0.
    if small_df['x'].nunique() == 1 or small_df['y'].nunique() == 1:
        return 0.0

    count = len(small_df)
    if count < 3:
        # count 取 0 x,y 同时不为空的一行也没有
        # count 取1 x,y同时不为空的行只有1，结果不可信
        return 0.0
    else:
        # 计算所有关系中的最大r2
        r2_max = 0.0

        for func_name, func in DETECT_CORRELATION_ML_FUNC_MAP.items():
            r2 = func(small_df['x'].values, small_df['y'].values, is_cv=is_cv)
            # print(f"func_name: {func_name}, r2: {r2}")
            # 选出最大的r2
            if r2 > r2_max:
                r2_max = r2

        # logger.info(f"r2_max: {r2_max}")
        # 确定r2 在 0-1 范围内
        if r2_max < 0 or r2_max > 1:
            return 0.0
        # 根据r2判段关系是否在设定关系中
        if r2_max >= r2_threshold:
            # 关系在DETECT_CORRELATION_FUNC_MAP之中
            return r2_max
        else:
            # 关系不在DETECT_CORRELATION_FUNC_MAP之中， 置为0.
            return 0.0


# 基本关系检测算法接口类
class CorrelationDetectAlgorithm(object):

    @staticmethod
    def get_corr_func(x: pd.Series, y: pd.Series) -> float:
        corr_func = partial(algo_detect_relation, r2_threshold=R2_THRESHOLD)
        return corr_func(x, y)



if __name__ == "__main__":
    X = pd.Series( [
    363.161117647059,
    357.947705882353,
    341.097176470588,
    334.979882352941,
    338.547,
    351.820705882353,
    336.675823529412,
    349.601529411765,
    344.887705882353,
    336.949882352941,
    343.741,
    339.098588235294,
    338.026882352941,
    341.021235294118,
    332.054,
    325.193411764706,
    323.692882352941,
    333.039705882353,
    330.291823529412,
    319.334058823529,
    324.269411764706
])
    y = pd.Series(
        [70] * len(X)

    )

    # X = pd.Series( [
    # -10.41062,
    # -10.50063,
    # -11.17438,
    # -10.22687,
    #
    #
    # ])

    # y = pd.Series( [
    #     0.00000000000001293977,
    #     - 0.00000000000000124864,
    #     0.000000000000005734738,
    #     0.00000000000005455348,
    #
    #
    # ])
    # y = 2 * X
    r2 = algo_detect_relation(X, y, r2_threshold=0.0, is_cv=True)


    print('处理后最后结果为', r2)

    print('linear polyfit ')
    z = np.polyfit(X, y, 1)
    fn = np.poly1d(z)

    print("r2 score polyfit ", r2_score(y, fn(X)))
    # print(y, fn(X))
    # print(np.corrcoef(y, fn(X)))

    print('quadratic polyfit ')
    z = np.polyfit(X, y, 2)
    fn = np.poly1d(z)

    print("r2 score polyfit ", r2_score(y, fn(X)))

    print('quadratic polyfit ')

    if y.min() < 0:
        print("exist negative value,r2 == 0 ")
    else:
        z = np.polyfit(X, y **2, 1)
        fn = np.poly1d(z)

        print("r2 score sqrt fit ", r2_score(y ** 2, fn(X)) )

    print("old realationship detection")
    print(algo_detect_relation_old(X, y, 0.0)  )