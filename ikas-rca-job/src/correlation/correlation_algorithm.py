# -*- coding: utf-8 -*-
"""
关系检测算法：
输入的两组一维异源数据， 检测是否存在如下关系：
    1. 线性关系： y= kx +b
    2. 二次项关系： y = ax^2 +b x +c
    3. 根号关系：y = \sqrt{x}  《===》 y^2 = kx +b
"""
import numpy as np


def check_linear_relation(x, y):
    """
    检查线性关系：y = kx + b

    参数:
        x (array): 数据集1（自变量）
        y (array): 数据集2（因变量）

    返回值:
        bool: 如果数据满足线性关系则返回True，否则False
    """
    fn = np.poly1d(np.polyfit(x, y, 1))
    y_hat = fn(x)
    r_value = np.corrcoef(y_hat, y)[0, 1]
    # 检查回归模型的R方值是否足够高（例如：0.9以上表示良好的拟合）
    return r_value


def check_quadratic_relation(x, y):
    """
    检查二次关系：y = ax^2 + bx + c

    参数:
        x (array): 数据集1（自变量）
        y (array): 数据集2（因变量）

    返回值:
        bool: 如果数据满足二次关系则返回True，否则False
    """

    fn = np.poly1d(np.polyfit(x, y, 2))
    y_hat = fn(x)
    r_value = np.corrcoef(y_hat, y)[0, 1]
    # 检查回归模型的R方值是否足够高（例如：0.9以上表示良好的拟合）
    return r_value


def check_sqrt_relation(x, y):
    """
    检查根号关系：y = sqrt(x)

    参数:
        x (array): 数据集1（自变量）
        y (array): 数据集2（因变量）

    返回值:
        bool: 如果数据满足根号关系则返回True，否则False
    """
    fn = np.poly1d(np.polyfit(x, y ** 2, 1))
    y_hat = fn(x)
    r_value = np.corrcoef(y_hat, y)[0, 1]

    return r_value


# todo:
"""
0. 跨数据源merge比较的情况 如何处置
1. 判断关系是以下几个中的那一个， 如果都不是，置为0.
 

"""


def test_relations(x, y):
    print("线性关系:", check_linear_relation(x, y))
    print("二次项关系:", check_quadratic_relation(x, y))
    print("根号关系:", check_sqrt_relation(x, y))


if __name__ == "__main__":
    # 模拟数据测试
    x = np.array([1, 2, 3, 4, 5])
    y_linear = np.array([0, 2, 4, 6, 8])  # 线性关系 y = 1.5x + b
    y_quad = np.array([1, 4, 9, 16, 25])  # 二次项关系 y = x^2
    y_sqrt = np.array([1, 1.41, 1.73, 2, 2.24])  # 根号关系 y = sqrt(x)

    test_relations(x, y_linear)

    test_relations(x, y_quad)

    test_relations(x, y_sqrt)
