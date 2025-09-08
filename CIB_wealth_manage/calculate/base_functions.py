# -*- encoding: utf-8 -*-
"""
@File: base_functions.py
@Modify Time: 2025/8/28 19:16       
@Author: Kevin-Chen
@Descriptions: 
"""
# 各种指标就计算的函数
from scipy import optimize
import numpy as np
import math


def funRun(x, scoreFunction, scoreArgs):
    if 'atan' == scoreFunction:
        a, b, c, d, k, t = scoreArgs
        return a * np.arctan(b * (k * x + t) + c) + d

    elif 'arctan' == scoreFunction:
        a, b, c, d, k, t = scoreArgs
        return a * np.arctan(b * (k * x + t) + c) + d

    elif 'morgan_mercer_flodin' == scoreFunction:
        a, b, c, d, k, t = scoreArgs
        x_ = (k * x + t)
        if np.isscalar(x_):
            x_ = 1e-6 if x_ < 0 else x_
        else:
            x_ = np.where(x_ > 0, x_, 1e-6)
        return (b * c + a * np.power(x_, d)) / (c + np.power(x_, d))

    elif 'morgan_mercer_flodin_customreturn' == scoreFunction:
        a, b, c, d, k, t = scoreArgs
        x_ = np.power(x, k)
        if np.isscalar(x_):
            x_ = 1e-6 if x_ < 0 else x_
        else:
            x_ = np.where(x_ > 0, x_, 1e-6)
        return (b * c + a * np.power(x_, d)) / (c + np.power(x_, d))

    elif 'tanh_neg' == scoreFunction:
        a, b, c, d, k, t = scoreArgs
        return -1 * a * np.tanh(b * (k * x + t) + c) + d

    else:
        print(f"不支持的评分函数{scoreFunction}")


def cal_disperse_a_b_c():
    # 单产品分散度
    return -1.5, -1.5, 0


# 给定2个点，求直线的斜率和截距
def line_k_b(x1, y1, x2, y2):
    k = (y2 - y1) / (x2 - x1)
    b = y1 - x1 * (y2 - y1) / (x2 - x1)
    return k, b


# 求根函数
# 求 scoreFunction(x, scoreArgs) - y = 0 的 x值
def root(y, scoreFunction, scoreArgs, type=None):
    """
    y: 函数值
    scoreFunction: 函数名称
    scoreArgs: 函数参数
    type: 函数类型，取值范围是：'现金类占比评分'，'预期收益率评分'，'预期波动率评分'，'分散度评分'， 不同类型涉及的搜索区间不同
    """

    def f(x):
        return funRun(x, scoreFunction, scoreArgs) - y

    res = optimize.root(f, 0.05)
    return res.x[0]


# 求根函数，用于复合函数
# 求 w1*scoreFunction1(x, scoreArgs1) + w2*scoreFunction2(x, scoreArgs2) - y = 0 的 x值
def root2(y, w1, scoreFunction1, scoreArgs1, w2, scoreFunction2, scoreArgs2, type=None):
    def f(x):
        return (w1 * funRun(x, scoreFunction1, scoreArgs1)
                + w2 * funRun(x, scoreFunction2, scoreArgs2) - y)

    res = optimize.root(f, [0.05, 0.05])
    return res.x
