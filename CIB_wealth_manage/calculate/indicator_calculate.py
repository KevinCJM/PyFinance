# -*- encoding: utf-8 -*-
"""
@File: indicator_calculate.py
@Modify Time: 2025/8/29 16:45       
@Author: Kevin-Chen
@Descriptions: 
"""
from calculate.base_functions import funRun
import numpy as np
import pandas as pd


def calDuration(portfolio, products, amt):
    """
    afger_df: 调仓后的持仓
    products: 持仓+产品池
    amt: 总资金量
    """
    # 计算组合久期
    df = pd.merge(products,
                  portfolio[['持仓权重', '持仓金额']],
                  left_index=True,
                  right_index=True,
                  how='left')

    # 不能减持的持仓资金的久期 (能减持的产品久期都设为0，不再算久期)
    hold_amounts = df['当前持仓金额'].fillna(0.0).values
    remain_term = df['持仓剩余期限'].fillna(0.0).values
    ClosedPeriod = df['产品周期'].fillna(0.0).values
    weight = df['持仓权重'].fillna(0.0).values

    # 已有持仓的久期
    holdDuration = np.minimum(weight * amt, hold_amounts) @ remain_term

    # 新增持仓的久期
    buyDuration = np.maximum(weight * amt - hold_amounts, 0) @ ClosedPeriod

    # 总久期
    duration = (buyDuration + holdDuration) / amt

    # 计算持仓久期 (优化前的久期)
    isHolding = (products['是否持仓'] == True)
    holdAmt = products.loc[isHolding, '当前持仓金额'].sum()

    if holdAmt > 1e-3:
        durationBefore = ((products.loc[isHolding, '当前持仓金额'].values / holdAmt) @
                          products.loc[isHolding, '持仓剩余期限'].values)
    else:
        durationBefore = 0.0

    return duration, durationBefore


def calCashScore(portfolio, scoreFunction=None, scoreArgs=None):
    # 计算现金类占比
    cash_ratio = portfolio['大类资产-现金类'].values @ portfolio['持仓权重'].values

    # 示例打分函数（注释掉的公式）
    # a, b, c, d = 55.91, 26.55, -0.26, 14.30
    # score = a * math.atan(b * cash_ratio + c) + d

    # 实际用传入的评分函数执行
    score = funRun(cash_ratio, scoreFunction, scoreArgs)

    return cash_ratio, score


def calReturnScore(portfolio, return_np, scoreFunction=None, scoreArgs=None):
    # 收益率评分
    w = portfolio['持仓权重'].values
    r = w @ return_np.values

    # 示例参数
    # a, b, c, d = 100.0, 0.99, 9e-7, 5.3
    # score = b * c * a * math.pow(r, d) / (c + math.pow(r, d))

    score = funRun(r, scoreFunction, scoreArgs)
    return r, score


def calVolatilityScore(portfolio, cov_matrix, scoreFunction=None, scoreArgs=None):
    # 波动率评分
    w = portfolio['持仓权重'].values
    cov = cov_matrix.values

    # 计算波动率
    volatility = (np.dot(cov, w) @ w) ** 0.5

    # 示例参数
    # a, b, c, d = 50.68, 14.85, -2.14, 50.69
    # score = -1 * a * math.tanh(b * volatility + c) + d

    score = funRun(volatility, scoreFunction, scoreArgs)
    return volatility, score


def calDisperseScore(portfolio, scoreFunction=None, scoreArgs=None):
    # 分散度评分
    if portfolio.shape[0] == 0:
        return 0.0, 100.0

    w = portfolio['持仓权重'].values
    disperse = -1 * w @ np.log(w + (1e-20))

    # 示例参数
    # a, b, c, d = 100.0, 0.0, 1.0, 2.75
    # score = b * c * a * math.pow(disperse, d) / (c + math.pow(disperse, d))

    score = funRun(disperse, scoreFunction, scoreArgs)
    return disperse, score
