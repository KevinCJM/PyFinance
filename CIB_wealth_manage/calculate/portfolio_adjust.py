# -*- encoding: utf-8 -*-
"""
@File: portfolio_adjust.py
@Modify Time: 2025/8/29 15:45       
@Author: Kevin-Chen
@Descriptions: 
"""
import pandas as pd
import numpy as np


def adjust_numerical_precision(portfolio, df_product_hold, amt):
    def _adj(x):
        if ((x['是否持仓'] == True) and (x['是否可买入'] == False)
                and (x['是否可卖出'] == False)):
            return 0.0
        elif ((x['是否持仓'] == True) and (x['是否可买入'] == True)
              and (x['是否可卖出'] == False) and (x['持仓金额'] < x['当前持仓金额'])):
            return 0.0
        elif ((x['是否持仓'] == True) and (x['是否可买入'] == False)
              and (x['是否可卖出'] == True) and (x['持仓金额'] > x['当前持仓金额'])):
            return 0.0
        elif x['是否卖完'] == True:
            return np.round(x['调仓金额'], 2)
        elif (x['是否持仓'] == True) and (abs(x['持仓金额'] - x['当前持仓金额']) < 100):
            return 0.0
        else:
            return np.round(x['调仓金额'], 0)

    def get_adj_key(df):
        if df[(df['大类资产-现金类'] == 1) & (df['持仓金额'] > 0)
              & (df['持仓金额'] - eps >= df['起购金额'])].shape[0] > 0:
            key = df[(df['大类资产-现金类'] == 1) & (df['持仓金额'] > 0) &
                     (df['持仓金额'] - eps >= df['起购金额'])]['持仓金额'].idxmax()
        elif df[(df['递增金额'] < 1) & (df['持仓金额'] > 0)
                & (df['持仓金额'] - eps >= df['起购金额'])].shape[0] > 0:
            key = df[(df['递增金额'] < 1) & (df['持仓金额'] > 0) &
                     (df['持仓金额'] - eps >= df['起购金额'])]['持仓金额'].idxmax()
        elif df[(df['持仓金额'] - eps >= df['起购金额'])].shape[0] > 0:
            key = df[(df['持仓金额'] - eps >= df['起购金额'])]['持仓金额'].idxmax()
        else:
            key = df['持仓金额'].idxmax()
        return key

    # 投资组合的数值精度调整，增持和减持金额至少1块钱起步
    df = pd.merge(portfolio, df_product_hold[
        ['是否持仓', '是否可买入', '是否可卖出', '当前持仓金额', '风险等级', '大类资产-现金类', '产品名称']],
                  left_index=True, right_index=True, how='right')
    print("``````````````````````````````````````````````````````````````````````````````````")

    for column in ['持仓权重', '持仓金额']:
        df[column] = df[column].fillna(0)

    df['是否卖完'] = df.apply(lambda x: (x['持仓金额'] < 1e-6) and (x['当前持仓金额'] > 0), axis=1)

    # 变化的金额 = 调整后持仓金额 - 调整前持仓金额
    df['调仓金额'] = df['持仓金额'] - df['当前持仓金额']

    # 对调仓金额做四舍五入处理
    # TODO 这里处理之后，就不再满足递增金额限制 !!!!
    df['处理后调仓金额'] = df.apply(lambda x: _adj(x), axis=1)

    # 修正调整后的持仓金额
    df['持仓金额'] = df['当前持仓金额'] + df['处理后调仓金额']

    # 计算误差，大于0表示钱数多了，小于0表示钱数少了
    eps = df['处理后调仓金额'].sum() - (amt - df['当前持仓金额'].sum())

    indexs = list(set(list(portfolio.index) + list(df_product_hold[df_product_hold['是否持仓'] == True].index)))
    key = get_adj_key(df.loc[indexs, :])

    df.loc[key, '持仓金额'] -= eps
    df.loc[:, '持仓权重'] = df.loc[:, '持仓金额'] / df.loc[:, '持仓金额'].sum()
    print(df)

    mask = df['持仓金额'] > 0
    portfolio_adj = df.loc[mask, portfolio.columns]
    info = []
    info.append(f"组合数值精度微调向产品: key={key}, 名称={df.loc[key, '产品名称']}, 增加资金额({-1 * eps})")

    return portfolio_adj, info
