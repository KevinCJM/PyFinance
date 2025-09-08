# -*- encoding: utf-8 -*-
"""
@File: object_functions.py
@Modify Time: 2025/8/29 10:24       
@Author: Kevin-Chen
@Descriptions: 
"""
import cvxpy as cvx
import numpy as np
from calculate.base_functions import funRun, line_k_b, cal_disperse_a_b_c


class ObjectScoreCal():
    def __init__(self, w=None, cov_matrix=None, scoreFuncParas=None, indicatorEstimateResult=None,
                 amt=None, product_info=None):
        self.w, self.cov_matrix = w, cov_matrix
        self.scoreFuncParas, self.indicatorEstimateResult = scoreFuncParas, indicatorEstimateResult
        self.amt, self.product_info = amt, product_info

    def cal_cash_k_b(self):
        scoreFunction = self.scoreFuncParas['现金类占比评分']['scoreFunction']
        scoreArgs = self.scoreFuncParas['现金类占比评分']['scoreArgs']
        min_value = self.indicatorEstimateResult['最小现金类比例']
        max_value = self.indicatorEstimateResult['最大现金类比例']
        min_score = funRun(min_value, scoreFunction, scoreArgs)
        max_score = funRun(max_value, scoreFunction, scoreArgs)
        k, b, *_ = line_k_b(min_value, min_score, max_value, max_score)
        return k, b, min_value, min_score, max_value, max_score

    def cal_return_k_b(self):
        scoreFunction = self.scoreFuncParas['预期收益率评分']['scoreFunction']
        scoreArgs = self.scoreFuncParas['预期收益率评分']['scoreArgs']
        min_value = self.indicatorEstimateResult['最小收益率']
        max_value = self.indicatorEstimateResult['最大收益率']
        min_score = funRun(min_value, scoreFunction, scoreArgs)
        max_score = funRun(max_value, scoreFunction, scoreArgs)
        k, b, *_ = line_k_b(min_value, min_score, max_value, max_score)
        return k, b, min_value, min_score, max_value, max_score

    def cal_volatility_k_b(self):
        x0, y0 = self.indicatorEstimateResult['最小收益率'], self.indicatorEstimateResult['最小波动率']
        x1, y1 = self.indicatorEstimateResult['最大收益率'], self.indicatorEstimateResult['最大波动率']
        # 收益率-波动率的关系
        k, b, *_ = line_k_b(x0, y0, x1, y1)
        if k < 0:
            k = 1e-6
        # 波动率与评分的关系
        scoreFunction = self.scoreFuncParas['预期波动率评分']['scoreFunction']
        scoreArgs = self.scoreFuncParas['预期波动率评分']['scoreArgs']
        min_value = self.indicatorEstimateResult['最小波动率']
        max_value = self.indicatorEstimateResult['最大波动率']
        min_score = funRun(min_value, scoreFunction, scoreArgs)
        max_score = funRun(max_value, scoreFunction, scoreArgs)
        k2, b2, *_ = line_k_b(min_value, min_score, max_value, max_score)
        return k2 * k, k2 * b + b2, x0, min_score, x1, max_score

    def cal_volatility_quad_k_b(self):
        # 波动率的平方 与 波动率评分的关系
        scoreFunction = self.scoreFuncParas['预期波动率评分']['scoreFunction']
        scoreArgs = self.scoreFuncParas['预期波动率评分']['scoreArgs']
        min_value = self.indicatorEstimateResult['最小波动率']
        max_value = self.indicatorEstimateResult['最大波动率']
        min_score = funRun(min_value, scoreFunction, scoreArgs)
        max_score = funRun(max_value, scoreFunction, scoreArgs)
        k, b, *_ = line_k_b(min_value ** 2, min_score, max_value ** 2, max_score)
        return k, b, min_value ** 2, min_score, max_value ** 2, max_score

    def cal_disperse_k_b(self):
        scoreFunction = self.scoreFuncParas['分散度评分']['scoreFunction']
        scoreArgs = self.scoreFuncParas['分散度评分']['scoreArgs']
        min_value = self.indicatorEstimateResult['最小分散度']
        max_value = self.indicatorEstimateResult['最大分散度']
        min_score = funRun(min_value, scoreFunction, scoreArgs)
        max_score = funRun(max_value, scoreFunction, scoreArgs)
        k, b, *_ = line_k_b(min_value, min_score, max_value, max_score)
        return k, b, min_value, min_score, max_value, max_score

    # 流动性评分
    def object_cash_score(self, cash_np, w):
        ratio = cash_np @ w
        k, b, *_ = self.cal_cash_k_b()
        obj = k * ratio + b
        return obj

    # 收益率评分
    def object_return_score(self, future_return_np, w):
        ratio = future_return_np @ w
        k, b, *_ = self.cal_return_k_b()
        obj = k * ratio + b
        return obj

    def object_volatility_is_dcp(self, w=None, cov_matrix=None):
        if w is None:
            w = self.w
        if cov_matrix is None:
            cov_matrix = self.cov_matrix
        return cvx.Problem(cvx.Maximize(-1 * cvx.quad_form(w, cov_matrix)), constraints=[]).is_dcp()

    # 波动率平方的评分
    def object_volatility_score_byquad(self, w, cov_matrix):
        vol_squares = cvx.quad_form(w, cov_matrix)
        k, b, *_ = self.cal_volatility_quad_k_b()
        obj = k * vol_squares + b
        return obj

    # 波动率的评分
    def object_volatility_score_byline(self, w, future_return_np):
        ratio = future_return_np @ w
        k, b, *_ = self.cal_volatility_k_b()
        obj = k * ratio + b
        return obj

    def object_disperse_score(self, w, z, make_object=True):
        if w.shape[0] == 0:
            return 100.0
        a, b, c = cal_disperse_a_b_c()
        if make_object:
            disperse = a * cvx.sum_squares(w) + b * cvx.sum(w)
        else:
            disperse = a * w @ w + b * np.sum(w)
        k, b, *_ = self.cal_disperse_k_b()
        obj = k * disperse + b
        return obj

    # 换手率评价
    def object_turnover_score(self, product_info, w, amt):
        w_m_0 = product_info["当前持仓金额"].fillna(0.0).values
        amt_0 = np.sum(w_m_0)
        fee_buy = product_info["买入换手费率"].values
        fee_sell = product_info["卖出换手费率"].values
        obj = -100 * (fee_sell @ cvx.neg((w * amt - w_m_0)) + fee_buy @ cvx.pos((w * amt - w_m_0))) / max(amt_0, amt)
        return obj

    # 持仓产品数量评价
    def object_product_num_score(self, w, z):
        obj = -1 * cvx.sum(z) / z.shape[0]
        return obj

    # 调仓产品数量评价
    def object_turnover_num_score(self, zb, zs):
        obj = -1 * cvx.sum(zb) / zb.shape[0] - cvx.sum(zs) / zs.shape[0]
        return obj

    def object_duration(self, w, w_inc, w_dec):
        hold_amounts = self.product_info["当前持仓金额"].fillna(0.0).values
        remain_term = self.product_info["持仓剩余期限"].fillna(0.0).values
        ClosedPeriod = self.product_info["产品周期"].fillna(0.0).values
        obj = (cvx.multiply(hold_amounts, (1 - w_dec)) @
               remain_term + (w_inc * self.amt) @ ClosedPeriod) / self.amt
        return obj

    # 通用评价
    def object_common(self, np_value, w):
        obj = np_value @ w
        return obj

    # 流动性损失值
    def cal_cash_loss_value(self, cash_np, w):
        return self.object_cash_score(cash_np, w)

    # 收益率损失值
    def cal_return_loss_value(self, future_return_np, w):
        return self.object_return_score(future_return_np, w)

    def cal_duration_loss_value(self, w, w_inc, w_dec):
        hold_amounts = self.product_info["当前持仓金额"].fillna(0.0).values
        remain_term = self.product_info["持仓剩余期限"].fillna(0.0).values
        ClosedPeriod = self.product_info["产品周期"].fillna(0.0).values
        obj = ((hold_amounts * (1 - w_dec)) @ remain_term + (w_inc * self.amt) @ ClosedPeriod) / self.amt
        return obj

    # 波动率评分损失值
    def cal_volatility_loss_value_byquad(self, w, cov_matrix):
        vol_squares = (np.dot(cov_matrix, w) @ w)
        k, b, *_ = self.cal_volatility_quad_k_b()
        score = k * vol_squares + b
        return score

    # 分散度分数拟合值
    def cal_disperse_fitness_value(self, w, z):
        a, b, c = cal_disperse_a_b_c()
        disperse = a * (w @ w) + b * np.sum(w)
        return disperse

    # 分散度损失值
    def cal_disperse_loss_value(self, w, z):
        # 注：w 要剔除为0的项
        if w.shape[0] == 0:
            return 100
        disperse = self.cal_disperse_fitness_value(w, z)
        k, b, *_ = self.cal_disperse_k_b()
        score = k * disperse + b
        return score

    def cal_product_num_loss_value(self, z):
        return -1 * np.sum(z)

    def cal_turnover_loss_value(self, product_info, w, amt):
        w_m_0 = product_info["当前持仓金额"].fillna(0.0).values
        amt_0 = np.sum(w_m_0)
        fee_buy = product_info["买入换手费率"].values
        fee_sell = product_info["卖出换手费率"].values
        loss_value = -100 * (fee_sell @ (-1 * np.minimum(w * amt - w_m_0, 0))
                             + fee_buy @ np.maximum((w * amt - w_m_0), 0)) / max(amt_0, amt)
        return loss_value

    def obj_piecewise(self, w):
        return np.piecewise(cvx.sum(w), [cvx.sum(w) < 0.05, cvx.sum(w) >= 0.05],
                            [1 * cvx.sum(w) + 1, 1.5 * cvx.sum(w) + 1])

    def output(self):
        dict = {'现金类占比评分': {}, '预期收益率评分': {}, '预期波动率评分': {}, '分散度评分': {}}

        (dict['现金类占比评分']['k'], dict['现金类占比评分']['b'],
         dict['现金类占比评分']['min_value'], dict['现金类占比评分']['min_score'],
         dict['现金类占比评分']['max_value'], dict['现金类占比评分']['max_score']) = self.cal_cash_k_b()

        (dict['预期收益率评分']['k'], dict['预期收益率评分']['b'],
         dict['预期收益率评分']['min_value'], dict['预期收益率评分']['min_score'],
         dict['预期收益率评分']['max_value'], dict['预期收益率评分']['max_score']) = self.cal_return_k_b()

        (dict['预期波动率评分']['k_quad'], dict['预期波动率评分']['b_quad'],
         dict['预期波动率评分']['min_value_quad'], dict['预期波动率评分']['min_score'],
         dict['预期波动率评分']['max_value_quad'], dict['预期波动率评分']['max_score']) = self.cal_volatility_quad_k_b()

        (dict['预期波动率评分']['k'], dict['预期波动率评分']['b'],
         dict['预期波动率评分']['min_value'], dict['预期波动率评分']['min_score'],
         dict['预期波动率评分']['max_value'], dict['预期波动率评分']['max_score']) = self.cal_volatility_k_b()

        (dict['分散度评分']['k'], dict['分散度评分']['b'],
         dict['分散度评分']['min_value'], dict['分散度评分']['min_score'],
         dict['分散度评分']['max_value'], dict['分散度评分']['max_score']) = self.cal_disperse_k_b()

        return dict
