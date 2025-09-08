# -*- encoding: utf-8 -*-
"""
@File: constratints.py
@Modify Time: 2025/8/29 11:24       
@Author: Kevin-Chen
@Descriptions: 
"""

import cvxpy as cvx
import numpy as np
import traceback
from calculate import base_functions


class ConstraintsClass():
    def __init__(self, w, w_c, w_d, w_inc, w_dec, a, z, zb, zs, zh,
                 nCPrd, nDPrd, DMask, amt, min_amt, max_amt, covMatrix, product_info,
                 scoreFuncParas, scoreWeightLevel2, indicatorEstimateResult, ASSET_CODE_DICT,
                 osc, repeat_product_id, zb_r, zs_r, zh_r):
        self.w = w
        self.w_c = w_c
        self.w_d = w_d
        self.w_inc = w_inc
        self.w_dec = w_dec
        self.a = a
        self.z = z
        self.zb = zb
        self.zs = zs
        self.zh = zh
        self.zb_r = zb_r
        self.zs_r = zs_r
        self.zh_r = zh_r
        self.repeat_product_id = repeat_product_id
        self.nCPrd = nCPrd
        self.nDPrd = nDPrd
        self.DMask = DMask
        self.MinAmt = np.where((product_info['是否持仓'].fillna(False).values == True) &
                               (product_info['是否可卖出'].values == False),
                               np.minimum(product_info['当前持仓金额'].values, product_info['起购金额'].values),
                               product_info['起购金额'].values)
        self.IncAmt = product_info['递增金额'].values
        self.amt = amt
        self.min_amt = min_amt
        self.max_amt = max_amt
        self.covMatrix = covMatrix
        self.product_info = product_info
        self.scoreFuncParas = scoreFuncParas
        self.scoreWeightLevel2 = scoreWeightLevel2
        self.indicatorEstimateResult = indicatorEstimateResult
        self.ASSET_CODE_DICT = ASSET_CODE_DICT
        self.osc = osc

    def constraint_check(self, z, zb, zs, zh, zb_r, zs_r, zh_r, a, w, w_c, w_d, w_inc, w_dec,
                         target_asset_alloc=None, longest_idle_time=None, other_args={}):
        check_result = {}

        Constraints = self.base_constraint(w, w_inc, w_dec, zb, zs, zh, zb_r, zs_r, zh_r, make_constraint=False)
        check_result['base_constraint'] = True if np.all([np.all(i) for i in Constraints]) else False

        Constraints = self.bound_constraint(self.nDPrd, self.nCPrd, self.DMask, a, w_c, w_d, z,
                                            self.MinAmt, self.IncAmt, self.amt)
        check_result['bound_constraint'] = True if np.all([np.all(i) for i in Constraints]) else False

        Constraints = self.class_bound_constraint(other_args.get('min_weight', None),
                                                  other_args.get('max_weight', None),
                                                  w=w, z=z, w_c=w_c, w_d=w_d, make_constraint=False)
        check_result['class_bound_constraint'] = True if np.all([np.all(i) for i in Constraints]) else False

        Constraints = self.budget_constraint(w, other_args.get('weight_sum', 0.99), make_constraint=False)
        check_result['budget_constraint'] = True if np.all([np.all(i) for i in Constraints]) else False

        Constraints = self.one_amount_constraint(other_args.get('min_amt', None), other_args.get('max_amt', None),
                                                 z=z, a=a, w=w, w_c=w_c, w_d=w_d, make_constraint=False)
        check_result['one_amount_constraint'] = True if np.all([np.all(i) for i in Constraints]) else False

        Constraints = self.hold_amount_constraint(w=w, zb=zb, zs=zs, w_inc=w_inc, w_dec=w_dec, make_constraint=False)
        check_result['hold_amount_constraint'] = True if np.all([np.all(i) for i in Constraints]) else False

        Constraints = self.portfolios_products_num_constraint(other_args.get('min_num', None),
                                                              other_args.get('max_num', None),
                                                              z=z, make_constraint=False)
        check_result['portfolios_products_num_constraint'] = True if np.all([np.all(i) for i in Constraints]) else False

        Constraints = self.portfolios_duration_constraint(longest_idle_time, w=w, make_constraint=False)
        check_result['portfolios_duration_constraint'] = True if np.all([np.all(i) for i in Constraints]) else False

        Constraints = self.asset_alloc_constraint(target_asset_alloc, other_args['asset_alloc_bias_detail'],
                                                  other_args.get('asset_alloc_bias_global', {}),
                                                  w=w, make_constraint=False, eps=1e-4)
        check_result['asset_alloc_constraint'] = True if np.all([np.all(i) for i in Constraints]) else False

        Constraints = self.cash_score_constraint(other_args['scoreConstraints'].get('cashScoreThresholdL', None),
                                                 other_args['scoreConstraints'].get('cashScoreThresholdU', None),
                                                 w=w, make_constraint=False)
        check_result['cash_score_constraint'] = True if np.all([np.all(i) for i in Constraints]) else False

        Constraints = self.return_score_constraint(other_args['scoreConstraints'].get('returnScoreThresholdL', None),
                                                   other_args['scoreConstraints'].get('returnScoreThresholdU', None),
                                                   w=w, make_constraint=False)
        check_result['return_score_constraint'] = True if np.all([np.all(i) for i in Constraints]) else False

        return check_result

    def base_constraint(self, w=None, w_inc=None, w_dec=None, zb=None, zs=None, zh=None,
                        zb_r=None, zs_r=None, zh_r=None, repeat_product_id=None, make_constraint=True):
        if w is None:
            w = self.w
        if w_inc is None:
            w_inc = self.w_inc
        if w_dec is None:
            w_dec = self.w_dec
        if zb is None:
            zb = self.zb
        if zs is None:
            zs = self.zs
        if zh is None:
            zh = self.zh
        if zb_r is None:
            zb_r = self.zb_r
        if zs_r is None:
            zs_r = self.zs_r
        if zh_r is None:
            zh_r = self.zh_r
        if repeat_product_id is None:
            repeat_product_id = self.repeat_product_id

        Constraints = []
        is_holding = self.product_info['是否持仓'].fillna(False).values == True
        w_m_0 = self.product_info['当前持仓金额'].fillna(0.0).values

        Constraints += [(zb + zs + zh) >= 1, (zb + zs + zh) <= 1]
        # 以下2个约束确保：当 zb=1,zs=0 时: 1>=w-w0 >= 0；当 zs=1, z=0 时: w-w0=<=0; 当 zb=0,zs=0时: w=w0; 实现zs,zb,与w联动
        eps = 1.0
        w0_amt = w_m_0
        if make_constraint:
            Constraints.append(
                zs * 0
                + cvx.multiply(zh, np.maximum(w0_amt - eps, 0))
                + cvx.multiply(zb, w0_amt + eps) <= w * self.amt
            )
            Constraints.append(
                w * self.amt <= cvx.multiply(zs, np.maximum(w0_amt - eps, 0))
                + cvx.multiply(zh, w0_amt + eps)
                + zb * self.amt
            )
        else:
            Constraints.append(zs * 0 + zh * np.maximum(w0_amt - eps, 0) + zb * (w0_amt + eps) <= w * self.amt)
            Constraints.append(w * self.amt <= zs * np.maximum(w0_amt - eps, 0) + zh * (w0_amt + eps) + zb * self.amt)

        # 重复产品的买入、卖出、持仓约束
        if repeat_product_id.shape[0] > 0:
            # 约束：不能同时发生买和卖
            Constraints += [(zb_r + zs_r) <= 1]
            for i, product_id in enumerate(repeat_product_id):
                mask = self.product_info['原始ID'].values == product_id
                # 以下约束确保当zb_r=0时，zb[mask]都为0；当zb[mask]有一个为1，则zb_r=1；当zb[mask]全部为0，则zb_r=0
                if make_constraint:
                    Constraints.append(cvx.sum(zb[mask]) / np.sum(mask) <= zb_r[i])
                    Constraints.append(cvx.sum(zb[mask]) >= zb_r[i])
                    Constraints.append(cvx.sum(zs[mask]) / np.sum(mask) <= zs_r[i])
                    Constraints.append(cvx.sum(zs[mask]) >= zs_r[i])
                else:
                    Constraints.append(np.sum(zb[mask]) / np.sum(mask) <= zb_r[i])
                    Constraints.append(np.sum(zb[mask]) >= zb_r[i])
                    Constraints.append(np.sum(zs[mask]) / np.sum(mask) <= zs_r[i])
                    Constraints.append(np.sum(zs[mask]) >= zs_r[i])

        # 以下约束保证：当减持时w_inc=0，当不减持时w_inc=(当前持仓金额-原始持仓金额)/总资产
        Constraints.append(w_inc >= 0)
        Constraints.append(w_inc <= 1.0)
        Constraints.append(w_inc <= zb)
        Constraints.append(w_inc <= (w - w_m_0 / self.amt + self.amt * (1 - zb) + 1e-4))
        Constraints.append(w_inc <= (w - w_m_0 / self.amt - self.amt * (1 - zb) - 1e-4))

        # 以下约束保证：当增持时w_dec=0，减持时w_dec=(原始持仓金额-当前持仓金额)/当前持仓金额
        Constraints.append(w_dec >= 0)
        Constraints.append(w_dec <= 1.0)
        Constraints.append(w_dec <= zs)

        mask = w_m_0 >= 0.01
        if np.any(mask):
            Constraints.append(
                w_dec[mask] <= ((w_m_0[mask] - w[mask] * self.amt) / w_m_0[mask] + self.amt * (1 - zs[mask]) + 1e-4))
            Constraints.append(
                w_dec[mask] >= ((w_m_0[mask] - w[mask] * self.amt) / w_m_0[mask] - self.amt * (1 - zs[mask]) - 1e-4))

        if np.any(~mask):
            Constraints.append(w_dec[~mask] <= (0.0 + self.amt * (1 - zs[~mask]) + 1e-4))
            Constraints.append(w_dec[~mask] >= (0.0 - self.amt * (1 - zs[~mask]) - 1e-4))

        return Constraints

    # 持仓权重边界约束
    def bound_constraint(self, nDPrd, nCPrd, DMask, a, w_c, w_d, z, MinAmt, IncAmt, amt):
        Constraints = []

        if nDPrd > 0:
            Constraints.append(a >= 0)
            Constraints.append(w_d >= 0.0)
            Constraints.append(np.diag(MinAmt[DMask]) @ z[DMask] + np.diag(IncAmt[DMask]) @ a <= amt * z[DMask])

        # 连续型产品
        if nCPrd > 0:
            Constraints.append(w_c >= 0.0)
            Constraints.append(w_c <= z[~DMask])
            Constraints.append(amt * w_c >= np.diag(MinAmt[~DMask]) @ z[~DMask])

        return Constraints

    # 大类资产权重边界约束
    def class_bound_constraint(self, min_weight, max_weight_dict,
                               w=None, z=None, w_c=None, w_d=None, make_constraint=True):
        if w is None:
            w = self.w
        if z is None:
            z = self.z
        if w_c is None:
            w_c = self.w_c
        if w_d is None:
            w_d = self.w_d
        Constraints = []

        if min_weight is not None:
            isno_hold = self.product_info['是否持仓'].fillna(False).values == False
            is_hold = (isno_hold == False)
            if self.nCPrd > 0:
                if np.sum(isno_hold[~self.DMask]) > 0:
                    # 没有持仓的产品最小权重 >= min_weight
                    Constraints.append(w_c[isno_hold[~self.DMask]] >= min_weight * z[(~self.DMask) & isno_hold])

                if np.sum(is_hold[~self.DMask]) > 0:
                    # 已经持仓的产品，有可能持仓权重很低，因此不要 >= min_weight，>=0即可
                    Constraints.append(w_c[is_hold[~self.DMask]] >= 0.0)

            if self.nDPrd > 0:
                if np.sum(isno_hold[self.DMask]) > 0:
                    # 没有持仓的产品最小权重 >= min_weight
                    Constraints.append(w_d[isno_hold[self.DMask]] >= min_weight * z[self.DMask & isno_hold])

                if np.sum(is_hold[self.DMask]) > 0:
                    # 已经持仓的产品，有可能持仓权重很低，因此不要 >= min_weight，>=0即可
                    Constraints.append(w_d[is_hold[self.DMask]] >= 0.0)

        if max_weight_dict is not None:
            is_hold = self.product_info['是否持仓'].fillna(False).values == True
            can_not_sell = (self.product_info['是否可卖出'].values == False)
            is_special = (is_hold & can_not_sell)
            ASSET_CODE_DICT_INV = dict(zip(map(lambda x: "clazz" + x,
                                               list(self.ASSET_CODE_DICT.values())), self.ASSET_CODE_DICT.keys()))

            columns = ["大类资产-"
                       + ASSET_CODE_DICT_INV[iAssetCode] for iAssetCode, max_weight in max_weight_dict.items()]
            max_weight_np = np.array([max_weight for iAssetCode, max_weight in max_weight_dict.items()])
            asset_expose_np = self.product_info.loc[:, columns].values
            product_max_weight_np = asset_expose_np.dot(max_weight_np)
            product_max_weight_np = np.where(is_special,
                                             np.maximum(self.product_info['当前持仓金额'].fillna(0.0) / self.amt + 1e-3,
                                                        product_max_weight_np), product_max_weight_np)

            if make_constraint:
                Constraints.append(w <= cvx.multiply(product_max_weight_np, z))
            else:
                Constraints.append(w <= (product_max_weight_np * z))

        return Constraints

    # 预算限制
    def budget_constraint(self, w, weight_sum, make_constraint=True):
        Constraints = []
        if make_constraint:
            Constraints.append(cvx.sum(w) <= 1.0)
            Constraints.append(cvx.sum(w) >= weight_sum)
        else:
            Constraints.append(np.sum(w) <= 1.0)
            Constraints.append(np.sum(w) >= weight_sum)
        return Constraints

    def one_amount_constraint(self, min_amt, max_amt,
                              z=None, a=None, w=None, w_c=None, w_d=None, make_constraint=True):
        # 单产品持仓金额限制
        if z is None:
            z = self.z
        if a is None:
            a = self.a
        if w is None:
            w = self.w
        if w_c is None:
            w_c = self.w_c
        if w_d is None:
            w_d = self.w_d
        Constraints = []

        if min_amt is not None:
            if self.nCPrd > 0:
                Constraints.append(self.amt * w_c >= min_amt * z[~self.DMask])
            if self.nDPrd > 0:
                if make_constraint:
                    Constraints.append(
                        cvx.multiply(self.MinAmt[self.DMask], z[self.DMask]) +
                        cvx.multiply(self.IncAmt[self.DMask], a) >= min_amt * z[self.DMask]
                    )
                else:
                    Constraints.append(
                        self.MinAmt[self.DMask] * z[self.DMask]
                        + self.IncAmt[self.DMask] * a >= min_amt * z[self.DMask]
                    )

        MAX_AMT = 10000e8
        if max_amt is not None:
            max_amt_np = np.where(
                self.product_info['最高持有金额'].fillna(MAX_AMT) > 0.0,
                self.product_info['最高持有金额'].fillna(MAX_AMT),
                max_amt
            )
        else:
            max_amt_np = self.product_info['最高持有金额'].fillna(MAX_AMT).values
        max_amt_np = np.maximum(max_amt_np, self.product_info['当前持仓金额'].fillna(0.0).values)
        valid = (max_amt_np > 0.0) & (max_amt_np < MAX_AMT)

        if np.sum(valid) > 0:
            Constraints.append(self.amt * w[valid] <= max_amt_np[valid])

        # 相同产品持有金额的和约束
        for id in set(self.product_info['原始ID'].values):
            same_product_np = self.product_info['原始ID'].values == id
            if np.sum(same_product_np) <= 1:
                continue
            else:
                max_mat_np = self.product_info.loc[same_product_np, '最高持有金额'].dropna().values
                max_hold_amt = np.max(self.product_info.loc[same_product_np, '当前持仓金额'].dropna().values)
                if max_mat_np.shape[0] > 0:
                    if make_constraint:
                        Constraints.append(
                            cvx.sum(w[same_product_np]) * self.amt <= max(np.max(max_mat_np), max_hold_amt))
                    else:
                        Constraints.append(
                            np.sum(w[same_product_np]) * self.amt <= max(np.max(max_mat_np), max_hold_amt))

        is_not_holding = (self.product_info['是否持仓'].fillna(False).values == False)
        is_can_not_buy = (self.product_info['是否可买入'].fillna(0).values == False)
        hold_amounts = self.product_info['当前持仓金额'].fillna(0.0).values
        if np.sum(is_not_holding & is_can_not_buy) > 0:
            Constraints.append(
                w[is_not_holding & is_can_not_buy] * self.amt <= hold_amounts[is_not_holding & is_can_not_buy] + 0.01)
        return Constraints

    # 单产品持仓金额限制
    def one_amount_constraint(self, min_amt, max_amt,
                              z=None, a=None, w=None, w_c=None, w_d=None, make_constraint=True):
        if z is None:
            z = self.z
        if a is None:
            a = self.a
        if w is None:
            w = self.w
        if w_c is None:
            w_c = self.w_c
        if w_d is None:
            w_d = self.w_d
        Constraints = []

        if min_amt is not None:
            if self.nCPrd > 0:
                Constraints.append(self.amt * w_c >= min_amt * z[~self.DMask])
            if self.nDPrd > 0:
                if make_constraint:
                    Constraints.append(
                        cvx.multiply(self.MinAmt[self.DMask], z[self.DMask])
                        + cvx.multiply(self.IncAmt[self.DMask], a) >= min_amt * z[self.DMask]
                    )
                else:
                    Constraints.append(
                        self.MinAmt[self.DMask] * z[self.DMask]
                        + self.IncAmt[self.DMask] * a >= min_amt * z[self.DMask]
                    )

        MAX_AMT = 10000e8
        if max_amt is not None:
            max_amt_np = np.where(
                self.product_info['最高持有金额'].fillna(MAX_AMT) > 0.0,
                self.product_info['最高持有金额'].fillna(MAX_AMT),
                max_amt
            )
        else:
            max_amt_np = self.product_info['最高持有金额'].fillna(MAX_AMT).values
        max_amt_np = np.maximum(max_amt_np, self.product_info['当前持仓金额'].fillna(0.0).values)
        valid = (max_amt_np > 0.0) & (max_amt_np < MAX_AMT)

        if np.sum(valid) > 0:
            Constraints.append(self.amt * w[valid] <= max_amt_np[valid])

        # 相同产品持有金额的和约束
        for id in set(self.product_info['原始ID'].values):
            same_product_np = self.product_info['原始ID'].values == id
            if np.sum(same_product_np) <= 1:
                continue
            else:
                max_mat_np = self.product_info.loc[same_product_np, '最高持有金额'].dropna().values
                max_hold_amt = np.max(self.product_info.loc[same_product_np, '当前持仓金额'].dropna().values)
                if max_mat_np.shape[0] > 0:
                    if make_constraint:
                        Constraints.append(
                            cvx.sum(w[same_product_np]) * self.amt <= max(np.max(max_mat_np), max_hold_amt))
                    else:
                        Constraints.append(
                            np.sum(w[same_product_np]) * self.amt <= max(np.max(max_mat_np), max_hold_amt))

        is_not_holding = (self.product_info['是否持仓'].fillna(False).values == False)
        is_can_not_buy = (self.product_info['是否可买入'].fillna(0).values == False)
        hold_amounts = self.product_info['当前持仓金额'].fillna(0.0).values
        if np.sum(is_not_holding & is_can_not_buy) > 0:
            Constraints.append(
                w[is_not_holding & is_can_not_buy] * self.amt <= hold_amounts[is_not_holding & is_can_not_buy] + 0.01)
        return Constraints

    # 持仓产品的金额约束
    def hold_amount_constraint(self, w=None, w_inc=None, w_dec=None, zb=None, zs=None,
                               holding_min_sell=1e4, holding_min_buy=1e4, make_constraint=True):
        if w is None:
            w = self.w
        if w_inc is None:
            w_inc = self.w_inc
        if w_dec is None:
            w_dec = self.w_dec
        Constraints = []

        # 买入和卖出最小金额约束 v0.2版本
        is_holding = self.product_info['是否持仓'].fillna(False).values == True
        w_m_0 = self.product_info['当前持仓金额'].fillna(0.0).values
        if np.any(is_holding):
            if make_constraint:
                Constraints.append((self.amt * w[is_holding]) <= (
                        cvx.multiply(np.maximum(w_m_0[is_holding] - holding_min_sell, 0),
                                     zs[is_holding]) + self.amt * (1 - zs[is_holding])
                ))
                Constraints.append(
                    (self.amt * w[is_holding]) >= cvx.multiply(
                        np.maximum(w_m_0[is_holding] + holding_min_buy, 0), zb[is_holding])
                )
            else:
                Constraints.append((self.amt * w[is_holding]) <= (
                        (np.maximum(w_m_0[is_holding] - holding_min_sell, 0) * zs[is_holding]) +
                        self.amt * zb[is_holding]
                ))
                Constraints.append(
                    (self.amt * w[is_holding]) >= (
                        (np.maximum(w_m_0[is_holding] + holding_min_buy, 0) * zb[is_holding])
                    )
                )

        # 产品金额下限约束：针对“是持仓、当前持仓金额大于零, 不能卖、能买”的产品，最低持仓金额=当前持仓金额
        conditions = ((self.product_info['是否持仓'].fillna(False).values == True) &
                      (self.product_info['当前持仓金额'].values > 0) &
                      (self.product_info['是否可卖出'].values == False) &
                      (self.product_info['是否可买入'].values == True))
        hold_amounts = self.product_info['当前持仓金额'].fillna(0.0).values
        if np.sum(conditions) > 0:
            Constraints.append(w[conditions] >= hold_amounts[conditions] / self.amt - 1e-3)

        # 产品金额上限约束：针对“是持仓、不能买、可以卖”的产品，最高持仓金额=当前持仓金额
        conditions = ((self.product_info['是否持仓'].fillna(False).values == True) &
                      (self.product_info['是否可买入'].values == False) &
                      (self.product_info['是否可卖出'].values == True))
        hold_amounts = self.product_info['当前持仓金额'].values
        if np.sum(conditions) > 0:
            Constraints.append(w[conditions] <= hold_amounts[conditions] / self.amt + 1e-3)

        # 产品金额锁定约束：针对“是持仓、不能买、不能卖”的产品，持仓金额权重=原始持仓权重
        conditions = ((self.product_info['是否持仓'].fillna(False).values == True) &
                      (self.product_info['是否可卖出'].values == False) &
                      (self.product_info['是否可买入'].values == False))
        if np.sum(conditions) > 0:
            Constraints.append(w[conditions] <= hold_amounts[conditions] / self.amt + 1e-3)
            Constraints.append(w[conditions] >= hold_amounts[conditions] / self.amt - 1e-3)

        return Constraints

    # 相同产品只做一次买入卖出的约束 (避免买入和卖出同一只产品)，注：相同产品只会出现在持仓里面，产品池里面不会出现
    def same_product_turnover_num_constraint(self, zb=None, zs=None, make_constraint=True):
        Constraints = []
        if (zb is None) or (zs is None):
            return Constraints

        # 判断哪些产品出现了重复
        for id in set(self.product_info['原始ID'].values):
            same_product_np = self.product_info['原始ID'].values == id
            if np.sum(same_product_np) <= 1:
                continue
            if make_constraint:
                # 买入只能有一次
                Constraints.append(cvx.sum(zb[same_product_np]) <= 1.01)
            else:
                # 买入只能有一次
                Constraints.append(np.sum(zb[same_product_np]) <= 1.01)

        return Constraints

    # 组合产品数量约束
    def portfolios_products_num_constraint(self, min_num, max_num, z=None, make_constraint=True):
        # 产品总数量限制
        if z is None:
            z = self.z
        Constraints = []

        if min_num is not None:
            if make_constraint:
                Constraints.append(cvx.sum(z) >= min_num)
            else:
                Constraints.append(np.sum(z) >= min_num)

        if max_num is not None:
            if make_constraint:
                Constraints.append(cvx.sum(z) <= max_num)
            else:
                Constraints.append(np.sum(z) <= max_num)

        return Constraints

    def portfolios_duration_constraint(self, duration, w=None, make_constraint=True):
        if w is None:
            w = self.w
        Constraints = []

        if duration is not None:
            # 剩余持仓金额 * 剩余期限 + 新增金额 * 投资期限
            hold_amounts = self.product_info["当前持仓金额"].fillna(0.0).values
            remain_term = self.product_info["持仓剩余期限"].fillna(0.0).values
            ClosedPeriod = self.product_info["产品周期"].fillna(0.0).values
            hold_cannot_sell = (
                    (self.product_info["是否持仓"].fillna(False).values == True) *
                    (self.product_info["是否可卖出"].fillna(False).values == False)
            )

            # 不能减持的持仓金额的久期 (能减持的产品久期都是0，不再计算久期)
            holdDuration = (hold_amounts * hold_cannot_sell) @ remain_term

            if make_constraint:
                buyDuration = cvx.pos(w * self.amt - hold_amounts) @ ClosedPeriod
            else:
                buyDuration = np.maximum(w * self.amt - hold_amounts, 0) @ ClosedPeriod

            Constraints.append(
                holdDuration + buyDuration <= duration * self.amt
            )

        return Constraints

    def asset_alloc_constraint(self, target_asset_alloc, asset_alloc_bias_detail, asset_alloc_bias_global,
                               w=None, make_constraint=True, eps=0.0):
        if w is None:
            w = self.w
        Constraints = []

        AssetInfo = self.product_info.loc[:, self.product_info.columns.str.contains("大类资产-")]

        if (target_asset_alloc is not None) and (AssetInfo.shape[1] > 0):
            for iAsset, iTarget in target_asset_alloc.items():
                if "大类资产-" + iAsset not in AssetInfo:
                    raise Exception(f"产品信息表中不存在大类资产 '{iAsset}' 的暴露数据！")

                iExpose = AssetInfo["大类资产-" + iAsset].values
                iPortfolioExpose = iExpose @ w

                if iAsset in asset_alloc_bias_global:
                    l_asset_alloc = asset_alloc_bias_global[iAsset]["l_asset_alloc"]
                    u_asset_alloc = asset_alloc_bias_global[iAsset]["u_asset_alloc"]
                    Constraints += [iPortfolioExpose >= l_asset_alloc,
                                    iPortfolioExpose <= u_asset_alloc]
                else:
                    l_asset_alloc_bias = asset_alloc_bias_detail[iAsset]["l_asset_alloc_bias"]
                    u_asset_alloc_bias = asset_alloc_bias_detail[iAsset]["u_asset_alloc_bias"]
                    Constraints += [iPortfolioExpose >= (max(0.0, iTarget - l_asset_alloc_bias) - eps),
                                    iPortfolioExpose <= (min(1.0, iTarget + u_asset_alloc_bias) + eps)]

        return Constraints

    # 收益率分数约束
    def return_score_constraint(self, itemScoreThreshold_l=None, itemScoreThreshold_u=None,
                                w=None, make_constraint=True):
        if w is None:
            w = self.w
        # 根据门限反求收益率
        scoreFunction = self.scoreFuncParas['预期收益率评分']['scoreFunction']
        scoreArgs = self.scoreFuncParas['预期收益率评分']['scoreArgs']
        constraints = []
        if itemScoreThreshold_l is not None:
            x = base_functions.root(y=itemScoreThreshold_l, scoreFunction=scoreFunction, scoreArgs=scoreArgs)
            constraints += [(self.product_info['基准指数收益率'].values @ w) >= x]
        if itemScoreThreshold_u is not None:
            x = base_functions.root(y=itemScoreThreshold_u, scoreFunction=scoreFunction, scoreArgs=scoreArgs)
            constraints += [(self.product_info['基准指数收益率'].values @ w) <= x]
        return constraints

    # 流动性分数约束
    def cash_score_constraint(self, itemScoreThreshold_l=None, itemScoreThreshold_u=None,
                              w=None, make_constraint=True):
        if w is None:
            w = self.w
        # 根据门限反求现金类占比
        scoreFunction = self.scoreFuncParas['现金类占比评分']['scoreFunction']
        scoreArgs = self.scoreFuncParas['现金类占比评分']['scoreArgs']
        constraints = []
        if itemScoreThreshold_l is not None:
            x = base_functions.root(y=itemScoreThreshold_l, scoreFunction=scoreFunction, scoreArgs=scoreArgs)
            constraints += [(self.product_info['大类资产-现金类'].values @ w) >= x]
        if itemScoreThreshold_u is not None:
            x = base_functions.root(y=itemScoreThreshold_u, scoreFunction=scoreFunction, scoreArgs=scoreArgs)
            constraints += [(self.product_info['大类资产-现金类'].values @ w) <= x]
        return constraints

    # 波动率分数约束
    def volatility_score_constraint(self, itemScoreThreshold, w=None, make_constraint=True):
        # 根据门限反求波动率
        if w is None:
            w = self.w

        scoreFunction = self.scoreFuncParas['预期波动率评分']['scoreFunction']
        scoreArgs = self.scoreFuncParas['预期波动率评分']['scoreArgs']

        x = base_functions.root(y=itemScoreThreshold, scoreFunction=scoreFunction, scoreArgs=scoreArgs)

        if make_constraint:
            return [cvx.quad_form(self.w, self.covMatrix) <= (x ** 2)]
        else:
            return [((np.dot(self.covMatrix, w)) @ w) <= (x ** 2)]

    # 风险分数约束，波动率 和 分散度的共同约束
    def risk_score_constraint(self, w, covMatrix, itemScoreThreshold, make_constraint=True):
        A, b, c = self.risk_score_param_comb(w, covMatrix)

        if make_constraint:
            return [(cvx.quad_form(w, A) + (w @ b) + c) >= itemScoreThreshold]
        else:
            return [((np.dot(covMatrix, w) @ w) + (w @ b) + c) >= itemScoreThreshold]

    def risk_score_param_comb(self, w, covMatrix):

        def _cal_disperse_k_b():
            scoreFunction = self.scoreFuncParas['分散度评分']['scoreFunction']
            scoreArgs = self.scoreFuncParas['分散度评分']['scoreArgs']
            min_value = self.indicatorEstimateResult['最小分散度']
            max_value = self.indicatorEstimateResult['最大分散度']
            min_score = base_functions.funRun(min_value, scoreFunction, scoreArgs)
            max_score = base_functions.funRun(max_value, scoreFunction, scoreArgs)
            k, b = base_functions.line_k_b(min_value, min_score, max_value, max_score)
            return k, b

        w1, w2 = self.scoreWeightLevel2['波动率'], self.scoreWeightLevel2['分散度']
        a_quad, b_quad, c_quad = base_functions.cal_disperse_a_b_c()
        k_vol, b_vol, *_ = self.osc.cal_volatility_quad_k_b()
        k_disperse, b_disperse = _cal_disperse_k_b()

        A_m = np.diag(w2 * a_quad * k_disperse * np.ones(w.shape[0])) + w1 * k_vol * covMatrix
        b_v = w2 * b_quad * k_disperse * np.ones(w.shape[0])
        c = w1 * b_vol + w2 * b_disperse

        return A_m, b_v, c

    def cal_risk_loss_value(self, w, covMatrix):
        A, b, c = self.risk_score_param_comb(w, covMatrix)
        return (np.dot(A, w) @ w) + (w @ b) + c
