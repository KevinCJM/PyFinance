# -*- encoding: utf-8 -*-
"""
@File: finetuning_config_setting.py
@Modify Time: 2025/8/28 19:38       
@Author: Kevin-Chen
@Descriptions: 
"""
# 针对微调场景的参数设置
import pandas as pd
from config_py import ASSET_CODE_DICT, CLAZZ_ORDER

CLAZZ = list(ASSET_CODE_DICT.keys())


class FineTuningConfigSetting():
    # 针对微调场景的参数设置
    def __init__(self, clazzInterestOfFineTuning, ProductInfo,
                 amt, obj_alloc, target_asset_alloc,
                 asset_alloc_rate, rate_limit, asset_alloc_bias, other_args):
        self.clazzInterestOfFineTuning = clazzInterestOfFineTuning
        self.ProductInfo = ProductInfo
        self.amt = amt
        self.obj_alloc = obj_alloc
        self.target_asset_alloc = target_asset_alloc
        self.asset_alloc_rate = asset_alloc_rate
        self.rate_limit = rate_limit
        self.asset_alloc_bias = asset_alloc_bias
        self.other_args = other_args

    # 提取没有偏好的类型
    def get_no_like(self):
        lst = []
        for i in CLAZZ:
            if i not in self.clazzInterestOfFineTuning:
                lst += [i]
        return lst

    # 针对新进资金场景的设置
    def set_addmoney_config(self):
        userHolding = self.other_args['userHolding']
        holdAmt = userHolding['当前持仓金额'].sum()
        if self.amt - holdAmt < 0:
            return False
        for index in userHolding.index:
            if 'xj_fixed' in index.lower():
                self.ProductInfo.loc[index, '当前持仓金额'] = self.amt - holdAmt
                self.other_args['userHolding'].loc[index, '当前持仓金额'] = self.amt - holdAmt
                break
        return True

    # 设置产品配置
    def set_product_info(self):
        # 不偏好的大类不能买
        holdAmt = self.other_args['userHolding']['当前持仓金额'].sum()
        no_like = self.get_no_like()
        for name in no_like:
            if (name == '现金类') and (self.amt > holdAmt):
                # 如果有新增资金，则允许现金买入（盈余的部分配到现金上）
                continue
            self.ProductInfo.loc[self.ProductInfo[f'大类资产-{name}'] == 1, '是否可买入'] = False

        # 不偏好的大类，买入换手率提高
        for name in no_like:
            self.ProductInfo.loc[self.ProductInfo[f'大类资产-{name}'] == 1, '买入换手率'] = 0.5

        # 设置偏好大类的产品优先度
        self.ProductInfo['产品优先度'] = self.ProductInfo['产品优先度'].fillna(0.0)
        for clazz in self.clazzInterestOfFineTuning:
            self.ProductInfo.loc[self.ProductInfo[f'大类资产-{clazz}'] == 1, '产品优先度'] += 10.0
        return True

    # 设置大类配置
    def set_asset_alloc_constraint(self):
        userHolding = self.other_args['userHolding']
        holdAmt = userHolding['当前持仓金额'].sum()
        for iAlloc in self.other_args['asset_alloc_bias_detail']:
            if iAlloc not in self.clazzInterestOfFineTuning:
                # 下限约束调整
                self.other_args['asset_alloc_bias_detail'][iAlloc]['l_asset_alloc_bias'] = self.target_asset_alloc[
                                                                                               iAlloc] + 1e-3

        # 不偏好的大类，上限约束为当前持仓的范围，如果有新进资金，默认加到现金类
        if (self.amt > holdAmt) and (iAlloc == '现金类'):
            df = userHolding[userHolding['是否持仓'] == True]
            ratio = (df[df[f'大类资产-{iAlloc}'] == 1]['当前持仓金额'].sum() + self.amt - holdAmt) / self.amt
        else:
            df = userHolding[userHolding['是否持仓'] == True]
            ratio = df[df[f'大类资产-{iAlloc}'] == 1]['当前持仓金额'].sum() / self.amt

        u_asset_alloc_bias = self.other_args['asset_alloc_bias_detail'][iAlloc]['u_asset_alloc_bias']
        iTargetNew = max(ratio, self.target_asset_alloc[iAlloc])
        self.other_args['asset_alloc_bias_detail'][iAlloc]['u_asset_alloc_bias'] = max(
            iTargetNew - self.target_asset_alloc[iAlloc], u_asset_alloc_bias) + 1e-3
        return True

    def set_budget_constraint(self):
        return True

    # 判断是否还需要进行微调
    def finetuning_check(self):
        msg = []
        # 选择所有大类，不进行微调
        if set(CLAZZ) == set(self.clazzInterestOfFineTuning):
            return False

        # 所有不偏好的大类都超配，可以不进行微调
        no_like = self.get_no_like()
        userHolding = self.other_args['userHolding']
        lst = []
        for iAlloc in no_like:
            df = userHolding[userHolding['是否持仓'] == True]
            ratio = df[df[f'大类资产-{iAlloc}'] == 1]['当前持仓金额'].sum() / self.amt
            if ratio > self.target_asset_alloc[iAlloc]:
                lst += [iAlloc]
        if set(lst) == set(no_like):
            return False

        # todo: 如果银行的大类都已经超配，不需要再进行调增，方案无效，抛出提示
        return True

    def set_max_weight(self):
        # 设置单一产品的最大限额：放开限制
        for key in self.other_args['max_weight']:
            self.other_args['max_weight'][key] = 1.0

    def set_obj_alloc(self):
        self.obj_alloc['换手费率'] = max(self.obj_alloc.values()) * 100

    def set_params(self):
        if self.finetuning_check():
            self.set_addmoney_config()  # 要放在最前面
            self.set_product_info()
            self.set_asset_alloc_constraint()
            self.set_budget_constraint()
            self.set_max_weight()
            self.set_obj_alloc()
            return True
        else:
            return False

    def get_params(self):
        return (self.ProductInfo, self.other_args['asset_alloc_bias_detail'],
                self.other_args['weight_sum'], self.obj_alloc)

    def result_check(self, portfolio):
        # 微调方案检测，如果最偏好的大类没有增持，则方案无效
        pass
        a = 1
        df = pd.merge(self.ProductInfo, portfolio[['持仓金额']], left_index=True, right_index=True, how='left')
        df['持仓金额'] = df['持仓金额'].fillna(0.0)
        df['增减金额'] = df['持仓金额'] - df['当前持仓金额']
        isok = True
        for name in CLAZZ:
            if name not in self.clazzInterestOfFineTuning:
                continue
            delta = df[df[f'大类资产-{name}'] == 1]['增减金额'].sum()
            if delta <= 0 and self.target_asset_alloc[name] > 0:
                # 减持了，且 目标暴露比例大于0
                isok = False
        return isok
