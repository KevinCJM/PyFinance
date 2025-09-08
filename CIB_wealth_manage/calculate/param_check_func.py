# -*- encoding: utf-8 -*-
"""
@File: param_check_func.py
@Modify Time: 2025/8/29 15:21       
@Author: Kevin-Chen
@Descriptions: 
"""
import copy


class ParaAdj():
    def __init__(self, target_asset_alloc, asset_alloc_bias, longest_idle_time, amt, other_args):
        self.target_asset_alloc = target_asset_alloc
        self.asset_alloc_bias = asset_alloc_bias
        self.longest_idle_time = longest_idle_time
        self.amt = amt
        self.other_args = other_args

    def adj_alloc_by_holding_v0(self):
        # 持仓中无法减持的品种与大类约束检测
        asset_alloc_bias_detail = self.other_args['asset_alloc_bias_detail']
        asset_alloc_bias_detail_new = copy.deepcopy(asset_alloc_bias_detail)
        userHolding = self.other_args['userHolding']

        adj_alloc_lst = []
        remainder = 1.0

        for iAlloc, iTarget in self.target_asset_alloc.items():
            ratio = (
                    userHolding[(userHolding[f"大类资产-{iAlloc}"] == 1) & (userHolding["是否可卖出"] == False)][
                        "当前持仓金额"].sum() / self.amt
            )
            if ratio > iTarget + asset_alloc_bias_detail[iAlloc]['u_asset_alloc_bias']:
                # 如果不能减持的产品持仓比例超过了上限，则提高上限
                asset_alloc_bias_detail_new[iAlloc]['u_asset_alloc_bias'] = ratio - iTarget + 1e-3
                adj_alloc_lst.append(iAlloc)
                remainder -= ratio + 1e-3

        if len(adj_alloc_lst) > 0:
            sum_ = sum([iTarget for iAlloc, iTarget in self.target_asset_alloc.items()
                        if iAlloc not in set(adj_alloc_lst)])
            for iAlloc, iTarget in self.target_asset_alloc.items():
                if iAlloc in set(adj_alloc_lst):
                    continue
                asset_alloc_bias_detail_new[iAlloc]['u_asset_alloc_bias'] = max(
                    0,
                    min(iTarget + asset_alloc_bias_detail[iAlloc]['u_asset_alloc_bias'], iTarget / sum_) - iTarget
                ) + 1e-3

        return asset_alloc_bias_detail_new

    def adj_alloc_by_holding(self):
        # 持仓中无法减持的产品 与 大类约束检测
        asset_alloc_bias_detail = self.other_args['asset_alloc_bias_detail']
        asset_alloc_bias_detail_new = copy.deepcopy(asset_alloc_bias_detail)
        target_asset_alloc_new = copy.deepcopy(self.target_asset_alloc)
        userHolding = self.other_args['userHolding']

        valid_hold_amt = userHolding[userHolding['是否可卖出'] == True]['当前持仓金额'].sum()
        if (valid_hold_amt <= 100) or (self.amt <= 100):
            # 流动资金如果小于30万，则不再进行参数调整
            return target_asset_alloc_new, asset_alloc_bias_detail_new

        adj_alloc_lst = []
        remainder = 1.0
        for iAlloc, iTarget in self.target_asset_alloc.items():
            ratio = userHolding[(userHolding[f'大类资产-{iAlloc}'] == 1)
                                & (userHolding['是否可卖出'] == False)]['当前持仓金额'].sum() / self.amt

            if ratio > iTarget + asset_alloc_bias_detail[iAlloc]['u_asset_alloc_bias']:
                # 如果不能减持的产品持仓比例超过了上限，则提高上限
                target_asset_alloc_new[iAlloc] = ratio
                asset_alloc_bias_detail_new[iAlloc]['l_asset_alloc_bias'] = 1e-3
                asset_alloc_bias_detail_new[iAlloc]['u_asset_alloc_bias'] = 1e-3
                adj_alloc_lst.append(iAlloc)
                remainder -= ratio

        if len(adj_alloc_lst) > 0:
            # 优先满足现金类的大类目标比例
            target_asset_alloc_new['现金类'] = min(self.target_asset_alloc['现金类'], remainder)
            remainder -= target_asset_alloc_new['现金类']

            sum_ = sum([iTarget for iAlloc, iTarget in self.target_asset_alloc.items()
                        if (iAlloc != '现金类' and iAlloc not in set(adj_alloc_lst))])

            for iAlloc, iTarget in self.target_asset_alloc.items():
                if iAlloc in set(adj_alloc_lst):
                    continue
                if iAlloc == '现金类':
                    continue
                target_asset_alloc_new[iAlloc] = remainder * iTarget / sum_

        return target_asset_alloc_new, asset_alloc_bias_detail_new

    def adj_duration_by_holding(self):
        # 调整久期值
        userHolding = self.other_args['userHolding']
        isHolding = (userHolding['是否持仓'] == True)
        is_can_not_sell = (userHolding['是否可卖出'] == False)
        holdAmt = userHolding.loc[:, '当前持仓金额'].sum()

        if holdAmt > 1e-3:
            duration = (userHolding.loc[isHolding & is_can_not_sell, '当前持仓金额'].values / holdAmt) @ \
                       userHolding.loc[isHolding & is_can_not_sell, '持仓剩余期限'].values
        else:
            duration = 0.0

        return max(self.longest_idle_time, duration + 1)


# 参数检测，用于判断是否需要进行资产配置优化
def main_param_check(userHolding, target_asset_alloc, AssetAllocBias, other_args):
    # 1. 大类资产配置是否在基准区间范围内
    # 2. 单一产品集中度超过门限
    # 3. 持仓久期大于基准久期
    # 4. 产品数量

    return False
