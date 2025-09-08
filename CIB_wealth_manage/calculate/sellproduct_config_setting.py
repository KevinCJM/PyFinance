# -*- encoding: utf-8 -*-
"""
@File: sellproduct_config_setting.py
@Modify Time: 2025/8/28 19:22       
@Author: Kevin-Chen
@Descriptions: 
"""
# 针对营销场景的参数配置
from config_py import ASSET_CODE_DICT, CLAZZ_ORDER

CLAZZ = list(ASSET_CODE_DICT.keys())


class SellProductConfigSetting(object):
    def __init__(
            self,
            ProductInfo,
            amt,
            obj_alloc,
            target_asset_alloc,
            longest_idle_time,
            asset_alloc_rate,
            rate_limit,
            asset_alloc_bias,
            other_args,
    ):
        self.ProductInfo = ProductInfo
        self.amt = amt
        self.obj_alloc = obj_alloc
        self.target_asset_alloc = target_asset_alloc
        self.longest_idle_time = longest_idle_time
        self.asset_alloc_rate = asset_alloc_rate
        self.rate_limit = rate_limit
        self.asset_alloc_bias = asset_alloc_bias
        self.other_args = other_args

    # 设置产品集
    def set_product_info(self):
        pass
        return True

    # 设置大类配置
    def set_asset_alloc_constraint(self):
        for iAsset, iTarget in self.target_asset_alloc.items():
            self.other_args['asset_alloc_bias_detail'][iAsset]['l_asset_alloc_bias'] = iTarget
            self.other_args['asset_alloc_bias_detail'][iAsset]['u_asset_alloc_bias'] = (
                1.0 / iTarget - 1 if iTarget > 1e-6 else 1.0)
        return True

    def set_budget_constraint(self):
        pass
        return True

    def set_min_max_num(self):
        self.other_args['min_num'] = 1
        self.other_args['max_num'] = 1e6

    def set_max_weight(self):
        # 设置单一产品集中度限制：放开限制
        for key in self.other_args['max_weight']:
            self.other_args['max_weight'][key] = 1.0

    def set_duration_limit(self):
        self.longest_idle_time = 1e8

    def set_obj_alloc(self):
        self.obj_alloc['产品评级'] = max(self.obj_alloc.values()) * 100

    def set_params(self):
        self.set_product_info()
        self.set_asset_alloc_constraint()
        self.set_budget_constraint()
        self.set_max_weight()
        self.set_duration_limit()
        self.set_min_max_num()
        return True

    def get_params(self):
        return (
            self.ProductInfo,
            self.other_args['asset_alloc_bias_detail'],
            self.other_args['weight_sum'],
            self.obj_alloc,
        )
