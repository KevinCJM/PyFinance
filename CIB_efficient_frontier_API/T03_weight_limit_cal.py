# -*- encoding: utf-8 -*-
"""
@File: T03_weight_limit_cal.py
@Modify Time: 2025/9/16 10:27       
@Author: Kevin-Chen
@Descriptions: 权重约束计算
"""
from typing import Any, Dict, Iterable, List, Tuple


def level_weight_limit_cal(asset_list, level_limit):
    # # 单资产上下限（示例：全部 [0, 1]）
    # single_limit_res: List[Tuple[float, float]] = [(0.0, 1.0)] * len(asset_list)
    # # 多资产联合约束（示例为空；可设如 {(0,1):(0.2,0.6)}）
    # multi_limit_res: Dict[Tuple[int, ...], Tuple[float, float]] = {}
    single_limit, multi_limit = [], {}
    return single_limit, multi_limit


def hold_weight_limit_cal(asset_list, user_holding):
    # # 单资产上下限（示例：全部 [0, 1]）
    # single_limit_res: List[Tuple[float, float]] = [(0.0, 1.0)] * len(asset_list)
    # # 多资产联合约束（示例为空；可设如 {(0,1):(0.2,0.6)}）
    # multi_limit_res: Dict[Tuple[int, ...], Tuple[float, float]] = {}
    single_limit, multi_limit = [], {}
    return single_limit, multi_limit


if __name__ == '__main__':
    ''' 1) 输入参数 '''
    # 大类列表
    the_asset_list = ["货币现金类", "固定收益类", "混合策略类", "权益投资类", "另类投资类"]
    # 标准组合
    c1 = {'货币现金类': (1.00, 1.00),
          '固定收益类': (0.00, 0.00),
          '混合策略类': (0.00, 0.00),
          '权益投资类': (0.00, 0.00),
          '另类投资类': (0.00, 0.00)
          }
    c6 = {'货币现金类': (0.00, 1.00),
          '固定收益类': (0.00, 1.00),
          '混合策略类': (0.00, 0.4),
          '权益投资类': (0.00, 0.7),
          '另类投资类': (0.00, 0.12)
          }
    # 持仓组合
    the_user_holding = {
        'C6': {  # 客户所属风险等级的标准组合上下限约束
            '货币现金类': (0.00, 1.00),
            '固定收益类': (0.00, 1.00),
            '混合策略类': (0.00, 0.4),
            '权益投资类': (0.00, 0.7),
            '另类投资类': (0.00, 0.12)
        },
        'holding': {  # 客户的当前持仓
            '货币现金类': 0.1,
            '固定收益类': 0.2,
            '混合策略类': 0.2,
            '权益投资类': 0.4,
            '另类投资类': 0.1
        },
        'can_sell': {  # 哪些大类持仓客户允许卖出
            '货币现金类': False,
            '固定收益类': True,
            '混合策略类': False,
            '权益投资类': True,
            '另类投资类': True
        },
        'can_buy': {  # 哪些大类持仓客户允许买入
            '货币现金类': True,
            '固定收益类': True,
            '混合策略类': False,
            '权益投资类': True,
            '另类投资类': False
        }
    }

    ''' 2) 计算约束 '''
    # 计算大类约束
    single_limit_c1, multi_limit_c1 = level_weight_limit_cal(the_asset_list, c1)
    single_limit_c6, multi_limit_c6 = level_weight_limit_cal(the_asset_list, c6)
    # 计算客户持仓约束
    single_limit_hold, multi_limit_hold = hold_weight_limit_cal(the_asset_list, the_user_holding)
