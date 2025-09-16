# -*- encoding: utf-8 -*-
"""
@File: T03_weight_limit_cal.py
@Modify Time: 2025/9/16 10:27       
@Author: Kevin-Chen
@Descriptions: 权重约束计算
"""
from typing import Any, Dict, Iterable, List, Tuple


def level_weight_limit_cal(asset_list, level_limit):
    """
    根据给定的资产列表和等级限制，计算单资产的权重上下限。

    :param asset_list: 资产列表
    :param level_limit: 包含各资产权重上下限的字典
    :return: 单资产上下限列表和多资产联合约束字典
    """
    # # 单资产上下限（示例：全部 [0, 1]）
    # single_limit_res: List[Tuple[float, float]] = [(0.0, 1.0)] * len(asset_list)
    # # 多资产联合约束（示例为空；可设如 {(0,1):(0.2,0.6)}）
    # multi_limit_res: Dict[Tuple[int, ...], Tuple[float, float]] = {}
    single_limit = [level_limit[asset] for asset in asset_list]
    multi_limit = {}
    return single_limit, multi_limit


def hold_weight_limit_cal(asset_list, user_holding):
    """
    根据用户持仓、买卖意愿和风险等级，计算各资产的权重上下限。

    :param asset_list: 资产列表
    :param user_holding: 包含用户持仓、意愿和风险等级的字典
    :return: 单资产上下限列表和多资产联合约束字典
    """
    # # 单资产上下限（示例：全部 [0, 1]）
    # single_limit_res: List[Tuple[float, float]] = [(0.0, 1.0)] * len(asset_list)
    # # 多资产联合约束（示例为空；可设如 {(0,1):(0.2,0.6)}）
    # multi_limit_res: Dict[Tuple[int, ...], Tuple[float, float]] = {}
    single_limit, multi_limit = [], {}

    # 识别风险等级key
    risk_level_key = None
    for key in user_holding:
        if key not in ['holding', 'can_sell', 'can_buy']:
            risk_level_key = key
            break
    if not risk_level_key:
        raise ValueError("在user_holding中找不到风险等级key")

    level_limits = user_holding[risk_level_key]
    holdings = user_holding['holding']
    can_sell_rules = user_holding['can_sell']
    can_buy_rules = user_holding['can_buy']

    for asset in asset_list:
        base_min, base_max = level_limits[asset]
        current_weight = holdings[asset]
        can_sell = can_sell_rules[asset]
        can_buy = can_buy_rules[asset]

        min_w = base_min
        max_w = base_max

        if not can_sell:
            min_w = max(min_w, current_weight)
        if not can_buy:
            max_w = min(max_w, current_weight)

        single_limit.append((min_w, max_w))

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
    print("标准组合C1的单资产约束：", single_limit_c1)
    print("标准组合C1的多资产约束：", multi_limit_c1)
    single_limit_c6, multi_limit_c6 = level_weight_limit_cal(the_asset_list, c6)
    print("标准组合C6的单资产约束：", single_limit_c6)
    print("标准组合C6的多资产约束：", multi_limit_c6)

    # 计算客户持仓约束
    single_limit_hold, multi_limit_hold = hold_weight_limit_cal(the_asset_list, the_user_holding)
    print("客户持仓的单资产约束：", single_limit_hold)
    print("客户持仓的多资产约束：", multi_limit_hold)
    