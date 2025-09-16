# -*- encoding: utf-8 -*-
"""
@File: A01_main_api.py
@Modify Time: 2025/9/16 11:05       
@Author: Kevin-Chen
@Descriptions: 
"""
import json

if __name__ == '__main__':
    # 字典格式入参
    dict_input = {
        'asset_list': [
            '货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类'
        ],
        'WeightRange': {
            'C1': {'货币现金类': [1.0, 1.0], '固定收益类': [0.0, 0.0], '混合策略类': [0.0, 0.0],
                   '权益投资类': [0.0, 0.0], '另类投资类': [0.0, 0.0]},
            'C2': {'货币现金类': [0.0, 1.0], '固定收益类': [0.0, 1.0], '混合策略类': [0.0, 0.0],
                   '权益投资类': [0.0, 0.0], '另类投资类': [0.0, 0.0]},
            'C3': {'货币现金类': [0.0, 1.0], '固定收益类': [0.0, 1.0], '混合策略类': [0.0, 0.5],
                   '权益投资类': [0.0, 0.0], '另类投资类': [0.0, 0.0]},
            'C4': {'货币现金类': [0.0, 1.0], '固定收益类': [0.0, 1.0], '混合策略类': [0.0, 0.6],
                   '权益投资类': [0.0, 0.2], '另类投资类': [0.0, 0.1]},
            'C5': {'货币现金类': [0.0, 1.0], '固定收益类': [0.0, 1.0], '混合策略类': [0.0, 0.8],
                   '权益投资类': [0.0, 0.5], '另类投资类': [0.0, 0.3]},
            'C6': {'货币现金类': [0.0, 1.0], '固定收益类': [0.0, 1.0], '混合策略类': [0.0, 1.0],
                   '权益投资类': [0.0, 0.7], '另类投资类': [0.0, 0.5]}
        },
        'StandardProportion': {
            'C1': {'货币现金类': 1.0, '固定收益类': 0.0, '混合策略类': 0.0, '权益投资类': 0.0, '另类投资类': 0.0},
            'C2': {'货币现金类': 0.2, '固定收益类': 0.8, '混合策略类': 0.0, '权益投资类': 0.0, '另类投资类': 0.0},
            'C3': {'货币现金类': 0.1, '固定收益类': 0.55, '混合策略类': 0.35, '权益投资类': 0.0, '另类投资类': 0.0},
            'C4': {'货币现金类': 0.05, '固定收益类': 0.4, '混合策略类': 0.3, '权益投资类': 0.2, '另类投资类': 0.05},
            'C5': {'货币现金类': 0.05, '固定收益类': 0.2, '混合策略类': 0.25, '权益投资类': 0.4, '另类投资类': 0.1},
            'C6': {'货币现金类': 0.05, '固定收益类': 0.1, '混合策略类': 0.15, '权益投资类': 0.6, '另类投资类': 0.1}
        },
        'user_holding': {
            'WeightRange': {
                '货币现金类': [0.0, 1.0], '固定收益类': [0.0, 1.0], '混合策略类': [0.0, 0.4],
                '权益投资类': [0.0, 0.7], '另类投资类': [0.0, 0.12]
            },
            'StandardProportion': {
                '货币现金类': 0.1, '固定收益类': 0.2, '混合策略类': 0.2, '权益投资类': 0.4, '另类投资类': 0.1
            },
            'can_sell': {
                '货币现金类': False, '固定收益类': True, '混合策略类': False, '权益投资类': True, '另类投资类': True
            },
            'can_buy': {
                '货币现金类': True, '固定收益类': True, '混合策略类': False, '权益投资类': True, '另类投资类': False
            }
        }
    }

    # 字典转Json
    json_str = json.dumps(dict_input, ensure_ascii=False)
    print(json_str)
    # Json转字典
    dict_input = json.loads(json_str)
    print(dict_input)
    # 分解参数
    asset_list = dict_input['asset_list']  # 大类列表
    weight_range = dict_input['WeightRange']  # 标准组合约束
    standard_proportion = dict_input['StandardProportion']  # 标准组合
    user_holding = dict_input['user_holding']  # 持仓组合
    user_holding_weight_range = user_holding['WeightRange']  # 客户所属风险等级的标准组合上下限约束
    user_holding_standard_proportion = user_holding['StandardProportion']  # 客户的当前持仓
    user_holding_can_sell = user_holding['can_sell']  # 哪些大类持仓客户允许卖出
    user_holding_can_buy = user_holding['can_buy']  # 哪些大类持仓客户允许买入
    print(asset_list)
    print(weight_range)
    print(standard_proportion)
    print(user_holding)
    print(user_holding_weight_range)
    print(user_holding_standard_proportion)
    print(user_holding_can_sell)
    print(user_holding_can_buy)
