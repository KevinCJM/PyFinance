# -*- encoding: utf-8 -*-
"""
@File: A01_main_api.py
@Modify Time: 2025/9/16 11:05       
@Author: Kevin-Chen
@Descriptions: 
"""
import json
from T02_other_tools import load_returns_from_excel
from T01_generate_random_weights import multi_level_random_walk_config
from T03_weight_limit_cal import level_weight_limit_cal, hold_weight_limit_cal

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

    ''' 0) 准备工作: 模拟json参数输入 & 模拟大类收益率输入 '''
    # 字典转Json, 模拟输入的Json参数
    json_str = json.dumps(dict_input, ensure_ascii=False)
    print(json_str)
    # 读取excel，生成日收益二维数组
    excel_path = '历史净值数据_万得指数.xlsx'
    excel_sheet = '历史净值数据'
    returns, assets = load_returns_from_excel(excel_path, excel_sheet, asset_list)

    ''' 1) 解析Json参数 '''
    # Json转字典
    dict_input = json.loads(json_str)
    # 分解参数
    asset_list = dict_input['asset_list']  # 大类列表
    weight_range = dict_input['WeightRange']  # 标准组合约束
    standard_proportion = dict_input['StandardProportion']  # 标准组合
    user_holding = dict_input['user_holding']  # 客户持仓组合

    ''' 2) 计算约束 '''
    # 计算标准组合的约束
    level_weight_limit = {}
    for k, v in weight_range.items():
        single_limit, multi_limit = level_weight_limit_cal(asset_list, v)
        level_weight_limit[k] = {'single_limit': single_limit, 'multi_limit': multi_limit}
    print("标准组合的约束：", level_weight_limit)
    # 计算客户持仓的约束
    single_limit_hold, multi_limit_hold = hold_weight_limit_cal(asset_list, user_holding)
    hold_weight_limit = {'single_limit': single_limit_hold, 'multi_limit': multi_limit_hold}
    print("客户持仓的约束：", hold_weight_limit)

    ''' 3) 计算标准组合的随机权重和有效前沿 '''
    # 循环计算各个标准组合的随机权重以及权重对应的收益率和波动率
