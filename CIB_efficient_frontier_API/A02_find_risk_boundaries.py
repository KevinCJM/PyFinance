# -*- encoding: utf-8 -*-
"""
@File: A02_find_risk_boundaries.py
@Modify Time: 2025/9/16 19:02       
@Author: Kevin-Chen
@Descriptions: 寻找在指定大类约束下的风险边界
"""
import json
import pandas as pd
import numpy as np

from T04_show_plt import plot_efficient_frontier
from T02_other_tools import load_returns_from_excel
from T01_generate_random_weights import multi_level_random_walk_config, compute_perf_arrays, \
    compute_var_parametric_arrays
from T03_weight_limit_cal import level_weight_limit_cal, hold_weight_limit_cal


# 解析Json参数 & 读取大类收益率
def analysis_json_and_read_data(json_input, excel_name, sheet_name):
    # Json转字典
    json_dict = json.loads(json_input)
    # 分解参数
    asset_list = json_dict['asset_list']  # 大类列表
    draw_plt = json_dict.get('draw_plt', None)  # 是否绘图展示
    draw_plt_filename = json_dict.get('draw_plt_filename', None)  # 绘图保存文件名, None表示不保存直接显示
    weight_range = json_dict.get('WeightRange', None)  # 标准组合约束
    # 读取excel，生成日收益二维数组
    returns, assets = load_returns_from_excel(excel_name, sheet_name, asset_list)
    return asset_list, draw_plt, draw_plt_filename, weight_range, returns


if __name__ == '__main__':
    ''' 准备工作: 模拟json参数输入 ------------------------------------------------------------------------------- '''
    # 字典格式入参
    dict_input = {
        'asset_list': [
            '货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类'
        ],
        'WeightRange': {  # 大类权重上下限
            '货币现金类': [0.05 * 0.8, 0.05 * 1.2], '固定收益类': [0.10 * 0.8, 0.10 * 1.2],
            '混合策略类': [0.15 * 0.8, 0.15 * 1.2], '权益投资类': [0.60 * 0.8, 0.60 * 1.2],
            '另类投资类': [0.10 * 0.8, 0.10 * 1.2]
        },
        'draw_plt': True,  # 是否绘图展示
        'draw_plt_filename': None,  # 绘图保存文件名，None表示不保存直接显示
    }
    # excel信息
    excel_path = '历史净值数据_万得指数.xlsx'
    excel_sheet = '历史净值数据'
    # 字典转Json, 模拟输入的Json参数
    json_str = json.dumps(dict_input, ensure_ascii=False)
    print(json_str)

    ''' 1. 处理Json & 读取数据 ----------------------------------------------------------------------------------- '''
    (asset_list, draw_plt, draw_plt_filename, weight_range,
     returns) = analysis_json_and_read_data(json_str, excel_path, excel_sheet)

    ''' 2) 计算约束 ----------------------------------------------------------------------------------------------- '''
    single_limit, multi_limit = level_weight_limit_cal(asset_list, weight_range)
    print(f"单层约束: {single_limit}; 多层约束: {multi_limit}")

    ''' 3) 计算两端风险水平 ---------------------------------------------------------------------------------------- '''
    # 生成随机权重 (用于SLSQP的热启动)
    # 计算最小风险组合
    # 计算最大风险组合