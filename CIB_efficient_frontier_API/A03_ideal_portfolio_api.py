# -*- encoding: utf-8 -*-
"""
@File: A03_ideal_portfolio_api.py
@Modify Time: 2025/9/17 08:05       
@Author: Kevin-Chen
@Descriptions: 根据已有持仓寻找组合配置:  风险不变收益最大,收益不变风险最小,持仓组合变动最小
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from T04_show_plt import plot_efficient_frontier
from T02_other_tools import load_returns_from_excel
from T01_generate_random_weights import compute_perf_arrays, compute_var_parametric_arrays
from T03_weight_limit_cal import hold_weight_limit_cal


def analysis_json_and_read_data(json_input, excel_name, sheet_name):
    params = json.loads(json_input)
    asset_list = params['asset_list']
    draw_plt = params.get('draw_plt', True)
    draw_plt_filename = params.get('draw_plt_filename', None)
    user_holding = params['user_holding']
    ef_data = params['ef_data']
    returns, assets = load_returns_from_excel(excel_name, sheet_name, asset_list)
    return asset_list, draw_plt, draw_plt_filename, user_holding, ef_data, returns


if __name__ == '__main__':
    ''' 准备工作: 模拟json参数输入 ------------------------------------------------------------------------------ '''
    # 读json文件
    with open('A03_input.json', 'r', encoding='utf-8') as f:
        json_str = f.read()
    # excel信息
    excel_path = '历史净值数据_万得指数.xlsx'
    excel_sheet = '历史净值数据'
    print(json_str)

    ''' 1. 解析json参数 & 读取excel数据 --------------------------------------------------------------------------- '''
    (asset_list, draw_plt, draw_plt_filename, user_holding, ef_data,
     returns) = analysis_json_and_read_data(json_str, excel_path, excel_sheet)

    ''' 2) 计算约束 ---------------------------------------------------------------------------------------------- '''
    single_limit, multi_limit = hold_weight_limit_cal(asset_list, user_holding)
    print(f"单层约束: {single_limit}; 多层约束: {multi_limit}")

    ''' 3) 计算推荐组合配置方案 ------------------------------------------------------------------------------------ '''
    ''' 3.1 计算客户当前持仓的风险和收益 '''
    ''' 3.2 计算客户当前持仓同等收益下风险最小的有效前沿上的点 '''
    ''' 3.3 计算客户当前持仓同等风险下收益最大的有效前沿上的点 '''
    ''' 3.4 计算与客户当前持仓换仓最小的有效前沿上的点 '''
