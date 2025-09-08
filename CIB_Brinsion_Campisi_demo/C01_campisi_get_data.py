# -*- encoding: utf-8 -*-
"""
@File: C01_campisi_get_data.py
@Modify Time: 2025/9/8 14:31       
@Author: Kevin-Chen
@Descriptions: Campisi归因, 取数
"""

import pandas as pd


def convert_duration(s):
    if '月' in s:
        return float(s.replace('月', '')) / 12
    elif '年' in s:
        return float(s.replace('年', ''))
    else:
        return None


if __name__ == '__main__':
    ''' ------------------ 基金大类信息 ------------------ '''
    fund_info_df = pd.read_parquet('./data/fund_info.parquet')
    _ = {'fund_code': '基金代码',
         'fund_name': '基金名称',
         'fund_type': '基金类型',  # '债券型', '商品型', '混合型', '股票型', '货币市场型'
         'fund_company': '基金公司'
         }
    print(fund_info_df)
    print(set(fund_info_df['fund_type']))
    print("基金信息数据获取完成 ... ")

    ''' ------------------ 最近两个报告期 ------------------ '''
    start = "20241231"
    end = "20250630"

    ''' ------------------ FOF组合持仓权重数据 ------------------ '''
    # 示例数据, 权重和小于等于1
    fof_holding = {
        '016295': 0.12,  # 债券型
        '217025': 0.07,  # 债券型
        '012240': 0.21,  # 债券型
        '018959': 0.10,  # 债券型
        'test1': 0.14,  # 非债券型
        'test2': 0.11,  # 非债券型
        'test3': 0.15,  # 非债券型
    }
    fof_holding_df = pd.DataFrame(fof_holding.items(), columns=['fund_code', 'weight'])
    _ = {'fund_code': '基金代码',
         'weight': '权重'
         }
    fof_holding_df.to_parquet('./data/fof_holding_campisi.parquet', index=False)
    print(fof_holding_df)
    print("FOF组合持仓权重数据获取完成 ... ")

    ''' ------------------ 当期期间数据 ------------------ '''
    # 从2025/06/30的公告数据, 获取 2024/13/31 ~ 2025/06/30 的区间内数据
    in_p_data = {
        "016295": [10, 150, 120],  # 利息收入, 债券投资收入, 债券公允价值变动
        "217025": [12, 100, -20],
        "012240": [8, 140, 180],
        "018959": [9, 145, -40],
    }
    data_list = []
    for fund_code, values in in_p_data.items():
        row = [fund_code] + values
        data_list.append(row)
    in_p_data_df = pd.DataFrame(data_list,
                                columns=['fund_code', 'interest_income', 'bond_invest_income', 'fair_value_change'])
    in_p_data_df['date'] = end
    _ = {'date': '报告日期',
         'fund_code': '基金代码',
         'interest_income': '利息收入',
         'bond_invest_income': '债券投资收入',
         'fair_value_change': '债券公允价值变动'
         }
    in_p_data_df.to_parquet('./data/in_p_data_campisi.parquet', index=False)
    print(in_p_data_df)
    print(f"{end}报告的数据获取完成 ... ")

    ''' ------------------ 期初数据 ------------------ '''
    # 从2024/12/31的公告数据, 获取 2024/12/31 的期初数据
    start_p_data = {
        "016295": [1000, 4.5],  # 期初债券市值, 基金期初久期
        "217025": [800, 3.2],
        "012240": [600, 5.1],
        "018959": [1200, 4.8],
    }
    data_list = []
    for fund_code, values in start_p_data.items():
        row = [fund_code] + values
        data_list.append(row)
    start_p_data_df = pd.DataFrame(data_list, columns=['fund_code', 'start_bond_mv', 'start_fund_duration'])
    start_p_data_df['date'] = start
    _ = {'date': '报告日期',
         'fund_code': '基金代码',
         'start_bond_mv': '期初债券市值',
         'start_fund_duration': '基金期初久期(年)'
         }
    start_p_data_df.to_parquet('./data/start_p_data_campisi.parquet', index=False)
    print(start_p_data_df)
    print(f"{start}报告的数据获取完成 ... ")

    ''' ------------------ 国债收益率曲线 ------------------ '''
    start_yield = {
        'date': [start] * 9,  # 日期列，所有行都相同
        'duration': ['3月', '6月', '1年', '2年', '3年', '5年', '7年', '10年', '30年'],
        'yield': [0.91, 0.96, 1.08, 1.14, 1.19, 1.42, 1.59, 1.68, 1.91]
    }
    end_yield = {
        'date': [end] * 9,  # 日期列，所有行都相同
        'duration': ['3月', '6月', '1年', '2年', '3年', '5年', '7年', '10年', '30年'],
        'yield': [1.32, 1.34, 1.34, 1.36, 1.40, 1.51, 1.61, 1.65, 1.86]
    }
    yield_df = pd.concat([pd.DataFrame(start_yield), pd.DataFrame(end_yield)])
    # 将duration转为float格式,单位为年
    yield_df['duration'] = yield_df['duration'].apply(convert_duration)
    yield_df[['duration', 'yield']] = yield_df[['duration', 'yield']].astype(float)
    _ = {
        'date': '日期',
        'duration': '期限(年)',
        'yield': '收益率(%)'
    }
    yield_df.to_parquet('./data/yield_curve_campisi.parquet', index=False)
    print(yield_df)
    print(f"国债收益率曲线数据获取完成 ... ")
