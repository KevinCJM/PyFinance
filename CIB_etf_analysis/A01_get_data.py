# -*- encoding: utf-8 -*-
"""
@File: A01_get_data.py
@Modify Time: 2025/9/9 15:40       
@Author: Kevin-Chen
@Descriptions: 
"""
import pandas as pd
import tushare as ts
from CIB_Brinsion_Campisi_demo.B01_category_brinsion_get_data import main_fetch_index_daily_return

# API 设置
ts.set_token('cdcff0dd57ef63b6e9a347481996ea8f555b0aae35088c9b921a06c9')
pro = ts.pro_api()

# 获取ETF列表
# etf_info_df = pro.fund_basic(market='E')
# etf_info_df.to_excel('etf_info_df.xlsx', index=False)
# print(etf_info_df)

res_list = []
code_dict = {'162207.SZ': '混合类',  # 混合
             '161017.SZ': '权益类',  # 500增强LOF, 股票
             '159001.SZ': '货基指数',  # 货币ETF
             '161216.SZ': '固收类',  # 国投双债LOF
             '159934.SZ': '另类',
             }
# 获取ETF日行情数据
for c, n in code_dict.items():
    etf_daily_df = pro.fund_nav(ts_code=c)
    etf_daily_df = etf_daily_df[['ts_code', 'nav_date', 'unit_nav', 'accum_nav', 'adj_nav']]
    etf_daily_df['nav_date'] = pd.to_datetime(etf_daily_df['nav_date'])
    etf_daily_df['date'] = pd.to_datetime(etf_daily_df['nav_date'])
    etf_daily_df['type'] = n
    etf_daily_df = etf_daily_df.sort_values('nav_date').reset_index(drop=True)
    print(etf_daily_df)
    res_list.append(etf_daily_df)

etf_daily_df = pd.concat(res_list, axis=0).reset_index(drop=True)
etf_daily_df.to_excel('etf_daily_df.xlsx', index=False)
etf_daily_df = pd.read_excel('etf_daily_df.xlsx')
etf_daily_df = etf_daily_df.pivot_table(index='date', columns='type', values='adj_nav')
etf_daily_df = etf_daily_df.dropna(axis=0)
etf_daily_df.to_excel('etf_daily_df_piv.xlsx')
print(etf_daily_df)
