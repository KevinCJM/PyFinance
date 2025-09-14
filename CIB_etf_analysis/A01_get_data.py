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
etf_info_df = pro.fund_basic(market='E')
etf_info_df.to_excel('etf_info_df.xlsx', index=False)
etf_info_dict = {i: j for i, j in zip(etf_info_df['ts_code'], etf_info_df['name'])}
print(etf_info_dict)

# 获取ETF日行情数据
res_list = []
for c, n in etf_info_dict.items():
    etf_daily_df = pro.fund_nav(ts_code=c)
    etf_daily_df = etf_daily_df[['ts_code', 'nav_date', 'unit_nav', 'accum_nav', 'adj_nav']]
    etf_daily_df['nav_date'] = pd.to_datetime(etf_daily_df['nav_date'])
    etf_daily_df['date'] = pd.to_datetime(etf_daily_df['nav_date'])
    etf_daily_df['name'] = n
    etf_daily_df = etf_daily_df.sort_values('nav_date').reset_index(drop=True)
    res_list.append(etf_daily_df)

etf_daily_df = pd.concat(res_list, axis=0).reset_index(drop=True)
etf_daily_df.to_parquet('etf_daily_df.parquet', index=False)
