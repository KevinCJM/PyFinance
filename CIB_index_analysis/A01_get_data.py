# -*- encoding: utf-8 -*-
"""
@File: A01_get_data.py
@Modify Time: 2025/9/9 10:34       
@Author: Kevin-Chen
@Descriptions: 
"""
import pandas as pd
import tushare as ts
from CIB_Brinsion_Campisi_demo.B01_category_brinsion_get_data import main_fetch_index_daily_return

# API 设置
ts.set_token('YOUR_TUSHARE_TOKEN')
pro = ts.pro_api()

# 获取中证指数基本信息
CSI_index_info = pro.index_basic(market='CSI')
CSI_index_info.to_excel('CSI_index_info.xlsx', index=False)
print(CSI_index_info)

# 获取中证指数日行情
start_date, end_date = '20050101', '20250909'
CSI_index_list = pd.read_excel('中证系列指数.xlsx')['指数代码'].tolist()
print(CSI_index_list)
CSI_index_return = main_fetch_index_daily_return(codes=CSI_index_list, start=start_date, end=end_date, max_workers=25)
CSI_index_return.to_parquet('CSI_index_return.parquet', index=False)

CSI_index_return = pd.read_parquet('CSI_index_return.parquet').reset_index(drop=True)
print(CSI_index_return)
