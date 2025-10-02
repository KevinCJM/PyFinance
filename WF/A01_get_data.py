# -*- encoding: utf-8 -*-
"""
@File: A01_get_data.py
@Modify Time: 2025/8/19 17:07       
@Author: Kevin-Chen
@Descriptions: 
"""

import akshare as ak
import pandas as pd

code_list = ['300008', '300106', '300405', '300752', '600121', '600570']
s_date = '20000101'
e_date = pd.to_datetime("today").strftime("%Y%m%d")

for code in code_list:
    ''' 获取个股日频行情数据 '''
    stock_zh_a_hist_df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=s_date,
        end_date=pd.to_datetime("today").strftime("%Y%m%d"),
        adjust="qfq"
    )
    stock_zh_a_hist_df.to_parquet(f"{code}.parquet", index=False)
    print(stock_zh_a_hist_df)

    ''' 获取个股详细信息 '''
    stock_individual_info_em_df = ak.stock_individual_info_em(symbol=code)
    print(stock_individual_info_em_df)
