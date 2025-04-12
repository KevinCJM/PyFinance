# -*- encoding: utf-8 -*-
"""
@File: Get_Data.py
@Modify Time: 2025/4/12 11:52       
@Author: Kevin-Chen
@Descriptions: 数据获取与准备
"""

from GetData.tushare_get_ETF_data import get_etf_daily_data_all, get_etf_daily_data_increment
from GetData.data_prepare import data_prepare

get_all = True  # 是否获取所有数据

if get_all:
    # 获取所有ETF数据
    get_etf_daily_data_all()
    # 数据预处理
    data_prepare()
else:
    # 增量获取所有ETF数据
    get_etf_daily_data_increment()
    # 数据预处理
    data_prepare()
