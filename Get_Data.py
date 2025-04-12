# -*- encoding: utf-8 -*-
"""
@File: Get_Data.py
@Modify Time: 2025/4/12 11:52       
@Author: Kevin-Chen
@Descriptions: 数据获取与准备
"""
import os

from GetData.tushare_get_ETF_data import get_etf_daily_data_all, get_etf_daily_data_increment
from GetData.data_prepare import data_prepare

daily_etf_file = "./Data/etf_daily.parquet"  # ETF日频数据文件名
wild_df_save_path = "./Data/"  # 预处理后的一系列宽表数据帧存放文件夹

# 判断 daily_etf_file 文件是否存在, 若不存在则全量取数
if not os.path.exists(daily_etf_file):
    print("获取所有ETF日频数据 (全量取数)")
    # 获取所有ETF数据
    get_etf_daily_data_all(daily_etf_file)
    # 数据预处理
    data_prepare(read_file=daily_etf_file,
                 save_path=wild_df_save_path
                 )
# 若文件已经存在则增量取数
else:
    print("获取所有ETF日频数据 (增量取数)")
    # 增量获取所有ETF数据
    get_etf_daily_data_increment(daily_etf_file)
    # 数据预处理
    data_prepare(read_file=daily_etf_file,
                 save_path=wild_df_save_path
                 )
