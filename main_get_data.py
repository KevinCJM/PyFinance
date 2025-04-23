# -*- encoding: utf-8 -*-
"""
@File: main_get_data.py
@Modify Time: 2025/4/12 11:52       
@Author: Kevin-Chen
@Descriptions: 主程序: 数据获取与准备
"""
import os
from GetData.data_prepare import data_prepare
from GetData.tushare_get_ETF_data import get_etf_daily_data_all, get_etf_daily_data_increment
from GetData.akshare_get_INDEX_data import akshare_index_main

'''
本文件有以下功能:
1) 从 Tushare 获取 ETF 日频数据 (自动增量/全量判断取数)
    具体数据获取的代码位于 GetData/tushare_get_ETF_data.py 文件中.
    目前仅仅支持 Tushare, 使用前需要在 set_tushare.py 中设置你的 Tushare token.
    获取的数据将以 dataframe 格式保存到 Data 文件夹下. 名称为 etf_daily.parquet.
    etf_daily.parquet 的 dataframe 包括以下字段: 
        ['ts_code', 'trade_date', 'pre_close', 'open', 'high', 'low', 'close', 'change', 'pct_chg', 'vol', 'amount']
    你如果没有 Tushare 的权限. 你可以自行准备上述格式的 dataframe 数据并保存为 etf_daily.parquet.
2) 对 ETF 日频数据进行预处理, 生成一系列宽表数据帧
    具体预处理的逻辑代码位于 GetData/data_prepare.py 文件中.
    预处理后的数据将以 dataframe 格式保存到 Data 文件夹下. 名称为 wide_***_df.parquet.
    预处理后的数据都是索引为日期, 字段为ETF代码的宽表数据帧.
3) 从AkShare 获取各类指数数据
    具体数据获取的代码位于 GetData/akshare_get_INDEX_data.py 文件中.
    目前仅仅支持 AkShare, AkShare 为免费接口, 无需设置 token.
    获取的数据将以 dataframe 格式保存到 Data/Index 文件夹下. 名称为 global_xxx_daily.parquet 或 china_xxx_daily.parquet.
    global_xxx_daily.parquet 和 china_xxx_daily.parquet 的字段并不统一
'''

daily_etf_file = "./Data/etf_daily.parquet"  # ETF日频数据文件名
wild_df_save_path = "./Data/"  # 预处理后的一系列宽表数据帧存放文件夹
index_df_save_path = "./Data/Index"  # 指数数据存放文件夹

# 判断 daily_etf_file 文件是否存在, 若不存在则全量取数
if not os.path.exists(daily_etf_file):
    print("获取所有ETF日频数据 (全量取数)")
    # 获取所有ETF数据
    get_etf_daily_data_all(daily_etf_file)
    # 数据预处理
    data_prepare(min_data_req=500,  # 最小数据量要求, 低于这个数据量的ETF会被剔除
                 read_file=daily_etf_file,  # 读取文件
                 save_path=wild_df_save_path  # 保存文件路径
                 )
# 若文件已经存在则增量取数
else:
    print("获取所有ETF日频数据 (增量取数)")
    # 增量获取所有ETF数据
    get_etf_daily_data_increment(daily_etf_file)
    # 数据预处理
    data_prepare(min_data_req=500,
                 read_file=daily_etf_file,
                 save_path=wild_df_save_path
                 )

# 从AkShare获取各类指数数据
akshare_index_main('index_df_save_path')
