# -*- encoding: utf-8 -*-
"""
@File: main_cal_metrics.py
@Modify Time: 2025/4/12 14:26       
@Author: Kevin-Chen
@Descriptions: 主程序: 计算各类指标
"""
import time
import pandas as pd
from MetricsFactory.metrics_factory import compute_metrics_for_period_initialize, compute_all_rolling_metrics

'''
本文件用于计算各类指标,
具体计算了什么指标什么区间, 以及指标的含义和简要公式可以查看 MetricsFactory/metrics_cal_config.py 文件; 
指标计算的逻辑代码位于 MetricsFactory/metrics_cal.py 文件; 
如果想要增加自己的指标, 可以在 metrics_cal_config.py 中配置, 然后在 metrics_cal.py 中加入逻辑代码.
指标计算的调用框架位于 MetricsFactory/metrics_factory.py 文件中, 支持单进程和多进程计算. 其中多进程计算使用了共享内存.
'''

# 配置路径
log_return_file = "./Data/wide_log_df.parquet"  # ETF日频对数收益率文件路径
open_price_file = "./Data/wide_open_df.parquet"  # ETF日频开盘价文件路径
close_price_file = "./Data/wide_close_df.parquet"  # ETF日频收盘价文件路径
high_price_file = "./Data/wide_high_df.parquet"  # ETF日频收盘价文件路径
low_price_file = "./Data/wide_low_df.parquet"  # ETF日频收盘价文件路径
volume_file = "./Data/wide_close_df.parquet"  # ETF日频收盘价文件路径
save_path = "./Data/Metrics"  # 计算完成的指标,存放的文件夹路径

log_return_df = pd.read_parquet(log_return_file)  # 读取ETF日频对数收益率数据
close_price_df = pd.read_parquet(close_price_file)  # 读取ETF日频收盘价数据
open_price_df = pd.read_parquet(open_price_file)  # 读取ETF日频开盘价数据
high_price_df = pd.read_parquet(high_price_file)  # 读取ETF日频收盘价数据
low_price_df = pd.read_parquet(low_price_file)  # 读取ETF日频收盘价数据
volume_df = pd.read_parquet(volume_file)  # 读取ETF日频收盘价数据

# 指定要计算的基金 (所有基金都要计算的话写None)
fund = ['510050.SH',    # 上证50ETF
        '159915.SZ',    # 创业板ETF
        '159912.SZ',    # 沪深300ETF
        '512500.SH',    # 中证500ETF华夏
        '511010.SH',    # 国债ETF
        '513100.SH',    # 纳指ETF
        '513030.SH',    # 德国ETF
        '513080.SH',    # 法国CAC40ETF
        '513520.SH',    # 日经ETF
        '518880.SH',    # 黄金ETF
        '161226.SZ',    # 国投白银LOF
        '501018.SH',    # 南方原油LOF
        '159981.SZ',    # 能源化工ETF
        '159985.SZ',    # 豆粕ETF
        '159980.SZ',    # 有色ETF
        ]

if __name__ == '__main__':
    # 计算区间指标
    compute_metrics_for_period_initialize(
        log_return_df,  # 对数收益率数据
        close_price_df,  # 收盘价数据
        high_price_df,  # 收盘价数据
        low_price_df,  # 收盘价数据
        volume_df,  # 收盘价数据
        save_path,  # 指标的保存文件路径
        p_list=None,  # 可以计算指定的区间列表, list格式
        metrics_list=None,  # 可以计算指定的指标列表, list格式
        fund_list=fund,  # 可以指定要计算的ETF列表, list格式
        spec_end_date=None,  # 可以指定的区间结束日期
        multi_process=True  # 是否使用多进程计算, True为使用多进程, False为单进程
    )

    # 计算滚动指标
    compute_all_rolling_metrics(
        open_price_df=open_price_df,  # ETF日频开盘价数据，DataFrame格式，包含多只ETF的日频开盘价信息
        close_price_df=close_price_df,  # ETF日频收盘价数据，DataFrame格式，包含多只ETF的日频收盘价信息
        high_price_df=high_price_df,  # ETF日频最高价数据，DataFrame格式，包含多只ETF的日频最高价信息
        low_price_df=low_price_df,  # ETF日频最低价数据，DataFrame格式，包含多只ETF的日频最低价信息
        volume_df=volume_df,  # ETF日频成交量数据，DataFrame格式，包含多只ETF的日频成交量信息
        save_path=save_path,  # 计算完成的滚动指标保存路径，字符串格式，指定指标结果存储的文件夹路径
        fund_list=fund,  # 指定要计算的ETF列表，list格式，包含ETF代码（如'510050.SH'），None表示计算所有ETF
        roll_list=None,  # 滚动窗口列表，list格式，指定需要计算的滚动窗口长度（如[20, 60, 120]），None表示使用默认配置
    )
