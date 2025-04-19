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
log_return_file = "./Data/wide_log_return_df.parquet"  # ETF日频对数收益率文件路径
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
fund = ['510050.SH', '159915.SZ', '159912.SZ', '512500.SH', '164701.SZ', '511010.SH', '513100.SH', '513030.SH',
        '513080.SH', '513520.SH', '518880.SH', '161226.SZ', '501018.SH', '159981.SZ', '159985.SZ', '159980.SZ',
        ]

if __name__ == '__main__':
    # 计算区间指标
    # s_t = time.time()
    # compute_metrics_for_period_initialize(log_return_df,  # 对数收益率数据
    #                                       close_price_df,  # 收盘价数据
    #                                       high_price_df,  # 收盘价数据
    #                                       low_price_df,  # 收盘价数据
    #                                       volume_df,  # 收盘价数据
    #                                       save_path,  # 指标的保存文件路径
    #                                       p_list=None,  # 可以计算指定的区间列表, list格式
    #                                       metrics_list=None,  # 可以计算指定的指标列表, list格式
    #                                       fund_list=fund,  # 可以指定要计算的ETF列表, list格式
    #                                       spec_end_date=None,  # 可以指定的区间结束日期
    #                                       multi_process=True  # 是否使用多进程计算, True为使用多进程, False为单进程
    #                                       )
    # print(f"所有区间指标计算完成, 耗时: {(time.time() - s_t) / 60:.2f} 分钟")

    compute_all_rolling_metrics(open_price_df=open_price_df,
                                close_price_df=close_price_df,
                                high_price_df=high_price_df,
                                low_price_df=low_price_df,
                                volume_df=volume_df,
                                save_path=save_path,
                                fund_list=fund,
                                num_workers=None,
                                roll_list=None,
                                multi_process=False,
                                )
