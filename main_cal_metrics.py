# -*- encoding: utf-8 -*-
"""
@File: main_cal_metrics.py
@Modify Time: 2025/4/12 14:26       
@Author: Kevin-Chen
@Descriptions: 主程序: 计算各类指标
"""
import time
import pandas as pd
from MetricsFactory.metrics_factory import compute_metrics_for_period_initialize

'''
本文件用于计算各类指标,
具体计算了什么指标什么区间, 以及指标的含义和简要公式可以查看 MetricsFactory/metrics_cal_config.py 文件; 
指标计算的逻辑代码位于 MetricsFactory/metrics_cal.py 文件; 
如果想要增加自己的指标, 可以在 metrics_cal_config.py 中配置, 然后在 metrics_cal.py 中加入逻辑代码.
指标计算的调用框架位于 MetricsFactory/metrics_factory.py 文件中, 支持单进程和多进程计算. 其中多进程计算使用了共享内存.
'''

# 配置路径
log_return_file = "./Data/wide_log_return_df.parquet"  # ETF日频对数收益率文件路径
close_price_file = "./Data/wide_close_df.parquet"  # ETF日频收盘价文件路径
save_path = "./Data/Metrics"  # 计算完成的指标,存放的文件夹路径

# 如果你有特定的时间段需要计算指标，可以在写区间列表, 例如 ['1m', '2m']. 写None则计算所有预设的区间
specific_period_list = None

log_return_df = pd.read_parquet(log_return_file)  # 读取ETF日频对数收益率数据
close_price_df = pd.read_parquet(close_price_file)  # 读取ETF日频收盘价数据

if __name__ == '__main__':
    # 计算指标
    s_t = time.time()
    compute_metrics_for_period_initialize(log_return_df,  # 对数收益率数据
                                          close_price_df,  # 收盘价数据
                                          save_path,  # 指标的保存文件路径
                                          p_list=specific_period_list,  # 计算指定的区间
                                          )
    print(f"所有指标计算完成, 耗时: {(time.time() - s_t) / 60:.2f} 分钟")
