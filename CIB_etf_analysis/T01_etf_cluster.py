# -*- encoding: utf-8 -*-
"""
@File: T01_etf_cluster.py
@Modify Time: 2025/9/18 08:45       
@Author: Kevin-Chen
@Descriptions: ETF聚类
"""

import pandas as pd

# 读取数据以及预处理
etf_data = pd.read_parquet("data/etf_daily_df.parquet")
etf_data = etf_data[["ts_code", "name", "date", "adj_nav"]]
etf_data['date'] = pd.to_datetime(etf_data['date'])
etf_data = etf_data[etf_data['date'] >= '2015-01-01']  # 只保留2015年以来的数据
etf_data = etf_data.sort_values(by=['ts_code', 'date'])
# 交易日历
trade_day_df = pd.read_parquet("data/trade_day_df.parquet")
trade_day_df = trade_day_df[trade_day_df['is_open'] == 1]
trade_day_df['cal_date'] = pd.to_datetime(trade_day_df['cal_date'])
# 提取交易日的数据
etf_data = etf_data[etf_data['date'].isin(trade_day_df['cal_date'])]
print(f"ETF数量: {etf_data['ts_code'].nunique()}")

# 剔除数据量不足50%的ETF
min_count = etf_data['date'].nunique() * 0.5
valid_etfs = etf_data.groupby('ts_code').filter(lambda x: len(x) >= min_count)['ts_code'].unique()
etf_data = etf_data[etf_data['ts_code'].isin(valid_etfs)]
print(f"剩余ETF数量: {etf_data['ts_code'].nunique()}")

# 剔除近20日无数据的ETF
latest_date = etf_data['date'].max()
recent_threshold = latest_date - pd.Timedelta(days=20)
recent_etfs = etf_data[etf_data['date'] >= recent_threshold]['ts_code'].unique()
etf_data = etf_data[etf_data['ts_code'].isin(recent_etfs)]
print(f"剩余ETF数量: {etf_data['ts_code'].nunique()}")

# 计算收益率&转pivot
etf_data['return'] = etf_data.groupby('ts_code')['adj_nav'].pct_change(fill_method=None)
etf_returns = etf_data.pivot(index='date', columns='ts_code', values='return')
print(etf_returns.tail())
