# -*- encoding: utf-8 -*-
"""
@File: data_prepare.py
@Modify Time: 2025/4/11 20:27       
@Author: Kevin-Chen
@Descriptions: 数据预处理, 生成一系列宽表数据帧
"""
import os
import gc
import numpy as np
import pandas as pd
from GetData.tushare_get_ETF_data import get_etf_info


# 数据预处理, 生成一系列宽表数据帧
def data_prepare(min_data_req=500,
                 read_file="../Data/etf_daily.parquet",
                 save_path="../Data/"
                 ):
    """
    数据预处理函数，用于对ETF基金数据进行清洗、筛选和转换，最终生成宽表数据帧。

    该函数的主要步骤包括：
    1. 读取原始数据并进行基本清洗（去重、处理缺失值、转换日期格式）。
    2. 剔除货币类基金。
    3. 剔除数据量不足的基金。
    4. 剔除一段时间内平均成交额在后25%的基金。
    5. 计算收益率。
    6. 建立宽表数据帧，方便后续分析。

    :param min_data_req: int, 可选参数，默认值为500。表示每只基金所需的最小数据量，低于此值的基金将被剔除。
    :param read_file: str, 原始etf日频数据文件名。
    :param save_path: str, 处理后的一系列宽表数据帧存放文件夹。
    :return: 返回一个包含多个宽表数据帧的元组，每个数据帧对应一个字段（如开盘价、收盘价、收益率等）。
    """
    ''' 数据预处理 '''
    # 读取数据并进行基本清洗
    etf_df = pd.read_parquet(read_file)
    etf_df = etf_df.dropna(subset=["trade_date", "ts_code"])
    etf_df["trade_date"] = pd.to_datetime(etf_df["trade_date"])
    etf_df = etf_df.drop_duplicates(subset=["trade_date", "ts_code"], keep="last")

    ''' 剔除货币类基金 '''
    # 获取货币型基金的 ts_code 并剔除
    etf_info = get_etf_info()
    mm_etf_df = etf_info[etf_info["invest_type"] == "货币型"]['ts_code'].tolist()
    etf_df = etf_df[~etf_df["ts_code"].isin(mm_etf_df)]
    del mm_etf_df
    gc.collect()

    ''' 剔除数据量不足的基金 '''
    # 统计每个 ts_code 的数据量并剔除数据量不足的基金
    ts_code_counts = etf_df["ts_code"].value_counts()
    etf_df = etf_df[etf_df["ts_code"].isin(ts_code_counts[ts_code_counts >= min_data_req].index)].reset_index(drop=True)
    del ts_code_counts
    gc.collect()

    ''' 剔除一段时间内平均成交额在后25%的基金 '''
    # 筛选近N天内的数据并计算平均成交额，剔除后25%的基金
    df_last_year = etf_df[etf_df["trade_date"] >= etf_df["trade_date"].max() - pd.Timedelta(days=min_data_req)]
    mean_amount = df_last_year.groupby("ts_code")["amount"].mean()
    valid_days = df_last_year.groupby("ts_code")["amount"].count()
    ts_codes_with_enough_data = valid_days[valid_days >= (valid_days.max() * 0.9)].index
    mean_amount = mean_amount.loc[ts_codes_with_enough_data]
    selected_ts_codes = mean_amount[mean_amount > mean_amount.quantile(0.25)].index
    etf_df = etf_df[etf_df["ts_code"].isin(selected_ts_codes)]
    del df_last_year, mean_amount, valid_days, ts_codes_with_enough_data, selected_ts_codes
    gc.collect()

    ''' 计算收益率 '''
    # 计算对数收益率和简单收益率
    etf_df["log_return"] = np.log(etf_df["close"] / etf_df["pre_close"])
    del etf_df['pct_chg']
    etf_df['pct_chg'] = etf_df['close'] / etf_df['pre_close'] - 1

    ''' 建立宽表数据帧 '''
    # 设置索引并生成宽表数据帧
    etf_df.set_index(["trade_date", "ts_code"], inplace=True)
    fields = ["open", "high", "low", "close", "change", "pct_chg", "vol", "amount", "log_return"]
    pivot_dfs = {
        field + "_df": etf_df[field].unstack(level="ts_code") for field in fields
    }
    pivot_dfs['etf_info'] = etf_info

    for key, value in pivot_dfs.items():
        file_name = f"wide_{key}.parquet"
        value.to_parquet(os.path.join(save_path, file_name), index=True)
