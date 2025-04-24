# -*- encoding: utf-8 -*-
"""
@File: data_prepare.py
@Modify Time: 2025/4/11 20:27       
@Author: Kevin-Chen
@Descriptions: 数据预处理, 生成一系列宽表数据帧
"""
import os
import gc
import warnings
import numpy as np
import pandas as pd
from functools import reduce

from GetData.tushare_get_ETF_data import get_etf_info

warnings.filterwarnings("ignore")


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
    fields = ["open", "high", "low", "close", "change", "pct", "vol", "amount", "log"]
    pivot_dfs = {
        field + "_df": etf_df[field].unstack(level="ts_code") for field in fields
    }
    pivot_dfs['etf_info'] = etf_info

    for key, value in pivot_dfs.items():
        file_name = f"wide_{key}.parquet"
        value.to_parquet(os.path.join(save_path, file_name), index=True)


# 指数数据预处理, 生成一系列宽表数据帧
def index_data_prepare(index_path="../Data/Index"):
    """
    准备指数数据并生成宽表格式的数据文件。

    读取指定路径下的 Parquet 文件，对每个文件进行数据清洗和处理，
    包括日期格式化、去重、排序、计算收益率等操作，然后将数据
    转换为宽表格式并保存回指定路径。

    参数:
    - index_path: 指数数据文件夹路径，默认为 "../Data/Index"。
    """
    # 初始化宽表列表
    wide_open = list()
    wide_close = list()
    wide_high = list()
    wide_low = list()
    wide_vol = list()
    wide_amount = list()
    wide_log = list()
    wide_pct = list()

    # 遍历文件夹中的文件
    for file in os.listdir(index_path):
        # 筛选符合条件的 Parquet 文件
        if (file.endswith(".parquet")
                and (file.startswith("china") or file.startswith("global"))
                and file != 'china_market_daily.parquet'
        ):
            print(f"处理文件: {file}")
            file_path = os.path.join(index_path, file)

            # 读取数据
            index_data = pd.read_parquet(file_path)
            index_data['ts_code'] = file.split(".")[0] + '_' + index_data['ts_code']
            print(index_data)

            # 数据预处理
            index_data['trade_date'] = pd.to_datetime(index_data['trade_date']).dt.normalize()
            index_data.drop_duplicates(subset=["trade_date", "ts_code"], keep="last", inplace=True)
            index_data = index_data.sort_values(by=['ts_code', 'trade_date'])
            index_data = index_data[index_data['trade_date'] <= pd.Timestamp.today()]

            # 计算收益率
            index_data['pct'] = index_data.groupby('ts_code')['close'].pct_change()
            index_data['log'] = index_data.groupby('ts_code')['close'].transform(
                lambda x: np.log(x / x.shift(1)))

            # 转换为宽表格式
            if 'open' in index_data.columns:
                wide_close.append(index_data.pivot(index='trade_date', columns='ts_code', values='open'))
            if 'close' in index_data.columns:
                wide_open.append(index_data.pivot(index='trade_date', columns='ts_code', values='close'))
            if 'high' in index_data.columns:
                wide_high.append(index_data.pivot(index='trade_date', columns='ts_code', values='high'))
            if 'low' in index_data.columns:
                wide_low.append(index_data.pivot(index='trade_date', columns='ts_code', values='low'))
            if 'vol' in index_data.columns:
                wide_vol.append(index_data.pivot(index='trade_date', columns='ts_code', values='vol'))
            if 'amount' in index_data.columns:
                wide_amount.append(index_data.pivot(index='trade_date', columns='ts_code', values='amount'))
            if 'log' in index_data.columns:
                wide_log.append(index_data.pivot(index='trade_date', columns='ts_code', values='log'))
            if 'pct' in index_data.columns:
                wide_pct.append(index_data.pivot(index='trade_date', columns='ts_code', values='pct'))

    print("数据处理完成，开始保存宽表数据。")

    # 合并宽表数据并保存
    for wide_name, the_list in {
        'wide_index_open': wide_open,
        'wide_index_close': wide_close,
        'wide_index_high': wide_high,
        'wide_index_low': wide_low,
        'wide_index_vol': wide_vol,
        'wide_index_amount': wide_amount,
        'wide_index_log': wide_log,
        'wide_index_pct': wide_pct
    }.items():
        if the_list:
            # 合并宽表数据
            merged_df = reduce(lambda left, right:
                               pd.merge(left, right, on='trade_date', how='outer'), the_list)
            file_name = os.path.join(index_path, f"{wide_name}.parquet")
            merged_df.to_parquet(file_name, index=True)
            print(f"保存文件: {file_name}")
        else:
            print(f"{wide_name} 列表为空。")


if __name__ == '__main__':
    pass
