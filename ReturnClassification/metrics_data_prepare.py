# -*- encoding: utf-8 -*-
"""
@File: metrics_data_prepare.py
@Modify Time: 2025/4/13 10:20       
@Author: Kevin-Chen
@Descriptions: 指标预处理
"""
import os
import duckdb
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option('display.width', 1000)  # 表格不分段显示
pd.set_option('display.max_columns', 1000)  # 显示字段的数量


def get_fund_close_price(selected_fund, data_folder_path):
    full_path = os.path.join(data_folder_path, 'wide_close_df.parquet')
    query = f"""
    SELECT "trade_date", "{selected_fund}"
    FROM '{full_path}'
    """
    df_selected = duckdb.query(query).to_df()
    df_selected['trade_date'] = pd.to_datetime(df_selected['trade_date'])
    df_selected = df_selected.sort_values('trade_date')
    df_selected = df_selected.dropna(subset=["trade_date", selected_fund]).reset_index(drop=True)
    return df_selected


def cal_future_log_return(df_selected, n_days=5):
    # 计算未来n天的对数收益率
    col = df_selected.columns[1]  # 假设第二列是价格数据
    price = df_selected[col]
    # 对数收益率计算: log(P_{t+n} / P_t) = log(P_{t+n}) - log(P_t)
    log_return = np.log(price.shift(-n_days)) - np.log(price)
    df_selected[f'log_return_forward_{n_days}d'] = log_return
    return df_selected


def get_fund_metrics_data(selected_fund, metrics_folder_path):
    df_final = None
    for idx, fil in enumerate(os.listdir(metrics_folder_path)):
        if not fil.endswith('.parquet'):
            continue

        full_path = os.path.join(metrics_folder_path, fil)

        query = f"""
        SELECT *
        FROM '{full_path}'
        WHERE ts_code = '{selected_fund}'
        """

        df = duckdb.query(query).to_df()
        # 删除无用字段，比如 __index_level_0__
        df = df.loc[:, ~df.columns.str.startswith("__")]

        # 第一个表作为基础
        if df_final is None:
            df_final = df
        else:
            df_final = pd.merge(df_final, df, on=['ts_code', 'date'], how='outer')

    df_final = df_final.dropna()
    # 按日期排序
    df_final = df_final.sort_values('date')
    # 将日期列转换为datetime格式
    df_final['date'] = pd.to_datetime(df_final['date'])
    return df_final


if __name__ == '__main__':
    # fund_code = '159919.SZ'  # 沪深300ETF
    # # 指定原始数据文件夹路径
    # folder_path = '../Data'
    # # 指定指标数据文件夹路径
    # metrics_folder = '../Data/Metrics'
    #
    # df_price = get_fund_close_price(fund_code, folder_path)
    # df_price = cal_future_log_return(df_price, n_days=5)
    # df_metrics = get_fund_metrics_data(fund_code, metrics_folder)
    # print(df_price.head())
    # print(df_metrics.head())
    #
    # # 训练集时间区间
    # train_start = df_metrics['date'].min()
    # train_end = pd.to_datetime('2023-12-31')
    # # 测试集时间区间
    # test_start = pd.to_datetime('2024-01-01')
    # test_end = pd.to_datetime('2024-12-31')
    #
    # print(train_start, train_end)
    # print(test_start, test_end)

    ddf = pd.read_parquet("/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/Metrics/5y.parquet")
    ddf = ddf[ddf['ts_code'] == '159919.SZ']
    print(ddf)
