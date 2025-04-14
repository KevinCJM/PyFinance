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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings("ignore")
pd.set_option('display.width', 1000)  # 表格不分段显示
pd.set_option('display.max_columns', 1000)  # 显示字段的数量


# 获取指定基金的收盘价数据
def get_fund_close_price(selected_fund, data_folder_path):
    """
    获取指定基金的收盘价数据。

    本函数从特定的数据文件中读取指定基金的收盘价，并按交易日期排序返回。

    参数:
    - selected_fund: 字符串，指定要查询的基金代码。
    - data_folder_path: 字符串，存放基金数据的文件夹路径。

    返回:
    - df_selected: DataFrame，包含指定基金的交易日期和收盘价。
    """
    # 构造完整文件路径
    full_path = os.path.join(data_folder_path, 'wide_close_df.parquet')

    # 构造SQL查询语句
    query = f"""
    SELECT "trade_date", "{selected_fund}"
    FROM '{full_path}'
    """

    # 执行查询并转换结果为DataFrame
    df_selected = duckdb.query(query).to_df()

    # 将交易日期列转换为datetime类型
    df_selected['trade_date'] = pd.to_datetime(df_selected['trade_date'])

    # 按交易日期升序排序
    df_selected = df_selected.sort_values('trade_date')

    # 删除缺失值行并重置索引
    df_selected = df_selected.dropna(subset=["trade_date", selected_fund]).reset_index(drop=True)

    # 返回处理后的DataFrame
    return df_selected


# 计算并添加未来n天的对数收益率到数据框中
def cal_future_log_return(df_selected, n_days=5):
    """
    计算并添加未来n天的对数收益率到数据框中。

    参数:
    df_selected: pandas.DataFrame，包含至少一列价格数据。
    n_days: int，计算对数收益率的天数，默认为5天。

    返回:
    df_selected: pandas.DataFrame，添加了未来n天对数收益率的列。
    """
    # 选择价格列，假设数据框中的第二列是价格数据
    col = df_selected.columns[1]
    price = df_selected[col]
    # 计算未来n天的对数收益率，使用对数变换来计算收益，以避免复利效应
    # 对数收益率计算: log(P_{t+n} / P_t) = log(P_{t+n}) - log(P_t)
    log_return = np.log(price.shift(-n_days)) - np.log(price)
    # 将计算出的对数收益率添加到原始数据框中，作为新列
    df_selected[f'log_return_forward_{n_days}d'] = log_return
    # 返回添加了对数收益率列的DataFrame
    return df_selected


# 获取基金指标数据
def get_fund_metrics_data(selected_fund, metrics_folder_path):
    """
    获取基金指标数据。

    本函数从指定文件夹中读取所有以.parquet结尾的文件，针对每个文件执行查询以获取选定基金的数据，
    并将结果合并为一个数据框。合并数据框后，进行数据清洗和格式转换，最终返回一个包含选定基金所有
    指标数据的干净数据框。

    参数:
    selected_fund (str): 选定的基金代码。
    metrics_folder_path (str): 包含基金指标数据的文件夹路径。

    返回:
    pandas.DataFrame: 包含选定基金所有指标数据的干净数据框。
    """
    # 初始化最终的数据框为None
    df_final = None
    # 遍历指标数据文件夹中的所有文件
    for idx, fil in enumerate(os.listdir(metrics_folder_path)):
        # 如果文件不是.parquet格式，则跳过
        if not fil.endswith('.parquet'):
            continue

        # 构建文件的完整路径
        full_path = os.path.join(metrics_folder_path, fil)

        # 构建SQL查询语句，以获取选定基金的数据
        query = f"""
        SELECT *
        FROM '{full_path}'
        WHERE ts_code = '{selected_fund}'
        """

        # 执行查询并将结果转换为pandas数据框
        df = duckdb.query(query).to_df()
        # 删除无用字段，比如 __index_level_0__
        df = df.loc[:, ~df.columns.str.startswith("__")]

        # 第一个表作为基础
        if df_final is None:
            df_final = df
        else:
            # 将当前数据框与最终数据框按基金代码和日期合并
            df_final = pd.merge(df_final, df, on=['ts_code', 'date'], how='outer')

    # 删除合并后存在空值的行
    df_final = df_final.dropna()
    # 按日期排序
    df_final = df_final.sort_values('date')
    # 将日期列转换为datetime格式
    df_final['date'] = pd.to_datetime(df_final['date'])
    # 返回清洗后的数据框
    return df_final


# 预处理指标数据
def preprocess_data(metrics_data):
    """
    预处理指标数据。

    对给定的指标数据进行清理和归一化/标准化处理，以确保数据适合进一步分析。

    参数:
    metrics_data: DataFrame, 包含指标数据的DataFrame，必须包含 'ts_code' 和 'date' 列。

    返回:
    DataFrame, 预处理后的DataFrame，包含归一化/标准化后的指标数据。
    """
    # 复制数据框以避免修改原始数据
    df_metrics = metrics_data.copy()

    # 移除 'ts_code' 和 'date' 列，这些列将被稍后重新插入到处理后的DataFrame中
    ts_code = df_metrics.pop('ts_code')
    date = df_metrics.pop('date')

    # 用每列的中位数填充 NaN、inf、-inf
    df_metrics = df_metrics.replace([np.inf, -np.inf], np.nan)  # 将 inf/-inf 转为 NaN
    medians = df_metrics.median(numeric_only=True)  # 计算每列中位数
    df_metrics = df_metrics.fillna(medians)  # 用中位数填充 NaN

    # 计算列的最大最小值
    col_min = df_metrics.min(axis=0, skipna=True)
    col_max = df_metrics.max(axis=0, skipna=True)

    # 判断每一列属于哪类变量
    pos_mask = (col_min >= 0)  # 全正值 → 最大最小归一化
    neg_mask = (col_max <= 0)  # 全负值 → 取绝对值后最大最小归一化
    mixed_mask = ~(pos_mask | neg_mask)  # 有正有负 → 标准化

    # 三类列名
    pos_cols = df_metrics.columns[pos_mask]
    neg_cols = df_metrics.columns[neg_mask]
    mixed_cols = df_metrics.columns[mixed_mask]

    # Z-score 标准化（有正有负）
    df_mixed = pd.DataFrame(
        StandardScaler().fit_transform(df_metrics[mixed_cols]),
        columns=mixed_cols,
        index=df_metrics.index) if len(mixed_cols) > 0 else pd.DataFrame(index=df_metrics.index)

    # Min-Max 归一化（全正）
    df_pos = pd.DataFrame(
        MinMaxScaler().fit_transform(df_metrics[pos_cols]),
        columns=pos_cols,
        index=df_metrics.index) if len(pos_cols) > 0 else pd.DataFrame(index=df_metrics.index)

    # Min-Max 归一化（全负 → 取绝对值）
    abs_data = df_metrics[neg_cols].abs() if len(neg_cols) > 0 else pd.DataFrame(index=df_metrics.index)
    df_neg = pd.DataFrame(
        MinMaxScaler().fit_transform(abs_data),
        columns=neg_cols,
        index=df_metrics.index) if len(neg_cols) > 0 else pd.DataFrame(index=df_metrics.index)

    # 合并结果
    df_processed = pd.concat([df_mixed, df_pos, df_neg], axis=1)[df_metrics.columns]
    df_processed.insert(0, 'date', date)
    df_processed.insert(0, 'ts_code', ts_code)
    return df_processed


if __name__ == '__main__':
    # fund_code = '159919.SZ'  # 沪深300ETF
    # # 指定原始数据文件夹路径
    # folder_path = '../Data'
    # # 指定指标数据文件夹路径
    # metrics_folder = '../Data/Metrics'
    # # 预测未来n天的收益率
    # future_days = 20
    #
    # ''' 价格数据预处理 '''
    # # 获取收盘价数据
    # df_price = get_fund_close_price(fund_code, folder_path)
    # # 滚动计算未来5天的对数收益率
    # df_price = cal_future_log_return(df_price, n_days=future_days)
    # # 生成目标标签
    # df_price[f"label_up_{future_days}d"] = (df_price[f"log_return_forward_{future_days}d"] > 0).astype(int)
    #
    # ''' 指标数据预处理 '''
    # # 获取指标数据
    # metrics_df = get_fund_metrics_data(fund_code, metrics_folder)
    # # 预处理指标数据
    # metrics_df = preprocess_data(metrics_df)

    pass
