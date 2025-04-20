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


# 获取指定基金的基本数据 (开盘价、收盘价、最高价、最低价、成交量)
def get_fund_basic_data(data_folder_path, selected_fund):
    """
    获取指定基金的基础数据。

    从给定的数据文件夹中读取所有以'wide'开头且以'.parquet'结尾的文件，
    并根据selected_fund参数筛选出指定基金的数据，最后将这些数据按交易日期合并。

    :param data_folder_path: 包含基金数据文件的文件夹路径。
    :param selected_fund: 需要提取的指定基金的代码或名称。
    :return: 包含指定基金所有合并数据的DataFrame。
    """
    # 获取文件列表
    file_list = os.listdir(data_folder_path)
    file_list = [f for f in file_list if f.startswith('wide') and f.endswith('.parquet')]

    all_info = None
    # 读取所有文件
    for file in file_list:
        file_path = os.path.join(data_folder_path, file)

        # 跳过 wide_etf_info.parquet 文件
        if file_path.endswith('wide_etf_info.parquet'):
            continue

        # 获取文件名中的信息名称
        _, name, _ = file.split('_')
        # 构造SQL查询语句
        query = f"""
        SELECT "trade_date", "{selected_fund}"
        FROM '{file_path}'
        """
        df = duckdb.query(query).to_df()
        df.columns = ['date', name]
        # 合并数据
        if all_info is None:
            all_info = df
        else:
            all_info = pd.merge(all_info, df, on='date', how='inner')

    # 数据整理
    all_info['date'] = pd.to_datetime(all_info['date'])
    all_info = all_info.sort_values('date')
    all_info = all_info.reset_index(drop=True)
    return all_info


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
def get_fund_metrics_data(selected_fund, metrics_folder_path, data_folder_path, basic_data_as_metric):
    """
    获取基金指标数据。

    本函数从指定文件夹中读取所有以.parquet结尾的文件，针对每个文件执行查询以获取选定基金的数据，
    并将结果合并为一个数据框。合并数据框后，进行数据清洗和格式转换，最终返回一个包含选定基金所有
    指标数据的干净数据框。

    参数:
    selected_fund (str): 选定的基金代码。
    metrics_folder_path (str): 包含基金指标数据的文件夹路径。
    data_folder_path (str): 包含基金基本信息的文件夹路径。
    basic_data_as_metric (bool): 是否将基本数据视为指标数据。

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

    if basic_data_as_metric:
        print(f"[INFO] {selected_fund} 基本数据也作为指标数据")
        # 获取 对数收益/开盘价/收盘价/最高价/最低价/成交量/成交额 的数据
        basic_data_df = get_fund_basic_data(data_folder_path, selected_fund)
        # 合并数据
        df_final = pd.merge(df_final, basic_data_df, on=['date'], how='outer')

    # 删除合并后存在空值的行
    df_final = df_final.dropna()
    # 按日期排序
    df_final = df_final.sort_values('date')
    # 将日期列转换为datetime格式
    df_final['date'] = pd.to_datetime(df_final['date'])

    n, m = df_final.shape
    print(f"[INFO] {selected_fund} 指标数据获取完成, 共 {n} 条记录, {m} 个指标")
    # 返回清洗后的数据框
    return df_final


# 预处理指标数据
def preprocess_data(metrics_data,
                    nan_method='median'
                    ):
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

    df_metrics = df_metrics.replace([np.inf, -np.inf], np.nan)  # 将 inf/-inf 转为 NaN
    if nan_method == 'median':
        # 用每列的中位数填充 NaN
        medians = df_metrics.median(numeric_only=True)  # 计算每列中位数
        df_metrics = df_metrics.fillna(medians)  # 用中位数填充 NaN
    elif nan_method == 'mean':
        # 用每列的均值填充 NaN
        means = df_metrics.mean(numeric_only=True)
        df_metrics = df_metrics.fillna(means)
    elif nan_method == 'drop':
        # 删除包含NaN的行
        df_metrics = df_metrics.dropna()
    df_metrics = df_metrics.dropna()

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
    # df = get_fund_metrics_data('510050.SH', '../Data/Metrics')
    # print(df)
    selected_fund = '510050.SH'

    # 获取Data文件夹下所有wide开头的文件
    data_folder_path = '../Data'
