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
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from ReturnClassification.build_collaborative_features import get_cross_metrics_transform, get_cross_metrics

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
def get_fund_metrics_data(selected_fund,
                          index_folder_path,
                          metrics_folder_path,
                          data_folder_path,
                          period_metrics=True,
                          rolling_metrics=True,
                          basic_data_as_metric=True,
                          index_close_as_metric=True,
                          ):
    """
    获取基金指标数据。

    本函数从指定文件夹中读取所有以.parquet结尾的文件，针对每个文件执行查询以获取选定基金的数据，
    并将结果合并为一个数据框。合并数据框后，进行数据清洗和格式转换，最终返回一个包含选定基金所有
    指标数据的干净数据框。

    参数:
    selected_fund (str): 选定的基金代码。
    index_folder_path (str): 指数数据的文件夹路径。
    metrics_folder_path (str): 包含基金指标数据的文件夹路径。
    data_folder_path (str): 包含基金基本信息的文件夹路径。
    period_metrics (bool): 是否使用区间指标作为训练参数。
    rolling_metrics (bool): 是否使用滚动指标作为训练参数。
    basic_data_as_metric (bool): 是否将基本数据视为指标数据。
    index_close_as_metric (bool): 是否将指数收盘价数据视为指标数据。

    返回:
    pandas.DataFrame: 包含选定基金所有指标数据的干净数据框。
    """
    print(f"[INFO] 获取指标数据 ...")
    # 初始化最终的数据框为None
    df_final = None

    # 使用 区间指标 作为训练参数
    if period_metrics:
        # 遍历指标数据文件夹中的所有文件
        for idx, fil in enumerate(os.listdir(metrics_folder_path)):
            # 如果文件不是.parquet格式，则跳过
            if not fil.endswith('.parquet'):
                continue
            # 如果是滚动指标则跳过
            if fil == 'rolling_metrics.parquet':
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

    # 使用 滚动指标 作为训练参数
    if rolling_metrics:
        # 构建文件的完整路径
        full_path = os.path.join(metrics_folder_path, 'rolling_metrics.parquet')

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

    # 如果将基金的基本数据也作为指标数据
    if basic_data_as_metric:
        print(f"[INFO] 基本数据也作为指标数据")
        # 获取 对数收益/开盘价/收盘价/最高价/最低价/成交量/成交额 的数据
        basic_data_df = get_fund_basic_data(data_folder_path, selected_fund)
        # 合并数据
        if df_final is None:
            df_final = basic_data_df
            df_final['ts_code'] = selected_fund
        else:
            # 将当前数据框与最终数据框按基金代码和日期合并
            df_final = pd.merge(df_final, basic_data_df, on=['date'], how='outer')

    # 如果将指数的收盘价数据也作为指标数据
    if index_close_as_metric:
        print(f"[INFO] 指数数据也作为指标数据")

        # 读取数据
        full_path = os.path.join(index_folder_path, 'wide_index_close.parquet')
        index_df = pd.read_parquet(full_path)
        # 将 inf 转为 nan
        index_df = index_df.replace([np.inf, -np.inf], np.nan)

        # 填充数据
        index_df = index_df.resample('D').asfreq()
        index_df = index_df.fillna(method='ffill')

        # 最小日期限制在 2015-01-01 之前
        min_date = pd.to_datetime('2015-01-01')
        # 从 index_df 找到每个指数的第一个非空数据的日期
        first_valid_dates = index_df.apply(lambda col: col.first_valid_index())
        # 找出需要剔除的列（第一个有效日期 >= min_date）
        cols_to_drop = first_valid_dates[first_valid_dates >= min_date].index
        # 从 index_df 中剔除这些列
        index_df = index_df.drop(columns=cols_to_drop)

        # 重建索引
        index_df = index_df.reset_index()
        index_df.columns.name = None
        index_df.rename(columns={'trade_date': 'date'}, inplace=True)
        # 删除全是nan的列
        index_df = index_df.dropna(axis=1, how='all')
        # 将当前数据框与最终数据框按基金代码和日期合并
        df_final = pd.merge(df_final, index_df, on=['date'], how='left')

    # 将 inf 转为 nan
    df_final = df_final.replace([np.inf, -np.inf], np.nan)
    # 删除合并后存在空值的行
    df_final = df_final.dropna()
    # 按日期排序
    df_final = df_final.sort_values('date')
    # 将日期列转换为datetime格式
    df_final['date'] = pd.to_datetime(df_final['date'])

    n, m = df_final.shape
    print(f"[INFO] 所有特征数据获取完成, 共 {n} 条记录, {m} 个指标")
    # 返回清洗后的数据框
    return df_final


# 预处理指标数据
def preprocess_data(metrics_data,
                    nan_method='median',
                    standardize_method='both'  # 'both'、'minmax'、'zscore'、'none',
                    ):
    """
    预处理指标数据。

    对给定的指标数据进行清理和归一化/标准化处理，以确保数据适合进一步分析。

    参数:
    metrics_data: DataFrame, 包含指标数据的DataFrame，必须包含 'ts_code' 和 'date' 列。
    nan_method: str, 处理NaN值的方法，默认为 'median'，可选['median', 'mean', 'drop']。
    standardize_method: str, 数据标准化/归一化的方法，默认为 'both'，可选['both', 'minmax', 'zscore', 'none']。

    返回:
    DataFrame, 预处理后的DataFrame，包含归一化/标准化后的指标数据。
    """
    # 复制数据框以避免修改原始数据
    df_metrics = metrics_data.copy()

    # 移除 'ts_code' 和 'date' 列，这些列将被稍后重新插入到处理后的DataFrame中
    ts_code = df_metrics.pop('ts_code')
    date = df_metrics.pop('date')

    # 将无穷大和无穷小的值替换为NaN
    df_metrics = df_metrics.replace([np.inf, -np.inf], np.nan)

    # 删除全是nan的列
    df_metrics = df_metrics.dropna(axis=1, how='all')

    # 根据nan_method参数处理NaN值
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

    # 再次删除全是nan的列，以防上一步产生新的全nan列
    df_metrics = df_metrics.dropna(axis=1, how='all')

    # 计算列的最大最小值
    col_min = df_metrics.min(axis=0, skipna=True)
    col_max = df_metrics.max(axis=0, skipna=True)

    # 根据standardize_method参数进行数据标准化/归一化
    if standardize_method == 'both':
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
    elif standardize_method == 'minmax':
        # Min-Max 归一化
        df_processed = pd.DataFrame(
            MinMaxScaler().fit_transform(df_metrics),
            columns=df_metrics.columns,
            index=df_metrics.index
        )
    elif standardize_method == 'zscore':
        # Z-score 标准化
        df_processed = pd.DataFrame(
            StandardScaler().fit_transform(df_metrics),
            columns=df_metrics.columns,
            index=df_metrics.index
        )
    else:
        # 不进行标准化/归一化
        df_processed = df_metrics.copy()

    # 将之前移除的 'ts_code' 和 'date' 列重新插入到DataFrame中
    df_processed.insert(0, 'date', date)
    df_processed.insert(0, 'ts_code', ts_code)

    # 返回预处理后的DataFrame
    return df_processed


# 划分训练集和测试集数据
def split_train_test_data(fund_code, future_days, df_price, metrics_df,
                          train_start=None,
                          train_end='2023-12-31',
                          test_start='2024-01-01',
                          test_end='2025-03-31'
                          ):
    """
    根据给定的基金代码、未来天数、价格数据和指标数据，划分训练集和测试集数据。

    :param fund_code: 基金代码，用于识别特定的基金。
    :param future_days: 未来的天数，用于预测。
    :param df_price: 价格数据，包含日期和基金价格信息。
    :param metrics_df: 指标数据，包含用于训练的特征。
    :param train_start: str, 训练集的区间开始日期
    :param train_end: str, 训练集的区间结束日期
    :param test_start: str, 测试集的区间开始日期
    :param test_end: str, 测试集的区间解释日期
    :return: 返回训练集特征、训练集标签、测试集特征和测试集标签。
    """
    ''' 合并特征 + 标签 '''
    # 重命名价格数据列，以便于后续合并和识别
    df_price_renamed = df_price.rename(columns={
        "trade_date": "date",
        fund_code: "price"
    })

    # 合并指标数据和价格数据，得到包含特征和标签的数据集
    df_all = pd.merge(metrics_df, df_price_renamed[["date", f"label_up_{future_days}d"]],
                      how="inner", on="date")

    ''' 划分数据集 '''
    # 定义训练集和测试集的时间区间
    train_start = pd.to_datetime(train_start) if train_start else df_all['date'].min()
    train_end = pd.to_datetime(train_end)
    test_start = pd.to_datetime(test_start)
    test_end = pd.to_datetime(test_end)
    print(f"[INFO] 训练集时间区间: {train_start} ~ {train_end}; 测试集时间区间: {test_start} ~ {test_end}")

    # 根据时间区间划分训练集和测试集
    train_df = df_all[(df_all['date'] >= train_start) & (df_all['date'] <= train_end)]
    test_df = df_all[(df_all['date'] >= test_start) & (df_all['date'] <= test_end)]

    ''' 准备特征和标签 '''
    # 选取特征列和目标列
    feature_cols = list(set([col for col in metrics_df.columns if col not in ['ts_code', 'date']]))
    target_col = f"label_up_{future_days}d"

    # 分离训练集和测试集的特征和标签
    x_train = train_df[feature_cols]
    y_train = train_df[target_col]

    x_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print(f"[INFO] "
          f"训练集数据量: {len(x_train)}条参数&{len(y_train)}条标签; 参数共{len(x_train.columns)}列; "
          f"测试集数据量: {len(x_test)}条参数&{len(y_test)}条标签; 参数共{len(x_test.columns)}列.")
    return x_train, y_train, x_test, y_test


def main_data_prepare(the_fund_code='159919.SZ',
                      n_days=20,
                      folder_path='../Data',
                      index_folder_path='../Data/Index',
                      metrics_folder='../Data/Metrics',
                      train_start=None,
                      train_end='2023-12-31',
                      test_start='2024-01-01',
                      test_end='2025-03-31',
                      nan_method='drop',
                      standardize_method='both',
                      period_metrics=True,
                      rolling_metrics=True,
                      basic_data_as_metric=False,
                      index_close_as_metric=True,
                      return_threshold=0.0,
                      dim_reduction=False, dim_reduction_limit=0.9, n_components=None,
                      cross_metrics=False, selected_metrics=None,
                      model_folder_path='../Data/Models',
                      joblib_file_name='autofeat_model_2_10.joblib',
                      ):
    """
    主要数据准备函数，用于准备基金数据以进行后续的机器学习模型训练和测试。

    :param the_fund_code: 基金代码，默认为'159919.SZ'
    :param n_days: 用于计算未来收益的天数，默认为20天
    :param folder_path: 基金价格数据的文件夹路径，默认为'../Data'
    :param index_folder_path: 指数数据的文件夹路径，默认为'../Data/Index'
    :param metrics_folder: 基金指标数据的文件夹路径，默认为'../Data/Metrics'
    :param train_start: 训练集开始日期，如果为None，则从数据的开始日期开始
    :param train_end: 训练集结束日期，默认为'2023-12-31'
    :param test_start: 测试集开始日期，默认为'2024-01-01'
    :param test_end: 测试集结束日期，默认为'2025-03-31'
    :param nan_method: 处理缺失值的方法，默认为 'drop'，可选值为 'median' 或 'mean'
    :param standardize_method: 数据标准化/归一化的方法，默认为 'both'，可选['both', 'minmax', 'zscore', 'none']。
    :param period_metrics: bool, 是否使用区间指标作为训练参数，默认为 True
    :param rolling_metrics: bool, 是否使用滚动指标作为训练参数，默认为 True
    :param basic_data_as_metric: bool, 是否将基本数据(例如:开盘价/收盘价/交易量等等)作为指标数据，默认为False
    :param index_close_as_metric: bool, 是否使用指数收盘价作为指标数据，默认为True
    :param return_threshold: float, 标签生成方法，默认为0, 表示使用未来收益率大于0的样本标记为1，否则为0;
                        如果写0.001, 则表示使用未来收益率大于+0.1%的样本标记为2，在-0.1%~0.1%之间的样本标记为1，否则为0;
    :param dim_reduction: bool, 是否做PCA数据降维，默认为 False
    :param dim_reduction_limit: float, PCA数据降维保留的解释方差比率，默认为0.9
    :param n_components: int, PCA数据降维到多少数量
    :param cross_metrics: bool, 是否构建交叉参数
    :param selected_metrics: list, 选择用于构建交叉参数的特征
    :param model_folder_path: str, 交叉参数模型保存的文件夹路径
    :param joblib_file_name: str, 交叉参数模型的文件名字

    :return: 返回训练集特征、训练集标签、测试集特征、测试集标签和原始指标数据
    """
    # 如果要做PCA, 数据预处理阶段不做标准化操作, 而是留到PCA的时候统一来做
    if dim_reduction:
        standardize_method = 'none'

    ''' 价格数据预处理 '''
    # 获取 收盘价 数据
    close_price = get_fund_close_price(the_fund_code, folder_path)
    # 滚动计算未来N天的对数收益率
    close_price = cal_future_log_return(close_price, n_days=n_days)
    # 生成目标标签: 未来收益大于0的样本标记为1，否则为0
    if return_threshold == 0:
        close_price[f"label_up_{n_days}d"] = (close_price[f"log_return_forward_{n_days}d"] > 0).astype(int)
    else:
        def classify_three_way(ret, threshold):
            if ret > threshold:
                return 2  # 上涨
            elif ret < -threshold:
                return 0  # 下跌
            else:
                return 1  # 横盘

        close_price[f"label_up_{n_days}d"] = close_price[f"log_return_forward_{n_days}d"].apply(
            lambda x: classify_three_way(x, return_threshold))

    ''' 指标数据预处理 '''
    # 获取指标数据
    metrics_data = get_fund_metrics_data(
        the_fund_code,  # 基金代码，用于指定需要处理的基金（如'510050.SH'）。
        index_folder_path,
        metrics_folder,  # 指标数据文件夹路径，包含用于训练模型的特征数据。
        folder_path,  # 数据文件夹路径，通常包含基金的价格数据和其他相关信息。
        period_metrics,  # 是否使用区间指标作为训练参数，默认为 True
        rolling_metrics,  # 是否使用滚动指标作为训练参数，默认为 True
        basic_data_as_metric,  # 是否将基本数据（如开盘价、收盘价、交易量等）作为特征数据，默认为False。
        index_close_as_metric,  # 是否将指数收盘价作为指标数据，默认为True。
    )
    # 预处理指标数据
    metrics_data = preprocess_data(
        metrics_data=metrics_data,  # 获取到的原始指标数据，需要进行预处理。
        nan_method=nan_method,  # 处理缺失值的方法，默认为 'drop'（删除缺失值），可选 'median' 或 'mean'。
        standardize_method=standardize_method  # 数据标准化的方法,可选: 'minmax', 'zscore', 'both', 'none'
    )

    ''' 测试集训练集划分 '''
    # 划分训练集和测试集数据
    x_train, y_train, x_test, y_test = split_train_test_data(
        the_fund_code, n_days, close_price, metrics_data,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end
    )

    if cross_metrics:
        if selected_metrics is None:
            selected_metrics = [
                'low', 'high', 'amount', 'open', 'close', 'vol',
                'TotalReturn:5d', 'TotalReturn:10d', 'TotalReturn:15d', 'TotalReturn:25d',
                'Volatility:5d', 'Volatility:10d', 'Volatility:15d', 'Volatility:25d'
            ]

        model_file_path = os.path.join(model_folder_path, joblib_file_name)

        # 判断文件是否存在
        if not os.path.exists(model_file_path):
            # 训练模型
            get_cross_metrics(x_train, y_train, selected_metrics, steps=2,
                              model_folder_path=model_folder_path, joblib_file_name=joblib_file_name)
        else:
            # 使用模型
            x_train = get_cross_metrics_transform(x_train, selected_metrics,
                                                  model_folder_path=model_folder_path,
                                                  joblib_file_name=joblib_file_name)
    if dim_reduction:
        print("[INFO] 数据降维中...")
        # 数据归一化
        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        if n_components is None:
            # dim_reduction_limit = 0.99
            pca_full = PCA().fit(x_train_scaled)
            explained_var_ratio_cum_sum = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(explained_var_ratio_cum_sum >= dim_reduction_limit) + 1
            print(f"[INFO] PCA降维到 {n_components} 维, 解释方差比率达到{dim_reduction_limit * 100}%")

        # 使用 PCA 降维
        pca = PCA(n_components=n_components)
        x_train = pca.fit_transform(x_train_scaled)
        x_test = pca.transform(x_test_scaled)
        print(f"[INFO] PCA降维完成, "
              f"训练集数据量: {len(x_train)}条参数&{len(y_train)}条标签; 参数共{x_train.shape[1]}列; "
              f"测试集数据量: {len(x_test)}条参数&{len(y_test)}条标签; 参数共{x_test.shape[1]}列.")

    # 返回划分好的数据集和原始指标数据
    return x_train, y_train, x_test, y_test, metrics_data


if __name__ == '__main__':
    df = pd.read_parquet('/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/Index/wide_index_close.parquet')
    print(df)
