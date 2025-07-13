# -*- encoding: utf-8 -*-
"""
@File: A01_DataPrepare.py
@Modify Time: 2025/7/11 16:00       
@Author: Kevin-Chen
@Descriptions: 
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
from A02_OperatorLibrary import winsorize  # 导入winsorize函数

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 构建Data目录的绝对路径
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'Data')

# 定义原始数据文件路径
basic_info_files = {
    "log": os.path.join(DATA_DIR, "wide_log_return_df.parquet"),
    "high": os.path.join(DATA_DIR, "wide_high_df.parquet"),
    "low": os.path.join(DATA_DIR, "wide_low_df.parquet"),
    "vol": os.path.join(DATA_DIR, "wide_vol_df.parquet"),
    "amount": os.path.join(DATA_DIR, "wide_amount_df.parquet"),
    "close": os.path.join(DATA_DIR, "wide_close_df.parquet"),
    "open": os.path.join(DATA_DIR, "wide_open_df.parquet"),
}

# 定义预处理后数据保存路径
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "Processed_ETF_Data")


def load_and_preprocess_data():
    """
    加载所有原始ETF数据，并在内存中进行预处理。
    **注意**: 此函数不保存文件，也不对log收益率的NaN进行填充。
    
    Returns:
        dict: 包含所有已处理DataFrame的字典。
    """
    print("--- 步骤1: 加载并预处理原始数据 ---")

    # 加载原始数据
    data_frames = {name: pd.read_parquet(path) for name, path in basic_info_files.items()}
    print("原始数据加载完成！")

    # --- 缺失值处理 ---
    data_frames['high'] = data_frames['high'].ffill()
    data_frames['low'] = data_frames['low'].ffill()
    data_frames['open'] = data_frames['open'].ffill()
    data_frames['close'] = data_frames['close'].ffill()
    data_frames['vol'] = data_frames['vol'].fillna(0)
    data_frames['amount'] = data_frames['amount'].fillna(0)
    # **重要**: 对数收益率的NaN保留

    # --- 异常值处理 ---
    def _winsorize_row(row, lower_percentile, upper_percentile):
        """一个更稳健的行缩尾函数，能正确处理NaN。"""
        valid_data = row.dropna()
        if valid_data.empty:
            return row

        lower_bound = valid_data.quantile(lower_percentile)
        upper_bound = valid_data.quantile(upper_percentile)
        return row.clip(lower_bound, upper_bound)

    data_frames['vol'] = data_frames['vol'].apply(lambda row: _winsorize_row(row, 0.01, 0.99), axis=1)
    data_frames['amount'] = data_frames['amount'].apply(lambda row: _winsorize_row(row, 0.01, 0.99), axis=1)
    data_frames['log'] = data_frames['log'].apply(lambda row: _winsorize_row(row, 0.01, 0.99), axis=1)

    print("内存中的数据预处理完成！")
    return data_frames


def _calculate_portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights


def _get_minimum_variance_weights(cov_matrix):
    n_assets = cov_matrix.shape[0]
    initial_weights = np.ones(n_assets) / n_assets
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    result = minimize(_calculate_portfolio_variance, initial_weights, args=(cov_matrix,),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


def _calculate_risk_contribution(weights, cov_matrix):
    portfolio_volatility = np.sqrt(_calculate_portfolio_variance(weights, cov_matrix))
    if portfolio_volatility == 0:
        return np.zeros(len(weights))
    marginal_contribution = cov_matrix @ weights
    risk_contribution = np.multiply(weights, marginal_contribution) / portfolio_volatility
    return risk_contribution


def _get_equal_risk_contribution_weights(cov_matrix):
    n_assets = cov_matrix.shape[0]
    initial_weights = np.ones(n_assets) / n_assets
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))

    def objective(weights, cov_matrix):
        risk_contributions = _calculate_risk_contribution(weights, cov_matrix)
        return np.sum((risk_contributions - risk_contributions.mean()) ** 2)

    result = minimize(objective, initial_weights, args=(cov_matrix,),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


def generate_and_save_benchmarks(log_returns_df,
                                 output_dir=PROCESSED_DATA_DIR,
                                 etf_list=None,
                                 rolling_window=252,
                                 rebalance_freq='M'):
    """
    基于已对齐、已填充的ETF收益率数据，生成并保存多种基准组合的收益率。
    """
    print("\n--- 步骤3: 生成并保存基准组合收益率 ---")

    benchmark_df = log_returns_df
    print(f"用于基准计算的数据维度: {benchmark_df.shape}")

    # 1. 等权重组合 (Equal Weight)
    n_assets = benchmark_df.shape[1]
    ew_weights = pd.DataFrame(np.ones_like(benchmark_df) / n_assets,
                              index=benchmark_df.index, columns=benchmark_df.columns)
    ew_returns = (benchmark_df * ew_weights).sum(axis=1)
    ew_returns.to_frame(name='log_return').to_parquet(os.path.join(output_dir, "benchmark_ew_log_returns.parquet"))
    print("已生成并保存：等权重组合 (EW) 基准")

    # 确定再平衡日期
    rebalance_dates = benchmark_df.resample(rebalance_freq).last().index

    # 2. 滚动最小方差组合 (Rolling Minimum Variance)
    min_var_weights_ts = pd.DataFrame(index=benchmark_df.index, columns=benchmark_df.columns, dtype=float)
    for i, date in enumerate(rebalance_dates):
        if i == 0:
            continue
        rolling_data = benchmark_df.loc[:date].iloc[-rolling_window:]
        if rolling_data.shape[0] < rolling_window * 0.8:
            continue
        cov_matrix = rolling_data.cov()
        weights = _get_minimum_variance_weights(cov_matrix)
        next_period_start = date + pd.Timedelta(days=1)
        next_period_end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else benchmark_df.index[-1]
        min_var_weights_ts.loc[next_period_start:next_period_end] = weights

    min_var_weights_ts = min_var_weights_ts.ffill().fillna(0)
    min_var_returns = (benchmark_df * min_var_weights_ts).sum(axis=1)
    min_var_returns.to_frame(name='log_return').to_parquet(
        os.path.join(output_dir, "benchmark_min_var_log_returns.parquet"))
    print("已生成并保存：滚动最小方差 (MinVar) 基准")

    # 3. 滚动等风险贡献组合 (Rolling Equal Risk Contribution)
    erc_weights_ts = pd.DataFrame(index=benchmark_df.index, columns=benchmark_df.columns, dtype=float)
    for i, date in enumerate(rebalance_dates):
        if i == 0:
            continue
        rolling_data = benchmark_df.loc[:date].iloc[-rolling_window:]
        if rolling_data.shape[0] < rolling_window * 0.8:
            continue
        cov_matrix = rolling_data.cov()
        weights = _get_equal_risk_contribution_weights(cov_matrix)
        next_period_start = date + pd.Timedelta(days=1)
        next_period_end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else benchmark_df.index[-1]
        erc_weights_ts.loc[next_period_start:next_period_end] = weights

    erc_weights_ts = erc_weights_ts.ffill().fillna(0)
    erc_returns = (benchmark_df * erc_weights_ts).sum(axis=1)
    erc_returns.to_frame(name='log_return').to_parquet(os.path.join(output_dir, "benchmark_erc_log_returns.parquet"))
    print("已生成并保存：滚动等风险贡献 (ERC) 基准")
    print("--- 基准组合收益率生成完毕 ---")


if __name__ == "__main__":
    # 1. 加载并进行内存预处理
    all_data_frames = load_and_preprocess_data()
    log_returns_df = all_data_frames['log']

    # 2. 定义基准ETF列表并确定统一的分析起点
    etf_list = [
        '510050.SH',  # 上证50ETF
        '159915.SZ',  # 创业板ETF
        '510300.SH',  # 沪深300ETF
        '512500.SH',  # 中证500ETF华夏
        '511010.SH',  # 国债ETF
        '513100.SH',  # 纳指ETF
        '513030.SH',  # 德国ETF
        '513080.SH',  # 法国CAC40ETF
        '513520.SH',  # 日经ETF
        '518880.SH',  # 黄金ETF
        '161226.SZ',  # 国投白银LOF
        '501018.SH',  # 南方原油LOF
        '159981.SZ',  # 能源化工ETF
        '159985.SZ',  # 豆粕ETF
        '159980.SZ',  # 有色ETF
        '511990.SH'  # 华宝添益货币ETF
    ]

    valid_etfs = [etf for etf in etf_list if etf in log_returns_df.columns]
    if not valid_etfs:
        raise ValueError("错误: 默认ETF列表中的所有ETF都不在收益率数据中，无法继续。")

    inception_dates = log_returns_df[valid_etfs].apply(lambda col: col.first_valid_index())

    print("\n--- 调试信息 ---")
    print(f"在log_returns_df中找到的有效ETF数量: {len(valid_etfs)}")
    # print(f"有效ETF列表: {valid_etfs}") # 列表可能过长，暂时注释
    print("计算出的各ETF成立日期:")
    print(inception_dates.to_string())

    start_date = inception_dates.max()
    print(f"计算出的最晚成立日 (start_date): {start_date}")
    print("--- 调试信息结束 ---\n")

    if pd.isna(start_date):
        raise ValueError("错误: 无法确定有效的开始日期，请检查ETF列表和数据。")

    print(f"\n--- 步骤2: 数据对齐与保存 ---")
    print(f"统一的分析开始日期为: {start_date.date()}")

    # 3. 对齐所有数据并保存
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    for name, df in all_data_frames.items():
        aligned_df = df.loc[start_date:].copy()
        # 对切片后的数据进行填充，以处理临时停牌等情况
        if name == 'log':
            aligned_df.fillna(0, inplace=True)

        save_path = os.path.join(PROCESSED_DATA_DIR, f"processed_{name}_df.parquet")
        aligned_df.to_parquet(save_path)

    print(f"所有数据已对齐并保存到: {PROCESSED_DATA_DIR}")

    # 4. 使用对齐后的数据生成基准
    final_log_returns = all_data_frames['log'].loc[start_date:, valid_etfs].fillna(0)
    generate_and_save_benchmarks(final_log_returns, etf_list=valid_etfs)

    print("\n所有数据准备和基准生成任务已完成。")
