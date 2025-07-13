# -*- encoding: utf-8 -*-
"""
@File: A01_DataPrepare.py
@Modify Time: 2025/7/11 19:00       
@Author: Kevin-Chen
@Descriptions: 数据准备模块，实现最严谨的生命周期和基准构建逻辑。
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize


def _winsorize_row(row, lower_percentile, upper_percentile):
    """一个更稳健的行缩尾函数，能正确处理NaN。"""
    valid_data = row.dropna()
    if valid_data.empty:
        return row
    lower = valid_data.quantile(lower_percentile)
    upper = valid_data.quantile(upper_percentile)
    # Clip only valid data to avoid introducing NaNs
    row[valid_data.index] = valid_data.clip(lower, upper)
    return row


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
    return np.multiply(weights, marginal_contribution) / portfolio_volatility


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


def generate_and_save_benchmarks(log_returns_df, output_dir, rolling_window, rebalance_freq='ME'):
    """
    基于已对齐、已填充的ETF收益率数据，生成并保存多种基准组合的收益率。
    """
    print("\n--- 步骤5: 生成并保存基准组合收益率 ---")
    benchmark_df = log_returns_df
    print(f"用于基准计算的数据维度: {benchmark_df.shape}")

    n_assets = benchmark_df.shape[1]
    ew_weights = pd.DataFrame(np.ones_like(benchmark_df) / n_assets, index=benchmark_df.index,
                              columns=benchmark_df.columns)
    ew_returns = (benchmark_df * ew_weights).sum(axis=1)
    ew_returns.to_frame(name='log_return').to_parquet(os.path.join(output_dir, "benchmark_ew_log_returns.parquet"))
    print("已生成并保存：等权重组合 (EW) 基准")

    rebalance_dates = benchmark_df.resample(rebalance_freq).last().index
    min_var_weights_ts = pd.DataFrame(index=benchmark_df.index, columns=benchmark_df.columns, dtype=float)
    for i, date in enumerate(rebalance_dates):
        if i == 0: continue
        rolling_data = benchmark_df.loc[:date].iloc[-rolling_window:]
        if rolling_data.shape[0] < rolling_window * 0.8: continue
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

    erc_weights_ts = pd.DataFrame(index=benchmark_df.index, columns=benchmark_df.columns, dtype=float)
    for i, date in enumerate(rebalance_dates):
        if i == 0: continue
        rolling_data = benchmark_df.loc[:date].iloc[-rolling_window:]
        if rolling_data.shape[0] < rolling_window * 0.8: continue
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


def prepare_main(data_dir, processed_dir, etf_list_for_benchmark=None, rolling_window=252, rebalance_freq='ME'):
    """
    数据准备和基准生成的总入口函数。
    """
    # 1. 加载所有原始数据
    print("--- 步骤1: 加载原始数据 ---")
    basic_info_files = {name: os.path.join(data_dir, f"{key}.parquet") for name, key in
                        (("log", "wide_log_return_df"), ("high", "wide_high_df"), ("low", "wide_low_df"),
                         ("vol", "wide_vol_df"), ("amount", "wide_amount_df"), ("close", "wide_close_df"),
                         ("open", "wide_open_df"))}
    all_data_frames = {name: pd.read_parquet(path) for name, path in basic_info_files.items()}
    log_returns_df = all_data_frames['log']
    print("原始数据加载完成！")

    # 2. 定义基准ETF列表并确定统一的分析起点
    print(f"\n--- 步骤2: 确定分析周期 ---")
    if etf_list_for_benchmark is None:
        etf_list_for_benchmark = [
            # '510050.SH',  # 上证50ETF, 20050223
            '159915.SZ',  # 创业板ETF, 20111124
            '159912.SZ',  # 沪深300ETF, 20111124
            # '512500.SH',  # 中证500ETF华夏, 20150529
            '511010.SH',  # 国债ETF, 20130325
            '513100.SH',  # 纳指ETF, 20130515
            '513030.SH',  # 德国ETF, 20140905
            # '513080.SH',  # 法国CAC40ETF, 20200612
            # '513520.SH',  # 日经ETF, 20190625
            '518880.SH',  # 黄金ETF, 20130729
            # '161226.SZ',  # 国投白银LOF, 20150817
            # '501018.SH',  # 南方原油LOF, 20160628
            # '159981.SZ',  # 能源化工ETF, 20200117
            # '159985.SZ',  # 豆粕ETF, 20191205
            # '159980.SZ',  # 有色ETF, 20191224
            '511990.SH',  # 华宝添益货币ETF, 20130128
        ]
    valid_benchmark_etfs = [etf for etf in etf_list_for_benchmark if etf in log_returns_df.columns]
    if not valid_benchmark_etfs:
        raise ValueError("错误: 指定的基准ETF列表在收益率数据中均无效。")
    benchmark_inception_dates = log_returns_df[valid_benchmark_etfs].apply(lambda col: col.first_valid_index())
    benchmark_start_date = benchmark_inception_dates.max()
    if pd.isna(benchmark_start_date):
        raise ValueError("错误: 无法为基准ETF确定有效的开始日期。")
    print(f"基准统一分析开始日期为: {benchmark_start_date.date()}")

    # 3. 识别幸存ETF并进行精确的预处理
    print("\n--- 步骤3: 数据预处理 ---")
    last_day = log_returns_df.index[-1]
    surviving_etfs = []
    for etf_code in log_returns_df.columns:
        if log_returns_df[etf_code].last_valid_index() >= last_day - pd.Timedelta(days=10):
            surviving_etfs.append(etf_code)
    print(f"识别出 {len(surviving_etfs)} 个存续ETF。")

    # 对所有存续ETF的数据进行切片和处理
    processed_data = {}
    for name, df in all_data_frames.items():
        # 4.1: 按日期和幸存者列表切片
        proc_df = df.loc[benchmark_start_date:, surviving_etfs].copy()

        # 4.2: 填充NaN
        if name in ['high', 'low', 'open', 'close', 'vol', 'amount']:
            proc_df.ffill(inplace=True)
        elif name == 'log':
            for etf_code in proc_df.columns:
                inception = proc_df[etf_code].first_valid_index()
                if pd.notna(inception):
                    proc_df.loc[inception:, etf_code] = proc_df.loc[inception:, etf_code].fillna(0)

        # 4.3: 缩尾
        if name in ['vol', 'amount', 'log']:
            proc_df = proc_df.apply(lambda row: _winsorize_row(row, 0.01, 0.99), axis=1)

        processed_data[name] = proc_df
    print("数据预处理完成 (切片、填充、缩尾)。")

    # 4. 保存所有处理好的数据
    os.makedirs(processed_dir, exist_ok=True)
    for name, df in processed_data.items():
        save_path = os.path.join(processed_dir, f"processed_{name}_df.parquet")
        df.to_parquet(save_path)
    print(f"所有处理后的数据已保存到: {processed_dir}")

    # 5. 使用最终数据生成基准
    final_benchmark_log_returns = processed_data['log'][valid_benchmark_etfs]
    generate_and_save_benchmarks(final_benchmark_log_returns, processed_dir, rolling_window, rebalance_freq)

    print("\n所有数据准备和基准生成任务已完成。")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'Data')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "Processed_ETF_Data")

    prepare_main(
        data_dir=DATA_DIR,
        processed_dir=PROCESSED_DATA_DIR
    )

    df_l = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "processed_log_df.parquet"))
    df_b = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "benchmark_ew_log_returns.parquet"))
    print(df_l.head())
    print(df_b.head())
