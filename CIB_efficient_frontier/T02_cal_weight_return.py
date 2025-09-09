# -*- encoding: utf-8 -*-
"""
@File: T02_cal_weight_return.py
@Modify Time: 2025/9/9 19:28       
@Author: Kevin-Chen
@Descriptions: 计算各个权重组合的收益率与波动率
"""
import time
import datetime
import numpy as np
import pandas as pd
from T01_simplex_grid_generator import generate_simplex_grid_numba


# 加权求和计算指数组合每日收益数据
def generate_alloc_perf(portfolio_alloc, hist_value_r):
    """
    计算投资组合的年度化收益率和年度化波动率

    参数:
        portfolio_alloc (Dict): 投资组合权重字典，键为资产代码，值为权重
        hist_value_r (DataFrame): 历史收益率数据，索引为日期，列为资产代码

    返回:
        Dict: 包含年度化收益率(ret_annual)和年度化波动率(vol_annual)的字典
    """
    # 获取历史价值数据并筛选投资组合中的资产
    port_daily = hist_value_r.copy()
    port_daily = port_daily[portfolio_alloc.keys()]

    # 计算投资组合每日收益率：各资产收益率与其权重的点积
    port_daily.loc[:, 'r'] = port_daily[portfolio_alloc.keys()].values @ np.array(list(portfolio_alloc.values()))

    # 计算累计价值：基于收益率的累积乘积
    port_daily['value'] = (1 + port_daily['r']).cumprod()

    # 设置初始价值为1，并重新排序索引
    port_daily.loc[port_daily.index[0] - datetime.timedelta(days=1), 'value'] = 1
    port_daily = port_daily.sort_index()

    # 计算年度化波动率：对数收益率的标准差乘以年化因子
    port_vol = np.std(port_daily['r'].apply(lambda r: np.log(r + 1)).dropna(), ddof=1) * np.sqrt(252)

    # 计算年度化收益率：基于累计价值的对数收益率并年化
    port_ret = np.log(port_daily['value'].iloc[-1] / port_daily['value'].iloc[0]) / (len(port_daily) - 1) * 252

    return {"ret_annual": float(port_ret), "vol_annual": float(port_vol)}


def cal_ef2_v4_ultra_fast(data):
    # 获取NumPy数组
    ret_values = data['ret_annual'].values
    vol_values = data['vol_annual'].values

    # 排序索引
    sorted_idx = np.argsort(ret_values)[::-1]

    # 向量化计算累计最小值
    sorted_vol = vol_values[sorted_idx]
    cummin_vol = np.minimum.accumulate(sorted_vol)

    # 识别有效前沿点
    on_ef_sorted = (sorted_vol == cummin_vol)

    # 重新映射到原始顺序
    on_ef = np.zeros(len(data), dtype=bool)
    on_ef[sorted_idx] = on_ef_sorted

    data['on_ef'] = on_ef
    return data


def find_best(weights, hist_value_r, assets_list):
    """
    根据给定的权重组合寻找最优投资组合

    参数:
        weights: 权重组合列表，每个元素是一个权重数组
        hist_value_r (DataFrame): 历史收益率数据，索引为日期，列为资产代码

    返回值:
        tuple: 包含两个元素的元组
            - 第一个元素是包含所有投资组合性能数据的DataFrame
            - 第二个元素是位于有效前沿上的投资组合的前5列数据
    """
    random_port_perf = []
    # 遍历所有权重组合，计算对应的投资组合性能
    for w in weights:
        asset_alloc = dict(zip(assets_list, list(w)))
        alloc_perf = generate_alloc_perf(asset_alloc, hist_value_r)
        random_port_perf.append(dict(**asset_alloc, **alloc_perf))

    # 将投资组合性能数据转换为DataFrame并初始化有效前沿标记
    random_port_perf = pd.DataFrame(random_port_perf)
    random_port_perf['on_ef'] = False

    # 调用高效算法计算有效前沿上的投资组合
    random_port_perf = cal_ef2_v4_ultra_fast(random_port_perf)

    return random_port_perf, random_port_perf[random_port_perf['on_ef'] == True].iloc[:, :5].values


def mark_empirical_frontier(df: pd.DataFrame, ret_col='ret_annual', vol_col='vol_annual', eps=1e-12):
    """收益降序 + 波动前缀最小扫描，标记经验有效前沿"""
    ret = df[ret_col].to_numpy()
    vol = df[vol_col].to_numpy()
    idx = np.argsort(ret)[::-1]  # 收益降序
    sv = vol[idx]
    cummin = np.minimum.accumulate(sv)
    on_sorted = sv <= cummin + eps  # 加数值容差更稳
    on = np.zeros(len(df), dtype=bool)
    on[idx] = on_sorted
    df['on_ef'] = on
    return df


def batch_perf_by_moments(hist_value_r: pd.DataFrame,
                          assets_list,
                          weights: np.ndarray,
                          ann_factor: float = 252.0,
                          use_log_returns: bool = True,
                          normalize_weights: bool = False):
    """
    一次性计算所有权重组合的年化收益与年化波动（矩法）：
    ret_ann = ann * (w @ μ),  vol_ann = sqrt(ann) * sqrt(w^T Σ w)

    参数
    ----
    hist_value_r : DataFrame, 形状 (T, N)，各资产“日简单收益率”，列名包含 assets_list
    assets_list  : 资产列名列表（长度 N，顺序固定）
    weights      : ndarray, 形状 (M, N)，M 个组合的权重
    ann_factor   : 年化因子（交易日 252）
    use_log_returns : True 表示用对数收益口径（推荐，与原代码一致）
    normalize_weights : 是否把每行权重强制归一到和为 1（可选）

    返回
    ----
    df : DataFrame，包含每个组合的权重、ret_annual、vol_annual、on_ef
    frontier_weights : ndarray，经验前沿上组合的权重 (K, N)
    """
    cols = list(assets_list)
    R = hist_value_r[cols].to_numpy(dtype=np.float64)  # (T, N)
    G = np.log1p(R) if use_log_returns else R  # (T, N)  — 日对数收益或简单收益

    # --- 1) 估计矩 ---
    # 均值 μ (N,)
    mu = np.nanmean(G, axis=0)
    # 协方差 Σ (N, N)；注意：若存在 NaN，需要先清洗或做成 pairwise 协方差
    Sigma = np.cov(G, rowvar=False, ddof=1)

    # --- 2) 批量组合绩效 ---
    W = np.asarray(weights, dtype=np.float64)  # (M, N)
    if normalize_weights:
        s = W.sum(axis=1, keepdims=True)
        s[s == 0.0] = 1.0
        W = W / s

    # 年化收益：ret = ann * (W @ μ)
    ret_annual = ann_factor * (W @ mu)  # (M,)
    # 年化波动：vol = sqrt(ann) * sqrt(diag(W Σ W^T))
    quad = np.einsum('ij,jk,ik->i', W, Sigma, W)  # (M,) 逐组合的 w^T Σ w
    vol_annual = np.sqrt(ann_factor) * np.sqrt(np.maximum(quad, 0.0))

    # --- 3) 汇总 & 标记经验前沿 ---
    df = pd.DataFrame(W, columns=cols)
    df['ret_annual'] = ret_annual
    df['vol_annual'] = vol_annual
    df = mark_empirical_frontier(df)

    frontier_weights = df.loc[df['on_ef'], cols].to_numpy()
    return df, frontier_weights


if __name__ == '__main__':
    # 读取数据
    hist_value = pd.read_excel('历史净值数据.xlsx', sheet_name='历史净值数据')
    # 数据输入处理
    hist_value = hist_value.set_index('date')
    hist_value.index = pd.to_datetime(hist_value.index)
    hist_value = hist_value.dropna().sort_index(ascending=True)
    hist_value = hist_value.rename({
        "货基指数": "货币现金类",
        '固收类': '固定收益类',
        '混合类': '混合策略类',
        '权益类': '权益投资类',
        '另类': '另类投资类',
        '安逸型': 'C1',
        '谨慎型': 'C2',
        '稳健型': 'C3',
        '增长型': 'C4',
        '进取型': 'C5',
        '激进型': 'C6'
    }, axis=1)

    # 预设的风险等级配比(proposed_alloc_df)，前端输入
    proposed_alloc = {
        'C1': {'货币现金类': 1.0},
        'C2': {'货币现金类': 0.2, '固定收益类': 0.8},
        'C3': {'货币现金类': 0.1, '固定收益类': 0.55, '混合策略类': 0.35},
        'C4': {'货币现金类': 0.05, '固定收益类': 0.4, '混合策略类': 0.3, '权益投资类': 0.2, '另类投资类': 0.05},
        'C5': {'货币现金类': 0.05, '固定收益类': 0.2, '混合策略类': 0.25, '权益投资类': 0.4, '另类投资类': 0.1},
        'C6': {'货币现金类': 0.05, '固定收益类': 0.1, '混合策略类': 0.15, '权益投资类': 0.6, '另类投资类': 0.1}
    }
    proposed_alloc_df = pd.DataFrame(proposed_alloc).T
    proposed_alloc_df = proposed_alloc_df.fillna(0)

    # 生成资产大类列表(assets_list)
    the_assets_list = proposed_alloc_df.columns.tolist()
    # 大类资产净值数据转为每日收益率数据
    the_hist_value_r = hist_value.pct_change().dropna()

    # 权重组合示例
    # w_r = np.array([[0., 0., 0., 0., 1.], [0., 0.6, 0., 0.3, 0.1], [0., 0.6, 0., 0.4, 0.]])
    w_r = generate_simplex_grid_numba(n_assets=5, resolution=20)
    print(w_r)
    print(w_r.shape)

    s_t = time.time()
    res_1, res_2 = find_best(w_r, the_hist_value_r, the_assets_list)
    print(res_1)
    print(res_2)
    print('======== Elapsed:', time.time() - s_t, '\n')

    s_t = time.time()
    res_3, res_4 = batch_perf_by_moments(the_hist_value_r, the_assets_list, w_r)
    print(res_3)
    print(res_4)
    print('======== Elapsed:', time.time() - s_t, '\n')
