# -*- encoding: utf-8 -*-
"""
@File: B01_random_weights_fast.py
@Modify Time: 2025/09/13
@Author: Kevin-Chen (optimized by assistant)
@Descriptions:
  高性能版本：全流程尽量使用 NumPy 数组、减少内存复制与 DataFrame 依赖。
  - 随机游走 + POCS/Dykstra 约束投影（向量化组半空间校正）
  - 批量指标计算（一次 GEMM，原地 log1p，单次归约）
  - 有效前沿识别（O(N log N)）
  - Plotly 绘图（使用 customdata + hovertemplate，无 DataFrame.apply）
"""
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Any, Dict, Tuple, Iterable


# ===================== 0) 工具函数 =====================

def cal_ef_mask(ret_annual: np.ndarray, vol_annual: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    识别有效前沿：对收益降序排序，保留波动率新的“前缀最小”点。
    返回布尔掩码（与原顺序对齐）。
    """
    idx = np.argsort(ret_annual)[::-1]
    vol_sorted = vol_annual[idx]
    cummin_vol = np.minimum.accumulate(vol_sorted)
    on_ef_sorted = vol_sorted <= (cummin_vol + eps)

    on_ef = np.zeros(ret_annual.shape[0], dtype=bool)
    on_ef[idx] = on_ef_sorted
    return on_ef


def build_group_matrix(n_assets: int, multi_limits: Dict[Tuple[int, ...], Tuple[float, float]]):
    """
    构建稠密组矩阵 G (m, n)，用于批量校验（权重 @ G.T）。
    n_assets 通常很小（本例为 5），用稠密矩阵足够快且简洁。
    """
    if not multi_limits:
        return None, None, None
    m = len(multi_limits)
    G = np.zeros((m, n_assets), dtype=np.float64)
    low = np.empty(m, dtype=np.float64)
    up = np.empty(m, dtype=np.float64)
    g = 0
    for idx_tuple, (lo, hi) in multi_limits.items():
        G[g, np.asarray(idx_tuple, dtype=np.int64)] = 1.0
        low[g] = float(lo)
        up[g] = float(hi)
        g += 1
    return G, low, up


def validate_weights_batch(
        W: np.ndarray,
        lows: np.ndarray,
        highs: np.ndarray,
        G: np.ndarray = None,
        low_g: np.ndarray = None,
        up_g: np.ndarray = None,
        atol: float = 1e-6
) -> np.ndarray:
    """
    向量化校验一批权重（行和=1、单资产上下限、组上下限）。
    返回布尔掩码（True 表示合法）。
    """
    # sum=1
    ok_sum = np.isclose(W.sum(axis=1), 1.0, atol=atol)

    # 单资产
    ok_low = (W >= (lows - atol)).all(axis=1)
    ok_up = (W <= (highs + atol)).all(axis=1)
    ok_single = ok_low & ok_up

    # 组约束
    if G is None:
        ok_group = np.ones(W.shape[0], dtype=bool)
    else:
        S = W @ G.T  # (M, m)
        ok_g_low = (S >= (low_g - atol)).all(axis=1)
        ok_g_up = (S <= (up_g + atol)).all(axis=1)
        ok_group = ok_g_low & ok_g_up

    return ok_sum & ok_single & ok_group


# ===================== 1) POCS 约束投影（高速） =====================

def _build_group_struct(multi_limits: Dict[Tuple[int, ...], Tuple[float, float]], n: int):
    """
    为 POCS 投影构建“拼接索引 + 组id”结构，供 bincount 做 segment reduce / scatter。
    """
    if not multi_limits:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    members_list, gid_list, gsize_list, low_list, up_list = [], [], [], [], []
    g = 0
    for idx_tuple, (lo, hi) in multi_limits.items():
        idx = np.asarray(idx_tuple, dtype=np.int64)
        if idx.size == 0:
            continue
        members_list.append(idx)
        gid_list.append(np.full(idx.size, g, dtype=np.int64))
        gsize_list.append(float(idx.size))
        low_list.append(float(lo))
        up_list.append(float(hi))
        g += 1

    if g == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    members = np.concatenate(members_list, axis=0)
    gid = np.concatenate(gid_list, axis=0)
    gsize = np.asarray(gsize_list, dtype=np.float64)
    low = np.asarray(low_list, dtype=np.float64)
    up = np.asarray(up_list, dtype=np.float64)
    return members, gid, gsize, low, up


def make_pocs_projector(
        single_limits: Iterable[Tuple[float, float]],
        multi_limits: Dict[Tuple[int, ...], Tuple[float, float]],
        max_iter: int = 200,
        tol: float = 1e-9,
        damping: float = 1.0
):
    """
    预编译 POCS 约束，返回高性能投影函数 project(v)->x 或 None。
    多次调用时摊销构建成本。
    """
    single_limits = tuple(single_limits)
    n = len(single_limits)

    lows = np.fromiter((a for a, _ in single_limits), count=n, dtype=np.float64)
    highs = np.fromiter((b for _, b in single_limits), count=n, dtype=np.float64)
    members, gid, gsize, low_g, up_g = _build_group_struct(multi_limits, n)
    m = low_g.size
    inv_n = 1.0 / n

    def project(v: np.ndarray):
        x = np.array(v, dtype=np.float64, copy=True)

        # 初始：盒约束 + sum=1
        np.clip(x, lows, highs, out=x)
        x += (1.0 - x.sum()) * inv_n

        x_prev = x.copy()
        for _ in range(max_iter):
            # 盒约束
            np.clip(x, lows, highs, out=x)
            # sum=1
            x += (1.0 - x.sum()) * inv_n

            if m:
                # 组和
                t = np.bincount(gid, weights=x[members], minlength=m).astype(np.float64, copy=False)
                over = np.maximum(t - up_g, 0.0)
                under = np.maximum(low_g - t, 0.0)
                net = (-over + under) / gsize
                if damping != 1.0:
                    net *= damping
                # 散射回资产
                delta_x = np.bincount(members, weights=net[gid], minlength=n).astype(np.float64, copy=False)
                x += delta_x
                # sum=1
                x += (1.0 - x.sum()) * inv_n

            # 收敛
            if np.max(np.abs(x - x_prev)) < tol:
                break
            x_prev[:] = x

        # 严格校验
        if (x < lows - 1e-6).any() or (x > highs + 1e-6).any():
            return None
        if m:
            t = np.bincount(gid, weights=x[members], minlength=m).astype(np.float64, copy=False)
            if (t < low_g - 1e-6).any() or (t > up_g + 1e-6).any():
                return None
        if not np.isclose(x.sum(), 1.0, atol=1e-6):
            return None
        return x

    return project


# ===================== 2) 批量指标计算（高效） =====================

def compute_perf_arrays(
        port_daily: np.ndarray,  # (T, N)
        portfolio_allocs: np.ndarray,  # (M, N)
        trading_days: float = 252.0,
        ddof: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (ret_annual, vol_annual)，不构建 DataFrame。
    """
    # dtype
    if port_daily.dtype != np.float64:
        port_daily = port_daily.astype(np.float64, copy=False)
    if portfolio_allocs.dtype != np.float64:
        portfolio_allocs = portfolio_allocs.astype(np.float64, copy=False)

    T = port_daily.shape[0]
    # 一次 GEMM
    R = port_daily @ portfolio_allocs.T  # (T, M)
    # 原地 log1p
    np.log1p(R, out=R)
    # 年化对数收益
    ret_annual = (R.sum(axis=0) / float(T)) * float(trading_days)
    # 年化波动
    vol_annual = R.std(axis=0, ddof=ddof) * np.sqrt(float(trading_days))
    return ret_annual, vol_annual


# ===================== 3) 绘图（纯数组 + hovertemplate） =====================

def plot_efficient_frontier_arrays(
        vol_all: np.ndarray,
        ret_all: np.ndarray,
        weights_all: np.ndarray,  # (M, N)
        ef_mask: np.ndarray,  # (M,)
        asset_names: List[str],
        title: str = '约束随机游走生成的投资组合与有效前沿'
):
    # 自定义 hover：使用 customdata 避免 Python 循环拼接
    # customdata: 权重矩阵 (M, N)
    hover_assets = "<br>".join([f"{name}: %{{customdata[{i}]:.1%}}" for i, name in enumerate(asset_names)])
    hovertemplate = (
            "年化波动率: %{x:.2%}<br>"
            "年化收益率: %{y:.2%}<br><br>"
            "<b>资产权重</b><br>" + hover_assets + "<extra></extra>"
    )

    fig = go.Figure()

    # 全部点
    fig.add_trace(go.Scatter(
        x=vol_all, y=ret_all,
        mode='markers',
        name='随机权重数据点',
        marker=dict(color='grey', size=2, opacity=0.45),
        customdata=weights_all,
        hovertemplate=hovertemplate
    ))

    # 有效前沿
    fig.add_trace(go.Scatter(
        x=vol_all[ef_mask], y=ret_all[ef_mask],
        mode='markers',
        name='有效前沿数据点',
        marker=dict(color='blue', size=3, opacity=0.85),
        customdata=weights_all[ef_mask],
        hovertemplate=hovertemplate
    ))

    fig.update_layout(
        title=title,
        xaxis_title='年化波动率 (Annual Volatility)',
        yaxis_title='年化收益率 (Annual Return)',
        legend_title='图例',
        hovermode='closest'
    )
    fig.show()


# ===================== 4) 主流程 =====================

if __name__ == '__main__':
    s_t = time.time()

    # --- 1. 数据加载 ---
    hist_value = pd.read_excel('历史净值数据.xlsx', sheet_name='历史净值数据')
    hist_value = hist_value.set_index('date')
    hist_value.index = pd.to_datetime(hist_value.index)
    hist_value = hist_value.dropna().sort_index(ascending=True)
    hist_value = hist_value.rename({
        "货基指数": "货币现金类", '固收类': '固定收益类', '混合类': '混合策略类',
        '权益类': '权益投资类', '另类': '另类投资类', '安逸型': 'C1',
        '谨慎型': 'C2', '稳健型': 'C3', '增长型': 'C4',
        '进取型': 'C5', '激进型': 'C6'
    }, axis=1)

    assets_list = ['货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类']

    # 使用 pandas 计算收益，再转 NumPy
    hist_ret_df = hist_value[assets_list].pct_change().dropna()
    port_daily_returns = hist_ret_df.values.astype(np.float64, copy=False)  # (T, N)
    T, N = port_daily_returns.shape

    # --- 2. 约束设置 ---
    # 单资产上下限（示例：0~100%）
    single_limits: List[Tuple[float, float]] = [(0.0, 1.0)] * N
    # 多资产联合约束（示例为空；如需可填 {(0,1):(0.2,0.6), (2,3,4):(0.1,0.5)} 等）
    multi_limits: Dict[Tuple[int, ...], Tuple[float, float]] = {}

    # 预编译校验矩阵（仅用于批量校验；投影使用拼接索引）
    G, low_g, up_g = build_group_matrix(N, multi_limits)

    # --- 3. 约束随机游走（高效 POCS 投影，零拷贝操作） ---
    print("开始通过约束随机游走生成投资组合...")
    rng = np.random.default_rng(12345)
    num_samples = 100
    step_size = 0.99

    lows = np.array([a for a, _ in single_limits], dtype=np.float64)
    highs = np.array([b for _, b in single_limits], dtype=np.float64)

    projector = make_pocs_projector(
        single_limits=single_limits,
        multi_limits=multi_limits,
        max_iter=200,
        tol=1e-9,
        damping=1.0
    )

    current_weights = np.full(N, 1.0 / N, dtype=np.float64)
    final_weights: List[np.ndarray] = []

    s_t_2 = time.time()
    for _ in range(num_samples):
        # 正态扰动（不创建多余中间对象）
        proposal = current_weights + rng.normal(loc=0.0, scale=step_size, size=N)
        adjusted = projector(proposal)
        if adjusted is not None:
            final_weights.append(adjusted)
            current_weights = adjusted

    print(f"成功生成 {len(final_weights)} 个候选投资组合。耗时 {time.time() - s_t_2:.2f} 秒。")

    if not final_weights:
        print("未能生成任何有效的投资组合，无法进行后续计算和绘图。")
        print(f"总耗时: {time.time() - s_t:.2f} 秒")
        raise SystemExit(0)

    # 合并为矩阵
    W = np.vstack(final_weights).astype(np.float64, copy=False)  # (M, N)

    # 按权重精度去重，显著降低后续计算量; 例如四位小数：
    W_round = np.round(W, 4)
    # 利用 view 对行去重
    Wc = np.ascontiguousarray(W_round)
    view = Wc.view(np.dtype((np.void, Wc.dtype.itemsize * Wc.shape[1])))
    _, uniq_idx = np.unique(view, return_index=True)
    W = W[np.sort(uniq_idx)]
    print(f"去重后权重数量: {W.shape[0]}")

    # --- 4. 显式批量校验（向量化） ---
    print("正在对所有生成的权重进行最终校验...")
    valid_mask = validate_weights_batch(W, lows, highs, G, low_g, up_g, atol=1e-6)
    W = W[valid_mask]
    print(f"校验完成。有效权重数量: {W.shape[0]}")

    if W.shape[0] == 0:
        print("有效权重为空。")
        print(f"总耗时: {time.time() - s_t:.2f} 秒")
        raise SystemExit(0)

    # --- 5. 批量计算收益与风险（数组版） ---
    print("正在批量计算所有组合的收益与风险...")
    s_t_3 = time.time()
    ret_annual, vol_annual = compute_perf_arrays(port_daily_returns, W, trading_days=252.0, ddof=1)

    # 过滤无效（例如包含 -100% 导致 -inf 的组合）
    finite_mask = np.isfinite(ret_annual) & np.isfinite(vol_annual)
    if not np.all(finite_mask):
        W = W[finite_mask]
        ret_annual = ret_annual[finite_mask]
        vol_annual = vol_annual[finite_mask]

    # 有效前沿布尔掩码
    ef_mask = cal_ef_mask(ret_annual, vol_annual)

    print(f"计算完成, 耗时: {time.time() - s_t_3:.2f} 秒")

    # --- 6. 绘图（数组直连） ---
    print("正在生成交互式图表...")
    plot_efficient_frontier_arrays(
        vol_all=vol_annual,
        ret_all=ret_annual,
        weights_all=W,
        ef_mask=ef_mask,
        asset_names=assets_list,
        title='约束随机游走生成的投资组合与有效前沿'
    )

    print(f"总耗时: {time.time() - s_t:.2f} 秒")
