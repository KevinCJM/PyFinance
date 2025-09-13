"""
B01_random_weights_faster.py

说明
- 在保证原有功能的基础上，重构为“函数化”结构，增加清晰注释与带时间戳的日志输出。
- 主要流程：读取数据 -> 约束设定 -> 随机游走生成权重 -> 指标计算 -> 有效前沿筛选 -> 绘图。
- 性能重点：尽量使用 NumPy 向量化、减少中间对象与 DataFrame 依赖。
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ===================== 0) 工具函数 =====================

def _ts() -> str:
    """返回当前时间戳字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    """标准化日志输出，带时间戳。"""
    print(f"[{_ts()}] {msg}")

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


def build_group_matrix(
    n_assets: int, multi_limits: Dict[Tuple[int, ...], Tuple[float, float]]
):
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
    G: np.ndarray | None = None,
    low_g: np.ndarray | None = None,
    up_g: np.ndarray | None = None,
    atol: float = 1e-6,
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

def _build_group_struct(
    multi_limits: Dict[Tuple[int, ...], Tuple[float, float]], n: int
):
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
    damping: float = 1.0,
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
    ddof: int = 1,
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
    title: str = "约束随机游走生成的投资组合与有效前沿",
    show: bool = True,
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
    if show:
        fig.show()


# ===================== 4) 数据与流程函数 =====================

def load_returns_from_excel(
    excel_path: str,
    sheet_name: str,
    assets_list: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """从 Excel 读取净值数据，生成日收益二维数组 (T,N)。"""
    log(f"加载数据: {excel_path} | sheet={sheet_name}")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.dropna().sort_index(ascending=True)
    df = df.rename(
        {
            "货基指数": "货币现金类",
            "固收类": "固定收益类",
            "混合类": "混合策略类",
            "权益类": "权益投资类",
            "另类": "另类投资类",
            "安逸型": "C1",
            "谨慎型": "C2",
            "稳健型": "C3",
            "增长型": "C4",
            "进取型": "C5",
            "激进型": "C6",
        },
        axis=1,
    )

    # 若部分资产列不存在，报错以避免静默问题
    missing = [c for c in assets_list if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列: {missing}")

    hist_ret_df = df[assets_list].pct_change().dropna()
    arr = hist_ret_df.values.astype(np.float64, copy=False)
    log(f"数据加载完成，样本天数={arr.shape[0]}，资产数={arr.shape[1]}")
    return arr, assets_list


def deduplicate_weights(W: np.ndarray, decimals: int = 4) -> np.ndarray:
    """按小数位去重行权重矩阵，减少后续计算量。"""
    W_round = np.round(W, decimals)
    Wc = np.ascontiguousarray(W_round)
    view = Wc.view(np.dtype((np.void, Wc.dtype.itemsize * Wc.shape[1])))
    _, uniq_idx = np.unique(view, return_index=True)
    return W[np.sort(uniq_idx)]


def generate_weights_random_walk(
    N: int,
    single_limits: List[Tuple[float, float]],
    multi_limits: Dict[Tuple[int, ...], Tuple[float, float]],
    seed: int = 12345,
    num_samples: int = 200,
    step_size: float = 0.99,
    projector_iters: int = 200,
    projector_tol: float = 1e-9,
    projector_damping: float = 1.0,
) -> np.ndarray:
    """使用带约束的随机游走生成一批合法权重 (M,N)。"""
    log(
        "开始通过约束随机游走生成投资组合: "
        f"N={N}, samples={num_samples}, step={step_size}"
    )
    rng = np.random.default_rng(seed)
    projector = make_pocs_projector(
        single_limits=single_limits,
        multi_limits=multi_limits,
        max_iter=projector_iters,
        tol=projector_tol,
        damping=projector_damping,
    )

    current = np.full(N, 1.0 / N, dtype=np.float64)
    out: List[np.ndarray] = []
    t0 = time.time()
    for _ in range(num_samples):
        proposal = current + rng.normal(loc=0.0, scale=step_size, size=N)
        adjusted = projector(proposal)
        if adjusted is not None:
            out.append(adjusted)
            current = adjusted
    dt = time.time() - t0
    log(f"随机游走完成，生成 {len(out)} 条候选，用时 {dt:.2f}s")

    if not out:
        return np.empty((0, N), dtype=np.float64)
    return np.vstack(out).astype(np.float64, copy=False)


def run_pipeline(
    excel_path: str,
    sheet_name: str,
    assets_list: List[str],
    single_limits: List[Tuple[float, float]],
    multi_limits: Dict[Tuple[int, ...], Tuple[float, float]],
    seed: int = 12345,
    num_samples: int = 200,
    step_size: float = 0.99,
    annual_trading_days: float = 252.0,
    drop_duplicates_decimals: int = 4,
    show_plot: bool = True,
) -> None:
    """端到端执行：数据 -> 权重 -> 指标 -> 前沿 -> 绘图。"""
    overall_t0 = time.time()

    # 1) 数据
    port_daily_returns, assets_list = load_returns_from_excel(
        excel_path, sheet_name, assets_list
    )
    T, N = port_daily_returns.shape
    log(f"数据准备完成：T={T}, N={N}")

    # 2) 约束与组矩阵
    G, low_g, up_g = build_group_matrix(N, multi_limits)
    lows = np.array([a for a, _ in single_limits], dtype=np.float64)
    highs = np.array([b for _, b in single_limits], dtype=np.float64)

    # 3) 随机游走生成权重
    W = generate_weights_random_walk(
        N=N,
        single_limits=single_limits,
        multi_limits=multi_limits,
        seed=seed,
        num_samples=num_samples,
        step_size=step_size,
    )
    if W.shape[0] == 0:
        log("未获得任何权重，流程终止。")
        return

    # 4) 去重 + 校验
    W = deduplicate_weights(W, decimals=drop_duplicates_decimals)
    log(f"去重后样本数: {W.shape[0]}")
    log("执行向量化约束校验...")
    valid_mask = validate_weights_batch(W, lows, highs, G, low_g, up_g, atol=1e-6)
    W = W[valid_mask]
    log(f"校验完成，有效权重: {W.shape[0]}")
    if W.shape[0] == 0:
        log("有效权重为空，流程终止。")
        return

    # 5) 批量计算绩效
    log("批量计算年化收益与年化波动...")
    t0 = time.time()
    ret_annual, vol_annual = compute_perf_arrays(
        port_daily_returns, W, trading_days=annual_trading_days, ddof=1
    )
    # 过滤无效
    finite_mask = np.isfinite(ret_annual) & np.isfinite(vol_annual)
    if not np.all(finite_mask):
        W = W[finite_mask]
        ret_annual = ret_annual[finite_mask]
        vol_annual = vol_annual[finite_mask]
        log("发现无效样本，已过滤。")
    log(f"绩效计算完成，用时 {time.time() - t0:.2f}s")

    # 6) 有效前沿
    ef_mask = cal_ef_mask(ret_annual, vol_annual)
    log(f"有效前沿点数: {ef_mask.sum()} / {ef_mask.size}")

    # 7) 绘图
    log("生成交互式图表...")
    plot_efficient_frontier_arrays(
        vol_all=vol_annual,
        ret_all=ret_annual,
        weights_all=W,
        ef_mask=ef_mask,
        asset_names=assets_list,
        title="约束随机游走生成的投资组合与有效前沿",
        show=show_plot,
    )

    log(f"流程完成，总耗时 {time.time() - overall_t0:.2f}s")


# ===================== 4) 主流程 =====================

if __name__ == "__main__":
    # 主要参数（可按需调整）
    EXCEL_PATH = "历史净值数据.xlsx"
    SHEET_NAME = "历史净值数据"
    ASSETS = ["货币现金类", "固定收益类", "混合策略类", "权益投资类", "另类投资类"]

    # 单资产上下限（示例：全部 [0, 1]）
    SINGLE_LIMITS: List[Tuple[float, float]] = [(0.0, 1.0)] * len(ASSETS)
    # 多资产联合约束（示例为空；可设如 {(0,1):(0.2,0.6)}）
    MULTI_LIMITS: Dict[Tuple[int, ...], Tuple[float, float]] = {}

    # 随机游走与指标参数
    RANDOM_SEED = 12345
    NUM_SAMPLES = 200
    STEP_SIZE = 0.99
    TRADING_DAYS = 252.0
    DEDUP_DECIMALS = 4

    # 是否显示图表（自测时可置 False，避免打开窗口）
    SHOW_PLOT = True

    log("程序开始运行")
    run_pipeline(
        excel_path=EXCEL_PATH,
        sheet_name=SHEET_NAME,
        assets_list=ASSETS,
        single_limits=SINGLE_LIMITS,
        multi_limits=MULTI_LIMITS,
        seed=RANDOM_SEED,
        num_samples=NUM_SAMPLES,
        step_size=STEP_SIZE,
        annual_trading_days=TRADING_DAYS,
        drop_duplicates_decimals=DEDUP_DECIMALS,
        show_plot=SHOW_PLOT,
    )
    log("程序结束")
