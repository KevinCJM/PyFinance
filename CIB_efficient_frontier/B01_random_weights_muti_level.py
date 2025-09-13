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
from typing import Any, Dict, Iterable, List, Tuple
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from numba import njit, prange
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

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


# ============== 可选：Numba 版 POCS 投影（释放 GIL，便于多线程） ==============
if HAS_NUMBA:
    @njit(cache=True, nogil=True)
    def _pocs_project_numba(
        v: np.ndarray,
        lows: np.ndarray,
        highs: np.ndarray,
        members: np.ndarray,           # 长度 L 的资产索引
        gid_for_member: np.ndarray,    # 长度 L 的对应组 id
        gsize: np.ndarray,             # 长度 m 的组规模
        low_g: np.ndarray,             # 长度 m 的组下限
        up_g: np.ndarray,              # 长度 m 的组上限
        inv_n: float,
        n: int,
        m: int,
        max_iter: int,
        tol: float,
        damping: float,
    ) -> Tuple[boolean, np.ndarray]:
        x = v.copy()

        # 初始盒约束 + sum=1
        for i in range(n):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        s = 0.0
        for i in range(n):
            s += x[i]
        adj = (1.0 - s) * inv_n
        for i in range(n):
            x[i] += adj

        x_prev = x.copy()
        L = members.shape[0]
        t = np.zeros(m, dtype=x.dtype)
        delta = np.zeros(n, dtype=x.dtype)

        for _ in range(max_iter):
            # 盒约束
            for i in range(n):
                if x[i] < lows[i]:
                    x[i] = lows[i]
                elif x[i] > highs[i]:
                    x[i] = highs[i]
            # sum=1
            s = 0.0
            for i in range(n):
                s += x[i]
            adj = (1.0 - s) * inv_n
            for i in range(n):
                x[i] += adj

            if m > 0:
                # 组和 t
                for j in range(m):
                    t[j] = 0.0
                for idx in range(L):
                    t[gid_for_member[idx]] += x[members[idx]]
                # net 校正
                for j in range(n):
                    delta[j] = 0.0
                for j in range(m):
                    over = t[j] - up_g[j]
                    under = low_g[j] - t[j]
                    corr = 0.0
                    if over > 0.0:
                        corr = -over / gsize[j]
                    elif under > 0.0:
                        corr = under / gsize[j]
                    if damping != 1.0:
                        corr *= damping
                    # 散射回资产
                    # 再次遍历成员
                    for idx in range(L):
                        if gid_for_member[idx] == j:
                            delta[members[idx]] += corr
                for i in range(n):
                    x[i] += delta[i]
                # sum=1
                s = 0.0
                for i in range(n):
                    s += x[i]
                adj = (1.0 - s) * inv_n
                for i in range(n):
                    x[i] += adj

            # 收敛
            md = 0.0
            for i in range(n):
                d = x[i] - x_prev[i]
                if d < 0:
                    d = -d
                if d > md:
                    md = d
                x_prev[i] = x[i]
            if md < tol:
                break

        # 严格校验
        for i in range(n):
            if x[i] < lows[i] - 1e-6 or x[i] > highs[i] + 1e-6:
                return False, x
        if m > 0:
            for j in range(m):
                t[j] = 0.0
            for idx in range(L):
                t[gid_for_member[idx]] += x[members[idx]]
            for j in range(m):
                if t[j] < low_g[j] - 1e-6 or t[j] > up_g[j] + 1e-6:
                    return False, x
        s = 0.0
        for i in range(n):
            s += x[i]
        if abs(s - 1.0) > 1e-6:
            return False, x
        return True, x

    def make_pocs_projector_numba(
        single_limits: Iterable[Tuple[float, float]],
        multi_limits: Dict[Tuple[int, ...], Tuple[float, float]],
        max_iter: int = 200,
        tol: float = 1e-9,
        damping: float = 1.0,
    ):
        single_limits = tuple(single_limits)
        n = len(single_limits)
        lows = np.array([a for a, _ in single_limits], dtype=np.float32)
        highs = np.array([b for _, b in single_limits], dtype=np.float32)
        members, gid, gsize, low_g, up_g = _build_group_struct(multi_limits, n)
        members = members.astype(np.int64)
        gid = gid.astype(np.int64)
        gsize = (gsize if gsize.size else np.array([], dtype=np.float32)).astype(np.float32)
        low_g = (low_g if low_g.size else np.array([], dtype=np.float32)).astype(np.float32)
        up_g = (up_g if up_g.size else np.array([], dtype=np.float32)).astype(np.float32)
        m = low_g.size
        inv_n = 1.0 / float(n)

        def project(v: np.ndarray):
            vv = np.array(v, dtype=np.float32, copy=True)
            ok, x = _pocs_project_numba(
                vv, lows, highs, members, gid, gsize, low_g, up_g,
                float(inv_n), int(n), int(m), int(max_iter), float(tol), float(damping)
            )
            if ok:
                return x.astype(np.float64, copy=False)
            return None

        return project

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
    返回 (ret_annual, vol_annual)，保持原逻辑（精确 log1p）：
    R = port_daily @ portfolio_allocs.T；对每列做 log1p 后统计均值与标准差并年化。
    为提升 BLAS 效率，将权重转置为 C 连续内存。
    """
    # 使用 float32 优先（SIMD/内存带宽更友好）；保持原公式(log1p)不变
    if port_daily.dtype != np.float32:
        port_daily = port_daily.astype(np.float32, copy=False)
    if portfolio_allocs.dtype != np.float32:
        portfolio_allocs = portfolio_allocs.astype(np.float32, copy=False)

    T = port_daily.shape[0]
    WT = np.ascontiguousarray(portfolio_allocs.T)  # (N, M) 连续内存利于 GEMM
    R = port_daily @ WT  # (T, M)
    np.log1p(R, out=R)
    ret_annual = (R.sum(axis=0, dtype=np.float32) / float(T)) * float(trading_days)
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

    # 云点数量过大时做等间距抽样以加速渲染（不影响前沿点）
    max_cloud = 5000
    M = vol_all.shape[0]
    if M > max_cloud:
        sel = np.linspace(0, M - 1, max_cloud, dtype=int)
        vol_cloud = vol_all[sel]
        ret_cloud = ret_all[sel]
        weights_cloud = weights_all[sel]
    else:
        vol_cloud, ret_cloud, weights_cloud = vol_all, ret_all, weights_all

    # 全部点
    fig.add_trace(go.Scattergl(
        x=vol_cloud, y=ret_cloud,
        mode='markers',
        name='随机权重数据点',
        marker=dict(color='grey', size=2, opacity=0.45),
        customdata=weights_cloud,
        hovertemplate=hovertemplate
    ))

    # 有效前沿
    fig.add_trace(go.Scattergl(
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
    # 全局采用 float32，以获得更好的吞吐（矩阵乘法/缓存）
    arr = hist_ret_df.values.astype(np.float32, copy=False)
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
    use_numba: bool | None = None,
) -> np.ndarray:
    """使用带约束的随机游走生成一批合法权重 (M,N)。"""
    log(
        "开始通过约束随机游走生成投资组合: "
        f"N={N}, samples={num_samples}, step={step_size}"
    )
    rng = np.random.default_rng(seed)
    if use_numba is None:
        use_numba = HAS_NUMBA
    projector = (
        make_pocs_projector_numba if (use_numba and HAS_NUMBA) else make_pocs_projector
    )(
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


def generate_weights_from_seeds(
    seeds: np.ndarray,
    single_limits: List[Tuple[float, float]],
    multi_limits: Dict[Tuple[int, ...], Tuple[float, float]],
    rng: np.random.Generator,
    samples_per_seed: int = 50,
    per_seed_quota: np.ndarray | None = None,
    step_size: float = 0.3,
    projector_iters: int = 200,
    projector_tol: float = 1e-9,
    projector_damping: float = 1.0,
    parallel_workers: int | None = None,
    use_numba: bool | None = None,
) -> np.ndarray:
    """
    从一组起始点（通常为上一轮的有效前沿点）出发，进行局部随机游走，生成新的合法权重。
    - seeds: (K, N) 起始权重集合（需满足约束）。
    - 返回: (M, N) 新采样的权重集合（可能 < K*samples_per_seed）。
    """
    if seeds.size == 0:
        return np.empty((0, 0), dtype=np.float64)

    N = seeds.shape[1]
    projector = make_pocs_projector(
        single_limits=single_limits,
        multi_limits=multi_limits,
        max_iter=projector_iters,
        tol=projector_tol,
        damping=projector_damping,
    )

    out: List[np.ndarray] = []
    quotas = (
        per_seed_quota.astype(int, copy=False)
        if per_seed_quota is not None
        else np.full(seeds.shape[0], int(samples_per_seed), dtype=int)
    )

    # 线程任务：从单个种子生成 q 个样本
    def _task(seed_idx: int) -> np.ndarray:
        q = int(quotas[seed_idx])
        if q <= 0:
            return np.empty((0, N), dtype=np.float64)
        local_rng = np.random.default_rng(rng.integers(2**63 - 1))
        current = seeds[seed_idx].copy()
        buf = []
        for _ in range(q):
            proposal = current + local_rng.normal(loc=0.0, scale=step_size, size=N)
            adjusted = projector(proposal)
            if adjusted is not None:
                buf.append(adjusted)
                current = adjusted
        if not buf:
            return np.empty((0, N), dtype=np.float64)
        return np.vstack(buf).astype(np.float64, copy=False)

    # 并发执行（默认使用 CPU 核数上限的 1/2~8 之间）
    if parallel_workers is None:
        cpu = os.cpu_count() or 4
        parallel_workers = min(8, max(2, cpu // 2))

    if use_numba is None:
        use_numba = HAS_NUMBA
    projector = (
        make_pocs_projector_numba if (use_numba and HAS_NUMBA) else make_pocs_projector
    )(
        single_limits=single_limits,
        multi_limits=multi_limits,
        max_iter=projector_iters,
        tol=projector_tol,
        damping=projector_damping,
    )

    with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
        futures = [ex.submit(_task, i) for i in range(seeds.shape[0])]
        for fu in as_completed(futures):
            arr = fu.result()
            if arr.size:
                out.append(arr)
    # 可选：如果该 seed 接受数太少，可追加少量尝试；为简洁先忽略
    if not out:
        return np.empty((0, N), dtype=np.float64)
    return np.vstack(out).astype(np.float64, copy=False)


def _compute_local_spacing(vol_sorted: np.ndarray) -> np.ndarray:
    """给定按升序排序的 vol，计算每点的局部间距尺度 d_i（端点用一侧差分）。"""
    n = vol_sorted.shape[0]
    if n == 1:
        return np.array([1.0], dtype=np.float64)
    d = np.empty(n, dtype=np.float64)
    d[0] = vol_sorted[1] - vol_sorted[0]
    d[-1] = vol_sorted[-1] - vol_sorted[-2]
    if n > 2:
        left = vol_sorted[1:-1] - vol_sorted[:-2]
        right = vol_sorted[2:] - vol_sorted[1:-1]
        d[1:-1] = 0.5 * (left + right)
    # 避免非正间距
    d = np.maximum(d, 0.0)
    return d


def _assign_quota_by_spacing(
    seeds_vol: np.ndarray,
    samples_total: int,
    *,
    d_target_bins: int | None = None,
    d_target: float | None = None,
    q_min: float = 0.0,
    q_max: float = 6.0,
    min_quota_per_seed: int = 0,
) -> np.ndarray:
    """
    基于邻近间距的配额分配：
    - 计算局部间距 d_i，与目标间距 d_target 比例转为权重 w_i=clip(d_i/d_target, q_min, q_max)
    - 将权重归一化到整数配额，总和为 samples_total；若需要，确保每个非零权重点至少 min_quota_per_seed。
    - 返回与 seeds_vol 同长的整型配额数组。
    """
    k = seeds_vol.shape[0]
    if k == 0 or samples_total <= 0:
        return np.zeros((0,), dtype=int)
    order = np.argsort(seeds_vol)
    vol_sorted = seeds_vol[order]
    d = _compute_local_spacing(vol_sorted)
    # 目标间距：优先 d_target；否则按分箱估计
    if d_target is None:
        if d_target_bins is None:
            d_target_bins = max(30, min(80, k))
        rng = vol_sorted[-1] - vol_sorted[0]
        d_target = (rng / float(d_target_bins)) if rng > 0 else 1.0
    w = d / float(d_target)
    # 限幅
    w = np.clip(w, q_min, q_max)
    # 若全为 0，退化为均分
    if not np.any(w > 0):
        quotas = np.full(k, samples_total // k, dtype=int)
        quotas[: samples_total % k] += 1
        # 还原原顺序
        out = np.empty_like(quotas)
        out[order] = quotas
        return out
    # 归一化为期望配额
    w_sum = w.sum()
    expected = w / w_sum * float(samples_total)
    base = np.floor(expected).astype(int)
    remainder = expected - base
    remaining = samples_total - int(base.sum())
    if remaining > 0:
        take = np.argsort(-remainder)[:remaining]
        base[take] += 1
    # 最小配额约束（仅对 w>0 的点）
    if min_quota_per_seed > 0:
        mask_pos = w > 0
        need = np.maximum(min_quota_per_seed - base[mask_pos], 0)
        inc_total = int(need.sum())
        if inc_total > 0:
            base[mask_pos] += need
            # 为保持总量，按最小 remainder 或权重小的点回收
            dec_candidates = np.where(~mask_pos, 0, base)[0]
            # 简化：从配额最高的点开始回收
            idx_sorted = np.argsort(-base)
            to_recover = inc_total
            for i in idx_sorted:
                if to_recover == 0:
                    break
                take = min(base[i], to_recover)
                base[i] -= take
                to_recover -= take
            base = np.maximum(base, 0)
    # 还原到原顺序
    out = np.empty_like(base)
    out[order] = base
    return out


def _assign_quota_by_vol_bins(
    seeds_vol: np.ndarray,
    samples_total: int,
    *,
    bins: int = 60,
    weight_mode: str = "inverse",  # "inverse" or "deficit"
    min_quota_per_seed: int = 0,
    max_quota_per_seed: int | None = None,
) -> np.ndarray:
    """
    根据波动率分桶策略为种子分配采样配额。

    该函数将输入的波动率数组划分为多个等宽的桶（bins），并根据每桶中种子的数量，
    为每个种子计算一个权重，进而决定其应得的采样配额。支持两种权重模式：
    - inverse：稀疏桶中的种子获得更高的权重；
    - deficit：基于目标密度与当前桶密度的差值分配权重。

    配额最终会被调整为整数，并满足最小和最大配额限制。

    参数:
        seeds_vol (np.ndarray): 种子的波动率数组，形状为 (k,)。
        samples_total (int): 总共需要分配的样本数量。
        bins (int): 分桶的数量，默认为 60。
        weight_mode (str): 权重分配模式，可选 "inverse" 或 "deficit"，默认为 "inverse"。
        min_quota_per_seed (int): 每个种子的最小配额，默认为 0。
        max_quota_per_seed (int | None): 每个种子的最大配额，若为 None 则无上限，默认为 None。

    返回:
        np.ndarray: 每个种子分配到的整数配额数组，形状为 (k,)。
    """
    k = seeds_vol.shape[0]
    if k == 0 or samples_total <= 0:
        return np.zeros((0,), dtype=int)

    vmin, vmax = float(seeds_vol.min()), float(seeds_vol.max())
    if vmax <= vmin:
        # 所有 vol 几乎相同，退化为均分
        quotas = np.full(k, samples_total // k, dtype=int)
        quotas[: samples_total % k] += 1
        return quotas

    edges = np.linspace(vmin, vmax, bins + 1)
    # digitize 返回 1..bins，修正到 0..bins-1
    bin_idx = np.clip(np.digitize(seeds_vol, edges, right=False) - 1, 0, bins - 1)
    counts = np.bincount(bin_idx, minlength=bins).astype(float)

    if weight_mode == "deficit":
        target = k / float(bins)
        w_bin = np.maximum(target - counts, 0.0)
        # 若全 0，降级为 inverse
        if not np.any(w_bin > 0):
            w_bin = 1.0 / (counts + 1e-9)
    else:  # inverse
        w_bin = 1.0 / (counts + 1e-9)

    w_seed = w_bin[bin_idx]
    # 若仍全 0（极端情况），均分
    if not np.any(w_seed > 0):
        quotas = np.full(k, samples_total // k, dtype=int)
        quotas[: samples_total % k] += 1
        return quotas

    # 归一化为期望配额
    w_sum = w_seed.sum()
    expected = w_seed / w_sum * float(samples_total)
    base = np.floor(expected).astype(int)
    remainder = expected - base
    remaining = samples_total - int(base.sum())
    if remaining > 0:
        take = np.argsort(-remainder)[:remaining]
        base[take] += 1

    # 最小/最大配额约束
    if min_quota_per_seed > 0:
        need = np.maximum(min_quota_per_seed - base, 0)
        inc_total = int(need.sum())
        base += need
        # 回收多出的配额
        if inc_total > 0:
            idx_sorted = np.argsort(-base)
            to_recover = inc_total
            for i in idx_sorted:
                if to_recover == 0:
                    break
                can = base[i] - min_quota_per_seed
                if can <= 0:
                    continue
                take = min(can, to_recover)
                base[i] -= take
                to_recover -= take
    if max_quota_per_seed is not None:
        over = np.maximum(base - int(max_quota_per_seed), 0)
        reduce_total = int(over.sum())
        base -= over
        if reduce_total > 0:
            # 把回收的 quota 再分配给未达上限者
            mask = base < int(max_quota_per_seed)
            if np.any(mask):
                w2 = (int(max_quota_per_seed) - base[mask]).astype(float)
                w2_sum = w2.sum()
                if w2_sum > 0:
                    expected2 = w2 / w2_sum * float(reduce_total)
                    add = np.floor(expected2).astype(int)
                    rem2 = int(reduce_total - add.sum())
                    if rem2 > 0:
                        idx2 = np.argsort(-(expected2 - add))[:rem2]
                        add[idx2] += 1
                    base[mask] += add

    return base



def multi_level_random_walk(
    port_daily_returns: np.ndarray,
    single_limits: List[Tuple[float, float]],
    multi_limits: Dict[Tuple[int, ...], Tuple[float, float]],
    *,
    seed: int = 12345,
    initial_samples: int = 200,
    rounds: int = 3,
    samples_per_round: int = 200,
    step_size_initial: float = 0.99,
    step_size_mid: float = 0.5,
    step_decay: float = 1.0,
    dedup_decimals: int = 4,
    annual_trading_days: float = 252.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    多层随机游走：
    1) 大步长探索整体边界，生成 initial_samples 个权重；
    2) 计算有效前沿；以这些前沿点为“种子”，中步长进行随机游走，生成 samples_per_round；
    3) 合并、去重、校验、重新计算有效前沿；重复若干轮；
    4) 返回最终去重后的权重、其收益与波动，以及最终有效前沿掩码。
    """
    T, N = port_daily_returns.shape
    G, low_g, up_g = build_group_matrix(N, multi_limits)
    lows = np.array([a for a, _ in single_limits], dtype=np.float64)
    highs = np.array([b for _, b in single_limits], dtype=np.float64)

    rng = np.random.default_rng(seed)

    # 第 1 步：大步长探索
    log(
        f"[第0轮] 大步长探索: samples={initial_samples}, step={step_size_initial}"
    )
    W = generate_weights_random_walk(
        N=N,
        single_limits=single_limits,
        multi_limits=multi_limits,
        seed=seed,
        num_samples=initial_samples,
        step_size=step_size_initial,
    )
    if W.size == 0:
        log("大步长探索未获得权重，终止。")
        return W, np.array([]), np.array([]), np.array([], dtype=bool)

    W = deduplicate_weights(W, decimals=dedup_decimals)
    valid_mask = validate_weights_batch(W, lows, highs, G, low_g, up_g, atol=1e-6)
    W = W[valid_mask]
    log(f"[第0轮] 去重且有效的权重数: {W.shape[0]}")

    ret, vol = compute_perf_arrays(
        port_daily_returns, W, trading_days=annual_trading_days, ddof=1
    )
    finite_mask = np.isfinite(ret) & np.isfinite(vol)
    if not np.all(finite_mask):
        W, ret, vol = W[finite_mask], ret[finite_mask], vol[finite_mask]
    ef_mask = cal_ef_mask(ret, vol)
    log(
        f"[第0轮] 有效前沿点数: {int(ef_mask.sum())} / {ef_mask.size}"
    )

    # 第 2..R 轮：前沿种子 + 中步长
    step_mid = step_size_mid
    for r in range(1, rounds + 1):
        seeds = W[ef_mask]
        if seeds.shape[0] == 0:
            log(f"[第{r}轮] 无有效前沿点可作为种子，跳过。")
            break

        # 按种子平均分配生成数
        samples_per_seed = max(1, int(np.ceil(samples_per_round / max(1, seeds.shape[0]))))
        log(
            f"[第{r}轮] 以 {seeds.shape[0]} 个前沿点为种子，"
            f"每个生成 {samples_per_seed}，步长={step_mid}"
        )

        new_W = generate_weights_from_seeds(
            seeds=seeds,
            single_limits=single_limits,
            multi_limits=multi_limits,
            rng=rng,
            samples_per_seed=samples_per_seed,
            step_size=step_mid,
        )
        if new_W.size == 0:
            log(f"[第{r}轮] 未生成新权重，提前停止。")
            break

        # 合并 + 去重 + 校验
        W = np.vstack([W, new_W])
        W = deduplicate_weights(W, decimals=dedup_decimals)
        valid_mask = validate_weights_batch(W, lows, highs, G, low_g, up_g, atol=1e-6)
        W = W[valid_mask]
        log(f"[第{r}轮] 合并后有效权重数: {W.shape[0]}")

        # 重新计算绩效与前沿
        ret, vol = compute_perf_arrays(
            port_daily_returns, W, trading_days=annual_trading_days, ddof=1
        )
        finite_mask = np.isfinite(ret) & np.isfinite(vol)
        if not np.all(finite_mask):
            W, ret, vol = W[finite_mask], ret[finite_mask], vol[finite_mask]
        ef_mask = cal_ef_mask(ret, vol)
        log(
            f"[第{r}轮] 有效前沿点数: {int(ef_mask.sum())} / {ef_mask.size}"
        )

        # 步长衰减（如需）
        step_mid *= float(step_decay)

    return W, ret, vol, ef_mask


def multi_level_random_walk_config(
    port_daily_returns: np.ndarray,
    single_limits: List[Tuple[float, float]],
    multi_limits: Dict[Tuple[int, ...], Tuple[float, float]],
    rounds_config: Dict[int, Dict[str, Any]],
    *,
    dedup_decimals: int = 4,
    annual_trading_days: float = 252.0,
    global_seed: int = 12345,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    基于“字典配置”的多层随机游走：
    - rounds_config: { 轮次: { 参数... } }
      - 第0轮（大步长）：需包含 `samples`（总采样数）与 `step_size`；可选 projector 参数：
        `projector_iters`/`projector_tol`/`projector_damping`/`seed`。
      - 第1..N轮（中步长）：包含 `step_size`，以下二选一：
        - `samples_per_seed`：每个种子生成的数量；
        - `samples_total`：本轮总体目标数量（会均分到各种子并向上取整）。
        同样支持 projector 参数与 `seed`（若未给，沿用 global_seed 生成器）。
    返回：最终 (W, ret, vol, ef_mask)。
    """
    if not rounds_config:
        raise ValueError("rounds_config 为空")

    # 预构建约束矩阵与上下界
    T, N = port_daily_returns.shape
    G, low_g, up_g = build_group_matrix(N, multi_limits)
    lows = np.array([a for a, _ in single_limits], dtype=np.float64)
    highs = np.array([b for _, b in single_limits], dtype=np.float64)

    # Round 0：大步长
    r0_cfg = rounds_config.get(0, {})
    init_samples = int(r0_cfg.get("samples", 200))
    init_step = float(r0_cfg.get("step_size", 0.99))
    p_iters = int(r0_cfg.get("projector_iters", 200))
    p_tol = float(r0_cfg.get("projector_tol", 1e-9))
    p_damping = float(r0_cfg.get("projector_damping", 1.0))
    seed0 = int(r0_cfg.get("seed", global_seed))

    log(f"[第0轮] 大步长探索: samples={init_samples}, step={init_step}")
    W = generate_weights_random_walk(
        N=N,
        single_limits=single_limits,
        multi_limits=multi_limits,
        seed=seed0,
        num_samples=init_samples,
        step_size=init_step,
        projector_iters=p_iters,
        projector_tol=p_tol,
        projector_damping=p_damping,
        use_numba=bool(r0_cfg.get("use_numba", False)),
    )
    if W.size == 0:
        log("[第0轮] 未获得任何权重，终止。")
        return W, np.array([]), np.array([]), np.array([], dtype=bool)

    W = deduplicate_weights(W, decimals=dedup_decimals)
    valid_mask = validate_weights_batch(W, lows, highs, G, low_g, up_g, atol=1e-6)
    W = W[valid_mask]
    log(f"[第0轮] 去重且有效的权重数: {W.shape[0]}")

    ret, vol = compute_perf_arrays(
        port_daily_returns, W, trading_days=annual_trading_days, ddof=1
    )
    finite_mask = np.isfinite(ret) & np.isfinite(vol)
    if not np.all(finite_mask):
        W, ret, vol = W[finite_mask], ret[finite_mask], vol[finite_mask]
    ef_mask = cal_ef_mask(ret, vol)
    log(f"[第0轮] 有效前沿点数: {int(ef_mask.sum())} / {ef_mask.size}")

    # 后续轮次：按配置执行
    max_round = max(rounds_config.keys())
    # 已见权重缓存（按四舍五入到 dedup_decimals 后的字节序列）
    seen: set[bytes] = set()
    if W.size:
        Wc0 = np.ascontiguousarray(np.round(W, dedup_decimals))
        for row in Wc0:
            seen.add(row.tobytes())
    for r in range(1, max_round + 1):
        cfg = rounds_config.get(r)
        if not cfg:
            log(f"[第{r}轮] 未提供配置，跳过。")
            continue

        seeds = W[ef_mask]
        if seeds.size == 0:
            log(f"[第{r}轮] 无有效前沿点可作为种子，提前停止。")
            break

        step_mid = float(cfg.get("step_size", 0.5))
        p_iters = int(cfg.get("projector_iters", 200))
        p_tol = float(cfg.get("projector_tol", 1e-9))
        p_damping = float(cfg.get("projector_damping", 1.0))
        r_seed = int(cfg.get("seed", global_seed))

        per_seed_quota = None
        desc = ""
        if "samples_total" in cfg:
            total = int(cfg.get("samples_total", 200))
            # 直接复用上一轮已计算的波动率，避免重复矩阵乘法
            seeds_vol = vol[ef_mask]
            per_seed_quota = _assign_quota_by_vol_bins(
                seeds_vol=seeds_vol,
                samples_total=total,
                bins=int(cfg.get("vol_bins", 60)),
                weight_mode=str(cfg.get("bin_weight_mode", "inverse")),
                min_quota_per_seed=int(cfg.get("bin_min_quota_per_seed", 0)),
                max_quota_per_seed=cfg.get("bin_max_quota_per_seed"),
            )
            zeros = int((per_seed_quota == 0).sum())
            desc = (
                f"总量 {total}，按 vol 分桶分配（bins={int(cfg.get('vol_bins',60))}, "
                f"min={per_seed_quota.min()}, median={int(np.median(per_seed_quota))}, max={per_seed_quota.max()}, zeros={zeros})"
            )
        else:
            sps = int(cfg.get("samples_per_seed", 1)) or 1
            per_seed_quota = np.full(seeds.shape[0], sps, dtype=int)
            desc = f"每种子 {sps} 次"

        log(f"[第{r}轮] 以 {seeds.shape[0]} 个种子，步长={step_mid}，{desc}")

        rng = np.random.default_rng(r_seed)
        new_W = generate_weights_from_seeds(
            seeds=seeds,
            single_limits=single_limits,
            multi_limits=multi_limits,
            rng=rng,
            per_seed_quota=per_seed_quota,
            samples_per_seed=1,
            step_size=step_mid,
            projector_iters=p_iters,
            projector_tol=p_tol,
            projector_damping=p_damping,
            parallel_workers=int(cfg.get("parallel_workers", 0)) or None,
            use_numba=bool(cfg.get("use_numba", False)),
        )
        if new_W.size == 0:
            log(f"[第{r}轮] 未生成新权重，提前停止。")
            break

        # 仅对新增进行去重与可行性校验，然后增量计算绩效
        if new_W.size:
            # 先在新增集合内部去重
            new_W = deduplicate_weights(new_W, decimals=dedup_decimals)
            # 过滤跨轮重复
            Wc = np.ascontiguousarray(np.round(new_W, dedup_decimals))
            mask_new = np.fromiter((row.tobytes() not in seen for row in Wc), count=Wc.shape[0], dtype=bool)
            new_W = new_W[mask_new]
        if new_W.size:
            # 约束校验
            vmask = validate_weights_batch(new_W, lows, highs, G, low_g, up_g, atol=1e-6)
            new_W = new_W[vmask]
        if new_W.size:
            # 增量计算绩效
            ret_new, vol_new = compute_perf_arrays(
                port_daily_returns, new_W, trading_days=annual_trading_days, ddof=1
            )
            finite_new = np.isfinite(ret_new) & np.isfinite(vol_new)
            if not np.all(finite_new):
                new_W = new_W[finite_new]
                ret_new = ret_new[finite_new]
                vol_new = vol_new[finite_new]
            if new_W.size:
                # 合并
                W = np.vstack([W, new_W])
                ret = np.concatenate([ret, ret_new])
                vol = np.concatenate([vol, vol_new])
                # 标记已见
                for row in np.ascontiguousarray(np.round(new_W, dedup_decimals)):
                    seen.add(row.tobytes())
        log(f"[第{r}轮] 合并后有效权重数: {W.shape[0]}")
        ef_mask = cal_ef_mask(ret, vol)
        log(f"[第{r}轮] 有效前沿点数: {int(ef_mask.sum())} / {ef_mask.size}")

    return W, ret, vol, ef_mask


def run_pipeline(
    excel_path: str,
    sheet_name: str,
    assets_list: List[str],
    single_limits: List[Tuple[float, float]],
    multi_limits: Dict[Tuple[int, ...], Tuple[float, float]],
    *,
    seed: int = 12345,
    # 单层（旧）路径参数（保留以兼容）：
    num_samples: int = 200,
    step_size: float = 0.99,
    # 多层（新）路径参数：
    use_multi_level: bool = True,
    rounds_config: Dict[int, Dict[str, Any]] | None = None,
    initial_samples: int = 200,
    rounds: int = 3,
    samples_per_round: int = 200,
    step_size_initial: float = 0.99,
    step_size_mid: float = 0.5,
    step_decay: float = 1.0,
    # 公共参数
    annual_trading_days: float = 252.0,
    drop_duplicates_decimals: int = 4,
    show_plot: bool = True,
) -> None:
    """
    端到端执行（单层或多层）：数据 -> 权重 -> 指标 -> 前沿 -> 绘图。
    - 默认采用多层随机游走策略；可通过 use_multi_level=False 退回旧流程。
    """
    overall_t0 = time.time()

    # 1) 数据
    port_daily_returns, assets_list = load_returns_from_excel(
        excel_path, sheet_name, assets_list
    )
    T, N = port_daily_returns.shape
    log(f"数据准备完成：T={T}, N={N}")

    if use_multi_level:
        if rounds_config is not None:
            # 基于字典配置的多层策略
            W, ret_annual, vol_annual, ef_mask = multi_level_random_walk_config(
                port_daily_returns=port_daily_returns,
                single_limits=SINGLE_LIMITS if 'SINGLE_LIMITS' in globals() else single_limits,
                multi_limits=MULTI_LIMITS if 'MULTI_LIMITS' in globals() else multi_limits,
                rounds_config=rounds_config,
                dedup_decimals=drop_duplicates_decimals,
                annual_trading_days=annual_trading_days,
                global_seed=seed,
            )
            if W.size == 0:
                log("多层（字典配置）流程未产出结果，终止。")
                return
        else:
            # 新：多层随机游走（参数化）
            W, ret_annual, vol_annual, ef_mask = multi_level_random_walk(
                port_daily_returns=port_daily_returns,
                single_limits=SINGLE_LIMITS if 'SINGLE_LIMITS' in globals() else single_limits,
                multi_limits=MULTI_LIMITS if 'MULTI_LIMITS' in globals() else multi_limits,
                seed=seed,
                initial_samples=initial_samples,
                rounds=rounds,
                samples_per_round=samples_per_round,
                step_size_initial=step_size_initial,
                step_size_mid=step_size_mid,
                step_decay=step_decay,
                dedup_decimals=drop_duplicates_decimals,
                annual_trading_days=annual_trading_days,
            )
            if W.size == 0:
                log("多层流程未产出结果，终止。")
                return
    else:
        # 旧：单层随机游走
        G, low_g, up_g = build_group_matrix(N, multi_limits)
        lows = np.array([a for a, _ in single_limits], dtype=np.float64)
        highs = np.array([b for _, b in single_limits], dtype=np.float64)

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

        W = deduplicate_weights(W, decimals=drop_duplicates_decimals)
        log(f"去重后样本数: {W.shape[0]}")
        log("执行向量化约束校验...")
        valid_mask = validate_weights_batch(W, lows, highs, G, low_g, up_g, atol=1e-6)
        W = W[valid_mask]
        log(f"校验完成，有效权重: {W.shape[0]}")
        if W.shape[0] == 0:
            log("有效权重为空，流程终止。")
            return

        log("批量计算年化收益与年化波动...")
        t0 = time.time()
        ret_annual, vol_annual = compute_perf_arrays(
            port_daily_returns, W, trading_days=annual_trading_days, ddof=1
        )
        finite_mask = np.isfinite(ret_annual) & np.isfinite(vol_annual)
        if not np.all(finite_mask):
            W = W[finite_mask]
            ret_annual = ret_annual[finite_mask]
            vol_annual = vol_annual[finite_mask]
            log("发现无效样本，已过滤。")
        log(f"绩效计算完成，用时 {time.time() - t0:.2f}s")

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

    # 随机游走与指标参数（多层）
    RANDOM_SEED = 12345
    # 旧单层参数（如需退回旧流程可用）：
    NUM_SAMPLES = 200
    STEP_SIZE = 0.99
    # 多层参数（两种方式二选一）
    USE_MULTI_LEVEL = True
    # A) 字典方式（推荐）：逐轮灵活配置
    ROUNDS_CONFIG: Dict[int, Dict[str, Any]] = {
        0: {"samples": 300, "step_size": 0.99},
        1: {"samples_total": 1000, "step_size": 0.1, "vol_bins": 100, "parallel_workers": 8},
        2: {"samples_total": 2000, "step_size": 0.1, "vol_bins": 200, "parallel_workers": 8},
        3: {"samples_total": 3000, "step_size": 0.05, "vol_bins": 300, "parallel_workers": 8},
        4: {"samples_total": 5000, "step_size": 0.01, "vol_bins": 500, "parallel_workers": 8},
    }
    # B) 旧参数化方式（保留兼容，不用可忽略）
    INITIAL_SAMPLES = 300
    ROUNDS = 3
    SAMPLES_PER_ROUND = 300
    STEP_SIZE_INITIAL = 0.99
    STEP_SIZE_MID = 0.5
    STEP_DECAY = 1.0
    TRADING_DAYS = 252.0
    DEDUP_DECIMALS = 4

    # 是否显示图表（自测时可置 False，避免打开窗口）
    SHOW_PLOT = False

    log("程序开始运行")
    run_pipeline(
        excel_path=EXCEL_PATH,
        sheet_name=SHEET_NAME,
        assets_list=ASSETS,
        single_limits=SINGLE_LIMITS,
        multi_limits=MULTI_LIMITS,
        seed=RANDOM_SEED,
        # 旧单层参数
        num_samples=NUM_SAMPLES,
        step_size=STEP_SIZE,
        # 多层参数
        use_multi_level=USE_MULTI_LEVEL,
        rounds_config=ROUNDS_CONFIG,
        initial_samples=INITIAL_SAMPLES,
        rounds=ROUNDS,
        samples_per_round=SAMPLES_PER_ROUND,
        step_size_initial=STEP_SIZE_INITIAL,
        step_size_mid=STEP_SIZE_MID,
        step_decay=STEP_DECAY,
        # 公共参数
        annual_trading_days=TRADING_DAYS,
        drop_duplicates_decimals=DEDUP_DECIMALS,
        show_plot=SHOW_PLOT,
    )
    log("程序结束")
