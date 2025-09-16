"""
B01_random_weights_faster.py

说明
- 在保证原有功能的基础上，重构为“函数化”结构，增加清晰注释与带时间戳的日志输出。
- 主要流程：读取数据 -> 约束设定 -> 随机游走生成权重 -> 指标计算 -> 有效前沿筛选 -> 绘图。
- 性能重点：尽量使用 NumPy 向量化、减少中间对象与 DataFrame 依赖。
"""

from __future__ import annotations  # 启用 Python 3.10+ 的类型注解特性

import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from numba import njit, prange

    HAS_NUMBA = True
    print("Numba 可用，启用 JIT 加速。")
except Exception:
    HAS_NUMBA = False
    print("Numba 不可用，使用纯 Python 版本。")

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


# ===================== 1) POCS 约束投影 =====================

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
            members: np.ndarray,  # 长度 L 的资产索引
            gid_for_member: np.ndarray,  # 长度 L 的对应组 id
            gsize: np.ndarray,  # 长度 m 的组规模
            low_g: np.ndarray,  # 长度 m 的组下限
            up_g: np.ndarray,  # 长度 m 的组上限
            inv_n: float,
            n: int,
            m: int,
            max_iter: int,
            tol: float,
            damping: float,
    ) -> Tuple[bool, np.ndarray]:
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


# ===================== 2) 批量指标计算 =====================

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


def compute_var_parametric_arrays(
        port_daily: np.ndarray,  # (T, N)
        portfolio_allocs: np.ndarray,  # (M, N)
        *,
        confidence: float = 0.95,
        horizon_days: float = 1.0,
        return_type: str = "simple",  # "simple" 或 "log"
        ddof: int = 1,
        clip_non_negative: bool = True,
) -> np.ndarray:
    """
    基于参数法（方差-协方差法，正态近似）计算投资组合 VaR（取非负值）。

    参数：
        port_daily : np.ndarray, shape (T, N)
            每日资产收益数据，T 为时间长度，N 为资产数量。
        portfolio_allocs : np.ndarray, shape (M, N)
            投资组合权重矩阵，M 为组合数量，N 为资产数量。
        confidence : float, default=0.95
            置信水平，用于确定分位点，取值范围 (0, 1)。
        horizon_days : float, default=1.0
            风险持有期天数，用于缩放均值和标准差。
        return_type : str, default="simple"
            收益类型，可选 "simple"（简单收益）或 "log"（对数收益）。
        ddof : int, default=1
            计算标准差时的自由度调整参数。
        clip_non_negative : bool, default=True
            是否将 VaR 结果裁剪为非负值。

    返回：
        np.ndarray, shape (M,)
            每个投资组合在指定置信水平下的 VaR 值。
    """
    # 安全性：限定参数范围
    confidence = float(confidence)
    confidence = min(max(confidence, 1e-6), 1 - 1e-6)
    horizon_days = max(float(horizon_days), 1e-12)

    # 组合日收益：R = port_daily @ W.T -> (T, M)
    if port_daily.dtype != np.float32:
        port_daily = port_daily.astype(np.float32, copy=False)
    if portfolio_allocs.dtype != np.float32:
        portfolio_allocs = portfolio_allocs.astype(np.float32, copy=False)

    WT = np.ascontiguousarray(portfolio_allocs.T)  # 转置并确保内存连续 (N, M)
    R = port_daily @ WT  # 得到每个组合在每个时间点的收益 (T, M)

    # 根据收益类型选择使用简单收益或对数收益
    if return_type == "log":
        X = np.log1p(R)
    else:
        X = R

    # 计算每组收益的均值和标准差
    mu = X.mean(axis=0, dtype=np.float32)
    sigma = X.std(axis=0, ddof=ddof)

    # 持有期缩放：对数收益可加总；简单收益使用平方根法则近似
    h = float(horizon_days)
    mu_h = mu * h
    sigma_h = sigma * np.sqrt(h)

    # 正态分布左尾分位数 z_{1 - confidence}
    try:
        from statistics import NormalDist

        z = NormalDist().inv_cdf(1.0 - confidence)
    except Exception:
        # 兜底：95% 左尾近似 -1.64485
        z = -1.645

    # 计算 VaR：VaR = max(0, -(mu_h + z * sigma_h))
    var_val = -(mu_h + z * sigma_h)
    if clip_non_negative:
        var_val = np.maximum(var_val, 0.0)
    # 取绝对值并转换为 float32 类型
    var_val = np.abs(var_val).astype(np.float32, copy=False)
    return var_val


# ===================== 2.5) 基于求解器的前沿（用于初始种子） =====================
def build_frontier_by_risk_grid(
        port_daily_returns: np.ndarray,
        single_limits: List[Tuple[float, float]],
        multi_limits: Dict[Tuple[int, ...], Tuple[float, float]],
        *,
        risk_metric: str = "vol",  # "vol" 或 "var"
        var_params: Dict[str, Any] | None = None,
        n_grid: int = 150,
        ridge: float = 1e-8,
        solver: str = "ECOS",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用凸优化求解在线性权重约束下的有效前沿点集，作为初始种子。
    - 当 risk_metric="vol" 时，问题为 QCQP/二阶锥：max μᵀw s.t. wᵀΣw ≤ r²
    - 当 risk_metric="var" 时，使用参数法 VaR（正态近似）建模为 SOCP：
        VaR(w) = α√h·s − h·μᵀw；约束 VaR(w) ≤ r，且 wᵀΣw ≤ s²
    返回： (W, ret_annual, risk_array)
    注意：本函数在内部仅在需要时 import cvxpy，若缺失将抛出可读错误信息。
    """
    try:
        import cvxpy as cp
    except Exception as e:
        raise RuntimeError("需要安装 cvxpy 以使用求解器初始化功能: pip install cvxpy") from e

    # 简单收益下的 μ 与协方差；Σ 做轻微 ridge 保证 PSD
    r = port_daily_returns.astype(np.float64, copy=False)
    mu = r.mean(axis=0)
    Sigma = np.cov(r, rowvar=False, ddof=1)
    Sigma = ((Sigma + Sigma.T) * 0.5) + float(ridge) * np.eye(Sigma.shape[0])
    N = Sigma.shape[0]

    # 构造 Σ^{1/2}，用于二阶锥约束：||Σ^{1/2} w||_2 ≤ s;   优先 Cholesky；若失败则用特征分解稳健构造
    try:
        Sigma_sqrt = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(Sigma)
        vals = np.clip(vals, 0.0, None)
        Sigma_sqrt = (vecs * np.sqrt(vals)) @ vecs.T

    lows = np.array([a for a, _ in single_limits], dtype=float)
    highs = np.array([b for _, b in single_limits], dtype=float)

    def add_group_constraints(cons, w_var):
        for idx_tuple, (lo, hi) in multi_limits.items():
            idx = list(idx_tuple)
            if idx:
                cons += [cp.sum(w_var[idx]) >= float(lo), cp.sum(w_var[idx]) <= float(hi)]
        return cons

    # 端点：最大收益（LP/QP 的线性目标）
    w1 = cp.Variable(N)
    cons1 = [cp.sum(w1) == 1, w1 >= lows, w1 <= highs]
    cons1 = add_group_constraints(cons1, w1)
    prob1 = cp.Problem(cp.Maximize(mu @ w1), cons1)
    prob1.solve(solver=getattr(cp, solver, cp.ECOS), verbose=False)
    if w1.value is None:
        log("[求解器初始化] 无法求得最大收益点，放弃求解器初始化。")
        return np.empty((0, N)), np.array([]), np.array([])
    w_retmax = np.asarray(w1.value, dtype=float)

    # 端点：最小风险
    risk_metric = (risk_metric or "vol").lower()
    if risk_metric == "vol":
        w2 = cp.Variable(N)
        cons2 = [cp.sum(w2) == 1, w2 >= lows, w2 <= highs]
        cons2 = add_group_constraints(cons2, w2)
        prob2 = cp.Problem(cp.Minimize(cp.quad_form(w2, Sigma)), cons2)
        prob2.solve(solver=getattr(cp, solver, cp.ECOS), verbose=False)
        if w2.value is None:
            log("[求解器初始化] 无法求得最小方差点，放弃求解器初始化。")
            return np.empty((0, N)), np.array([]), np.array([])
        w_riskmin = np.asarray(w2.value, dtype=float)
    else:
        # VaR 参数
        vp = var_params or {}
        from statistics import NormalDist

        alpha = abs(NormalDist().inv_cdf(1.0 - float(vp.get("confidence", 0.95))))
        sqrt_h = float(vp.get("horizon_days", 1.0)) ** 0.5
        h = float(vp.get("horizon_days", 1.0))

        w2 = cp.Variable(N)
        s2 = cp.Variable(nonneg=True)
        cons2 = [cp.sum(w2) == 1, w2 >= lows, w2 <= highs,
                 cp.norm(Sigma_sqrt @ w2, 2) <= s2]
        cons2 = add_group_constraints(cons2, w2)
        obj2 = cp.Minimize(alpha * sqrt_h * s2 - h * (mu @ w2))
        prob2 = cp.Problem(obj2, cons2)
        prob2.solve(solver=getattr(cp, solver, cp.ECOS), verbose=False)
        if w2.value is None:
            log("[求解器初始化] 无法求得最小 VaR 点，放弃求解器初始化。")
            return np.empty((0, N)), np.array([]), np.array([])
        w_riskmin = np.asarray(w2.value, dtype=float)

    # 风险端点
    if risk_metric == "vol":
        def risk_of(wv: np.ndarray) -> float:
            return float(np.sqrt(wv @ Sigma @ wv))
    else:
        vp = var_params or {}
        from statistics import NormalDist

        alpha = abs(NormalDist().inv_cdf(1.0 - float(vp.get("confidence", 0.95))))
        sqrt_h = float(vp.get("horizon_days", 1.0)) ** 0.5
        h = float(vp.get("horizon_days", 1.0))

        def risk_of(wv: np.ndarray) -> float:
            sigma = float(np.sqrt(max(wv @ Sigma @ wv, 0.0)))
            muw = float(mu @ wv)
            return max(0.0, alpha * sqrt_h * sigma - h * muw)

    r_min = risk_of(w_riskmin)
    r_max = risk_of(w_retmax)
    if r_max < r_min:
        r_max = r_min

    # 风险栅格
    grid = np.linspace(r_min, r_max, int(n_grid))
    W = []

    for r_cap in grid:
        ww = cp.Variable(N)
        cons = [cp.sum(ww) == 1, ww >= lows, ww <= highs]
        cons = add_group_constraints(cons, ww)
        if risk_metric == "vol":
            cons += [cp.quad_form(ww, Sigma) <= (float(r_cap) ** 2)]
            obj = cp.Maximize(mu @ ww)
            prob = cp.Problem(obj, cons)
        else:
            vp = var_params or {}
            from statistics import NormalDist

            alpha = abs(NormalDist().inv_cdf(1.0 - float(vp.get("confidence", 0.95))))
            sqrt_h = float(vp.get("horizon_days", 1.0)) ** 0.5
            h = float(vp.get("horizon_days", 1.0))

            ss = cp.Variable(nonneg=True)
            cons += [cp.norm(Sigma_sqrt @ ww, 2) <= ss]
            cons += [alpha * sqrt_h * ss - h * (mu @ ww) <= float(r_cap)]
            obj = cp.Maximize(h * (mu @ ww))
            prob = cp.Problem(obj, cons)

        prob.solve(solver=getattr(cp, solver, cp.ECOS), verbose=False)
        if ww.value is not None:
            W.append(np.asarray(ww.value, dtype=float))

    if not W:
        return np.empty((0, N)), np.array([]), np.array([])

    W = np.vstack(W).astype(np.float64, copy=False)
    # 回算年化收益与风险（口径使用现有函数）
    ret_ann, vol_ann = compute_perf_arrays(
        port_daily_returns.astype(np.float32, copy=False),
        W.astype(np.float32, copy=False),
        trading_days=252.0, ddof=1,
    )
    if risk_metric == "vol":
        risk_arr = vol_ann
    else:
        vp = var_params or {}
        risk_arr = compute_var_parametric_arrays(
            port_daily_returns.astype(np.float32, copy=False),
            W.astype(np.float32, copy=False),
            confidence=float(vp.get("confidence", 0.95)),
            horizon_days=float(vp.get("horizon_days", 1.0)),
            return_type=str(vp.get("return_type", "simple")),
            ddof=int(vp.get("ddof", 1)),
            clip_non_negative=bool(vp.get("clip_non_negative", True)),
        )
    return W, ret_ann, risk_arr


# ===================== 3) 绘图 =====================

def plot_efficient_frontier_arrays(
        vol_all: np.ndarray,
        ret_all: np.ndarray,
        weights_all: np.ndarray,  # (M, N)
        ef_mask: np.ndarray,  # (M,)
        asset_names: List[str],
        title: str = "约束随机游走生成的投资组合与有效前沿",
        show: bool = True,
        x_label: str | None = None,
):
    # 自定义 hover：使用 customdata 避免 Python 循环拼接; customdata: 权重矩阵 (M, N)
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
        xaxis_title=(x_label or '年化波动率 (Annual Volatility)'),
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


# ===================== 权重量化（与 B02 一致语义） =====================

def _parse_precision(choice: str) -> float:
    """将 '0.1%'/'0.2%'/'0.5%' 或 '0.001' 解析为步长浮点数。"""
    choice = str(choice).strip()
    if choice.endswith('%'):
        val = float(choice[:-1]) / 100.0
    else:
        val = float(choice)
    return float(val)


def _snap_to_grid_simplex(w: np.ndarray, step: float, single_limits: List[Tuple[float, float]]) -> np.ndarray | None:
    """
    将权重向量吸附到网格（步长 step）上，同时满足：单资产上下限与 sum=1。
    采用最大余数法在整数格上分配，保持总和与边界。
    说明：不处理组约束；组约束由外层 POCS-量化循环修复。
    """
    R = int(round(1.0 / step))
    w = np.clip(w, 0.0, 1.0)
    k_float = w / step
    k_floor = np.floor(k_float).astype(np.int64)
    frac = k_float - k_floor
    lows = np.array([a for a, _ in single_limits], dtype=np.float64)
    highs = np.array([b for _, b in single_limits], dtype=np.float64)
    lo_units = np.ceil(lows / step - 1e-12).astype(np.int64)
    hi_units = np.floor(highs / step + 1e-12).astype(np.int64)
    k = np.clip(k_floor, lo_units, hi_units)
    diff = R - int(k.sum())
    if diff > 0:
        cap = hi_units - k
        idx = np.argsort(-frac)
        for i in idx:
            if diff == 0:
                break
            add = int(min(cap[i], diff))
            if add > 0:
                k[i] += add
                diff -= add
        if diff != 0:
            return None
    elif diff < 0:
        cap = k - lo_units
        idx = np.argsort(frac)
        for i in idx:
            if diff == 0:
                break
            sub = int(min(cap[i], -diff))
            if sub > 0:
                k[i] -= sub
                diff += sub
        if diff != 0:
            return None
    wq = k.astype(np.float64) / R
    if (wq < lows - 1e-12).any() or (wq > highs + 1e-12).any():
        return None
    if not np.isclose(wq.sum(), 1.0, atol=1e-12):
        return None
    return wq


def quantize_with_projection(
    w: np.ndarray,
    step: float,
    single_limits: List[Tuple[float, float]],
    multi_limits: Dict[Tuple[int, ...], Tuple[float, float]],
    *,
    projector=None,
    rounds: int = 5,
) -> np.ndarray | None:
    """
    迭代执行：
      1) 约束投影（盒+sum=1+组） 2) 网格吸附（sum=1+单资产）
    多轮循环后若收敛则返回量化后的可行权重；若失败返回 None。
    """
    if projector is None:
        projector = make_pocs_projector(
            single_limits=single_limits,
            multi_limits=multi_limits,
            max_iter=300,
            tol=1e-10,
            damping=0.9,
        )
    x = w.copy()
    for _ in range(rounds):
        x = projector(x)
        if x is None:
            return None
        xq = _snap_to_grid_simplex(x, step, single_limits)
        if xq is None:
            return None
        if np.max(np.abs(xq - x)) < step * 0.5:
            return xq
        x = xq
    return x


def _quantize_weights_batch_if_needed(
    W: np.ndarray,
    precision_choice: str | None,
    single_limits: List[Tuple[float, float]],
    multi_limits: Dict[Tuple[int, ...], Tuple[float, float]],
    *,
    rounds: int = 4,
    use_numba: bool | None = None,
    parallel_workers: int | None = None,
) -> np.ndarray:
    """按需对一批权重量化；失败的行丢弃。precision_choice=None 时原样返回。"""
    if not precision_choice:
        return W
    step = _parse_precision(precision_choice)
    if step <= 0 or step >= 1:
        return W
    if use_numba is None:
        use_numba = HAS_NUMBA
    projector = (
        make_pocs_projector_numba if (use_numba and HAS_NUMBA) else make_pocs_projector
    )(
        single_limits=single_limits,
        multi_limits=multi_limits,
        max_iter=300,
        tol=1e-10,
        damping=0.9,
    )

    def _one(w: np.ndarray) -> np.ndarray | None:
        return quantize_with_projection(
            w, step, single_limits, multi_limits, projector=projector, rounds=rounds
        )

    out: list[np.ndarray] = []
    M = W.shape[0]
    if parallel_workers is None:
        cpu = os.cpu_count() or 4
        # 小批量串行更快，批量大时并行
        parallel_workers = 0 if M < 64 else min(8, max(2, cpu // 2))
    if parallel_workers and M > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
            futures = [ex.submit(_one, w) for w in W]
            for fu in as_completed(futures):
                xq = fu.result()
                if xq is not None:
                    out.append(xq)
    else:
        for w in W:
            xq = _one(w)
            if xq is not None:
                out.append(xq)
    if not out:
        return np.empty((0, W.shape[1]), dtype=np.float64)
    return np.vstack(out).astype(np.float64, copy=False)


def generate_extreme_weight_seeds(
        N: int,
        single_limits: List[Tuple[float, float]],
        multi_limits: Dict[Tuple[int, ...], Tuple[float, float]],
        *,
        projector_iters: int = 200,
        projector_tol: float = 1e-9,
        projector_damping: float = 1.0,
        use_numba: bool | None = None,
        dedup_decimals: int = 8,
) -> np.ndarray:
    """
    生成“极端权重”可行种子：对每个资产 i，构造 e_i（仅第 i 个权重为 1），
    再投影到约束可行域，得到最多 N 个合法起点；若部分冲突导致重复/不可行，会自动去重/剔除。
    """
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

    seeds: List[np.ndarray] = []
    for i in range(N):
        v = np.zeros(N, dtype=np.float64)
        v[i] = 1.0
        x = projector(v)
        if x is not None:
            seeds.append(x)

    if not seeds:
        return np.empty((0, N), dtype=np.float64)

    S = np.vstack(seeds).astype(np.float64, copy=False)
    # 极端点之间可能因约束而投影为同一点，这里做一次去重
    S = deduplicate_weights(S, decimals=dedup_decimals)
    return S


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
        local_rng = np.random.default_rng(rng.integers(2 ** 63 - 1))
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
        risk_metric: str = "vol",  # "vol" 或 "var"
        var_params: Dict[str, Any] | None = None,
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
    # 可选：改用 VaR 作为风险维度
    if (risk_metric or "vol").lower() == "var":
        vp = var_params or {}
        risk_arr = compute_var_parametric_arrays(
            port_daily_returns,
            W,
            confidence=float(vp.get("confidence", 0.95)),
            horizon_days=float(vp.get("horizon_days", 1.0)),
            return_type=str(vp.get("return_type", "simple")),
            ddof=int(vp.get("ddof", 1)),
            clip_non_negative=bool(vp.get("clip_non_negative", True)),
        )
    else:
        risk_arr = vol
    finite_mask = np.isfinite(ret) & np.isfinite(vol)
    if not np.all(finite_mask):
        W, ret, vol = W[finite_mask], ret[finite_mask], vol[finite_mask]
    ef_mask = cal_ef_mask(ret, risk_arr)
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
        # 更新风险度量
        if (risk_metric or "vol").lower() == "var":
            vp = var_params or {}
            risk_arr = compute_var_parametric_arrays(
                port_daily_returns,
                W,
                confidence=float(vp.get("confidence", 0.95)),
                horizon_days=float(vp.get("horizon_days", 1.0)),
                return_type=str(vp.get("return_type", "simple")),
                ddof=int(vp.get("ddof", 1)),
                clip_non_negative=bool(vp.get("clip_non_negative", True)),
            )
        else:
            risk_arr = vol
        ef_mask = cal_ef_mask(ret, risk_arr)
        log(
            f"[第{r}轮] 有效前沿点数: {int(ef_mask.sum())} / {ef_mask.size}"
        )

        # 步长衰减（如需）
        step_mid *= float(step_decay)

    return W, ret, (risk_arr if (risk_metric or "vol").lower() == "var" else vol), ef_mask


def multi_level_random_walk_config(
        port_daily_returns: np.ndarray,
        single_limits: List[Tuple[float, float]],
        multi_limits: Dict[Tuple[int, ...], Tuple[float, float]],
        rounds_config: Dict[int, Dict[str, Any]],
        *,
        dedup_decimals: int = 4,
        annual_trading_days: float = 252.0,
        global_seed: int = 12345,
        extreme_seed_config: Dict[str, Any] | None = None,
        risk_metric: str = "vol",  # "vol" 或 "var"
        var_params: Dict[str, Any] | None = None,
        precision_choice: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    基于“字典配置”的多层随机游走：
    - rounds_config: { 轮次: { 参数... } }
      - 第0轮（初始化）：两种模式
        a) 随机探索（exploration，默认）：需包含 `samples`（总采样数）与 `step_size`；可选
           `projector_iters`/`projector_tol`/`projector_damping`/`seed`/`use_numba`。
        b) 求解器（solver）：设置 `init_mode="solver"`，并在 `solver_params` 中配置
           `n_grid`（风险栅格点数）、`solver`（ECOS/SCS/MOSEK）、`ridge` 等；
           风险度量由 `risk_metric`/`var_params` 控制（波动率或参数法 VaR）。
      - 第1..N轮（中步长）：包含 `step_size`，以下二选一：
        - `samples_per_seed`：每个种子生成的数量；
        - `samples_total`：本轮总体目标数量（会均分到各种子并向上取整）。
        同样支持 projector 参数与 `seed`（若未给，沿用 global_seed 生成器）。
      - 额外的 extreme_seed_config（可选，默认 None）：
        {"enable": bool, "samples_per_seed": int, 可选 "step_size" 及 projector/seed/并行参数}
    返回：最终 (W, ret, vol, ef_mask)。
    """
    if not rounds_config:
        raise ValueError("rounds_config 为空")

    # 预构建约束矩阵与上下界
    T, N = port_daily_returns.shape
    G, low_g, up_g = build_group_matrix(N, multi_limits)
    lows = np.array([a for a, _ in single_limits], dtype=np.float64)
    highs = np.array([b for _, b in single_limits], dtype=np.float64)

    # Round 0：初始化（两种模式：exploration/solver）
    r0_cfg = rounds_config.get(0, {})
    init_mode = str(r0_cfg.get("init_mode", "exploration")).lower()
    if init_mode == "solver":
        sp = dict(r0_cfg.get("solver_params", {}))
        n_grid = int(sp.get("n_grid", 150))
        ridge = float(sp.get("ridge", 1e-8))
        solver_name = str(sp.get("solver", "ECOS"))
        log(f"[第0轮] 求解器初始化前沿点: grid={n_grid}, risk={risk_metric}, solver={solver_name}")
        try:
            W, ret, risk_arr = build_frontier_by_risk_grid(
                port_daily_returns=port_daily_returns,
                single_limits=single_limits,
                multi_limits=multi_limits,
                risk_metric=risk_metric,
                var_params=var_params,
                n_grid=n_grid,
                ridge=ridge,
                solver=solver_name,
            )
        except Exception as e:
            log(f"[第0轮] 求解器初始化失败：{e}. 回退到随机探索。")
            init_mode = "exploration"

    if init_mode != "solver":
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
    # 可选：量化精度
    if W.size and precision_choice:
        W = _quantize_weights_batch_if_needed(W, precision_choice, single_limits, multi_limits)
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
    # 选择风险度量
    if (risk_metric or "vol").lower() == "var":
        vp = var_params or {}
        risk_arr = compute_var_parametric_arrays(
            port_daily_returns,
            W,
            confidence=float(vp.get("confidence", 0.95)),
            horizon_days=float(vp.get("horizon_days", 1.0)),
            return_type=str(vp.get("return_type", "simple")),
            ddof=int(vp.get("ddof", 1)),
            clip_non_negative=bool(vp.get("clip_non_negative", True)),
        )
        risk_name = f"VaR@{vp.get('confidence', 0.95)}({vp.get('return_type', 'simple')})"
    else:
        risk_arr = vol
        risk_name = "波动率"
    ef_mask = cal_ef_mask(ret, risk_arr)
    log(f"[第0轮] 有效前沿点数: {int(ef_mask.sum())} / {ef_mask.size}")

    # 额外：极端权重种子 -> 局部随机游走（可选）
    if extreme_seed_config and bool(extreme_seed_config.get("enable", False)):
        sps = int(extreme_seed_config.get("samples_per_seed", 0))
        if sps > 0:
            step_e = float(
                extreme_seed_config.get(
                    "step_size",
                    float(rounds_config.get(1, {}).get("step_size", 0.3)),
                )
            )
            p_iters_e = int(extreme_seed_config.get("projector_iters", 200))
            p_tol_e = float(extreme_seed_config.get("projector_tol", 1e-9))
            p_damping_e = float(extreme_seed_config.get("projector_damping", 1.0))
            r_seed_e = int(extreme_seed_config.get("seed", global_seed))
            parallel_e = int(extreme_seed_config.get("parallel_workers", 0)) or None
            use_numba_e = bool(extreme_seed_config.get("use_numba", False))

            log(
                f"[极端种子] 生成并投影极端权重种子，"
                f"每种子 {sps} 次，步长={step_e}"
            )
            extreme_seeds = generate_extreme_weight_seeds(
                N=N,
                single_limits=single_limits,
                multi_limits=multi_limits,
                projector_iters=p_iters_e,
                projector_tol=p_tol_e,
                projector_damping=p_damping_e,
                use_numba=use_numba_e,
                dedup_decimals=max(6, dedup_decimals),
            )
            if extreme_seeds.size:
                log(f"[极端种子] 可行种子数: {extreme_seeds.shape[0]}")
                # 从极端种子出发进行随机游走
                rng_e = np.random.default_rng(r_seed_e)
                per_seed_quota = np.full(extreme_seeds.shape[0], sps, dtype=int)
                ext_W = generate_weights_from_seeds(
                    seeds=extreme_seeds,
                    single_limits=single_limits,
                    multi_limits=multi_limits,
                    rng=rng_e,
                    per_seed_quota=per_seed_quota,
                    samples_per_seed=1,
                    step_size=step_e,
                    projector_iters=p_iters_e,
                    projector_tol=p_tol_e,
                    projector_damping=p_damping_e,
                    parallel_workers=parallel_e,
                    use_numba=use_numba_e,
                )
                if ext_W.size:
                    # 过滤与现有 W 重复的样本
                    if W.size:
                        seen_local: set[bytes] = set()
                        Wc_seen = np.ascontiguousarray(np.round(W, dedup_decimals))
                        for row in Wc_seen:
                            seen_local.add(row.tobytes())
                        Wc_new = np.ascontiguousarray(np.round(ext_W, dedup_decimals))
                        mask_new = np.fromiter((row.tobytes() not in seen_local for row in Wc_new),
                                               count=Wc_new.shape[0], dtype=bool)
                        ext_W = ext_W[mask_new]
                if ext_W.size and precision_choice:
                    ext_W = _quantize_weights_batch_if_needed(ext_W, precision_choice, single_limits, multi_limits)
                if ext_W.size:
                    # 去重、去已见、校验
                    ext_W = deduplicate_weights(ext_W, decimals=dedup_decimals)
                    # 将已见集合在后面统一构建后去重；先暂存，稍后统一处理
                    # 先直接按可行性过滤
                    vmask_e = validate_weights_batch(ext_W, lows, highs, G, low_g, up_g, atol=1e-6)
                    ext_W = ext_W[vmask_e]
                    if ext_W.size:
                        ret_e, vol_e = compute_perf_arrays(
                            port_daily_returns, ext_W, trading_days=annual_trading_days, ddof=1
                        )
                        finite_e = np.isfinite(ret_e) & np.isfinite(vol_e)
                        ext_W, ret_e, vol_e = ext_W[finite_e], ret_e[finite_e], vol_e[finite_e]
                        if ext_W.size:
                            W = np.vstack([W, ext_W])
                            ret = np.concatenate([ret, ret_e])
                            vol = np.concatenate([vol, vol_e])
                            # 更新风险度量与前沿
                            if (risk_metric or "vol").lower() == "var":
                                vp2 = var_params or {}
                                risk_arr = compute_var_parametric_arrays(
                                    port_daily_returns,
                                    W,
                                    confidence=float(vp2.get("confidence", 0.95)),
                                    horizon_days=float(vp2.get("horizon_days", 1.0)),
                                    return_type=str(vp2.get("return_type", "simple")),
                                    ddof=int(vp2.get("ddof", 1)),
                                    clip_non_negative=bool(vp2.get("clip_non_negative", True)),
                                )
                            else:
                                risk_arr = vol
                            ef_mask = cal_ef_mask(ret, risk_arr)
                            log(
                                f"[极端种子] 合并后有效权重数: {W.shape[0]}；"
                                f"有效前沿点数: {int(ef_mask.sum())}/{ef_mask.size}"
                            )
            else:
                log("[极端种子] 未生成任何可行极端种子，跳过。")

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

        if "samples_total" in cfg:
            total = int(cfg.get("samples_total", 200))
            # 直接复用上一轮已计算的风险度量，避免重复矩阵乘法
            seeds_vol = (risk_arr if (risk_metric or "vol").lower() == "var" else vol)[ef_mask]
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
                f"总量 {total}，按 vol 分桶分配（bins={int(cfg.get('vol_bins', 60))}, "
                f"min={per_seed_quota.min()}, median={int(np.median(per_seed_quota))}, "
                f"max={per_seed_quota.max()}, zeros={zeros})"
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
        # 可选：量化新权重
        if new_W.size and precision_choice:
            new_W = _quantize_weights_batch_if_needed(new_W, precision_choice, single_limits, multi_limits)
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
        # 更新风险度量
        if (risk_metric or "vol").lower() == "var":
            vp = var_params or {}
            risk_arr = compute_var_parametric_arrays(
                port_daily_returns,
                W,
                confidence=float(vp.get("confidence", 0.95)),
                horizon_days=float(vp.get("horizon_days", 1.0)),
                return_type=str(vp.get("return_type", "simple")),
                ddof=int(vp.get("ddof", 1)),
                clip_non_negative=bool(vp.get("clip_non_negative", True)),
            )
        else:
            risk_arr = vol
        ef_mask = cal_ef_mask(ret, risk_arr)
        log(f"[第{r}轮] 有效前沿点数: {int(ef_mask.sum())} / {ef_mask.size}")

    return W, ret, (risk_arr if (risk_metric or "vol").lower() == "var" else vol), ef_mask


def run_pipeline(
        excel_path: str,
        sheet_name: str,
        assets_list: List[str],
        single_limits: List[Tuple[float, float]],
        multi_limits: Dict[Tuple[int, ...], Tuple[float, float]],
        *,
        seed: int = 12345,
        rounds_config: Dict[int, Dict[str, Any]] | None = None,
        extreme_seed_config: Dict[str, Any] | None = None,
        risk_metric: str = "vol",
        var_params: Dict[str, Any] | None = None,
        precision_choice: str | None = None,
        # 公共参数
        annual_trading_days: float = 252.0,
        drop_duplicates_decimals: int = 4,
        show_plot: bool = True,
) -> None:
    """
    端到端执行：数据 -> 权重 -> 指标 -> 前沿 -> 绘图。
    仅支持字典式 rounds_config 配置多轮参数。
    """
    overall_t0 = time.time()

    # 1) 数据
    port_daily_returns, assets_list = load_returns_from_excel(
        excel_path, sheet_name, assets_list
    )
    T, N = port_daily_returns.shape
    log(f"数据准备完成：T={T}, N={N}")

    if rounds_config is None:
        raise ValueError("rounds_config 不能为空（仅支持字典配置模式）")

    W, ret_annual, risk_arr, ef_mask = multi_level_random_walk_config(
        port_daily_returns=port_daily_returns,
        single_limits=single_limits,
        multi_limits=multi_limits,
        rounds_config=rounds_config,
        dedup_decimals=drop_duplicates_decimals,
        annual_trading_days=annual_trading_days,
        global_seed=seed,
        extreme_seed_config=extreme_seed_config,
        risk_metric=risk_metric,
        var_params=var_params,
        precision_choice=precision_choice,
    )
    if W.size == 0:
        log("多层（字典配置）流程未产出结果，终止。")
        return

    # 7) 绘图
    log("生成交互式图表...")
    xlabel = '年化波动率 (Annual Volatility)'
    if (risk_metric or "vol").lower() == "var":
        vp = var_params or {}
        conf = float(vp.get("confidence", 0.95))
        h = float(vp.get("horizon_days", 1.0))
        rtype = str(vp.get("return_type", "simple"))
        xlabel = f"VaR@{conf:.2f}({rtype}, 持有期{h:g}天)"
    plot_efficient_frontier_arrays(
        vol_all=risk_arr,
        ret_all=ret_annual,
        weights_all=W,
        ef_mask=ef_mask,
        asset_names=assets_list,
        title="约束随机游走生成的投资组合与有效前沿",
        show=show_plot,
        x_label=xlabel,
    )

    log(f"流程完成，总耗时 {time.time() - overall_t0:.2f}s")


# ===================== 4) 主流程 =====================

if __name__ == "__main__":
    # 主要参数（可按需调整）
    EXCEL_PATH = "历史净值数据_万得指数.xlsx"
    SHEET_NAME = "历史净值数据"
    ASSETS = ["货币现金类", "固定收益类", "混合策略类", "权益投资类", "另类投资类"]

    # 单资产上下限（示例：全部 [0, 1]）
    SINGLE_LIMITS: List[Tuple[float, float]] = [(0.0, 1.0)] * len(ASSETS)
    # 多资产联合约束（示例为空；可设如 {(0,1):(0.2,0.6)}）
    MULTI_LIMITS: Dict[Tuple[int, ...], Tuple[float, float]] = {}

    # 随机游走与指标参数（仅字典方式）
    RANDOM_SEED = 12345
    # 字典式多轮配置（参数含义见 multi_level_random_walk_config 注释）
    ROUNDS_CONFIG: Dict[int, Dict[str, Any]] = {
        # 第0轮：初始化方式二选一：
        0: {
            "init_mode": "exploration",  # "exploration" 随机探索 或 "solver" 求解器
            # exploration 参数（当 init_mode=="exploration" 生效）：
            "samples": 300,
            "step_size": 0.99,
            # solver 参数（当 init_mode=="solver" 生效）：
            "solver_params": {
                "n_grid": 300,
                "solver": "ECOS",  # ECOS/SCS/MOSEK
                "ridge": 1e-8,
            },
        },
        1: {"samples_total": 1000, "step_size": 0.1, "vol_bins": 100, "parallel_workers": 100},
        2: {"samples_total": 2000, "step_size": 0.8, "vol_bins": 200, "parallel_workers": 100},
        3: {"samples_total": 3000, "step_size": 0.05, "vol_bins": 300, "parallel_workers": 100},
        4: {"samples_total": 4000, "step_size": 0.03, "vol_bins": 400, "parallel_workers": 100},
        5: {"samples_total": 5000, "step_size": 0.01, "vol_bins": 500, "parallel_workers": 100},
    }
    TRADING_DAYS = 252.0  # 年化换算用交易天数
    DEDUP_DECIMALS = 2  # 在“权重去重”时对每行权重进行四舍五入保留的小数位数

    # 是否显示图表（自测/批处理可为 False）
    SHOW_PLOT = True

    # 是否启用“极端权重”种子，以及每个种子生成的数量
    EXTREME_SEED_CONFIG: Dict[str, Any] = {
        "enable": False,
        # 每个极端种子（例如 5 个大类 -> 5 个种子）生成多少权重
        "samples_per_seed": 100,
        # 可选步长：未指定时默认采用第1轮 step_size（否则 0.3）
        # "step_size": 0.3,
        # 其他可选项：projector_iters/projector_tol/projector_damping/seed/parallel_workers/use_numba
    }

    # 风险度量与 VaR 参数
    RISK_METRIC = "var"  # 可选："vol"（波动率）或 "var"（参数法 VaR）
    VAR_PARAMS: Dict[str, Any] = {
        "confidence": 0.95,
        "horizon_days": 1.0,
        "return_type": "log",  # 或 "simple"
        "ddof": 1,
        "clip_non_negative": True,  # 对“无下跌”情形，VaR 取 0
    }

    # 权重精度（量化）选择：'0.1%'、'0.2%'、'0.5%' 或 None（不量化）
    PRECISION_CHOICE: str | None = None

    log("程序开始运行")
    run_pipeline(
        excel_path=EXCEL_PATH,
        sheet_name=SHEET_NAME,
        assets_list=ASSETS,
        single_limits=SINGLE_LIMITS,
        multi_limits=MULTI_LIMITS,
        seed=RANDOM_SEED,
        rounds_config=ROUNDS_CONFIG,
        extreme_seed_config=EXTREME_SEED_CONFIG,
        risk_metric=RISK_METRIC,
        var_params=VAR_PARAMS,
        precision_choice=PRECISION_CHOICE,
        # 公共参数
        annual_trading_days=TRADING_DAYS,
        drop_duplicates_decimals=DEDUP_DECIMALS,
        show_plot=SHOW_PLOT,
    )
    log("程序结束")
