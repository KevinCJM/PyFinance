"""
C01_efficient_frontier_solver.py

使用凸优化求解器（QCQP/SOCP）构造在给定线性权重约束下的有效前沿，并绘制展示。

特性
- 风险度量可选：
  - 波动率（Variance/Volatility）：标准马科维茨 QCQP（max μᵀw s.t. wᵀΣw ≤ r²）
  - 参数法 VaR（正态近似）：SOCP 形式（‖Σ^{1/2}w‖₂ ≤ s，且 α√h·s − h·μᵀw ≤ r）
- 支持单资产上下限（SINGLE_LIMITS）、分组上下限（MULTI_LIMITS）、权重和=1。
- 读取 Excel 净值数据，生成日收益矩阵，优化阶段采用简单收益的 μ、Σ；绘图回算使用仓库里统一的 log1p 年化口径或 VaR 计算，保证展示一致。

依赖：cvxpy、numpy、pandas、plotly
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize


# ===================== 工具与通用函数 =====================

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}")


def load_returns_from_excel(
        excel_path: str, sheet_name: str, assets_list: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """从 Excel 读取净值数据，生成日收益二维数组 (T,N)。"""
    log(f"加载数据: {excel_path} | sheet={sheet_name}")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.dropna().sort_index(ascending=True)

    # 列重命名对齐仓库全局口径（可按需拓展）
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

    missing = [c for c in assets_list if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列: {missing}")

    hist_ret_df = df[assets_list].pct_change().dropna()
    arr = hist_ret_df.values.astype(np.float32, copy=False)
    log(f"数据加载完成，样本天数={arr.shape[0]}，资产数={arr.shape[1]}")
    return arr, assets_list


def compute_perf_arrays(
        port_daily: np.ndarray,  # (T, N)
        portfolio_allocs: np.ndarray,  # (M, N)
        trading_days: float = 252.0,
        ddof: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    与仓库口径一致：对组合日收益 R_t = Σ w_i r_{t,i}，使用 log1p 年化。
    返回 (ret_annual, vol_annual)。
    """
    if port_daily.dtype != np.float32:
        port_daily = port_daily.astype(np.float32, copy=False)
    if portfolio_allocs.dtype != np.float32:
        portfolio_allocs = portfolio_allocs.astype(np.float32, copy=False)

    T = port_daily.shape[0]
    WT = np.ascontiguousarray(portfolio_allocs.T)
    R = port_daily @ WT  # (T, M)
    np.log1p(R, out=R)
    ret_annual = (R.sum(axis=0, dtype=np.float32) / float(T)) * float(trading_days)
    vol_annual = R.std(axis=0, ddof=ddof) * np.sqrt(float(trading_days))
    return ret_annual, vol_annual


def compute_var_parametric_arrays(
        port_daily: np.ndarray,
        portfolio_allocs: np.ndarray,
        *,
        confidence: float = 0.95,
        horizon_days: float = 1.0,
        return_type: str = "simple",
        ddof: int = 1,
        clip_non_negative: bool = True,
) -> np.ndarray:
    """参数法 VaR（正态近似），返回 VaR 数组（非负）。"""
    confidence = float(confidence)
    confidence = min(max(confidence, 1e-6), 1 - 1e-6)
    horizon_days = max(float(horizon_days), 1e-12)

    if port_daily.dtype != np.float32:
        port_daily = port_daily.astype(np.float32, copy=False)
    if portfolio_allocs.dtype != np.float32:
        portfolio_allocs = portfolio_allocs.astype(np.float32, copy=False)

    WT = np.ascontiguousarray(portfolio_allocs.T)
    R = port_daily @ WT
    X = np.log1p(R) if return_type == "log" else R
    mu = X.mean(axis=0, dtype=np.float32)
    sigma = X.std(axis=0, ddof=ddof)

    h = float(horizon_days)
    mu_h = mu * h
    sigma_h = sigma * np.sqrt(h)

    try:
        from statistics import NormalDist

        z = NormalDist().inv_cdf(1.0 - confidence)
    except Exception:
        z = -1.6448536269514722 if abs(confidence - 0.95) < 1e-6 else -2.3263478740408408

    var_val = -(mu_h + z * sigma_h)
    if clip_non_negative:
        var_val = np.maximum(var_val, 0.0)
    return np.abs(var_val).astype(np.float32, copy=False)


def cal_ef_mask(ret_annual: np.ndarray, risk_array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """对收益降序排序，保留风险的前缀最小点，返回布尔掩码。"""
    idx = np.argsort(ret_annual)[::-1]
    risk_sorted = risk_array[idx]
    cummin_risk = np.minimum.accumulate(risk_sorted)
    on_ef_sorted = risk_sorted <= (cummin_risk + eps)
    on_ef = np.zeros(ret_annual.shape[0], dtype=bool)
    on_ef[idx] = on_ef_sorted
    return on_ef


def plot_frontier(
        risk_arr: np.ndarray,
        ret_arr: np.ndarray,
        weights: np.ndarray,
        asset_names: List[str],
        *,
        title: str,
        x_label: str,
        show: bool = True,
        save_html: str | None = None,
        frontier_only: bool = True,
):
    hover_assets = "<br>".join(
        [f"{name}: %{{customdata[{i}]:.1%}}" for i, name in enumerate(asset_names)]
    )
    hovertemplate = (
            "年化风险: %{x:.2%}<br>" "年化收益率: %{y:.2%}<br><br>" "<b>资产权重</b><br>" + hover_assets + "<extra></extra>"
    )

    ef_mask = cal_ef_mask(ret_arr, risk_arr)
    fig = go.Figure()
    if not frontier_only:
        fig.add_trace(
            go.Scattergl(
                x=risk_arr[~ef_mask],
                y=ret_arr[~ef_mask],
                mode="markers",
                name="非前沿点",
                marker=dict(color="grey", size=3, opacity=0.4),
                customdata=weights[~ef_mask],
                hovertemplate=hovertemplate,
            )
        )
    fig.add_trace(
        go.Scattergl(
            x=risk_arr[ef_mask],
            y=ret_arr[ef_mask],
            mode="markers+lines",
            name="有效前沿",
            marker=dict(color="blue", size=5, opacity=0.9),
            line=dict(color="blue"),
            customdata=weights[ef_mask],
            hovertemplate=hovertemplate,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="年化收益率 (Annual Return)",
        legend_title="图例",
        hovermode="closest",
    )
    if save_html:
        fig.write_html(save_html, include_plotlyjs="cdn")
        log(f"图表已保存: {save_html}")
    if show:
        fig.show()


# ===================== 求解器前沿构建（QCQP/SOCP） =====================

def _mu_cov_from_daily(port_daily: np.ndarray, ridge: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    r = port_daily.astype(np.float64, copy=False)
    mu = r.mean(axis=0)
    Sigma = np.cov(r, rowvar=False, ddof=1)
    Sigma = ((Sigma + Sigma.T) * 0.5) + float(ridge) * np.eye(Sigma.shape[0])
    return mu, Sigma


def _sigma_sqrt_psd(Sigma: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(Sigma)
        vals = np.clip(vals, 0.0, None)
        return (vecs * np.sqrt(vals)) @ vecs.T


def build_frontier_by_solver(
        port_daily_returns: np.ndarray,
        single_limits: List[Tuple[float, float]],
        multi_limits: Dict[Tuple[int, ...], Tuple[float, float]],
        *,
        risk_metric: str = "vol",  # "vol" 或 "var"
        var_params: Dict[str, Any] | None = None,
        n_interior: int = 100,
        ridge: float = 1e-8,
        solver: str = "ECOS",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    - vol: QCQP（max μᵀw s.t. wᵀΣw ≤ r²）
    - var: SOCP（‖Σ^{1/2}w‖₂ ≤ s；α√h·s − h·μᵀw ≤ r）
    返回 (W, ret_annual, risk_array)。
    """
    try:
        import cvxpy as cp
    except Exception as e:
        raise RuntimeError("需要安装 cvxpy: pip install cvxpy") from e

    mu, Sigma = _mu_cov_from_daily(port_daily_returns, ridge=ridge)
    N = Sigma.shape[0]
    lows = np.array([a for a, _ in single_limits], dtype=float)
    highs = np.array([b for _, b in single_limits], dtype=float)

    def add_group_constraints(cons, w_var):
        for idx_tuple, (lo, hi) in multi_limits.items():
            idx = list(idx_tuple)
            if idx:
                cons += [w_var[idx].sum() >= float(lo), w_var[idx].sum() <= float(hi)]
        return cons

    # 端点：最大收益（LP）
    w1 = cp.Variable(N)
    cons1 = [cp.sum(w1) == 1, w1 >= lows, w1 <= highs]
    cons1 = add_group_constraints(cons1, w1)
    prob1 = cp.Problem(cp.Maximize(mu @ w1), cons1)
    prob1.solve(solver=getattr(cp, solver, cp.ECOS), verbose=False)
    if w1.value is None:
        raise RuntimeError("求解最大收益组合失败（可能约束不可行）。")
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
            raise RuntimeError("求解最小方差组合失败（可能约束不可行）。")
        w_riskmin = np.asarray(w2.value, dtype=float)
    else:
        from statistics import NormalDist

        vp = var_params or {}
        alpha = abs(NormalDist().inv_cdf(1.0 - float(vp.get("confidence", 0.95))))
        sqrt_h = float(vp.get("horizon_days", 1.0)) ** 0.5
        h = float(vp.get("horizon_days", 1.0))

        Sigma_sqrt = _sigma_sqrt_psd(Sigma)
        w2 = cp.Variable(N)
        s2 = cp.Variable(nonneg=True)
        cons2 = [cp.sum(w2) == 1, w2 >= lows, w2 <= highs, cp.norm(Sigma_sqrt @ w2, 2) <= s2]
        cons2 = add_group_constraints(cons2, w2)
        obj2 = cp.Minimize(alpha * sqrt_h * s2 - h * (mu @ w2))
        prob2 = cp.Problem(obj2, cons2)
        prob2.solve(solver=getattr(cp, solver, cp.ECOS), verbose=False)
        if w2.value is None:
            raise RuntimeError("求解最小 VaR 组合失败（可能约束不可行）。")
        w_riskmin = np.asarray(w2.value, dtype=float)

    # 端点风险值
    if risk_metric == "vol":
        def risk_of(wv: np.ndarray) -> float:
            return float(np.sqrt(max(wv @ Sigma @ wv, 0.0)))
    else:
        from statistics import NormalDist

        vp = var_params or {}
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
    # 在端点之间切 n_interior 份，总点数 = n_interior + 2
    grid = np.linspace(r_min, r_max, int(n_interior) + 2)

    # 每个风险上限下最大化收益，得到前沿
    W = []
    if risk_metric == "vol":
        for r_cap in grid:
            ww = cp.Variable(N)
            cons = [cp.sum(ww) == 1, ww >= lows, ww <= highs]
            cons = add_group_constraints(cons, ww)
            cons += [cp.quad_form(ww, Sigma) <= float(r_cap) ** 2]
            prob = cp.Problem(cp.Maximize(mu @ ww), cons)
            prob.solve(solver=getattr(cp, solver, cp.ECOS), verbose=False)
            if ww.value is not None:
                W.append(np.asarray(ww.value, dtype=float))
    else:
        from statistics import NormalDist

        vp = var_params or {}
        alpha = abs(NormalDist().inv_cdf(1.0 - float(vp.get("confidence", 0.95))))
        sqrt_h = float(vp.get("horizon_days", 1.0)) ** 0.5
        h = float(vp.get("horizon_days", 1.0))
        Sigma_sqrt = _sigma_sqrt_psd(Sigma)

        for r_cap in grid:
            ww = cp.Variable(N)
            ss = cp.Variable(nonneg=True)
            cons = [cp.sum(ww) == 1, ww >= lows, ww <= highs]
            cons = add_group_constraints(cons, ww)
            cons += [cp.norm(Sigma_sqrt @ ww, 2) <= ss]
            cons += [alpha * sqrt_h * ss - h * (mu @ ww) <= float(r_cap)]
            prob = cp.Problem(cp.Maximize(h * (mu @ ww)), cons)
            prob.solve(solver=getattr(cp, solver, cp.ECOS), verbose=False)
            if ww.value is not None:
                W.append(np.asarray(ww.value, dtype=float))

    if not W:
        return np.empty((0, N)), np.array([]), np.array([])

    W = np.vstack(W).astype(np.float64, copy=False)
    # 展示口径：严格按用户选择的收益与风险指标
    ret_disp, vol_disp = compute_perf_arrays(
        port_daily_returns.astype(np.float32, copy=False),
        W.astype(np.float32, copy=False),
        trading_days=252.0,
        ddof=1,
    )
    if risk_metric == "vol":
        risk_disp = vol_disp
    else:
        vp = var_params or {}
        risk_disp = compute_var_parametric_arrays(
            port_daily_returns.astype(np.float32, copy=False),
            W.astype(np.float32, copy=False),
            confidence=float(vp.get("confidence", 0.95)),
            horizon_days=float(vp.get("horizon_days", 1.0)),
            return_type=str(vp.get("return_type", "simple")),
            ddof=int(vp.get("ddof", 1)),
            clip_non_negative=bool(vp.get("clip_non_negative", True)),
        )
    return W, ret_disp.astype(np.float64, copy=False), risk_disp.astype(np.float64, copy=False)


# ===================== SLSQP 前沿构建（按展示口径直接优化） =====================

def build_frontier_by_slsqp(
        port_daily_returns: np.ndarray,
        single_limits: List[Tuple[float, float]],
        multi_limits: Dict[Tuple[int, ...], Tuple[float, float]],
        *,
        risk_metric: str = "vol",  # "vol" 或 "var"
        var_params: Dict[str, Any] | None = None,
        n_interior: int = 100,
        maxiter: int = 400,
        tol: float = 1e-8,
        verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 SLSQP 直接在“展示口径”的收益/风险下优化并构造前沿：
    - 先通过简单启发式分别求最小风险点与最大收益点，确定 [Rmin, Rmax]
    - 在其间等分 n_interior 份，每个风险上限 r_cap 下最大化展示口径收益
    说明：SLSQP 是局部优化器，建议使用相邻网格的解作 warm-start，以提高一致性。
    """
    T, N = port_daily_returns.shape
    lows = np.array([a for a, _ in single_limits], dtype=float)
    highs = np.array([b for _, b in single_limits], dtype=float)
    bounds = list(zip(lows.tolist(), highs.tolist()))
    risk_metric = (risk_metric or "vol").lower()
    vp = var_params or {}

    def ret_disp_of(w: np.ndarray) -> float:
        W = w.reshape(1, -1).astype(np.float32)
        ret, _ = compute_perf_arrays(port_daily_returns, W, trading_days=252.0, ddof=1)
        return float(ret[0])

    def risk_disp_of(w: np.ndarray) -> float:
        W = w.reshape(1, -1).astype(np.float32)
        if risk_metric == "vol":
            _, vol = compute_perf_arrays(port_daily_returns, W, trading_days=252.0, ddof=1)
            return float(vol[0])
        else:
            arr = compute_var_parametric_arrays(
                port_daily_returns,
                W,
                confidence=float(vp.get("confidence", 0.95)),
                horizon_days=float(vp.get("horizon_days", 1.0)),
                return_type=str(vp.get("return_type", "simple")),
                ddof=int(vp.get("ddof", 1)),
                clip_non_negative=bool(vp.get("clip_non_negative", True)),
            )
            return float(arr[0])

    # 组约束（不等式形式）：for each group sum(w[idx]) in [lo, hi]
    group_ineqs = []
    for idx_tuple, (lo, hi) in multi_limits.items():
        idx = np.array(idx_tuple, dtype=int)
        if idx.size == 0:
            continue

        def make_lo(idx=idx, lo=lo):
            return dict(type='ineq', fun=lambda w, idx=idx, lo=lo: float(np.sum(w[idx]) - float(lo)))

        def make_hi(idx=idx, hi=hi):
            return dict(type='ineq', fun=lambda w, idx=idx, hi=hi: float(float(hi) - np.sum(w[idx])))

        group_ineqs.append(make_lo())
        group_ineqs.append(make_hi())

    # 等式约束：sum(w)=1
    eq_sum = dict(type='eq', fun=lambda w: float(np.sum(w) - 1.0))

    # 启动点：均匀权重并裁剪到边界
    w0 = np.clip(np.full(N, 1.0 / N, dtype=float), lows, highs)
    # 若 sum!=1，用等量平移到满足 sum=1（在 box 内部通常可行）
    w0 += (1.0 - w0.sum()) / N
    w0 = np.clip(w0, lows, highs)

    # 端点1：最大收益（仅线性目标+线性约束）
    res_ret = minimize(
        lambda w: -ret_disp_of(w),
        x0=w0,
        method='SLSQP',
        bounds=bounds,
        constraints=[eq_sum] + group_ineqs,
        options=dict(maxiter=maxiter, ftol=tol, disp=verbose),
    )
    w_retmax = (res_ret.x if res_ret.success else w0).astype(float)
    rmax = risk_disp_of(w_retmax)

    # 端点2：最小风险（仅风险目标）
    res_risk = minimize(
        lambda w: risk_disp_of(w),
        x0=w0,
        method='SLSQP',
        bounds=bounds,
        constraints=[eq_sum] + group_ineqs,
        options=dict(maxiter=maxiter, ftol=tol, disp=verbose),
    )
    w_riskmin = (res_risk.x if res_risk.success else w0).astype(float)
    rmin = risk_disp_of(w_riskmin)
    if rmax < rmin:
        rmax = rmin

    # 风险网格
    grid = np.linspace(rmin, rmax, int(n_interior) + 2)
    r_span = float(rmax - rmin)
    slack = max(1e-9, 1e-6 * r_span)

    W_list: List[np.ndarray] = []
    cur = w_riskmin.copy()
    for k, r_cap in enumerate(grid):
        # 风险约束：risk(w) <= r_cap
        cons = [eq_sum] + group_ineqs + [
            dict(type='ineq', fun=lambda w, r=r_cap, s=slack: float((r + s) - risk_disp_of(w)))
        ]
        # 分段线性插值的 warm-start，有助于避免陷在局部最优
        t = 0.0 if r_span <= 0 else float((r_cap - rmin) / r_span)
        x0 = ((1.0 - t) * w_riskmin + t * w_retmax).astype(float)
        res = minimize(
            lambda w: -ret_disp_of(w),
            x0=x0 if k > 0 else cur,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options=dict(maxiter=maxiter, ftol=tol, disp=verbose),
        )
        if res.success:
            cur = res.x.astype(float)
        # 即便失败，保留当前解（可为上一解），保持前沿连贯
        W_list.append(cur.copy())

    W = np.vstack(W_list).astype(np.float64, copy=False)
    ret_disp, vol_disp = compute_perf_arrays(
        port_daily_returns.astype(np.float32, copy=False),
        W.astype(np.float32, copy=False),
        trading_days=252.0,
        ddof=1,
    )
    if risk_metric == 'vol':
        risk_disp = vol_disp
    else:
        risk_disp = compute_var_parametric_arrays(
            port_daily_returns.astype(np.float32, copy=False),
            W.astype(np.float32, copy=False),
            confidence=float(vp.get("confidence", 0.95)),
            horizon_days=float(vp.get("horizon_days", 1.0)),
            return_type=str(vp.get("return_type", "simple")),
            ddof=int(vp.get("ddof", 1)),
            clip_non_negative=bool(vp.get("clip_non_negative", True)),
        )
    return W, ret_disp.astype(np.float64, copy=False), risk_disp.astype(np.float64, copy=False)


# ===================== 主流程 =====================

if __name__ == "__main__":
    # 数据与资产
    EXCEL_PATH = "历史净值数据_万得指数.xlsx"
    SHEET_NAME = "历史净值数据"
    ASSETS = ["货币现金类", "固定收益类", "混合策略类", "权益投资类", "另类投资类"]

    # 约束
    SINGLE_LIMITS: List[Tuple[float, float]] = [(0.0, 1.0)] * len(ASSETS)
    MULTI_LIMITS: Dict[Tuple[int, ...], Tuple[float, float]] = {}

    # 风险与求解器配置
    RISK_METRIC = "var"  # "vol" 或 "var"
    VAR_PARAMS: Dict[str, Any] = {
        "confidence": 0.95,
        "horizon_days": 1.0,
        "return_type": "log",  # simple 或 "log"
        "ddof": 1,
        "clip_non_negative": True,
    }
    SOLVER_PARAMS: Dict[str, Any] = {
        "n_interior": 1000,
        "solver": "ECOS",  # ECOS/SCS/MOSEK
        "ridge": 1e-8,
    }

    SHOW_PLOT = True
    SAVE_HTML = None  # 可设为 None 不落盘
    USE_SLSQP = True  # 设为 True 切换为 SLSQP 直接按展示口径优化

    t0 = time.time()
    log("开始：读取数据")
    port_daily, assets = load_returns_from_excel(EXCEL_PATH, SHEET_NAME, ASSETS)
    log("开始：构造前沿 (QCQP/SOCP 或 SLSQP)")
    if not USE_SLSQP:
        W, ret_arr, risk_arr = build_frontier_by_solver(
            port_daily_returns=port_daily,
            single_limits=SINGLE_LIMITS,
            multi_limits=MULTI_LIMITS,
            risk_metric=RISK_METRIC,
            var_params=VAR_PARAMS,
            n_interior=int(SOLVER_PARAMS.get("n_interior", 1000)),
            ridge=float(SOLVER_PARAMS.get("ridge", 1e-8)),
            solver=str(SOLVER_PARAMS.get("solver", "ECOS")),
        )
    else:
        W, ret_arr, risk_arr = build_frontier_by_slsqp(
            port_daily_returns=port_daily,
            single_limits=SINGLE_LIMITS,
            multi_limits=MULTI_LIMITS,
            risk_metric=RISK_METRIC,
            var_params=VAR_PARAMS,
            n_interior=int(SOLVER_PARAMS.get("n_interior", 1000)),
            maxiter=5000,
            tol=1e-8,
            verbose=False,
        )
    if W.size == 0:
        log("未能构造任何前沿点，请检查约束或参数。")
    else:
        xlabel = "年化波动率 (Annual Volatility)"
        if (RISK_METRIC or "vol").lower() == "var":
            c = VAR_PARAMS.get("confidence", 0.95)
            h = VAR_PARAMS.get("horizon_days", 1.0)
            rt = VAR_PARAMS.get("return_type", "simple")
            xlabel = f"VaR@{float(c):.2f}({rt}, 持有期{float(h):g}天)"
        log("绘制图表...")
        plot_frontier(
            risk_arr=risk_arr,
            ret_arr=ret_arr,
            weights=W,
            asset_names=assets,
            title="求解器构造的有效前沿",
            x_label=xlabel,
            show=SHOW_PLOT,
            save_html=SAVE_HTML,
            frontier_only=True,
        )
        log(f"完成。总耗时 {time.time() - t0:.2f}s")
