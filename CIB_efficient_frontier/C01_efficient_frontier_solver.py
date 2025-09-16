# -*- coding: utf-8 -*-
"""
SLSQP 刻画 (Annualized Log Return, VaR_log) 有效前沿
- 优化目标使用平滑 VaR_raw = -(mu_h + z*sigma_h)（不abs、不clip）
- 用 SLSQP 先解可达收益区间 [ret_min, ret_max]，再扫描目标收益
- 不允许空卖：sum(w)=1, 0<=w<=1（可在 BOUNDS 修改上限）
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pathlib import Path

# ===================== 0) 参数 =====================
EXCEL_PATH = "历史净值数据_万得指数.xlsx"
SHEET_NAME = "历史净值数据"
ASSETS = ["货币现金类", "固定收益类", "混合策略类", "权益投资类", "另类投资类"]

N_TARGETS = 60
CONFIDENCE = 0.95
HORIZON_DAYS = 1.0
TRADING_DAYS = 252.0
DDOF = 1
MAXITER = 800
FTOL = 1e-12
USE_WARM_START = True
BOUNDS = [(0.0, 1.0)] * len(ASSETS)  # 无空卖；可改为如 (0,0.4)

SAVE_CSV_PATH = "frontier_by_var.csv"
SAVE_FIG_PATH = "frontier_return_var.png"


# ===================== 1) 基础工具 =====================
def log(msg: str) -> None:
    print(msg, flush=True)


def load_returns_from_excel(
        excel_path: str, sheet_name: str, assets_list: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """从 Excel 读取净值数据，生成日简单收益二维数组 (T,N)。"""
    log(f"加载数据: {excel_path} | sheet={sheet_name}")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.dropna().sort_index(ascending=True)

    # 列重命名口径统一（按需拓展）
    df = df.rename(
        {
            "货基指数": "货币现金类",
            "固收类": "固定收益类",
            "混合类": "混合策略类",
            "权益类": "权益投资类",
            "另类": "另类投资类",
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
        port_daily: np.ndarray,  # (T, N) —— 简单日收益
        portfolio_allocs: np.ndarray,  # (M, N)
        trading_days: float = 252.0,
        ddof: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    与口径一致：组合日收益 R_t = Σ w_i r_{t,i}，使用 log1p 年化。
    返回 (ret_annual, vol_annual)。
    """
    if port_daily.dtype != np.float32:
        port_daily = port_daily.astype(np.float32, copy=False)
    if portfolio_allocs.dtype != np.float32:
        portfolio_allocs = portfolio_allocs.astype(np.float32, copy=False)

    T = port_daily.shape[0]
    WT = np.ascontiguousarray(portfolio_allocs.T)  # (N,M)
    R = port_daily @ WT  # (T,M)
    np.log1p(R, out=R)  # 就地转换为对数收益
    ret_annual = (R.sum(axis=0, dtype=np.float32) / float(T)) * float(trading_days)
    vol_annual = R.std(axis=0, ddof=ddof) * np.sqrt(float(trading_days))
    return ret_annual, vol_annual


def compute_var_parametric_arrays(
        port_daily: np.ndarray,
        portfolio_allocs: np.ndarray,
        *,
        confidence: float = 0.95,
        horizon_days: float = 1.0,
        return_type: str = "log",
        ddof: int = 1,
        clip_non_negative: bool = True,
) -> np.ndarray:
    """参数法 VaR（正态近似），返回 VaR 数组（非负）。用于结果展示。"""
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


# ===================== 2) 优化用“平滑 VaR 目标” =====================
def _var_raw_for_opt(
        port_daily: np.ndarray,
        w: np.ndarray,
        *,
        confidence: float = 0.95,
        horizon_days: float = 1.0,
        return_type: str = "log",
        ddof: int = 1,
) -> float:
    """
    平滑 VaR 目标（不abs、不clip）： -(mu_h + z*sigma_h)
    仅用于优化，展示时再 clip>=0。
    """
    w = np.asarray(w, dtype=np.float32, order='C')
    R = port_daily @ w.astype(np.float32, copy=False)  # (T,)
    X = np.log1p(R) if return_type == "log" else R
    mu = float(X.mean(dtype=np.float32))
    sigma = float(X.std(ddof=ddof))

    h = float(horizon_days)
    mu_h = mu * h
    sigma_h = sigma * np.sqrt(h)

    try:
        from statistics import NormalDist
        z = NormalDist().inv_cdf(1.0 - confidence)
    except Exception:
        z = -1.6448536269514722 if abs(confidence - 0.95) < 1e-6 else -2.3263478740408408

    return -(mu_h + z * sigma_h)


def _perf_of_w(port_daily: np.ndarray,
               w: np.ndarray,
               trading_days: float = TRADING_DAYS,
               ddof: int = DDOF) -> Tuple[float, float]:
    """标量版 (ret_annual, vol_annual)。"""
    w = np.asarray(w, dtype=np.float32, order='C')
    ret, vol = compute_perf_arrays(port_daily, w[None, :], trading_days=trading_days, ddof=ddof)
    return float(ret[0]), float(vol[0])


# ===================== 3) 极值收益（确定扫描区间） =====================
def _solve_extreme_return(
        port_daily: np.ndarray,
        maximize: bool,
        *,
        bounds: List[Tuple[float, float]],
        ddof: int = DDOF,
        maxiter: int = 300,
        ftol: float = 1e-12
) -> Tuple[np.ndarray, float]:
    """
    maximize=True  -> 最大化 年化对数收益
    maximize=False -> 最小化 年化对数收益
    """
    T, N = port_daily.shape
    w0 = np.full(N, 1.0 / N, dtype=np.float64)

    def ret_only(w):
        r, _ = _perf_of_w(port_daily, w, trading_days=TRADING_DAYS, ddof=ddof)
        return -r if maximize else r

    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w, dtype=np.float64) - 1.0}]
    res = minimize(ret_only, w0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': int(maxiter), 'ftol': float(ftol), 'disp': False})
    w_star = np.asarray(res.x, dtype=np.float64, order='C')
    r_star, _ = _perf_of_w(port_daily, w_star, TRADING_DAYS, ddof)
    return w_star.astype(np.float32), r_star


# ===================== 4) 给定目标收益，最小化 VaR =====================
def _solve_min_var_given_return(
        port_daily: np.ndarray,
        target_ret_annual: float,
        *,
        w0: Optional[np.ndarray],
        bounds: List[Tuple[float, float]],
        confidence: float = CONFIDENCE,
        horizon_days: float = HORIZON_DAYS,
        ddof: int = DDOF,
        maxiter: int = MAXITER,
        ftol: float = FTOL
) -> Tuple[np.ndarray, float, float, float]:
    """
    minimize   VaR_raw(w) = -(mu_h + z*sigma_h)
    s.t.       sum(w)=1,  ret_annual(w) >= target_ret_annual, bounds
    返回: (w, ret_annual, vol_annual, var_raw)
    """
    T, N = port_daily.shape
    if w0 is None:
        w0 = np.full(N, 1.0 / N, dtype=np.float64)
    else:
        w0 = np.asarray(w0, dtype=np.float64, order='C')

    def obj(w: np.ndarray) -> float:
        return _var_raw_for_opt(port_daily, w,
                                confidence=confidence, horizon_days=horizon_days,
                                return_type="log", ddof=ddof)

    cons = [
        {'type': 'eq', 'fun': lambda w: np.sum(w, dtype=np.float64) - 1.0},
        {'type': 'ineq', 'fun': lambda w: _perf_of_w(port_daily, w, TRADING_DAYS, ddof)[0] - target_ret_annual},
    ]

    res = minimize(
        obj, w0, method='SLSQP', bounds=bounds, constraints=cons,
        options={'maxiter': int(maxiter), 'ftol': float(ftol), 'disp': False}
    )

    # 若失败，多起点重启（各顶点+等权）
    if not res.success:
        inits = [np.eye(len(w0), dtype=np.float64)[i] for i in range(len(w0))] + [np.full_like(w0, 1 / len(w0))]
        best_val = np.inf
        best_x = None
        for guess in inits:
            res2 = minimize(obj, guess, method='SLSQP', bounds=bounds, constraints=cons,
                            options={'maxiter': int(maxiter), 'ftol': float(ftol), 'disp': False})
            if res2.success:
                val = obj(res2.x)
                if val < best_val:
                    best_val, best_x = val, res2.x.copy()
        if best_x is not None:
            res.x = best_x
        else:
            # 实在不可行：回退初值
            res.x = w0

    w_star = np.asarray(res.x, dtype=np.float64, order='C')
    # 数值修正到 simplex
    if np.any(w_star < 0):
        w_star = np.maximum(w_star, 0.0)
    s = w_star.sum()
    if s <= 0:
        w_star[:] = 1.0 / w_star.size
    else:
        w_star /= s

    ret_a, vol_a = _perf_of_w(port_daily, w_star, TRADING_DAYS, ddof)
    var_raw = _var_raw_for_opt(port_daily, w_star,
                               confidence=confidence, horizon_days=horizon_days,
                               return_type="log", ddof=ddof)
    return w_star.astype(np.float32, copy=False), ret_a, vol_a, var_raw


# ===================== 5) 前沿扫描 =====================
def efficient_frontier_slsqp_var(
        port_daily: np.ndarray,
        *,
        n_targets: int = N_TARGETS,
        bounds: Optional[List[Tuple[float, float]]] = None,
        confidence: float = CONFIDENCE,
        horizon_days: float = HORIZON_DAYS,
        ddof: int = DDOF,
        maxiter: int = MAXITER,
        ftol: float = FTOL,
        use_warm_start: bool = USE_WARM_START
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回:
        W  : (M, N)   —— 每个目标收益下的最优权重
        R  : (M,)     —— 年化对数收益
        VOL: (M,)     —— 年化对数波动
        VaR: (M,)     —— 展示口径 VaR = max(VaR_raw, 0)
    """
    port_daily = np.asarray(port_daily, dtype=np.float32, order='C')
    T, N = port_daily.shape
    bounds = bounds or [(0.0, 1.0)] * N

    # 可达收益区间（在给定 bounds 下）
    _, ret_max = _solve_extreme_return(port_daily, maximize=True, bounds=bounds, ddof=ddof)
    _, ret_min = _solve_extreme_return(port_daily, maximize=False, bounds=bounds, ddof=ddof)
    if ret_max - ret_min < 1e-8:
        ret_min -= 1e-3
        ret_max += 1e-3
    target_rets = np.linspace(ret_min, ret_max, int(n_targets), dtype=np.float64)

    W_list, R_list, VOL_list, VaRraw_list = [], [], [], []
    w_prev = np.full(N, 1.0 / N, dtype=np.float64)

    for k, tgt in enumerate(target_rets, 1):
        w0 = (w_prev if use_warm_start else None)
        w_star, ret_a, vol_a, var_raw = _solve_min_var_given_return(
            port_daily, tgt, w0=w0, bounds=bounds,
            confidence=confidence, horizon_days=horizon_days,
            ddof=ddof, maxiter=maxiter, ftol=ftol
        )
        W_list.append(w_star)
        R_list.append(ret_a)
        VOL_list.append(vol_a)
        VaRraw_list.append(var_raw)
        w_prev = w_star.astype(np.float64, copy=False)
        log(f"[{k}/{len(target_rets)}] target_ret={tgt:.6f} | ret={ret_a:.6f} | VaR_raw={var_raw:.6f}")

    W = np.vstack(W_list).astype(np.float32, copy=False)
    R = np.asarray(R_list, dtype=np.float32)
    VOL = np.asarray(VOL_list, dtype=np.float32)
    VaR_raw = np.asarray(VaRraw_list, dtype=np.float32)
    VaR_disp = np.maximum(VaR_raw, 0.0).astype(np.float32, copy=False)  # 展示口径
    return W, R, VOL, VaR_disp


# ===================== 6) 主流程 =====================
def main():
    port_daily, names = load_returns_from_excel(EXCEL_PATH, SHEET_NAME, ASSETS)

    W, R, VOL, VaR = efficient_frontier_slsqp_var(
        port_daily,
        n_targets=N_TARGETS,
        bounds=BOUNDS,
        confidence=CONFIDENCE,
        horizon_days=HORIZON_DAYS,
        ddof=DDOF,
        maxiter=MAXITER,
        ftol=FTOL,
        use_warm_start=USE_WARM_START
    )

    # 汇总结果
    df = pd.DataFrame({"ret_annual": R, "vol_annual": VOL, "VaR_h": VaR})
    for j, name in enumerate(names):
        df[f"w_{name}"] = W[:, j]

    log("\n前沿样例（前5行）：")
    print(df.head(5).to_string(index=False))

    if SAVE_CSV_PATH:
        df.to_csv(SAVE_CSV_PATH, index=False, encoding="utf-8-sig")
        log(f"\n已保存前沿到: {SAVE_CSV_PATH}")

    if SAVE_FIG_PATH:
        plt.figure(figsize=(7.2, 5.0), dpi=140)
        plt.scatter(VaR, R, s=14)
        plt.xlabel(f"VaR (h={HORIZON_DAYS:g}d, conf={CONFIDENCE:.2f}, log)")
        plt.ylabel("Annualized Log Return")
        plt.title("Efficient Frontier: Min-VaR under Return Target (SLSQP)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(SAVE_FIG_PATH)
        log(f"已保存散点图: {SAVE_FIG_PATH}")


if __name__ == "__main__":
    if not Path(EXCEL_PATH).exists():
        log(f"找不到 Excel 文件: {EXCEL_PATH} —— 请确认路径。")
    else:
        main()
