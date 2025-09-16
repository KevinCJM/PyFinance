# -*- coding: utf-8 -*-
"""
按指定逻辑构建有效前沿（风险: 参数法VaR_log；收益: 年化对数收益）
流程:
1) 最小风险组合
2) 最大收益组合
3) 用二者的风险作为区间, 均分100个风险点
4) 在每个目标风险 = 常数 的等式约束下, 最大化收益
5) 汇总100+2个点构成前沿并绘图

依赖: numpy pandas scipy matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List
from scipy.optimize import minimize
from statistics import NormalDist

# ============ 0) 生成演示数据 (你可替换为真实 daily_returns: shape (T,N)) ============
np.random.seed(42)
T = 252 * 3
asset_names = ['股票', '债券', '黄金', '商品']
N = len(asset_names)

daily_returns = np.random.normal(0.0005, 0.015, (T, N)).astype(np.float64)
daily_returns[:, 1] = np.random.normal(0.0002, 0.005, T)  # 债券更稳

# ============ 1) 基础函数(收益/风险) ============

TRADING_DAYS = 252.0
DDOF = 1
CONFIDENCE = 0.95
HORIZON_DAYS = 1.0
Z = NormalDist().inv_cdf(1.0 - CONFIDENCE)  # 注意Z<0


def annualized_log_return(weights: np.ndarray, R: np.ndarray) -> float:
    """
    年化对数收益:
    先组合简单日收益 Rt = sum_i w_i r_{t,i}, 再对数化 log1p(Rt),
    年化 = 平均日对数收益 * 252
    """
    w = np.asarray(weights, dtype=np.float64)
    Rt = R @ w  # (T,)
    Xt = np.log1p(Rt)
    return float(Xt.mean(dtype=np.float64) * TRADING_DAYS)


def var_log_raw(weights: np.ndarray, R: np.ndarray) -> float:
    """
    参数法VaR（对数收益口径）的“平滑目标”:  VaR_raw = -(mu_h + Z * sigma_h)
    - 不做 abs/clip, 仅用于优化(可导/平滑)；
    - 展示时再做 clip>=0.
    """
    w = np.asarray(weights, dtype=np.float64)
    Rt = R @ w  # (T,)
    Xt = np.log1p(Rt)
    mu = float(Xt.mean(dtype=np.float64))
    sigma = float(Xt.std(ddof=DDOF))
    mu_h = mu * HORIZON_DAYS
    sigma_h = sigma * np.sqrt(HORIZON_DAYS)
    return -(mu_h + Z * sigma_h)


def var_log_display(val_raw: float) -> float:
    """展示口径: VaR>=0"""
    return float(max(val_raw, 0.0))


# ============ 2) 端点组合: 最小风险 & 最大收益 ============

BOUNDS = [(0.0, 1.0)] * N


def minimize_risk(R: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """最小化 VaR_raw, 约束: sum(w)=1, 0<=w<=1"""
    w0 = np.full(N, 1.0 / N)
    obj = lambda w: var_log_raw(w, R)
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    res = minimize(obj, w0, method='SLSQP', bounds=BOUNDS, constraints=cons,
                   options={'maxiter': 800, 'ftol': 1e-12, 'disp': False})
    w = res.x
    # 修正到单纯形
    w = np.maximum(w, 0.0);
    s = w.sum();
    w = (w / s) if s > 0 else np.full(N, 1.0 / N)
    risk_raw = var_log_raw(w, R)
    ret_ann = annualized_log_return(w, R)
    return w, risk_raw, ret_ann


def maximize_return(R: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """最大化年化对数收益, 约束: sum(w)=1, 0<=w<=1"""
    w0 = np.full(N, 1.0 / N)
    obj = lambda w: -annualized_log_return(w, R)
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    res = minimize(obj, w0, method='SLSQP', bounds=BOUNDS, constraints=cons,
                   options={'maxiter': 800, 'ftol': 1e-12, 'disp': False})
    w = res.x
    w = np.maximum(w, 0.0)
    s = w.sum()
    w = (w / s) if s > 0 else np.full(N, 1.0 / N)
    risk_raw = var_log_raw(w, R)
    ret_ann = annualized_log_return(w, R)
    return w, risk_raw, ret_ann


# ============ 3) 扫描固定风险=常数, 最大化收益 ============

def maximize_return_with_risk_eq(R: np.ndarray, target_risk_raw: float,
                                 w0: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    在“风险等于目标”的等式约束下最大化收益:
       max Ret(w)  s.t.  VaR_raw(w) = target_risk_raw, sum(w)=1, 0<=w<=1
    说明:
    - 使用等式能确保每个目标点约束激活, 曲线不会塌缩成一个点；
    - 使用上一点的最优解作为初值(warm start), 提升稳定性；
    - 若等式数值不可行, 自动退化为 <= 约束 + 轻微惩罚 (稳健处理)。
    """
    # 目标: 最大化收益 -> 最小化负收益
    obj = lambda w: -annualized_log_return(w, R)

    # 等式: VaR_raw(w) - target = 0
    cons = [
        {'type': 'eq', 'fun': lambda w, tr=target_risk_raw: var_log_raw(w, R) - tr},
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
    ]

    res = minimize(obj, w0, method='SLSQP', bounds=BOUNDS, constraints=cons,
                   options={'maxiter': 800, 'ftol': 1e-12, 'disp': False})

    if not res.success:
        # 退路: 放宽为 <= 目标 + 惩罚项 (数值更稳)
        lam = 100.0  # 小惩罚系数, 促使VaR尽量贴合target

        def obj_relax(w):
            ret = annualized_log_return(w, R)
            slack = max(0.0, var_log_raw(w, R) - target_risk_raw)  # 超过目标的惩罚
            return -(ret) + lam * (slack ** 2)

        cons2 = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        res2 = minimize(obj_relax, w0, method='SLSQP', bounds=BOUNDS, constraints=cons2,
                        options={'maxiter': 800, 'ftol': 1e-12, 'disp': False})
        w = res2.x
    else:
        w = res.x

    # 修正并返回
    w = np.maximum(w, 0.0);
    s = w.sum();
    w = (w / s) if s > 0 else np.full(N, 1.0 / N)
    ret_ann = annualized_log_return(w, R)
    risk_raw = var_log_raw(w, R)
    return w, risk_raw, ret_ann


# ============ 4) 主流程: 构建前沿 ============

def build_frontier_by_risk_scan(R: np.ndarray, n_points: int = 100) -> pd.DataFrame:
    # 端点
    w_min_risk, risk_min_raw, ret_at_minrisk = minimize_risk(R)
    w_max_ret, risk_at_maxret_raw, ret_max = maximize_return(R)

    # 目标风险网格 (含端点)
    rlo, rhi = float(min(risk_min_raw, risk_at_maxret_raw)), float(max(risk_min_raw, risk_at_maxret_raw))
    targets = np.linspace(rlo, rhi, n_points, dtype=np.float64)

    # 扫描
    rows = []
    w_prev = w_min_risk.copy()  # 热启动
    for k, tr in enumerate(targets, 1):
        w_star, risk_raw, ret_ann = maximize_return_with_risk_eq(R, tr, w_prev)
        w_prev = w_star.copy()
        rows.append({
            'point_type': 'scan',
            'VaR_raw': risk_raw,
            'VaR': var_log_display(risk_raw),
            'ret_annual': ret_ann,
            **{f'w_{name}': w_star[i] for i, name in enumerate(asset_names)}
        })

    # 加入两个端点
    rows.append({
        'point_type': 'min_risk',
        'VaR_raw': float(risk_min_raw),
        'VaR': var_log_display(risk_min_raw),
        'ret_annual': float(ret_at_minrisk),
        **{f'w_{name}': w_min_risk[i] for i, name in enumerate(asset_names)}
    })
    rows.append({
        'point_type': 'max_return',
        'VaR_raw': float(risk_at_maxret_raw),
        'VaR': var_log_display(risk_at_maxret_raw),
        'ret_annual': float(ret_max),
        **{f'w_{name}': w_max_ret[i] for i, name in enumerate(asset_names)}
    })

    df = pd.DataFrame(rows).sort_values('VaR').reset_index(drop=True)
    return df


df_frontier = build_frontier_by_risk_scan(daily_returns, n_points=100)

# ============ 5) 可视化 ============

plt.figure(figsize=(9, 6))
mask_scan = (df_frontier['point_type'] == 'scan')
plt.scatter(df_frontier.loc[mask_scan, 'VaR'], df_frontier.loc[mask_scan, 'ret_annual'],
            c='tab:blue', s=16, label='Frontier (scan)')

# 标注端点
p_min = df_frontier.loc[df_frontier['point_type'] == 'min_risk'].iloc[0]
p_max = df_frontier.loc[df_frontier['point_type'] == 'max_return'].iloc[0]
plt.scatter(p_min['VaR'], p_min['ret_annual'], marker='*', s=180, color='red',
            label=f"Min-Risk ({p_min['VaR']:.4f}, {p_min['ret_annual']:.4f})")
plt.scatter(p_max['VaR'], p_max['ret_annual'], marker='*', s=180, color='green',
            label=f"Max-Return ({p_max['VaR']:.4f}, {p_max['ret_annual']:.4f})")

plt.title('有效前沿: 固定 VaR_log = 常数，最大化年化对数收益')
plt.xlabel(f"VaR (h={HORIZON_DAYS:g}d, conf={CONFIDENCE:.2f}, log)")
plt.ylabel("Annualized Log Return")
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
