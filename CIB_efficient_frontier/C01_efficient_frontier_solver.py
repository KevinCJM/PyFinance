# -*- coding: utf-8 -*-
"""
按指定逻辑构建有效前沿（风险: 参数法VaR_log；收益: 年化对数收益）
流程:
1) 最小风险组合
2) 最大收益组合
3) 用二者的风险作为区间, 均分100个风险点
4) 在每个目标风险 = 常数 的等式约束下, 最大化收益
5) 汇总100+2个点构成前沿并用 plotly 展示

依赖: numpy pandas scipy plotly openpyxl
"""

import os
from typing import Tuple, List
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statistics import NormalDist
import plotly.graph_objects as go

# ===================== 0) Excel 参数 =====================
EXCEL_PATH = "历史净值数据_万得指数.xlsx"
SHEET_NAME = "历史净值数据"
ASSETS = ["货币现金类", "固定收益类", "混合策略类", "权益投资类", "另类投资类"]

# ===================== 1) 读数与基本配置 =====================
TRADING_DAYS = 252.0
DDOF = 1
CONFIDENCE = 0.95
HORIZON_DAYS = 1.0
Z = NormalDist().inv_cdf(1.0 - CONFIDENCE)  # 注意Z<0

MAXITER = 800
FTOL = 1e-12
N_SCAN = 100


def log(msg: str) -> None:
    print(msg, flush=True)


def load_returns_from_excel(
        excel_path: str, sheet_name: str, assets_list: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """从 Excel 读取净值数据，生成日收益二维数组 (T,N)。"""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"找不到 Excel 文件: {excel_path}")
    log(f"加载数据: {excel_path} | sheet={sheet_name}")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if "date" not in df.columns:
        raise ValueError("Excel中缺少 'date' 列")

    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.dropna().sort_index(ascending=True)

    # 列重命名对齐
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
    R = hist_ret_df.values.astype(np.float64, copy=False)  # (T, N)
    log(f"数据加载完成，样本天数={R.shape[0]}，资产数={R.shape[1]}")
    return R, assets_list


# ===================== 2) 基础函数(收益/风险) =====================
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


# ===================== 3) 端点组合: 最小风险 & 最大收益 =====================
def _project_simplex(w: np.ndarray) -> np.ndarray:
    w = np.maximum(np.asarray(w, dtype=np.float64), 0.0)
    s = w.sum()
    if s <= 0:
        w[:] = 1.0 / w.size
    else:
        w /= s
    return w


def minimize_risk(R: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """最小化 VaR_raw, 约束: sum(w)=1, 0<=w<=1"""
    N = R.shape[1]
    BOUNDS = [(0.0, 1.0)] * N
    w0 = np.full(N, 1.0 / N)
    obj = lambda w: var_log_raw(w, R)
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    res = minimize(obj, w0, method='SLSQP', bounds=BOUNDS, constraints=cons,
                   options={'maxiter': MAXITER, 'ftol': FTOL, 'disp': False})
    w = _project_simplex(res.x)
    risk_raw = var_log_raw(w, R)
    ret_ann = annualized_log_return(w, R)
    return w, risk_raw, ret_ann


def maximize_return(R: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """最大化年化对数收益, 约束: sum(w)=1, 0<=w<=1"""
    N = R.shape[1]
    BOUNDS = [(0.0, 1.0)] * N
    w0 = np.full(N, 1.0 / N)
    obj = lambda w: -annualized_log_return(w, R)
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    res = minimize(obj, w0, method='SLSQP', bounds=BOUNDS, constraints=cons,
                   options={'maxiter': MAXITER, 'ftol': FTOL, 'disp': False})
    w = _project_simplex(res.x)
    risk_raw = var_log_raw(w, R)
    ret_ann = annualized_log_return(w, R)
    return w, risk_raw, ret_ann


# ===================== 4) 扫描固定风险=常数, 最大化收益 =====================
def maximize_return_with_risk_eq(R: np.ndarray, target_risk_raw: float,
                                 w0: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    在“风险等于目标”的等式约束下最大化收益:
       max Ret(w)  s.t.  VaR_raw(w) = target_risk_raw, sum(w)=1, 0<=w<=1
    """
    N = R.shape[1]
    BOUNDS = [(0.0, 1.0)] * N

    # 目标: 最大化收益 -> 最小化负收益
    obj = lambda w: -annualized_log_return(w, R)

    # 等式: VaR_raw(w) - target = 0
    cons = [
        {'type': 'eq', 'fun': lambda w, tr=target_risk_raw: var_log_raw(w, R) - tr},
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
    ]

    res = minimize(obj, w0, method='SLSQP', bounds=BOUNDS, constraints=cons,
                   options={'maxiter': MAXITER, 'ftol': FTOL, 'disp': False})

    if not res.success:
        # 退路: 放宽为 <= 目标 + 惩罚项 (数值更稳)
        lam = 100.0  # 小惩罚系数, 促使VaR尽量贴合target

        def obj_relax(w):
            ret = annualized_log_return(w, R)
            slack = max(0.0, var_log_raw(w, R) - target_risk_raw)  # 超过目标的惩罚
            return -(ret) + lam * (slack ** 2)

        cons2 = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        res2 = minimize(obj_relax, w0, method='SLSQP', bounds=BOUNDS, constraints=cons2,
                        options={'maxiter': MAXITER, 'ftol': FTOL, 'disp': False})
        w = res2.x
    else:
        w = res.x

    # 修正并返回
    w = _project_simplex(w)
    ret_ann = annualized_log_return(w, R)
    risk_raw = var_log_raw(w, R)
    return w, risk_raw, ret_ann


# ===================== 5) 主流程: 构建前沿 =====================
def build_frontier_by_risk_scan(R: np.ndarray, n_points: int = 100) -> pd.DataFrame:
    # 端点
    w_min_risk, risk_min_raw, ret_at_minrisk = minimize_risk(R)
    w_max_ret, risk_at_maxret_raw, ret_max = maximize_return(R)

    log(f"端点：最小风险 VaR_raw={risk_min_raw:.6f}, 其年化对数收益={ret_at_minrisk:.6f}")
    log(f"端点：最大收益 年化对数收益={ret_max:.6f}, 其 VaR_raw={risk_at_maxret_raw:.6f}")

    # 目标风险网格 (含端点)
    rlo = float(min(risk_min_raw, risk_at_maxret_raw))
    rhi = float(max(risk_min_raw, risk_at_maxret_raw))
    targets = np.linspace(rlo, rhi, n_points, dtype=np.float64)

    # 扫描
    rows = []
    w_prev = w_min_risk.copy()  # 热启动
    asset_names = ASSETS
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


# ===================== 6) Plotly 可视化 =====================
def plot_frontier_plotly(df: pd.DataFrame) -> None:
    mask_scan = (df['point_type'] == 'scan')
    mask_min = (df['point_type'] == 'min_risk')
    mask_max = (df['point_type'] == 'max_return')

    fig = go.Figure()

    # 前沿扫描点
    fig.add_trace(go.Scatter(
        x=df.loc[mask_scan, 'VaR'],
        y=df.loc[mask_scan, 'ret_annual'],
        mode='markers',
        name='Frontier (scan)',
        marker=dict(size=6),
        hovertemplate='VaR=%{x:.6f}<br>Annual Log Return=%{y:.6f}<extra></extra>'
    ))

    # 端点：最小风险
    if mask_min.any():
        fig.add_trace(go.Scatter(
            x=df.loc[mask_min, 'VaR'],
            y=df.loc[mask_min, 'ret_annual'],
            mode='markers+text',
            name='Min-Risk',
            marker=dict(size=12, symbol='star', color='red'),
            text=['Min-Risk'],
            textposition='top center',
            hovertemplate='VaR=%{x:.6f}<br>Annual Log Return=%{y:.6f}<extra></extra>'
        ))

    # 端点：最大收益
    if mask_max.any():
        fig.add_trace(go.Scatter(
            x=df.loc[mask_max, 'VaR'],
            y=df.loc[mask_max, 'ret_annual'],
            mode='markers+text',
            name='Max-Return',
            marker=dict(size=12, symbol='star', color='green'),
            text=['Max-Return'],
            textposition='bottom center',
            hovertemplate='VaR=%{x:.6f}<br>Annual Log Return=%{y:.6f}<extra></extra>'
        ))

    fig.update_layout(
        title=f"有效前沿: 固定 VaR_log = 常数，最大化年化对数收益<br><sub>h={HORIZON_DAYS:g}d, conf={CONFIDENCE:.2f}</sub>",
        xaxis_title=f"VaR (display)",
        yaxis_title="Annualized Log Return",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    fig.show()


# ===================== 7) main =====================
def main():
    R, _ = load_returns_from_excel(EXCEL_PATH, SHEET_NAME, ASSETS)
    df_frontier = build_frontier_by_risk_scan(R, n_points=N_SCAN)

    log("\n前沿（前6行预览）：")
    print(df_frontier.head(6).to_string(index=False))

    plot_frontier_plotly(df_frontier)


if __name__ == "__main__":
    main()
