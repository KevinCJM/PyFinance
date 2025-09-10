# -*- encoding: utf-8 -*-
"""
@File: B01_random_weights.py
@Modify Time: 2025/9/10 12:00
@Author: Kevin-Chen
@Descriptions: 结合了约束随机游走和性能计算的优化脚本。
             采纳建议，使用“前沿感知”混合采样策略，精确刻画并填充有效前沿。
"""
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go


# =====================================================================================
# 1. 性能计算与基础函数
# =====================================================================================

def generate_alloc_perf_batch(port_daily: np.ndarray, portfolio_allocs: np.ndarray, chunk_size: int = 20000) -> pd.DataFrame:
    """(已优化) 分块计算，避免 T×N 爆内存；并且对 log 计算做 clip，清理 ±inf. """
    T, n = port_daily.shape
    N = portfolio_allocs.shape[0]
    res_list = []
    for s in range(0, N, chunk_size):
        e = min(N, s + chunk_size)
        W = portfolio_allocs[s:e]
        R = port_daily @ W.T
        one_plus_R = np.clip(1.0 + R, 1e-12, None)
        port_cum = np.cumprod(one_plus_R, axis=0)
        final_ret = port_cum[-1, :]
        log_total = np.log(np.clip(final_ret, 1e-12, None))
        ret_annual = (log_total / T) * 252.0
        log_daily = np.log(one_plus_R)
        vol_annual = np.std(log_daily, axis=0, ddof=1) * np.sqrt(252.0)
        df = pd.DataFrame({"ret_annual": ret_annual, "vol_annual": vol_annual})
        wdf = pd.DataFrame(W, columns=[f"w_{i}" for i in range(n)])
        res_list.append(pd.concat([wdf, df], axis=1))
    out = pd.concat(res_list, axis=0, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def cal_ef2_v4_ultra_fast(data: pd.DataFrame) -> pd.DataFrame:
    """
    从给定的投资组合点中，高效识别出位于有效前沿上的点。
    """
    ret_values = data['ret_annual'].values
    vol_values = data['vol_annual'].values
    sorted_idx = np.argsort(ret_values)[::-1]
    sorted_vol = vol_values[sorted_idx]
    cummin_vol = np.minimum.accumulate(sorted_vol)
    on_ef_sorted = (sorted_vol <= cummin_vol + 1e-6)
    on_ef = np.zeros(len(data), dtype=bool)
    on_ef[sorted_idx] = on_ef_sorted
    data['on_ef'] = on_ef
    return data


# =====================================================================================
# 2. 核心投影与采样函数 (采纳同事最终建议)
# =====================================================================================

def project_to_constraints_pocs(v: np.ndarray, single_limits, multi_limits: dict,
                                max_iter=200, tol=1e-9, damping=1.0):
    """(已优化) 用 POCS 交替投影到：盒约束 ∩ 总和=1 ∩ 各组半空间 之交集. """
    x = v.copy()
    n = x.size
    lows = np.array([a for a, _ in single_limits], dtype=np.float64)
    highs = np.array([b for _, b in single_limits], dtype=np.float64)
    groups = []
    for idx_tuple, (low, up) in multi_limits.items():
        idx = np.array(idx_tuple, dtype=np.int64)
        a_norm2 = float(len(idx))
        groups.append((idx, float(low), float(up), a_norm2))
    x = np.clip(x, lows, highs)
    x += (1.0 - x.sum()) / n
    for _ in range(max_iter):
        x_prev = x.copy()
        x = np.clip(x, lows, highs)
        x += (1.0 - x.sum()) / n
        for idx, low, up, a_norm2 in groups:
            if a_norm2 == 0: continue
            t = x[idx].sum()
            if t > up + 1e-12:
                delta = (t - up) / a_norm2
                x[idx] -= damping * delta
            elif t < low - 1e-12:
                delta = (low - t) / a_norm2
                x[idx] += damping * delta
        x += (1.0 - x.sum()) / n
        if np.linalg.norm(x - x_prev, ord=np.inf) < tol:
            break
    if np.any(x < lows - 1e-6) or np.any(x > highs + 1e-6):
        return None
    for idx, low, up, _ in groups:
        if len(idx) == 0: continue
        t = x[idx].sum()
        if t < low - 1e-6 or t > up + 1e-6:
            return None
    if not np.isclose(x.sum(), 1.0, atol=1e-6):
        return None
    return x

def _largest_eigval_est(Sigma, iters=30):
    n = Sigma.shape[0]
    v = np.ones(n) / n
    for _ in range(iters):
        v = Sigma @ v
        nv = np.linalg.norm(v)
        if nv < 1e-18: break
        v /= nv
    return float(v @ (Sigma @ v))

def _lambda_grid(mu, Sigma, n_lams=60):
    mu_s = np.median(np.abs(mu)) + 1e-12
    sig_s = np.median(np.diag(Sigma)) + 1e-12
    lam_min = 0.02 * sig_s / mu_s
    lam_max = 50.0 * sig_s / mu_s
    return np.geomspace(lam_min, lam_max, num=n_lams)

def solve_anchor_pgd(Sigma, mu, single_limits, multi_limits, lam,
                     w0=None, max_iter=2000, tol=1e-9, eta=None):
    n = Sigma.shape[0]
    w = (np.ones(n) / n) if w0 is None else w0.copy()
    if eta is None:
        L = 2.0 * _largest_eigval_est(Sigma)
        eta = 1.0 / max(L, 1e-12)
    for _ in range(max_iter):
        grad = 2.0 * (Sigma @ w) - lam * mu
        w_next = w - eta * grad
        w_next = project_to_constraints_pocs(w_next, single_limits, multi_limits,
                                             max_iter=300, tol=1e-10, damping=0.9)
        if w_next is None:
            eta *= 0.5
            continue
        if np.linalg.norm(w_next - w, ord=np.inf) < tol:
            w = w_next
            break
        w = w_next
    return w

def generate_frontier_anchors(Sigma, mu, single_limits, multi_limits, n_lams=80):
    """步骤1: 用PGD扫描lambda参数，生成有效前沿的锚点."""
    print(f"步骤1: 正在扫描{n_lams}个lambda值以生成前沿锚点...")
    lams = _lambda_grid(mu, Sigma, n_lams)
    anchors = []
    w0 = None
    for i, lam in enumerate(lams):
        w = solve_anchor_pgd(Sigma, mu, single_limits, multi_limits, lam, w0=w0)
        anchors.append(w)
        w0 = w
    return np.asarray(anchors)

def filter_efficient(perf_df, anchors):
    """步骤2: 过滤锚点，确保其单调有效."""
    print("步骤2: 正在过滤锚点以确保单调有效性...")
    ret = perf_df['ret_annual'].values
    vol = perf_df['vol_annual'].values
    idx = np.argsort(vol)
    keep = np.zeros_like(vol, dtype=bool)
    best_ret = -np.inf
    for i in idx:
        if ret[i] > best_ret + 1e-10:
            keep[i] = True
            best_ret = ret[i]
    print(f"过滤后剩余 {np.sum(keep)} 个有效锚点。")
    return anchors[keep], ret[keep], vol[keep]

def convex_mix_adjacent(sorted_anchors, n_samples, seed=42):
    """步骤3a: 沿前沿进行邻段凸组合，均匀“加粗”前沿."""
    rng = np.random.default_rng(seed)
    M, n = sorted_anchors.shape
    if M < 2: return np.empty((0, n))
    out = np.empty((n_samples, n))
    for t in range(n_samples):
        j = rng.integers(0, M - 1)
        w1, w2 = sorted_anchors[j], sorted_anchors[j + 1]
        a = rng.beta(2.0, 2.0)
        out[t] = a * w1 + (1.0 - a) * w2
    return out

def jitter_near_anchors(anchors, single_limits, multi_limits, m_each=30, noise=0.01, seed=123):
    """步骤3b: 在锚点附近抖动并再投影，形成贴近前沿的薄带."""
    rng = np.random.default_rng(seed)
    M, n = anchors.shape
    out = []
    for i in range(M):
        w = anchors[i]
        eps = rng.normal(0.0, noise, size=(m_each, n))
        eps -= eps.mean(axis=1, keepdims=True)
        for k in range(m_each):
            w_try = w + eps[k]
            w_new = project_to_constraints_pocs(w_try, single_limits, multi_limits)
            if w_new is not None:
                out.append(w_new)
    return np.asarray(out) if out else np.empty((0, n))


# =====================================================================================
# 主程序
# =====================================================================================

if __name__ == '__main__':
    # --- 1. 数据加载和预处理 ---
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
    hist_value_r = hist_value[assets_list].pct_change().dropna()

    # --- 2. 定义约束并使用“前沿感知”策略生成权重 ---
    print("--- 开始通过‘前沿感知’策略生成投资组合 ---")
    num_of_asset = len(assets_list)
    single_limits = [(0.0, 1.0), (0.0, 0.8), (0.0, 0.7), (0.0, 0.6), (0.0, 0.2)]
    multi_limits = {(0, 1): (0.1, 1.0)}

    # 计算优化所需的 mu (期望收益) 和 Sigma (协方差矩阵)
    log_r = np.log1p(hist_value_r[assets_list])
    mu = log_r.mean().values * 252
    Sigma = log_r.cov().values * 252

    # 步骤1+2: 生成并过滤有效锚点
    anchors_raw = generate_frontier_anchors(Sigma, mu, single_limits, multi_limits, n_lams=100)
    perf_anchors = generate_alloc_perf_batch(hist_value_r[assets_list].values, anchors_raw)
    anchors_eff, ret_eff, vol_eff = filter_efficient(perf_anchors, anchors_raw)
    
    # 步骤3: 在有效锚点周围密集采样
    n_total = 50000
    n_mix = int(n_total * 0.7)
    n_jitter = int(n_total * 0.3)
    print(f"\n步骤3: 正在生成 {n_total} 个样本 (邻段凸组合: {n_mix}, 锚点抖动: {n_jitter})...")
    
    weights_mix = convex_mix_adjacent(anchors_eff, n_mix, seed=42)
    weights_jitter = jitter_near_anchors(anchors_eff, single_limits, multi_limits, m_each=max(1, n_jitter // len(anchors_eff)), seed=123)
    
    weights_array = np.vstack([anchors_eff, weights_mix, weights_jitter])
    print(f"采样完成，共得到 {weights_array.shape[0]} 个有效权重。")

    # --- 4. 批量计算最终所有组合的性能 ---
    if weights_array.shape[0] > 0:
        print("\n正在批量计算所有组合的收益与风险...")
        port_daily_returns = hist_value_r[assets_list].values
        results_df = generate_alloc_perf_batch(port_daily_returns, weights_array)
        results_df = cal_ef2_v4_ultra_fast(results_df)
        print("计算完成。")

        # --- 5. 使用 Plotly 进行交互式可视化 ---
        print("正在生成交互式图表...")
        weight_cols = {f'w_{i}': assets_list[i] for i in range(len(assets_list))}
        results_df = results_df.rename(columns=weight_cols)

        def create_hover_text(df_row):
            text = f"年化收益率: {df_row['ret_annual']:.2%}<br>年化波动率: {df_row['vol_annual']:.2%}<br><br><b>资产权重</b>:<br>"
            for asset in assets_list:
                if asset in df_row and df_row[asset] > 1e-4:
                    text += f"  {asset}: {df_row[asset]:.1%}<br>"
            return text

        results_df['hover_text'] = results_df.apply(create_hover_text, axis=1)
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=results_df['vol_annual'], y=results_df['ret_annual'],
            hovertext=results_df['hover_text'], hoverinfo='text',
            mode='markers', marker=dict(color='lightblue', size=2, opacity=0.7),
            name='随机有效组合'
        ))

        ef_df = results_df[results_df['on_ef'] == True]
        fig.add_trace(go.Scatter(
            x=ef_df['vol_annual'], y=ef_df['ret_annual'],
            hovertext=ef_df['hover_text'], hoverinfo='text',
            mode='markers', marker=dict(color='gold', size=2, line=dict(width=1, color='darkslategrey')),
            name='有效前沿'
        ))

        fig.update_layout(
            title='约束随机游走生成的投资组合与有效前沿',
            xaxis_title='年化波动率 (Annual Volatility)',
            yaxis_title='年化收益率 (Annual Return)',
            legend_title="图例", hovermode='closest'
        )
        fig.show()
    else:
        print("未能生成任何有效的投资组合，无法进行后续计算和绘图。")
