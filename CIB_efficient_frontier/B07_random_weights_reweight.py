# -*- coding: utf-8 -*-
import numpy as np
import cvxpy as cp
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple

''' 一、通用工具 & 画图 '''


def plot_efficient_frontier(
        scatter_points_data: List[Dict[str, Any]],
        title: str = '投资组合与有效前沿',
        x_axis_title: str = '年化波动率 (Annual Volatility)',
        y_axis_title: str = '年化收益率 (Annual Return)',
        x_col: str = 'vol_annual',
        y_col: str = 'ret_annual',
        hover_text_col: str = 'hover_text'
):
    fig = go.Figure()
    for point_set in scatter_points_data:
        df = point_set["data"]
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[y_col],
            hovertext=df[hover_text_col], hoverinfo='text',
            mode='markers',
            marker=dict(
                color=point_set["color"],
                size=point_set["size"],
                opacity=point_set["opacity"],
                line=point_set.get("marker_line")
            ),
            name=point_set["name"]
        ))
    fig.update_layout(
        title=title, xaxis_title=x_axis_title, yaxis_title=y_axis_title,
        legend_title="图例", hovermode='closest'
    )
    fig.show()


def generate_alloc_perf_batch(port_daily: np.ndarray, portfolio_allocs: np.ndarray,
                              chunk_size: int = 20000) -> pd.DataFrame:
    assert port_daily.shape[1] == portfolio_allocs.shape[1]
    T, n = port_daily.shape
    N = portfolio_allocs.shape[0]
    res_list = []
    for s in range(0, N, chunk_size):
        e = min(N, s + chunk_size)
        W = portfolio_allocs[s:e]  # [m, n]
        R = port_daily @ W.T  # [T, m]
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
    data = data.copy()
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


''' 二、POCS 投影 (盒约束 ∩ 和=1 ∩ 多资产组约束) '''


def project_to_constraints_pocs(v: np.ndarray,
                                single_limits,  # list[(low, high)]
                                multi_limits: dict,  # {(tuple_idx): (low, high)}
                                max_iter=200, tol=1e-9, damping=0.9):
    x = v.astype(np.float64).copy()
    n = x.size
    lows = np.array([a for a, _ in single_limits], dtype=np.float64)
    highs = np.array([b for _, b in single_limits], dtype=np.float64)
    groups = []
    for idx_tuple, (low, up) in multi_limits.items():
        idx = np.array(idx_tuple, dtype=np.int64)
        a2 = float(len(idx))
        groups.append((idx, float(low), float(up), a2))
    x = np.clip(x, lows, highs)
    x += (1.0 - x.sum()) / n
    for _ in range(max_iter):
        x_prev = x
        x = np.clip(x, lows, highs)
        x += (1.0 - x.sum()) / n
        for idx, low, up, a2 in groups:
            if a2 == 0: continue
            s = x[idx].sum()
            if s > up + 1e-12:
                x[idx] -= damping * (s - up) / a2
            elif s < low - 1e-12:
                x[idx] += damping * (low - s) / a2
        x += (1.0 - x.sum()) / n
        if np.max(np.abs(x - x_prev)) < tol:
            break
    if (x < lows - 1e-6).any() or (x > highs + 1e-6).any(): return None
    for idx, low, up, _ in groups:
        if len(idx) == 0: continue
        s = x[idx].sum()
        if s < low - 1e-6 or s > up + 1e-6: return None
    if not np.isclose(x.sum(), 1.0, atol=1e-6): return None
    return x


''' 三、统计量与前沿刻画 (QCQP 逐风险扫描) '''


def ann_mu_sigma(log_returns: np.ndarray):
    mu = log_returns.mean(axis=0) * 252.0
    Sigma = np.cov(log_returns, rowvar=False, ddof=1) * 252.0
    return mu, Sigma


def port_stats(W: np.ndarray, mu: np.ndarray, Sigma: np.ndarray):
    if W.ndim == 1:
        ret = float(W @ mu);
        vol = float(np.sqrt(W @ Sigma @ W))
        return np.array([ret]), np.array([vol])
    rets = W @ mu
    vols = np.sqrt(np.einsum('ij,jk,ik->i', W, Sigma, W))
    return rets, vols


def solve_min_variance(Sigma, single_limits, multi_limits):
    n = Sigma.shape[0]
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1]
    for i, (lo, hi) in enumerate(single_limits):
        cons += [w[i] >= lo, w[i] <= hi]
    for idxs, (low, up) in multi_limits.items():
        cons += [cp.sum(w[list(idxs)]) >= low, cp.sum(w[list(idxs)]) <= up]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), cons)
    prob.solve(solver=cp.ECOS, warm_start=True, abstol=1e-8, reltol=1e-8, feastol=1e-8)
    return w.value


def solve_max_return(mu, single_limits, multi_limits):
    n = mu.size
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1]
    for i, (lo, hi) in enumerate(single_limits):
        cons += [w[i] >= lo, w[i] <= hi]
    for idxs, (low, up) in multi_limits.items():
        cons += [cp.sum(w[list(idxs)]) >= low, cp.sum(w[list(idxs)]) <= up]
    prob = cp.Problem(cp.Maximize(mu @ w), cons)
    prob.solve(solver=cp.ECOS, warm_start=True, abstol=1e-8, reltol=1e-8, feastol=1e-8)
    return w.value


def solve_max_return_at_risk(mu, Sigma, s_target, single_limits, multi_limits, w0=None):
    n = mu.size
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, cp.quad_form(w, Sigma) <= float(s_target ** 2)]
    for i, (lo, hi) in enumerate(single_limits):
        cons += [w[i] >= lo, w[i] <= hi]
    for idxs, (low, up) in multi_limits.items():
        cons += [cp.sum(w[list(idxs)]) >= low, cp.sum(w[list(idxs)]) <= up]
    prob = cp.Problem(cp.Maximize(mu @ w), cons)
    if w0 is not None:
        try:
            w.value = w0
        except Exception:
            pass
    prob.solve(solver=cp.ECOS, warm_start=True, abstol=5e-8, reltol=5e-8, feastol=5e-8, max_iters=1000)
    return w.value


def sweep_frontier_by_risk(mu, Sigma, single_limits, multi_limits, n_grid=1200):
    w_minv = solve_min_variance(Sigma, single_limits, multi_limits)
    w_maxr = solve_max_return(mu, single_limits, multi_limits)
    _, s_min = port_stats(w_minv, mu, Sigma);
    s_min = s_min[0]
    _, s_max = port_stats(w_maxr, mu, Sigma);
    s_max = float(max(s_min, s_max))
    grid = np.linspace(s_min, s_max, n_grid)
    W = []
    w0 = w_minv
    for s in grid:
        w = solve_max_return_at_risk(mu, Sigma, s, single_limits, multi_limits, w0=w0)
        W.append(w);
        w0 = w
    W = np.asarray(W)
    R, S = port_stats(W, mu, Sigma)
    return grid, W, R, S, w_minv, w_maxr


def make_upper_envelope_fn(R: np.ndarray, S: np.ndarray):
    order = np.argsort(S)
    S_sorted = S[order];
    R_sorted = R[order]

    def f(sig):
        sig = np.atleast_1d(sig)
        return np.interp(sig, S_sorted, R_sorted, left=R_sorted[0], right=R_sorted[-1])

    return f


''' 四、量化到指定精度 & 去重 （保留原逻辑，便于后续使用） '''


def _parse_precision(choice: str) -> float:
    choice = str(choice).strip()
    return float(choice[:-1]) / 100.0 if choice.endswith('%') else float(choice)


def _snap_to_grid_simplex(w: np.ndarray, step: float, single_limits) -> np.ndarray | None:
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
            if diff == 0: break
            add = min(cap[i], diff)
            if add > 0: k[i] += add; diff -= add
        if diff != 0: return None
    elif diff < 0:
        cap = k - lo_units
        idx = np.argsort(frac)
        for i in idx:
            if diff == 0: break
            sub = min(cap[i], -diff)
            if sub > 0: k[i] -= sub; diff += sub
        if diff != 0: return None
    wq = k.astype(np.float64) / R
    if (wq < lows - 1e-12).any() or (wq > highs + 1e-12).any(): return None
    if not np.isclose(wq.sum(), 1.0, atol=1e-12): return None
    return wq


def quantize_with_projection(w: np.ndarray, step: float,
                             single_limits, multi_limits,
                             rounds: int = 5) -> np.ndarray | None:
    x = w.copy()
    for _ in range(rounds):
        x = project_to_constraints_pocs(x, single_limits, multi_limits,
                                        max_iter=300, tol=1e-10, damping=0.9)
        if x is None: return None
        xq = _snap_to_grid_simplex(x, step, single_limits)
        if xq is None: return None
        if np.max(np.abs(xq - x)) < step * 0.5: return xq
        x = xq
    return x


def dedup_by_grid(W: np.ndarray, step: float) -> np.ndarray:
    if W.size == 0: return W
    K = np.rint(W / step).astype(np.int64)
    _, idx = np.unique(K, axis=0, return_index=True)
    return W[np.sort(idx)]


''' 五、以前沿锚点为种子：小步随机游走 + POCS（可选精度）填厚前沿之下 '''


def random_walk_below_frontier(W_anchor: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
                               single_limits, multi_limits,
                               per_anchor: int = 30, step: float = 0.01,
                               sigma_tol: float = 1e-4, seed: int = 123,
                               precision: str | float | None = None):
    rng = np.random.default_rng(seed)
    R_anchor, S_anchor = port_stats(W_anchor, mu, Sigma)
    f_upper = make_upper_envelope_fn(R_anchor, S_anchor)
    step_grid = None
    if precision is not None:
        step_grid = _parse_precision(precision)
    collected = []
    for w0 in W_anchor:
        _, s0 = port_stats(w0, mu, Sigma);
        s0 = s0[0];
        s_bar = s0 + sigma_tol
        for _ in range(per_anchor):
            eps = rng.normal(0.0, step, size=w0.size);
            eps -= eps.mean()
            w_try = project_to_constraints_pocs(w0 + eps, single_limits, multi_limits,
                                                max_iter=200, tol=1e-9, damping=0.9)
            if w_try is None: continue
            if step_grid is not None:
                w_try = quantize_with_projection(w_try, step_grid, single_limits, multi_limits, rounds=5)
                if w_try is None: continue
            r, s = port_stats(w_try, mu, Sigma)
            if (s[0] <= s_bar + 1e-12) and (r[0] <= f_upper(s)[0] + 1e-8):
                collected.append(w_try)
    W = np.array(collected) if collected else np.empty((0, W_anchor.shape[1]))
    if step_grid is not None and W.size:
        W = dedup_by_grid(W, step_grid)
    return W


def generate_constrained_portfolios(num_points: int, single_limits, multi_limits):
    generated_weights = []
    lows = np.array([l for l, h in single_limits])
    highs = np.array([h for l, h in single_limits])
    while len(generated_weights) < num_points:
        proposal = np.random.uniform(lows, highs)
        adjusted = project_to_constraints_pocs(proposal, single_limits, multi_limits)
        if adjusted is not None:
            generated_weights.append(adjusted)
    return np.array(generated_weights)


''' 六、平台“极限约束” = 各档 ±band 后逐资产取全局最宽区间 '''


def make_platform_limits_from_all_levels(proposed_alloc: Dict[str, Dict[str, float]],
                                         assets_list: List[str],
                                         band: float = 0.20,
                                         mode: str = "relative",
                                         unlock_zero: bool = False,
                                         zero_cap: float | None = None
                                         ) -> Tuple[
    List[Tuple[float, float]], Dict[Tuple[int, ...], Tuple[float, float]]]:
    """
    根据 C1~C6 基准 + 相对/绝对带宽，生成平台统一的单资产极限约束 single_limits。
    multi_limits 此处返回空字典；如需平台组约束，可在此函数内派生后返回。
    """
    levels = list(proposed_alloc.keys())
    B = np.array([[proposed_alloc[level].get(a, 0.0) for a in assets_list] for level in levels], dtype=float)  # [K,n]
    if mode == "relative":
        L = np.maximum(0.0, (1.0 - band) * B)
        U = np.minimum(1.0, (1.0 + band) * B)
        if unlock_zero:
            cap = band if zero_cap is None else min(band, zero_cap)
            U = np.where(B == 0.0, np.minimum(U, cap), U)
    elif mode == "absolute":
        L = np.maximum(0.0, B - band)
        U = np.minimum(1.0, B + band)
    else:
        raise ValueError("mode must be 'relative' or 'absolute'")
    lo = L.min(axis=0)
    hi = U.max(axis=0)
    # 可行性修正：确保 sum(lo) <= 1 <= sum(hi)
    s_lo, s_hi = lo.sum(), hi.sum()
    n = lo.size
    if s_lo > 1 + 1e-12:
        lo = lo * (1.0 / s_lo)
    if s_hi < 1 - 1e-12:
        slack = 1.0 - s_hi
        room = 1.0 - hi
        if room.sum() > 1e-12:
            hi = hi + slack * (room / room.sum())
        else:
            hi = np.minimum(1.0, hi + slack / n)
    single_limits = [(float(lo[i]), float(hi[i])) for i in range(n)]
    multi_limits: Dict[Tuple[int, ...], Tuple[float, float]] = {}  # 如需组约束可在此构造
    return single_limits, multi_limits


def intersect_single_limits(A: List[Tuple[float, float]],
                            B: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """ 两个盒约束逐维求交集（确保 lo<=hi） """
    out = []
    for (a, b), (c, d) in zip(A, B):
        lo = max(a, c);
        hi = min(b, d)
        if hi < lo:  # 退化，取点在交叠边界
            hi = lo
        out.append((lo, hi))
    return out


''' 七、主程序 '''

if __name__ == '__main__':
    # --- 1) 数据加载与预处理 ---
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
    port_daily_returns = hist_value_r.values
    log_r = np.log1p(port_daily_returns)
    mu, Sigma = ann_mu_sigma(log_r)

    # --- 2) C1~C6 基准配置（用于构造平台极限约束） ---
    proposed_alloc = {
        'C1': {'货币现金类': 1.0, '固定收益类': 0.0, '混合策略类': 0.0, '权益投资类': 0.0, '另类投资类': 0.0},
        'C2': {'货币现金类': 0.2, '固定收益类': 0.8, '混合策略类': 0.0, '权益投资类': 0.0, '另类投资类': 0.0},
        'C3': {'货币现金类': 0.1, '固定收益类': 0.55, '混合策略类': 0.35, '权益投资类': 0.0, '另类投资类': 0.0},
        'C4': {'货币现金类': 0.05, '固定收益类': 0.4, '混合策略类': 0.3, '权益投资类': 0.2, '另类投资类': 0.05},
        'C5': {'货币现金类': 0.05, '固定收益类': 0.2, '混合策略类': 0.25, '权益投资类': 0.4, '另类投资类': 0.1},
        'C6': {'货币现金类': 0.05, '固定收益类': 0.1, '混合策略类': 0.15, '权益投资类': 0.6, '另类投资类': 0.1}
    }

    # --- 3) 生成“平台极限约束” ---
    band = 0.20  # ±20%
    single_limits_platform, multi_limits_platform = make_platform_limits_from_all_levels(
        proposed_alloc, assets_list, band=band, mode="relative",
        unlock_zero=False  # 若希望给“全档为0的资产”开个小上限，可设 True 并给 zero_cap
    )

    # 打印查看（可选）
    print("平台极限单资产约束：")
    for a, (lo, hi) in zip(assets_list, single_limits_platform):
        print(f"  {a}: [{lo:.4f}, {hi:.4f}]")
    print("（组约束此处为空，如需可在 make_platform_limits_from_all_levels 内派生）")

    # --- 4) 基于平台极限约束刻画有效前沿（锚点） ---
    print("\n开始刻画【平台受限】有效前沿...")
    risk_grid, W_frontier, R_frontier, S_frontier, w_minv, w_maxr = sweep_frontier_by_risk(
        mu, Sigma, single_limits_platform, multi_limits_platform, n_grid=500
    )
    idx = np.argsort(S_frontier)
    S_sorted, R_sorted, W_sorted = S_frontier[idx], R_frontier[idx], W_frontier[idx]
    cummax_R = np.maximum.accumulate(R_sorted)
    keep = np.isclose(R_sorted, cummax_R, atol=1e-10)
    W_anchors = W_sorted[keep]
    R_anchors, S_anchors = R_sorted[keep], S_sorted[keep]
    print(f"有效前沿锚点数量: {len(W_anchors)}")

    # --- 5) 以前沿锚点为种子：小步随机游走 + POCS（可选精度）填厚前沿之下 ---
    precision_choice = None  # 可选：'0.1%'/'0.2%'/'0.5%'/None
    print(f"\n开始填充前沿之下的可行空间（precision={precision_choice}）...")
    W_below = random_walk_below_frontier(
        W_anchor=W_anchors, mu=mu, Sigma=Sigma,
        single_limits=single_limits_platform, multi_limits=multi_limits_platform,
        per_anchor=100, step=0.10, sigma_tol=1e-4, seed=123,
        precision=precision_choice
    )
    print(f"填充样本数量（量化&去重后）: {len(W_below)}")
    W_all = np.vstack([W_anchors, W_below]) if len(W_below) else W_anchors

    # --- 6) 批量计算收益与风险 ---
    print("\n批量计算绩效指标...")
    perf_df = generate_alloc_perf_batch(port_daily_returns, W_all)
    anchor_perf = generate_alloc_perf_batch(port_daily_returns, W_anchors);
    anchor_perf['is_anchor'] = True
    perf_df['is_anchor'] = False
    full_df = pd.concat([perf_df, anchor_perf], ignore_index=True).drop_duplicates()

    # --- 7) 组装悬停文本与分层 ---
    weight_cols = {f"w_{i}": assets_list[i] for i in range(len(assets_list))}
    full_df = full_df.rename(columns=weight_cols)


    def create_hover_text(row):
        s = f"年化收益率: {row['ret_annual']:.2%}<br>年化波动率: {row['vol_annual']:.2%}<br><br><b>权重</b>:<br>"
        for asset in assets_list:
            if asset in row and row[asset] > 1e-4:
                s += f"{asset}: {row[asset]:.1%}<br>"
        s += f"<br>锚点: {'是' if row.get('is_anchor', False) else '否'}"
        return s


    full_df['hover_text'] = full_df.apply(create_hover_text, axis=1)
    full_df = cal_ef2_v4_ultra_fast(full_df)
    df_anchor = full_df[full_df['is_anchor'] == True]
    df_ef = full_df[(full_df['on_ef'] == True) & (full_df['is_anchor'] == False)]
    df_fill = full_df[(full_df['on_ef'] == False) & (full_df['is_anchor'] == False)]

    scatter_data = [
        {"data": df_fill, "name": "前沿之下填充样本（平台约束）", "color": "lightblue", "size": 3, "opacity": 0.8},
        {"data": df_ef, "name": "识别出的有效前沿（平台约束）", "color": "deepskyblue", "size": 3, "opacity": 0.8},
        {"data": df_anchor, "name": "前沿锚点", "color": "crimson", "size": 5, "opacity": 0.9,
         "marker_line": dict(width=1, color='black')},
    ]

    # --- 8) C1~C6 档位：与平台约束取交集后生成“可配置空间” ---
    print("\n--- 为每个风险等级生成可配置空间（与平台约束求交） ---")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    num_points_per_level = 6000

    # 平台约束复制一份，供交集使用
    platform_limits = single_limits_platform

    for i, (risk_level, base_alloc_map) in enumerate(proposed_alloc.items()):
        print(f"  处理 {risk_level} ...")
        base_weights = np.array([base_alloc_map.get(asset, 0.0) for asset in assets_list])
        level_limits = [(max(0.0, w * (1 - band)), min(1.0, w * (1 + band))) for w in base_weights]
        # 与平台约束逐维求交
        specific_single_limits = intersect_single_limits(level_limits, platform_limits)

        # 采样（仍可选择是否添加平台 multi_limits）
        risk_level_weights = generate_constrained_portfolios(num_points_per_level,
                                                             specific_single_limits,
                                                             {})

        if len(risk_level_weights) > 0:
            perf_df_lv = generate_alloc_perf_batch(port_daily_returns, risk_level_weights)
            perf_df_lv = perf_df_lv.rename(columns=weight_cols)
            perf_df_lv['hover_text'] = perf_df_lv.apply(lambda row: create_hover_text(row), axis=1)
            scatter_data.append({
                "data": perf_df_lv, "name": f"{risk_level} 可配置空间(∩平台约束)",
                "color": colors[i % len(colors)], "size": 2, "opacity": 0.45
            })

        # 中心基准点（仅标记）
        base_perf_df = generate_alloc_perf_batch(port_daily_returns, base_weights.reshape(1, -1))
        base_perf_df = base_perf_df.rename(columns=weight_cols)
        base_perf_df['hover_text'] = base_perf_df.apply(lambda row: create_hover_text(row), axis=1)
        scatter_data.append({
            "data": base_perf_df, "name": f"{risk_level} 基准点",
            "color": colors[i % len(colors)], "size": 4, "opacity": 1.0,
            "marker_line": dict(width=1.5, color='black')
        })

    # --- 9) 画图 ---
    plot_efficient_frontier(
        scatter_points_data=scatter_data,
        title=f"平台极限约束：±{int(band * 100)}% 外包盒 | QCQP 前沿 + 随机填充（precision={precision_choice}）"
    )
