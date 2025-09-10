# -*- encoding: utf-8 -*-
"""
@File: B01_frontier_then_fill.py
@Author: Kevin-Chen
@Descriptions:
  1) 用 QCQP 逐风险扫描，刻准有效前沿（带线性约束与多资产联合约束）
  2) 以前沿锚点为种子，小步随机游走 + POCS 投影，填充前沿之下的可行空间
  3) 支持生成样本的“权重精度”可选（0.1% / 0.2% / 0.5%），并去重
  4) 批量计算绩效，并作图
"""
import numpy as np
import cvxpy as cp
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any


# =====================================================================================
# 一、通用工具 & 画图
# =====================================================================================

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
            x=df[x_col],
            y=df[y_col],
            hovertext=df[hover_text_col],
            hoverinfo='text',
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
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        legend_title="图例",
        hovermode='closest'
    )
    fig.show()


def generate_alloc_perf_batch(port_daily: np.ndarray, portfolio_allocs: np.ndarray,
                              chunk_size: int = 20000) -> pd.DataFrame:
    """
    分块计算避免 T×N 爆内存；对 log 计算做 clip 并清理 ±inf。
    """
    assert port_daily.shape[1] == portfolio_allocs.shape[1]
    T, n = port_daily.shape
    N = portfolio_allocs.shape[0]

    res_list = []
    for s in range(0, N, chunk_size):
        e = min(N, s + chunk_size)
        W = portfolio_allocs[s:e]  # [m, n]
        R = port_daily @ W.T  # [T, m]

        one_plus_R = np.clip(1.0 + R, 1e-12, None)
        port_cum = np.cumprod(one_plus_R, axis=0)  # [T, m]
        final_ret = port_cum[-1, :]

        log_total = np.log(np.clip(final_ret, 1e-12, None))
        ret_annual = (log_total / T) * 252.0

        log_daily = np.log(one_plus_R)
        vol_annual = np.std(log_daily, axis=0, ddof=1) * np.sqrt(252.0)

        df = pd.DataFrame({
            "ret_annual": ret_annual,
            "vol_annual": vol_annual,
        })
        wdf = pd.DataFrame(W, columns=[f"w_{i}" for i in range(n)])
        res_list.append(pd.concat([wdf, df], axis=1))

    out = pd.concat(res_list, axis=0, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def cal_ef2_v4_ultra_fast(data: pd.DataFrame) -> pd.DataFrame:
    """
    从散点中识别有效前沿（按收益降序，保留波动率的前缀最小值）
    """
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


# =====================================================================================
# 二、POCS 投影（盒约束 ∩ 和=1 ∩ 多资产组约束）
# =====================================================================================

def project_to_constraints_pocs(v: np.ndarray,
                                single_limits,  # list[(low, high)]
                                multi_limits: dict,  # {(tuple_idx): (low, high)}
                                max_iter=200, tol=1e-9, damping=0.9):
    """
    POCS 交替投影：盒约束、和=1、每个组的上下半空间
    """
    x = v.astype(np.float64).copy()
    n = x.size

    lows = np.array([a for a, _ in single_limits], dtype=np.float64)
    highs = np.array([b for _, b in single_limits], dtype=np.float64)

    groups = []
    for idx_tuple, (low, up) in multi_limits.items():
        idx = np.array(idx_tuple, dtype=np.int64)
        a2 = float(len(idx))  # ||a||^2
        groups.append((idx, float(low), float(up), a2))

    # 初始粗投影
    x = np.clip(x, lows, highs)
    x += (1.0 - x.sum()) / n

    for _ in range(max_iter):
        x_prev = x

        # 盒
        x = np.clip(x, lows, highs)
        # 和=1
        x += (1.0 - x.sum()) / n
        # 组
        for idx, low, up, a2 in groups:
            if a2 == 0:
                continue
            s = x[idx].sum()
            if s > up + 1e-12:
                x[idx] -= damping * (s - up) / a2
            elif s < low - 1e-12:
                x[idx] += damping * (low - s) / a2
        # 再修正一次和=1（组步会破坏总和）
        x += (1.0 - x.sum()) / n

        if np.max(np.abs(x - x_prev)) < tol:
            break

    # 末端校验
    if (x < lows - 1e-6).any() or (x > highs + 1e-6).any():
        return None
    for idx, low, up, _ in groups:
        if len(idx) == 0:
            continue
        s = x[idx].sum()
        if s < low - 1e-6 or s > up + 1e-6:
            return None
    if not np.isclose(x.sum(), 1.0, atol=1e-6):
        return None
    return x


# =====================================================================================
# 三、统计量与前沿刻画（QCQP 逐风险扫描）
# =====================================================================================

def ann_mu_sigma(log_returns: np.ndarray):
    """
    log_returns: [T, n]
    """
    mu = log_returns.mean(axis=0) * 252.0
    Sigma = np.cov(log_returns, rowvar=False, ddof=1) * 252.0
    return mu, Sigma


def port_stats(W: np.ndarray, mu: np.ndarray, Sigma: np.ndarray):
    """
    W: [m, n] 或 [n,], 返回 (ret, vol)
    """
    if W.ndim == 1:
        ret = float(W @ mu)
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
    prob.solve(solver=cp.ECOS, warm_start=True, abstol=5e-8, reltol=5e-8, feastol=5e-8, max_iters=300)
    return w.value


def sweep_frontier_by_risk(mu, Sigma, single_limits, multi_limits, n_grid=1200):
    """
    扫描风险网格 [σ_min, σ_max]，每个 σ 上最大化收益
    """
    # 端点
    w_minv = solve_min_variance(Sigma, single_limits, multi_limits)
    w_maxr = solve_max_return(mu, single_limits, multi_limits)
    _, s_min = port_stats(w_minv, mu, Sigma);
    s_min = s_min[0]
    _, s_max = port_stats(w_maxr, mu, Sigma);
    s_max = float(max(s_min, s_max))
    # 风险网格
    grid = np.linspace(s_min, s_max, n_grid)

    W = []
    w0 = w_minv
    for s in grid:
        w = solve_max_return_at_risk(mu, Sigma, s, single_limits, multi_limits, w0=w0)
        W.append(w)
        w0 = w
    W = np.asarray(W)
    R, S = port_stats(W, mu, Sigma)
    return grid, W, R, S, w_minv, w_maxr


def make_upper_envelope_fn(R: np.ndarray, S: np.ndarray):
    """
    用 (σ, μ) 的分段线性插值给出“上包络”函数 R_upper(σ)
    """
    order = np.argsort(S)
    S_sorted = S[order]
    R_sorted = R[order]

    def f(sig):
        sig = np.atleast_1d(sig)
        return np.interp(sig, S_sorted, R_sorted, left=R_sorted[0], right=R_sorted[-1])

    return f


# =====================================================================================
# 四、量化到指定精度 & 去重
# =====================================================================================

def _parse_precision(choice: str) -> float:
    """
    '0.1%' -> 0.001, '0.2%' -> 0.002, '0.5%' -> 0.005
    也允许直接传 float(0.001) 的字符串
    """
    choice = str(choice).strip()
    if choice.endswith('%'):
        val = float(choice[:-1]) / 100.0
    else:
        val = float(choice)
    # 规范到合理小数
    return float(val)


def _snap_to_grid_simplex(w: np.ndarray, step: float, single_limits) -> np.ndarray | None:
    """
    最大余数法把 w 吸附到步长 step 的网格上，同时保证和=1、单资产盒约束。
    不处理“组约束”，组约束交给外层 POCS-量化循环去修复。
    """
    n = w.size
    R = int(round(1.0 / step))  # 分辨率（总单位数）
    w = np.clip(w, 0.0, 1.0)  # 安全裁剪
    k_float = w / step
    k_floor = np.floor(k_float).astype(np.int64)
    frac = k_float - k_floor

    # 单资产上下界在格子上的整数边界
    lows = np.array([a for a, _ in single_limits], dtype=np.float64)
    highs = np.array([b for _, b in single_limits], dtype=np.float64)
    lo_units = np.ceil(lows / step - 1e-12).astype(np.int64)
    hi_units = np.floor(highs / step + 1e-12).astype(np.int64)

    k = np.clip(k_floor, lo_units, hi_units)
    diff = R - int(k.sum())

    if diff > 0:
        # 需要在容量内 +1，优先给 frac 大的
        cap = hi_units - k
        idx = np.argsort(-frac)  # 大到小
        for i in idx:
            if diff == 0:
                break
            add = min(cap[i], diff)
            if add > 0:
                k[i] += add
                diff -= add
        if diff != 0:
            return None  # 容量不足
    elif diff < 0:
        # 需要 -1，优先减 frac 小的
        cap = k - lo_units
        idx = np.argsort(frac)  # 小到大
        for i in idx:
            if diff == 0:
                break
            sub = min(cap[i], -diff)
            if sub > 0:
                k[i] -= sub
                diff += sub
        if diff != 0:
            return None  # 无法满足和=1
    # 回到权重
    wq = k.astype(np.float64) / R
    # 单资产再检一次（数值安全）
    if (wq < lows - 1e-12).any() or (wq > highs + 1e-12).any():
        return None
    if not np.isclose(wq.sum(), 1.0, atol=1e-12):
        return None
    return wq


def quantize_with_projection(w: np.ndarray, step: float,
                             single_limits, multi_limits,
                             rounds: int = 5) -> np.ndarray | None:
    """
    循环：POCS（修组约束） -> 网格量化（修和=1 & 盒约束 & 精度）
    最多 rounds 次，若失败返回 None
    """
    x = w.copy()
    for _ in range(rounds):
        # 先在连续域修组约束
        x = project_to_constraints_pocs(x, single_limits, multi_limits,
                                        max_iter=300, tol=1e-10, damping=0.9)
        if x is None:
            return None
        # 再吸附到网格
        xq = _snap_to_grid_simplex(x, step, single_limits)
        if xq is None:
            return None
        # 如果量化后几乎不变，则结束
        if np.max(np.abs(xq - x)) < step * 0.5:
            return xq
        x = xq
    # 最后一次校验
    return x


def dedup_by_grid(W: np.ndarray, step: float) -> np.ndarray:
    if W.size == 0:
        return W
    K = np.rint(W / step).astype(np.int64)
    _, idx = np.unique(K, axis=0, return_index=True)
    return W[np.sort(idx)]


# =====================================================================================
# 五、以前沿锚点为种子：小步随机游走 + POCS（可选精度）填厚前沿之下
# =====================================================================================

def random_walk_below_frontier(W_anchor: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
                               single_limits, multi_limits,
                               per_anchor: int = 30, step: float = 0.01,
                               sigma_tol: float = 1e-4, seed: int = 123,
                               precision: str | float | None = None):
    """
    对每个前沿锚点 w0，做 per_anchor 次零和小步扰动。
    若 precision 不为 None，则把样本吸附到对应精度网格并去重。
    """
    rng = np.random.default_rng(seed)
    R_anchor, S_anchor = port_stats(W_anchor, mu, Sigma)
    f_upper = make_upper_envelope_fn(R_anchor, S_anchor)

    step_grid = None
    if precision is not None:
        step_grid = _parse_precision(precision)

    collected = []

    for w0 in W_anchor:
        _, s0 = port_stats(w0, mu, Sigma);
        s0 = s0[0]
        s_bar = s0 + sigma_tol
        for _ in range(per_anchor):
            eps = rng.normal(0.0, step, size=w0.size)
            eps -= eps.mean()  # 零和扰动
            w_try = project_to_constraints_pocs(w0 + eps, single_limits, multi_limits,
                                                max_iter=200, tol=1e-9, damping=0.9)
            if w_try is None:
                continue

            # 如果需要“精度”，先量化（包含一次 POCS-量化循环保证组约束）
            if step_grid is not None:
                w_try = quantize_with_projection(w_try, step_grid, single_limits, multi_limits, rounds=5)
                if w_try is None:
                    continue

            r, s = port_stats(w_try, mu, Sigma)
            if (s[0] <= s_bar + 1e-12) and (r[0] <= f_upper(s)[0] + 1e-8):
                collected.append(w_try)

    W = np.array(collected) if collected else np.empty((0, W_anchor.shape[1]))
    # 去重（仅当有精度时按网格整数去重）
    if step_grid is not None and W.size:
        W = dedup_by_grid(W, step_grid)
    return W


# 生成服从指定约束的随机权重
def generate_constrained_portfolios(num_points: int, single_limits, multi_limits):
    """
    在给定约束下，高效生成指定数量的投资组合权重。
    """
    generated_weights = []
    lows = np.array([l for l, h in single_limits])
    highs = np.array([h for l, h in single_limits])

    while len(generated_weights) < num_points:
        proposal = np.random.uniform(lows, highs)
        adjusted_weights = project_to_constraints_pocs(proposal, single_limits, multi_limits)
        if adjusted_weights is not None:
            generated_weights.append(adjusted_weights)

    return np.array(generated_weights)


# =====================================================================================
# 六、主程序
# =====================================================================================

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

    # 用 log 收益率估计 μ, Σ（和风险最优化保持一致）
    log_r = np.log1p(port_daily_returns)
    mu, Sigma = ann_mu_sigma(log_r)

    # --- 2) 约束定义（可按需修改/添加组约束） ---
    n_assets = len(assets_list)
    single_limits = [(0.0, 1.0)] * n_assets
    multi_limits = {
        # (0,1,2): (0.3, 1.0),   # 示例：前三类合计至少 30%
        # (3,4):   (0.0, 0.7),   # 示例：权益+另类合计不超过 70%
    }

    # --- 3) 扫描风险网格，刻准有效前沿（锚点） ---
    print("开始刻画有效前沿（QCQP 逐风险扫描）...")
    risk_grid, W_frontier, R_frontier, S_frontier, w_minv, w_maxr = sweep_frontier_by_risk(
        mu, Sigma, single_limits, multi_limits, n_grid=500
    )
    # 去支配，得到真正的有效锚点（按 σ 升序保持收益前缀上包络）
    idx = np.argsort(S_frontier)
    S_sorted, R_sorted, W_sorted = S_frontier[idx], R_frontier[idx], W_frontier[idx]
    cummax_R = np.maximum.accumulate(R_sorted)
    keep = np.isclose(R_sorted, cummax_R, atol=1e-10)
    W_anchors = W_sorted[keep]
    R_anchors, S_anchors = R_sorted[keep], S_sorted[keep]
    print(f"有效前沿锚点数量: {len(W_anchors)}")

    # --- 4) 以前沿锚点为种子：小步随机游走 + POCS（可选精度）填厚前沿之下 ---
    # 选择精度：'0.1%' / '0.2%' / '0.5%' / None
    precision_choice = '0.5%'  # <- 你可以改成 '0.2%'、'0.5%' 或 None（表示不量化）
    print(f"开始填充前沿之下的可行空间（precision={precision_choice}) ...")

    W_below = random_walk_below_frontier(
        W_anchor=W_anchors, mu=mu, Sigma=Sigma,
        single_limits=single_limits, multi_limits=multi_limits,
        per_anchor=100, step=0.12, sigma_tol=1e-4, seed=123,
        precision=precision_choice
    )
    print(f"填充样本数量（量化&去重后）: {len(W_below)}")

    # 合并（把锚点也纳入样本，便于一起评估和可视化）
    W_all = np.vstack([W_anchors, W_below]) if len(W_below) else W_anchors

    # --- 5) 批量计算收益与风险（用于画图与再次识别前沿） ---
    print("批量计算绩效指标...")
    perf_df = generate_alloc_perf_batch(port_daily_returns, W_all)

    # 标记“是否为锚点”
    anchor_perf = generate_alloc_perf_batch(port_daily_returns, W_anchors)
    anchor_perf['is_anchor'] = True

    perf_df['is_anchor'] = False
    # 合并后再以 is_anchor 标色
    full_df = pd.concat([perf_df, anchor_perf], ignore_index=True).drop_duplicates()

    # 标记有效前沿（可选：对全样本做一次快速识别）
    full_df = cal_ef2_v4_ultra_fast(full_df)

    # --- 6) 准备悬停文本并作图 ---
    weight_cols = {f"w_{i}": assets_list[i] for i in range(n_assets)}
    full_df = full_df.rename(columns=weight_cols)


    def create_hover_text(row):
        s = f"年化收益率: {row['ret_annual']:.2%}<br>年化波动率: {row['vol_annual']:.2%}<br><br><b>权重</b>:<br>"
        for asset in assets_list:
            if asset in row and row[asset] > 1e-4:
                s += f"  {asset}: {row[asset]:.1%}<br>"
        s += f"<br>锚点: {'是' if row.get('is_anchor', False) else '否'}"
        return s


    full_df['hover_text'] = full_df.apply(create_hover_text, axis=1)

    # 分层：前沿锚点、前沿点（自动识别）、填充样本
    df_anchor = full_df[full_df['is_anchor'] == True]
    df_ef = full_df[(full_df['on_ef'] == True) & (full_df['is_anchor'] == False)]
    df_fill = full_df[(full_df['on_ef'] == False) & (full_df['is_anchor'] == False)]

    scatter_data = [
        {"data": df_fill, "name": "前沿之下填充样本", "color": "lightblue", "size": 3, "opacity": 0.8},
        {"data": df_ef, "name": "识别出的有效前沿", "color": "deepskyblue", "size": 3, "opacity": 0.8},
        {"data": df_anchor, "name": "前沿锚点", "color": "crimson", "size": 5, "opacity": 0.8,
         "marker_line": dict(width=1, color='black')},
    ]

    ''' C1~C6 刻画 '''
    proposed_alloc = {
        'C1': {'货币现金类': 1.0, '固定收益类': 0.0, '混合策略类': 0.0, '权益投资类': 0.0, '另类投资类': 0.0},
        'C2': {'货币现金类': 0.2, '固定收益类': 0.8, '混合策略类': 0.0, '权益投资类': 0.0, '另类投资类': 0.0},
        'C3': {'货币现金类': 0.1, '固定收益类': 0.55, '混合策略类': 0.35, '权益投资类': 0.0, '另类投资类': 0.0},
        'C4': {'货币现金类': 0.05, '固定收益类': 0.4, '混合策略类': 0.3, '权益投资类': 0.2, '另类投资类': 0.05},
        'C5': {'货币现金类': 0.05, '固定收益类': 0.2, '混合策略类': 0.25, '权益投资类': 0.4, '另类投资类': 0.1},
        'C6': {'货币现金类': 0.05, '固定收益类': 0.1, '混合策略类': 0.15, '权益投资类': 0.6, '另类投资类': 0.1}
    }
    weight_cols_map = {f'w_{j}': assets_list[j] for j in range(len(assets_list))}

    print("\n--- 正在为每个风险等级生成可配置空间 ---")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    deviation = 0.5
    num_points_per_level = 10000
    base_points_to_plot = []

    for i, (risk_level, base_alloc_map) in enumerate(proposed_alloc.items()):
        print(f"--- 正在处理: {risk_level} ---")
        base_weights = np.array([base_alloc_map.get(asset, 0.0) for asset in assets_list])
        specific_single_limits = [(max(0.0, w * (1 - deviation)), min(1.0, w * (1 + deviation))) for w in base_weights]

        risk_level_weights = generate_constrained_portfolios(num_points_per_level, specific_single_limits, {})

        if len(risk_level_weights) > 0:
            perf_df = generate_alloc_perf_batch(port_daily_returns, risk_level_weights)
            perf_df = perf_df.rename(columns=weight_cols_map)
            perf_df['hover_text'] = perf_df.apply(lambda row: create_hover_text(row), axis=1)
            scatter_data.append({
                "data": perf_df, "name": f"{risk_level} 可配置空间", "color": colors[i % len(colors)],
                "size": 2, "opacity": 0.5
            })

        # 准备中心基准点
        base_perf_df = generate_alloc_perf_batch(port_daily_returns, base_weights.reshape(1, -1))
        base_perf_df = base_perf_df.rename(columns=weight_cols_map)
        base_perf_df['hover_text'] = base_perf_df.apply(lambda row: create_hover_text(row), axis=1)
        scatter_data.append({
            "data": base_perf_df, "name": f"{risk_level} 基准点", "color": colors[i % len(colors)],
            "size": 3, "opacity": 1.0, "symbol": "star", "marker_line": dict(width=1.5, color='black')
        })

    ''' 图像画点 '''
    plot_efficient_frontier(
        scatter_points_data=scatter_data,
        title=f"QCQP 刻准有效前沿 + 小步随机游走填充（precision={precision_choice}）"
    )
