# -*- encoding: utf-8 -*-
"""
@File: B07_random_weights_final.py
@Author: Kevin-Chen
@Descriptions:
  1) 用 QCQP 逐风险扫描，刻准有效前沿（带线性约束与多资产联合约束）
  2) 以前沿锚点为种子，小步随机游走 + POCS 投影，填充前沿之下的可行空间
  3) 支持生成样本的“权重精度”可选（0.1% / 0.2% / 0.5%），并去重
  4) 批量计算绩效，并作图
  5) ✅ C1~C6 等级硬编码上下限；全局前沿服从等级边界极值包络
  6) ✅ 等级可配置空间：QCQP 刻前沿 → 随机游走 + POCS 填充；并在图上展示各等级前沿
"""
import numpy as np
import cvxpy as cp
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple

''' 一、通用工具 & 画图 '''


def dict_alloc_to_vector(assets: List[str], alloc_map: Dict[str, float]) -> np.ndarray:
    """
    按 assets 顺序把 {'资产名': 权重} 映射成 np.ndarray；缺省资产按 0 处理。
    """
    w = np.array([alloc_map.get(a, 0.0) for a in assets], dtype=np.float64)
    return w


def project_baseline_to_level(w_base: np.ndarray,
                              level_limits: List[Tuple[float, float]]) -> np.ndarray:
    """
    将给定基准权重投影到该等级的盒约束 ∩ sum=1；若已可行则原样返回。
    """
    # 轻度数值清洗
    w = np.clip(w_base, 0.0, 1.0).astype(np.float64)
    if not np.isclose(w.sum(), 1.0, atol=1e-12):
        # 先归一化，再投影更稳定
        s = w.sum()
        w = w / s if s > 0 else np.full_like(w, 1.0 / w.size)

    w_proj = project_to_constraints_pocs(w, level_limits, {},
                                         max_iter=500, tol=1e-12, damping=0.9)
    if w_proj is None:
        # 兜底：以中点作为起点再投影一次
        w_proj = level_midpoint_weights(level_limits)
    return w_proj


def plot_efficient_frontier(
        scatter_points_data: List[Dict[str, Any]],
        title: str = '投资组合与有效前沿',
        x_axis_title: str = '年化波动率 (Annual Volatility)',
        y_axis_title: str = '年化收益率 (Annual Return)',
        x_col: str = 'vol_annual',
        y_col: str = 'ret_annual',
        hover_text_col: str = 'hover_text',
        output_filename: str = None
):
    """
    绘制投资组合有效前沿图（增强：支持 marker.symbol）
    """
    fig = go.Figure()
    for point_set in scatter_points_data:
        df = point_set["data"]
        marker_cfg = dict(
            color=point_set["color"],
            size=point_set["size"],
            opacity=point_set["opacity"],
        )
        if "marker_line" in point_set:
            marker_cfg["line"] = point_set["marker_line"]
        if "symbol" in point_set:
            marker_cfg["symbol"] = point_set["symbol"]

        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            hovertext=df[hover_text_col],
            hoverinfo='text',
            mode='markers',
            marker=marker_cfg,
            name=point_set["name"]
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        legend_title="图例",
        hovermode='closest'
    )
    if output_filename:
        fig.write_html(output_filename)
        print(f"图表已保存到: {output_filename}")
    else:
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


''' 二、POCS 投影 (盒约束 ∩ 和=1 ∩ 多资产组约束) '''


def project_to_constraints_pocs(v: np.ndarray,
                                single_limits,  # list[(low, high)]
                                multi_limits: dict,  # {(tuple_idx): (low, high)}
                                max_iter=200, tol=1e-9, damping=0.9):
    """
    POCS 投影到：单资产盒约束 ∩ sum=1 ∩ 组约束。
    """
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
            if a2 == 0:
                continue
            s = x[idx].sum()
            if s > up + 1e-12:
                x[idx] -= damping * (s - up) / a2
            elif s < low - 1e-12:
                x[idx] += damping * (low - s) / a2

        x += (1.0 - x.sum()) / n

        if np.max(np.abs(x - x_prev)) < tol:
            break

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


''' 三、统计量与前沿刻画 (QCQP 逐风险扫描) '''


def ann_mu_sigma(log_returns: np.ndarray):
    mu = log_returns.mean(axis=0) * 252.0
    Sigma = np.cov(log_returns, rowvar=False, ddof=1) * 252.0
    return mu, Sigma


def port_stats(W: np.ndarray, mu: np.ndarray, Sigma: np.ndarray):
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
    prob.solve(solver=cp.ECOS, warm_start=True, abstol=5e-8, reltol=5e-8, feastol=5e-8, max_iters=1000)
    return w.value


def sweep_frontier_by_risk(mu, Sigma, single_limits, multi_limits, n_grid=300):
    """
    （保留你的实现）逐风险扫描得到前沿曲线
    """
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


''' 四、量化到指定精度 & 去重 '''


def _parse_precision(choice: str) -> float:
    choice = str(choice).strip()
    if choice.endswith('%'):
        val = float(choice[:-1]) / 100.0
    else:
        val = float(choice)
    return float(val)


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
            if add > 0:
                k[i] += add;
                diff -= add
        if diff != 0: return None
    elif diff < 0:
        cap = k - lo_units
        idx = np.argsort(frac)
        for i in idx:
            if diff == 0: break
            sub = min(cap[i], -diff)
            if sub > 0:
                k[i] -= sub;
                diff += sub
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
        if np.max(np.abs(xq - x)) < step * 0.5:
            return xq
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
        s0 = s0[0]
        s_bar = s0 + sigma_tol
        for _ in range(per_anchor):
            eps = rng.normal(0.0, step, size=w0.size)
            eps -= eps.mean()
            w_try = project_to_constraints_pocs(w0 + eps, single_limits, multi_limits,
                                                max_iter=200, tol=1e-9, damping=0.9)
            if w_try is None:
                continue
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


''' 六、C1~C6 等级边界：硬编码 + 全局包络 '''


def bounds_dict_to_limits(assets: List[str],
                          level_bounds: Dict[str, Dict[str, Tuple[float, float]]]) -> Dict[
    str, List[Tuple[float, float]]]:
    """
    把 {等级: {资产: (L,H)}} 规范成 {等级: [(L_j,H_j)]_j 与 assets_list 对齐}
    """
    out = {}
    for level, bmap in level_bounds.items():
        out[level] = [(bmap.get(a, (0.0, 1.0))[0], bmap.get(a, (0.0, 1.0))[1]) for a in assets]
    return out


def global_envelope_limits(per_level_limits: Dict[str, List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    """
    对每个资产维度取 L 的最小值、H 的最大值，形成全局硬边界（用于整体有效前沿）。
    """
    levels = list(per_level_limits.keys())
    n = len(per_level_limits[levels[0]])
    lows = np.min(np.array([[per_level_limits[L][j][0] for L in levels] for j in range(n)]), axis=1)
    highs = np.max(np.array([[per_level_limits[L][j][1] for L in levels] for j in range(n)]), axis=1)
    return [(float(l), float(h)) for l, h in zip(lows, highs)]


def level_midpoint_weights(limits_1d: List[Tuple[float, float]]) -> np.ndarray:
    """
    用每个维度 (L,H) 的中点构造等级“基准点”，再投影到 sum=1（与盒约束）以保证可行。
    """
    mids = np.array([(l + h) * 0.5 for (l, h) in limits_1d], dtype=np.float64)
    w0 = project_to_constraints_pocs(mids, limits_1d, {}, max_iter=500, tol=1e-12, damping=0.9)
    if w0 is None:
        w0 = mids / np.sum(mids) if np.sum(mids) > 0 else np.full_like(mids, 1.0 / len(mids))
        w0 = project_to_constraints_pocs(w0, limits_1d, {}, max_iter=500, tol=1e-12, damping=0.9)
    return w0


def compute_frontier_anchors(mu: np.ndarray, Sigma: np.ndarray,
                             single_limits, multi_limits,
                             n_grid: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    给定边界约束，刻画（等级或全局的）有效前沿锚点：
    返回 (W_anchors, R_anchors, S_anchors)
    """
    _, W_frontier, R_frontier, S_frontier, _, _ = sweep_frontier_by_risk(
        mu, Sigma, single_limits, multi_limits, n_grid=n_grid
    )
    idx = np.argsort(S_frontier)
    S_sorted, R_sorted, W_sorted = S_frontier[idx], R_frontier[idx], W_frontier[idx]
    cummax_R = np.maximum.accumulate(R_sorted)
    keep = np.isclose(R_sorted, cummax_R, atol=1e-10)
    W_anchors = W_sorted[keep]
    R_anchors, S_anchors = R_sorted[keep], S_sorted[keep]
    return W_anchors, R_anchors, S_anchors


if __name__ == '__main__':
    ''' --- 1) 数据加载与预处理 --- '''
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

    ''' --- 2) 等级边界硬编码（示例占位，请按业务改数值） --- '''
    # --- 等级“基准点” —— 由业务给定的权重映射 ---
    proposed_alloc_base = {
        'C1': {'货币现金类': 1.0, '固定收益类': 0.0, '混合策略类': 0.0, '权益投资类': 0.0, '另类投资类': 0.0},
        'C2': {'货币现金类': 0.2, '固定收益类': 0.8, '混合策略类': 0.0, '权益投资类': 0.0, '另类投资类': 0.0},
        'C3': {'货币现金类': 0.1, '固定收益类': 0.55, '混合策略类': 0.35, '权益投资类': 0.0, '另类投资类': 0.0},
        'C4': {'货币现金类': 0.05, '固定收益类': 0.4, '混合策略类': 0.3, '权益投资类': 0.2, '另类投资类': 0.05},
        'C5': {'货币现金类': 0.05, '固定收益类': 0.2, '混合策略类': 0.25, '权益投资类': 0.4, '另类投资类': 0.1},
        'C6': {'货币现金类': 0.05, '固定收益类': 0.1, '混合策略类': 0.15, '权益投资类': 0.6, '另类投资类': 0.1}
    }
    risk_level_bounds = {
        'C1': {
            '货币现金类': (1.00, 1.00),
            '固定收益类': (0.00, 0.00),
            '混合策略类': (0.00, 0.00),
            '权益投资类': (0.00, 0.00),
            '另类投资类': (0.00, 0.00),
        },
        'C2': {
            '货币现金类': (0.00, 1.00),
            '固定收益类': (0.00, 1.00),
            '混合策略类': (0.00, 0.00),
            '权益投资类': (0.00, 0.00),
            '另类投资类': (0.00, 0.00),
        },
        'C3': {
            '货币现金类': (0.00, 1.00),
            '固定收益类': (0.00, 1.00),
            '混合策略类': (0.00, 0.35 * 1.2),
            '权益投资类': (0.00, 0.00),
            '另类投资类': (0.00, 0.00),
        },
        'C4': {
            '货币现金类': (0.00, 1.00),
            '固定收益类': (0.00, 1.00),
            '混合策略类': (0.00, 0.35 * 1.2),
            '权益投资类': (0.00, 0.2 * 1.2),
            '另类投资类': (0.00, 0.05 * 1.2),
        },
        'C5': {
            '货币现金类': (0.00, 1.00),
            '固定收益类': (0.00, 1.00),
            '混合策略类': (0.00, 0.35 * 1.2),
            '权益投资类': (0.00, 0.4 * 1.2),
            '另类投资类': (0.00, 0.1 * 1.2),
        },
        'C6': {
            '货币现金类': (0.00, 1.00),
            '固定收益类': (0.00, 1.00),
            '混合策略类': (0.00, 0.35 * 1.2),
            '权益投资类': (0.00, 0.6 * 1.2),
            '另类投资类': (0.00, 0.1 * 1.2),
        },
    }

    # 规范化为与 assets_list 对齐的列表形式
    per_level_limits = bounds_dict_to_limits(assets_list, risk_level_bounds)
    # 全局有效前沿服从的极值包络
    single_limits_global = global_envelope_limits(per_level_limits)
    multi_limits_global = {
        # 你可以在此添加全局组约束
        # (assets_list.index('权益投资类'), assets_list.index('另类投资类')): (0.0, 0.70),
    }

    ''' --- 3) 全局有效前沿（服从全局极值边界） --- '''
    print("刻画【全局】有效前沿（QCQP 逐风险扫描）...")
    risk_grid, W_frontier, R_frontier, S_frontier, w_minv, w_maxr = sweep_frontier_by_risk(
        mu, Sigma, single_limits_global, multi_limits_global, n_grid=300
    )
    idx = np.argsort(S_frontier)
    S_sorted, R_sorted, W_sorted = S_frontier[idx], R_frontier[idx], W_frontier[idx]
    cummax_R = np.maximum.accumulate(R_sorted)
    keep = np.isclose(R_sorted, cummax_R, atol=1e-10)
    W_anchors_glb = W_sorted[keep]
    R_anchors_glb, S_anchors_glb = R_sorted[keep], S_sorted[keep]
    print(f"全局有效前沿锚点数量: {len(W_anchors_glb)}")

    ''' --- 4) 全局：随机游走 + POCS 填厚前沿之下区域（可选精度） --- '''
    precision_choice = None  # 可改 '0.2%'、'0.5%' 或 None
    print(f"全局：填充前沿之下的可行空间（precision={precision_choice}) ...")
    W_below_glb = random_walk_below_frontier(
        W_anchor=W_anchors_glb, mu=mu, Sigma=Sigma,
        single_limits=single_limits_global, multi_limits=multi_limits_global,
        per_anchor=100, step=0.12, sigma_tol=1e-4, seed=123,
        precision=precision_choice
    )

    ''' --- 5) 批量计算收益与风险（全局） --- '''
    perf_df_glb = generate_alloc_perf_batch(port_daily_returns,
                                            np.vstack([W_anchors_glb, W_below_glb]) if len(
                                                W_below_glb) else W_anchors_glb)
    anchor_perf_glb = generate_alloc_perf_batch(port_daily_returns, W_anchors_glb)
    anchor_perf_glb['is_anchor'] = True
    perf_df_glb['is_anchor'] = False
    full_df_glb = pd.concat([perf_df_glb, anchor_perf_glb], ignore_index=True).drop_duplicates()
    full_df_glb = cal_ef2_v4_ultra_fast(full_df_glb)

    ''' --- 6) 悬停文本（共用） --- '''
    weight_cols = {f"w_{i}": assets_list[i] for i in range(len(assets_list))}


    def create_hover_text(row):
        s = f"年化收益率: {row['ret_annual']:.2%}<br>年化波动率: {row['vol_annual']:.2%}<br><br><b>权重</b>:<br>"
        for asset in assets_list:
            if asset in row and row[asset] > 1e-4:
                s += f"  {asset}: {row[asset]:.1%}<br>"
        s += f"<br>锚点: {'是' if row.get('is_anchor', False) else '否'}"
        return s


    full_df_glb = full_df_glb.rename(columns=weight_cols)
    full_df_glb['hover_text'] = full_df_glb.apply(create_hover_text, axis=1)

    df_anchor_glb = full_df_glb[full_df_glb['is_anchor'] == True]
    df_ef_glb = full_df_glb[(full_df_glb['on_ef'] == True) & (full_df_glb['is_anchor'] == False)]
    df_fill_glb = full_df_glb[(full_df_glb['on_ef'] == False) & (full_df_glb['is_anchor'] == False)]

    scatter_data = [
        {"data": df_fill_glb, "name": "全局：前沿之下填充样本", "color": "lightblue", "size": 3, "opacity": 0.45},
        {"data": df_ef_glb, "name": "全局：识别出的有效前沿", "color": "deepskyblue", "size": 3, "opacity": 0.9},
        {"data": df_anchor_glb, "name": "全局：前沿锚点", "color": "crimson", "size": 5, "opacity": 0.9,
         "marker_line": dict(width=1, color='black')},
    ]

    ''' --- 7) 等级可配置空间：QCQP 刻前沿 → 随机游走 + POCS 填充，并展示“等级有效前沿” --- '''
    print("\n--- 为每个风险等级生成：有效前沿（QCQP） + 前沿之下填充样本 ---")
    colors = {
        'C1': '#1f77b4', 'C2': '#ff7f0e', 'C3': '#2ca02c',
        'C4': '#d62728', 'C5': '#9467bd', 'C6': '#8c564b'
    }
    weight_cols_map = {f'w_{j}': assets_list[j] for j in range(len(assets_list))}

    for level in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']:
        print(f"--- 处理等级：{level} ---")
        level_limits = per_level_limits[level]
        level_multi_limits = {}  # 如需等级专属组约束，在此填写

        # 7.1 刻画该等级的有效前沿锚点
        W_anchors_lv, R_anchors_lv, S_anchors_lv = compute_frontier_anchors(
            mu, Sigma, single_limits=level_limits, multi_limits=level_multi_limits, n_grid=200
        )
        print(f"{level} 有效前沿锚点数量: {len(W_anchors_lv)}")

        # 7.2 基于等级锚点做随机游走 + POCS（严格服从该等级边界）
        W_below_lv = random_walk_below_frontier(
            W_anchor=W_anchors_lv, mu=mu, Sigma=Sigma,
            single_limits=level_limits, multi_limits=level_multi_limits,
            per_anchor=80, step=0.08, sigma_tol=1e-4, seed=2024,
            precision=None  # 需要整数网格时可设如 '0.2%'
        )

        # 7.3 计算绩效并加入图层
        # 锚点
        perf_anchor_lv = generate_alloc_perf_batch(port_daily_returns, W_anchors_lv)
        perf_anchor_lv = perf_anchor_lv.rename(columns=weight_cols_map)
        perf_anchor_lv['hover_text'] = perf_anchor_lv.apply(lambda row: create_hover_text(row), axis=1)

        # 填充样本
        if len(W_below_lv) > 0:
            perf_fill_lv = generate_alloc_perf_batch(port_daily_returns, W_below_lv)
            perf_fill_lv = perf_fill_lv.rename(columns=weight_cols_map)
            perf_fill_lv['hover_text'] = perf_fill_lv.apply(lambda row: create_hover_text(row), axis=1)
            scatter_data.append({
                "data": perf_fill_lv, "name": f"{level} 可配置空间",
                "color": colors[level], "size": 2, "opacity": 0.35
            })

        # 等级有效前沿（锚点可视为前沿抽样）
        scatter_data.append({
            "data": perf_anchor_lv, "name": f"{level} 有效前沿",
            "color": colors[level], "size": 4, "opacity": 0.9,
            "marker_line": dict(width=1, color='black')
        })

        # 7.4 等级“基准点”（来自 proposed_alloc_base，并投影到该等级硬边界）
        if level in proposed_alloc_base:
            base_w_raw = dict_alloc_to_vector(assets_list, proposed_alloc_base[level])
        else:
            # 若某等级未给基准字典，退回到中点方案
            base_w_raw = level_midpoint_weights(level_limits)
        # 可选：检查基准点是否明显越界并提示
        sum_low = sum(l for l, h in level_limits)
        sum_high = sum(h for l, h in level_limits)
        if sum_low > 1.0 + 1e-12 or sum_high < 1.0 - 1e-12:
            print(f"[警告] {level} 的盒约束与 sum=1 可能不可行：ΣL={sum_low:.3f}, ΣH={sum_high:.3f}")
        base_w = project_baseline_to_level(base_w_raw, level_limits)

        base_perf_df = generate_alloc_perf_batch(port_daily_returns, base_w.reshape(1, -1))
        base_perf_df = base_perf_df.rename(columns=weight_cols_map)
        base_perf_df['hover_text'] = base_perf_df.apply(lambda row: create_hover_text(row), axis=1)
        scatter_data.append({
            "data": base_perf_df, "name": f"{level} 基准点",
            "color": colors[level], "size": 9, "opacity": 1.0,
            "symbol": "star", "marker_line": dict(width=1.5, color='black')
        })

    ''' --- 8) 作图 --- '''
    plot_efficient_frontier(
        scatter_points_data=scatter_data,
        title=f"全局与 C1~C6 等级：有效前沿（QCQP）+ 前沿下可行空间（随机游走+POCS）",
        # output_filename="efficient_frontier_2.html"
    )
