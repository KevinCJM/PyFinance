# -*- encoding: utf-8 -*-
"""
@File: B09_random_weights_customer_final.py
@Modify Time: 2025/9/11 18:34
@Author: Kevin-Chen
@Descriptions:
  1) 用 QCQP 逐风险扫描，刻准有效前沿（带线性约束与多资产联合约束）
  2) 以前沿锚点为种子，小步随机游走 + POCS 投影，填充前沿之下的可行空间
  3) 支持生成样本的“权重精度”可选（0.1% / 0.2% / 0.5%），并去重
  4) 批量计算绩效，并作图与 Excel 导出（全局 + C1~C6 多 Sheet）
"""
import time
import numpy as np
import cvxpy as cp
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple, Optional


# ========= 工具：加载 & 基础统计 =========

def load_returns_from_excel(
        filepath: str,
        sheet_name: str,
        assets_list: List[str],
        rename_map: Dict[str, str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """读取净值，转日收益，计算年化 μ、Σ。"""
    print(f"{str_time()} [加载数据] 读取excel文件数据...")
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index)
    df = df.dropna().sort_index(ascending=True).rename(rename_map, axis=1)
    print(f"{str_time()} [加载数据] 计算日涨跌收益率...")
    hist_value_r = df[assets_list].pct_change().dropna()
    daily_returns = hist_value_r.values
    print(f"{str_time()} [加载数据] 将日收益率转换为对数收益...")
    log_r = np.log1p(daily_returns)
    print(f"{str_time()} [加载数据] 计算年化收益与协方差...")
    miu, cov = ann_mu_sigma(log_r)
    return df, daily_returns, miu, cov


def ann_mu_sigma(log_returns: np.ndarray):
    miu = log_returns.mean(axis=0) * 252.0
    cov = np.cov(log_returns, rowvar=False, ddof=1) * 252.0
    return miu, cov


def str_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# ========= 约束 & 投影 =========

def bounds_dict_to_limits(assets: List[str],
                          level_bounds: Dict[str, Dict[str, Tuple[float, float]]]) -> Dict[
    str, List[Tuple[float, float]]]:
    out = {}
    for level, bmap in level_bounds.items():
        print(f"{str_time()} [构造边界] 处理等级:", level)
        out[level] = [(bmap.get(a, (0.0, 1.0))[0], bmap.get(a, (0.0, 1.0))[1]) for a in assets]
    return out


def global_envelope_limits(per_level_limits: Dict[str, List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    print(f"{str_time()} [构造边界] 处理全局边界")
    levels = list(per_level_limits.keys())
    n = len(per_level_limits[levels[0]])
    lows = np.min(np.array([[per_level_limits[L][j][0] for L in levels] for j in range(n)]), axis=1)
    highs = np.max(np.array([[per_level_limits[L][j][1] for L in levels] for j in range(n)]), axis=1)
    return [(float(l), float(h)) for l, h in zip(lows, highs)]


def project_to_constraints_pocs(v: np.ndarray,
                                single_limits,  # list[(low, high)]
                                multi_limits: dict,  # {(tuple_idx): (low, high)}
                                max_iter=200, tol=1e-9, damping=0.9):
    """
    使用 POCS（Projection Onto Convex Sets）方法将向量投影到多个约束集合的交集上。

    该函数处理以下三类约束：
    1. 每个变量的上下界约束（盒约束）；
    2. 所有变量之和等于 1 的约束；
    3. 多个子集的和的上下界约束（组约束）。

    参数：
        v : np.ndarray
            初始向量，将被投影到满足所有约束的空间中。
        single_limits : list of tuple (low, high)
            每个元素表示对应位置变量的上下界限制。
        multi_limits : dict {tuple of int : (low, high)}
            键为索引元组，表示一个子集；值为该子集和的上下界。
        max_iter : int, optional
            最大迭代次数，默认为 200。
        tol : float, optional
            收敛判断的容忍度，当变量变化小于该值时停止迭代，默认为 1e-9。
        damping : float, optional
            阻尼系数，用于控制组约束投影步长的衰减，默认为 0.9。

    返回：
        np.ndarray or None
            如果成功找到满足所有约束的解，则返回投影后的向量；
            否则返回 None。
    """
    x = v.astype(np.float64).copy()
    n = x.size

    # 提取单变量上下界
    lows = np.array([a for a, _ in single_limits], dtype=np.float64)
    highs = np.array([b for _, b in single_limits], dtype=np.float64)

    # 解析组约束信息
    groups = []
    for idx_tuple, (low, up) in multi_limits.items():
        idx = np.array(idx_tuple, dtype=np.int64)
        a2 = float(len(idx))
        groups.append((idx, float(low), float(up), a2))

    # 初始投影：裁剪至单变量边界并调整总和为1
    x = np.clip(x, lows, highs)
    x += (1.0 - x.sum()) / n

    # 迭代进行 POCS 投影
    for _ in range(max_iter):
        x_prev = x
        # 投影到单变量边界
        x = np.clip(x, lows, highs)
        # 投影到总和为1的约束
        x += (1.0 - x.sum()) / n
        # 投影到各组约束
        for idx, low, up, a2 in groups:
            s = x[idx].sum()
            if s > up + 1e-12:
                x[idx] -= damping * (s - up) / a2
            elif s < low - 1e-12:
                x[idx] += damping * (low - s) / a2
        # 再次投影到总和为1的约束
        x += (1.0 - x.sum()) / n
        # 判断是否收敛
        if np.max(np.abs(x - x_prev)) < tol:
            break

    # 验证最终结果是否满足所有约束
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


def level_midpoint_weights(limits_1d: List[Tuple[float, float]]) -> np.ndarray:
    """
    计算水平中点权重，通过投影到约束集合来获得满足约束条件的权重分布。

    参数:
        limits_1d: 一维限制范围列表，每个元素为(l, h)元组，表示下限和上限

    返回:
        np.ndarray: 满足约束条件的权重数组
    """
    # 计算每个限制区间的中点值
    mids = np.array([(l + h) * 0.5 for (l, h) in limits_1d], dtype=np.float64)

    # 将中点值投影到约束集合中，获取满足约束的权重
    w0 = project_to_constraints_pocs(mids, limits_1d, {}, max_iter=500, tol=1e-12, damping=0.9)

    # 如果投影失败，则进行归一化处理后重新投影
    if w0 is None:
        w0 = mids / np.sum(mids) if np.sum(mids) > 0 else np.full_like(mids, 1.0 / len(mids))
        w0 = project_to_constraints_pocs(w0, limits_1d, {}, max_iter=500, tol=1e-12, damping=0.9)
    return w0


def project_baseline_to_level(w_base: np.ndarray, level_limits: List[Tuple[float, float]]) -> np.ndarray:
    """
    将基础权重向量投影到指定级别的约束空间中

    该函数首先对输入的权重进行归一化处理，确保其在[0,1]范围内且和为1，
    然后使用POCS算法将权重投影到给定的级别约束空间中。

    参数:
        w_base: 基础权重向量，形状为(n,)的numpy数组
        level_limits: 级别约束限制列表，每个元素为(min_limit, max_limit)元组，
                     表示对应权重的取值范围

    返回:
        投影后的权重向量，形状为(n,)的numpy数组，满足约束条件
    """
    # 对基础权重进行裁剪和归一化处理
    w = np.clip(w_base, 0.0, 1.0).astype(np.float64)
    if not np.isclose(w.sum(), 1.0, atol=1e-12):
        s = w.sum()
        w = w / s if s > 0 else np.full_like(w, 1.0 / w.size)

    # 使用POCS算法将权重投影到约束空间
    w_proj = project_to_constraints_pocs(w, level_limits, {}, max_iter=500, tol=1e-12, damping=0.9)

    # 如果投影失败，则使用级别中点权重作为备选方案
    if w_proj is None:
        w_proj = level_midpoint_weights(level_limits)

    return w_proj


# ========= 前沿刻画（QCQP） =========

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


def sweep_frontier_by_risk(mu, Sigma, single_limits, multi_limits, n_grid=300):
    """逐风险扫描得到前沿曲线。"""
    w_minv = solve_min_variance(Sigma, single_limits, multi_limits)
    w_maxr = solve_max_return(mu, single_limits, multi_limits)
    _, s_min = port_stats(w_minv, mu, Sigma)
    s_min = s_min[0]
    _, s_max = port_stats(w_maxr, mu, Sigma)
    s_max = np.maximum(s_min, s_max).item()

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


def compute_frontier_anchors(mu: np.ndarray, Sigma: np.ndarray,
                             single_limits, multi_limits,
                             n_grid: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (W_anchors, R_anchors, S_anchors)。"""
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


# ========= 网格量化 & 去重 =========

def _parse_precision(choice: str | float) -> float:
    choice = str(choice).strip()
    if choice.endswith('%'):
        val = float(choice[:-1]) / 100.0
    else:
        val = float(choice)
    return float(val)


def _snap_to_grid_simplex(w: np.ndarray, step: float, single_limits) -> Optional[np.ndarray]:
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
                             rounds: int = 5) -> Optional[np.ndarray]:
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


def quantize_df_for_export(df_in: pd.DataFrame,
                           assets: List[str],
                           step: float,
                           single_limits,
                           multi_limits,
                           port_daily: np.ndarray) -> pd.DataFrame:
    """
    对输入的资产权重进行网格量化处理，并重新计算绩效指标和前沿识别。

    该函数首先对输入 DataFrame 中的资产列执行 POCS 投影与网格量化操作，
    包括边界约束、 simplex 约束（权重和为1）以及组约束。若量化失败，则回退到简单投影和网格化策略。
    随后基于量化后的权重重新计算投资组合的绩效指标（如年化收益、波动率等），
    并重新进行前沿识别。最终返回一个结构与输入一致但内容经过量化处理的 DataFrame。

    参数:
        df_in (pd.DataFrame): 输入的包含资产权重及相关绩效列的 DataFrame。
        assets (List[str]): 资产名称列表，用于指定需要处理的列。
        step (float): 网格量化的步长。
        single_limits: 单资产约束条件（如上下限）。
        multi_limits: 多资产组约束条件（如组权重范围）。
        port_daily (np.ndarray): 日度收益矩阵，用于绩效计算。

    返回:
        pd.DataFrame: 经过量化处理后的 DataFrame，保留原有附加列并更新权重及绩效相关字段。
    """
    W = df_in[assets].to_numpy(dtype=np.float64)
    Wq = []
    for w in W:
        # 尝试使用带投影的量化方法进行处理
        wq = quantize_with_projection(w, step, single_limits, multi_limits, rounds=5)
        if wq is None:
            # 若失败则先投影到约束空间再进行简单网格化
            w_proj = project_to_constraints_pocs(w, single_limits, multi_limits)
            wq = _snap_to_grid_simplex(w_proj, step, single_limits) if w_proj is not None else w
        Wq.append(wq)
    Wq = np.vstack(Wq)

    # 基于量化后的权重批量生成新的绩效数据
    perf_q = generate_alloc_perf_batch(port_daily, Wq)
    rename_map = {f"w_{i}": assets[i] for i in range(len(assets))}
    perf_q = perf_q.rename(columns=rename_map)

    # 保留非资产权重和绩效相关的列
    keep_cols = [c for c in df_in.columns if c not in (assets + ['ret_annual', 'vol_annual', 'on_ef'])]
    out = pd.concat([perf_q[assets + ['ret_annual', 'vol_annual']], df_in[keep_cols].reset_index(drop=True)], axis=1)

    # 重新进行前沿识别
    out = cal_ef2_v4_ultra_fast(out)

    # 对极小权重置零以提升数值稳定性
    eps = step * 1e-6
    for a in assets:
        col = out[a].to_numpy()
        col[np.abs(col) < eps] = 0.0
        out[a] = col
    return out


# ========= 绩效批量、前沿识别、作图 =========

def generate_alloc_perf_batch(port_daily: np.ndarray, portfolio_allocs: np.ndarray,
                              chunk_size: int = 20000) -> pd.DataFrame:
    T, n = port_daily.shape
    assert n == portfolio_allocs.shape[1]
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


def plot_efficient_frontier(
        scatter_points_data: List[Dict[str, Any]],
        title: str = '投资组合与有效前沿',
        x_axis_title: str = '年化波动率 (Annual Volatility)',
        y_axis_title: str = '年化收益率 (Annual Return)',
        x_col: str = 'vol_annual',
        y_col: str = 'ret_annual',
        hover_text_col: str = 'hover_text',
        output_filename: Optional[str] = None
):
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
        print(f"{str_time()} 图表已保存到: {output_filename}")
    else:
        fig.show()


# ========= 其他小工具 =========

def dict_alloc_to_vector(assets: List[str], alloc_map: Dict[str, float]) -> np.ndarray:
    return np.array([alloc_map.get(a, 0.0) for a in assets], dtype=np.float64)


def make_upper_envelope_fn(R: np.ndarray, S: np.ndarray):
    order = np.argsort(S)
    S_sorted = S[order];
    R_sorted = R[order]

    def f(sig):
        sig = np.atleast_1d(sig)
        return np.interp(sig, S_sorted, R_sorted, left=R_sorted[0], right=R_sorted[-1])

    return f


def create_hover_text_factory(assets_list: List[str]):
    def _create(row: pd.Series) -> str:
        s = f"年化收益率: {row['ret_annual']:.2%}<br>年化波动率: {row['vol_annual']:.2%}<br><br><b>权重</b>:<br>"
        for asset in assets_list:
            if asset in row and row[asset] > 1e-4:
                s += f"  {asset}: {row[asset]:.1%}<br>"
        s += f"<br>锚点: {'是' if row.get('is_anchor', False) else '否'}"
        return s

    return _create


def build_export_view(df_named: pd.DataFrame, assets: List[str]) -> pd.DataFrame:
    """
    构建用于导出的视图DataFrame

    该函数从输入的DataFrame中提取指定的资产列及相关指标，进行列名重命名、排序等处理，
    生成适合导出的标准化格式DataFrame。

    参数:
        df_named (pd.DataFrame): 包含资产配置和风险收益指标的原始数据框
        assets (List[str]): 需要包含在导出视图中的资产类别列名列表

    返回:
        pd.DataFrame: 处理后的导出视图数据框，包含重命名的资产比例列、年化收益率、
                     年化波动率和有效前沿点标识，并按波动率和收益率排序

    异常:
        ValueError: 当输入DataFrame缺少必要的列时抛出
    """
    # 定义需要的列并检查是否存在缺失列
    need_cols = assets + ['ret_annual', 'vol_annual', 'on_ef']
    miss = [c for c in need_cols if c not in df_named.columns]
    if miss:
        raise ValueError(f"导出缺少必要列: {miss}")

    # 提取需要的列并创建副本
    out = df_named[need_cols].copy()

    # 定义资产列名的重命名映射
    rename_assets = {
        '货币现金类': '货币现金类比例',
        '固定收益类': '固定收益类比例',
        '混合策略类': '混合策略类比例',
        '权益投资类': '权益投资类比例',
        '另类投资类': '另类投资类比例',
    }

    # 重命名列名，将资产列名和指标列名统一转换为中文
    out = out.rename(columns=rename_assets | {
        'ret_annual': '年化收益率',
        'vol_annual': '年化波动率',
        'on_ef': '是否有效前沿点',
    })

    # 按年化波动率升序、年化收益率降序进行排序，并重置索引
    out = out.sort_values(by=['年化波动率', '年化收益率'], ascending=[True, False], kind='mergesort').reset_index(
        drop=True)
    return out


# ========= 随机游走（前沿之下填充） =========

def random_walk_below_frontier(W_anchor: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
                               single_limits, multi_limits,
                               per_anchor: int = 30, step: float = 0.01,
                               sigma_tol: float = 1e-4, seed: int = 123,
                               precision: str | float | None = None):
    """
    通过在有效前沿下方进行随机游走采样投资组合权重。

    该函数从给定的锚点集合出发，在满足约束条件下进行随机扰动，生成新的投资组合权重，
    并保留那些位于有效前沿下方 (包括前沿上)。

    参数:
        W_anchor (np.ndarray): 形状为 (n_anchor, n_assets) 的锚点权重矩阵。
        mu (np.ndarray): 资产预期收益向量，形状为 (n_assets,)。
        Sigma (np.ndarray): 资产协方差矩阵，形状为 (n_assets, n_assets)。
        single_limits: 单个资产约束条件，用于投影函数。
        multi_limits: 多资产联合约束条件，用于投影函数。
        per_anchor (int): 每个锚点尝试生成的样本数量，默认为 30。
        step (float): 随机步长的标准差，默认为 0.01。
        sigma_tol (float): 允许超出锚点风险的容忍度，默认为 1e-4。
        seed (int): 随机数种子，默认为 123。
        precision (str | float | None): 权重精度设置，用于量化权重网格，默认为 None。

    返回:
        np.ndarray: 符合条件的投资组合权重矩阵，形状为 (n_samples, n_assets)。
    """
    rng = np.random.default_rng(seed)
    # 计算锚点组合的收益和标准差，并构建上包络函数
    R_anchor, S_anchor = port_stats(W_anchor, mu, Sigma)
    f_upper = make_upper_envelope_fn(R_anchor, S_anchor)

    step_grid = None
    if precision is not None:
        # 解析精度参数，生成步长网格
        step_grid = _parse_precision(precision)

    collected = []
    # 对每个锚点执行随机游走采样
    for w0 in W_anchor:
        _, s0 = port_stats(w0, mu, Sigma)
        s0 = s0[0]
        s_bar = s0 + sigma_tol
        for _ in range(per_anchor):
            # 生成零均值正态扰动
            eps = rng.normal(0.0, step, size=w0.size)
            eps -= eps.mean()
            # 将扰动后的权重投影到约束空间内
            w_try = project_to_constraints_pocs(w0 + eps, single_limits, multi_limits,
                                                max_iter=200, tol=1e-9, damping=0.9)
            if w_try is None:
                continue
            if step_grid is not None:
                # 如果设置了精度，则对权重进行量化处理
                w_try = quantize_with_projection(w_try, step_grid, single_limits, multi_limits, rounds=5)
                if w_try is None:
                    continue
            # 计算尝试权重的风险与收益
            r, s = port_stats(w_try, mu, Sigma)
            # 判断是否在有效前沿下方且风险未显著增加
            if (s[0] <= s_bar + 1e-12) and (r[0] <= f_upper(s)[0] + 1e-8):
                collected.append(w_try)

    # 构造最终输出矩阵并去重
    W = np.array(collected) if collected else np.empty((0, W_anchor.shape[1]))
    if step_grid is not None and W.size:
        W = dedup_by_grid(W, step_grid)
    return W


# ========= 核心流程（全局 & 等级） =========

def run_global_layer(cfg: Dict[str, Any],
                     assets_list: List[str],
                     mu: np.ndarray, Sigma: np.ndarray,
                     port_daily_returns: np.ndarray,
                     single_limits_global, multi_limits_global) -> tuple[
    pd.DataFrame, List[Dict[str, Any]], pd.DataFrame]:
    """计算全局层：前沿、填充、作图数据与导出 DataFrame（未转中文列名）。"""
    # 前沿（锚点）
    print(f"{str_time()} [资产配置全局层] 计算全局有效前沿锚点...")
    _, W_frontier, R_frontier, S_frontier, *_ = sweep_frontier_by_risk(
        mu, Sigma, single_limits_global, multi_limits_global, n_grid=cfg['n_grid']
    )
    idx = np.argsort(S_frontier)
    S_sorted, R_sorted, W_sorted = S_frontier[idx], R_frontier[idx], W_frontier[idx]
    keep = np.isclose(R_sorted, np.maximum.accumulate(R_sorted), atol=1e-10)
    W_anchors = W_sorted[keep]

    # 填充
    print(f"{str_time()} [资产配置全局层] 随机游走填充前沿之下的可行空间...")
    W_below = random_walk_below_frontier(
        W_anchor=W_anchors, mu=mu, Sigma=Sigma,
        single_limits=single_limits_global, multi_limits=multi_limits_global,
        per_anchor=cfg['per_anchor'], step=cfg['step_rw'],
        sigma_tol=cfg['sigma_tol'], seed=cfg['seed'],
        precision=cfg['precision_choice']
    )

    # 绩效
    print(f"{str_time()} [资产配置全局层] 批量计算绩效指标与识别前沿...")
    perf_df = generate_alloc_perf_batch(
        port_daily_returns, np.vstack([W_anchors, W_below]) if len(W_below) else W_anchors
    )
    # 单独给锚点计算绩效指标
    anchor_perf = generate_alloc_perf_batch(port_daily_returns, W_anchors)
    anchor_perf['is_anchor'] = True
    perf_df['is_anchor'] = False
    full_df = pd.concat([perf_df, anchor_perf], ignore_index=True).drop_duplicates()
    full_df = cal_ef2_v4_ultra_fast(full_df)

    # 作图数据准备
    print(f"{str_time()} [资产配置全局层] 作图数据准备...")
    weight_cols = {f"w_{i}": assets_list[i] for i in range(len(assets_list))}
    full_df_named = full_df.rename(columns=weight_cols)
    create_hover_text = create_hover_text_factory(assets_list)
    full_df_named['hover_text'] = full_df_named.apply(create_hover_text, axis=1)

    df_anchor = full_df_named[full_df_named['is_anchor'] == True]
    df_ef = full_df_named[(full_df_named['on_ef'] == True) & (full_df_named['is_anchor'] == False)]
    df_fill = full_df_named[(full_df_named['on_ef'] == False) & (full_df_named['is_anchor'] == False)]

    scatter_data = [
        {"data": df_fill, "name": "全局：前沿之下填充样本", "color": "lightblue", "size": 3, "opacity": 0.45},
        {"data": df_ef, "name": "全局：识别出的有效前沿", "color": "deepskyblue", "size": 3, "opacity": 0.9},
        {"data": df_anchor, "name": "全局：前沿锚点", "color": "crimson", "size": 5, "opacity": 0.9,
         "marker_line": dict(width=1, color='black')},
    ]
    return full_df_named, scatter_data, df_anchor


def run_level_layer(level: str,
                    cfg_lv: Dict[str, Any],
                    limits_lv,
                    multi_limits_lv,
                    assets_list: List[str],
                    mu: np.ndarray, Sigma: np.ndarray,
                    port_daily_returns: np.ndarray,
                    base_alloc_map: Dict[str, Dict[str, float]]) -> tuple[List[Dict[str, Any]], pd.DataFrame]:
    """计算单个等级：作图数据与导出 DataFrame（中文列名）。"""
    colors = cfg_lv['color']
    create_hover_text = create_hover_text_factory(assets_list)
    weight_cols_map = {f'w_{j}': assets_list[j] for j in range(len(assets_list))}

    # 1) 前沿锚点
    print(f"{str_time()} [资产配置等级层] 处理等级: {level}  计算有效前沿锚点...")
    W_anchors_lv, _, _ = compute_frontier_anchors(
        mu, Sigma, single_limits=limits_lv, multi_limits=multi_limits_lv, n_grid=cfg_lv['n_grid']
    )

    # 2) 随机游走 + POCS +（可选）精度
    print(f"{str_time()} [资产配置等级层] 处理等级: {level}  填充前沿下的可配置区域...")
    W_below_lv = random_walk_below_frontier(
        W_anchor=W_anchors_lv, mu=mu, Sigma=Sigma,
        single_limits=limits_lv, multi_limits=multi_limits_lv,
        per_anchor=cfg_lv['per_anchor'], step=cfg_lv['step_rw'],
        sigma_tol=cfg_lv['sigma_tol'], seed=cfg_lv['seed'],
        precision=cfg_lv['precision_choice']
    )
    if cfg_lv['precision_choice'] is not None and len(W_below_lv):
        W_below_lv = dedup_by_grid(W_below_lv, _parse_precision(cfg_lv['precision_choice']))

    # 3) 绩效
    print(f"{str_time()} [资产配置等级层] 处理等级: {level}  计算绩效指标与识别前沿...")
    perf_anchor_lv = generate_alloc_perf_batch(port_daily_returns, W_anchors_lv).rename(columns=weight_cols_map)
    perf_anchor_lv['hover_text'] = perf_anchor_lv.apply(create_hover_text, axis=1)

    scatter_data = [{
        "data": perf_anchor_lv, "name": f"{level} 有效前沿",
        "color": colors, "size": 4, "opacity": 0.9, "marker_line": dict(width=1, color='black')
    }]

    if len(W_below_lv) > 0:
        perf_fill_lv = generate_alloc_perf_batch(port_daily_returns, W_below_lv).rename(columns=weight_cols_map)
        perf_fill_lv['hover_text'] = perf_fill_lv.apply(create_hover_text, axis=1)
        scatter_data.append({
            "data": perf_fill_lv, "name": f"{level} 可配置空间",
            "color": colors, "size": 2, "opacity": 0.35
        })
    else:
        perf_fill_lv = pd.DataFrame(columns=list(perf_anchor_lv.columns))

    # 4) 基准点
    if level in base_alloc_map:
        base_w_raw = dict_alloc_to_vector(assets_list, base_alloc_map[level])
    else:
        base_w_raw = level_midpoint_weights(limits_lv)
    base_w = project_baseline_to_level(base_w_raw, limits_lv)
    base_perf_df = generate_alloc_perf_batch(port_daily_returns, base_w.reshape(1, -1)).rename(columns=weight_cols_map)
    base_perf_df['hover_text'] = base_perf_df.apply(create_hover_text, axis=1)
    scatter_data.append({
        "data": base_perf_df, "name": f"{level} 基准点",
        "color": colors, "size": 9, "opacity": 1.0,
        "symbol": "star", "marker_line": dict(width=1.5, color='black')
    })

    # 5) 导出：合并→（若设精度）量化→重算绩效→识别前沿→去重
    frames = [perf_anchor_lv.assign(is_anchor=True)]
    if len(perf_fill_lv):
        frames.append(perf_fill_lv.assign(is_anchor=False))
    level_full_df = pd.concat(frames, ignore_index=True)

    if cfg_lv['precision_choice'] is not None:
        step_val = _parse_precision(cfg_lv['precision_choice'])
        level_full_df = quantize_df_for_export(
            df_in=level_full_df, assets=assets_list, step=step_val,
            single_limits=limits_lv, multi_limits=multi_limits_lv, port_daily=port_daily_returns
        )
    else:
        level_full_df = cal_ef2_v4_ultra_fast(level_full_df)

    level_full_df = level_full_df.drop_duplicates(subset=assets_list, keep='first').reset_index(drop=True)
    export_df = build_export_view(level_full_df, assets_list)

    return scatter_data, export_df


# ========= 主程序 =========

if __name__ == '__main__':
    # ---- 统一配置（可按需调整）----
    CONFIG = {
        # 1) 数据
        "input_excel": "历史净值数据.xlsx",
        "sheet_name": "历史净值数据",
        "rename_map": {
            "货基指数": "货币现金类", "固收类": "固定收益类", "混合类": "混合策略类",
            "权益类": "权益投资类", "另类": "另类投资类",
            "安逸型": "C1", "谨慎型": "C2", "稳健型": "C3",
            "增长型": "C4", "进取型": "C5", "激进型": "C6"
        },
        "assets_list": ['货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类'],

        # 2) 等级边界（示例；请按业务改数值）
        "proposed_alloc_base": {
            'C1': {'货币现金类': 1.0, '固定收益类': 0.0, '混合策略类': 0.0, '权益投资类': 0.0, '另类投资类': 0.0},
            'C2': {'货币现金类': 0.2, '固定收益类': 0.8, '混合策略类': 0.0, '权益投资类': 0.0, '另类投资类': 0.0},
            'C3': {'货币现金类': 0.1, '固定收益类': 0.55, '混合策略类': 0.35, '权益投资类': 0.0, '另类投资类': 0.0},
            'C4': {'货币现金类': 0.05, '固定收益类': 0.4, '混合策略类': 0.3, '权益投资类': 0.2, '另类投资类': 0.05},
            'C5': {'货币现金类': 0.05, '固定收益类': 0.2, '混合策略类': 0.25, '权益投资类': 0.4, '另类投资类': 0.1},
            'C6': {'货币现金类': 0.05, '固定收益类': 0.1, '混合策略类': 0.15, '权益投资类': 0.6, '另类投资类': 0.1}
        },
        "risk_level_bounds": {
            'C1': {'货币现金类': (1.00, 1.00), '固定收益类': (0.00, 0.00), '混合策略类': (0.00, 0.00),
                   '权益投资类': (0.00, 0.00), '另类投资类': (0.00, 0.00)},
            'C2': {'货币现金类': (0.00, 1.00), '固定收益类': (0.00, 1.00), '混合策略类': (0.00, 0.00),
                   '权益投资类': (0.00, 0.00), '另类投资类': (0.00, 0.00)},
            'C3': {'货币现金类': (0.00, 1.00), '固定收益类': (0.00, 1.00), '混合策略类': (0.00, 0.35 * 1.2),
                   '权益投资类': (0.00, 0.00), '另类投资类': (0.00, 0.00)},
            'C4': {'货币现金类': (0.00, 1.00), '固定收益类': (0.00, 1.00), '混合策略类': (0.00, 0.35 * 1.2),
                   '权益投资类': (0.00, 0.2 * 1.2), '另类投资类': (0.00, 0.05 * 1.2)},
            'C5': {'货币现金类': (0.00, 1.00), '固定收益类': (0.00, 1.00), '混合策略类': (0.00, 0.35 * 1.2),
                   '权益投资类': (0.00, 0.4 * 1.2), '另类投资类': (0.00, 0.1 * 1.2)},
            'C6': {'货币现金类': (0.00, 1.00), '固定收益类': (0.00, 1.00), '混合策略类': (0.00, 0.35 * 1.2),
                   '权益投资类': (0.00, 0.6 * 1.2), '另类投资类': (0.00, 0.1 * 1.2)},
        },

        # 3) 全局层参数
        "global_layer": {
            "n_grid": 300,  # 全局层网格点数量
            "per_anchor": 100,  # 每个锚点的采样数量
            "step_rw": 0.12,  # 随机游走步长
            "sigma_tol": 1e-4,  # 标准差容忍度阈值
            "seed": 123,  # 随机数生成器种子
            "precision_choice": "0.2%",  # 精度选择参数
        },
        "global_multi_limits": {  # 例： (assets.index('权益投资类'), assets.index('另类投资类')): (0.0, 0.70),
        },

        # 4) 等级层公共默认参数（各等级可覆盖）
        "level_defaults": {
            "n_grid": 200,  # 网格点数量，默认值为200
            "per_anchor": 80,  # 每个锚点的数量，默认值为80
            "step_rw": 0.08,  # 随机游走步长，默认值为0.08
            "sigma_tol": 1e-4,  # sigma容忍度阈值，默认值为1e-4
            "seed": 2024,  # 随机种子
            "precision_choice": "0.2%",  # 精度选择选项, 支持 "0.1%", "0.2%", "0.5%", None
        },
        # 每个等级的颜色（作图用）
        "level_colors": {
            'C1': '#1f77b4', 'C2': '#ff7f0e', 'C3': '#2ca02c',
            'C4': '#d62728', 'C5': '#9467bd', 'C6': '#8c564b'
        },

        # 5) 输出
        "plot_output_html": None,  # 如: "efficient_frontier.html"
        "export_excel": "前沿与等级可配置空间导出.xlsx",
    }
    s_t = time.time()

    # ---- 运行：加载数据 ----
    print(f"{str_time()} [加载数据] ... ")
    assets_list = CONFIG["assets_list"]
    hist_df, port_daily_returns, mu, Sigma = load_returns_from_excel(
        CONFIG["input_excel"], CONFIG["sheet_name"], assets_list, CONFIG["rename_map"]
    )

    # ---- 构造边界 ----
    print(f"{str_time()} [构造边界] ... ")
    per_level_limits = bounds_dict_to_limits(assets_list, CONFIG["risk_level_bounds"])
    single_limits_global = global_envelope_limits(per_level_limits)
    multi_limits_global = CONFIG["global_multi_limits"]

    # ---- 资产配置全局层 ----
    print(f"{str_time()} [资产配置全局层]")
    glb_cfg = CONFIG["global_layer"] | {"n_grid": CONFIG["global_layer"]["n_grid"]}
    full_df_glb, scatter_data, df_anchor_glb = run_global_layer(
        glb_cfg, assets_list, mu, Sigma,
        port_daily_returns, single_limits_global, multi_limits_global
    )
    print(f"{str_time()} [资产配置全局层] 量化与导出视图构建...")
    export_sheets: Dict[str, pd.DataFrame] = {}
    quantize_df = quantize_df_for_export(full_df_glb, assets_list, _parse_precision(glb_cfg["precision_choice"]),
                                         single_limits_global, multi_limits_global, port_daily_returns)
    export_sheets['全局'] = build_export_view(
        quantize_df if glb_cfg["precision_choice"] is not None else build_export_view(full_df_glb, assets_list),
        assets_list
    )
    # ---- 资产配置等级层 ----
    print(f"{str_time()} [资产配置等级层]")
    level_defaults = CONFIG["level_defaults"]
    for level in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']:
        print(f"{str_time()} [资产配置等级层] 处理等级:", level)
        lv_cfg = {
            **level_defaults,
            "color": CONFIG["level_colors"][level],
        }
        limits_lv = per_level_limits[level]
        multi_limits_lv = {}  # 如需等级专属组约束，在此填写
        lv_scatter, lv_export_df = run_level_layer(
            level, lv_cfg, limits_lv, multi_limits_lv,
            assets_list, mu, Sigma, port_daily_returns,
            CONFIG["proposed_alloc_base"]
        )
        scatter_data.extend(lv_scatter)
        export_sheets[level] = lv_export_df

    # ---- 作图 ----
    print(f"{str_time()} [作图]")
    plot_efficient_frontier(
        scatter_points_data=scatter_data,
        title="全局与 C1~C6 等级：有效前沿（QCQP）+ 前沿下可行空间（随机游走+POCS）",
        output_filename=CONFIG["plot_output_html"]
    )

    # ---- 导出 Excel ----
    print(f"{str_time()} [导出 Excel]")
    with pd.ExcelWriter(CONFIG["export_excel"]) as writer:
        for sheet_name, df_out in export_sheets.items():
            safe_name = sheet_name[:31]
            df_out.to_excel(writer, sheet_name=safe_name, index=False)
    print(f"{str_time()} [完成] ✅ 耗时 {time.time() - s_t:.2f}s")
