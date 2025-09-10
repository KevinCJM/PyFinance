# -*- encoding: utf-8 -*-
"""
@File: B01_random_weights.py
@Modify Time: 2025/9/10 10:30
@Author: Kevin-Chen
@Descriptions: 结合了（1）通用随机组合与有效前沿 和（2）基于建议配置生成各风险等级可配置空间的功能
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any


# ==============================================================================
# 绘图与辅助函数
# ==============================================================================
def plot_efficient_frontier(
        scatter_points_data: List[Dict[str, Any]],
        title: str = '投资组合与有效前沿',
        x_axis_title: str = '年化波动率 (Annual Volatility)',
        y_axis_title: str = '年化收益率 (Annual Return)',
        x_col: str = 'vol_annual',
        y_col: str = 'ret_annual',
        hover_text_col: str = 'hover_text'
):
    """
    使用 Plotly 绘制可定制的有效前沿和投资组合散点图。
    """
    fig = go.Figure()

    for point_set in scatter_points_data:
        df = point_set["data"]
        marker_config = dict(
            color=point_set["color"],
            size=point_set["size"],
            opacity=point_set["opacity"],
            line=point_set.get("marker_line"),
            symbol=point_set.get("symbol", "circle")
        )

        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            hovertext=df[hover_text_col],
            hoverinfo='text',
            mode='markers',
            marker=marker_config,
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


def create_hover_text(df_row, assets_list):
    text = f"年化收益率: {df_row['ret_annual']:.2%}<br>年化波动率: {df_row['vol_annual']:.2%}<br><br><b>资产权重</b>:<br>"
    for asset in assets_list:
        if asset in df_row and pd.notna(df_row[asset]) and df_row[asset] > 1e-4:
            text += f"  {asset}: {df_row[asset]:.1%}<br>"
    return text


# ==============================================================================
# 核心计算函数
# ==============================================================================

def generate_alloc_perf_batch(port_daily: np.ndarray, portfolio_allocs: np.ndarray,
                              chunk_size: int = 20000) -> pd.DataFrame:
    """
    分块计算，避免 T×N 爆内存；并且对 log 计算做 clip，清理 ±inf。
    """
    assert port_daily.shape[1] == portfolio_allocs.shape[1]
    T, n = port_daily.shape
    N = portfolio_allocs.shape[0]

    res_list = []
    for s in range(0, N, chunk_size):
        e = min(N, s + chunk_size)
        W = portfolio_allocs[s:e]  # [m, n]
        R = port_daily @ W.T  # [T, m]

        # 安全 log：避免 <= -1 的极端值
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
    # 清理 ±inf
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


# 识别出位于有效前沿上的点
def cal_ef2_v4_ultra_fast(data: pd.DataFrame) -> pd.DataFrame:
    """
    从给定的投资组合点中，高效识别出位于有效前沿上的点。
    """
    data = data.copy()
    ret_values = data['ret_annual'].values
    vol_values = data['vol_annual'].values

    sorted_idx = np.argsort(ret_values)[::-1]

    sorted_vol = vol_values[sorted_idx]
    cummin_vol = np.minimum.accumulate(sorted_vol)

    on_ef_sorted = (sorted_vol <= cummin_vol + 1e-6)  # 加上一个小的容差

    on_ef = np.zeros(len(data), dtype=bool)
    on_ef[sorted_idx] = on_ef_sorted

    data['on_ef'] = on_ef
    return data


# 约束满足函数 (老)
def primal_dual_interior_point(proposal, the_single_limits, the_multi_limits, max_iter=100):
    """
    使用迭代投影法将一个可能无效的权重修正为满足所有约束的有效权重。
    """
    num_assets = len(proposal)
    num_constraints = 2 * num_assets + 2 * len(the_multi_limits) + 2

    A = np.zeros((num_constraints, num_assets))
    b = np.zeros(num_constraints)

    idx = 0
    for i_ in range(num_assets):
        A[idx, i_] = 1
        b[idx] = the_single_limits[i_][1]
        idx += 1
        A[idx, i_] = -1
        b[idx] = -the_single_limits[i_][0]
        idx += 1

    for indices, (lower, upper) in the_multi_limits.items():
        A[idx, list(indices)] = 1
        b[idx] = upper
        idx += 1
        A[idx, list(indices)] = -1
        b[idx] = -lower
        idx += 1

    A[idx, :] = 1
    b[idx] = 1
    A[idx + 1, :] = -1
    b[idx + 1] = -1

    x = np.copy(proposal)

    for _ in range(max_iter):
        Ax_b = A.dot(x) - b
        violating = Ax_b > 1e-6  # 添加容差

        if not np.any(violating):
            return x / np.sum(x)  # 返回前最后归一化

        if not np.any(A[violating]):  # 如果违反约束的行全为0，则无法修正
            return None

        correction = np.linalg.lstsq(A[violating], Ax_b[violating], rcond=None)[0]
        x -= correction

        x = np.clip(x, a_min=[lim[0] for lim in the_single_limits], a_max=[lim[1] for lim in the_single_limits])
        x /= np.sum(x)

    return None


# 约束满足函数 (POCS/Dykstra 投影)
def project_to_constraints_pocs(v: np.ndarray,
                                single_limits,  # list[(low, high)]
                                multi_limits: dict,  # {(tuple_idx): (low, high)}
                                max_iter=200, tol=1e-9, damping=1.0):
    """
    用 POCS 交替投影到：盒约束 ∩ 总和=1 ∩ 各组半空间 之交集。
    支持多资产联合约束。damping∈(0,1] 可缓解过冲。
    """
    x = v.copy()
    n = x.size

    # 预取单资产上下界向量
    lows = np.array([a for a, _ in single_limits], dtype=np.float64)
    highs = np.array([b for _, b in single_limits], dtype=np.float64)

    # 预编译组约束（a 向量的稀疏结构：组里为 1，其他为 0）
    groups = []
    for idx_tuple, (low, up) in multi_limits.items():
        idx = np.array(idx_tuple, dtype=np.int64)
        a_norm2 = float(len(idx))  # ||a||^2 = 组大小
        groups.append((idx, float(low), float(up), a_norm2))

    # 初始：投影到盒约束+总和=1，减少后续振荡
    x = np.clip(x, lows, highs)
    s = x.sum()
    x = x + (1.0 - s) / n

    for _ in range(max_iter):
        x_prev = x

        # 1) 盒约束
        x = np.clip(x, lows, highs)

        # 2) 总和=1 的超平面投影
        s = x.sum()
        x = x + (1.0 - s) / n

        # 3) 每个组的上下半空间投影
        # 上界：a^T x ≤ up；若超出则投影：x ← x - ((a^T x - up)/||a||^2) a
        # 下界：a^T x ≥ low；若低于则投影：x ← x + ((low - a^T x)/||a||^2) a
        for idx, low, up, a_norm2 in groups:
            if a_norm2 == 0: continue
            t = x[idx].sum()
            if t > up + 1e-12:
                delta = (t - up) / a_norm2
                x[idx] -= damping * delta  # 等价于沿 -a 方向走
            elif t < low - 1e-12:
                delta = (low - t) / a_norm2
                x[idx] += damping * delta  # 等价于沿 +a 方向走

        # 再次总和=1（组投影会打破总和），保证可行
        s = x.sum()
        x = x + (1.0 - s) / n

        # 收敛判据 + 约束校验（可选更严格）
        if np.linalg.norm(x - x_prev, ord=np.inf) < tol:
            break

    # 最终一次严格校验（允许极小容差）
    if np.any(x < lows - 1e-6) or np.any(x > highs + 1e-6):
        return None
    for idx, low, up, _ in groups:
        if len(idx) == 0:
            continue
        t = x[idx].sum()
        if t < low - 1e-6 or t > up + 1e-6:
            return None
    # 总和
    if not np.isclose(x.sum(), 1.0, atol=1e-6):
        return None
    return x


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
    port_daily_returns = hist_value_r[assets_list].values

    proposed_alloc = {
        'C1': {'货币现金类': 1.0, '固定收益类': 0.0, '混合策略类': 0.0, '权益投资类': 0.0, '另类投资类': 0.0},
        'C2': {'货币现金类': 0.2, '固定收益类': 0.8, '混合策略类': 0.0, '权益投资类': 0.0, '另类投资类': 0.0},
        'C3': {'货币现金类': 0.1, '固定收益类': 0.55, '混合策略类': 0.35, '权益投资类': 0.0, '另类投资类': 0.0},
        'C4': {'货币现金类': 0.05, '固定收益类': 0.4, '混合策略类': 0.3, '权益投资类': 0.2, '另类投资类': 0.05},
        'C5': {'货币现金类': 0.05, '固定收益类': 0.2, '混合策略类': 0.25, '权益投资类': 0.4, '另类投资类': 0.1},
        'C6': {'货币现金类': 0.05, '固定收益类': 0.1, '混合策略类': 0.15, '权益投资类': 0.6, '另类投资类': 0.1}
    }

    scatter_data_to_plot = []
    weight_cols_map = {f'w_{j}': assets_list[j] for j in range(len(assets_list))}

    # --- 2. 生成通用随机组合与有效前沿 (背景) ---
    print("--- 正在生成通用随机组合与有效前沿 (背景) ---")
    general_single_limits = [(0.0, 1.0)] * len(assets_list)
    general_multi_limits = {}  # 保持原始脚本的示例多资产约束
    num_general_points = 50000

    general_weights = generate_constrained_portfolios(num_general_points,
                                                      general_single_limits, general_multi_limits)
    if len(general_weights) > 0:
        all_results_df = generate_alloc_perf_batch(port_daily_returns, general_weights)
        all_results_df = cal_ef2_v4_ultra_fast(all_results_df)

        all_results_df = all_results_df.rename(columns=weight_cols_map)
        all_results_df['hover_text'] = all_results_df.apply(lambda row: create_hover_text(row, assets_list), axis=1)

        scatter_data_to_plot.append({
            "data": all_results_df,
            "name": "随机生成组合", "color": "lightgrey", "size": 2, "opacity": 0.5
        })
        ef_df = all_results_df[all_results_df['on_ef']].copy()
        scatter_data_to_plot.append({
            "data": ef_df,
            "name": "有效前沿", "color": "#0000FF", "size": 2, "opacity": 0.5
        })

    # --- 3. 为每个风险等级生成可配置空间 (前景) ---
    print("\n--- 正在为每个风险等级生成可配置空间 ---")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    deviation = 0.20
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
            perf_df['hover_text'] = perf_df.apply(lambda row: create_hover_text(row, assets_list), axis=1)
            scatter_data_to_plot.append({
                "data": perf_df, "name": f"{risk_level} 可配置空间", "color": colors[i % len(colors)],
                "size": 2, "opacity": 0.5
            })

        # 准备中心基准点
        base_perf_df = generate_alloc_perf_batch(port_daily_returns, base_weights.reshape(1, -1))
        base_perf_df = base_perf_df.rename(columns=weight_cols_map)
        base_perf_df['hover_text'] = base_perf_df.apply(lambda row: create_hover_text(row, assets_list), axis=1)
        base_points_to_plot.append({
            "data": base_perf_df, "name": f"{risk_level} 基准点", "color": colors[i % len(colors)],
            "size": 5, "opacity": 1.0, "symbol": "star", "marker_line": dict(width=1.5, color='black')
        })

    # 将基准点添加到绘图列表的顶层
    scatter_data_to_plot.extend(base_points_to_plot)

    # --- 4. 统一可视化 ---
    if scatter_data_to_plot:
        print("\n--- 正在生成最终的组合图表 ---")
        plot_efficient_frontier(
            scatter_points_data=scatter_data_to_plot,
            title='各风险等级配置空间、有效前沿与随机组合'
        )
    else:
        print("未能生成任何有效的投资组合，无法进行绘图。")
