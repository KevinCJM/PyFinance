# -*- encoding: utf-8 -*-
"""
@File: B01_random_weights.py
@Modify Time: 2025/9/9 20:23       
@Author: Kevin-Chen
@Descriptions: 结合了约束随机游走和性能计算的优化脚本
"""
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go


# 指标计算函数 (老)
def generate_alloc_perf_batch_old(port_daily: np.ndarray, portfolio_allocs: np.ndarray) -> pd.DataFrame:
    """
    批量计算多个资产组合的性能指标，使用向量化操作以提高效率。
    """
    assert port_daily.shape[1] == portfolio_allocs.shape[1]
    # 步骤1: 所有组合的日收益率 (矩阵乘法)
    port_return_daily = port_daily @ portfolio_allocs.T

    # 步骤2: 年化收益率 (基于对数总收益)
    # 添加一个极小值防止log(0)
    port_cum_returns = np.cumprod(1 + port_return_daily, axis=0)
    final_returns = port_cum_returns[-1, :]
    log_total_ret = np.log(final_returns, where=(final_returns > 0), out=np.full_like(final_returns, -np.inf))
    port_ret_annual = log_total_ret / (port_return_daily.shape[0]) * 252

    # 步骤3: 年化波动率 (基于对数日收益率)
    log_returns = np.log(1 + port_return_daily)
    port_vol_annual = np.std(log_returns, axis=0, ddof=1) * np.sqrt(252)

    # 步骤4: 打包成DataFrame
    ret_df = pd.DataFrame({
        "ret_annual": port_ret_annual,
        "vol_annual": port_vol_annual,
    })

    # 步骤5: 合并权重数据
    weight_df = pd.DataFrame(portfolio_allocs, columns=[f'w_{i}' for i in range(portfolio_allocs.shape[1])])

    return pd.concat([weight_df, ret_df], axis=1).dropna()


# 指标计算函数 (内存友好)
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
    (同事建议的更优算法)
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
        if len(idx) == 0: continue
        t = x[idx].sum()
        if t < low - 1e-6 or t > up + 1e-6:
            return None
    # 总和
    if not np.isclose(x.sum(), 1.0, atol=1e-6):
        return None
    return x


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

    # --- 2. 约束随机游走 ---
    print("开始通过约束随机游走生成投资组合...")
    num_of_asset = len(assets_list)
    current_weights = np.array([1 / num_of_asset] * num_of_asset)
    step_size = 0.02
    single_limits = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]  # 示例：每个资产的权重限制在0%到100%
    multi_limits = {(0, 1, 2): (0.3, 1.0)}

    final_weights = []
    for i in range(100000):
        new_proposal = current_weights + np.random.normal(0, step_size, len(current_weights))
        # adjusted_weights = primal_dual_interior_point(new_proposal, single_limits, multi_limits, max_iter=100)
        adjusted_weights = project_to_constraints_pocs(new_proposal, single_limits, multi_limits)

        if adjusted_weights is not None:
            final_weights.append(adjusted_weights)
            current_weights = adjusted_weights

    print(f"成功生成 {len(final_weights)} 个候选投资组合。")

    # --- 3. 显式校验和过滤权重 (新功能) ---
    print("正在对所有生成的权重进行最终校验...")

    validated_weights = []
    for w in final_weights:
        # 校验1: 权重和为1
        if not np.isclose(np.sum(w), 1.0, atol=1e-6):
            continue

        # 校验2: 单资产上下限
        single_valid = all(
            single_limits[i][0] - 1e-6 <= w[i] <= single_limits[i][1] + 1e-6 for i in range(num_of_asset))
        if not single_valid:
            continue

        # 校验3: 多资产组合上下限
        multi_valid = all(lower - 1e-6 <= np.sum(w[list(indices)]) <= upper + 1e-6 for indices, (lower, upper) in
                          multi_limits.items())
        if not multi_valid:
            continue

        validated_weights.append(w)

    print(f"校验完成。有效权重数量: {len(validated_weights)} / {len(final_weights)}")

    # 后续计算使用校验过的权重
    final_weights = validated_weights

    # --- 4. 批量计算收益和风险 ---
    if final_weights:
        print("正在批量计算所有组合的收益与风险...")
        weights_array = np.array(final_weights)
        port_daily_returns = hist_value_r[assets_list].values

        # 批量计算
        results_df = generate_alloc_perf_batch(port_daily_returns, weights_array)

        # 找出有效前沿
        results_df = cal_ef2_v4_ultra_fast(results_df)
        print("计算完成。")

        # --- 4. 使用 Plotly 进行交互式可视化 (新功能) ---
        print("正在生成交互式图表...")

        # 为权重列重命名
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

        # 添加所有随机生成的点
        fig.add_trace(go.Scatter(
            x=results_df['vol_annual'], y=results_df['ret_annual'],
            hovertext=results_df['hover_text'], hoverinfo='text',
            mode='markers', marker=dict(color='lightblue', size=2, opacity=0.7),
            name='随机有效组合'
        ))

        # 添加有效前沿上的点
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
