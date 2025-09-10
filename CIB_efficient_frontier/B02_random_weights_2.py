# -*- encoding: utf-8 -*-
"""
@File: B01_random_weights.py
@Modify Time: 2025/9/10 10:00
@Author: Kevin-Chen
@Descriptions: 结合了约束随机游走和性能计算的优化脚本。
             采纳建议A，使用高效的“有界单纯形投影”算法。
"""
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go


# =====================================================================================
# 性能计算函数
# =====================================================================================

def generate_alloc_perf_batch(port_daily: np.ndarray, portfolio_allocs: np.ndarray) -> pd.DataFrame:
    """
    批量计算多个资产组合的性能指标，使用向量化操作以提高效率。
    """
    assert port_daily.shape[1] == portfolio_allocs.shape[1]
    port_return_daily = port_daily @ portfolio_allocs.T
    port_cum_returns = np.cumprod(1 + port_return_daily, axis=0)
    final_returns = port_cum_returns[-1, :]
    log_total_ret = np.log(final_returns, where=(final_returns > 0), out=np.full_like(final_returns, -np.inf))
    port_ret_annual = log_total_ret / (port_return_daily.shape[0]) * 252
    log_returns = np.log(1 + port_return_daily)
    port_vol_annual = np.std(log_returns, axis=0, ddof=1) * np.sqrt(252)
    ret_df = pd.DataFrame({"ret_annual": port_ret_annual, "vol_annual": port_vol_annual})
    weight_df = pd.DataFrame(portfolio_allocs, columns=[f'w_{i}' for i in range(portfolio_allocs.shape[1])])
    return pd.concat([weight_df, ret_df], axis=1).dropna()


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
# 约束满足函数 (采纳建议A：有界单纯形投影)
# =====================================================================================

def project_on_bounded_simplex(v, single_limits, max_iter=100, tolerance=1e-9):
    """
    (新算法) 使用高效的“有界单纯形投影”算法（基于对偶问题的二分搜索）
    将一个可能无效的权重修正为满足所有约束的有效权重。
    该算法旨在解决: min ||x - v||^2, s.t. l <= x <= u, sum(x) = 1

    Args:
        v (np.array): 提议的权重向量.
        single_limits (list of tuples): 单资产的上下限 [(l_1, u_1), ...].
        max_iter (int): 二分搜索的最大迭代次数.
        tolerance (float): 收敛的容忍度.

    Returns:
        np.array: 投影后的有效权重, 如果无解则返回None.
    """
    l = np.array([lim[0] for lim in single_limits])
    u = np.array([lim[1] for lim in single_limits])

    # 初始检查：如果上下限本身就无法满足和为1，则无解
    if np.sum(l) > 1.0 + tolerance or np.sum(u) < 1.0 - tolerance:
        return None

    # 二分法寻找拉格朗日乘子 lambda
    lambda_min = np.min(v - u)
    lambda_max = np.max(v - l)

    for _ in range(max_iter):
        lambda_mid = (lambda_min + lambda_max) / 2.0
        x_projected = np.clip(v - lambda_mid, l, u)
        
        current_sum = np.sum(x_projected)

        if abs(current_sum - 1.0) < tolerance:
            break
        
        if current_sum > 1.0:
            lambda_min = lambda_mid
        else:
            lambda_max = lambda_mid
    
    # 最终投影结果
    x_final = np.clip(v - lambda_max, l, u)
    
    # 由于二分法的精度限制，最后可能需要微调使总和严格为1
    x_final /= np.sum(x_final)
    
    return x_final


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

    # --- 2. 约束随机游走 ---
    print("开始通过约束随机游走生成投资组合...")
    num_of_asset = len(assets_list)
    current_weights = np.array([1 / num_of_asset] * num_of_asset)
    step_size = 0.02
    # 定义单资产权重上下限 (盒约束)
    single_limits = [(0.0, 1.0), (0.0, 0.8), (0.0, 0.7), (0.0, 0.6), (0.0, 0.2)]
    # multi_limits 已被移除，因为新的投影算法不处理此类约束

    final_weights = []
    # 为了能快速看到结果，这里暂时只迭代10000次
    for i in range(10000):
        new_proposal = current_weights + np.random.normal(0, step_size, len(current_weights))
        
        # 调用新的、高效的投影函数
        adjusted_weights = project_on_bounded_simplex(new_proposal, single_limits)
        
        if adjusted_weights is not None:
            final_weights.append(adjusted_weights)
            current_weights = adjusted_weights

    print(f"成功生成 {len(final_weights)} 个候选投资组合。")

    # --- 3. 显式校验和过滤权重 ---
    print("正在对所有生成的权重进行最终校验...")
    
    validated_weights = []
    for w in final_weights:
        if w is None: continue
        # 校验1: 权重和为1
        if not np.isclose(np.sum(w), 1.0, atol=1e-6):
            continue
        
        # 校验2: 单资产上下限
        single_valid = all(single_limits[i][0] - 1e-6 <= w[i] <= single_limits[i][1] + 1e-6 for i in range(num_of_asset))
        if not single_valid:
            continue
            
        # 校验3: 多资产组合上下限 (已移除)
        validated_weights.append(w)
        
    print(f"校验完成。有效权重数量: {len(validated_weights)} / {len(final_weights)}")
    
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

        # --- 5. 使用 Plotly 进行交互式可视化 ---
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
            mode='markers', marker=dict(color='lightblue', size=5, opacity=0.7),
            name='随机有效组合'
        ))

        # 添加有效前沿上的点
        ef_df = results_df[results_df['on_ef'] == True]
        fig.add_trace(go.Scatter(
            x=ef_df['vol_annual'], y=ef_df['ret_annual'],
            hovertext=ef_df['hover_text'], hoverinfo='text',
            mode='markers', marker=dict(color='gold', size=7, line=dict(width=1, color='darkslategrey')),
            name='有效前沿'
        ))

        fig.update_layout(
            title='约束随机游走生成的投资组合与有效前沿 (高效投影算法)',
            xaxis_title='年化波动率 (Annual Volatility)',
            yaxis_title='年化收益率 (Annual Return)',
            legend_title="图例", hovermode='closest'
        )
        fig.show()
    else:
        print("未能生成任何有效的投资组合，无法进行后续计算和绘图。")