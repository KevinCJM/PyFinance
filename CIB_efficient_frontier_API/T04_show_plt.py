# -*- encoding: utf-8 -*-
"""
@File: T04_show_plt.py
@Modify Time: 2025/9/16 10:50       
@Author: Kevin-Chen
@Descriptions: 展示图表
"""
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional


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
    else:
        fig.show()


if __name__ == '__main__':
    import numpy as np
    import pandas as pd

    # 1. 生成随机投资组合数据 (点云)
    num_portfolios = 2500
    vols = np.random.uniform(0.1, 0.35, num_portfolios)
    rets = vols * np.random.uniform(0.3, 0.9, num_portfolios) + np.random.normal(0, 0.03, num_portfolios)

    random_portfolios_df = pd.DataFrame({
        'vol_annual': vols,
        'ret_annual': rets,
        'hover_text': [f"随机组合<br>收益率: {r:.2%}<br>波动率: {v:.2%}" for r, v in zip(rets, vols)]
    })

    # 2. 生成有效前沿数据 (曲线)
    frontier_vols = np.linspace(0.1, 0.35, 100)
    frontier_rets = 1.5 * (frontier_vols - 0.09) ** 2 + 0.05  # A simple parabola

    efficient_frontier_df = pd.DataFrame({
        'vol_annual': frontier_vols,
        'ret_annual': frontier_rets,
        'hover_text': [f"有效前沿<br>收益率: {r:.2%}<br>波动率: {v:.2%}" for r, v in zip(frontier_rets, frontier_vols)]
    })

    # 3. 找到并标记特殊点
    # 最小波动率点
    min_vol_point_df = efficient_frontier_df.iloc[[efficient_frontier_df['vol_annual'].idxmin()]].copy()
    min_vol_point_df['hover_text'] = f"波动最小组合<br>收益率: {min_vol_point_df['ret_annual'].iloc[0]:.2%}<br>波动率: {min_vol_point_df['vol_annual'].iloc[0]:.2%}"

    # 夏普率最大点 (模拟一个)
    risk_free_rate = 0.02
    sharpe_ratios = (efficient_frontier_df['ret_annual'] - risk_free_rate) / efficient_frontier_df['vol_annual']
    max_sharpe_point_df = efficient_frontier_df.iloc[[sharpe_ratios.idxmax()]].copy()
    max_sharpe_point_df['hover_text'] = f"夏普率最大组合<br>收益率: {max_sharpe_point_df['ret_annual'].iloc[0]:.2%}<br>波动率: {max_sharpe_point_df['vol_annual'].iloc[0]:.2%}<br>夏普率: {sharpe_ratios.max():.2f}"

    # 4. 组装 scatter_points_data
    scatter_points_data = [
        {
            "data": random_portfolios_df,
            "name": "随机投资组合",
            "color": "lightblue",
            "size": 5,
            "opacity": 0.6,
        },
        {
            "data": efficient_frontier_df,
            "name": "有效前沿 (模拟)",
            "color": "darkblue",
            "size": 4,
            "opacity": 1.0,
        },
        {
            "data": max_sharpe_point_df,
            "name": "夏普率最大组合 (模拟)",
            "color": "red",
            "size": 12,
            "opacity": 1.0,
            "symbol": "star",
            "marker_line": dict(width=1, color='black')
        },
        {
            "data": min_vol_point_df,
            "name": "波动最小组合 (模拟)",
            "color": "green",
            "size": 12,
            "opacity": 1.0,
            "symbol": "diamond",
            "marker_line": dict(width=1, color='black')
        }
    ]

    # 5. 添加一个当前投资组合点
    current_portfolio = pd.DataFrame({
        'vol_annual': [0.18],
        'ret_annual': [0.08],
        'hover_text': ['当前组合<br>收益率: 8.00%<br>波动率: 18.00%']
    })
    scatter_points_data.append({
        "data": current_portfolio,
        "name": "当前组合",
        "color": "purple",
        "size": 12,
        "opacity": 1.0,
        "symbol": "cross",
        "marker_line": dict(width=2, color='black')
    })

    # 6. 调用绘图函数
    plot_efficient_frontier(
        scatter_points_data,
        title='模拟有效前沿与投资组合',
        output_filename=None
    )
    print("已生成示例图表: efficient_frontier_example.html")
