# -*- encoding: utf-8 -*-
"""
@File: A02_index_analysis_cluster.py
@Modify Time: 2025/9/9 13:24       
@Author: Kevin-Chen
@Descriptions: 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyecharts.charts import Line
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

# ===================== 0) 读取数据 =====================
# 指数基本信息（建议包含: 指数代码, 指数简称/名称, 若有资产类别更好）
index_info = pd.read_excel('中证系列指数.xlsx')

# 指数日行情数据（包含列: date(YYYYMMDD), index_code, close）
index_nv = pd.read_parquet('CSI_index_return.parquet').reset_index(drop=True)
# 剔除数据量小于100的指数
counts = index_nv['index_code'].value_counts()
valid_codes = counts[counts >= 100].index
index_nv = index_nv[index_nv['index_code'].isin(valid_codes)]
index_info = index_info[index_info['指数代码'].isin(index_nv['index_code'].unique())]
index_info.to_excel('当前有数据的指数列表.xlsx', index=False)

index_nv['date'] = pd.to_datetime(index_nv['date'], format='%Y%m%d')
index_nv = index_nv[index_nv['date'] >= pd.to_datetime('2012-01-01')]
index_nv = index_nv[index_nv['index_code'].isin([
    'H11021',  # 股票基金, 中证股票型基金指数
    'H11022',  # 混合基金, 中证混合型基金指数
    'H11023',  # 债券基金, 中证债券型基金指数
    'H11025',  # 货币基金, 中证货币基金指数
    'H11026',  # QDII基金	中证QDII基金指数
    # '932047',  # 中证REITs全收益	中证REITs全收益指数
    # 'H30457',  # 沪港深通综合, 中证沪港深互联互通综合指数
    # 'H11061',  # 商品CFCI, 中证商品期货综合指数
    # 'H30228',  # 商品OCFI, 中证优化展期商品期货成份指数
    'H30009',  # 商品CFI, 中证商品期货成份指数
    # 'H30072',  # 贵金CFI, 中证贵金属期货成份指数
    # 'H30224',  # 粮油CFI, 中证粮油期货成份指数
    # 'H11180',  # 合成期权, 中证股债动态合成期权策略指数
    # '930957',  # 港股通中国100, 中证港股通中国100指数
    # 'H30322',  # 股债RP, 中证股债风险平价指数
    'H30345',  # 300波控1, 沪深300波动率控制20%指数
    # 'H30347',  # 500波控1, 中证500波动率控制25%指数
])]

# 指数收盘价宽表：行=日期，列=指数代码
index_nv_wide = index_nv.pivot_table(index='date', columns='index_code', values='close')

# 取有行情的指数信息
index_info = index_info[index_info['指数代码'].isin(index_nv['index_code'].unique())]
print(index_info)

# ===================== 1) 计算归一化净值 =====================
# 归一化处理，使所有指数的起始净值都为1，方便比较走势
normalized_nv = index_nv_wide / index_nv_wide.iloc[0]

# ===================== 2) 扩充到所有自然日并绘图 =====================
# 创建完整的日历日期范围
all_dates = pd.date_range(start=normalized_nv.index.min(), end=normalized_nv.index.max(), freq='D')
# 重新索引数据，非交易日将填充为NaN
normalized_nv_reindexed = normalized_nv.reindex(all_dates)

# 准备 x 轴数据 (所有日历日期)
x_data = normalized_nv_reindexed.index.strftime("%Y-%m-%d").tolist()

# 创建 Line 图表
line_chart = Line(init_opts=opts.InitOpts(width="1200px", height="500px", theme=ThemeType.WHITE))
line_chart.add_xaxis(xaxis_data=x_data)

# 循环添加 y 轴数据 (归一化净值)
for col in normalized_nv_reindexed.columns:
    # 获取指数简称
    series_name = index_info.loc[index_info['指数代码'] == col, '指数简称'].iloc[0]

    # 将NaN替换为None，以便pyecharts正确断开线条
    y_values = normalized_nv_reindexed[col].round(4)
    y_axis_data = [v if pd.notna(v) else None for v in y_values]

    line_chart.add_yaxis(
        series_name=series_name,
        y_axis=y_axis_data,
        is_smooth=False,  # 使用非平滑线以清晰地显示断点
        label_opts=opts.LabelOpts(is_show=False),
    )

# 设置图表全局选项
line_chart.set_global_opts(
    title_opts=opts.TitleOpts(title="指数净值走势", subtitle="起始净值=1"),
    tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
    legend_opts=opts.LegendOpts(pos_left="center", pos_top="1%", orient="horizontal"),
    datazoom_opts=[
        opts.DataZoomOpts(is_show=True, range_start=0, range_end=100),
        opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
    ],
    xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
    yaxis_opts=opts.AxisOpts(
        type_="value",
        axistick_opts=opts.AxisTickOpts(is_show=True),
        splitline_opts=opts.SplitLineOpts(is_show=True),
    ),
)

# ===================== 3) 渲染图表 =====================
output_filename = "代表指数归一化净值_交互图.html"
line_chart.render(output_filename)

print(f"图表已成功生成：{output_filename}")
