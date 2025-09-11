# -*- encoding: utf-8 -*-
"""
@File: A05_index_analysis_risk_return.py
@Modify Time: 2025/9/11 08:27
@Author: Kevin-Chen
@Descriptions: 分析各个指数的风险收益特征, 刻画在风险收益图上 + 选项化过滤与虚拟净值展示
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
TRADING_DAYS = 252


# ===================== 工具函数 =====================
def load_info_map(xlsx_path: str, sheet=0) -> pd.DataFrame:
    """
    读取指数信息表，返回以 '指数代码' 为索引、只包含 ['short_name','asset_class'] 的映射表。
    会做：dtype=str、去空白、去重(保留最后一条)。
    """
    info_raw = pd.read_excel(xlsx_path, sheet_name=sheet, dtype={'指数代码': str}).copy()
    # 兼容不同列名（如有差异，请在此处改成你真实列名）
    rename_map = {'指数简称': 'short_name', '资产类别': 'asset_class'}
    info_raw = info_raw.rename(columns=rename_map)

    # 只保留需要的列；不存在的列用空串占位
    for col in ['指数代码', 'short_name', 'asset_class']:
        if col not in info_raw.columns:
            info_raw[col] = ''

    info_raw['指数代码'] = info_raw['指数代码'].astype(str).str.strip()
    info_raw['short_name'] = info_raw['short_name'].astype(str).str.strip()
    info_raw['asset_class'] = info_raw['asset_class'].astype(str).str.strip()

    # 去重：保留最后一条
    info_raw = info_raw.drop_duplicates(subset=['指数代码'], keep='last')

    info_map = (info_raw
                .set_index('指数代码')
                [['short_name', 'asset_class']]
                .sort_index())
    return info_map


def align_info_to_codes(info_map: pd.DataFrame, codes_index: pd.Index) -> pd.DataFrame:
    """
    将 info_map 严格对齐到 codes_index 的顺序与长度；缺失用空串。
    """
    codes = pd.Index(codes_index.astype(str))  # 强制字符串对齐
    info_aligned = info_map.reindex(codes)
    info_aligned = info_aligned.fillna({'short_name': '', 'asset_class': ''})
    # 安全检查：长度必须一致
    if len(info_aligned) != len(codes):
        raise ValueError(f"[对齐失败] info_aligned 长度={len(info_aligned)} 与 codes 长度={len(codes)} 不一致")
    return info_aligned


# ===================== 配置区 =====================
# 1) 是否剔除极端收益/风险的指数（基于分位数）
EXCLUDE_EXTREMES = True
EXTREME_RULES = {  # 指标: (下分位, 上分位)
    'cagr': (0.01, 0.99),  # 剔除极端高/低长期收益
    'vol_ann': (0.01, 0.99),  # 剔除极端高/低波动
    # 如需更严可加 'mdd': (0.01, 0.99)（注意 mdd 为负值，若启用建议改用 abs(mdd) 判别）
}

# 2) 设定起始日期；若为 None 则不启用此规则, 启用后将剔除“首条净值在起始日之后”的指数，以保证统一样本起点。
START_DATE = '2015-01-01'  # 示例: '2015-01-01' 或 None

# 3) 第二张图的展示集合：指定指数代码列表；若为 None 则取 Top K（按 CAGR）
SELECTED_CODES = [  # 'H30345', 'H11021', '000510',
    # 'H30076',  # 期指两倍
    # '931790',  # 中韩半导体
    'H30228',  # 商品OCFI
    'h00979.CSI',  # 大宗商品全收益
    '000300.SH',  # 沪深300
    'h30322.CSI',  # 股债RP
    # 'H11001',   # 中证全债
    'H11023',  # 债券基金
    'H11025',   # 货币基金
]  # 例如: ['000300.SH', '000905.SH']
TOPK_FOR_NAV = 20

# 4) 指数信息excel相关
INFO_XLSX = '中证系列指数.xlsx'  # 新增：指数基础信息文件（含 指数代码/指数简称/资产类别）
INFO_SHEET = '指数列表'  # 如有命名的 sheet，也可填 sheet 名

# 5) 可选：按净值缺失比例剔除指数（在 START_DATE 过滤之后、收益计算之前执行）
DROP_BY_MISSING = True  # 是否启用
MAX_MISSING_RATIO = 0.2  # 缺失占比阈值（0~1），比如 0.20 表示缺失超过 20% 的指数被剔除

# ===================== 1) 原始数据 -> 宽表 close_df =====================
close_df = pd.read_parquet('CSI_index_return.parquet')[['date', 'index_code', 'close']]
close_df['date'] = pd.to_datetime(close_df['date'], format='%Y%m%d')
close_df = close_df.drop_duplicates(['date', 'index_code'], keep='last').reset_index(drop=True)
close_df = close_df.pivot(index='date', columns='index_code', values='close')
close_df = close_df.sort_index()

# --------- 起始日过滤（可选）---------
if START_DATE is not None:
    start_ts = pd.Timestamp(START_DATE)
    # 每列首个非空日期（需先过滤全 NaN 列）
    has_data = close_df.notna().sum() > 0
    first_valid = close_df.loc[:, has_data].notna().idxmax()
    keep_cols = first_valid[first_valid <= start_ts].index
    close_df = close_df.loc[close_df.index >= start_ts, keep_cols]

# --------- 按净值缺失比例剔除指数（可选）---------
if DROP_BY_MISSING:
    # 以当前样本期的净值宽表为准，统计每列缺失占比
    total_days = close_df.shape[0]
    # 若 total_days 很小，避免除零
    if total_days > 0:
        missing_ratio = close_df.isna().sum() / float(total_days)
        # 保留“缺失占比 <= 阈值”的列；若全列缺失也会被自然剔除
        keep_cols_missing = (missing_ratio <= MAX_MISSING_RATIO)
        # 至少保留一些列，防止极端条件下全被剔除
        if keep_cols_missing.any():
            close_df = close_df.loc[:, keep_cols_missing.index[keep_cols_missing]]
        else:
            # 若没有任何列满足条件，保留缺失最少的前若干列（兜底，可按需改成直接报错）
            topk = min(10, missing_ratio.size)
            close_df = close_df.loc[:, missing_ratio.nsmallest(topk).index]
    else:
        # 没有可用日期，给出提示（不抛错以便后续逻辑处理）
        print("[WARN] 按缺失比例剔除：当前样本期没有可用日期。")

# ===================== 2) 宽表 -> 日收益，加工预处理 =====================
ret_df = close_df.pct_change()

# 2.1 保留至少 min_obs 个有效交易日的指数（避免统计失真）
min_obs = 252
valid_cols = ret_df.count() >= min_obs
ret_df = ret_df.loc[:, valid_cols]

# 2.2 统一对齐：交集日期（横向可比）
ret_df = ret_df.dropna(how='any')

# 2.3 极值裁剪（winsorize）：按列对称 0.5% 分位，稳健对抗离群点
clip_q = 0.005
lower = ret_df.quantile(clip_q)
upper = ret_df.quantile(1 - clip_q)
ret_df = ret_df.clip(lower=lower, upper=upper, axis=1)

# ===================== 3) 指标计算（全向量化） =====================
mu_ann = ret_df.mean() * TRADING_DAYS  # 年化算术收益

wealth = (1.0 + ret_df).cumprod()
T = ret_df.shape[0]
cagr = (wealth.iloc[-1] / wealth.iloc[0]) ** (TRADING_DAYS / T) - 1.0  # 年化几何收益

vol_ann = ret_df.std(ddof=1) * np.sqrt(TRADING_DAYS)  # 年化波动

rf_ann = 0.0
rf_daily = rf_ann / TRADING_DAYS
down_leg = np.minimum(ret_df - rf_daily, 0.0)
downside_ann = np.sqrt((down_leg ** 2).mean()) * np.sqrt(TRADING_DAYS)  # 年化下行波动

rolling_max = wealth.cummax()
drawdown = wealth / rolling_max - 1.0
mdd = drawdown.min()  # 负数

skew = ret_df.skew()
kurt = ret_df.kurt()

eps = 1e-12
sharpe = (mu_ann - rf_ann) / (vol_ann.replace(0, np.nan) + eps)
sortino = (mu_ann - rf_ann) / (downside_ann.replace(0, np.nan) + eps)

metrics = pd.DataFrame({
    'mu_ann': mu_ann,
    'cagr': cagr,
    'vol_ann': vol_ann,
    'downside_ann': downside_ann,
    'mdd': mdd,
    'skew': skew,
    'kurtosis_excess': kurt,
    'sharpe': sharpe,
    'sortino': sortino,
}).sort_values('cagr', ascending=False)

# ===================== 3.1 读取指数信息并对齐 =====================
# 期待 Excel 至少包含列：'指数代码'、'指数简称'、'资产类别'
info_df = pd.read_excel(INFO_XLSX, sheet_name=INFO_SHEET, dtype={'指数代码': str}).copy()
# 标准化代码：去空白，保持为字符串
info_df['指数代码'] = info_df['指数代码'].astype(str).str.strip()

# 与 metrics.index（指数代码）对齐；缺失则用空串占位
info_df = (info_df
           .set_index('指数代码')
           .reindex(metrics.index)
           .rename(columns={'指数简称': 'short_name', '资产类别': 'asset_class'}))
info_df[['short_name', 'asset_class']] = info_df[['short_name', 'asset_class']].fillna('')
info_map = load_info_map(INFO_XLSX, INFO_SHEET)  # 全量映射
info_aligned = align_info_to_codes(info_map, metrics.index)  # 严格对齐到 metrics.index

# ===================== 4) 可选：剔除极端收益/风险的指数 =====================
if EXCLUDE_EXTREMES and len(metrics) > 0:
    mask = pd.Series(True, index=metrics.index)
    for col, (q_lo, q_hi) in EXTREME_RULES.items():
        if col not in metrics.columns:
            continue
        lo = metrics[col].quantile(q_lo)
        hi = metrics[col].quantile(q_hi)
        mask &= metrics[col].between(lo, hi)
    # 至少保留若干个，防止全被剔除
    if mask.sum() >= max(10, int(0.2 * len(mask))):
        metrics = metrics.loc[mask]
        # 同步缩减收益与净值数据（用于后续第二张图）
        ret_df = ret_df.loc[:, metrics.index]
        wealth = wealth.loc[:, metrics.index]
        drawdown = drawdown.loc[:, metrics.index]
# —— 在“极端值剔除”块之后立刻重新对齐，确保 info_aligned 与最新 metrics.index 等长 ——
info_map = load_info_map(INFO_XLSX, INFO_SHEET)  # 仍用同一份 Excel 映射
info_aligned = align_info_to_codes(info_map, metrics.index)  # 关键：重新对齐到过滤后的 metrics.index

# ===================== 5) 图一：风险-收益散点（支持按资产类别筛选） =====================
size_scale = 40.0
mdd_abs = metrics['mdd'].abs()
size_full = (mdd_abs / (mdd_abs.max() + 1e-12) * size_scale + 6.0)  # 与 metrics 对齐的全量尺寸

# 合并绘图所需信息
plot_df = metrics.join(info_aligned)  # 列含: 指标 + short_name + asset_class
plot_df['asset_class'] = plot_df['asset_class'].replace('', '未分类')
categories = sorted(plot_df['asset_class'].unique().tolist())

# 为 hover 组合 customdata（先准备数值矩阵，后面子集切片即可）
base_customdata = np.column_stack([
    plot_df['mu_ann'].to_numpy(),
    plot_df['downside_ann'].to_numpy(),
    plot_df['mdd'].to_numpy(),
    plot_df['sharpe'].to_numpy(),
    plot_df['sortino'].to_numpy(),
    plot_df['skew'].to_numpy(),
    plot_df['kurtosis_excess'].to_numpy(),
    plot_df['short_name'].to_numpy(),  # 7
    plot_df['asset_class'].to_numpy(),  # 8
])

fig = go.Figure()
vis_flags = []  # 每个按钮需要的可见性模板

# 按类别逐个 trace 添加（每个类别一条散点，点为该类别的所有指数）
for cat in categories:
    mask = (plot_df['asset_class'] == cat).to_numpy()
    # 没有点则跳过
    if not mask.any():
        continue

    fig.add_trace(go.Scatter(
        x=plot_df.loc[mask, 'vol_ann'],
        y=plot_df.loc[mask, 'cagr'],
        mode='markers',
        name=str(cat),  # 图例显示类别
        text=plot_df.index[mask].astype(str),  # hover 中显示“指数代码”
        marker=dict(
            size=size_full[mask].to_numpy(),
            sizemode='diameter',
            opacity=0.9,
            line=dict(width=0.5)
        ),
        customdata=base_customdata[mask, :],
        hovertemplate=(
            "<b>指数代码</b>: %{text}<br>"
            "<b>指数简称</b>: %{customdata[7]}<br>"
            "<b>资产类别</b>: %{customdata[8]}<br>"
            "年化波动率 σ: %{x:.2%}<br>"
            "CAGR: %{y:.2%}<br>"
            "年化算术收益 μ: %{customdata[0]:.2%}<br>"
            "下行波动 (年化): %{customdata[1]:.2%}<br>"
            "最大回撤 MDD: %{customdata[2]:.2%}<br>"
            "Sharpe: %{customdata[3]:.2f}<br>"
            "Sortino: %{customdata[4]:.2f}<br>"
            "偏度: %{customdata[5]:.2f} | 超额峰度: %{customdata[6]:.2f}"
            "<extra></extra>"
        )
    ))

# —— 下拉按钮（单选），配合图例可实现多选 ——
n_traces = len(fig.data)
buttons = []

# 0) 全部
buttons.append(dict(
    label="全部",
    method="update",
    args=[{"visible": [True] * n_traces},
          {"title": "指数风险–收益特征图（全部类别）"}]
))

# 1) 清空
buttons.append(dict(
    label="清空",
    method="update",
    args=[{"visible": [False] * n_traces},
          {"title": "指数风险–收益特征图（无类别可见）"}]
))

# 2) 各类别单独显示
for i, cat in enumerate(categories):
    vis = [False] * n_traces
    if i < n_traces:
        vis[i] = True
    buttons.append(dict(
        label=str(cat),
        method="update",
        args=[{"visible": vis},
              {"title": f"指数风险–收益特征图（{cat}）"}]
    ))

title_suffix = "（已剔除极端值）" if EXCLUDE_EXTREMES else "（未剔除极端值）"
fig.update_layout(
    title=f"指数风险–收益特征图 {title_suffix}",
    xaxis_title="年化波动率 σ",
    yaxis_title="年化几何收益 CAGR",
    hovermode='closest',
    template='plotly_white',
    legend=dict(title="资产类别", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    updatemenus=[dict(
        type="dropdown",
        direction="down",
        x=1.0, xanchor="right",
        y=1.15, yanchor="top",
        buttons=buttons,
        showactive=True,
        bgcolor="white",
        bordercolor="#ccc"
    )]
)

fig.add_hline(y=0.0, line_width=1, line_dash="dash", opacity=0.6)
fig.add_vline(x=float(plot_df['vol_ann'].median()), line_width=1, line_dash="dot", opacity=0.4)

fig.show()

# ===================== 6) 图二：指定指数的虚拟净值（从 1 开始） =====================
# 选择展示集合
if SELECTED_CODES is not None and len(SELECTED_CODES) > 0:
    codes_to_plot = [c for c in SELECTED_CODES if c in wealth.columns]
else:
    # 默认按 CAGR 取 TopK
    codes_to_plot = metrics.index[:TOPK_FOR_NAV].tolist()

if len(codes_to_plot) > 0:
    nav = (1.0 + ret_df[codes_to_plot]).cumprod()
    # 归一化到 1：确保首日显示为 1
    nav = nav / nav.iloc[0]

    fig2 = go.Figure()
    for code in codes_to_plot:
        fig2.add_trace(go.Scatter(
            x=nav.index,
            y=nav[code],
            mode='lines',
            name=str(code),
            hovertemplate="<b>%{fullData.name}</b><br>%{x|%Y-%m-%d}  虚拟净值: %{y:.3f}<extra></extra>"
        ))
    fig2.update_layout(
        title=f"虚拟净值（起点=1） | 展示 {len(codes_to_plot)} 条",
        xaxis_title="日期",
        yaxis_title="虚拟净值",
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0)
    )
    fig2.show()
else:
    print("无可展示的指数：请检查 SELECTED_CODES 或 TOPK_FOR_NAV 设置。")
