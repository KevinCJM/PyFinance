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
index_nv['date'] = pd.to_datetime(index_nv['date'], format='%Y%m%d')
index_nv = index_nv[index_nv['date'] >= pd.to_datetime('2010-01-01')]

# 取有行情的指数信息
index_info = index_info[index_info['指数代码'].isin(index_nv['index_code'].unique())]
print(len(index_info))

# 指数收盘价宽表：行=日期，列=指数代码
index_nv_wide = index_nv.pivot_table(index='date', columns='index_code', values='close')

# ===================== 1) 构造对数收益 =====================
# 对数收益：相关性更稳定，可加性强
log_px = np.log(index_nv_wide)
ret_df = log_px.diff()  # 第一行 NaN

# 单指标期数过滤（避免噪声指数）
min_obs = int(0.60 * ret_df.shape[0])  # 至少覆盖 60% 样本期
valid_mask = (ret_df.notna().sum(axis=0).to_numpy() >= min_obs)
ret_df = ret_df.loc[:, valid_mask]

# 剔除几乎常数的序列
stds = ret_df.std(axis=0, skipna=True).to_numpy()
ret_df = ret_df.loc[:, stds > 1e-12]

codes = ret_df.columns.to_numpy(copy=False)
n = codes.size
if n < 6:
    raise RuntimeError(f"有效指数数量过少（n={n}），无法稳定聚成5类。请降低 min_obs 或检查数据质量。")

# ===================== 2) 相关性/距离矩阵（稳健版） =====================
T = ret_df.shape[0]
min_pair_obs = max(90, int(0.25 * T))  # 两两相关最少重叠样本数

corr = ret_df.corr(min_periods=min_pair_obs)  # pairwise complete
corr_values = corr.to_numpy(copy=True)

# 重叠不足或不重叠 → 相关性视作 -1（相似度=0）
np.putmask(corr_values, ~np.isfinite(corr_values), -1.0)
corr_values = np.clip(corr_values, -1.0, 1.0)
np.fill_diagonal(corr_values, 1.0)

# 距离矩阵 D=1-rho（供 silhouette 等）
D = 1.0 - corr_values
np.fill_diagonal(D, 0.0)

# 相似度矩阵 S ∈ [0,1]（谱聚类使用），去自环并对称化
S = 0.5 * (corr_values + 1.0)
S = np.clip(S, 0.0, 1.0)
np.fill_diagonal(S, 0.0)
S = 0.5 * (S + S.T)

# ===================== 3) KNN 稀疏化，增强连通稳定性 =====================
topk = max(1, int(np.ceil(np.sqrt(n))))  # 每点保留 ~sqrt(n) 个近邻
order = np.argsort(S, axis=1)  # 每行从小到大索引

knn_mask = np.zeros_like(S, dtype=bool)
for i in range(n):  # O(n^2) 但只做一次，n通常不大
    knn_mask[i, order[i, -topk:]] = True

# 对称化 KNN 图：并集
knn_sym = knn_mask | knn_mask.T
S_knn = np.where(knn_sym, S, 0.0)

# 孤立点兜底：若某行全零，连向其“最相似”的一个点
row_sum = S_knn.sum(axis=1)
iso = (row_sum == 0)
if iso.any():
    for i in np.where(iso)[0]:
        j = int(np.argmax(S[i, :]))
        if i != j and S[i, j] > 0:
            S_knn[i, j] = S[i, j]
            S_knn[j, i] = S[i, j]

# ===================== 4') 五大类半监督分层归类（替换原 4~8 步） =====================
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples
from pyecharts.charts import Line
from pyecharts import options as opts
from pyecharts.globals import ThemeType

# ---------- 基础准备 ----------
# ret_df: 日对数收益，列=代码（上文已算好）
codes = ret_df.columns.to_numpy(copy=False)
n = codes.size

# 年化波动（跳过缺失）
ann_vol = ret_df.std(axis=0, skipna=True).to_numpy() * np.sqrt(252)

# 动态阈值（“分层粗分”）
q10, q35, q60 = np.nanpercentile(ann_vol, [10, 35, 60])
MONEY_TH = min(0.012, q10)  # 货币：年化波动 ≲ 1.2% 或处于最底部分位
BOND_TH = min(0.07, q35)  # 固收：年化波动 ≲ 7% 或处于次低分位
MID_TH = q60  # 中等波（潜在“混合”上界）

money_mask = (ann_vol <= MONEY_TH)
bond_mask = (~money_mask) & (ann_vol <= BOND_TH)
mid_mask = (~money_mask) & (~bond_mask) & (ann_vol <= MID_TH)
high_mask = (~money_mask) & (~bond_mask) & (~mid_mask)  # 高波：权益/商品 候选


# ---------- 在高波组里再做 2 类相似度聚类（权益 vs 商品） ----------
def build_knn_graph(S_full, idx, topk=None):
    if topk is None:
        topk = max(1, int(np.ceil(np.sqrt(len(idx)))))
    S = S_full[np.ix_(idx, idx)].copy()
    np.fill_diagonal(S, 0.0)
    order = np.argsort(S, axis=1)
    knn_mask = np.zeros_like(S, dtype=bool)
    for i in range(S.shape[0]):
        knn_mask[i, order[i, -topk:]] = True
    sym = knn_mask | knn_mask.T
    S_knn = np.where(sym, S, 0.0)
    row_sum = S_knn.sum(axis=1)
    iso = (row_sum == 0)
    if iso.any():
        for i in np.where(iso)[0]:
            j = int(np.argmax(S[i, :]))
            if i != j and S[i, j] > 0:
                S_knn[i, j] = S[i, j]
                S_knn[j, i] = S[i, j]
    return S_knn


eq_mask = np.zeros(n, dtype=bool)
cmd_mask = np.zeros(n, dtype=bool)

hi_idx = np.flatnonzero(high_mask)
if hi_idx.size > 0:
    S_hi = build_knn_graph(S, hi_idx)
    # 2 类谱聚类
    sc_hi = SpectralClustering(n_clusters=2, affinity='precomputed',
                               assign_labels='kmeans', n_init=20, random_state=42)
    hi_labels = sc_hi.fit_predict(S_hi)
    # 用“波动的波动”（vol-of-vol）区分：商品通常 vol-of-vol 更高
    roll_std = ret_df.iloc[:, hi_idx].rolling(21, min_periods=10).std()
    vov = roll_std.std(axis=0, skipna=True).to_numpy()  # 每列的“std 的 std”
    vov0 = vov[hi_labels == 0].mean()
    vov1 = vov[hi_labels == 1].mean()
    cmd_cluster = 0 if vov0 > vov1 else 1
    # 赋类
    cmd_mask[hi_idx[hi_labels == cmd_cluster]] = True
    eq_mask[hi_idx[hi_labels != cmd_cluster]] = True
else:
    # 没有高波组，兜底：都不标为权益/商品
    pass

# ---------- 先取“货币/固收”，再将“高波”拆成 权益/商品，剩余与中波判为“混合” ----------
macro_group = np.array(["未分类"] * n, dtype=object)
macro_group[money_mask] = "货币"
macro_group[bond_mask] = "固收"
macro_group[cmd_mask] = "商品"
macro_group[eq_mask] = "权益"
# 中波 & 其余未归入者 → “混合”
macro_group[(macro_group == "未分类") | mid_mask] = "混合"


# ---------- 每类内部用 Medoid 选代表 ----------
def medoid_from_D(D_full, member_idx):
    if member_idx.size == 0:
        return None
    subD = D_full[np.ix_(member_idx, member_idx)]
    return member_idx[np.argmin(subD.sum(axis=1))]


classes = ["权益", "固收", "货币", "商品", "混合"]
class_members = {c: np.flatnonzero(macro_group == c) for c in classes}
class_reps = {}
for c in classes:
    midx = class_members[c]
    rep = medoid_from_D(D, midx)
    class_reps[c] = codes[rep] if rep is not None else None

# ---------- 打印结果 ----------
print("\n=== 五大类规模（条数） ===")
for c in classes:
    print(f"{c}: {class_members[c].size}")

print("\n=== 每类代表指数（Medoid） ===")
for c in classes:
    print(f"{c}: {class_reps[c]}")

# ===================== 5') pyecharts 绘图：五大类代表指数的净值 =====================
from pyecharts.charts import Line
from pyecharts import options as opts
from pyecharts.globals import ThemeType

# 1) 代表代码列表（按五大类顺序，跳过 None）
classes = ["权益", "固收", "货币", "商品", "混合"]
rep_pairs = [(c, class_reps.get(c)) for c in classes if class_reps.get(c) is not None]
# 仅保留实际存在于宽表中的代码
rep_pairs = [(c, code) for (c, code) in rep_pairs if code in index_nv_wide.columns]
if not rep_pairs:
    raise RuntimeError("未选出用于绘图的代表指数，请检查上游分类逻辑。")

rep_codes = [code for (_, code) in rep_pairs]

# 2) 名称映射（指数简称/名称列自动探测）
if '指数代码' in index_info.columns:
    _name_map_df = index_info.set_index('指数代码')
else:
    _name_map_df = index_info.set_index(index_info.columns[0])
_short_name_col = None
for _cand in ['指数简称', '指数名称', '名称', 'name', 'short_name']:
    if _cand in _name_map_df.columns:
        _short_name_col = _cand
        break


def code2name(code: str) -> str:
    if _short_name_col is not None and code in _name_map_df.index:
        return str(_name_map_df.loc[code, _short_name_col])
    return ""


# 3) 取价格并对齐区间（无公共交集则放宽到≥80%列非空）
px_rep = index_nv_wide.loc[:, rep_codes]
valid_mask_all = ~px_rep.isna()
common = valid_mask_all.all(axis=1)
if not common.any():
    thresh = int(0.8 * px_rep.shape[1])
    common = (valid_mask_all.sum(axis=1) >= thresh)
px_rep = px_rep.loc[common]

# 4) 各列用“各自首个非空值”归一为 1.0（稳健应对不对齐）
arr = px_rep.to_numpy(copy=False, dtype=float)


def _first_valid(x: np.ndarray) -> float:
    m = ~np.isnan(x)
    return x[np.argmax(m)] if m.any() else np.nan


scales = np.array([_first_valid(arr[:, j]) for j in range(arr.shape[1])], dtype=float)
scales[~np.isfinite(scales)] = np.nan
nav = arr / scales

# 5) 组织 legend 名称：`大类 | 代码 名称`
series_labels = [
    f"{cls}｜{code} {code2name(code)}".strip()
    for (cls, code) in rep_pairs
]

# 6) x 轴
x_data = [d.strftime("%Y-%m-%d") for d in px_rep.index.to_pydatetime()]

# 7) 画图
line = Line(init_opts=opts.InitOpts(
    width="1200px",
    height="640px",
    theme=ThemeType.LIGHT,
    page_title="五大类代表指数净值"
))
line.add_xaxis(xaxis_data=x_data)

for j, code in enumerate(rep_codes):
    y = nav[:, j]
    if not np.isfinite(y).any():
        continue
    y_list = [float(v) if np.isfinite(v) else None for v in y]
    line.add_yaxis(
        series_name=series_labels[j],
        y_axis=y_list,
        is_connect_nones=False,  # 缺失不连线
        is_smooth=False,
        symbol="none",
        linestyle_opts=opts.LineStyleOpts(width=2),
        label_opts=opts.LabelOpts(is_show=False),
    )

line.set_global_opts(
    title_opts=opts.TitleOpts(
        title="五大类代表指数的历史净值（各自首值归一=1.0）",
        subtitle="代表=各类 Medoid；区间=公共或≥80%列可用"
    ),
    tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
    legend_opts=opts.LegendOpts(pos_top="4%", pos_left="center"),
    datazoom_opts=[opts.DataZoomOpts(type_="inside"), opts.DataZoomOpts(type_="slider")],
    xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
    yaxis_opts=opts.AxisOpts(type_="value", is_scale=True)
)

out_path = "五大类代表指数_净值_交互图.html"
line.render(out_path)
print(f"已生成交互图: {out_path}")
