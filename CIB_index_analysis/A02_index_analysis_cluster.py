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

# ===================== 4) 谱聚类（5类） =====================
k = 5
sc = SpectralClustering(
    n_clusters=k,
    affinity='precomputed',
    assign_labels='kmeans',
    n_init=20,
    random_state=42
)
labels = sc.fit_predict(S_knn)  # shape=(n,)

cluster_ids = np.unique(labels)

# ===================== 5) 离群点判定（轮廓系数 + 类内平均相似度） =====================
sil = silhouette_samples(D, labels, metric='precomputed')  # 距离矩阵 D

# 类内平均相似度（自环=0 已去除）
intra_mean_sim = np.full(n, -np.inf, dtype=np.float64)
for cid in cluster_ids:
    idx = np.flatnonzero(labels == cid)
    if idx.size <= 1:
        continue
    sub = S[np.ix_(idx, idx)]
    intra_mean_sim[idx] = sub.sum(axis=1) / (idx.size - 1)

sil_th = 0.05
sim_th = 0.15
is_outlier = (sil < sil_th) | (intra_mean_sim < sim_th)

# ===================== 6) 代表指数（Medoid）与簇成员 =====================
representatives = {}
cluster_members = {}
for cid in cluster_ids:
    idx = np.flatnonzero((labels == cid) & (~is_outlier))
    if idx.size == 0:
        continue
    subD = D[np.ix_(idx, idx)]
    medoid_local = idx[np.argmin(subD.sum(axis=1))]
    representatives[int(cid)] = codes[medoid_local]
    cluster_members[int(cid)] = codes[idx]

# ===================== 7) 汇总输出表 =====================
cluster_map_df = pd.DataFrame({
    'index_code': codes,
    'cluster': labels,
    'is_outlier': is_outlier,
    'intra_mean_sim': intra_mean_sim,
    'silhouette': sil
})

# 指数简称绑定（若不存在该列也不会报错）
if '指数代码' in index_info.columns:
    name_map = index_info.set_index('指数代码')
else:
    # 如果你的列名不是“指数代码”，改这里
    name_map = index_info.set_index(index_info.columns[0])

short_name_col = None
for cand in ['指数简称', '指数名称', '名称', 'name', 'short_name']:
    if cand in name_map.columns:
        short_name_col = cand
        break

if short_name_col is not None:
    cluster_map_df = cluster_map_df.join(
        name_map[[short_name_col]].rename(columns={short_name_col: 'index_name'}),
        on='index_code'
    )
else:
    cluster_map_df['index_name'] = ''

print("各簇规模（排除离群后）:")
for cid in sorted(cluster_members.keys()):
    print(f"  Cluster {cid}: {cluster_members[cid].size} 条")

print("\n离群点数量:", int(is_outlier.sum()))

print("\n每个簇的典型指数（Medoid）:")
for cid in sorted(representatives.keys()):
    code = representatives[cid]
    name = cluster_map_df.loc[cluster_map_df['index_code'] == code, 'index_name']
    name = name.iloc[0] if len(name) else ''
    print(f"  Cluster {cid} -> {code} {name}")

# 如需保存聚类结果：
# cluster_map_df.to_csv('index_cluster_result.csv', index=False, encoding='utf-8-sig')

# ===================== 8) pyecharts 绘图：代表指数的历史净值曲线 =====================
rep_codes = [representatives[cid] for cid in sorted(representatives.keys())]
if len(rep_codes) == 0:
    raise RuntimeError("未选出代表指数（可能阈值过严或数据质量问题）。请调低 sil_th/sim_th 或检查数据。")

px_rep = index_nv_wide.loc[:, rep_codes]

# 公共区间：若完全没有交集，则放宽为“≥80% 列非空”的日期
valid_mask_all = ~px_rep.isna()
common = valid_mask_all.all(axis=1)
if not common.any():
    thresh = int(0.8 * px_rep.shape[1])
    common = (valid_mask_all.sum(axis=1) >= thresh)
px_rep = px_rep.loc[common]

# 各列用“各自首个非空值”归一到 1.0（更稳健）
arr = px_rep.to_numpy(copy=False, dtype=float)


def first_valid(x: np.ndarray) -> float:
    m = ~np.isnan(x)
    if not m.any():
        return np.nan
    return x[np.argmax(m)]


scales = np.array([first_valid(arr[:, j]) for j in range(arr.shape[1])], dtype=float)
scales[~np.isfinite(scales)] = np.nan
nav = arr / scales  # 广播除法

# x 轴日期
x_data = [d.strftime("%Y-%m-%d") for d in px_rep.index.to_pydatetime()]


# 名称映射
def code2name(code: str) -> str:
    nm = ''
    if short_name_col is not None and code in name_map.index:
        nm = str(name_map.loc[code, short_name_col])
    return (f"{code} {nm}").strip()


line = Line(init_opts=opts.InitOpts(
    width="1200px",
    height="640px",
    theme=ThemeType.LIGHT,
    page_title="五大聚类代表指数净值")
)
line.add_xaxis(xaxis_data=x_data)

for j, code in enumerate(rep_codes):
    y = nav[:, j]
    if not np.isfinite(y).any():
        continue
    y_list = [float(v) if np.isfinite(v) else None for v in y]
    line.add_yaxis(
        series_name=code2name(code),
        y_axis=y_list,
        is_connect_nones=False,
        is_smooth=False,
        symbol="none",
        linestyle_opts=opts.LineStyleOpts(width=2),
        label_opts=opts.LabelOpts(is_show=False),
    )

line.set_global_opts(
    title_opts=opts.TitleOpts(
        title="五大聚类代表指数的历史净值（起点=1.0）",
        subtitle="代表指数=各簇Medoid；区间按公共/80%可用日期"
    ),
    tooltip_opts=opts.TooltipOpts(
        trigger="axis",
        axis_pointer_type="cross"
    ),
    legend_opts=opts.LegendOpts(pos_top="4%", pos_left="center"),
    datazoom_opts=[
        opts.DataZoomOpts(type_="inside"),
        opts.DataZoomOpts(type_="slider")
    ],
    xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
    yaxis_opts=opts.AxisOpts(type_="value", is_scale=True)  # ← 这里用 is_scale
)

out_path = "代表指数净值_交互图.html"
line.render(out_path)
print(f"已生成交互图: {out_path}")
