# -*- encoding: utf-8 -*-
"""
A04_expand_and_cluster_similar.py

本脚本作为新闻聚类流程的最终整合与扩展步骤。

核心逻辑:
1.  **数据整合**: 动态加载A01, A02, A03的输出，在内存中合并成一个完整的数据集，
    包含原始文本、独立新闻标记、相同新闻组ID。
2.  **选种(Seeding)**: 从每个“相同新闻”组中，选取一篇最长的文章作为该事件的“代表性新闻"(种子)。
3.  **扩展(Expand)**: 
    a. 通过分析“种子”之间的语义相似度，自动学习一个用于区分不同事件的阈值 `T_expand`。
    b. 以每个“种子”为核心，在所有“其他新闻”中搜索，将相似度高于 `T_expand` 的新闻拉入该事件组。
4.  **再聚类(Re-cluster)**: 
    a. 对那些未被任何“种子”吸纳的“其他新闻”，进行独立的内部相似性分析。
    b. 自动学习一个适用于这部分数据的阈值 `T_cluster`，并将它们独立聚类。
5.  **整合与输出**: 合并所有分组结果，生成最终的权威分类结果。

用法:
    python A04_expand_and_cluster_similar.py
"""

import os
import warnings
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

# -------------------- 依赖与辅助函数 --------------------
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    import hnswlib
    HAS_HNSW = True
except ImportError:
    HAS_HNSW = False


def current_time_str() -> str:
    return pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _read_any(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        warnings.warn(f"文件不存在: {path}")
        return None
    try:
        if path.endswith('.parquet'):
            return pd.read_parquet(path)
        else:
            return pd.read_excel(path)
    except Exception as e:
        warnings.warn(f"读取文件失败 {path}: {e}")
        return None


def _encode_sentences(texts: List[str], model_name: str, batch_size: int = 64) -> Optional[np.ndarray]:
    if not HAS_ST:
        warnings.warn("sentence-transformers 未安装，无法进行语义编码。" )
        return None
    try:
        device = 'cuda' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu'
        print(f"{current_time_str()} [INFO] 使用设备: {device} 加载模型: {model_name}")
        model = SentenceTransformer(model_name, device=device)
        emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
        return emb.astype(np.float32)
    except Exception as e:
        warnings.warn(f"加载或编码模型 {model_name} 失败: {e}")
        return None


def find_topk_neighbors(emb: np.ndarray, k: int, use_hnsw: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    n, d = emb.shape
    k_query = min(k + 1, n)
    if use_hnsw and HAS_HNSW and n > k:
        index = hnswlib.Index(space='cosine', dim=d)
        index.init_index(max_elements=n, ef_construction=200, M=48)
        index.add_items(emb, np.arange(n))
        index.set_ef(max(k_query * 2, 100))
        labels, distances = index.knn_query(emb, k=k_query)
        similarities = 1.0 - distances
    else:
        nn = NearestNeighbors(n_neighbors=k_query, metric='cosine', algorithm='brute', n_jobs=-1)
        nn.fit(emb)
        distances, labels = nn.kneighbors(emb)
        similarities = 1.0 - distances

    is_self = (labels == np.arange(n)[:, None])
    non_self_labels = np.full((n, k), -1, dtype=np.int64)
    non_self_sims = np.full((n, k), -1.0, dtype=np.float32)
    for i in range(n):
        mask = ~is_self[i, :]
        row_labels, row_sims = labels[i, mask], similarities[i, mask]
        num_neighbors = min(k, len(row_labels))
        non_self_labels[i, :num_neighbors] = row_labels[:num_neighbors]
        non_self_sims[i, :num_neighbors] = row_sims[:num_neighbors]
    return non_self_labels, non_self_sims


def thr_leftmost_valley(scores: np.ndarray, bins: int = 256, smooth_sigma: float = 2.0) -> Optional[float]:
    v = np.clip(np.asarray(scores, float), 0.0, 1.0)
    if v.size < 20:
        return float(np.quantile(v, 0.8))
    hist, edges = np.histogram(v, bins=bins, range=(0.0, 1.0), density=True)
    r = int(max(1, round(3 * smooth_sigma)))
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-0.5 * (x / smooth_sigma) ** 2);
    k /= k.sum()
    Hs = np.convolve(np.pad(hist, (r, r), mode='reflect'), k, mode='same')[r:-r]
    peaks = [i for i in range(1, bins - 1) if Hs[i - 1] < Hs[i] >= Hs[i + 1]]
    if len(peaks) < 2: return None
    p1, p2 = peaks[0], peaks[1]
    j = p1 + int(np.argmin(Hs[p1:p2 + 1]))
    return float(0.5 * (edges[j] + edges[j + 1]))


def union_find_groups(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    parent = np.arange(n, dtype=np.int32)

    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j: parent[root_i] = root_j

    for u, v in edges: union(u, v)
    for i in range(n): parent[i] = find(i)
    return parent


def plot_hist(scores: np.ndarray, threshold: Optional[float], title: str, out_png: str):
    ensure_dir(out_png)
    plt.figure(figsize=(10, 5))
    plt.hist(np.clip(scores, 0, 1), bins=128, density=True, alpha=0.8, label="Score Distribution")
    if threshold is not None and np.isfinite(threshold):
        plt.axvline(threshold, color='r', linestyle='--', label=f"Threshold: {threshold:.3f}")
    plt.title(title);
    plt.legend();
    plt.grid(True, alpha=0.5);
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# -------------------- 主流程 --------------------

def main(a01_path: str, a02_path: str, a03_path: str, 
         output_dir: str, model_name: str, text_col: str,
         indep_col: str, same_col: str, quick_test: Optional[int]):
    
    os.makedirs(output_dir, exist_ok=True)

    # 1. 数据整合 (Data Synthesis)
    print(f"{current_time_str()} [INFO] 1. 动态整合A01, A02, A03的输出...")
    df_base = _read_any(a01_path)
    if df_base is None:
        print(f"{current_time_str()} [ERROR] 基础文件A01数据加载失败，流程终止。")
        return

    df_indep_info = _read_any(a02_path)
    df_same_info = _read_any(a03_path)

    # 使用基础数据左连接另外两个结果，确保全集
    df = df_base
    if df_indep_info is not None:
        df = pd.merge(df, df_indep_info[[text_col, indep_col]], on=text_col, how='left')
    else:
        df[indep_col] = np.nan # 如果文件不存在，则创建空列

    if df_same_info is not None:
        df = pd.merge(df, df_same_info[[text_col, same_col]], on=text_col, how='left')
    else:
        df[same_col] = np.nan

    # 填充缺失值，确保流程健壮性
    df[indep_col] = df[indep_col].fillna(0) # 假设非-1的都是待定
    
    if quick_test:
        df = df.sample(n=min(quick_test, len(df)), random_state=42).reset_index(drop=True)

    df['original_index'] = df.index
    df_indep = df[df[indep_col] == -1].copy()
    df_work = df[df[indep_col] != -1].reset_index(drop=True)
    print(f"{current_time_str()} [INFO] 总计: {len(df)}, 独立: {len(df_indep)}, 待处理: {len(df_work)}")

    if df_work.empty:
        print(f"{current_time_str()} [INFO] 没有需要处理的数据，流程结束。")
        return

    # 2. 语义编码
    print(f"{current_time_str()} [INFO] 2. 对{len(df_work)}条待处理新闻进行语义编码...")
    embeddings = _encode_sentences(df_work[text_col].tolist(), model_name)
    if embeddings is None: return

    # 3. 选种与扩展 (Seed and Expand)
    print(f"{current_time_str()} [INFO] 3. 从“相同新闻”组中选种并扩展“相似新闻”...")
    df_work_same = df_work.dropna(subset=[same_col])
    seeds = []
    if not df_work_same.empty:
        df_work_same['text_len'] = df_work_same[text_col].str.len()
        seed_indices_in_work = df_work_same.loc[df_work_same.groupby(same_col)['text_len'].idxmax()].index.tolist()
        seeds = df_work.iloc[seed_indices_in_work].copy()
        print(f"{current_time_str()} [INFO] 找到 {len(seeds)} 个“相同新闻”组的代表性种子。")

    edges = []
    if not seeds.empty:
        seed_embeddings = embeddings[seeds.index]
        sim_matrix_seeds = seed_embeddings @ seed_embeddings.T
        np.fill_diagonal(sim_matrix_seeds, -1)
        inter_seed_scores = sim_matrix_seeds[sim_matrix_seeds >= 0]
        T_expand = float(np.quantile(inter_seed_scores, 0.95)) if inter_seed_scores.size > 0 else 0.7
        print(f"{current_time_str()} [INFO] 学习到种子扩展阈值 T_expand = {T_expand:.4f}")
        plot_hist(inter_seed_scores, T_expand, "Inter-Seed Similarity Distribution", 
                  os.path.join(output_dir, "A04_inter_seed_hist.png"))

        sim_to_seeds = embeddings @ seed_embeddings.T
        for i in range(len(df_work)):
            for j_seed_idx, seed_original_idx in enumerate(seeds.index):
                if sim_to_seeds[i, j_seed_idx] > T_expand:
                    edges.append((i, seed_original_idx))

    # 4. 聚类“孤儿”新闻 (Orphan Clustering)
    print(f"{current_time_str()} [INFO] 4. 聚类剩余的“孤儿”新闻...")
    orphan_mask = df_work[same_col].isna()
    df_orphans = df_work[orphan_mask].copy()
    orphan_indices_in_work = df_orphans.index.to_numpy()
    orphan_embeddings = embeddings[orphan_mask]

    if not df_orphans.empty:
        _, orphan_top1_sim = find_topk_neighbors(orphan_embeddings, k=1, use_hnsw=HAS_HNSW)
        T_cluster = thr_leftmost_valley(orphan_top1_sim[:, 0]) or float(np.quantile(orphan_top1_sim[:, 0], 0.85))
        print(f"{current_time_str()} [INFO] 学习到孤儿聚类阈值 T_cluster = {T_cluster:.4f}")
        plot_hist(orphan_top1_sim[:, 0], T_cluster, "Orphan Top-1 Similarity Distribution",
                  os.path.join(output_dir, "A04_orphan_top1_hist.png"))

        orphan_neighbor_idx, orphan_neighbor_sim = find_topk_neighbors(orphan_embeddings, k=5, use_hnsw=HAS_HNSW)
        for i_orphan, orphan_work_idx in enumerate(orphan_indices_in_work):
            for j_neighbor_local_idx, sim in zip(orphan_neighbor_idx[i_orphan], orphan_neighbor_sim[i_orphan]):
                if j_neighbor_local_idx != -1 and sim > T_cluster:
                    neighbor_work_idx = orphan_indices_in_work[j_neighbor_local_idx]
                    if orphan_work_idx < neighbor_work_idx:
                        edges.append((orphan_work_idx, neighbor_work_idx))
                else:
                    break

    # 5. 整合与输出
    print(f"{current_time_str()} [INFO] 5. 整合所有分组并生成最终结果...")
    group_roots = union_find_groups(len(df_work), edges)
    unique_roots, final_group_ids_work = pd.factorize(group_roots)
    df_work['final_group_id'] = final_group_ids_work

    df_work['final_group_type'] = 'similar'
    df_work.loc[df_work[same_col].notna(), 'final_group_type'] = 'same'

    group_counts = df_work['final_group_id'].value_counts()
    single_member_groups = group_counts[group_counts == 1].index
    single_mask = df_work['final_group_id'].isin(single_member_groups)
    df_work.loc[single_mask, 'final_group_type'] = 'independent'
    df_work.loc[single_mask, 'final_group_id'] = -1

    df_final = df.copy()
    df_final['final_group_id'] = -1
    df_final['final_group_type'] = 'independent'

    df_final.loc[df_work['original_index'], ['final_group_id', 'final_group_type']] = \
        df_work[['final_group_id', 'final_group_type']].values

    out_path_base = os.path.join(output_dir, "A04_final_event_groups")
    df_final.drop(columns=['original_index'], errors='ignore').to_parquet(f"{out_path_base}.parquet", index=False)
    df_final.drop(columns=['original_index'], errors='ignore').to_excel(f"{out_path_base}.xlsx", index=False)
    print(f"{current_time_str()} [INFO] 结果已保存到 {output_dir}")
    print(f"{current_time_str()} [INFO] 分类统计:\n{df_final['final_group_type'].value_counts()}")
    print(f"{current_time_str()} [INFO] 流程执行完毕。")


if __name__ == '__main__':
    # 使用命名参数直接调用main函数
    # 请确保以下文件路径是A01, A02, A03脚本的实际输出路径
    main(
        a01_path='A01_test_news_clean.parquet',
        a02_path='A02_independent_news/news_with_groups.xlsx',
        a03_path='A02_news_same_groups_containment.parquet', # 这是A03脚本的输出
        output_dir='A04_final_event_groups',
        model_name='BAAI/bge-large-zh-v1.5',
        text_col='doc_norm',
        indep_col='group', # A02输出的独立新闻列
        same_col='same_group_id', # A03输出的相同新闻列
        quick_test=50000  # 设置为None则处理全部数据
    )
