import os
import math
import hashlib
from typing import List, Tuple, Optional, Iterable, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

try:
    from datasketch import MinHash, MinHashLSHForest

    HAS_LSHFOREST = True
except Exception:
    HAS_LSHFOREST = False
try:
    from datasketch import MinHashLSH

    HAS_LSH = True
except Exception:
    HAS_LSH = False


# ========= 基础：字符 n-gram 指纹（去重集合） =========

def _hash64(b: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), 'little', signed=False)


def char_ngrams(s: str, n: int):
    L = len(s)
    if L <= n:
        if s:
            yield s
        return
    for i in range(L - n + 1):
        yield s[i:i + n]


def shingle_set(text: str, n: int = 5) -> np.ndarray:
    if not isinstance(text, str):
        text = '' if text is None else str(text)
    xs = {_hash64(g.encode('utf-8')) for g in char_ngrams(text, n)}
    if not xs:
        return np.empty(0, dtype=np.uint64)
    arr = np.fromiter(xs, dtype=np.uint64, count=len(xs))
    arr.sort()
    return arr


def inter_size_sorted(a: np.ndarray, b: np.ndarray) -> int:
    i = j = cnt = 0
    na, nb = a.size, b.size
    while i < na and j < nb:
        ai, bj = a[i], b[j]
        if ai == bj:
            cnt += 1;
            i += 1;
            j += 1
        elif ai < bj:
            i += 1
        else:
            j += 1
    return cnt


def jaccard_from_sets(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 and b.size == 0:
        return 0.0
    inter = inter_size_sorted(a, b)
    union = a.size + b.size - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def max_containment(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    inter = inter_size_sorted(a, b)
    denom = max(1, min(a.size, b.size))
    return float(inter / denom)


# ========= MinHash 签名与候选检索 =========

def build_minhash_signatures(texts: List[str], ngram: int = 4, num_perm: int = 256, seed: int = 1) -> List[MinHash]:
    sigs: List[MinHash] = []
    for t in texts:
        m = MinHash(num_perm=num_perm, seed=seed)
        for g in char_ngrams(t, n=ngram):
            m.update(g.encode('utf-8'))
        sigs.append(m)
    return sigs


def build_index_and_query_topk(sigs: List[MinHash], k_neighbors: int = 40):
    n = len(sigs)
    ids = [str(i) for i in range(n)]
    if HAS_LSHFOREST:
        forest = MinHashLSHForest(num_perm=sigs[0].num_perm)
        for i, m in enumerate(sigs):
            forest.add(ids[i], m)
        forest.index()

        def query(i: int) -> List[int]:
            res = forest.query(sigs[i], k_neighbors + 1)
            out = [int(r) for r in res if int(r) != i]
            return out[:k_neighbors]

        return query
    if HAS_LSH:
        lsh = MinHashLSH(threshold=0.5, num_perm=sigs[0].num_perm)
        for i, m in enumerate(sigs):
            lsh.insert(ids[i], m)

        def query(i: int) -> List[int]:
            res = lsh.query(sigs[i])
            out = [int(r) for r in res if int(r) != i]
            if len(out) > k_neighbors:
                out.sort(key=lambda j: -sigs[i].jaccard(sigs[j]))
                return out[:k_neighbors]
            return out

        return query
    from heapq import nlargest
    def query(i: int) -> List[int]:
        scores = [(sigs[i].jaccard(sigs[j]), j) for j in range(n) if j != i]
        return [j for _, j in nlargest(k_neighbors, scores)]

    return query


# ========= 阈值方法（在 top1 上求） =========

def thr_gmm_posterior(scores: np.ndarray, Kmax: int = 4, tau: float = 0.95, random_state: int = 42) -> Optional[float]:
    v = np.clip(scores, 0.0, 1.0).reshape(-1, 1)
    best, best_bic = None, np.inf
    for K in range(2, Kmax + 1):
        gm = GaussianMixture(n_components=K, covariance_type='full',
                             reg_covar=1e-6, max_iter=300, random_state=random_state)
        gm.fit(v)
        bic = gm.bic(v)
        if bic < best_bic:
            best, best_bic = gm, bic
    if best is None:
        return None
    w = best.weights_.ravel()
    mu = best.means_.ravel()
    sd = np.sqrt(best.covariances_.reshape(-1))
    dup_idx = int(np.argmax(mu))

    xs = np.linspace(0.0, 1.0, 2001)
    dens = []
    for k in range(best.n_components):
        var = sd[k] ** 2 + 1e-12
        dens.append(w[k] * np.exp(-0.5 * (xs - mu[k]) ** 2 / var) / np.sqrt(2 * np.pi * var))
    dens = np.vstack(dens)
    post = dens[dup_idx] / (dens.sum(axis=0) + 1e-12)
    idx = np.argmax(post >= tau)
    if post[idx] < tau:
        return None
    return float(xs[idx])


def thr_rightmost_valley(scores: np.ndarray, bins: int = 256, smooth_sigma: float = 2.0, min_span: int = 10) -> \
        Optional[float]:
    v = np.clip(scores, 0.0, 1.0)
    hist, edges = np.histogram(v, bins=bins, range=(0.0, 1.0))
    if smooth_sigma and smooth_sigma > 0:
        r = int(3 * smooth_sigma)
        x = np.arange(-r, r + 1, dtype=float)
        k = np.exp(-0.5 * (x / smooth_sigma) ** 2);
        k /= k.sum()
        hist = np.convolve(hist.astype(float), k, mode='same')
    H = hist
    peaks = np.where((H[1:-1] > H[:-2]) & (H[1:-1] >= H[2:]))[0] + 1
    if peaks.size == 0:
        return None
    rp = peaks.max()
    lp = peaks[peaks < rp]
    left = lp.max() if lp.size else 1
    if rp - left < max(min_span, bins // 50):
        left = max(1, rp - max(min_span, bins // 50))
    seg = H[left:rp + 1]
    if seg.size == 0:
        return None
    vi = int(np.argmin(seg)) + left
    thr = 0.5 * (edges[vi] + edges[vi + 1])
    return float(thr)


def thr_upper_maxgap(scores: np.ndarray, q0: float = 0.6) -> Optional[float]:
    s = np.sort(np.clip(scores, 0.0, 1.0))
    s = s[s >= q0]
    if s.size < 2:
        return None
    gaps = s[1:] - s[:-1]
    i = int(np.argmax(gaps))
    return float(s[i])


def threshold_by_fpr(bg_scores: np.ndarray, k_neighbors: int, fp_per_node: float = 0.05) -> float:
    q = (1.0 - fp_per_node) ** (1.0 / max(1, k_neighbors))
    return float(np.quantile(bg_scores, q))


# ========= 并查集 =========

def union_find_groups(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int8)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb: return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for u, v in edges:
        if u != v: union(u, v)
    for i in range(n):
        parent[i] = find(i)

    roots, inv, counts = np.unique(parent, return_inverse=True, return_counts=True)
    sizes = counts[inv]
    gid = np.where(sizes >= 2, inv, -1).astype(np.int32)
    return gid


# ========= 主流程 =========

def main(input_parquet='test_news_clean.parquet',
         out_parquet='news_same_groups_containment.parquet',
         out_excel='news_same_groups_containment.xlsx',
         hist_png_cont='top1_containment_hist.png',
         hist_png_jacc='top1_jaccard_hist.png',
         ngram_lsh=4,
         ngram_cont=5,
         num_perm=256,
         k_neighbors=40,
         topM=4,
         fp_per_node=0.05,
         neg_pairs=200_000,
         mutual_required=True,
         quick_test=False):
    df = pd.read_parquet(input_parquet)
    if quick_test:
        df = df.iloc[:200].copy()
    if 'doc_norm' not in df.columns:
        if {'title_norm', 'text_norm'}.issubset(df.columns):
            df['doc_norm'] = df['title_norm'].astype(str).str.cat(df['text_norm'].astype(str), sep='。').str.strip()
        else:
            raise ValueError("需要存在 doc_norm 或 (title_norm, text_norm)")

    docs = df['doc_norm'].astype(str).tolist()
    n = len(docs)
    print(f"[INFO] 样本数 n = {n}")

    print(f"[INFO] 构建字符 {ngram_cont}-gram 指纹集合（去重）...")
    sets = [shingle_set(t, n=ngram_cont) for t in docs]

    print(f"[INFO] 构建 MinHash（ngram={ngram_lsh}, num_perm={num_perm}）并建立索引...")
    sigs = build_minhash_signatures(docs, ngram=ngram_lsh, num_perm=num_perm, seed=1)
    query = build_index_and_query_topk(sigs, k_neighbors=k_neighbors)

    print("[INFO] 计算每篇的 top-1 最大包含度 / Jaccard ...")
    top1_cont = np.zeros(n, dtype=np.float32)
    top1_jacc = np.zeros(n, dtype=np.float32)
    cand_lists: Dict[int, List[int]] = {}
    for i in range(n):
        js = query(i)
        cand_lists[i] = js
        if not js:
            continue
        Ai = sets[i]
        best_c = 0.0
        best_j = 0.0
        for j in js:
            Bj = sets[j]
            c = max_containment(Ai, Bj)
            if c > best_c: best_c = c
            jacc = jaccard_from_sets(Ai, Bj)
            if jacc > best_j: best_j = jacc
        top1_cont[i] = best_c
        top1_jacc[i] = best_j

    # 负类分布（用于 FPR 地板）
    m_neg = min(int(neg_pairs), max(10_000, 20 * n))
    print(f"[INFO] 采样负类随机对 m = {m_neg} ...")
    rng = np.random.default_rng(42)
    i_idx = rng.integers(0, n, size=m_neg, endpoint=False)
    j_idx = rng.integers(0, n, size=m_neg, endpoint=False)
    mask = (i_idx != j_idx)
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    neg_cont = np.empty(i_idx.size, dtype=np.float32)
    B = 2048
    w = 0
    for s in range(0, i_idx.size, B):
        e = min(i_idx.size, s + B)
        for k in range(s, e):
            a = sets[i_idx[k]];
            b = sets[j_idx[k]]
            neg_cont[w] = max_containment(a, b)
            w += 1
    neg_cont = neg_cont[:w]

    # ====== 阈值（基于 top1_cont） ======
    scores = top1_cont.copy()
    print("[INFO] 计算自动阈值（基于 top1_cont）...")
    T1 = thr_gmm_posterior(scores, Kmax=4, tau=0.95)
    T2 = thr_rightmost_valley(scores, bins=256, smooth_sigma=2.0, min_span=10)
    T3 = thr_upper_maxgap(scores, q0=0.6)
    T_fpr = threshold_by_fpr(neg_cont, k_neighbors=k_neighbors, fp_per_node=fp_per_node)

    Ts = [t for t in [T1, T2, T3] if t is not None and np.isfinite(t)]
    T_cont = (float(np.median(Ts)) if Ts else float(np.quantile(scores, 0.98)))
    # 分位地板 & 最终阈值
    q90_cont = float(np.quantile(scores, 0.90))
    q98_cont = float(np.quantile(scores, 0.98))
    T_cont = max(T_cont, q90_cont)
    T_final = float(max(T_cont, T_fpr))
    T_final = float(np.clip(T_final, 0.0, 0.995))

    print(f"[INFO] 阈值：GMM_post={T1}, rightmost_valley={T2}, upper_maxgap={T3}, FPR_floor={T_fpr:.6f}")
    print(f"[INFO] q90_cont={q90_cont:.6f}, q98_cont={q98_cont:.6f}, T_cont_fused={T_cont:.6f}, T_final={T_final:.6f}")

    # ====== 画图 1：top1_cont 直方图 + 阈值/分位线 ======
    print(f"[INFO] 绘制直方图（containment） → {hist_png_cont}")
    plt.figure(figsize=(9, 4.8))
    plt.hist(scores, bins=128, range=(0.0, 1.0), density=True, alpha=0.85)
    for name, val in [('GMM_post', T1), ('right_valley', T2), ('upper_gap', T3), ('FPR', T_fpr), ('final', T_final)]:
        if val is not None and np.isfinite(val):
            plt.axvline(val, linestyle='--', linewidth=1.2, label=f"{name}: {val:.3f}")
    plt.xlabel("Top-1 max-containment (character n-gram sets)")
    plt.ylabel("density")
    plt.title("Top-1 Containment & thresholds / quantiles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_png_cont, dpi=150)
    plt.close()

    # ====== Jaccard 护栏阈值与直方图 ======
    # 负类 Jaccard（复用你采的随机对；上面已有 neg_cont）
    neg_jacc = np.empty(i_idx.size, dtype=np.float32)
    w = 0
    for s in range(0, i_idx.size, B):
        e = min(i_idx.size, s + B)
        for k in range(s, e):
            a = sets[i_idx[k]];
            b = sets[j_idx[k]]
            neg_jacc[w] = jaccard_from_sets(a, b)
            w += 1
    neg_jacc = neg_jacc[:w]

    def threshold_by_fpr_new(bg_scores, k_neighbors, fp_per_node=0.05):
        q = (1.0 - fp_per_node) ** (1.0 / max(1, k_neighbors))
        return float(np.quantile(bg_scores, q))

    # 用更紧的护栏（比如单独给 Jaccard 设 ε_j）
    T_jacc_guard = threshold_by_fpr_new(neg_jacc, k_neighbors=k_neighbors, fp_per_node=0.01)
    print(f"[INFO] Jaccard guard by FPR: {T_jacc_guard:.6f}")

    print(f"[INFO] 绘制直方图（jaccard） → {hist_png_jacc}")
    plt.figure(figsize=(9, 4.8))
    plt.hist(top1_jacc, bins=128, range=(0.0, 1.0), density=True, alpha=0.85)
    for name, val in [('fpr', T_jacc_guard)]:
        plt.axvline(val, linestyle='--', linewidth=1.2, label=f"{name}: {val:.3f}")
    plt.xlabel("Top-1 Jaccard (character n-gram sets)")
    plt.ylabel("density")
    plt.title("Top-1 Jaccard & quantiles (guard marked)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_png_jacc, dpi=150)
    plt.close()

    # ====== 建图：互为近邻 + per-node topM + 双阈值 ======
    print("[INFO] 建图（互为近邻 + per-node topM + 双阈值）...")
    edges_set = set()
    for i in range(n):
        js = cand_lists[i]
        if not js:
            continue
        Ai = sets[i]
        conts = np.empty(len(js), dtype=np.float32)
        jaccs = np.empty(len(js), dtype=np.float32)
        for t, j in enumerate(js):
            Bj = sets[j]
            conts[t] = max_containment(Ai, Bj)
            jaccs[t] = jaccard_from_sets(Ai, Bj)
        mutual = np.array([i in cand_lists.get(j, []) for j in js], dtype=bool
                          ) if mutual_required else np.ones(len(js), dtype=bool)
        mask = (conts >= T_final) & (jaccs >= T_jacc_guard) & mutual
        idxs = np.where(mask)[0]
        if idxs.size > topM:
            top_idx = np.argpartition(-conts[idxs], topM - 1)[:topM]
            idxs = idxs[top_idx]
        for t in idxs:
            j = js[t]
            u, v = (i, j) if i < j else (j, i)
            edges_set.add((u, v))

    edges = list(edges_set)
    print(f"[INFO] 保留边数：{len(edges)}")

    groups = union_find_groups(n, edges)
    df_out = df.copy()
    df_out['same_group_id'] = groups.astype(np.int32)
    df_out['same_group_id'] = df_out['same_group_id'].where(df_out['same_group_id'] >= 0, np.nan)

    n_groups = int(df_out['same_group_id'].nunique(dropna=True))
    n_in_groups = int(df_out['same_group_id'].notna().sum())
    print(f"[INFO] 分组完成：{n_groups} 个相同新闻组，共 {n_in_groups} 篇进入组。")

    df_out.to_parquet(out_parquet, index=False)
    try:
        df_out.to_excel(out_excel, index=False)
    except Exception as e:
        print(f"[WARN] 写 Excel 失败：{e}")

    meta = {
        # 'q90_cont': q90_cont, 'q98_cont': q98_cont,
        'T_gmm_posterior': T1, 'T_rightmost_valley': T2, 'T_upper_maxgap': T3,
        'T_fpr_floor': float(T_fpr),
        'T_cont_fused': float(T_cont),
        'T_final_used': float(T_final),
        'T_jacc_guard': float(T_jacc_guard),
        'ngram_lsh': ngram_lsh, 'ngram_cont': ngram_cont,
        'num_perm': num_perm, 'k_neighbors': k_neighbors, 'topM': topM,
        'fp_per_node': fp_per_node, 'mutual_required': mutual_required, 'neg_pairs': int(m_neg)
    }
    pd.DataFrame([meta]).to_csv(os.path.splitext(out_parquet)[0] + "_thresholds.csv", index=False)
    print("[INFO] 全部完成。")


if __name__ == '__main__':
    main(
        input_parquet='test_news_clean.parquet',
        out_parquet='news_same_groups_containment.parquet',
        out_excel='news_same_groups_containment.xlsx',
        hist_png_cont='top1_containment_hist.png',
        hist_png_jacc='top1_jaccard_hist.png',
        ngram_lsh=4,  # LSH 检索 n-gram
        ngram_cont=5,  # 精确验证 n-gram（更严）
        num_perm=256,  # MinHash 置换数
        k_neighbors=40,  # 候选个数（足够大以保证召回）
        topM=4,  # 每节点最多保留的强边数
        fp_per_node=0.05,  # 目标每节点误边概率（FPR 地板）
        neg_pairs=200_000,  # 负样本随机对采样数
        mutual_required=True,  # 只保留互为近邻的边
        quick_test=True  # 调试用截断
    )
