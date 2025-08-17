# -*- encoding: utf-8 -*-
"""
@File: A02_find_independence.py
@Modify Time: 2025/8/17 10:36       
@Author: Kevin-Chen
@Descriptions: 
"""
# -*- coding: utf-8 -*-
"""
Independent News Detection (Two-Stage)
-------------------------------------
Stage-1 (lexical only):
  - char n-gram + Containment / Jaccard (Numba, MinHash-LSH 召回)
  - TF-IDF cosine
  - BM25 cosine (sparse BM25 -> L2 normalize -> cosine)
  - SimHash (char/word) + 1 - Hamming/64 近似召回

Stage-2 (semantic on preselected):
  - Multiple sentence-transformers models (ANN/HNSW)
  - Majority voting with veto guard

Outputs:
  - independent_stage1.parquet / .xlsx
  - independent_final.parquet / .xlsx
  - news_for_dedup.parquet / .xlsx
  - histograms & thresholds csv
"""

import os
import math
import gc
import hashlib
import warnings
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

import jieba
import hnswlib
from sentence_transformers import SentenceTransformer
from numba import njit, prange
from datasketch import MinHash, MinHashLSHForest, MinHashLSH

HAS_JIEBA = True
HAS_HNSW = True
HAS_ST = True
HAS_NUMBA = True
HAS_DATASKETCH = True


# ------------------------- Utils -------------------------

def current_time_str():
    return pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')


def ensure_dir(p):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def plot_hist(scores: np.ndarray, thresholds: Dict[str, Optional[float]],
              title: str, xlabel: str, out_png: str, bins: int = 128):
    ensure_dir(out_png)
    v = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    plt.figure(figsize=(9, 4.8))
    plt.hist(v, bins=bins, range=(0.0, 1.0), density=True, alpha=0.85)
    for name, val in thresholds.items():
        if val is not None and np.isfinite(val):
            plt.axvline(float(val), linestyle='--', linewidth=1.2, label=f"{name}: {float(val):.3f}")
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.title(title)
    if thresholds:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def thr_leftmost_valley(scores: np.ndarray, bins: int = 256,
                        smooth_sigma: float = 2.0, min_span: int = 10) -> Optional[float]:
    """与“最右谷值”对称：在最左峰与其右侧第一个峰之间找谷底。"""
    v = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    hist, edges = np.histogram(v, bins=bins, range=(0.0, 1.0))
    if smooth_sigma and smooth_sigma > 0:
        r = int(3 * smooth_sigma)
        x = np.arange(-r, r + 1, dtype=float)
        k = np.exp(-0.5 * (x / smooth_sigma) ** 2)
        k /= k.sum()
        hist = np.convolve(hist.astype(float), k, mode='same')
    H = hist
    # 寻峰
    peaks = np.where((H[1:-1] > H[:-2]) & (H[1:-1] >= H[2:]))[0] + 1
    if peaks.size < 2:
        return None
    lp = peaks.min()
    rp_candidates = peaks[peaks > lp]
    rp = rp_candidates.min()
    if rp - lp < max(min_span, bins // 50):
        rp = min(bins - 2, lp + max(min_span, bins // 50))
    seg = H[lp:rp + 1]
    if seg.size == 0:
        return None
    vi = int(np.argmin(seg)) + lp
    thr = 0.5 * (edges[vi] + edges[vi + 1])
    return float(thr)


def threshold_by_fpr(bg_scores: np.ndarray, k_neighbors: int, fp_per_node: float = 0.05) -> float:
    """同你原脚本：给定随机负对分布，目标每节点误报率上限 -> 分位数阈值"""
    q = (1.0 - fp_per_node) ** (1.0 / max(1, k_neighbors))
    return float(np.quantile(np.clip(bg_scores, 0.0, 1.0), q))


# ------------------------- Lexical: char n-gram -------------------------

# 为了自包含，这里内嵌必要的 numba 版本；若无 numba，退回纯 Python（速度慢，但可运行）
if HAS_NUMBA:

    @njit
    def _inter_size_sorted_region(S, a0, a1, b0, b1) -> int:
        i = a0
        j = b0
        cnt = 0
        while i < a1 and j < b1:
            ai = S[i]
            bj = S[j]
            if ai == bj:
                cnt += 1
                i += 1
                j += 1
            elif ai < bj:
                i += 1
            else:
                j += 1
        return cnt


    @njit(parallel=True)
    def compute_top1_cont_jacc(S_flat, S_offs, C_flat, C_offs):
        n = S_offs.shape[0] - 1
        top_cont = np.zeros(n, dtype=np.float32)
        top_jacc = np.zeros(n, dtype=np.float32)
        for i in prange(n):
            s = C_offs[i]
            e = C_offs[i + 1]
            if e <= s:
                continue
            ai0 = S_offs[i]
            ai1 = S_offs[i + 1]
            a_len = ai1 - ai0
            best_c = 0.0
            best_j = 0.0
            for p in range(s, e):
                j = int(C_flat[p])
                bj0 = S_offs[j]
                bj1 = S_offs[j + 1]
                inter = _inter_size_sorted_region(S_flat, ai0, ai1, bj0, bj1)
                b_len = bj1 - bj0
                # containment
                min_len = a_len if a_len < b_len else b_len
                denom_c = 1 if min_len <= 0 else min_len
                c = inter / denom_c
                if c > best_c:
                    best_c = c
                # jaccard
                union = a_len + b_len - inter
                jacc = 0.0 if union <= 0 else inter / union
                if jacc > best_j:
                    best_j = jacc
            top_cont[i] = best_c
            top_jacc[i] = best_j
        return top_cont, top_jacc


    @njit(parallel=True)
    def compute_pairs_cont_jacc(S_flat, S_offs, i_idx, j_idx):
        m = i_idx.shape[0]
        cont = np.empty(m, dtype=np.float32)
        jacc = np.empty(m, dtype=np.float32)
        for k in prange(m):
            i = int(i_idx[k])
            j = int(j_idx[k])
            ai0 = S_offs[i]
            ai1 = S_offs[i + 1]
            bj0 = S_offs[j]
            bj1 = S_offs[j + 1]
            inter = _inter_size_sorted_region(S_flat, ai0, ai1, bj0, bj1)
            a_len = ai1 - ai0
            b_len = bj1 - bj0
            # containment
            min_len = a_len if a_len < b_len else b_len
            denom_c = 1 if min_len <= 0 else min_len
            cont[k] = inter / denom_c
            # jaccard
            union = a_len + b_len - inter
            jacc[k] = 0.0 if union <= 0 else inter / union
        return cont, jacc
else:
    def _inter_size_sorted_region(S, a0, a1, b0, b1) -> int:
        i = a0;
        j = b0;
        cnt = 0
        while i < a1 and j < b1:
            ai = S[i];
            bj = S[j]
            if ai == bj:
                cnt += 1;
                i += 1;
                j += 1
            elif ai < bj:
                i += 1
            else:
                j += 1
        return cnt


    def compute_top1_cont_jacc(S_flat, S_offs, C_flat, C_offs):
        n = S_offs.shape[0] - 1
        top_cont = np.zeros(n, dtype=np.float32)
        top_jacc = np.zeros(n, dtype=np.float32)
        for i in range(n):
            s, e = C_offs[i], C_offs[i + 1]
            if e <= s: continue
            ai0, ai1 = S_offs[i], S_offs[i + 1]
            a_len = ai1 - ai0
            best_c = 0.0;
            best_j = 0.0
            for p in range(s, e):
                j = int(C_flat[p])
                bj0, bj1 = S_offs[j], S_offs[j + 1]
                inter = _inter_size_sorted_region(S_flat, ai0, ai1, bj0, bj1)
                b_len = bj1 - bj0
                min_len = a_len if a_len < b_len else b_len
                denom_c = 1 if min_len <= 0 else min_len
                c = inter / denom_c
                if c > best_c: best_c = c
                union = a_len + b_len - inter
                jacc = 0.0 if union <= 0 else inter / union
                if jacc > best_j: best_j = jacc
            top_cont[i], top_jacc[i] = best_c, best_j
        return top_cont, top_jacc


    def compute_pairs_cont_jacc(S_flat, S_offs, i_idx, j_idx):
        m = i_idx.shape[0]
        cont = np.empty(m, dtype=np.float32)
        jacc = np.empty(m, dtype=np.float32)
        for k in range(m):
            i, j = int(i_idx[k]), int(j_idx[k])
            ai0, ai1 = S_offs[i], S_offs[i + 1]
            bj0, bj1 = S_offs[j], S_offs[j + 1]
            inter = _inter_size_sorted_region(S_flat, ai0, ai1, bj0, bj1)
            a_len = ai1 - ai0;
            b_len = bj1 - bj0
            min_len = a_len if a_len < b_len else b_len
            denom_c = 1 if min_len <= 0 else min_len
            cont[k] = inter / denom_c
            union = a_len + b_len - inter
            jacc[k] = 0.0 if union <= 0 else inter / union
        return cont, jacc


def _hash64(b: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), 'little', signed=False)


def _char_ngrams(s: str, n: int):
    L = len(s)
    if L <= n:
        if s:
            yield s
        return
    for i in range(L - n + 1):
        yield s[i:i + n]


def shingle_set_py(text: str, n: int = 5) -> np.ndarray:
    if not isinstance(text, str):
        text = '' if text is None else str(text)
    xs = {_hash64(g.encode('utf-8')) for g in _char_ngrams(text, n)}
    if not xs:
        return np.empty(0, dtype=np.uint64)
    arr = np.fromiter(xs, dtype=np.uint64, count=len(xs))
    arr.sort()
    return arr


def pack_sets_uint64(sets_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(sets_list)
    offs = np.empty(n + 1, dtype=np.int64)
    offs[0] = 0
    total = 0
    for i in range(n):
        total += sets_list[i].size
        offs[i + 1] = total
    flat = np.empty(total, dtype=np.uint64)
    pos = 0
    for arr in sets_list:
        m = arr.size
        if m:
            flat[pos:pos + m] = arr
            pos += m
    return flat, offs


def pack_candidates_int32(cand_lists: Dict[int, List[int]], n: int) -> Tuple[np.ndarray, np.ndarray]:
    offs = np.empty(n + 1, dtype=np.int64)
    offs[0] = 0
    total = 0
    for i in range(n):
        total += len(cand_lists.get(i, []))
        offs[i + 1] = total
    flat = np.empty(total, dtype=np.int32)
    pos = 0
    for i in range(n):
        js = cand_lists.get(i, [])
        if js:
            a = np.asarray(js, dtype=np.int32)
            flat[pos:pos + a.size] = a
            pos += a.size
    return flat, offs


# ---- Unicode codepoints + numba窗口哈希（更快的 shingle） ----
if HAS_NUMBA:

    @njit
    def _fnv1a_u32_slice(arr_u32, start, length) -> np.uint64:
        h = np.uint64(1469598103934665603)
        fnv = np.uint64(1099511628211)
        for t in range(length):
            h ^= np.uint64(arr_u32[start + t])
            h *= fnv
        return h


    @njit(parallel=True)
    def _hash_windows_for_docs(flat_cp, offs_cp, ngram, out_hashes, offs_hash):
        n_docs = offs_cp.shape[0] - 1
        for i in prange(n_docs):
            s = offs_cp[i]
            e = offs_cp[i + 1]
            L = e - s
            dst = offs_hash[i]
            if L <= 0:
                continue
            if L <= ngram:
                out_hashes[dst] = _fnv1a_u32_slice(flat_cp, s, L)
            else:
                m = L - ngram + 1
                for p in range(m):
                    out_hashes[dst + p] = _fnv1a_u32_slice(flat_cp, s + p, ngram)


    def _to_codepoints_flat(docs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        n = len(docs)
        offs = np.empty(n + 1, dtype=np.int64)
        offs[0] = 0
        cps = []
        total = 0
        for i, s in enumerate(docs):
            a = np.fromiter((ord(ch) for ch in s), dtype=np.uint32, count=len(s))
            cps.append(a)
            total += a.size
            offs[i + 1] = total
        if total == 0:
            return np.empty(0, dtype=np.uint32), offs
        flat = np.empty(total, dtype=np.uint32)
        pos = 0
        for a in cps:
            m = a.size
            if m:
                flat[pos:pos + m] = a
                pos += m
        return flat, offs


    def build_shingle_sets_numba(docs: List[str], ngram: int = 5) -> List[np.ndarray]:
        n = len(docs)
        flat_cp, offs_cp = _to_codepoints_flat(docs)
        lens = (offs_cp[1:] - offs_cp[:-1]).astype(np.int64)
        counts = np.where(lens <= 0, 0,
                          np.where(lens <= ngram, 1, lens - ngram + 1)).astype(np.int64)
        offs_hash = np.empty(n + 1, dtype=np.int64)
        offs_hash[0] = 0
        np.cumsum(counts, out=offs_hash[1:])
        total = int(offs_hash[-1])
        out_hashes = np.empty(total, dtype=np.uint64)
        if total:
            _hash_windows_for_docs(flat_cp, offs_cp, ngram, out_hashes, offs_hash)
        sets: List[np.ndarray] = []
        for i in range(n):
            a = out_hashes[offs_hash[i]:offs_hash[i + 1]]
            if a.size == 0:
                sets.append(np.empty(0, dtype=np.uint64))
            else:
                sets.append(np.unique(a))
        return sets
else:
    def build_shingle_sets_numba(docs: List[str], ngram: int = 5) -> List[np.ndarray]:
        return [shingle_set_py(t, n=ngram) for t in docs]


def _make_hash_params(num_perm: int, seed: int = 1):
    rng = np.random.RandomState(seed)
    A = rng.randint(1, np.iinfo(np.uint64).max, size=num_perm, dtype=np.uint64)
    B = rng.randint(0, np.iinfo(np.uint64).max, size=num_perm, dtype=np.uint64)
    A |= np.uint64(1)
    return A, B


if HAS_NUMBA:

    @njit(parallel=True)
    def _minhash_from_sets_numba(S_flat, S_offs, A, B):
        n = S_offs.shape[0] - 1
        num_perm = A.shape[0]
        sigs = np.empty((n, num_perm), dtype=np.uint64)
        MAXU = np.uint64(0xFFFFFFFFFFFFFFFF)
        for i in prange(n):
            for p in range(num_perm):
                sigs[i, p] = MAXU
            s = S_offs[i];
            e = S_offs[i + 1]
            if e <= s:
                continue
            for t in range(s, e):
                x = S_flat[t]
                for p in range(num_perm):
                    hv = A[p] * x + B[p]
                    if hv < sigs[i, p]:
                        sigs[i, p] = hv
        return sigs
else:
    def _minhash_from_sets_numba(S_flat, S_offs, A, B):
        n = S_offs.shape[0] - 1
        num_perm = A.shape[0]
        MAXU = np.uint64(0xFFFFFFFFFFFFFFFF)
        sigs = np.full((n, num_perm), MAXU, dtype=np.uint64)
        for i in range(n):
            s, e = S_offs[i], S_offs[i + 1]
            for t in range(s, e):
                x = S_flat[t]
                for p in range(num_perm):
                    hv = A[p] * x + B[p]
                    if hv < sigs[i, p]:
                        sigs[i, p] = hv
        return sigs


def build_minhash_signatures_numba(texts: List[str], ngram: int = 4, num_perm: int = 128, seed: int = 1) -> List[
    'MinHash']:
    if not HAS_DATASKETCH:
        raise RuntimeError("datasketch 未安装，无法构建 LSH Forest。")
    sets = build_shingle_sets_numba(texts, ngram=ngram)
    S_flat, S_offs = pack_sets_uint64(sets)
    A, B = _make_hash_params(num_perm=num_perm, seed=seed)
    sig_mat = _minhash_from_sets_numba(S_flat, S_offs, A, B)
    sigs: List[MinHash] = []
    n = sig_mat.shape[0]
    for i in range(n):
        m = MinHash(num_perm=num_perm, seed=seed)
        m.hashvalues = sig_mat[i].copy()
        sigs.append(m)
    return sigs, sets, (S_flat, S_offs)


def build_index_and_query_topk(sigs: List['MinHash'], k_neighbors: int = 40):
    n = len(sigs)
    ids = [str(i) for i in range(n)]
    if HAS_DATASKETCH:
        try:
            forest = MinHashLSHForest(num_perm=sigs[0].num_perm)
            for i, m in enumerate(sigs):
                forest.add(ids[i], m)
            forest.index()

            def query(i: int) -> List[int]:
                res = forest.query(sigs[i], k_neighbors + 1)
                out = [int(r) for r in res if int(r) != i]
                return out[:k_neighbors]

            return query
        except Exception:
            pass

    # 退化：暴力
    def query(i: int) -> List[int]:
        scores = [(sigs[i].jaccard(sigs[j]), j) for j in range(n) if j != i]
        scores.sort(reverse=True)
        return [j for _, j in scores[:k_neighbors]]

    return query


def lexical_char_scores(docs: List[str],
                        ngram_lsh=4, ngram_cont=5,
                        num_perm=128, k_neighbors=40,
                        neg_pairs=200_000):
    print(f"{current_time_str()} [INFO] (lex-char) 构建 MinHash & LSH 召回 ...")
    sigs, sets, (S_flat, S_offs) = build_minhash_signatures_numba(docs, ngram=ngram_lsh, num_perm=num_perm, seed=1)
    query = build_index_and_query_topk(sigs, k_neighbors=k_neighbors)
    print(f"{current_time_str()} [INFO] (lex-char) 计算候选并精算 top1(cont/jacc) ...")
    cand_lists: Dict[int, List[int]] = {i: query(i) for i in range(len(docs))}
    C_flat, C_offs = pack_candidates_int32(cand_lists, len(docs))
    top1_cont, top1_jacc = compute_top1_cont_jacc(S_flat, S_offs, C_flat, C_offs)
    # 负对
    rng = np.random.default_rng(42)
    n = len(docs)
    m_neg = min(int(neg_pairs), max(10_000, 20 * n))
    i_idx = rng.integers(0, n, size=m_neg, endpoint=False)
    j_idx = rng.integers(0, n, size=m_neg, endpoint=False)
    mask = (i_idx != j_idx)
    cont_neg, jacc_neg = compute_pairs_cont_jacc(S_flat, S_offs, i_idx[mask].astype(np.int64),
                                                 j_idx[mask].astype(np.int64))
    return top1_cont, top1_jacc, cont_neg, jacc_neg


# ------------------------- Lexical: TF-IDF / BM25 -------------------------

def _cosine_top1_sparse(X: sparse.csr_matrix, n_neighbors: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """返回 top1 余弦与对应索引（排除自身）"""
    # X 已经 L2 归一化
    nn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X)
    dists, inds = nn.kneighbors(X, return_distance=True)
    # 第一个是自己，取第二个
    top1 = 1.0 - dists[:, 1]
    idx1 = inds[:, 1]
    return top1.astype(np.float32), idx1.astype(np.int32)


def _pair_cosine_sparse(X: sparse.csr_matrix, i_idx: np.ndarray, j_idx: np.ndarray) -> np.ndarray:
    """给定若干行对 (i,j)，计算余弦（X 已 L2 归一化）。"""
    # 稀疏行点乘
    vals = np.empty(i_idx.size, dtype=np.float32)
    for k, (i, j) in enumerate(zip(i_idx, j_idx)):
        xi = X.getrow(int(i))
        xj = X.getrow(int(j))
        vals[k] = float(xi.multiply(xj).sum())
    return vals


def lexical_tfidf_scores(docs: List[str], ngram_range=(1, 2), min_df=3, max_features=None,
                         neg_pairs=200_000):
    print(f"{current_time_str()} [INFO] (lex-tfidf) 构建 TF-IDF ...")
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_features=max_features,
                                 norm='l2', use_idf=True)
    X = vectorizer.fit_transform(docs)  # L2 normalized
    print(f"{current_time_str()} [INFO] (lex-tfidf) 最近邻 ...")
    top1, _ = _cosine_top1_sparse(X, n_neighbors=2)
    # 负对
    n = len(docs)
    rng = np.random.default_rng(123)
    m_neg = min(int(neg_pairs), max(10_000, 20 * n))
    i_idx = rng.integers(0, n, size=m_neg, endpoint=False)
    j_idx = rng.integers(0, n, size=m_neg, endpoint=False)
    mask = (i_idx != j_idx)
    neg = _pair_cosine_sparse(X, i_idx[mask], j_idx[mask])
    return top1, neg


def lexical_bm25_scores(docs: List[str], k1=1.2, b=0.75, min_df=3, ngram_range=(1, 2),
                        neg_pairs=200_000):
    print(f"{current_time_str()} [INFO] (lex-bm25) 生成稀疏 BM25 权重 ...")
    cv = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
    TF = cv.fit_transform(docs).astype(np.float32)  # csr
    N, dl = TF.shape[0], np.asarray(TF.sum(axis=1)).ravel()  # 每文档长度
    avgdl = float(dl.mean()) + 1e-9
    # df
    df = np.diff(TF.tocsc().indptr).astype(np.float32)
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)  # shape [V]
    # 构造 BM25 加权矩阵（逐非零）
    TF = TF.tocsr()
    TF.sort_indices()
    data = TF.data
    indptr = TF.indptr
    indices = TF.indices
    bm25_data = np.empty_like(data)
    for i in range(N):
        s, e = indptr[i], indptr[i + 1]
        tf_i = data[s:e]
        cols = indices[s:e]
        denom = tf_i + k1 * (1.0 - b + b * (dl[i] / avgdl))
        bm25_data[s:e] = idf[cols] * (tf_i * (k1 + 1.0) / denom)
    X = sparse.csr_matrix((bm25_data, indices, indptr), shape=TF.shape, dtype=np.float32)
    # L2 normalize
    row_norm = np.sqrt(X.multiply(X).sum(axis=1)).A1 + 1e-12
    X = sparse.diags(1.0 / row_norm).dot(X)
    print(f"{current_time_str()} [INFO] (lex-bm25) 最近邻 ...")
    top1, _ = _cosine_top1_sparse(X, n_neighbors=2)
    # 负对
    n = N
    rng = np.random.default_rng(321)
    m_neg = min(int(neg_pairs), max(10_000, 20 * n))
    i_idx = rng.integers(0, n, size=m_neg, endpoint=False)
    j_idx = rng.integers(0, n, size=m_neg, endpoint=False)
    mask = (i_idx != j_idx)
    neg = _pair_cosine_sparse(X, i_idx[mask], j_idx[mask])
    return top1, neg


# ------------------------- Lexical: SimHash -------------------------

def _tokenize(text: str, level: str = 'char') -> List[str]:
    if level == 'word':
        if HAS_JIEBA:
            return [w for w in jieba.lcut(text) if w.strip()]
        # fallback: 空格分
        return [w for w in text.split() if w.strip()]
    # char
    return list(text)


def _simhash64(tokens: List[str]) -> np.uint64:
    if not tokens:
        return np.uint64(0)
    acc = np.zeros(64, dtype=np.int64)
    for t in tokens:
        h = int.from_bytes(hashlib.blake2b(t.encode('utf-8'), digest_size=8).digest(), 'little', signed=False)
        for b in range(64):
            if (h >> b) & 1:
                acc[b] += 1
            else:
                acc[b] -= 1
    sig = 0
    for b in range(64):
        if acc[b] >= 0:
            sig |= (1 << b)
    return np.uint64(sig)


def _hamming64(a: np.uint64, b: np.uint64) -> int:
    return int((a ^ b).bit_count())


def lexical_simhash_scores(docs: List[str], level: str = 'char',
                           bands: int = 8, band_bits: int = 8,
                           neg_pairs=200_000):
    """简易 LSH：8*8 分段，召回候选再算 Hamming。"""
    print(f"{current_time_str()} [INFO] (lex-simhash) 计算 64bit SimHash ({level}) ...")
    toks = [_tokenize(t, level=level) for t in docs]
    sigs = np.array([_simhash64(ts) for ts in toks], dtype=np.uint64)
    n = len(sigs)
    # 建桶
    print(f"{current_time_str()} [INFO] (lex-simhash) 构建桶 ...")
    buckets: List[Dict[int, List[int]]] = [dict() for _ in range(bands)]
    mask = (1 << band_bits) - 1
    for i, h in enumerate(sigs):
        for b in range(bands):
            key = int((h >> (b * band_bits)) & mask)
            L = buckets[b].setdefault(key, [])
            L.append(i)
    # 每篇取候选
    print(f"{current_time_str()} [INFO] (lex-simhash) 召回并计算 top1 ...")
    top1 = np.zeros(n, dtype=np.float32)
    for i, hi in enumerate(sigs):
        cand = set()
        for b in range(bands):
            key = int((hi >> (b * band_bits)) & mask)
            for j in buckets[b].get(key, []):
                if j != i:
                    cand.add(j)
        best = 0.0
        for j in cand:
            ham = _hamming64(hi, sigs[j])
            sim = 1.0 - ham / 64.0
            if sim > best:
                best = sim
        top1[i] = best
    # 负对
    rng = np.random.default_rng(777)
    m_neg = min(int(neg_pairs), max(10_000, 20 * n))
    i_idx = rng.integers(0, n, size=m_neg, endpoint=False)
    j_idx = rng.integers(0, n, size=m_neg, endpoint=False)
    maskp = (i_idx != j_idx)
    i_idx = i_idx[maskp];
    j_idx = j_idx[maskp]
    neg = np.empty(i_idx.size, dtype=np.float32)
    for k in range(i_idx.size):
        ham = _hamming64(sigs[int(i_idx[k])], sigs[int(j_idx[k])])
        neg[k] = 1.0 - ham / 64.0
    return top1, neg


# ------------------------- Semantic (Stage-2) -------------------------

def _encode_sentences(texts: List[str], model_name: str, batch_size: int = 256) -> Optional[np.ndarray]:
    if not HAS_ST:
        warnings.warn(f"sentence-transformers 未安装，跳过 {model_name}")
        return None
    try:
        device = 'cuda' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu'
        model = SentenceTransformer(model_name, device=device)
        emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
        return emb.astype(np.float32)
    except Exception as e:
        warnings.warn(f"加载/编码模型 {model_name} 失败：{e}")
        return None


def _ann_top1_cosine(emb: np.ndarray, use_hnsw=True) -> np.ndarray:
    n, d = emb.shape
    if use_hnsw and HAS_HNSW:
        idx = hnswlib.Index(space='cosine', dim=d)
        idx.init_index(max_elements=n, ef_construction=200, M=48)
        idx.add_items(emb, np.arange(n))
        idx.set_ef(100)
        lbls, dists = idx.knn_query(emb, k=2)
        top1 = 1.0 - dists[:, 1]
        return top1.astype(np.float32)
    # fallback: brute cosine
    nn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=2, n_jobs=-1)
    nn.fit(emb)
    dists, inds = nn.kneighbors(emb, return_distance=True)
    top1 = 1.0 - dists[:, 1]
    return top1.astype(np.float32)


def semantic_model_scores(texts: List[str], model_name: str,
                          neg_pairs=200_000, use_hnsw=True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    emb = _encode_sentences(texts, model_name)
    if emb is None:
        return None, None
    top1 = _ann_top1_cosine(emb, use_hnsw=use_hnsw)
    # 负对
    n = len(texts)
    rng = np.random.default_rng(2468)
    m_neg = min(int(neg_pairs), max(10_000, 20 * n))
    i_idx = rng.integers(0, n, size=m_neg, endpoint=False)
    j_idx = rng.integers(0, n, size=m_neg, endpoint=False)
    mask = (i_idx != j_idx)
    i_idx = i_idx[mask];
    j_idx = j_idx[mask]
    neg = np.sum(emb[i_idx] * emb[j_idx], axis=1).astype(np.float32)  # 余弦
    return top1, neg


# ------------------------- Threshold Fusion -------------------------

def estimate_indep_threshold(scores: np.ndarray, neg_scores: Optional[np.ndarray],
                             bins=256, smooth_sigma=2.0,
                             k_neighbors=40, fp_per_node=0.05) -> float:
    t_left = thr_leftmost_valley(scores, bins=bins, smooth_sigma=smooth_sigma)
    t_fpr = None
    if neg_scores is not None and neg_scores.size > 0:
        t_fpr = threshold_by_fpr(neg_scores, k_neighbors=k_neighbors, fp_per_node=fp_per_node)
    if t_left is None and t_fpr is None:
        return float(np.quantile(np.clip(scores, 0.0, 1.0), 0.15))
    if t_left is None:
        return float(t_fpr)
    if t_fpr is None:
        return float(t_left)
    return float(max(t_left, t_fpr))


# ------------------------- Main Pipeline -------------------------

def main(input_parquet='test_news_clean.parquet',
         text_col='doc_norm', title_col='title_norm', body_col='text_norm',
         # stage-1 params
         ngram_lsh=4, ngram_cont=5, num_perm=128, k_neighbors=40,
         tfidf_min_df=3, bm25_min_df=3, ngram_range=(1, 2),
         simhash_level='char', simhash_bands=8, simhash_band_bits=8,
         lex_neg_pairs=200_000, lex_delta=0.02,
         # stage-2 params
         sem_model_names=("moka-ai/m3e-base", "BAAI/bge-large-zh-v1.5"),
         sem_use_hnsw=True, sem_neg_pairs=200_000, sem_delta=0.02,
         sem_vote='majority', sem_vote_ratio=0.67, sem_veto_fpr=0.001,
         # outputs
         out_dir='independent_news',
         quick_test=None):
    """
    sem_vote: 'and' | 'majority'
    """
    os.makedirs(out_dir, exist_ok=True)

    # ----- Load data -----
    df = pd.read_parquet(input_parquet)
    if quick_test:
        df = df.iloc[:quick_test].copy()
    if text_col not in df.columns:
        if {title_col, body_col}.issubset(df.columns):
            df[text_col] = df[title_col].astype(str).str.cat(df[body_col].astype(str), sep='。').str.strip()
        else:
            raise ValueError(f"需要存在 {text_col} 或 ({title_col}, {body_col})")
    docs = df[text_col].astype(str).tolist()
    n = len(docs)
    print(f"{current_time_str()} [INFO] 样本数 n = {n}")

    meta_rows = []

    # ===================== Stage-1 Lexical =====================
    print(f"{current_time_str()} [INFO] ===== 阶段1：字面通道 =====")

    # 1) char n-gram containment/jaccard
    try:
        cont1, jacc1, cont_neg, jacc_neg = lexical_char_scores(
            docs, ngram_lsh=ngram_lsh, ngram_cont=ngram_cont, num_perm=num_perm,
            k_neighbors=k_neighbors, neg_pairs=lex_neg_pairs
        )
        lex_char = np.maximum(cont1, jacc1)
        T_char = estimate_indep_threshold(lex_char, np.maximum(cont_neg, jacc_neg),
                                          k_neighbors=k_neighbors, fp_per_node=0.05)
        plot_hist(lex_char, {'left+fpr': T_char}, "Lexical(char-ngrams) Top-1", "score",
                  os.path.join(out_dir, 'hist_lex_char.png'))
        meta_rows.append({'route': 'lex_char', 'T_indep': T_char, 'mean': float(np.mean(lex_char)),
                          'median': float(np.median(lex_char))})
    except Exception as e:
        warnings.warn(f"(lex-char) 失败：{e}")
        lex_char = np.zeros(n, dtype=np.float32)
        T_char = 0.0

    # 2) TF-IDF cosine
    try:
        tfidf1, tfidf_neg = lexical_tfidf_scores(docs, ngram_range=ngram_range, min_df=tfidf_min_df,
                                                 neg_pairs=lex_neg_pairs)
        T_tfidf = estimate_indep_threshold(tfidf1, tfidf_neg, k_neighbors=k_neighbors, fp_per_node=0.05)
        plot_hist(tfidf1, {'left+fpr': T_tfidf}, "Lexical(TF-IDF) Top-1", "cosine",
                  os.path.join(out_dir, 'hist_lex_tfidf.png'))
        meta_rows.append({'route': 'lex_tfidf', 'T_indep': T_tfidf, 'mean': float(np.mean(tfidf1)),
                          'median': float(np.median(tfidf1))})
    except Exception as e:
        warnings.warn(f"(lex-tfidf) 失败：{e}")
        tfidf1 = np.zeros(n, dtype=np.float32)
        T_tfidf = 0.0

    # 3) BM25 cosine
    try:
        bm251, bm25_neg = lexical_bm25_scores(docs, min_df=bm25_min_df, ngram_range=ngram_range,
                                              neg_pairs=lex_neg_pairs)
        T_bm25 = estimate_indep_threshold(bm251, bm25_neg, k_neighbors=k_neighbors, fp_per_node=0.05)
        plot_hist(bm251, {'left+fpr': T_bm25}, "Lexical(BM25) Top-1", "cosine",
                  os.path.join(out_dir, 'hist_lex_bm25.png'))
        meta_rows.append({'route': 'lex_bm25', 'T_indep': T_bm25, 'mean': float(np.mean(bm251)),
                          'median': float(np.median(bm251))})
    except Exception as e:
        warnings.warn(f"(lex-bm25) 失败：{e}")
        bm251 = np.zeros(n, dtype=np.float32)
        T_bm25 = 0.0

    # 4) SimHash
    try:
        sim1, sim_neg = lexical_simhash_scores(docs, level=simhash_level, bands=simhash_bands,
                                               band_bits=simhash_band_bits, neg_pairs=lex_neg_pairs)
        T_sim = estimate_indep_threshold(sim1, sim_neg, k_neighbors=k_neighbors, fp_per_node=0.05)
        plot_hist(sim1, {'left+fpr': T_sim}, f"Lexical(SimHash-{simhash_level}) Top-1", "1 - Hamming/64",
                  os.path.join(out_dir, 'hist_lex_simhash.png'))
        meta_rows.append({'route': f'lex_simhash_{simhash_level}', 'T_indep': T_sim, 'mean': float(np.mean(sim1)),
                          'median': float(np.median(sim1))})
    except Exception as e:
        warnings.warn(f"(lex-simhash) 失败：{e}")
        sim1 = np.zeros(n, dtype=np.float32)
        T_sim = 0.0

    # 预选规则（任一通道判独立）
    indep_stage1_mask = ((lex_char < T_char - lex_delta) |
                         (tfidf1 < T_tfidf - lex_delta) |
                         (bm251 < T_bm25 - lex_delta) |
                         (sim1 < T_sim - lex_delta))

    df_stage1 = df[indep_stage1_mask].copy()
    df_rest_for_sem = df[~indep_stage1_mask].copy()
    print(f"{current_time_str()} [INFO] 阶段1预选：独立候选 {len(df_stage1)} / {n}，其余 {len(df_rest_for_sem)}")

    df_stage1.to_parquet(os.path.join(out_dir, 'independent_stage1.parquet'), index=False)
    df_stage1.to_excel(os.path.join(out_dir, 'independent_stage1.xlsx'), index=False)
    df_rest_for_sem.to_parquet(os.path.join(out_dir, 'rest_for_semantic.parquet'), index=False)
    df_rest_for_sem.to_excel(os.path.join(out_dir, 'rest_for_semantic.xlsx'), index=False)

    # ===================== Stage-2 Semantic =====================
    print(f"{current_time_str()} [INFO] ===== 阶段2：语义通道（在预选池上精筛） =====")
    if df_stage1.empty or not sem_model_names:
        print(f"{current_time_str()} [INFO] 预选池为空或未配置语义模型，直接输出 Stage-1 结果为最终独立。")
        df_indep_final = df_stage1
        df_for_dedup = df_rest_for_sem
    else:
        sem_scores: Dict[str, np.ndarray] = {}
        sem_indep_T: Dict[str, float] = {}
        sem_dup_guard_T: Dict[str, float] = {}
        texts_pre = df_stage1[text_col].astype(str).tolist()
        m = len(texts_pre)

        for name in sem_model_names:
            print(f"{current_time_str()} [INFO] (sem) 模型 {name} ...")
            s_top1, s_neg = semantic_model_scores(texts_pre, model_name=name,
                                                  neg_pairs=sem_neg_pairs, use_hnsw=sem_use_hnsw)
            if s_top1 is None:
                continue
            T_indep = estimate_indep_threshold(s_top1, s_neg, k_neighbors=k_neighbors, fp_per_node=0.05)
            # 右侧否决护栏（更严格）：fp_per_node 更小
            T_dup = threshold_by_fpr(s_neg, k_neighbors=k_neighbors,
                                     fp_per_node=sem_veto_fpr) if s_neg is not None else None
            plot_hist(s_top1, {'left+fpr': T_indep, 'dup_guard': T_dup},
                      f"Semantic({name}) Top-1", "cosine",
                      os.path.join(out_dir, f'hist_sem_{name.replace("/", "_")}.png'))

            sem_scores[name] = s_top1
            sem_indep_T[name] = float(T_indep)
            sem_dup_guard_T[name] = float(T_dup) if T_dup is not None else None
            meta_rows.append({'route': f'sem_{name}', 'T_indep': float(T_indep),
                              'T_dup_guard': float(T_dup) if T_dup is not None else np.nan,
                              'mean': float(np.mean(s_top1)), 'median': float(np.median(s_top1))})

        if not sem_scores:
            warnings.warn("所有语义模型均不可用或失败；将 Stage-1 结果作为最终独立。")
            df_indep_final = df_stage1
            df_for_dedup = df_rest_for_sem
        else:
            # 否决护栏：任一模型超过 dup_guard -> 非独立
            veto = np.zeros(m, dtype=bool)
            for name, s in sem_scores.items():
                t_dup = sem_dup_guard_T.get(name, None)
                if t_dup is not None:
                    veto |= (s >= t_dup)

            # 判定：AND 或 多数投票
            votes = np.zeros((m, len(sem_scores)), dtype=np.uint8)
            for j, (name, s) in enumerate(sem_scores.items()):
                t_indep = sem_indep_T[name]
                votes[:, j] = (s < t_indep - sem_delta).astype(np.uint8)

            if sem_vote == 'and':
                ok = (votes.sum(axis=1) == votes.shape[1])
            else:  # majority
                need = max(1, int(math.ceil(votes.shape[1] * float(sem_vote_ratio))))
                ok = (votes.sum(axis=1) >= need)
            final_mask = ok & (~veto)

            df_indep_final = df_stage1.loc[final_mask].copy()
            # 未通过语义精筛的 + 阶段1未入池的 -> 下一阶段去做去重
            df_for_dedup = pd.concat([df_rest_for_sem, df_stage1.loc[~final_mask]], ignore_index=True)

    # 保存最终结果
    df_indep_final.to_parquet(os.path.join(out_dir, 'independent_final.parquet'), index=False)
    df_indep_final.to_excel(os.path.join(out_dir, 'independent_final.xlsx'), index=False)
    df_for_dedup.to_parquet(os.path.join(out_dir, 'news_for_dedup.parquet'), index=False)
    df_for_dedup.to_excel(os.path.join(out_dir, 'news_for_dedup.xlsx'), index=False)

    # 保存阈值元数据
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(os.path.join(out_dir, 'independent_thresholds_meta.csv'), index=False)

    print(f"{current_time_str()} [INFO] 阶段1独立候选：{len(df_stage1)}")
    print(f"{current_time_str()} [INFO] 最终独立：{len(df_indep_final)}，剩余进入去重：{len(df_for_dedup)}")
    print(f"{current_time_str()} [INFO] 全部完成。输出目录：{out_dir}")


if __name__ == '__main__':
    # 典型执行：保持命名入参
    main(
        input_parquet='A01_test_news_clean.parquet',
        out_dir='A02_independent_news',
        # ----- Stage-1 (lexical) -----
        ngram_lsh=4, ngram_cont=5, num_perm=128, k_neighbors=40,
        tfidf_min_df=3, bm25_min_df=3, ngram_range=(1, 2),
        simhash_level='char', simhash_bands=8, simhash_band_bits=8,
        lex_neg_pairs=200_000, lex_delta=0.02,
        # ----- Stage-2 (semantic) -----
        sem_model_names=("moka-ai/m3e-base", "BAAI/bge-large-zh-v1.5"),
        sem_use_hnsw=True, sem_neg_pairs=200_000, sem_delta=0.02,
        sem_vote='majority', sem_vote_ratio=0.67, sem_veto_fpr=0.001,
        # ----- misc -----
        quick_test=50_000  # 例如 50000 做快速实验
    )
