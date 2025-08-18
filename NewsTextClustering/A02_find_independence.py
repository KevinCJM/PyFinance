# -*- encoding: utf-8 -*-
"""
@File: A02_find_independence.py
@Modify Time: 2025/8/17 10:36       
@Author: Kevin-Chen
@Descriptions: 
"""
import traceback
import os
import math
import warnings
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
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

def _to_1d(a):
    return np.asarray(a).ravel()


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


def threshold_by_fpr(bg_scores: np.ndarray, k_neighbors: int, fp_per_node: float = 0.2) -> float:
    """同你原脚本：给定随机负对分布，目标每节点误报率上限 -> 分位数阈值"""
    q = (1.0 - fp_per_node) ** (1.0 / max(1, k_neighbors))
    return float(np.quantile(np.clip(bg_scores, 0.0, 1.0), q))


def _gaussian_kernel(sigma: float) -> np.ndarray:
    r = int(max(1, round(3 * float(sigma))))
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    k /= k.sum()
    return k


def thr_leftmost_valley_smooth(scores: np.ndarray,
                               bins: int = 256,
                               sigma: float = 1.4,  # 小σ，保护左窄峰
                               eps_head: float = 0.01,  # 跳过极靠左的毛刺（x<eps_head）
                               min_peak_rel: float = 0.04,  # 峰相对高度阈（相对全局振幅）
                               min_sep_frac: float = 0.03  # 两峰的最小间隔（相对 bins）
                               ) -> float | None:
    """
    在平滑密度 Hs 上，取“从左到右的第1个主要峰 p1”和“其右第1个主要峰 p2”，
    返回 p1 与 p2 之间的谷底。主要峰的判定：高度 ≥ min_peak_rel * (max(H)-min(H))，
    且中心位置 x>=eps_head，且与上一个峰的距离 ≥ min_sep。
    """
    v = np.clip(np.asarray(scores, float), 0.0, 1.0)
    H, edges = np.histogram(v, bins=bins, range=(0.0, 1.0), density=True)

    # 高斯平滑（反射边界）
    r = int(max(1, round(3 * sigma)))
    xk = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-0.5 * (xk / sigma) ** 2);
    k /= k.sum()
    Hs = np.convolve(np.pad(H, (r, r), mode='reflect'), k, mode='same')[r:-r]

    xs = 0.5 * (edges[:-1] + edges[1:])
    amp = Hs.max() - Hs.min()
    thr_peak = Hs.min() + min_peak_rel * amp
    min_sep = max(1, int(round(min_sep_frac * bins)))

    # 用一阶差分找极值
    d = np.diff(Hs)
    peaks = []
    for i in range(1, bins - 1):
        if d[i - 1] > 0 and d[i] <= 0:  # 上升转下降
            if xs[i] >= eps_head and Hs[i] >= thr_peak:
                # 与上一主峰距离检查
                if not peaks or (i - peaks[-1]) >= min_sep:
                    peaks.append(i)

    if len(peaks) < 2:
        return None

    p1, p2 = peaks[0], peaks[1]
    # p1 与 p2 之间的谷
    j = p1 + int(np.argmin(Hs[p1:p2 + 1]))
    return float(0.5 * (edges[j] + edges[j + 1]))


def plot_hist_with_smooth(scores, thresholds: dict, title, xlabel, out_png,
                          bins=256, sigma=2.5):
    v = np.clip(np.asarray(scores, float), 0.0, 1.0)
    H, edges = np.histogram(v, bins=bins, range=(0.0, 1.0), density=True)
    # 平滑
    k = _gaussian_kernel(sigma)
    r = (len(k) - 1) // 2
    Hs = np.convolve(np.pad(H, (r, r), mode='reflect'), k, mode='same')[r:-r]
    xs = 0.5 * (edges[1:] + edges[:-1])

    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 4.8))
    plt.hist(v, bins=bins, range=(0.0, 1.0), density=True, alpha=0.75)
    plt.plot(xs, Hs, linewidth=1.5)  # 平滑密度曲线
    for name, val in thresholds.items():
        if val is not None and np.isfinite(val):
            plt.axvline(float(val), linestyle='--', linewidth=1.2, label=f"{name}: {float(val):.3f}")
    plt.xlabel(xlabel);
    plt.ylabel("density");
    plt.title(title);
    plt.legend();
    plt.tight_layout();
    plt.savefig(out_png, dpi=150);
    plt.close()


def thr_gmm_valley(scores: np.ndarray,
                   Kmax: int = 4,
                   random_state: int = 42,
                   min_prom_frac: float = 0.03) -> float | None:
    """
    1) 对 logit(s) 做 1D GMM（K=2..Kmax，BIC 选型）
    2) 把混合密度映回 s∈(0,1)：p_s(s)=Σ w_k * N(logit(s); μ_k, σ_k^2) * |dz/ds|
       其中 z=logit(s), |dz/ds|=1/(s*(1-s))
    3) 取“最左两座主峰”之间的谷底作为阈值
    """
    v = np.clip(np.asarray(scores, float), 1e-6, 1 - 1e-6)
    z = np.log(v / (1 - v)).reshape(-1, 1)

    # 拟合 GMM（BIC 选型）
    best, best_bic = None, np.inf
    for K in range(2, Kmax + 1):
        gm = GaussianMixture(n_components=K, covariance_type='full',
                             reg_covar=1e-6, max_iter=500, random_state=random_state)
        gm.fit(z)
        bic = gm.bic(z)
        if bic < best_bic:
            best, best_bic = gm, bic
    if best is None:
        print(f"{current_time_str()} [WARN] (gmm-valley) 输入数据太少，无法拟合 GMM。")
        return None

    # 在 s∈(0,1) 网格上评估混合密度（手算 1D 高斯 pdf，避免私有 API）
    xs = np.linspace(1e-6, 1 - 1e-6, 4001)  # s
    zs = np.log(xs / (1 - xs))  # z=logit(s)
    mu = best.means_.ravel().astype(float)  # (K,)
    # 兼容 full/diag/spherical：都还原成每个分量的一维方差
    cov = best.covariances_
    if cov.ndim == 3:  # full: (K,1,1)
        var = cov.reshape(-1)
    elif cov.ndim == 2:  # diag: (K,1)
        var = cov[:, 0]
    else:  # spherical: (K,)
        var = np.asarray(cov).ravel()
    sd = np.sqrt(var + 1e-12)
    w = best.weights_.ravel().astype(float)

    # p_z(z) = Σ w_k * N(z; μ_k, σ_k^2)
    dens_z = np.zeros_like(xs, dtype=float)
    for k in range(len(w)):
        dens_z += w[k] * (np.exp(-0.5 * ((zs - mu[k]) / sd[k]) ** 2) /
                          (np.sqrt(2 * np.pi) * sd[k]))
    # 变量替换：p_s(s) = p_z(z) * |dz/ds|, 其中 |dz/ds| = 1 / (s*(1-s))
    dens_s = dens_z * (1.0 / (xs * (1.0 - xs)))

    # 找局部峰，并按显著性过滤毛刺
    H = dens_s
    n = H.size
    peaks = [i for i in range(1, n - 1) if H[i - 1] < H[i] >= H[i + 1]]
    if len(peaks) < 2:
        print(f"{current_time_str()} [WARN] (gmm-valley) 峰数太少，无法找到合适的阈值。")
        return None

    def valley_between(a, b):
        if a > b: a, b = b, a
        j = a + int(np.argmin(H[a:b + 1]))
        return j, H[j]

    amp = H.max() - H.min()
    prom = []
    for pi, p in enumerate(peaks):
        lp = peaks[pi - 1] if pi - 1 >= 0 else 0
        rp = peaks[pi + 1] if pi + 1 < len(peaks) else n - 1
        _, lv = valley_between(lp, p) if lp != p else (0, H[:p + 1].min() if p > 0 else H[p])
        _, rv = valley_between(p, rp) if rp != p else (n - 1, H[p:].min())
        prom.append(H[p] - max(lv, rv))
    keep = np.asarray(prom) >= (min_prom_frac * amp)
    peaks = [p for p, k in zip(peaks, keep) if k]
    if len(peaks) < 2:
        print(f"{current_time_str()} [WARN] (gmm-valley) 峰数太少，无法找到合适的阈值。")
        return None

    p1, p2 = peaks[0], peaks[1]
    vi, _ = valley_between(p1, p2)
    return float(xs[vi])


def thr_otsu(scores: np.ndarray, bins: int = 256) -> float | None:
    """
    大津法（Otsu）：在 [0,1] 分箱后最大化类间方差，返回对应阈值。
    """
    v = np.clip(np.asarray(scores, float), 0.0, 1.0)
    hist, edges = np.histogram(v, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    tot = hist.sum()
    if tot <= 0:
        print(f"{current_time_str()} [WARN] (otsu) 输入数据太少，无法计算 Otsu 阈值。")
        return None
    p = hist / tot
    # 用 bin 中心作为灰度
    centers = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(p)  # 累计权重
    mu = np.cumsum(p * centers)  # 累计均值
    mu_T = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom < 1e-12] = np.nan  # 避免除零/极端端点
    sigma_b2 = (mu_T * omega - mu) ** 2 / denom  # 类间方差
    # 避开两端极端 bin
    idx = int(np.nanargmax(sigma_b2[1:-1]) + 1)
    return float(centers[idx])


# ------------------------- Lexical: char n-gram -------------------------
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


def _make_hash_params(num_perm: int, seed: int = 1):
    rng = np.random.RandomState(seed)
    A = rng.randint(1, np.iinfo(np.uint64).max, size=num_perm, dtype=np.uint64)
    B = rng.randint(0, np.iinfo(np.uint64).max, size=num_perm, dtype=np.uint64)
    A |= np.uint64(1)
    return A, B


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
    """
    基于字符级别的MinHash与LSH技术，计算文档间的相似度得分，并生成正负样本对的连续性与Jaccard相似度。

    参数:
        docs (List[str]): 输入的文档列表，每个元素是一个字符串。
        ngram_lsh (int): 用于构建MinHash签名时的n-gram长度，默认为4。
        ngram_cont (int): 用于计算连续性相似度时的n-gram长度，默认为5。
        num_perm (int): MinHash中使用的哈希函数数量，默认为128。
        k_neighbors (int): LSH召回阶段每个查询点返回的近邻数量，默认为40。
        neg_pairs (int): 生成的负样本对数量上限，默认为200,000。

    返回:
        tuple:
            top1_cont (np.ndarray): 每个文档与其最相似邻居的连续性相似度。
            top1_jacc (np.ndarray): 每个文档与其最相似邻居的Jaccard相似度。
            cont_neg (np.ndarray): 负样本对的连续性相似度。
            jacc_neg (np.ndarray): 负样本对的Jaccard相似度。
    """

    print(f"{current_time_str()} [INFO] (lex-char) 构建 MinHash & LSH 召回 ...")
    # 构建MinHash签名和对应的集合表示，并生成扁平化结构用于后续处理
    sigs, sets, (S_flat, S_offs) = build_minhash_signatures_numba(docs, ngram=ngram_lsh, num_perm=num_perm, seed=1)
    # 构建LSH索引并查询每个文档的top-k近邻
    query = build_index_and_query_topk(sigs, k_neighbors=k_neighbors)

    print(f"{current_time_str()} [INFO] (lex-char) 计算候选并精算 top1(cont/jacc) ...")
    # 查询每个文档的候选邻居列表
    cand_lists: Dict[int, List[int]] = {i: query(i) for i in range(len(docs))}
    # 将候选列表打包为扁平数组和偏移数组，便于批量处理
    C_flat, C_offs = pack_candidates_int32(cand_lists, len(docs))
    # 计算每个文档与其最相似邻居（top1）的连续性和Jaccard相似度
    top1_cont, top1_jacc = compute_top1_cont_jacc(S_flat, S_offs, C_flat, C_offs)

    # 负对采样与计算
    rng = np.random.default_rng(42)
    n = len(docs)
    # 控制负样本数量不超过设定值，并保证至少有10,000或20倍文档数的负样本
    m_neg = min(int(neg_pairs), max(10_000, 20 * n))
    # 随机生成负样本对索引
    i_idx = rng.integers(0, n, size=m_neg, endpoint=False)
    j_idx = rng.integers(0, n, size=m_neg, endpoint=False)
    # 过滤掉相同索引对
    mask = (i_idx != j_idx)
    # 计算负样本对的连续性和Jaccard相似度
    cont_neg, jacc_neg = compute_pairs_cont_jacc(S_flat, S_offs, i_idx[mask].astype(np.int64),
                                                 j_idx[mask].astype(np.int64))

    return top1_cont, top1_jacc, cont_neg, jacc_neg


# ------------------------- Lexical: TF-IDF / BM25 -------------------------
def _pair_cosine_sparse(X: sparse.csr_matrix, i_idx: np.ndarray, j_idx: np.ndarray) -> np.ndarray:
    """给定若干行对 (i,j)，计算余弦（X 已 L2 归一化）。"""
    # 稀疏行点乘
    vals = np.empty(i_idx.size, dtype=np.float32)
    for k, (i, j) in enumerate(zip(i_idx, j_idx)):
        xi = X.getrow(int(i))
        xj = X.getrow(int(j))
        vals[k] = float(xi.multiply(xj).sum())
    return vals


# ---- TF-IDF (fixed for Chinese) ----
def lexical_tfidf_scores(docs: List[str],
                         analyzer_kind: str = 'char',  # 'char' 或 'jieba'
                         ngram_range=(2, 4),  # char n-gram 推荐 2~4
                         min_df=3,
                         neg_pairs=200_000):
    print(f"{current_time_str()} [INFO] (lex-tfidf) 构建 TF-IDF (analyzer={analyzer_kind}) ...")

    if analyzer_kind == 'jieba':
        if not HAS_JIEBA:
            raise RuntimeError("未安装 jieba：请改用 analyzer_kind='char' 或安装 jieba")
        vectorizer = TfidfVectorizer(tokenizer=jieba.lcut, token_pattern=None,
                                     ngram_range=(1, 2), min_df=min_df, norm='l2')
    else:
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=ngram_range,
                                     min_df=min_df, norm='l2')

    X = vectorizer.fit_transform(docs)  # L2 normalized (稀疏)

    # 仅非零行做邻居搜索；零行 top1=0，避免 0/0 ⇒ cos=1 的假高峰
    nz_mask = _to_1d(X.getnnz(axis=1)) > 0

    top1 = np.zeros(X.shape[0], dtype=np.float32)
    if nz_mask.any():
        nn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=2, n_jobs=-1)
        nn.fit(X[nz_mask])
        dists, _ = nn.kneighbors(X[nz_mask], return_distance=True)
        top1[nz_mask] = (1.0 - dists[:, 1]).astype(np.float32)

    # 负对：只在非零行里抽样
    rng = np.random.default_rng(123)
    nz_idx = np.flatnonzero(nz_mask)
    if nz_idx.size >= 2:
        m_neg = min(int(neg_pairs), max(10_000, 20 * nz_idx.size))
        i_idx = rng.choice(nz_idx, size=m_neg, replace=True)
        j_idx = rng.choice(nz_idx, size=m_neg, replace=True)
        mask = (i_idx != j_idx)
        neg = _pair_cosine_sparse(X, i_idx[mask], j_idx[mask])
    else:
        neg = np.zeros(0, dtype=np.float32)

    return top1, neg


# ---- BM25 (fixed for Chinese) ----
def lexical_bm25_scores(docs: List[str],
                        analyzer_kind: str = 'char',  # 'char' 或 'jieba'
                        ngram_range=(2, 4),  # char n-gram 推荐 2~4
                        min_df=3,
                        k1=1.2, b=0.75,
                        neg_pairs=200_000):
    print(f"{current_time_str()} [INFO] (lex-bm25) 生成稀疏 BM25 (analyzer={analyzer_kind}) ...")

    if analyzer_kind == 'jieba':
        if not HAS_JIEBA:
            raise RuntimeError("未安装 jieba：请改用 analyzer_kind='char' 或安装 jieba")
        cv = CountVectorizer(tokenizer=jieba.lcut, token_pattern=None,
                             ngram_range=(1, 2), min_df=min_df)
    else:
        cv = CountVectorizer(analyzer='char', ngram_range=ngram_range, min_df=min_df)

    TF = cv.fit_transform(docs).astype(np.float32)  # csr
    N = TF.shape[0]
    dl = _to_1d(TF.sum(axis=1))  # 每文档长度
    avgdl = float(dl.mean()) + 1e-9

    # df & idf
    df = np.diff(TF.tocsc().indptr).astype(np.float32)  # 每列文档频次
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)

    # BM25 权重
    TF = TF.tocsr();
    TF.sort_indices()
    data = TF.data;
    indptr = TF.indptr;
    indices = TF.indices
    bm25_data = np.empty_like(data)
    for i in range(N):
        s, e = indptr[i], indptr[i + 1]
        if s == e:
            continue
        tf_i = data[s:e]
        cols = indices[s:e]
        denom = tf_i + k1 * (1.0 - b + b * (dl[i] / avgdl))
        bm25_data[s:e] = idf[cols] * (tf_i * (k1 + 1.0) / denom)

    X = sparse.csr_matrix((bm25_data, indices, indptr), shape=TF.shape, dtype=np.float32)

    # L2 归一化：对所有行统一缩放（避免切片赋值的稀疏坑）
    row_norm = _to_1d(np.sqrt(X.multiply(X).sum(axis=1)))
    scale = np.where(row_norm > 0, 1.0 / (row_norm + 1e-12), 0.0)
    X = sparse.diags(scale).dot(X)

    # KNN 仅对非零行
    nz_mask = scale > 0
    top1 = np.zeros(N, dtype=np.float32)
    if nz_mask.any():
        nn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=2, n_jobs=-1)
        nn.fit(X[nz_mask])
        dists, _ = nn.kneighbors(X[nz_mask], return_distance=True)
        top1[nz_mask] = (1.0 - dists[:, 1]).astype(np.float32)

    # 负对：只采样非零行
    rng = np.random.default_rng(321)
    nz_idx = np.flatnonzero(nz_mask)
    if nz_idx.size >= 2:
        m_neg = min(int(neg_pairs), max(10_000, 20 * nz_idx.size))
        i_idx = rng.choice(nz_idx, size=m_neg, replace=True)
        j_idx = rng.choice(nz_idx, size=m_neg, replace=True)
        mask = (i_idx != j_idx)
        neg = _pair_cosine_sparse(X, i_idx[mask], j_idx[mask])
    else:
        neg = np.zeros(0, dtype=np.float32)

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
                             bins=256, smooth_sigmas=(1.5, 2.5, 3.0),
                             k_neighbors=40, fp_per_node=None) -> float:
    """
    估计独立阈值，用于区分正负样本的分离点

    该函数通过多种策略计算阈值：首先尝试基于分数分布的最左谷点，如果不可用则使用
    基于负样本的FPR（假阳性率）方法，最后根据多种情况选择最优阈值。

    参数:
        scores: numpy数组，包含所有样本的分数值
        neg_scores: 可选的numpy数组，包含负样本的分数值，用于FPR计算
        bins: int，直方图分箱数量，默认为256
        smooth_sigma: float，平滑高斯核的标准差，默认为2.0
        k_neighbors: int，用于FPR计算的邻居数量，默认为40
        fp_per_node: float，每个节点允许的假阳性率，默认为0.2

    返回:
        float: 计算得到的阈值
    """
    # 计算分数分布最左侧的谷点作为阈值
    # t_left = thr_leftmost_valley(scores, bins=bins, smooth_sigma=smooth_sigma)
    t_left = thr_leftmost_valley_smooth(scores, bins=bins)

    # 如果提供了负样本分数，则计算基于FPR的阈值
    t_fpr = None
    if fp_per_node is not None and neg_scores is not None and neg_scores.size > 0:
        t_fpr = threshold_by_fpr(neg_scores, k_neighbors=k_neighbors, fp_per_node=fp_per_node)

    # 根据不同情况选择返回的阈值
    if t_left is None and t_fpr is None:
        # 如果两种方法都失败，返回分数第15百分位数作为默认阈值
        print(f"{current_time_str()} [INFO] 阈值：无有效阈值，使用第15百分位数作为默认阈值")
        return float(np.quantile(np.clip(scores, 0.0, 1.0), 0.15))
    if t_left is None:
        # 如果只有FPR阈值可用，返回FPR阈值
        print(f"{current_time_str()} [INFO] 阈值：无左谷点阈值，使用FPR阈值={t_fpr:.6f}")
        return float(t_fpr)
    if t_fpr is None:
        # 如果只有左谷点阈值可用，返回左谷点阈值
        print(f"{current_time_str()} [INFO] 阈值：无FPR阈值，使用左谷点阈值={t_left:.6f}")
        return float(t_left)
    # 如果两种阈值都可用，返回两者中的较大值
    print(f"{current_time_str()} [INFO] 阈值：左谷点={t_left:.6f}, FPR={t_fpr:.6f}")
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
        print(f"{current_time_str()} [INFO] (lex-char) char n-gram 完成")
        T_char = estimate_indep_threshold(lex_char, np.maximum(cont_neg, jacc_neg),
                                          k_neighbors=k_neighbors, fp_per_node=None)
        T_gmm = thr_gmm_valley(lex_char, Kmax=4)
        T_otsu = thr_otsu(lex_char, bins=256)
        thresholds = {
            "left-valley": T_char,
            "GMM": T_gmm,
            "Otsu": T_otsu,
        }
        print(f"{current_time_str()} [INFO] (lex-char) 字符 n-gram 阈值 T_char = {T_char:.6f}")
        plot_hist_with_smooth(lex_char, thresholds, "Lexical(char-ngrams) Top-1",
                              "score", os.path.join(out_dir, 'hist_lex_char.png'))
        meta_rows.append({'route': 'lex_char', 'T_indep': T_char, 'mean': float(np.mean(lex_char)),
                          'median': float(np.median(lex_char))})
    except Exception as e:
        warnings.warn(f"(lex-char) 失败")
        print(traceback.format_exc())
        lex_char = np.zeros(n, dtype=np.float32)
        T_char = 0.0

    # 2) TF-IDF cosine
    try:
        tfidf1, tfidf_neg = lexical_tfidf_scores(docs, ngram_range=ngram_range, min_df=tfidf_min_df,
                                                 neg_pairs=lex_neg_pairs)
        print(f"{current_time_str()} [INFO] (lex-tfidf) TF-IDF 完成")
        T_tfidf = estimate_indep_threshold(tfidf1, tfidf_neg, k_neighbors=k_neighbors,
                                           fp_per_node=None)
        T_gmm = thr_gmm_valley(tfidf1, Kmax=4)
        T_otsu = thr_otsu(tfidf1, bins=256)
        thresholds = {
            "left-valley": T_tfidf,
            "GMM": T_gmm,
            "Otsu": T_otsu,
        }
        print(f"{current_time_str()} [INFO] (lex-tfidf) 阈值 T_tfidf = {T_tfidf:.6f}")
        plot_hist_with_smooth(tfidf1, thresholds, "Lexical(TF-IDF) Top-1", "cosine",
                              os.path.join(out_dir, 'hist_lex_tfidf.png'))
        meta_rows.append({'route': 'lex_tfidf', 'T_indep': T_tfidf, 'mean': float(np.mean(tfidf1)),
                          'median': float(np.median(tfidf1))})
    except Exception as e:
        warnings.warn(f"(lex-tfidf) 失败")
        print(traceback.format_exc())
        tfidf1 = np.zeros(n, dtype=np.float32)
        T_tfidf = 0.0

    # 3) BM25 cosine
    try:
        bm251, bm25_neg = lexical_bm25_scores(docs, min_df=bm25_min_df, ngram_range=ngram_range,
                                              neg_pairs=lex_neg_pairs)
        print(f"{current_time_str()} [INFO] (lex-bm25) BM25 完成")
        T_bm25 = estimate_indep_threshold(bm251, bm25_neg, k_neighbors=k_neighbors, fp_per_node=None)
        T_gmm = thr_gmm_valley(bm251, Kmax=4)
        T_otsu = thr_otsu(bm251, bins=256)
        thresholds = {
            "left-valley": T_bm25,
            "GMM": T_gmm,
            "Otsu": T_otsu,
        }
        print(f"{current_time_str()} [INFO] (lex-bm25) 阈值 T_bm25 = {T_bm25:.6f}")
        plot_hist_with_smooth(bm251, thresholds, "Lexical(BM25) Top-1", "cosine",
                              os.path.join(out_dir, 'hist_lex_bm25.png'))
        meta_rows.append({'route': 'lex_bm25', 'T_indep': T_bm25, 'mean': float(np.mean(bm251)),
                          'median': float(np.median(bm251))})
    except Exception as e:
        warnings.warn(f"(lex-bm25) 失败")
        print(traceback.format_exc())
        bm251 = np.zeros(n, dtype=np.float32)
        T_bm25 = 0.0

    # 预选规则（任一通道判独立）
    indep_stage1_mask = ((lex_char < T_char - lex_delta) |
                         (tfidf1 < T_tfidf - lex_delta) |
                         (bm251 < T_bm25 - lex_delta))

    df_stage1 = df[indep_stage1_mask].copy()
    df_rest_for_sem = df[~indep_stage1_mask].copy()
    print(f"{current_time_str()} [INFO] 阶段1预选：独立候选 {len(df_stage1)} / {n}，其余 {len(df_rest_for_sem)}")

    # ===================== Stage-2 Semantic =====================
    print(f"{current_time_str()} [INFO] ===== 阶段2：语义通道（在预选池上精筛） =====")
    if df_stage1.empty or not sem_model_names:
        print(f"{current_time_str()} [INFO] 预选池为空或未配置语义模型，直接输出 Stage-1 结果为最终独立。")
        df_indep_final = df_stage1
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
            T_indep = estimate_indep_threshold(s_top1, s_neg, k_neighbors=k_neighbors, fp_per_node=0.1)
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

    # ===== 只输出一个 Excel/Parquet，带 group 字段 =====
    mask_indep = pd.Series(False, index=df.index)
    # group 规则：独立新闻 = -1，其他 = NaN（后续“相同/相似新闻”再填充具体组号）
    if not df_indep_final.empty:
        mask_indep.loc[df_indep_final.index] = True

    df_out = df.copy()
    df_out['group'] = np.nan
    df_out.loc[mask_indep, 'group'] = -1

    # 仅输出一个 Excel（你需要的话，也可以顺带保存一个同构的 parquet）
    out_xlsx = os.path.join(out_dir, 'news_with_groups.xlsx')
    df_out.to_excel(out_xlsx, index=False)

    # 阈值元数据仍然保留，便于追溯
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(os.path.join(out_dir, 'independent_thresholds_meta.csv'), index=False)

    print(f"{current_time_str()} [INFO] 阶段1独立候选：{len(df_stage1)}")
    print(f"{current_time_str()} [INFO] 最终独立：{int(mask_indep.sum())}，其余待后续分组：{int((~mask_indep).sum())}")
    print(f"{current_time_str()} [INFO] 单表输出完成：{out_xlsx}")


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
        quick_test=1_000  # 例如 50000 做快速实验
    )
