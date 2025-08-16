import os
import hashlib
import numpy as np
import pandas as pd
from heapq import nlargest
from numba import njit, prange
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from typing import List, Tuple, Optional, Dict

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


def current_time_str():
    return pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')


# -------- CSR 打包：把变长的指纹集 / 候选列表压平成一维数组 + 偏移量 --------
def pack_sets_uint64(sets_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    # sets_list：每篇文档的去重升序 uint64 指纹集合
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


# -------- Numba 核心：两指针交集 + top1 计算（并行） --------
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
    """
    S_flat: 所有文档的指纹串联数组 (uint64)
    S_offs: 每文档在 S_flat 的 [start,end) 偏移 (int64), 长度 n+1
    C_flat: 所有候选的串联数组 (int32)
    C_offs: 每文档候选的 [start,end) 偏移 (int64), 长度 n+1
    返回：top1_cont, top1_jacc （float32）
    """
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


# ---------- 文本 -> Unicode 码点扁平化 ----------
def _to_codepoints_flat(docs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回:
      flat_cp: 所有文档拼接后的 uint32 码点数组
      offs_cp: 每文档在 flat_cp 中的 [start, end) 偏移, 长度 n+1
    """
    n = len(docs)
    offs = np.empty(n + 1, dtype=np.int64)
    offs[0] = 0
    cps = []
    total = 0
    for i, s in enumerate(docs):
        # Python 端转码：对 CJK 安全，逐字符 ord
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


# ---------- Numba: FNV-1a 64bit 对 u32 序列做滑窗哈希 ----------
@njit
def _fnv1a_u32_slice(arr_u32, start, length) -> np.uint64:
    h = np.uint64(1469598103934665603)  # FNV offset basis
    fnv = np.uint64(1099511628211)  # FNV prime
    for t in range(length):
        h ^= np.uint64(arr_u32[start + t])
        h *= fnv  # uint64 溢出自动按 2^64 wrap
    return h


@njit(parallel=True)
def _hash_windows_for_docs(flat_cp, offs_cp, ngram, out_hashes, offs_hash):
    """
    对每个文档并行计算所有长度为 ngram 的窗口哈希。
    规则与原 char_ngrams 一致：若 L==0 -> 无片段；若 0<L<=ngram -> 仅一个窗口(整篇)。
    结果写入 out_hashes 的每个文档片段 [offs_hash[i], offs_hash[i+1])
    """
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


def build_shingle_sets_numba(docs: List[str], ngram: int = 5) -> List[np.ndarray]:
    """
    返回与旧版 shingle_set 一致的结果类型：每篇是升序去重后的 uint64 数组。
    """
    n = len(docs)
    flat_cp, offs_cp = _to_codepoints_flat(docs)

    # 先算每篇窗口数用于一次性分配
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

    # 分文档 unique & sort（C 端实现，足够快）
    sets: List[np.ndarray] = []
    for i in range(n):
        a = out_hashes[offs_hash[i]:offs_hash[i + 1]]
        if a.size == 0:
            sets.append(np.empty(0, dtype=np.uint64))
        else:
            sets.append(np.unique(a))  # unique 本身会排序
    return sets


# ---------- 把 sets/candidates 压平成 CSR ----------


def pack_candidates_int32_sorted(cand_lists, n):
    # 排序去重，便于 binary search 做 mutual 检查
    offs = np.empty(n + 1, dtype=np.int64)
    offs[0] = 0
    tot = 0
    tmp = []
    for i in range(n):
        js = cand_lists.get(i, [])
        if len(js):
            a = np.asarray(js, dtype=np.int32)
            a = np.unique(a)  # 排序 + 去重
            tmp.append(a)
            tot += a.size
        else:
            tmp.append(np.empty(0, dtype=np.int32))
        offs[i + 1] = tot
    flat = np.empty(tot, dtype=np.int32)
    p = 0
    for a in tmp:
        m = a.size
        if m:
            flat[p:p + m] = a
            p += m
    return flat, offs


# ---------- 低层两指针交集（操作扁平数组的片段） ----------
@njit
def _inter_size_sorted_region(S, a0, a1, b0, b1):
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


@njit
def _contains_sorted(A, s, e, x):
    lo = s
    hi = e - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        v = A[mid]
        if v == x:
            return True
        elif v < x:
            lo = mid + 1
        else:
            hi = mid - 1
    return False


# ---------- Numba 并行：筛边 + per-node topM ----------
@njit(parallel=True)
def build_edges_numba(S_flat, S_offs, C_flat, C_offs,
                      T_cont, T_jacc, topM, mutual_required):
    """
    返回两个长度为 n*topM 的数组 (u,v)，多余位置为 -1
    只写 i<j 的边，天然去重
    """
    n = S_offs.shape[0] - 1
    edges_u = np.full(n * topM, -1, dtype=np.int32)
    edges_v = np.full(n * topM, -1, dtype=np.int32)

    for i in prange(n):
        ai0 = S_offs[i]
        ai1 = S_offs[i + 1]
        a_len = ai1 - ai0
        cs = C_offs[i]
        ce = C_offs[i + 1]

        # 本节点的 topM（按 containment 排序）
        best_js = np.full(topM, -1, dtype=np.int32)
        best_sc = np.full(topM, -1.0, dtype=np.float32)

        for p in range(cs, ce):
            j = int(C_flat[p])
            # 只让小编号指向大编号，避免重复边
            if j <= i:
                continue

            bj0 = S_offs[j]
            bj1 = S_offs[j + 1]
            inter = _inter_size_sorted_region(S_flat, ai0, ai1, bj0, bj1)
            b_len = bj1 - bj0

            # containment / jaccard
            min_len = a_len if a_len < b_len else b_len
            denom_c = 1 if min_len <= 0 else min_len
            c = inter / denom_c
            union = a_len + b_len - inter
            jacc = 0.0 if union <= 0 else inter / union

            if c < T_cont or jacc < T_jacc:
                continue

            if mutual_required:
                # 检查 i 是否出现在 j 的候选中（C_flat[C_offs[j]:C_offs[j+1]) 是升序）
                if not _contains_sorted(C_flat, C_offs[j], C_offs[j + 1], i):
                    continue

            # 线性维护本节点 topM（M 很小，线性足够快）
            placed = False
            for k in range(topM):
                if best_js[k] == -1:
                    best_js[k] = j
                    best_sc[k] = c
                    placed = True
                    break
            if not placed:
                # 找出当前最小的一个，若 c 更大则替换
                minpos = 0
                minval = best_sc[0]
                for k in range(1, topM):
                    if best_sc[k] < minval:
                        minval = best_sc[k]
                        minpos = k
                if c > minval:
                    best_js[minpos] = j
                    best_sc[minpos] = c

        # 写回（每节点最多 topM 条，与 i<j 规则一起避免重复）
        base = i * topM
        outc = 0
        for k in range(topM):
            j = best_js[k]
            if j != -1:
                edges_u[base + outc] = i
                edges_v[base + outc] = j
                outc += 1

    return edges_u, edges_v


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
            cnt += 1
            i += 1
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
        k = np.exp(-0.5 * (x / smooth_sigma) ** 2)
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
        if u != v:
            union(u, v)
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
         quick_test=None):
    df = pd.read_parquet(input_parquet)
    if quick_test:
        df = df.iloc[:quick_test].copy()
    if 'doc_norm' not in df.columns:
        if {'title_norm', 'text_norm'}.issubset(df.columns):
            df['doc_norm'] = df['title_norm'].astype(str).str.cat(df['text_norm'].astype(str), sep='。').str.strip()
        else:
            raise ValueError("需要存在 doc_norm 或 (title_norm, text_norm)")

    docs = df['doc_norm'].astype(str).tolist()
    n = len(docs)
    print(f"{current_time_str()} [INFO] 样本数 n = {n}")

    print(f"{current_time_str()} [INFO] 构建字符 {ngram_cont}-gram 指纹集合（去重）...")
    sets = build_shingle_sets_numba(docs, ngram=ngram_cont)

    print(f"{current_time_str()} [INFO] 构建 MinHash（ngram={ngram_lsh}, num_perm={num_perm}）并建立索引...")
    sigs = build_minhash_signatures(docs, ngram=ngram_lsh, num_perm=num_perm, seed=1)
    query = build_index_and_query_topk(sigs, k_neighbors=k_neighbors)

    print(f"{current_time_str()} [INFO] 计算候选（仅召回，不算分数）...")
    cand_lists: Dict[int, List[int]] = {}
    for i in range(n):
        cand_lists[i] = query(i)

    print(f"{current_time_str()} [INFO] 打包 CSR 结构并并行计算 top-1 分数 ...")
    S_flat, S_offs = pack_sets_uint64(sets)  # 指纹集合打包
    C_flat, C_offs = pack_candidates_int32(cand_lists, n)  # 候选打包
    top1_cont, top1_jacc = compute_top1_cont_jacc(S_flat, S_offs, C_flat, C_offs)

    # 负类分布（用于 FPR 地板）
    m_neg = min(int(neg_pairs), max(10_000, 20 * n))
    print(f"{current_time_str()} [INFO] 采样负类随机对 m = {m_neg} ...")
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
            a = sets[i_idx[k]]
            b = sets[j_idx[k]]
            neg_cont[w] = max_containment(a, b)
            w += 1
    neg_cont = neg_cont[:w]

    # ====== 阈值（基于 top1_cont） ======
    scores = top1_cont.copy()  # 复制top1_cont得分用于后续阈值计算
    print(f"{current_time_str()} [INFO] 计算自动阈值（基于 top1_cont）...")

    # 使用多种方法计算候选阈值
    T1 = thr_gmm_posterior(scores, Kmax=4, tau=0.95)  # GMM后验概率方法计算阈值
    T2 = thr_rightmost_valley(scores, bins=256, smooth_sigma=2.0, min_span=10)  # 最右谷值方法计算阈值
    T3 = thr_upper_maxgap(scores, q0=0.6)  # 上界最大间隔方法计算阈值
    T_fpr = threshold_by_fpr(neg_cont, k_neighbors=k_neighbors, fp_per_node=fp_per_node)  # 基于FPR计算阈值下限

    # # 过滤无效阈值并计算融合阈值
    # Ts = [t for t in [T1, T2, T3] if t is not None and np.isfinite(t)]
    # # 如果有有效阈值则取中位数，否则使用98%分位数作为候选阈值
    # T_cont = (float(np.median(Ts)) if Ts else float(np.quantile(scores, 0.98)))
    #
    # # 计算90%和98%分位数作为阈值下限参考
    # q90_cont = float(np.quantile(scores, 0.90))
    # q98_cont = float(np.quantile(scores, 0.98))
    # # 确保T_cont不低于90%分位数
    # T_cont = max(T_cont, q90_cont)
    # # 最终阈值取T_cont和T_fpr中的较大值
    # T_final = float(max(T_cont, T_fpr))
    # # 将最终阈值限制在合理范围内
    # T_final = float(np.clip(T_final, 0.0, 0.995))
    #
    # # 打印各方法计算的阈值信息
    # print(f"{current_time_str()} [INFO] 阈值：GMM_post={T1}, rightmost_valley={T2}, "
    #       f"upper_maxgap={T3}, FPR_floor={T_fpr:.6f}")
    # print(f"{current_time_str()} [INFO] q90_cont={q90_cont:.6f}, q98_cont={q98_cont:.6f}, "
    #       f"T_cont_fused={T_cont:.6f}")
    T_final = thr_rightmost_valley(scores, bins=256, smooth_sigma=2.0, min_span=10)
    print(f"{current_time_str()} [INFO] 阈值 T_final={T_final:.6f}")

    # ====== 画图 1：top1_cont 直方图 + 阈值/分位线 ======
    print(f"{current_time_str()} [INFO] 绘制直方图（containment） → {hist_png_cont}")
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
    neg_jacc = np.empty(i_idx.size, dtype=np.float32)
    w = 0
    for s in range(0, i_idx.size, B):
        e = min(i_idx.size, s + B)
        for k in range(s, e):
            a = sets[i_idx[k]]
            b = sets[j_idx[k]]
            neg_jacc[w] = jaccard_from_sets(a, b)
            w += 1
    neg_jacc = neg_jacc[:w]

    def threshold_by_fpr_new(bg_scores, k_neighbors, fp_per_node=0.01):
        q = (1.0 - fp_per_node) ** (1.0 / max(1, k_neighbors))
        return float(np.quantile(bg_scores, q))

    # 用更紧的护栏（比如单独给 Jaccard 设 ε_j）
    T_jacc_guard = threshold_by_fpr_new(neg_jacc, k_neighbors=k_neighbors, fp_per_node=0.001)
    print(f"{current_time_str()} [INFO] Jaccard guard by FPR: {T_jacc_guard:.6f}")

    print(f"{current_time_str()} [INFO] 绘制直方图（jaccard） → {hist_png_jacc}")
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
    print(f"{current_time_str()} [INFO] 建图（numba 并行，互为近邻 + per-node topM + 双阈值）...")

    # 先把 sets / candidates 打包成 CSR
    S_flat, S_offs = pack_sets_uint64(sets)
    C_flat, C_offs = pack_candidates_int32_sorted(cand_lists, n)

    # Numba 并行筛边（只产出 i<j 的边，天然去重）
    edges_u, edges_v = build_edges_numba(
        S_flat, S_offs, C_flat, C_offs,
        T_cont=T_final, T_jacc=T_jacc_guard,
        topM=int(topM), mutual_required=bool(mutual_required)
    )

    # 回到 Python 侧，整理有效边
    mask = edges_u >= 0
    edges = [(int(u), int(v)) for u, v in zip(edges_u[mask], edges_v[mask])]
    print(f"{current_time_str()} [INFO] 保留边数：{len(edges)}")

    groups = union_find_groups(n, edges)
    df_out = df.copy()
    df_out['same_group_id'] = groups.astype(np.int32)
    df_out['same_group_id'] = df_out['same_group_id'].where(df_out['same_group_id'] >= 0, np.nan)

    n_groups = int(df_out['same_group_id'].nunique(dropna=True))
    n_in_groups = int(df_out['same_group_id'].notna().sum())
    print(f"{current_time_str()} [INFO] 分组完成：{n_groups} 个相同新闻组，共 {n_in_groups} 篇进入组。")

    df_out.to_parquet(out_parquet, index=False)
    print(f"{current_time_str()} [INFO] 结果保存 → {out_parquet}")
    df_out.to_excel(out_excel, index=False)
    print(f"{current_time_str()} [INFO] 结果保存 → {out_excel}")

    meta = {
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
    print(f"{current_time_str()} [INFO] 全部完成。")


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
        quick_test=50_000  # 调试用的样本数量 (None则表示全量)
    )
