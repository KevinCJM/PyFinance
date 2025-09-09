# -*- encoding: utf-8 -*-
"""
@File: T01_simplex_grid_generator.py
@Modify Time: 2025/9/9 19:04       
@Author: Kevin-Chen
@Descriptions: 单纯形网格生成器 - 用于生成资产权重的离散组合 - 两种实现（纯 Python + Numba 加速）
"""
import time
import numpy as np
from numba import njit, prange
from itertools import combinations


# ---------- 组合数 C(n, k) ----------
@njit(inline='always')
def comb_nk(n: int, k: int) -> np.int64:
    if k < 0 or k > n:
        return np.int64(0)
    k = min(k, n - k)
    res = np.int64(1)
    # 逐步整除，避免大数溢出（n 适中时 int64 足够）
    for i in range(1, k + 1):
        # res *= (n - k + i); res //= i
        res = (res * np.int64(n - k + i)) // np.int64(i)
    return res


# ---------- 组合反排（lex 次序）：给 rank -> 还原 r 组合 ----------
@njit(inline='always')
def unrank_combination_lex(rank: np.int64, n: int, r: int):
    """
    在按字典序（lexicographic）排序的所有 C(n, r) 组合中，
    给定 rank (0-based) 返回对应的一组 r 个递增下标。
    """
    out = np.empty(r, dtype=np.int64)
    x = 0  # 当前可选的最小值
    for j in range(r):
        # 尝试第 j 个元素，从 x 开始往上试
        for v in range(x, n):
            cnt = comb_nk(n - 1 - v, r - 1 - j)
            if rank < cnt:
                out[j] = v
                x = v + 1
                break
            else:
                rank -= cnt
    return out


# ---------- 生成单纯形网格 ----------
@njit(parallel=True)
def generate_simplex_grid_numba(n_assets: int, resolution: int) -> np.ndarray:
    """
    返回 shape = (C(resolution+n_assets-1, n_assets-1), n_assets) 的权重矩阵。
    每行是一个权重向量，非负、各元素是 k/resolution，且和为 1。
    """
    total_slots = resolution + n_assets - 1
    r = n_assets - 1
    n_rows = comb_nk(total_slots, r)  # 组合总数
    out = np.empty((n_rows, n_assets), dtype=np.float64)

    # 并行遍历每个 rank，反排得到隔板 bars，再转换为权重
    for idx in prange(n_rows):
        bars_core = unrank_combination_lex(np.int64(idx), total_slots, r)

        # 用“哨兵”思想，避免构造 Python tuple：prev=-1, 末尾=total_slots
        prev = -1
        # 前 r 段
        for i in range(r):
            cur = bars_core[i]
            # 段长 = cur - prev - 1
            out[idx, i] = (cur - prev - 1) / resolution
            prev = cur
        # 最后一段（到 total_slots）
        out[idx, n_assets - 1] = (total_slots - prev - 1) / resolution

    return out


def generate_simplex_grid(n_assets: int, resolution: int):
    """
    生成单纯形网格点，用于表示资产权重的离散组合

    该函数生成所有可能的资产权重组合，其中每个权重都是1/resolution的整数倍，
    且所有资产权重之和等于1。这在投资组合优化中常用于生成可行的权重空间。

    参数:
        n_assets (int): 资产类别数量（维度）
        resolution (int): 分辨率，将1划分为resolution份，即Δ=1/resolution

    返回:
        numpy.ndarray: 形状为(n_points, n_assets)的数组，每一行表示一个有效的
                      资产权重组合，所有元素非负且和为1
    """
    total_slots = resolution + n_assets - 1
    grid = []
    # 使用组合数学方法生成所有可能的分割点
    for bars in combinations(range(total_slots), n_assets - 1):
        bars = (-1,) + bars + (total_slots,)
        # 计算相邻分割点之间的间隔作为各资产的权重
        vec = [bars[i + 1] - bars[i] - 1 for i in range(n_assets)]
        grid.append(np.array(vec) / resolution)
    return np.array(grid)


if __name__ == '__main__':
    ''' njit版本 '''
    s_t = time.time()
    print(generate_simplex_grid_numba(5, 100))
    print('Elapsed:', time.time() - s_t)

    ''' 原始版本 '''
    s_t = time.time()
    print(generate_simplex_grid(5, 100))
    print('Elapsed:', time.time() - s_t)
