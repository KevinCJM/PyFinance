# -*- encoding: utf-8 -*-
"""
@File: 工具对比01_单纯形网格点生成.py
@Modify Time: 2025/9/19 10:29       
@Author: Kevin-Chen
@Descriptions: 生成网格的方法
"""

import time
import numpy as np
from numba import njit, prange
from itertools import combinations


# 生成单纯形网格点，用于构建资产配置的离散化组合空间
def generate_simplex_grid(n_assets: int, resolution: int):
    """
    生成单纯形网格点，用于构建资产配置的离散化组合空间

    该函数生成所有满足以下条件的资产组合：
    1. 每个资产的权重都是 1/resolution 的整数倍
    2. 所有资产权重之和等于 1
    3. 每个资产权重非负

    参数:
        n_assets (int): 资产类别数量（问题的维度）
        resolution (int): 分辨率，将1划分为resolution份，权重间隔为1/resolution

    返回:
        np.array: 形状为(n_points, n_assets)的数组，每一行代表一个有效的资产组合，
                 每行元素之和为1，每个元素都是1/resolution的整数倍
    """
    # 计算总的插槽数量，用于组合生成
    total_slots = resolution + n_assets - 1
    grid = []

    # 通过组合数学方法生成所有可能的分隔点组合
    for bars in combinations(range(total_slots), n_assets - 1):
        # 添加边界标记以便计算各段长度
        bars = (-1,) + bars + (total_slots,)
        # 计算相邻分隔点之间的间隔，得到各资产的权重
        vec = [bars[i + 1] - bars[i] - 1 for i in range(n_assets)]
        # 将权重转换为实际比例并添加到结果网格中
        grid.append(np.array(vec) / resolution)

    return np.array(grid)


# 使用 Numba 加速网格生成，用于构建资产配置的离散化组合空间
def generate_simplex_grid_numba(n_assets: int, resolution: int):
    """
    使用 Numba(njit+prange) 加速的网格生成函数。

    本函数用于在 n_assets 维空间中生成所有满足权重和为 1 的离散单纯形网格点。
    采用“星与杠”（Stars and Bars）方法枚举非负整数解，并映射为浮点权重向量。
    利用 Numba 的 njit 和并行 prange 加速组合生成过程，适用于高维、高分辨率场景。

    参数:
        n_assets (int): 资产维度，必须 ≥ 2。
        resolution (int): 权重划分精度，即将 1 划分为 resolution 份，步长 Δ = 1/resolution。

    返回:
        np.ndarray: shape 为 (C(resolution+n_assets-1, n_assets-1), n_assets) 的二维数组，
                    每一行表示一个权重向量，所有元素非负且和为 1。

    说明：
    - 旧实现基于 Python 的 itertools.combinations，生成 400 万级组合会非常慢；
    - 新实现预分配结果矩阵，使用组合字典序枚举，并在第一维上并行分块填充；
    - 需 numba==0.49.1（见 requirements.txt）。
    """

    @njit
    def _comb(n: int, k: int) -> np.int64:
        """
        计算组合数 C(n, k)，使用数值稳定的方式避免溢出。

        参数:
            n (int): 总数。
            k (int): 选取数。

        返回:
            np.int64: 组合数 C(n, k)。
        """
        if k < 0 or k > n:
            return np.int64(0)
        if k > n - k:
            k = n - k
        res = 1
        for i in range(1, k + 1):
            res = (res * (n - k + i)) // i
        return np.int64(res)

    @njit
    def _fill_block_given_first(out: np.ndarray, start_row: int,
                                first_bar: int, total_slots: int,
                                k: int, n_assets: int, resolution: int) -> int:
        """
        填充所有以 first_bar 作为第一个“杠”的组合块。

        参数:
            out (np.ndarray): 输出数组。
            start_row (int): 当前块起始行索引。
            first_bar (int): 第一个“杠”的位置。
            total_slots (int): 总槽位数（resolution + n_assets - 1）。
            k (int): 需要放置的“杠”数（n_assets - 1）。
            n_assets (int): 资产数。
            resolution (int): 分辨率。

        返回:
            int: 写入的行数。
        """
        # 处理只剩一个杠的情况（递归终止条件）
        k2 = k - 1
        if k2 == 0:
            prev = -1
            row = start_row
            out[row, 0] = (first_bar - prev - 1) / resolution
            prev = first_bar
            out[row, n_assets - 1] = (total_slots - prev - 1) / resolution
            for j in range(1, n_assets - 1):
                out[row, j] = 0.0
            return 1

        # 初始化剩余“杠”的位置索引
        base = first_bar + 1
        n2 = total_slots - base
        idx2 = np.empty(k2, dtype=np.int64)
        for i in range(k2):
            idx2[i] = i

        row = start_row
        while True:
            # 根据当前“杠”位置构造权重向量
            prev = -1
            out[row, 0] = (first_bar - prev - 1) / resolution
            prev = first_bar
            for j in range(k2):
                b = base + idx2[j]
                out[row, j + 1] = (b - prev - 1) / resolution
                prev = b
            out[row, n_assets - 1] = (total_slots - prev - 1) / resolution
            row += 1

            # 更新下一个组合索引（字典序递增）
            p = k2 - 1
            while p >= 0 and idx2[p] == p + n2 - k2:
                p -= 1
            if p < 0:
                break
            idx2[p] += 1
            for j in range(p + 1, k2):
                idx2[j] = idx2[j - 1] + 1

        return row - start_row

    @njit(parallel=True)
    def _generate(n_assets: int, resolution: int) -> np.ndarray:
        """
        并行生成整个单纯形网格。

        参数:
            n_assets (int): 资产维度。
            resolution (int): 权重划分精度。

        返回:
            np.ndarray: 生成的权重矩阵。
        """
        # 总槽位数和杠数
        total_slots = resolution + n_assets - 1
        k = n_assets - 1

        # 预分配输出数组
        M = _comb(total_slots, k)
        out = np.empty((M, n_assets), dtype=np.float64)

        # 计算每个 first_bar 对应的组合数和偏移量
        first_max = total_slots - k
        F = first_max + 1
        counts = np.empty(F, dtype=np.int64)
        for f in range(F):
            counts[f] = _comb(total_slots - (f + 1), k - 1)

        offsets = np.empty(F + 1, dtype=np.int64)
        offsets[0] = 0
        for i in range(F):
            offsets[i + 1] = offsets[i] + counts[i]

        # 并行填充每个 first_bar 对应的块
        for f in prange(F):
            start = offsets[f]
            _ = _fill_block_given_first(out, start, f, total_slots, k, n_assets, resolution)

        return out

    return _generate(n_assets, resolution)


if __name__ == '__main__':
    s_t_1 = time.time()
    grid1 = generate_simplex_grid(5, 100)
    print(f"生成网格点数量 (itertools): {grid1.shape}, 耗时: {time.time() - s_t_1:.2f} 秒")
    # 前5个网格点
    print(grid1[:5])
    # 后5个网格点
    print(grid1[-5:])
    print("-" * 50)

    s_t_2 = time.time()
    grid2 = generate_simplex_grid_numba(5, 100)
    print(f"生成网格点数量 (numba): {grid2.shape}, 耗时: {time.time() - s_t_2:.2f} 秒")
    # 前5个网格点
    print(grid2[:5])
    # 后5个网格点
    print(grid2[-5:])
