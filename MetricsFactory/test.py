import time
import numpy as np
import pandas as pd
from numba import float64, njit, prange


@njit(float64[:, :](float64[:, :], float64), parallel=True, cache=True)
def rolling_sum_2d_numba(arr: np.ndarray, window: float) -> np.ndarray:
    """
    计算二维ndarray的滚动和，忽略NaN值。

    参数:
        arr (np.ndarray): 输入的二维ndarray (dtype=float64)。
        window (float): 滚动窗口的大小 (将自动转换为整数)。

    返回:
        np.ndarray: 包含每列滚动和的二维ndarray (dtype=float64)。
    """
    rows, cols = arr.shape
    results = np.empty_like(arr, dtype=np.float64)
    win_int = int(window)

    for j in prange(cols):
        for i in range(rows):
            start = max(0, i - win_int + 1)
            end = i + 1
            window_data = arr[start:end, j]
            valid_values = window_data[~np.isnan(window_data)]
            if valid_values.size >= 1:
                results[i, j] = np.nansum(valid_values)
            else:
                results[i, j] = np.nan
    return results


if __name__ == '__main__':
    price_df = pd.read_parquet('/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/wide_close_df.parquet')

    s_t = time.time()
    print(price_df.rolling(window=10, min_periods=1).sum())
    print('Pandas rolling std time:', time.time() - s_t)

    s_t = time.time()
    print(rolling_sum_2d_numba(price_df.values, 10))
    print('Numba rolling std time:', time.time() - s_t)
