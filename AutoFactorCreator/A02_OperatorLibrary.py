# -*- encoding: utf-8 -*-
"""
@File: A02_OperatorLibrary.py
@Modify Time: 2025/7/10 09:49       
@Author: Kevin-Chen
@Descriptions: 基础算子, 入参和出参必须是(np.ndarray, float, int), 必须支持nan的计算
算子函数的注释必须遵循下面的格式:
    功能描述: 计算a的p次幂。

    参数:
        a (Union[np.ndarray, pd.DataFrame, float, int]): 基数，可以是NumPy数组、浮点数或整数。
        p (Union[np.ndarray, pd.DataFrame, float, int]): 指数，可以是NumPy数组、浮点数或整数。

    返回:
        Union[np.ndarray, pd.DataFrame, float, int]: 幂运算的结果，类型与输入a和p的类型兼容。
"""

import numpy as np
from typing import Union, Optional, Sequence


# --- 基础数学算子 ---

def add(a: Union[np.ndarray, float, int], b: Union[np.ndarray, float, int]) -> Union[np.ndarray, float, int]:
    """
    功能描述: 执行两个输入之间的加法运算。

    参数:
        a (Union[np.ndarray, float, int]): 第一个加数。
        b (Union[np.ndarray, float, int]): 第二个加数。

    返回:
        Union[np.ndarray, float, int]: 两数之和。
    """
    return np.add(a, b)


def subtract(a: Union[np.ndarray, float, int], b: Union[np.ndarray, float, int]) -> Union[np.ndarray, float, int]:
    """
    功能描述: 执行两个输入之间的减法运算。

    参数:
        a (Union[np.ndarray, float, int]): 被减数。
        b (Union[np.ndarray, float, int]): 减数。

    返回:
        Union[np.ndarray, float, int]: 两数之差。
    """
    return np.subtract(a, b)


def multiply(a: Union[np.ndarray, float, int], b: Union[np.ndarray, float, int]) -> Union[np.ndarray, float, int]:
    """
    功能描述: 执行两个输入之间的乘法运算。

    参数:
        a (Union[np.ndarray, float, int]): 第一个乘数。
        b (Union[np.ndarray, float, int]): 第二个乘数。

    返回:
        Union[np.ndarray, float, int]: 两数之积。
    """
    return np.multiply(a, b)


def divide(a: Union[np.ndarray, float, int], b: Union[np.ndarray, float, int]) -> Union[np.ndarray, float, int]:
    """
    功能描述: 执行两个输入之间的除法运算，并处理除零情况。

    参数:
        a (Union[np.ndarray, float, int]): 被除数。
        b (Union[np.ndarray, float, int]): 除数。

    返回:
        Union[np.ndarray, float, int]: 两数之商，除零结果为NaN。
    """
    result = np.divide(a, b)
    # 用NaN替换除零产生的无限值
    return np.where(np.isinf(result), np.nan, result)


def log(a: Union[np.ndarray, float, int]) -> Union[np.ndarray, float, int]:
    """
    功能描述: 计算输入值的自然对数，并处理非正数取对数的情况。

    参数:
        a (Union[np.ndarray, float, int]): 输入值，可以是NumPy数组、浮点数或整数。

    返回:
        Union[np.ndarray, float, int]: 自然对数运算的结果。当输入值非正时，对应位置的结果为NaN。
    """
    return np.log(a, out=np.full_like(a, np.nan), where=a > 0)


def abs_val(a: Union[np.ndarray, float, int]) -> Union[np.ndarray, float, int]:
    """
    功能描述: 计算输入值的绝对值。

    参数:
        a (Union[np.ndarray, float, int]): 输入值，可以是NumPy数组、浮点数或整数。

    返回:
        Union[np.ndarray, float, int]: 绝对值运算的结果，类型与输入a的类型兼容。
    """
    return np.abs(a)


def power(a: Union[np.ndarray, float, int], p: Union[np.ndarray, float, int]) -> Union[
    np.ndarray, float, int]:
    """
    功能描述: 计算a的p次幂。

    参数:
        a (Union[np.ndarray, float, int]): 基数，可以是NumPy数组、浮点数或整数。
        p (Union[np.ndarray, float, int]): 指数，可以是NumPy数组、浮点数或整数。

    返回:
        Union[np.ndarray, float, int]: 幂运算的结果，类型与输入a和p的类型兼容。
    """
    return np.power(a, p)


def sqrt(a: Union[np.ndarray, float, int]) -> Union[np.ndarray, float, int]:
    """
    功能描述: 计算输入值的平方根，并处理负数开方的情况。

    参数:
        a (Union[np.ndarray, float, int]): 输入值，可以是NumPy数组、浮点数或整数。

    返回:
        Union[np.ndarray, float, int]: 平方根运算的结果。当输入值为负时，对应位置的结果为NaN。
    """
    return np.sqrt(a, out=np.full_like(a, np.nan), where=a >= 0)


# --- 累积算子 (从序列起始点开始累积，无窗口限制) ---
# 这些算子执行的是从序列开始到当前点的累积操作，不接受窗口参数，计算时忽略NaN。

def cumulative_sum(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算数据的累积和（从序列起始点开始累积），计算时忽略NaN值 (将NaN视为0)。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        axis (int): 累积的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 累积和的结果。
    """
    return np.nancumsum(data, axis=axis)


def cumulative_product(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算数据的累积乘积（从序列起始点开始累积），计算时忽略NaN值 (将NaN视为1)。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        axis (int): 累积的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 累积乘积的结果。
    """
    return np.nancumprod(data, axis=axis)


def cumulative_max(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算数据的累积最大值（从序列起始点开始累积），计算时忽略NaN值。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        axis (int): 累积的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 累积最大值的结果。
    """
    temp_data = np.where(np.isnan(data), -np.inf, data)
    result = np.maximum.accumulate(temp_data, axis=axis)
    result[result == -np.inf] = np.nan
    return result


def cumulative_min(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算数据的累积最小值（从序列起始点开始累积），计算时忽略NaN值。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        axis (int): 累积的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 累积最小值的结果。
    """
    temp_data = np.where(np.isnan(data), np.inf, data)
    result = np.minimum.accumulate(temp_data, axis=axis)
    result[result == np.inf] = np.nan
    return result


# --- 滚动算子 (在指定窗口内进行累积/统计，支持上下限参数，如 \sum_{i=1}^{N}[...]) ---
# 这些算子通过 'window' 参数定义了操作的上下限范围，实现了类似 LaTeX 中带上下标的求和/累积概念。

def _rolling_window(a: np.ndarray, window: int) -> np.ndarray:
    """
    功能描述: 对输入数组进行滑动窗口操作，生成连续子数组的视图。

    参数:
        a (np.ndarray): 输入的NumPy数组。
        window (int): 滑动窗口的大小。

    返回:
        np.ndarray: 包含滑动窗口视图的数组。
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides[:-1] + (a.strides[-1], a.strides[-1])
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_sum(data: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 在指定窗口内计算数据的滚动累加和，计算时忽略NaN值。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        window (int): 滚动窗口的大小。
        axis (int): 计算的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动累加和的结果。
    """
    if data.shape[axis] < window:
        return np.full_like(data, np.nan)

    if axis == 0:
        data = data.T

    windows = _rolling_window(data, window)
    sum_values = np.nansum(windows, axis=-1)

    result = np.full(data.shape, np.nan)
    result[:, window - 1:] = sum_values

    if axis == 0:
        result = result.T
    return result


def rolling_product(data: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 在指定窗口内计算数据的滚动累积乘积，计算时忽略NaN值。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        window (int): 滚动窗口的大小。
        axis (int): 计算的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动累积乘积的结果。
    """
    if data.shape[axis] < window:
        return np.full_like(data, np.nan)

    if axis == 0:
        data = data.T

    windows = _rolling_window(data, window)
    prod_values = np.nanprod(windows, axis=-1)

    result = np.full(data.shape, np.nan)
    result[:, window - 1:] = prod_values

    if axis == 0:
        result = result.T
    return result


def rolling_max(data: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 在指定窗口内计算数据的滚动最大值，计算时忽略NaN值。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        window (int): 滚动窗口的大小。
        axis (int): 计算的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动最大值的结果。
    """
    if data.shape[axis] < window:
        return np.full_like(data, np.nan)

    if axis == 0:
        data = data.T

    windows = _rolling_window(data, window)
    max_values = np.nanmax(windows, axis=-1)

    result = np.full(data.shape, np.nan)
    result[:, window - 1:] = max_values

    if axis == 0:
        result = result.T
    return result


def rolling_min(data: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 在指定窗口内计算数据的滚动最小值，计算时忽略NaN值。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        window (int): 滚动窗口的大小。
        axis (int): 计算的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动最小值的结果。
    """
    if data.shape[axis] < window:
        return np.full_like(data, np.nan)

    if axis == 0:
        data = data.T

    windows = _rolling_window(data, window)
    min_values = np.nanmin(windows, axis=-1)

    result = np.full(data.shape, np.nan)
    result[:, window - 1:] = min_values

    if axis == 0:
        result = result.T
    return result


# --- 统计算子 ---

def mean(data: np.ndarray, axis: int = 0) -> Union[np.ndarray, float]:
    """
    功能描述: 计算数据的均值。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        axis (int): 计算的轴向。0表示按列（时间序列），1表示按行（横截面）。默认为0。

    返回:
        Union[np.ndarray, float]: 均值。
    """
    return np.nanmean(data, axis=axis)


def std_dev(data: np.ndarray, axis: int = 0, ddof: int = 1) -> Union[np.ndarray, float]:
    """
    功能描述: 计算数据的标准差。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        axis (int): 计算的轴向。0表示按列（时间序列），1表示按行（横截面）。默认为0。
        ddof (int): 自由度。默认为1。

    返回:
        Union[np.ndarray, float]: 标准差。
    """
    return np.nanstd(data, axis=axis, ddof=ddof)


def variance(data: np.ndarray, axis: int = 0, ddof: int = 1) -> Union[np.ndarray, float]:
    """
    功能描述: 计算数据的方差。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        axis (int): 计算的轴向。0表示按列（时间序列），1表示按行（横截面）。默认为0。
        ddof (int): 自由度。默认为1。

    返回:
        Union[np.ndarray, float]: 方差。
    """
    return np.nanvar(data, axis=axis, ddof=ddof)


def correlation(a: np.ndarray, b: np.ndarray, axis: int = 0) -> Union[np.ndarray, float]:
    """
    功能描述: 计算两个输入之间的相关系数（横截面或时间序列）。

    参数:
        a (np.ndarray): 输入数组a。
        b (np.ndarray): 输入数组b，必须与a具有相同的形状。
        axis (int): 计算的轴向。0表示按列（时间序列），1表示按行（横截面）。默认为0。

    返回:
        Union[np.ndarray, float]: 相关系数。如果axis=0且输入为2D，则返回一个1D数组。
    """
    if a.shape != b.shape:
        raise ValueError("输入数组 a 和 b 必须具有相同的形状。")

    if axis == 0:  # 时间序列相关性 (逐列计算)
        if a.ndim == 1:
            valid_indices = ~np.isnan(a) & ~np.isnan(b)
            if np.sum(valid_indices) < 2:
                return np.nan
            return np.corrcoef(a[valid_indices], b[valid_indices])[0, 1]
        else:
            results = np.full(a.shape[1], np.nan)
            for i in range(a.shape[1]):
                col_a, col_b = a[:, i], b[:, i]
                valid_indices = ~np.isnan(col_a) & ~np.isnan(col_b)
                if np.sum(valid_indices) >= 2:
                    results[i] = np.corrcoef(col_a[valid_indices], col_b[valid_indices])[0, 1]
            return results

    elif axis == 1:  # 横截面相关性 (逐行计算)
        results = np.full(a.shape[0], np.nan)
        for i in range(a.shape[0]):
            row_a, row_b = a[i, :], b[i, :]
            valid_indices = ~np.isnan(row_a) & ~np.isnan(row_b)
            if np.sum(valid_indices) >= 2:
                results[i] = np.corrcoef(row_a[valid_indices], row_b[valid_indices])[0, 1]
        return results
    else:
        raise ValueError("Axis 必须是 0 (时间序列) 或 1 (横截面)。")


def covariance(a: np.ndarray, b: np.ndarray, axis: int = 0) -> Union[np.ndarray, float]:
    """
    功能描述: 计算两个输入之间的协方差（横截面或时间序列）。

    参数:
        a (np.ndarray): 输入数组a。
        b (np.ndarray): 输入数组b，必须与a具有相同的形状。
        axis (int): 计算的轴向。0表示按列（时间序列），1表示按行（横截面）。默认为0。

    返回:
        Union[np.ndarray, float]: 协方差。如果axis=0且输入为2D，则返回一个1D数组。
    """
    if a.shape != b.shape:
        raise ValueError("输入数组 a 和 b 必须具有相同的形状。")

    if axis == 0:  # 时间序列协方差 (逐列计算)
        if a.ndim == 1:
            valid_indices = ~np.isnan(a) & ~np.isnan(b)
            if np.sum(valid_indices) < 2:
                return np.nan
            return np.cov(a[valid_indices], b[valid_indices])[0, 1]
        else:
            results = np.full(a.shape[1], np.nan)
            for i in range(a.shape[1]):
                col_a, col_b = a[:, i], b[:, i]
                valid_indices = ~np.isnan(col_a) & ~np.isnan(col_b)
                if np.sum(valid_indices) >= 2:
                    results[i] = np.cov(col_a[valid_indices], col_b[valid_indices])[0, 1]
            return results

    elif axis == 1:  # 横截面协方差 (逐行计算)
        results = np.full(a.shape[0], np.nan)
        for i in range(a.shape[0]):
            row_a, row_b = a[i, :], b[i, :]
            valid_indices = ~np.isnan(row_a) & ~np.isnan(row_b)
            if np.sum(valid_indices) >= 2:
                results[i] = np.cov(row_a[valid_indices], row_b[valid_indices])[0, 1]
        return results
    else:
        raise ValueError("Axis 必须是 0 (时间序列) 或 1 (横截面)。")


def max_val(data: np.ndarray, axis: int = 0) -> Union[np.ndarray, float]:
    """
    功能描述: 计算数据的最大值。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        axis (int): 计算的轴向。0表示按列（时间序列），1表示按行（横截面）。默认为0。

    返回:
        Union[np.ndarray, float]: 最大值。
    """
    return np.nanmax(data, axis=axis)


def min_val(data: np.ndarray, axis: int = 0) -> Union[np.ndarray, float]:
    """
    功能描述: 计算数据的最小值。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        axis (int): 计算的轴向。0表示按列（时间序列），1表示按行（横截面）。默认为0。

    返回:
        Union[np.ndarray, float]: 最小值。
    """
    return np.nanmin(data, axis=axis)


def median(data: np.ndarray, axis: int = 0) -> Union[np.ndarray, float]:
    """
    功能描述: 计算数据的中位数。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        axis (int): 计算的轴向。0表示按列（时间序列），1表示按行（横截面）。默认为0。

    返回:
        Union[np.ndarray, float]: 中位数。
    """
    return np.nanmedian(data, axis=axis)


def quantile(data: np.ndarray, q: Union[float, Sequence[float]], axis: int = 0) -> Union[np.ndarray, float]:
    """
    功能描述: 计算数据的分位数。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        q (Union[float, Sequence[float]]): 要计算的分位数，介于0和1之间。
        axis (int): 计算的轴向。0表示按列（时间序列），1表示按行（横截面）。默认为0。

    返回:
        Union[np.ndarray, float]: 分位数。
    """
    return np.nanquantile(data, q, axis=axis)


def excess_return(data: np.ndarray, benchmark_data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    功能描述: 获取股票的每日 excess return。即每日超额收益率。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        benchmark_data (np.ndarray): 基准数据，必须与data具有相同的形状。
        axis (int): 计算的轴向。目前只支持0（时间序列）。

    返回:
        np.ndarray: 股票的 excess return。
    """
    if axis != 0:
        raise ValueError("excess_return only supports axis=0 (time-series).")

    if data.shape != benchmark_data.shape:
        raise ValueError("data and benchmark_data must have the same shape.")

    # 计算超额收益
    excess_returns = data - benchmark_data
    return excess_returns


# --- 时间序列算子 (通常作用于axis=0，即时间轴) ---

def ts_delay(data: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 获取过去第N期的数据（时间序列延迟）。也称为 'shift' 或 'lag'。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        window (int): 延迟的期数。
        axis (int): 计算的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 延迟后的数据。
    """
    if axis == 0:
        result = np.full_like(data, np.nan, dtype=float)
        result[window:] = data[:-window]
    elif axis == 1:
        result = np.full_like(data, np.nan, dtype=float)
        result[:, window:] = data[:, :-window]
    else:
        raise ValueError("Axis must be 0 or 1.")
    return result


def ts_delta(data: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算当前数据与N期前数据的差值。也称为 'diff'。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        window (int): 差分的期数。
        axis (int): 计算的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 差分后的数据。
    """
    delayed_data = ts_delay(data, window, axis)
    return data - delayed_data


def moving_average(data: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算数据的移动平均。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        window (int): 移动窗口的大小。
        axis (int): 计算的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 移动平均值。
    """
    if axis == 0:
        data = data.T

    result = np.full(data.shape, np.nan)
    for i in range(data.shape[0]):
        if data.shape[1] >= window:
            windows = _rolling_window(data[i], window)
            result[i, window - 1:] = np.nanmean(windows, axis=-1)

    if axis == 0:
        result = result.T
    return result


def ts_std(data: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算数据的滚动标准差。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        window (int): 滚动窗口的大小。
        axis (int): 计算的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动标准差。
    """
    if axis == 0:
        data = data.T

    result = np.full(data.shape, np.nan)
    for i in range(data.shape[0]):
        if data.shape[1] >= window:
            windows = _rolling_window(data[i], window)
            result[i, window - 1:] = np.nanstd(windows, axis=-1)

    if axis == 0:
        result = result.T
    return result


# Alias ts_mean to moving_average
ts_mean = moving_average


def exponential_moving_average(data: np.ndarray, span: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算数据的指数移动平均。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        span (int): 时间跨度。
        axis (int): 计算的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 指数移动平均值。
    """
    if axis == 0:
        data = data.T

    alpha = 2 / (span + 1)
    ema_data = np.full_like(data, np.nan, dtype=float)

    for i in range(data.shape[0]):
        current_ema = np.nan  # Initialize with NaN
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                ema_data[i, j] = np.nan
            else:
                if np.isnan(current_ema):
                    current_ema = data[i, j]
                else:
                    current_ema = alpha * data[i, j] + (1 - alpha) * current_ema
                ema_data[i, j] = current_ema

    if axis == 0:
        ema_data = ema_data.T
    return ema_data


def rolling_skew(data: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 在指定窗口内计算数据的滚动偏度。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        window (int): 滚动窗口的大小。
        axis (int): 计算的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动偏度。
    """
    if axis == 0:
        data = data.T

    result = np.full(data.shape, np.nan)
    for i in range(data.shape[0]):
        if data.shape[1] >= window:
            windows = _rolling_window(data[i], window)
            # Calculate skew for each window, ignoring NaNs
            for j, win in enumerate(windows):
                # Remove NaNs from the window
                clean_win = win[~np.isnan(win)]
                if len(clean_win) >= 3:  # Skew requires at least 3 data points
                    m3 = np.nanmean((clean_win - np.nanmean(clean_win)) ** 3)
                    m2 = np.nanmean((clean_win - np.nanmean(clean_win)) ** 2)
                    if m2 > 0:
                        result[i, window - 1 + j] = m3 / (m2 ** 1.5)

    if axis == 0:
        result = result.T
    return result


def rolling_kurt(data: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 在指定窗口内计算数据的滚动峰度。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        window (int): 滚动窗口的大小。
        axis (int): 计算的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动峰度。
    """
    if axis == 0:
        data = data.T

    result = np.full(data.shape, np.nan)
    for i in range(data.shape[0]):
        if data.shape[1] >= window:
            windows = _rolling_window(data[i], window)
            # Calculate kurtosis for each window, ignoring NaNs
            for j, win in enumerate(windows):
                # Remove NaNs from the window
                clean_win = win[~np.isnan(win)]
                if len(clean_win) >= 4:  # Kurtosis requires at least 4 data points
                    m4 = np.nanmean((clean_win - np.nanmean(clean_win)) ** 4)
                    m2 = np.nanmean((clean_win - np.nanmean(clean_win)) ** 2)
                    if m2 > 0:
                        result[i, window - 1 + j] = m4 / (m2 ** 2) - 3  # Excess kurtosis

    if axis == 0:
        result = result.T
    return result


def rolling_quantile(data: np.ndarray, window: int, quantile_level: float, axis: int = 0) -> np.ndarray:
    """
    功能描述: 在指定窗口内计算数据的滚动分位数。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        window (int): 滚动窗口的大小。
        quantile_level (float): 要计算的分位数，介于0和1之间。
        axis (int): 计算的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动分位数。
    """
    if axis == 0:
        data = data.T

    result = np.full(data.shape, np.nan)
    for i in range(data.shape[0]):
        if data.shape[1] >= window:
            windows = _rolling_window(data[i], window)
            result[i, window - 1:] = np.nanquantile(windows, quantile_level, axis=-1)

    if axis == 0:
        result = result.T
    return result


def rolling_corr(a: np.ndarray, b: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 在指定窗口内计算两个序列的滚动相关系数。

    参数:
        a (np.ndarray): 输入数组a。
        b (np.ndarray): 输入数组b，必须与a具有相同的形状。
        window (int): 滚动窗口的大小。
        axis (int): 计算的轴向。目前只支持0（时间序列）。

    返回:
        np.ndarray: 滚动相关系数。
    """
    if axis != 0:
        raise ValueError("rolling_corr only supports axis=0 (time-series correlation).")

    if a.shape != b.shape:
        raise ValueError("Inputs a and b must have the same shape.")

    result = np.full(a.shape, np.nan)

    for i in range(a.shape[1]):  # Iterate over columns (stocks)
        for j in range(window - 1, a.shape[0]):  # Iterate over time
            a_slice = a[j - window + 1: j + 1, i]
            b_slice = b[j - window + 1: j + 1, i]

            # Remove NaNs for correlation calculation
            valid_indices = ~np.isnan(a_slice) & ~np.isnan(b_slice)

            if np.sum(valid_indices) >= 2:
                result[j, i] = np.corrcoef(a_slice[valid_indices], b_slice[valid_indices])[0, 1]
    return result


def rolling_cov(a: np.ndarray, b: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 在指定窗口内计算两个序列的滚动协方差。

    参数:
        a (np.ndarray): 输入数组a。
        b (np.ndarray): 输入数组b，必须与a具有相同的形状。
        window (int): 滚动窗口的大小。
        axis (int): 计算的轴向。目前只支持0（时间序列）。

    返回:
        np.ndarray: 滚动协方差。
    """
    if axis != 0:
        raise ValueError("rolling_cov only supports axis=0 (time-series covariance).")

    if a.shape != b.shape:
        raise ValueError("Inputs a and b must have the same shape.")

    result = np.full(a.shape, np.nan)

    for i in range(a.shape[1]):  # Iterate over columns (stocks)
        for j in range(window - 1, a.shape[0]):  # Iterate over time
            a_slice = a[j - window + 1: j + 1, i]
            b_slice = b[j - window + 1: j + 1, i]

            # Remove NaNs for covariance calculation
            valid_indices = ~np.isnan(a_slice) & ~np.isnan(b_slice)

            if np.sum(valid_indices) >= 2:
                result[j, i] = np.cov(a_slice[valid_indices], b_slice[valid_indices])[0, 1]
    return result


def ts_rank(data: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算数据在过去N期时间序列窗口中的百分比排名。
              其行为旨在模拟 pandas.rank(pct=True)。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        window (int): 滚动窗口的大小。
        axis (int): 计算的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动百分比排名的结果。
    """
    if data.shape[axis] < window:
        return np.full_like(data, np.nan)

    if axis == 0:
        data = data.T

    # 创建滚动窗口
    windows = _rolling_window(data, window)

    # 获取每个窗口的最后一个值，这是我们要计算排名的目标值
    last_values = windows[:, :, -1]

    # 计算每个窗口内非NaN值的数量
    n_valid = np.sum(~np.isnan(windows), axis=-1)

    # 计算每个窗口中严格小于目标值的数量
    n_less = np.nansum(windows < last_values[..., np.newaxis], axis=-1)

    # 计算每个窗口中等于目标值的数量
    n_equal = np.nansum(windows == last_values[..., np.newaxis], axis=-1)

    # 计算平均排名 (处理并列情况)
    rank_ = n_less + (n_equal - 1) * 0.5

    # 计算百分比排名
    # 减1是为了使分母为(n-1)，与pandas的pct=True行为一致
    pct_rank = rank_ / (n_valid - 1)

    # 处理特殊情况
    pct_rank[n_valid <= 1] = np.nan  # 如果窗口内有效值不多于1个，则无排名
    pct_rank[np.isnan(last_values)] = np.nan  # 如果目标值是NaN，则排名也是NaN

    # 准备结果数组
    result = np.full(data.shape, np.nan)
    result[:, window - 1:] = pct_rank

    if axis == 0:
        result = result.T

    return result


def time_series_decay(data: np.ndarray, halflife: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算数据的时间序列衰减加权平均 (指数加权移动平均)。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        halflife (int): 半衰期。
        axis (int): 计算的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 时间序列衰减加权平均值。
    """
    if data.ndim == 1:
        data = data[:, np.newaxis]  # Make it 2D for consistent processing

    alpha = 1 - np.exp(np.log(0.5) / halflife)
    decay_data = np.full_like(data, np.nan, dtype=float)

    if axis == 0:  # Apply EMA along rows (time-series for each column)
        for col_idx in range(data.shape[1]):
            current_series = data[:, col_idx]
            current_ema = np.nan
            for i in range(data.shape[0]):
                if np.isnan(current_series[i]):
                    decay_data[i, col_idx] = np.nan
                else:
                    if np.isnan(current_ema):
                        current_ema = current_series[i]
                    else:
                        current_ema = alpha * current_series[i] + (1 - alpha) * current_ema
                    decay_data[i, col_idx] = current_ema
    elif axis == 1:  # Apply EMA along columns (cross-sectional for each row)
        for row_idx in range(data.shape[0]):
            current_series = data[row_idx, :]
            current_ema = np.nan
            for i in range(data.shape[1]):
                if np.isnan(current_series[i]):
                    decay_data[row_idx, i] = np.nan
                else:
                    if np.isnan(current_ema):
                        current_ema = current_series[i]
                    else:
                        current_ema = alpha * current_series[i] + (1 - alpha) * current_ema
                    decay_data[row_idx, i] = current_ema
    else:
        raise ValueError("Axis must be 0 or 1.")

    return decay_data


# --- 复合财务指标算子 (Complex Financial Metrics) ---
# 这些算子通常用于评估收益序列的风险和表现，其计算逻辑是路径依赖的。

def rolling_max_drawdown(data: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算滚动最大回撤。最大回撤衡量了在指定窗口内资产净值从峰值回落的最大百分比。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        window (int): 滚动窗口的大小。
        axis (int): 计算的轴向。目前只支持0（时间序列）。

    返回:
        np.ndarray: 滚动最大回撤。
    """
    if axis != 0:
        raise ValueError("rolling_max_drawdown only supports axis=0 (time-series).")

    result = np.full_like(data, np.nan, dtype=float)

    for i in range(data.shape[1]):  # Iterate over columns (assets)
        asset_series = data[:, i]

        for j in range(window - 1, len(asset_series)):  # Iterate over time
            window_data = asset_series[j - window + 1: j + 1]

            # Remove NaNs for calculation
            clean_window = window_data[~np.isnan(window_data)]

            if len(clean_window) > 0:
                cumulative_returns = np.cumprod(1 + clean_window) - 1
                peak = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - peak) / (1 + peak)  # Adjusted for percentage drawdown
                result[j, i] = np.nanmin(drawdown)
    return result


def downside_deviation(data: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算下行标准差（下行波动率）。只针对窗口期内小于0的收益率计算标准差。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        window (int): 滚动窗口的大小。
        axis (int): 计算的轴向。目前只支持0（时间序列）。

    返回:
        np.ndarray: 下行标准差。
    """
    if axis != 0:
        raise ValueError("downside_deviation only supports axis=0 (time-series).")

    result = np.full_like(data, np.nan, dtype=float)

    for i in range(data.shape[1]):  # Iterate over columns (assets)
        asset_series = data[:, i]

        for j in range(window - 1, len(asset_series)):  # Iterate over time
            window_data = asset_series[j - window + 1: j + 1]

            # Filter for negative returns and remove NaNs
            downside_returns = window_data[~np.isnan(window_data) & (window_data < 0)]

            if len(downside_returns) >= 2:  # Need at least 2 data points for std
                result[j, i] = np.nanstd(downside_returns, ddof=1)
    return result


def sharpe_ratio(data: np.ndarray, window: int, periods_per_year: int = 252, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算滚动夏普比率。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        window (int): 滚动窗口的大小。
        periods_per_year (int): 每年的周期数，用于年化。默认为252。
        axis (int): 计算的轴向。目前只支持0（时间序列）。

    返回:
        np.ndarray: 滚动夏普比率。
    """
    if axis != 0:
        raise ValueError("sharpe_ratio only supports axis=0 (time-series).")

    rolling_mean = moving_average(data, window, axis=axis)
    rolling_std = ts_std(data, window, axis=axis)

    # Avoid division by zero
    sharpe = (rolling_mean / np.where(rolling_std == 0, np.nan, rolling_std)) * np.sqrt(periods_per_year)
    return sharpe


def sortino_ratio(data: np.ndarray, window: int, periods_per_year: int = 252, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算滚动索提诺比率。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        window (int): 滚动窗口的大小。
        periods_per_year (int): 每年的周期数，用于年化。默认为252。
        axis (int): 计算的轴向。目前只支持0（时间序列）。

    返回:
        np.ndarray: 滚动索提诺比率。
    """
    if axis != 0:
        raise ValueError("sortino_ratio only supports axis=0 (time-series).")

    rolling_mean = moving_average(data, window, axis=axis)
    rolling_downside_std = downside_deviation(data, window, axis=axis)

    # Avoid division by zero
    sortino = (rolling_mean / np.where(rolling_downside_std == 0, np.nan, rolling_downside_std)) * np.sqrt(
        periods_per_year)
    return sortino


def calmar_ratio(data: np.ndarray, window: int, periods_per_year: int = 252, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算滚动卡玛比率 (年化收益 / 最大回撤)。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        window (int): 滚动窗口的大小。
        periods_per_year (int): 每年的周期数，用于年化。默认为252。
        axis (int): 计算的轴向。目前只支持0（时间序列）。

    返回:
        np.ndarray: 滚动卡玛比率。
    """
    if axis != 0:
        raise ValueError("calmar_ratio only supports axis=0 (time-series).")

    annualized_return = moving_average(data, window, axis=axis) * periods_per_year
    max_dd = rolling_max_drawdown(data, window, axis=axis)

    # Avoid division by zero, and take the absolute value of max_dd
    calmar = annualized_return / np.where(np.abs(max_dd) == 0, np.nan, np.abs(max_dd))
    return calmar


def _ols_regression(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    功能描述: 执行普通最小二乘回归 (OLS)，计算回归系数。

    参数:
        y (np.ndarray): 响应变量，必须是一维数组。
        X (np.ndarray): 自变量，必须是二维数组。

    返回:
        np.ndarray: 回归系数，包括截距和斜率。如果无法计算，则返回NaN。
    """
    # Combine y and X to easily drop rows with NaNs
    combined = np.concatenate((y[:, np.newaxis], X), axis=1)
    clean_combined = combined[~np.isnan(combined).any(axis=1)]

    if clean_combined.shape[0] < X.shape[1]:  # Not enough data points for regression
        return np.full(X.shape[1], np.nan)

    y_clean = clean_combined[:, 0]
    X_clean = clean_combined[:, 1:]

    # Add a constant (intercept) term to X if not already present
    # For alpha/beta, we expect X to be benchmark returns, and we need an intercept
    # So, we'll add it here if X_clean is just 1D (single benchmark)
    if X_clean.ndim == 1:
        X_clean = X_clean[:, np.newaxis]
    X_clean = np.hstack((np.ones((X_clean.shape[0], 1)), X_clean))

    try:
        # OLS formula: beta = (X.T @ X)^-1 @ X.T @ y
        beta_hat = np.linalg.inv(X_clean.T @ X_clean) @ X_clean.T @ y_clean
        return beta_hat
    except np.linalg.LinAlgError:
        # Handle singular matrix (e.g., perfect multicollinearity or not enough data)
        return np.full(X_clean.shape[1], np.nan)


def alpha(data: np.ndarray, benchmark_data: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算滚动Alpha。通过在每个时间点上使用最小二乘回归(OLS)方法，将资产收益率与基准收益率进行回归分析，从而估计出资产相对于基准的超额收益。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        benchmark_data (np.ndarray): 基准数据，必须与data具有相同的形状。
        window (int): 滚动窗口的大小。
        axis (int): 计算的轴向。目前只支持0（时间序列）。

    返回:
        np.ndarray: 滚动Alpha。
    """
    if axis != 0:
        raise ValueError("alpha only supports axis=0 (time-series).")

    if data.shape != benchmark_data.shape:
        raise ValueError("data and benchmark_data must have the same shape.")

    result = np.full_like(data, np.nan, dtype=float)

    for i in range(data.shape[1]):  # Iterate over columns (assets)
        asset_returns = data[:, i]
        benchmark_returns = benchmark_data[:, i]
        # 执行滚动回归
        for j in range(window - 1, len(asset_returns)):  # Iterate over time
            y_slice = asset_returns[j - window + 1: j + 1]
            x_slice = benchmark_returns[j - window + 1: j + 1]

            # Perform OLS regression
            # _ols_regression expects X to be without constant, it adds it internally
            coeffs = _ols_regression(y_slice, x_slice[:, np.newaxis])  # Ensure x_slice is 2D

            if not np.isnan(coeffs[0]):  # Check if intercept is valid
                result[j, i] = coeffs[0]  # Alpha is the intercept
    return result


def beta(data: np.ndarray, benchmark_data: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算滚动Beta (对基准的系统性风险)。通过在每个时间点上使用最小二乘回归(OLS)方法，得到的Beta值表示资产收益率对基准收益率的敏感度。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        benchmark_data (np.ndarray): 基准数据，必须与data具有相同的形状。
        window (int): 滚动窗口的大小。
        axis (int): 计算的轴向。目前只支持0（时间序列）。

    返回:
        np.ndarray: 滚动Beta。
    """
    if axis != 0:
        raise ValueError("beta only supports axis=0 (time-series).")

    if data.shape != benchmark_data.shape:
        raise ValueError("data and benchmark_data must have the same shape.")

    result = np.full_like(data, np.nan, dtype=float)

    for i in range(data.shape[1]):  # Iterate over columns (assets)
        asset_returns = data[:, i]
        benchmark_returns = benchmark_data[:, i]

        for j in range(window - 1, len(asset_returns)):  # Iterate over time
            y_slice = asset_returns[j - window + 1: j + 1]
            x_slice = benchmark_returns[j - window + 1: j + 1]

            # Perform OLS regression
            coeffs = _ols_regression(y_slice, x_slice[:, np.newaxis])  # Ensure x_slice is 2D

            if not np.isnan(coeffs[1]):  # Check if beta is valid
                result[j, i] = coeffs[1]  # Beta is the slope
    return result


# --- 数据预处理与因子结果处理算子 (仅用于原始数据预处理和因子结果评估前端处理，不用于因子计算逻辑的构建) ---

def neutralize(factor_data: np.ndarray, risk_factors_data: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    功能描述: 对因子数据进行中性化处理，消除其中包含的风险因子暴露。

    参数:
        factor_data (np.ndarray): 因子数据。
        risk_factors_data (np.ndarray): 风险因子数据，必须与factor_data具有相同的形状。
        axis (int): 中性化的轴向。1表示横截面中性化，0表示时间序列中性化。默认为1。

    返回:
        np.ndarray: 中性化后的因子数据。
    """
    if factor_data.shape != risk_factors_data.shape:
        raise ValueError("factor_data and risk_factors_data must have the same shape.")

    neutralized_factor = np.full_like(factor_data, np.nan, dtype=float)

    if axis == 1:  # 横截面中性化 (对每个日期进行回归)
        for i in range(factor_data.shape[0]):  # Iterate over rows (dates)
            y = factor_data[i, :]
            X = risk_factors_data[i, :]

            # Combine y and X to easily drop columns with NaNs
            combined = np.vstack((y, X)).T  # Transpose to have columns as variables
            clean_combined = combined[~np.isnan(combined).any(axis=1)]

            if clean_combined.shape[0] < 2:  # Not enough data points for regression
                continue

            y_clean = clean_combined[:, 0]
            X_clean = clean_combined[:, 1:]

            # Add a constant term
            X_clean = np.hstack((np.ones((X_clean.shape[0], 1)), X_clean))

            try:
                beta_hat = np.linalg.inv(X_clean.T @ X_clean) @ X_clean.T @ y_clean
                residuals = y_clean - (X_clean @ beta_hat)

                # Map residuals back to original positions
                original_indices = np.where(~np.isnan(combined).any(axis=1))[0]
                neutralized_factor[i, original_indices] = residuals
            except np.linalg.LinAlgError:
                pass  # Keep NaNs if regression fails

    elif axis == 0:  # 时间序列中性化 (对每个金融产品进行回归)
        for i in range(factor_data.shape[1]):  # Iterate over columns (assets)
            y = factor_data[:, i]
            X = risk_factors_data[:, i]

            # Combine y and X to easily drop rows with NaNs
            combined = np.vstack((y, X)).T
            clean_combined = combined[~np.isnan(combined).any(axis=1)]

            if clean_combined.shape[0] < 2:  # Not enough data points for regression
                continue

            y_clean = clean_combined[:, 0]
            X_clean = clean_combined[:, 1:]

            # Add a constant term
            X_clean = np.hstack((np.ones((X_clean.shape[0], 1)), X_clean))

            try:
                beta_hat = np.linalg.inv(X_clean.T @ X_clean) @ X_clean.T @ y_clean
                residuals = y_clean - (X_clean @ beta_hat)

                # Map residuals back to original positions
                original_indices = np.where(~np.isnan(combined).any(axis=1))[0]
                neutralized_factor[original_indices, i] = residuals
            except np.linalg.LinAlgError:
                pass  # Keep NaNs if regression fails
    else:
        raise ValueError("Axis must be 0 (time-series) or 1 (cross-sectional).")

    return neutralized_factor


def winsorize(data: np.ndarray, lower_percentile: float = 0.01, upper_percentile: float = 0.99,
              axis: Optional[int] = None) -> np.ndarray:
    """
    功能描述: 对数据进行缩尾处理，将极端值替换为指定分位数的值。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        lower_percentile (float): 下分位数，介于0和1之间。默认为0.01。
        upper_percentile (float): 上分位数，介于0和1之间。默认为0.99。
        axis (Optional[int]): 计算分位数的轴向。默认为None，对整个数组计算。

    返回:
        np.ndarray: 缩尾处理后的数据。
    """
    lower_bound = np.nanpercentile(data, lower_percentile * 100, axis=axis, keepdims=True)
    upper_bound = np.nanpercentile(data, upper_percentile * 100, axis=axis, keepdims=True)

    winsorized_data = np.clip(data, lower_bound, upper_bound)
    return winsorized_data


def clip(data: np.ndarray, lower_bound: Optional[Union[float, int]] = None,
         upper_bound: Optional[Union[float, int]] = None) -> np.ndarray:
    """
    功能描述: 将数据值限制在指定的上下限范围内。

    参数:
        data (np.ndarray): 输入数据，可以是NumPy数组。
        lower_bound (Optional[Union[float, int]]): 数据的下限值。默认为None（无下限）。
        upper_bound (Optional[Union[float, int]]): 数据的上限值。默认为None（无上限）。

    返回:
        np.ndarray: 截断/裁剪后的数据。
    """
    return np.clip(data, lower_bound, upper_bound)


def fill_na(data: np.ndarray, method: str = 'ffill', value: Optional[Union[float, int]] = None,
            axis: Optional[int] = None) -> np.ndarray:
    """
    功能描述: 填充数据中的缺失值（NaN）。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        method (str): 填充方法。可选值: 'ffill', 'bfill', 'zero', 'mean', 'median', 'value'。默认为'ffill'。
        value (Optional[Union[float, int]]): 当method为'value'时，用于填充的特定值。
        axis (Optional[int]): 'ffill', 'bfill', 'mean', 'median'的计算轴向。默认为None。

    返回:
        np.ndarray: 填充缺失值后的数据。
    """
    if method == 'zero':
        return np.nan_to_num(data, nan=0)
    elif method == 'value':
        return np.where(np.isnan(data), value, data)
    elif method == 'mean':
        if axis is None:
            fill_value = np.nanmean(data)
        else:
            fill_value = np.nanmean(data, axis=axis, keepdims=True)
        return np.where(np.isnan(data), fill_value, data)
    elif method == 'median':
        if axis is None:
            fill_value = np.nanmedian(data)
        else:
            fill_value = np.nanmedian(data, axis=axis, keepdims=True)
        return np.where(np.isnan(data), fill_value, data)
    elif method == 'ffill':
        if axis == 0:  # Fill along columns (time-series)
            output = np.copy(data)
            for col_idx in range(output.shape[1]):
                mask = np.isnan(output[:, col_idx])
                idx = np.where(~mask, np.arange(len(mask)), 0)
                np.maximum.accumulate(idx, out=idx)
                output[:, col_idx] = output[idx, col_idx]
            return output
        elif axis == 1:  # Fill along rows (cross-sectional)
            output = np.copy(data)
            for row_idx in range(output.shape[0]):
                mask = np.isnan(output[row_idx, :])
                idx = np.where(~mask, np.arange(len(mask)), 0)
                np.maximum.accumulate(idx, out=idx)
                output[row_idx, :] = output[row_idx, idx]
            return output
        else:
            raise ValueError("Axis must be 0 or 1 for ffill/bfill.")
    elif method == 'bfill':
        if axis == 0:  # Fill along columns (time-series)
            output = np.copy(data)
            for col_idx in range(output.shape[1]):
                mask = np.isnan(output[:, col_idx])
                idx = np.where(~mask, np.arange(len(mask)), len(mask) - 1)
                np.minimum.accumulate(idx[::-1], out=idx[::-1])
                output[:, col_idx] = output[idx, col_idx]
            return output
        elif axis == 1:  # Fill along rows (cross-sectional)
            output = np.copy(data)
            for row_idx in range(output.shape[0]):
                mask = np.isnan(output[row_idx, :])
                idx = np.where(~mask, np.arange(len(mask)), len(mask) - 1)
                np.minimum.accumulate(idx[::-1], out=idx[::-1])
                output[row_idx, :] = output[row_idx, idx]
            return output
        else:
            raise ValueError("Axis must be 0 or 1 for ffill/bfill.")
    else:
        raise ValueError(f"Unsupported fill method: {method}")


def cross_sectional_rank(data: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    功能描述: 对数据进行横截面排名（通常对每个日期进行排名）。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        axis (int): 排名的轴向。1表示横截面排名，0表示时间序列排名。默认为1。

    返回:
        np.ndarray: 排名后的数据（百分比形式）。
    """
    if axis == 1:
        ranked_data = np.full_like(data, np.nan, dtype=float)
        for i in range(data.shape[0]):
            # Get non-NaN values and their original indices
            row = data[i, :]
            non_nan_indices = ~np.isnan(row)
            clean_row = row[non_nan_indices]

            if len(clean_row) > 0:
                # Calculate ranks for non-NaN values
                # np.argsort returns indices that would sort an array
                # np.argsort(np.argsort(arr)) gives ranks (0-indexed)
                ranks = np.argsort(np.argsort(clean_row)).astype(float)

                # Convert to percentage ranks (0 to 1)
                if len(clean_row) > 1:
                    percentage_ranks = ranks / (len(clean_row) - 1)
                else:
                    percentage_ranks = np.array([0.0])  # Single element is rank 0

                # Place ranks back into the original array shape
                ranked_data[i, non_nan_indices] = percentage_ranks
        return ranked_data
    elif axis == 0:
        ranked_data = np.full_like(data, np.nan, dtype=float)
        for i in range(data.shape[1]):
            col = data[:, i]
            non_nan_indices = ~np.isnan(col)
            clean_col = col[non_nan_indices]

            if len(clean_col) > 0:
                ranks = np.argsort(np.argsort(clean_col)).astype(float)
                if len(clean_col) > 1:
                    percentage_ranks = ranks / (len(clean_col) - 1)
                else:
                    percentage_ranks = np.array([0.0])
                ranked_data[non_nan_indices, i] = percentage_ranks
        return ranked_data
    else:
        raise ValueError("Axis must be 0 or 1.")


def cross_sectional_scale(data: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    功能描述: 对数据进行横截面标准化（Z-score标准化）。

    参数:
        data (np.ndarray): 输入数据，NumPy数组格式。
        axis (int): 标准化的轴向。1表示横截面标准化，0表示时间序列标准化。默认为1。

    返回:
        np.ndarray: 标准化后的数据。
    """
    if axis == 1:
        mean_vals = np.nanmean(data, axis=axis, keepdims=True)
        std_vals = np.nanstd(data, axis=axis, keepdims=True)
        # Avoid division by zero
        scaled_data = np.divide(data - mean_vals, std_vals, out=np.full_like(data, np.nan), where=std_vals != 0)
        return scaled_data
    elif axis == 0:
        mean_vals = np.nanmean(data, axis=axis, keepdims=True)
        std_vals = np.nanstd(data, axis=axis, keepdims=True)
        scaled_data = np.divide(data - mean_vals, std_vals, out=np.full_like(data, np.nan), where=std_vals != 0)
        return scaled_data
    else:
        raise ValueError("Axis must be 0 or 1.")


if __name__ == '__main__':
    # 测试用例
    print(rolling_sum(np.array([[1, 2, np.nan], [4, 5, 6], [7, 8, 9]]), window=2, axis=0))
