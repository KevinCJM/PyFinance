# -*- encoding: utf-8 -*-
"""
@File: A02_OperatorLibrary.py
@Modify Time: 2025/7/10 09:49       
@Author: Kevin-Chen
@Descriptions: 
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm  # 导入statsmodels
from typing import Union, Optional, Sequence


# --- 基础数学算子 ---

def add(a, b):
    return pd.DataFrame(a).add(pd.DataFrame(b) if not isinstance(b, (pd.DataFrame, pd.Series)) else b, fill_value=0)


def subtract(a, b):
    return pd.DataFrame(a).sub(pd.DataFrame(b) if not isinstance(b, (pd.DataFrame, pd.Series)) else b, fill_value=0)


def multiply(a, b):
    return pd.DataFrame(a).mul(pd.DataFrame(b) if not isinstance(b, (pd.DataFrame, pd.Series)) else b, fill_value=0)


def divide(a, b):
    """
    功能描述: 执行两个输入之间的除法运算，并处理除零情况。
    Pandas的 .div() 方法会自动处理索引对齐和除零（返回 inf）。
    我们随后将 inf 替换为 NaN。
    """
    # Ensure both are pandas objects for safe division
    a = pd.DataFrame(a) if not isinstance(a, (pd.DataFrame, pd.Series)) else a
    b = pd.DataFrame(b) if not isinstance(b, (pd.DataFrame, pd.Series)) else b
    
    result = a.div(b)
    # Replace infinite values resulting from division by zero with NaN
    return result.replace([np.inf, -np.inf], np.nan)


def log(a: Union[np.ndarray, pd.DataFrame, float, int]) -> Union[np.ndarray, pd.DataFrame, float, int]:
    """
    功能描述: 计算输入值的自然对数，并处理非正数取对数的情况。

    参数:
        a (Union[np.ndarray, pd.DataFrame, float, int]): 输入值，可以是NumPy数组、Pandas DataFrame、浮点数或整数。

    返回:
        Union[np.ndarray, pd.DataFrame, float, int]: 自然对数运算的结果。当输入值非正时，对应位置的结果为NaN。
    """
    return np.log(a, out=np.full_like(a, np.nan), where=a > 0)


def abs_val(a: Union[np.ndarray, pd.DataFrame, float, int]) -> Union[np.ndarray, pd.DataFrame, float, int]:
    """
    功能描述: 计算输入值的绝对值。

    参数:
        a (Union[np.ndarray, pd.DataFrame, float, int]): 输入值，可以是NumPy数组、Pandas DataFrame、浮点数或整数。

    返回:
        Union[np.ndarray, pd.DataFrame, float, int]: 绝对值运算的结果，类型与输入a的类型兼容。
    """
    return np.abs(a)


def power(a: Union[np.ndarray, pd.DataFrame, float, int], p: Union[np.ndarray, pd.DataFrame, float, int]) -> Union[
    np.ndarray, pd.DataFrame, float, int]:
    """
    功能描述: 计算a的p次幂。

    参数:
        a (Union[np.ndarray, pd.DataFrame, float, int]): 基数，可以是NumPy数组、Pandas DataFrame、浮点数或整数。
        p (Union[np.ndarray, pd.DataFrame, float, int]): 指数，可以是NumPy数组、Pandas DataFrame、浮点数或整数。

    返回:
        Union[np.ndarray, pd.DataFrame, float, int]: 幂运算的结果，类型与输入a和p的类型兼容。
    """
    return np.power(a, p)


def sqrt(a: Union[np.ndarray, pd.DataFrame, float, int]) -> Union[np.ndarray, pd.DataFrame, float, int]:
    """
    功能描述: 计算输入值的平方根，并处理负数开方的情况。

    参数:
        a (Union[np.ndarray, pd.DataFrame, float, int]): 输入值，可以是NumPy数组、Pandas DataFrame、浮点数或整数。

    返回:
        Union[np.ndarray, pd.DataFrame, float, int]: 平方根运算的结果。当输入值为负时，对应位置的结果为NaN。
    """
    return np.sqrt(a, out=np.full_like(a, np.nan), where=a >= 0)


# --- 累积算子 (从序列起始点开始累积，无窗口限制) ---
# 这些算子执行的是从序列开始到当前点的累积操作，不接受窗口参数。

def cumulative_sum(data: Union[np.ndarray, pd.DataFrame], axis: int = 0) -> Union[np.ndarray, pd.DataFrame]:
    """
    功能描述: 计算数据的累积和（从序列起始点开始累积）。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        axis (int): 累积的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        Union[np.ndarray, pd.DataFrame]: 累积和的结果，类型与输入data的类型兼容。
    """
    return np.cumsum(data, axis=axis)


def cumulative_product(data: Union[np.ndarray, pd.DataFrame], axis: int = 0) -> Union[np.ndarray, pd.DataFrame]:
    """
    功能描述: 计算数据的累积乘积（从序列起始点开始累积）。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        axis (int): 累积的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        Union[np.ndarray, pd.DataFrame]: 累积乘积的结果，类型与输入data的类型兼容。
    """
    return np.cumprod(data, axis=axis)


def cumulative_max(data: Union[np.ndarray, pd.DataFrame], axis: int = 0) -> Union[np.ndarray, pd.DataFrame]:
    """
    功能描述: 计算数据的累积最大值（从序列起始点开始累积）。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        axis (int): 累积的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        Union[np.ndarray, pd.DataFrame]: 累积最大值的结果，类型与输入data的类型兼容。
    """
    return np.maximum.accumulate(data, axis=axis)


def cumulative_min(data: Union[np.ndarray, pd.DataFrame], axis: int = 0) -> Union[np.ndarray, pd.DataFrame]:
    """
    功能描述: 计算数据的累积最小值（从序列起始点开始累积）。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        axis (int): 累积的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        Union[np.ndarray, pd.DataFrame]: 累积最小值的结果，类型与输入data的类型兼容。
    """
    return np.minimum.accumulate(data, axis=axis)


# --- 滚动算子 (在指定窗口内进行累积/统计，支持上下限参数，如 \sum_{i=1}^{N}[...]) ---
# 这些算子通过 'window' 参数定义了操作的上下限范围，实现了类似 LaTeX 中带上下标的求和/累积概念。

def rolling_sum(data: Union[np.ndarray, pd.DataFrame], window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 在指定窗口内计算数据的滚动累加和。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        window (int): 滚动窗口的大小。
        axis (int): 滚动的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动累加和的结果，为NumPy数组。
    """
    return pd.DataFrame(data).rolling(window=window, axis=axis).sum().values


def rolling_product(data: Union[np.ndarray, pd.DataFrame], window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 在指定窗口内计算数据的滚动累积乘积。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        window (int): 滚动窗口的大小。
        axis (int): 滚动的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动累积乘积的结果，为NumPy数组。
    """
    return pd.DataFrame(data).rolling(window=window, axis=axis).prod().values


def rolling_max(data: Union[np.ndarray, pd.DataFrame], window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 在指定窗口内计算数据的滚动最大值。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        window (int): 滚动窗口的大小。
        axis (int): 滚动的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动最大值的结果，为NumPy数组。
    """
    return pd.DataFrame(data).rolling(window=window, axis=axis).max().values


def rolling_min(data: Union[np.ndarray, pd.DataFrame], window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 在指定窗口内计算数据的滚动最小值。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        window (int): 滚动窗口的大小。
        axis (int): 滚动的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动最小值的结果，为NumPy数组。
    """
    return pd.DataFrame(data).rolling(window=window, axis=axis).min().values


# --- 统计算子 ---

def mean(data: Union[np.ndarray, pd.DataFrame], axis: int = 0) -> Union[np.ndarray, float]:
    """
    功能描述: 计算数据的均值。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        axis (int): 计算均值的轴向。0表示按列，1表示按行。默认为0。

    返回:
        Union[np.ndarray, float]: 均值结果。如果axis为None，返回浮点数；否则返回NumPy数组。
    """
    return np.mean(data, axis=axis)


def std_dev(data: Union[np.ndarray, pd.DataFrame], axis: int = 0, ddof: int = 1) -> Union[np.ndarray, float]:
    """
    功能描述: 计算数据的标准差。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        axis (int): 计算标准差的轴向。0表示按行，1表示按列。默认为0。
        ddof (int): 自由度。默认为1（样本标准差）。

    返回:
        Union[np.ndarray, float]: 标准差结果。如果axis为None，返回浮点数；否则返回NumPy数组。
    """
    return np.std(data, axis=axis, ddof=ddof)


def variance(data: Union[np.ndarray, pd.DataFrame], axis: int = 0, ddof: int = 1) -> Union[np.ndarray, float]:
    """
    功能描述: 计算数据的方差。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        axis (int): 计算方差的轴向。0表示按列，1表示按行。默认为0。
        ddof (int): 自由度。默认为1（样本方差）。

    返回:
        Union[np.ndarray, float]: 方差结果。如果axis为None，返回浮点数；否则返回NumPy数组。
    """
    return np.var(data, axis=axis, ddof=ddof)


def correlation(a: Union[np.ndarray, pd.DataFrame], b: Union[np.ndarray, pd.DataFrame], axis: int = 0) -> Union[
    np.ndarray, float]:
    """
    功能描述: 计算两个输入之间的相关系数（横截面或时间序列）。

    参数:
        a (Union[np.ndarray, pd.DataFrame]): 第一个输入数据，可以是NumPy数组或Pandas DataFrame。
        b (Union[np.ndarray, pd.DataFrame]): 第二个输入数据，可以是NumPy数组或Pandas DataFrame。
        axis (int): 计算相关系数的轴向。0表示按列（时间序列），1表示按行（横截面）。默认为0。

    返回:
        Union[np.ndarray, float]: 相关系数结果。如果输入是1D数组，返回浮点数；否则返回NumPy数组。
    """


def covariance(a: Union[np.ndarray, pd.DataFrame], b: Union[np.ndarray, pd.DataFrame], axis: int = 0) -> Union[
    np.ndarray, float]:
    """
    功能描述: 计算两个输入之间的协方差（横截面或时间序列）。

    参数:
        a (Union[np.ndarray, pd.DataFrame]): 第一个输入数据，可以是NumPy数组或Pandas DataFrame。
        b (Union[np.ndarray, pd.DataFrame]): 第二个输入数据，可以是NumPy数组或Pandas DataFrame。
        axis (int): 计算协方差的轴向。0表示按列（时间序列），1表示按行（横截面）。默认为0。

    返回:
        Union[np.ndarray, float]: 协方差结果。如果输入是1D数组，返回浮点数；否则返回NumPy数组。
    """


def max_val(data: Union[np.ndarray, pd.DataFrame], axis: int = 0) -> Union[np.ndarray, float]:
    """
    功能描述: 计算数据的最大值。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        axis (int): 计算最大值的轴向。0表示按列，1表示按行。默认为0。

    返回:
        Union[np.ndarray, float]: 最大值结果。如果axis为None，返回浮点数；否则返回NumPy数组。
    """
    return np.max(data, axis=axis)


def min_val(data: Union[np.ndarray, pd.DataFrame], axis: int = 0) -> Union[np.ndarray, float]:
    """
    功能描述: 计算数据的最小值。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        axis (int): 计算最小值的轴向。0表示按列，1表示按行。默认为0。

    返回:
        Union[np.ndarray, float]: 最小值结果。如果axis为None，返回浮点数；否则返回NumPy数组。
    """
    return np.min(data, axis=axis)


def median(data: Union[np.ndarray, pd.DataFrame], axis: int = 0) -> Union[np.ndarray, float]:
    """
    功能描述: 计算数据的中位数。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        axis (int): 计算中位数的轴向。0表示按列，1表示按行。默认为0。

    返回:
        Union[np.ndarray, float]: 中位数结果。如果axis为None，返回浮点数；否则返回NumPy数组。
    """
    return np.median(data, axis=axis)


def quantile(data: Union[np.ndarray, pd.DataFrame], q: Union[float, Sequence[float]], axis: int = 0) -> Union[
    np.ndarray, float]:
    """
    功能描述: 计算数据的分位数。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        q (Union[float, Sequence[float]]): 0到1之间的分位数或分位数序列。
        axis (int): 计算分位数的轴向。0表示按列，1表示按行。默认为0。

    返回:
        Union[np.ndarray, float]: 分位数结果。如果q是单个浮点数且axis为None，返回浮点数；否则返回NumPy数组。
    """
    return np.quantile(data, q, axis=axis)


# --- 时间序列算子 (通常作用于axis=0，即时间轴) ---

def ts_delay(data: Union[np.ndarray, pd.DataFrame], window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 获取过去第N期的数据（时间序列延迟）。也称为 'shift' 或 'lag'。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据。
        window (int): 延迟的期数。
        axis (int): 滚动的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 延迟N期后的数据。
    """
    return pd.DataFrame(data).shift(periods=window, axis=axis).values


def ts_delta(data: Union[np.ndarray, pd.DataFrame], window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算当前数据与N期前数据的差值。也称为 'diff'。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据。
        window (int): 差值的期数。
        axis (int): 滚动的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: N期差值的结果。
    """
    return pd.DataFrame(data).diff(periods=window, axis=axis).values


def moving_average(data: Union[np.ndarray, pd.DataFrame], window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算数据的移动平均。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        window (int): 移动平均的窗口大小。
        axis (int): 计算移动平均的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 移动平均的结果，为NumPy数组。
    """
    return pd.DataFrame(data).rolling(window=window, axis=axis).mean().values


def ts_std(data: Union[np.ndarray, pd.DataFrame], window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算数据的滚动标准差。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据。
        window (int): 滚动窗口大小。
        axis (int): 滚动的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动标准差的结果。
    """
    return pd.DataFrame(data).rolling(window=window, axis=axis).std().values


# Alias ts_mean to moving_average
ts_mean = moving_average


def exponential_moving_average(data: Union[np.ndarray, pd.DataFrame], span: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算数据的指数移动平均。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        span (int): 指数移动平均的跨度（span）。
        axis (int): 计算指数移动平均的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 指数移动平均的结果，为NumPy数组。
    """
    return pd.DataFrame(data).ewm(span=span, axis=axis).mean().values


def rolling_skew(data: Union[np.ndarray, pd.DataFrame], window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 在指定窗口内计算数据的滚动偏度。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据。
        window (int): 滚动窗口的大小。
        axis (int): 滚动的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动偏度的结果。
    """
    return pd.DataFrame(data).rolling(window=window, axis=axis).skew().values


def rolling_kurt(data: Union[np.ndarray, pd.DataFrame], window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 在指定窗口内计算数据的滚动峰度。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据。
        window (int): 滚动窗口的大小。
        axis (int): 滚动的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动峰度的结果。
    """
    return pd.DataFrame(data).rolling(window=window, axis=axis).kurt().values


def rolling_quantile(data: Union[np.ndarray, pd.DataFrame], window: int, quantile_level: float, axis: int = 0) -> np.ndarray:
    """
    功能描述: 在指定窗口内计算数据的滚动分位数。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据。
        window (int): 滚动窗口的大小。
        quantile_level (float): 要计算的分位数 (0.0 到 1.0)。
        axis (int): 滚动的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动分位数的结果。
    """
    return pd.DataFrame(data).rolling(window=window, axis=axis).quantile(quantile_level).values


def rolling_corr(a: Union[np.ndarray, pd.DataFrame], b: Union[np.ndarray, pd.DataFrame], window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 在指定窗口内计算两个序列的滚动相关系数。

    参数:
        a (Union[np.ndarray, pd.DataFrame]): 第一个输入数据。
        b (Union[np.ndarray, pd.DataFrame]): 第二个输入数据。
        window (int): 滚动窗口的大小。
        axis (int): 滚动的轴向。必须为0（时间序列）。默认为0。

    返回:
        np.ndarray: 滚动相关系数的结果。
    """
    if axis != 0:
        raise ValueError("rolling_corr only supports axis=0 (time-series correlation).")
    df_a = pd.DataFrame(a)
    df_b = pd.DataFrame(b)
    return df_a.rolling(window=window, axis=0).corr(df_b).values


def rolling_cov(a: Union[np.ndarray, pd.DataFrame], b: Union[np.ndarray, pd.DataFrame], window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 在指定窗口内计算两个序列的滚动协方差。

    参数:
        a (Union[np.ndarray, pd.DataFrame]): 第一个输入数据。
        b (Union[np.ndarray, pd.DataFrame]): 第二个输入数据。
        window (int): 滚动窗口的大小。
        axis (int): 滚动的轴向。必须为0（时间序列）。默认为0。

    返回:
        np.ndarray: 滚动协方差的结果。
    """
    if axis != 0:
        raise ValueError("rolling_cov only supports axis=0 (time-series covariance).")
    df_a = pd.DataFrame(a)
    df_b = pd.DataFrame(b)
    return df_a.rolling(window=window, axis=0).cov(df_b).values


def ts_rank(data: Union[np.ndarray, pd.DataFrame], window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算数据在过去N期时间序列窗口中的百分比排名。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据。
        window (int): 滚动窗口的大小。
        axis (int): 滚动的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 时间序列排名的结果 (0到1之间)。
    """
    return pd.DataFrame(data).rolling(window=window, axis=axis).rank(pct=True).values


def time_series_decay(data: Union[np.ndarray, pd.DataFrame], halflife: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算数据的时间序列衰减加权平均 (指数加权移动平均)。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        halflife (int): 半衰期，用于计算衰减权重。
        axis (int): 计算衰减加权平均的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 时间序列衰减加权平均的结果，为NumPy数组。
    """
    return pd.DataFrame(data).ewm(halflife=halflife, axis=axis).mean().values


# --- 复合财务指标算子 (Complex Financial Metrics) ---
# 这些算子通常用于评估收益序列的风险和表现，其计算逻辑是路径依赖的。

def rolling_max_drawdown(data: Union[np.ndarray, pd.DataFrame], window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算滚动最大回撤。最大回撤衡量了在指定窗口内资产净值从峰值回落的最大百分比。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入的净值或价格序列。
        window (int): 滚动窗口的大小。
        axis (int): 滚动的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动最大回撤的结果 (通常为负值)。
    """
    def max_drawdown(series):
        cumulative = (1 + series).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    # rolling.apply 只能在 axis=0 上操作
    if axis != 0:
        raise ValueError("rolling_max_drawdown only supports axis=0 (time-series).")
    
    return pd.DataFrame(data).rolling(window=window, axis=0).apply(max_drawdown, raw=False).values


def downside_deviation(data: Union[np.ndarray, pd.DataFrame], window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算下行标准差（下行波动率）。只针对窗口期内小于0的收益率计算标准差。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入的收益率序列。
        window (int): 滚动窗口的大小。
        axis (int): 滚动的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动下行标准差的结果。
    """
    def calculate_downside_std(series):
        downside_returns = series[series < 0]
        return downside_returns.std(ddof=1)

    if axis != 0:
        raise ValueError("downside_deviation only supports axis=0 (time-series).")
        
    return pd.DataFrame(data).rolling(window=window, axis=0).apply(calculate_downside_std, raw=False).values


def sharpe_ratio(data: Union[np.ndarray, pd.DataFrame], window: int, periods_per_year: int = 252, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算滚动夏普比率。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入的收益率序列。
        window (int): 滚动窗口的大小。
        periods_per_year (int): 每年的周期数（如日频为252，月频为12）。用于年化。
        axis (int): 滚动的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动夏普比率的结果。
    """
    if axis != 0:
        raise ValueError("sharpe_ratio only supports axis=0 (time-series).")
        
    rolling_mean = pd.DataFrame(data).rolling(window=window, axis=0).mean()
    rolling_std = pd.DataFrame(data).rolling(window=window, axis=0).std(ddof=1)
    
    # 避免除以零
    sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(periods_per_year)
    return sharpe.values


def sortino_ratio(data: Union[np.ndarray, pd.DataFrame], window: int, periods_per_year: int = 252, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算滚动索提诺比率。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入的收益率序列。
        window (int): 滚动窗口的大小。
        periods_per_year (int): 每年的周期数。
        axis (int): 滚动的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动索提诺比率的结果。
    """
    if axis != 0:
        raise ValueError("sortino_ratio only supports axis=0 (time-series).")
        
    rolling_mean = pd.DataFrame(data).rolling(window=window, axis=0).mean()
    rolling_downside_std = downside_deviation(data, window, axis=0)
    
    # 避免除以零
    sortino = (rolling_mean / pd.DataFrame(rolling_downside_std).replace(0, np.nan)) * np.sqrt(periods_per_year)
    return sortino.values


def calmar_ratio(data: Union[np.ndarray, pd.DataFrame], window: int, periods_per_year: int = 252, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算滚动卡玛比率 (年化收益 / 最大回撤)。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入的收益率序列。
        window (int): 滚动窗口的大小。
        periods_per_year (int): 每年的周期数。
        axis (int): 滚动的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动卡玛比率的结果。
    """
    if axis != 0:
        raise ValueError("calmar_ratio only supports axis=0 (time-series).")
        
    annualized_return = pd.DataFrame(data).rolling(window=window, axis=0).mean() * periods_per_year
    max_dd = rolling_max_drawdown(data, window, axis=0)
    
    # 避免除以零, 并取最大回撤的绝对值
    calmar = annualized_return / pd.DataFrame(np.abs(max_dd)).replace(0, np.nan)
    return calmar.values


def alpha(data: Union[np.ndarray, pd.DataFrame], benchmark_data: Union[np.ndarray, pd.DataFrame], window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算滚动Alpha (对基准的超额收益)。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 资产收益率序列。
        benchmark_data (Union[np.ndarray, pd.DataFrame]): 基准收益率序列。
        window (int): 滚动窗口的大小。
        axis (int): 滚动的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动Alpha的结果。
    """
    def calculate_alpha(y, x):
        x_with_const = sm.add_constant(x)
        model = sm.OLS(y, x_with_const).fit()
        return model.params.iloc[0] # 返回截距项 (alpha)

    if axis != 0:
        raise ValueError("alpha only supports axis=0 (time-series).")

    df_data = pd.DataFrame(data)
    df_benchmark = pd.DataFrame(benchmark_data)
    
    # 使用expanding来确保有足够的数据点进行回归
    results = pd.Series(index=df_data.index, dtype=float)
    for i in range(window, len(df_data)):
        y_slice = df_data.iloc[i-window:i]
        x_slice = df_benchmark.iloc[i-window:i]
        results.iloc[i] = calculate_alpha(y_slice, x_slice)
        
    return results.values


def beta(data: Union[np.ndarray, pd.DataFrame], benchmark_data: Union[np.ndarray, pd.DataFrame], window: int, axis: int = 0) -> np.ndarray:
    """
    功能描述: 计算滚动Beta (对基准的系统性风险)。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 资产收益率序列。
        benchmark_data (Union[np.ndarray, pd.DataFrame]): 基准收益率序列。
        window (int): 滚动窗口的大小。
        axis (int): 滚动的轴向。0表示按行（时间序列），1表示按列（横截面）。默认为0。

    返回:
        np.ndarray: 滚动Beta的结果。
    """
    def calculate_beta(y, x):
        x_with_const = sm.add_constant(x)
        model = sm.OLS(y, x_with_const).fit()
        return model.params.iloc[1] # 返回斜率项 (beta)

    if axis != 0:
        raise ValueError("beta only supports axis=0 (time-series).")

    df_data = pd.DataFrame(data)
    df_benchmark = pd.DataFrame(benchmark_data)
    
    results = pd.Series(index=df_data.index, dtype=float)
    for i in range(window, len(df_data)):
        y_slice = df_data.iloc[i-window:i]
        x_slice = df_benchmark.iloc[i-window:i]
        results.iloc[i] = calculate_beta(y_slice, x_slice)
        
    return results.values




# --- 数据预处理与因子结果处理算子 (仅用于原始数据预处理和因子结果评估前端处理，不用于因子计算逻辑的构建) ---

def neutralize(factor_data: pd.DataFrame, risk_factors_data: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
    """
    功能描述: 对因子数据进行中性化处理，消除其中包含的风险因子暴露。

    参数:
        factor_data (pd.DataFrame): 因子值数据，Pandas DataFrame。
        risk_factors_data (pd.DataFrame): 风险因子数据，Pandas DataFrame，其索引和列应与factor_data对齐。
        axis (int): 中性化的轴向。0表示时间序列中性化（按列回归），1表示横截面中性化（按行回归）。默认为1。

    返回:
        pd.DataFrame: 中性化后的因子数据，Pandas DataFrame。
    """
    if not isinstance(factor_data, pd.DataFrame) or not isinstance(risk_factors_data, pd.DataFrame):
        raise TypeError("factor_data and risk_factors_data must be pandas DataFrames.")

    # 确保索引和列对齐
    if not factor_data.index.equals(risk_factors_data.index) or \
            not factor_data.columns.equals(risk_factors_data.columns):
        # 尝试重新索引对齐，缺失值填充NaN
        common_index = factor_data.index.intersection(risk_factors_data.index)
        common_columns = factor_data.columns.intersection(risk_factors_data.columns)

        factor_data = factor_data.loc[common_index, common_columns]
        risk_factors_data = risk_factors_data.loc[common_index, common_columns]

        if factor_data.empty or risk_factors_data.empty:
            print("Warning: No common index/columns found for neutralization. Returning NaN.")
            return pd.DataFrame(np.nan, index=factor_data.index, columns=factor_data.columns)

    neutralized_factor = pd.DataFrame(np.nan, index=factor_data.index, columns=factor_data.columns)

    if axis == 1:  # 横截面中性化 (对每个日期进行回归)
        for date in factor_data.index:
            y = factor_data.loc[date].dropna()  # 移除NaN值
            X = risk_factors_data.loc[date].loc[y.index].dropna()  # 风险因子也只取y中非NaN的部分

            # 确保X和y有共同的非NaN索引
            common_non_nan_index = y.index.intersection(X.index)
            if len(common_non_nan_index) < 2:  # 至少需要两个点才能回归
                continue

            y_clean = y.loc[common_non_nan_index]
            X_clean = X.loc[common_non_nan_index]

            # 添加常数项
            X_clean = sm.add_constant(X_clean, has_constant='add')

            try:
                model = sm.OLS(y_clean, X_clean, missing='drop')  # missing='drop' 确保处理NaN
                results = model.fit()
                # 残差即为中性化后的因子值
                neutralized_factor.loc[date, common_non_nan_index] = results.resid
            except Exception as e:
                # print(f"Warning: Regression failed for date {date}: {e}")
                pass  # 失败时保持NaN
        return neutralized_factor

    elif axis == 0:  # 时间序列中性化 (对每个金融产品进行回归)
        for col in factor_data.columns:
            y = factor_data[col].dropna()  # 移除NaN值
            X = risk_factors_data[col].loc[y.index].dropna()  # 风险因子也只取y中非NaN的部分

            # 确保X和y有共同的非NaN索引
            common_non_nan_index = y.index.intersection(X.index)
            if len(common_non_nan_index) < 2:  # 至少需要两个点才能回归
                continue

            y_clean = y.loc[common_non_nan_index]
            X_clean = X.loc[common_non_nan_index]

            # 添加常数项
            X_clean = sm.add_constant(X_clean, has_constant='add')

            try:
                model = sm.OLS(y_clean, X_clean, missing='drop')
                results = model.fit()
                # 残差即为中性化后的因子值
                neutralized_factor.loc[common_non_nan_index, col] = results.resid
            except Exception as e:
                # print(f"Warning: Regression failed for column {col}: {e}")
                pass  # 失败时保持NaN
        return neutralized_factor
    else:
        raise ValueError("Axis must be 0 (time-series) or 1 (cross-sectional).")


def winsorize(data: Union[np.ndarray, pd.DataFrame], lower_percentile: float = 0.01, upper_percentile: float = 0.99,
              axis: Optional[int] = None) -> Union[np.ndarray, pd.DataFrame]:
    """
    功能描述: 对数据进行缩尾处理，将极端值替换为指定分位数的值。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        lower_percentile (float): 下限分位数（例如 0.01）。默认为0.01。
        upper_percentile (float): 上限分位数（例如 0.99）。默认为0.99。
        axis (Optional[int]): 缩尾的轴向。0表示按列（时间序列），1表示按行（横截面），None表示全局。默认为None。

    返回:
        Union[np.ndarray, pd.DataFrame]: 缩尾处理后的数据，类型与输入data的类型兼容。
    """
    if isinstance(data, pd.DataFrame):
        if axis is None:
            lower_bound = data.quantile(lower_percentile).min()
            upper_bound = data.quantile(upper_percentile).max()
        elif axis == 0:  # 按列（股票）缩尾
            lower_bound = data.quantile(lower_percentile, axis=0)
            upper_bound = data.quantile(upper_percentile, axis=0)
        elif axis == 1:  # 按行（日期）缩尾
            lower_bound = data.quantile(lower_percentile, axis=1)
            upper_bound = data.quantile(upper_percentile, axis=1)

        winsorized_data = data.clip(lower=lower_bound, upper=upper_bound, axis=axis)
        return winsorized_data

    # For numpy arrays
    lower_bound = np.percentile(data, lower_percentile * 100, axis=axis)
    upper_bound = np.percentile(data, upper_percentile * 100, axis=axis)

    winsorized_data = np.clip(data, lower_bound, upper_bound)
    return winsorized_data


def clip(data: Union[np.ndarray, pd.DataFrame], lower_bound: Optional[Union[float, int]] = None,
         upper_bound: Optional[Union[float, int]] = None) -> Union[np.ndarray, pd.DataFrame]:
    """
    功能描述: 将数据值限制在指定的上下限范围内。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        lower_bound (Optional[Union[float, int]]): 数据的下限值。默认为None（无下限）。
        upper_bound (Optional[Union[float, int]]): 数据的上限值。默认为None（无上限）。

    返回:
        Union[np.ndarray, pd.DataFrame]: 截断/裁剪后的数据，类型与输入data的类型兼容。
    """
    return np.clip(data, lower_bound, upper_bound)


def fill_na(data: Union[np.ndarray, pd.DataFrame], method: str = 'ffill', value: Optional[Union[float, int]] = None,
            axis: Optional[int] = None) -> Union[np.ndarray, pd.DataFrame]:
    """
    功能描述: 填充数据中的缺失值（NaN）。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        method (str): 填充缺失值的方法。可选值包括 'ffill' (前向填充), 'bfill' (后向填充), 'zero' (零值填充), 'mean' (均值填充), 'median' (中位数填充), 'value' (指定值填充)。默认为'ffill'。
        value (Optional[Union[float, int]]): 当method='value'时，用于填充的值。默认为None。
        axis (Optional[int]): 填充的轴向。0表示按列，1表示按行。默认为None。

    返回:
        Union[np.ndarray, pd.DataFrame]: 填充缺失值后的数据，类型与输入data的类型兼容。
    """
    if isinstance(data, pd.DataFrame):
        if method == 'ffill':
            return data.ffill(axis=axis)
        elif method == 'bfill':
            return data.bfill(axis=axis)
        elif method == 'zero':
            return data.fillna(0)
        elif method == 'mean':
            return data.fillna(data.mean(axis=axis))
        elif method == 'median':
            return data.fillna(data.median(axis=axis))
        elif method == 'value':
            return data.fillna(value)

    # For numpy arrays (simplified, often requires pandas for sophisticated fillna)
    if method == 'zero':
        return np.nan_to_num(data, nan=0)
    elif method == 'value':
        return np.where(np.isnan(data), value, data)
    # ffill/bfill for numpy arrays are more complex, often done via pandas or custom loops
    return data  # Placeholder for other methods


def cross_sectional_rank(data: Union[np.ndarray, pd.DataFrame], axis: int = 1) -> Union[np.ndarray, pd.DataFrame]:
    """
    功能描述: 对数据进行横截面排名（通常对每个日期进行排名）。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        axis (int): 排名的轴向。通常为1表示横截面排名。默认为1。

    返回:
        Union[np.ndarray, pd.DataFrame]: 排名后的数据，类型与输入data的类型兼容。
    """
    if isinstance(data, pd.DataFrame):
        return data.rank(axis=axis, pct=True)
    # For numpy arrays, assuming 2D array where axis=1 is cross-section
    if axis == 1:
        ranked_data = np.zeros_like(data, dtype=float)
        for i in range(data.shape[0]):
            temp_series = pd.Series(data[i, :])
            ranked_data[i, :] = temp_series.rank(pct=True).values
        return ranked_data
    return data  # Placeholder for other axes


def cross_sectional_scale(data: Union[np.ndarray, pd.DataFrame], axis: int = 1) -> Union[np.ndarray, pd.DataFrame]:
    """
    功能描述: 对数据进行横截面标准化（Z-score标准化）。

    参数:
        data (Union[np.ndarray, pd.DataFrame]): 输入数据，可以是NumPy数组或Pandas DataFrame。
        axis (int): 标准化的轴向。通常为1表示横截面标准化。默认为1。

    返回:
        Union[np.ndarray, pd.DataFrame]: 标准化后的数据，类型与输入data的类型兼容。
    """
    if isinstance(data, pd.DataFrame):
        return (data - data.mean(axis=axis)) / data.std(axis=axis)

    # For numpy arrays, assuming 2D array where axis=1 is cross-section
    if axis == 1:
        mean_vals = np.mean(data, axis=axis, keepdims=True)
        std_vals = np.std(data, axis=axis, keepdims=True)
        # 避免除以零
        scaled_data = np.divide(data - mean_vals, std_vals, out=np.full_like(data, np.nan), where=std_vals != 0)
        return scaled_data
    return data  # Placeholder for other axes
