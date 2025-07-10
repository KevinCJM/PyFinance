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


# --- 基础数学算子 ---

def add(a, b):
    """加法: a + b"""
    return a + b


def subtract(a, b):
    """减法: a - b"""
    return a - b


def multiply(a, b):
    """乘法: a * b"""
    return a * b


def divide(a, b):
    """除法: a / b, 避免除零"""
    return np.divide(a, b, out=np.full_like(a, np.nan), where=b != 0)


def log(a):
    """对数: ln(a), 避免对非正数取对数"""
    return np.log(a, out=np.full_like(a, np.nan), where=a > 0)


def abs_val(a):
    """绝对值: |a|"""
    return np.abs(a)


def power(a, p):
    """幂运算: a^p"""
    return np.power(a, p)


def sqrt(a):
    """平方根: sqrt(a), 避免对负数开方"""
    return np.sqrt(a, out=np.full_like(a, np.nan), where=a >= 0)


# --- 累积算子 (从序列起始点开始累积，无窗口限制) ---
# 这些算子执行的是从序列开始到当前点的累积操作，不接受窗口参数。

def cumulative_sum(data, axis=0):
    """累加 (从序列起始点开始累积)"""
    return np.cumsum(data, axis=axis)


def cumulative_product(data, axis=0):
    """累乘 (从序列起始点开始累积)"""
    return np.cumprod(data, axis=axis)


def cumulative_max(data, axis=0):
    """累积最大值 (从序列起始点开始累积)"""
    return np.maximum.accumulate(data, axis=axis)


def cumulative_min(data, axis=0):
    """累积最小值 (从序列起始点开始累积)"""
    return np.minimum.accumulate(data, axis=axis)


# --- 滚动算子 (在指定窗口内进行累积/统计，支持上下限参数，如 \sum_{i=1}^{N}[...]) ---
# 这些算子通过 'window' 参数定义了操作的上下限范围，实现了类似 LaTeX 中带上下标的求和/累积概念。

def rolling_sum(data, window, axis=0):
    """滚动累加 (在指定窗口内求和，window参数定义了上下限范围)"""
    return pd.DataFrame(data).rolling(window=window, axis=axis).sum().values


def rolling_product(data, window, axis=0):
    """滚动累乘 (在指定窗口内求积，window参数定义了上下限范围)"""
    return pd.DataFrame(data).rolling(window=window, axis=axis).prod().values


def rolling_max(data, window, axis=0):
    """滚动最大值 (在指定窗口内求最大值，window参数定义了上下限范围)"""
    return pd.DataFrame(data).rolling(window=window, axis=axis).max().values


def rolling_min(data, window, axis=0):
    """滚动最小值 (在指定窗口内求最小值，window参数定义了上下限范围)"""
    return pd.DataFrame(data).rolling(window=window, axis=axis).min().values


# --- 统计算子 ---

def mean(data, axis=0):
    """均值"""
    return np.mean(data, axis=axis)


def std_dev(data, axis=0, ddof=1):
    """标准差"""
    return np.std(data, axis=axis, ddof=ddof)


def variance(data, axis=0, ddof=1):
    """方差"""
    return np.var(data, axis=axis, ddof=ddof)


def correlation(a, b, axis=0):
    """相关系数 (横截面或时间序列)"""
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        return a.corrwith(b, axis=axis)
    # For 1D arrays, np.corrcoef returns a 2x2 matrix, we need the off-diagonal element
    return np.corrcoef(a, b)[0, 1] if a.ndim == 1 and b.ndim == 1 else np.nan  # Handle non-1D arrays gracefully


def covariance(a, b, axis=0):
    """协方差 (横截面或时间序列)"""
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        return a.cov(b, axis=axis)
    # For 1D arrays, np.cov returns a 2x2 matrix, we need the off-diagonal element
    return np.cov(a, b)[0, 1] if a.ndim == 1 and b.ndim == 1 else np.nan  # Handle non-1D arrays gracefully


def max_val(data, axis=0):
    """最大值"""
    return np.max(data, axis=axis)


def min_val(data, axis=0):
    """最小值"""
    return np.min(data, axis=axis)


def median(data, axis=0):
    """中位数"""
    return np.median(data, axis=axis)


def quantile(data, q, axis=0):
    """分位数"""
    return np.quantile(data, q, axis=axis)


# --- 时间序列算子 (通常作用于axis=0，即时间轴) ---

def moving_average(data, window, axis=0):
    """移动平均"""
    return pd.DataFrame(data).rolling(window=window, axis=axis).mean().values


def exponential_moving_average(data, span, axis=0):
    """指数移动平均"""
    return pd.DataFrame(data).ewm(span=span, axis=axis).mean().values


def rolling_rank(data, window, axis=0):
    """滚动排名"""
    return pd.DataFrame(data).rolling(window=window, axis=axis).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1],
                                                                      raw=False).values


def time_series_decay(data, halflife, axis=0):
    """时间序列衰减加权平均"""
    return pd.DataFrame(data).ewm(halflife=halflife, axis=axis).mean().values


# --- 数据预处理与因子结果处理算子 (仅用于原始数据预处理和因子结果评估前端处理，不用于因子计算逻辑的构建) ---

def neutralize(factor_data, risk_factors_data, axis=1):
    """
    中性化: 消除因子中包含的风险因子暴露。
    factor_data: 因子值 (DataFrame)
    risk_factors_data: 风险因子数据 (DataFrame, 与factor_data对齐)
    axis: 0为时间序列中性化，1为横截面中性化
    
    注意: risk_factors_data应与factor_data具有相同的索引和列，
          且其列应代表不同的风险因子。
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


def winsorize(data, lower_percentile=0.01, upper_percentile=0.99, axis=None):
    """
    缩尾处理: 将极端值替换为指定分位数的值。
    lower_percentile: 下限分位数 (例如 0.01)
    upper_percentile: 上限分位数 (例如 0.99)
    axis: 0为时间序列，1为横截面，None为全局
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


def clip(data, lower_bound=None, upper_bound=None):
    """截断/裁剪: 将值限制在指定范围内"""
    return np.clip(data, lower_bound, upper_bound)


def fill_na(data, method='ffill', value=None, axis=None):
    """
    缺失值填充。
    method: 'ffill', 'bfill', 'zero', 'mean', 'median', 'value'
    value: 当method='value'时指定填充值
    axis: 0为时间序列，1为横截面
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


def cross_sectional_rank(data, axis=1):
    """横截面排名 (通常对每个日期进行排名)"""
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


def cross_sectional_scale(data, axis=1):
    """
    横截面标准化 (Z-score标准化)。
    对每个交易日的所有金融产品，将因子值转换为均值为0、标准差为1的分布。
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
