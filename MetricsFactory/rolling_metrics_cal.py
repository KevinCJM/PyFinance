# -*- encoding: utf-8 -*-
"""
@File: rolling_metrics_cal.py
@Modify Time: 2025/4/15 17:11       
@Author: Kevin-Chen
@Descriptions: 滚动指标的计算逻辑
"""
import time
import warnings
import traceback
import numpy as np
import pandas as pd
from functools import wraps
from numba import prange, float64, njit
from scipy.stats import norm, kurtosis, skew
from numpy.lib.stride_tricks import sliding_window_view
from MetricsFactory.metrics_cal_config import return_ann_factor, risk_ann_factor, log_ann_return, log_daily_return

warnings.filterwarnings("ignore")
pd.set_option('display.width', 1000)  # 表格不分段显示
pd.set_option('display.max_columns', 1000)  # 显示字段的数量


@njit(float64[:, :, :](float64[:, :], float64, float64), cache=True, parallel=True)
def kdj_recursive(rsv, alpha_k, alpha_d):
    n_days, n_funds = rsv.shape

    # 三维数组：最后一维表示 [K, D, J]
    kdj = np.empty((n_days, n_funds, 3), dtype=np.float64)

    # 初始化第0天：K = D = 50，J = 3K - 2D = 50
    for i in prange(n_funds):
        kdj[0, i, 0] = 50.0  # K
        kdj[0, i, 1] = 50.0  # D
        kdj[0, i, 2] = 50.0  # J

    for t in range(1, n_days):
        for i in prange(n_funds):
            if np.isfinite(rsv[t, i]):
                k_prev = kdj[t - 1, i, 0]
                d_prev = kdj[t - 1, i, 1]

                k_t = alpha_k * rsv[t, i] + (1 - alpha_k) * k_prev
                d_t = alpha_d * k_t + (1 - alpha_d) * d_prev
                j_t = 3 * k_t - 2 * d_t
            else:
                k_t = kdj[t - 1, i, 0]
                d_t = kdj[t - 1, i, 1]
                j_t = 3 * k_t - 2 * d_t

            kdj[t, i, 0] = k_t
            kdj[t, i, 1] = d_t
            kdj[t, i, 2] = j_t

    return kdj


# 计算移动平均
@njit(float64[:, :](float64[:, :], float64), parallel=True, cache=True)
def rolling_mean_2d_numba(arr: np.ndarray, window: float) -> np.ndarray:
    n_days, n_assets = arr.shape
    result = np.empty((n_days, n_assets), dtype=np.float64)
    result[:, :] = np.nan  # 先全部填 nan

    w = int(window)

    for j in prange(n_assets):  # 并发处理每列
        for i in range(w - 1, n_days):
            s = 0.0
            count = 0
            for k in range(i - w + 1, i + 1):
                val = arr[k, j]
                if not np.isnan(val):
                    s += val
                    count += 1
            result[i, j] = s / count if count > 0 else np.nan

    return result


# 计算移动标准差
@njit(float64[:, :](float64[:, :], float64), parallel=True, cache=True)
def rolling_std_2d_numba(data: np.ndarray, rolling_days: float) -> np.ndarray:
    """
    计算二维ndarray的滚动样本标准差（有自由度），忽略NaN值，
    并使用 Numba 的 @njit 和 parallel=True。

    参数:
        data (np.ndarray): 输入的二维ndarray (dtype=float64)。
        rolling_days (float): 滚动窗口的天数 (将自动转换为整数)。

    返回:
        np.ndarray: 包含每列滚动样本标准差的二维ndarray (dtype=float64)。
    """
    rows, cols = data.shape
    results = np.empty_like(data, dtype=np.float64)
    window = int(rolling_days)  # 将滚动天数转换为整数

    for j in prange(cols):
        for i in range(rows):
            start = max(0, i - window + 1)
            end = i + 1
            window_data = data[start:end, j]
            valid_values = window_data[~np.isnan(window_data)]
            n = valid_values.size
            if n >= 2:  # 至少需要两个有效值才能计算样本标准差
                mean = np.sum(valid_values) / n
                variance = np.sum((valid_values - mean) ** 2) / (n - 1)
                results[i, j] = np.sqrt(variance)
            elif n == 1:
                results[i, j] = 0.0  # 如果只有一个有效值，样本标准差为 0
            else:
                results[i, j] = np.nan
    return results


# 计算滚动最小值
@njit(float64[:, :](float64[:, :], float64), parallel=True, cache=True)
def rolling_min_2d_numba(arr: np.ndarray, window: float) -> np.ndarray:
    """
    计算二维ndarray的滚动最小值，忽略NaN值。

    参数:
        arr (np.ndarray): 输入的二维ndarray (dtype=float64)。
        window (float): 滚动窗口的大小 (将自动转换为整数)。

    返回:
        np.ndarray: 包含每列滚动最小值的二维ndarray (dtype=float64)。
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
                results[i, j] = np.nanmin(valid_values)
            else:
                results[i, j] = np.nan
    return results


# 计算滚动最大值
@njit(float64[:, :](float64[:, :], float64), parallel=True, cache=True)
def rolling_max_2d_numba(arr: np.ndarray, window: float) -> np.ndarray:
    """
    计算二维ndarray的滚动最大值，忽略NaN值。

    参数:
        arr (np.ndarray): 输入的二维ndarray (dtype=float64)。
        window (float): 滚动窗口的大小 (将自动转换为整数)。

    返回:
        np.ndarray: 包含每列滚动最大值的二维ndarray (dtype=float64)。
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
                results[i, j] = np.nanmax(valid_values)
            else:
                results[i, j] = np.nan
    return results


# 计算指数加权移动平均
@njit(float64[:, :](float64[:, :], float64), parallel=True, cache=True)
def rolling_ewm_2d_numba(arr: np.ndarray, span: float) -> np.ndarray:
    """
    计算二维ndarray的指数加权移动平均 (EWMA)，忽略NaN值，adjust=False。

    参数:
        arr (np.ndarray): 输入的二维ndarray (dtype=float64)。
        span (float): EWMA 的 span 参数。

    返回:
        np.ndarray: 包含每列 EWMA 的二维ndarray (dtype=float64)。
    """
    rows, cols = arr.shape
    results = np.empty_like(arr, dtype=np.float64)
    alpha = 2.0 / (span + 1)

    for j in prange(cols):
        ewm_value = np.nan
        for i in range(rows):
            current_value = arr[i, j]
            if np.isnan(current_value):
                results[i, j] = ewm_value
            else:
                if np.isnan(ewm_value):
                    ewm_value = current_value
                else:
                    ewm_value = alpha * current_value + (1 - alpha) * ewm_value
                results[i, j] = ewm_value
    return results


# 计算滚动数据的总和
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


# 装饰器函数，用于缓存被装饰函数的计算结果
def cache_rolling_metric(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):

        # 布林带特殊处理
        if func.__name__ in ['cal_Boll', 'cal_KDJ', 'cal_MTMMA', 'cal_MAPSY', 'cal_MACR']:
            metric_name = kwargs['metric_name']
        else:
            # 从函数名中提取指标名称，去掉前缀 "cal_"
            metric_name = func.__name__.replace("cal_", "")

        # 如果指标已经存在于缓存中，则直接返回缓存中的值
        if metric_name in self.res_dict:
            return self.res_dict[metric_name]

        # 调用被装饰函数计算结果，并将结果缓存到 `self.res_dict` 中
        result = func(self, *args, **kwargs)
        self.res_dict[metric_name] = result

        return result

    return wrapper


class CalRollingMetrics:
    def __init__(self, fund_codes,
                 log_return_array,
                 close_price_array,
                 high_price_array,
                 low_price_array,
                 volume_array,
                 rolling_days,
                 days_array,
                 trans_to_cumulative_return=False):
        # 基金代码数组 (一维ndarray)
        self.fund_codes = fund_codes
        #  2维ndarray, 行表示日期, 列表示基金, 值为基金的对数收益率
        self.return_array = log_return_array
        # 2维ndarray, 行表示日期, 列表示基金, 值为基金的每日收盘价
        self.price_array = close_price_array
        # 2维ndarray, 行表示日期, 列表示基金, 值为基金的每日最高价
        self.high_array = high_price_array
        # 2维ndarray, 行表示日期, 列表示基金, 值为基金的每日最低价
        self.low_array = low_price_array
        # 2维ndarray, 行表示日期, 列表示基金, 值为基金的每日交易量
        self.volume_array = volume_array
        # 滚动天数, int
        self.rolling_days = rolling_days
        # 日期序列 (一维ndarray)
        self.days_array = days_array
        # 存储是否将收益率转换为累计收益率的选项
        self.cum_rtn = trans_to_cumulative_return
        # 原始数据的行数和列数，分别代表天数和基金数量
        self.n_days, self.n_funds = self.return_array.shape
        # 初始化结果字典，用于存储后续计算的业绩评估指标
        self.res_dict = dict()

        # self.price_df = pd.DataFrame(self.price_array)
        # self.high_df = pd.DataFrame(self.high_array)
        # self.low_df = pd.DataFrame(self.low_array)
        # self.volume_df = pd.DataFrame(self.volume_array)

    @cache_rolling_metric  # 滚动N日的收盘价标准差
    def cal_PriceSigma(self, **kwargs):
        return rolling_std_2d_numba(self.price_array, self.rolling_days)

    @cache_rolling_metric  # 滚动N日的收盘价均值
    def cal_CloseMA(self, **kwargs):
        return rolling_mean_2d_numba(self.price_array, self.rolling_days)

    @cache_rolling_metric  # 滚动N日的成交量均值
    def cal_VolMA(self, **kwargs):
        return rolling_mean_2d_numba(self.volume_array, self.rolling_days)

    @cache_rolling_metric  # 布林带上轨
    def cal_Boll(self, metric_name, **kwargs):
        try:
            k = int(metric_name.split('-')[1])
        except (IndexError, ValueError):
            raise ValueError(
                f"Invalid Boll metric_name format: '{metric_name}'. Should be like 'BollUp-2' or 'BollDo-2'")
        ma = self.cal_CloseMA()
        sigma = self.cal_PriceSigma()
        self.res_dict[f'BollUp-{k}'] = ma + k * sigma
        self.res_dict[f'BollDo-{k}'] = ma - k * sigma
        return self.res_dict[metric_name]

    @cache_rolling_metric  # 滚动N日的最低价
    def cal_L(self, **kwargs):
        return rolling_min_2d_numba(self.low_array, self.rolling_days)

    @cache_rolling_metric  # 滚动N日的最高价
    def cal_H(self, **kwargs):
        return rolling_max_2d_numba(self.high_array, self.rolling_days)

    @cache_rolling_metric  # 滚动N日的 RSV指标
    def cal_RSV(self, **kwargs):
        c = self.price_array  # 收盘价
        the_h = self.cal_H()  # N日最高价
        the_l = self.cal_L()  # N日最低价
        # 分母：避免除以 0
        range_ = the_h - the_l
        invalid_mask = range_ == 0  # 定位“价格没有波动”的点

        range_[range_ == 0] = np.nan  # 临时置 nan，稍后用 where 控制
        # RSV = (C -L) / (H - L) * 100
        rsv = (c - the_l) / range_ * 100
        # 对 nan 的位置，令 RSV = 50
        rsv = np.where(invalid_mask, 50, rsv)  # 对没有价格波动的数据填充50
        return rsv

    @cache_rolling_metric  # 滚动N日的 KDJ指标
    def cal_KDJ(self, metric_name, **kwargs):
        _, _, m = metric_name.split('-')
        m1, m2 = int(m), int(m)
        alpha_k, alpha_d = 1 / m1, 1 / m2

        rsv = self.cal_RSV()
        kdj = kdj_recursive(rsv, alpha_k, alpha_d)

        # 拆分成 K / D / J 并缓存
        self.res_dict[f'KDJ-K-{m}'] = kdj[:, :, 0]
        self.res_dict[f'KDJ-D-{m}'] = kdj[:, :, 1]
        self.res_dict[f'KDJ-J-{m}'] = kdj[:, :, 2]

        return self.res_dict[metric_name]

    @cache_rolling_metric  # 滚动N日的数移动平均收盘价
    def cal_EMA(self, **kwargs):
        return rolling_ewm_2d_numba(self.price_array, self.rolling_days)

    @cache_rolling_metric  # 计算RSI
    def cal_RSI(self, **kwargs):
        price = self.price_array  # shape: (n_days, n_assets)
        n_days, n_assets = price.shape
        window = self.rolling_days

        # 计算收盘价变动 (等价于 df.diff())
        diff = np.empty_like(price)
        diff[0, :] = np.nan
        diff[1:, :] = price[1:, :] - price[:-1, :]

        # 拆出涨跌部分
        gain = np.where(diff > 0, diff, 0)
        loss = np.where(diff < 0, -diff, 0)

        # 平均（忽略 NaN）
        avg_gain = rolling_mean_2d_numba(gain, window)
        avg_loss = rolling_mean_2d_numba(loss, window)

        # 前 window-1 行补 NaN，保持 shape 一致
        pad = np.full((window - 1, n_assets), np.nan)
        avg_gain = np.vstack([pad, avg_gain])
        avg_loss = np.vstack([pad, avg_loss])

        # 计算 RS = avg_gain / avg_loss，避免除0
        rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)

        # RSI = 100 - (100 / (1 + RS))
        rsi = 100 - 100 / (1 + rs)

        return rsi

    @cache_rolling_metric
    def cal_OBV(self, **kwargs):
        close = self.price_array  # shape: (n_days, n_funds)
        volume = self.volume_array  # shape: (n_days, n_funds)

        # 价格变动方向：+1 (涨), -1 (跌), 0 (持平)
        price_diff = np.diff(close, axis=0)
        direction = np.sign(price_diff)

        # 将方向扩展为 (n_days, n_funds)（第一行补 0）
        direction = np.vstack([np.zeros((1, self.n_funds)), direction])

        # 构造 OBV 增量
        obv_delta = direction * volume

        # 初始 OBV 为 0，累加方向量
        obv = np.nancumsum(obv_delta, axis=0)

        return obv

    @cache_rolling_metric  # 计算 MTM 动量指标
    def cal_MTM(self, **kwargs):
        out = np.full_like(self.price_array, np.nan, dtype=np.float64)
        out[self.rolling_days:, :] = self.price_array[self.rolling_days:, :] - self.price_array[:-self.rolling_days, :]
        return out

    @cache_rolling_metric  # 平滑动量指标
    def cal_MTMMA(self, metric_name, **kwargs):
        _, smooth = metric_name.split('-')
        smooth = int(smooth)
        mtm = self.cal_MTM()
        return rolling_mean_2d_numba(mtm, smooth)

    @cache_rolling_metric  # 三重指数平滑移动平均
    def cal_TRIX(self, **kwargs):
        ema1 = rolling_ewm_2d_numba(self.price_array, self.rolling_days)
        ema2 = rolling_ewm_2d_numba(ema1, self.rolling_days)
        ema3 = rolling_ewm_2d_numba(ema2, self.rolling_days)

        result = np.full_like(ema3, np.nan, dtype=np.float64)
        result[1:, :] = (ema3[1:, :] - ema3[:-1, :]) / ema3[:-1, :]
        return result * 100

    @cache_rolling_metric  # 三重指数平滑移动平均的移动平均
    def cal_MATRIX(self, metric_name, **kwargs):
        _, smooth = metric_name.split('-')
        # 获取 TRIX 值（自动从缓存或重新计算）
        trix_array = self.cal_TRIX()
        return rolling_mean_2d_numba(trix_array, int(smooth))

    @cache_rolling_metric  # 心理线
    def cal_PSY(self, **kwargs):
        # 1. 计算收盘价变化（等价于 df.diff()）
        diff = np.empty_like(self.price_array, dtype=np.float64)
        diff[0, :] = np.nan
        diff[1:, :] = self.price_array[1:, :] - self.price_array[:-1, :]

        # 2. 标记上涨日（大于0为1，否则为0）
        up_days = np.where(diff > 0, 1.0, 0.0)

        # 3. 计算过去 N 日的上涨天数（使用你已有的 numba 滚动和）
        up_days_sum = rolling_sum_2d_numba(up_days, self.rolling_days)

        # 4. 转为百分比
        psy = up_days_sum / self.rolling_days * 100

        return psy

    @cache_rolling_metric  # 心理线的移动均线
    def cal_MAPSY(self, metric_name, **kwargs):
        _, smooth = metric_name.split('-')
        psy = self.cal_PSY()
        # 对每列做 EMA 平滑
        return rolling_mean_2d_numba(psy, int(smooth))

    @cache_rolling_metric  # 计算 CCI 顺势指标
    def cal_CCI(self, **kwargs):
        # 1. 计算 TP = (H + L + C) / 3
        tp = (self.high_array + self.low_array + self.price_array) / 3  # shape: (n_days, n_assets)
        # 2. 计算 TP 的 N 日滚动均值（MA）
        tp_ma = rolling_mean_2d_numba(tp, self.rolling_days)  # shape: (n_days, n_assets)
        # 3. 计算 |TP - MA|，然后再对其做滚动均值 → MD
        tp_deviation = np.abs(tp - tp_ma)
        tp_md = rolling_mean_2d_numba(tp_deviation, self.rolling_days)
        # 4. CCI = (TP - MA) / (0.015 * MD)
        denominator = 0.015 * tp_md
        # 防止除以 0 或过小值
        denominator = np.where(denominator == 0, np.nan, denominator)
        cci = (tp - tp_ma) / denominator
        return cci

    @cache_rolling_metric  # 计算 CurrentRatio 现价能量强度指标
    def cal_CR(self, **kwargs):
        high = self.high_array
        low = self.low_array
        window = self.rolling_days
        n_days, n_assets = high.shape

        # 1. 构造前一日中间价 MP_{t-1} = (H_{t-1} + L_{t-1}) / 2
        mp = (high + low) / 2

        # 向后移动一行，首行补 nan
        mp_prev = np.empty_like(mp)
        mp_prev[0, :] = np.nan
        mp_prev[1:, :] = mp[:-1, :]

        # 2. 分子部分: max(0, H - MP_prev)
        up = np.maximum(0, high - mp_prev)

        # 3. 分母部分: max(0, MP_prev - L)
        down = np.maximum(0, mp_prev - low)

        # 4. 滚动求和（忽略 NaN）
        sum_up = rolling_sum_2d_numba(up, window)
        sum_down = rolling_sum_2d_numba(down, window)

        # 5. 计算 CR = sum_up / sum_down * 100，防除零
        denominator = np.where(sum_down == 0, np.nan, sum_down)
        cr = sum_up / denominator * 100

        return cr

    @cache_rolling_metric  # CR指标的移动平均
    def cal_MACR(self, metric_name, **kwargs):
        _, n, m = metric_name.split('-')
        cr = self.cal_CR()
        # 计算 CR 的 N日 移动平均
        ma_cr = rolling_mean_2d_numba(cr, int(n))
        # 后移 M 天
        shifted_ma_cr = np.full_like(ma_cr, np.nan)
        shifted_ma_cr[int(m):, :] = ma_cr[:-int(m), :]
        return shifted_ma_cr

    @cache_rolling_metric  # 计算 VR 指标
    def cal_VR(self, **kwargs):
        # 1. 计算涨跌状态：相对前一日
        diff = np.empty_like(self.price_array, dtype=np.float64)
        diff[0, :] = np.nan
        diff[1:, :] = self.price_array[1:, :] - self.price_array[:-1, :]
        # 2. 构造掩码
        up_mask = (diff > 0).astype(np.float64)  # 涨
        down_mask = (diff < 0).astype(np.float64)  # 跌
        even_mask = (diff == 0).astype(np.float64)  # 平
        # 3. 提取三种日的成交量
        vol_up = self.volume_array * up_mask
        vol_down = self.volume_array * down_mask
        vol_even = self.volume_array * even_mask
        # 4. 分别对3类成交量进行 N 日滚动求和
        sum_up = rolling_sum_2d_numba(vol_up, self.rolling_days)
        sum_down = rolling_sum_2d_numba(vol_down, self.rolling_days)
        sum_even = rolling_sum_2d_numba(vol_even, self.rolling_days)
        # 5. 按照公式计算 VR，避免除 0
        numerator = sum_up + 0.5 * sum_even
        denominator = sum_down + 0.5 * sum_even
        denominator = np.where(denominator == 0, np.nan, denominator)
        vr = numerator / denominator * 100
        return vr

    @cache_rolling_metric  # 计算 VR 指标
    def cal_MAVR(self, metric_name, **kwargs):
        _, n = metric_name.split('-')
        vr = self.cal_VR()
        return rolling_mean_2d_numba(vr, int(n))

    # 根据指标名 计算相应的指标值
    def cal_metric(self, metric_name, **kwargs):
        # 布林带计算处理, 会同时计算上下两个轨道
        if metric_name.startswith('Boll'):
            func_name = 'cal_Boll'
        elif metric_name.startswith('KDJ-'):
            func_name = 'cal_KDJ'
        elif metric_name.startswith('MTMMA-'):
            func_name = 'cal_MTMMA'
        elif metric_name.startswith('MATRIX-'):
            func_name = 'cal_MATRIX'
        elif metric_name.startswith('MAPSY-'):
            func_name = 'cal_MAPSY'
        elif metric_name.startswith('MACR-'):
            func_name = 'cal_MACR'
        elif metric_name.startswith('MAVR-'):
            func_name = 'cal_MAVR'
        else:
            func_name = f'cal_{metric_name}'
        kwargs['metric_name'] = metric_name

        try:
            return getattr(self, func_name)(**kwargs)
        except Exception as e:
            print(f"Error when calling '{func_name}': {e}")
            print(traceback.format_exc())

    def cal_all_metrics(self, metric_name_list, **kwargs):
        df_list = list()
        # 计算各个指标值
        for metric_name in metric_name_list:
            res_array = self.cal_metric(metric_name, **kwargs)
            sub_df = pd.DataFrame(
                data=res_array,  # 二维数组（值）
                columns=self.fund_codes,  # 列名（日期）
                index=self.days_array  # 索引（基金代码）
            )
            long_df = sub_df.stack().reset_index()
            long_df.columns = ['date', 'ts_code', f'{metric_name}:{self.rolling_days}']
            df_list.append(long_df)

        # 按关联列合并所有 DataFrame
        final_df = df_list[0]  # 从第一个 DataFrame 开始
        for df in df_list[1:]:
            final_df = pd.merge(final_df, df, on=['date', 'ts_code'], how='outer')
        return final_df


if __name__ == '__main__':
    price_df = pd.read_parquet('/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/wide_close_df.parquet')
    return_df = pd.read_parquet('/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/wide_log_return_df.parquet')
    high_df = pd.read_parquet('/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/wide_high_df.parquet')
    low_df = pd.read_parquet('/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/wide_low_df.parquet')
    vol_df = pd.read_parquet('/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/wide_vol_df.parquet')

    fund = ['510050.SH', '159915.SZ', '159912.SZ', '512500.SH', '164701.SZ', '511010.SH', '513100.SH', '513030.SH',
            '513080.SH', '513520.SH', '518880.SH', '161226.SZ', '501018.SH', '159981.SZ', '159985.SZ', '159980.SZ',
            ]
    price_df = price_df[fund]
    return_df = return_df[fund]
    high_df = high_df[fund]
    low_df = low_df[fund]
    vol_df = vol_df[fund]

    fund_codes_array = np.array(price_df.columns.tolist())
    days = price_df.index

    log_return = return_df.values
    close_price = price_df.values
    high_df = high_df.values
    low_df = low_df.values
    vol_df = vol_df.values

    cal = CalRollingMetrics(fund_codes=fund_codes_array,
                            log_return_array=log_return,
                            close_price_array=close_price,
                            high_price_array=high_df,
                            low_price_array=low_df,
                            volume_array=vol_df,
                            rolling_days=26,
                            days_array=days,
                            )
    # 计算所有指标
    res = cal.cal_all_metrics(['MAVR-6'])
    print(res[res['ts_code'] == '159980.SZ'])

    # from MetricsFactory.metrics_cal_config import create_rolling_metrics_map
    #
    # res_df = None  # 初始化空的 final_df
    # for roll_day, metrics_list in create_rolling_metrics_map().items():  # 滚动天数 和 指标列表
    #     print(roll_day)
    #     print(metrics_list)
    #     cal = CalRollingMetrics(fund_codes=fund_codes_array,
    #                             log_return_array=log_return,
    #                             close_price_array=close_price,
    #                             high_price_array=high_df,
    #                             low_price_array=low_df,
    #                             volume_array=vol_df,
    #                             rolling_days=roll_day,
    #                             days_array=days,
    #                             )
    #     # 计算所有指标
    #     res = cal.cal_all_metrics(metrics_list)
    #
    #     # 第一次循环直接赋值，后续循环按关联列合并
    #     if res_df is None:
    #         res_df = res
    #     else:
    #         res_df = pd.merge(
    #             res_df,
    #             res,
    #             on=['date', 'ts_code'],
    #             how='outer'  # 保留所有数据（即使某些行缺少部分指标）
    #         )
    #
    # res_df = res_df.reset_index(drop=True)  # 重置索引
    # print(res_df)
