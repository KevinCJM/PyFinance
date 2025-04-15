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


@njit
def kdj_recursive(rsv, alpha_k, alpha_d):
    n_days, n_funds = rsv.shape
    K = np.empty((n_days, n_funds))
    D = np.empty((n_days, n_funds))
    J = np.empty((n_days, n_funds))

    # 初始化
    K[0, :] = 50.0
    D[0, :] = 50.0

    for t in range(1, n_days):
        for i in range(n_funds):
            if np.isfinite(rsv[t, i]):
                K[t, i] = alpha_k * rsv[t, i] + (1 - alpha_k) * K[t - 1, i]
                D[t, i] = alpha_d * K[t, i] + (1 - alpha_d) * D[t - 1, i]
                J[t, i] = 3 * K[t, i] - 2 * D[t, i]
            else:
                K[t, i] = K[t - 1, i]
                D[t, i] = D[t - 1, i]
                J[t, i] = 3 * K[t, i] - 2 * D[t, i]

    return K, D, J


# 装饰器函数，用于缓存被装饰函数的计算结果
def cache_rolling_metric(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):

        # 布林带特殊处理
        if func.__name__ == 'cal_Boll':
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
                 rolling_days,
                 trans_to_cumulative_return=False):
        # 基金代码数组 (一维ndarray)
        self.fund_codes = fund_codes
        # 移除对数收益率数组中的全NaN行，并存储为返回数组 (2维ndarray, 行表示日期, 列表示基金, 值为基金的对数收益率)
        self.return_array = log_return_array[~np.all(np.isnan(log_return_array), axis=1)]
        # 移除收盘价数组中的全NaN行，并存储为价格数组   (2维ndarray, 行表示日期, 列表示基金, 值为基金的每日收盘价)
        self.price_array = close_price_array[~np.all(np.isnan(close_price_array), axis=1)]
        # 移除收盘价数组中的全NaN行，并存储为价格数组   (2维ndarray, 行表示日期, 列表示基金, 值为基金的每日最高价)
        self.high_array = high_price_array[~np.all(np.isnan(high_price_array), axis=1)]
        # 移除收盘价数组中的全NaN行，并存储为价格数组   (2维ndarray, 行表示日期, 列表示基金, 值为基金的每日最低价)
        self.low_array = low_price_array[~np.all(np.isnan(low_price_array), axis=1)]
        # 滚动天数, int
        self.rolling_days = rolling_days
        # 存储是否将收益率转换为累计收益率的选项
        self.cum_rtn = trans_to_cumulative_return
        # 原始数据的行数和列数，分别代表天数和基金数量
        self.n_days, self.n_funds = self.return_array.shape
        # 初始化结果字典，用于存储后续计算的业绩评估指标
        self.res_dict = dict()

        self.price_df = pd.DataFrame(self.price_array)
        self.high_df = pd.DataFrame(self.high_array)
        self.low_df = pd.DataFrame(self.low_array)

    @cache_rolling_metric  # 滚动N日的收盘价标准差
    def cal_PriceSigma(self, **kwargs):
        # 逐列进行 rolling 标准差计算，忽略 NaN（min_periods=1），得到新的 DataFrame
        rolling_std = self.price_df.rolling(window=self.rolling_days, min_periods=1).std()
        # 再转回 numpy，并返回
        return rolling_std.to_numpy()

    @cache_rolling_metric  # 滚动N日的收盘价均值
    def cal_CloseMA(self, **kwargs):
        rolling_mean = self.price_df.rolling(window=self.rolling_days, min_periods=1).mean()
        return rolling_mean.to_numpy()

    @cache_rolling_metric  # 布林带上轨
    def cal_Boll(self, metric_name, **kwargs):
        try:
            k = int(metric_name.split('-')[1])
        except (IndexError, ValueError):
            raise ValueError(
                f"Invalid Boll metric_name format: '{metric_name}'. Should be like 'BollUp-2' or 'BollDo-2'")
        ma = self.cal_CloseMA(self)
        sigma = self.cal_PriceSigma(self)
        self.res_dict[f'BollUp-{k}'] = ma + k * sigma
        self.res_dict[f'BollDo-{k}'] = ma - k * sigma
        return self.res_dict[metric_name]

    @cache_rolling_metric  # 滚动N日的最低价
    def cal_L(self):
        rolling_min = self.low_df.rolling(window=self.rolling_days, min_periods=1).min()
        return rolling_min.to_numpy()

    @cache_rolling_metric  # 滚动N日的最高价
    def cal_H(self):
        rolling_max = self.high_df.rolling(window=self.rolling_days, min_periods=1).max()
        return rolling_max.to_numpy()

    @cache_rolling_metric  # 滚动N日的 RSV指标
    def cal_RSV(self):
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
    def cal_KDJ(self):
        M1, M2 = 3, 3
        alpha_k, alpha_d = 1 / M1, 1 / M2

        rsv = self.cal_RSV()
        K, D, J = kdj_recursive(rsv, alpha_k, alpha_d)

        self.res_dict['K'] = K
        self.res_dict['D'] = D
        self.res_dict['J'] = J

        return K

    # 根据指标名 计算相应的指标值
    def cal_metric(self, metric_name, **kwargs):
        # 布林带计算处理, 会同时计算上下两个轨道
        if metric_name.startswith('Boll'):
            func_name = 'cal_Boll'
            kwargs['metric_name'] = metric_name
        else:
            func_name = f'cal_{metric_name}'
        try:
            return getattr(self, func_name)(**kwargs)
        except Exception as e:
            print(f"Error when calling '{func_name}': {e}")
            print(traceback.format_exc())


if __name__ == '__main__':
    price_df = pd.read_parquet('/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/wide_close_df.parquet')
    return_df = pd.read_parquet('/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/wide_log_return_df.parquet')
    high_df = pd.read_parquet('/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/wide_high_df.parquet')
    low_df = pd.read_parquet('/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/wide_low_df.parquet')

    # print(price_df.head())
    # print(return_df.head())

    price_df = price_df.resample('D').asfreq()
    return_df = return_df.resample('D').asfreq()

    fund_codes_array = np.array(price_df.columns.tolist())
    log_return = return_df.values
    close_price = price_df.values
    high_df = high_df.values
    low_df = low_df.values

    roll_d = 5

    cal = CalRollingMetrics(fund_codes=fund_codes_array,
                            log_return_array=log_return,
                            close_price_array=close_price,
                            high_price_array=high_df,
                            low_price_array=low_df,
                            rolling_days=roll_d,
                            )
    s_t = time.time()
    res = cal.cal_metric('RSV')
    print(res)
    # res = cal.cal_metric('L')
    # print(res)
    print(f"共 {len(res)} 条记录，耗时 {(time.time() - s_t):.4f} 秒")
    print(cal.res_dict)
