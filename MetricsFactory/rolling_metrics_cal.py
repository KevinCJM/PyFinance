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


# 装饰器函数，用于缓存被装饰函数的计算结果
def cache_rolling_metric(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):

        # 布林带特殊处理
        if func.__name__ in ['cal_Boll', 'cal_KDJ', 'cal_MTMMA']:
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

        self.price_df = pd.DataFrame(self.price_array)
        self.high_df = pd.DataFrame(self.high_array)
        self.low_df = pd.DataFrame(self.low_array)
        self.volume_df = pd.DataFrame(self.volume_array)

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

    @cache_rolling_metric  # 滚动N日的成交量均值
    def cal_VolMA(self, **kwargs):
        rolling_mean = self.volume_df.rolling(window=self.rolling_days, min_periods=1).mean()
        return rolling_mean.to_numpy()

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
        rolling_min = self.low_df.rolling(window=self.rolling_days, min_periods=1).min()
        return rolling_min.to_numpy()

    @cache_rolling_metric  # 滚动N日的最高价
    def cal_H(self, **kwargs):
        rolling_max = self.high_df.rolling(window=self.rolling_days, min_periods=1).max()
        return rolling_max.to_numpy()

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
        span = self.rolling_days  # 滚动窗口长度，常见值如 12、26
        # 使用 pandas 的 ewm（exponential weighted moving average）
        ema_df = self.price_df.ewm(span=span, adjust=False, min_periods=1).mean()
        # 转回 numpy 并返回
        return ema_df.to_numpy()

    @cache_rolling_metric
    def cal_RSI(self, **kwargs):
        # 1) 计算每日价格变动
        df_diff = price_df.diff()

        # 2) 计算收益和损失
        #    gain：大于0的diff部分，其他为0
        #    loss：小于0的diff取绝对值，其他为0
        df_gain = df_diff.clip(lower=0)  # 等价于 np.where(df_diff>0, df_diff, 0)
        df_loss = (-df_diff).clip(lower=0)  # 等价于 np.where(df_diff<0, -df_diff, 0)

        # 3) 分别对 gain 和 loss 进行简单移动平均 (SMA)
        avg_gain = df_gain.rolling(window=self.rolling_days, min_periods=1).mean()
        avg_loss = df_loss.rolling(window=self.rolling_days, min_periods=1).mean()

        # 4) 计算 RS = AvgGain / AvgLoss
        #    注意避免除零，这里简单做个防零处理：如果 avg_loss 为0，可视为RS极大；也可改用更严谨方法
        rs = avg_gain / avg_loss.replace(0, 1e-10)

        # 5) 计算 RSI = 100 - (100 / (1 + RS))
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
        # 1. 当前价格 - N天前价格
        mtm = self.price_df - self.price_df.shift(self.rolling_days)
        # 2. 转回 ndarray 返回
        return mtm.to_numpy()

    @cache_rolling_metric  # 平滑动量指标
    def cal_MTMMA(self, metric_name, **kwargs):
        _, smooth = metric_name.split('-')
        smooth = int(smooth)
        mtm = self.cal_MTM()
        df_mtm = pd.DataFrame(mtm, columns=self.fund_codes)
        mtmma = df_mtm.rolling(window=smooth, min_periods=1).mean()
        return mtmma.to_numpy()

    @cache_rolling_metric  # 三重指数平滑移动平均
    def cal_TRIX(self, **kwargs):
        # 一次 EMA
        ema1 = self.price_df.ewm(span=self.rolling_days, adjust=False, min_periods=1).mean()
        ema2 = ema1.ewm(span=self.rolling_days, adjust=False, min_periods=1).mean()
        ema3 = ema2.ewm(span=self.rolling_days, adjust=False, min_periods=1).mean()

        # 三重 EMA 的一阶变化率（百分比）
        trix = ema3.pct_change() * 100  # 百分比形式

        return trix.to_numpy()

    @cache_rolling_metric  # 三重指数平滑移动平均的移动平均
    def cal_MATRIX(self, metric_name, **kwargs):
        _, smooth = metric_name.split('-')
        # 获取 TRIX 值（自动从缓存或重新计算）
        trix_array = self.cal_TRIX()
        # 转回 DataFrame 做 EMA 平滑
        df_trix = pd.DataFrame(trix_array, columns=self.fund_codes)
        # 对每列做 EMA 平滑
        matrix_df = df_trix.rolling(window=int(smooth), min_periods=1).mean()
        return matrix_df.to_numpy()

    @cache_rolling_metric
    def cal_PSY(self, window: int = 12):
        # 收盘价变动（t - t-1）
        diff = self.price_df.diff()
        # 涨则记为 1，跌或持平记为 0
        up_days = (diff > 0).astype(int)
        # 过去 N 天上涨天数占比
        psy = up_days.rolling(window=window, min_periods=1).sum() / window * 100
        return psy.to_numpy()

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
                            rolling_days=12,
                            days_array=days,
                            )
    # 计算所有指标
    res = cal.cal_all_metrics(['MATRIX-9'])
    print(res)

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
