# -*- encoding: utf-8 -*-
"""
@File: metrics_cal.py
@Modify Time: 2025/4/9 08:37       
@Author: Kevin-Chen
@Descriptions: 金融指标计算器
"""
import time
import warnings
import traceback
import numpy as np
import pandas as pd
from functools import wraps
from numba import prange, float64, njit
from scipy.stats import norm, kurtosis, skew
from MetricsFactory.metrics_cal_config import return_ann_factor, risk_ann_factor, log_ann_return, log_daily_return

warnings.filterwarnings("ignore")
pd.set_option('display.width', 1000)  # 表格不分段显示
pd.set_option('display.max_columns', 1000)  # 显示字段的数量


# 装饰器函数，用于缓存被装饰函数的计算结果
def cache_metric(func):
    """
    装饰器函数，用于缓存被装饰函数的计算结果。

    该装饰器会将被装饰函数的计算结果缓存到 `self.res_dict` 中，以避免重复计算。
    如果缓存中已经存在该函数的计算结果，则直接返回缓存中的值。

    Args:
        func (function): 需要被装饰的函数，通常是一个计算指标的函数。

    Returns:
        function: 返回一个包装函数 `wrapper`，该函数会执行缓存逻辑。
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        """
        包装函数，用于执行缓存逻辑。

        该函数会从被装饰函数的名称中提取指标名称，并检查该指标是否已经存在于 `self.res_dict` 中。
        如果存在，则直接返回缓存中的值；如果不存在，则调用被装饰函数计算结果，并将结果缓存到 `self.res_dict` 中。

        Args:
            self: 类的实例对象。
            *args: 传递给被装饰函数的位置参数。
            **kwargs: 传递给被装饰函数的关键字参数。

        Returns:
            Any: 返回被装饰函数的计算结果或缓存中的值。
        """
        # 判断是否已经传入指标名称了, 例如: cal_max_draw_down_for_all() 和 cal_k_ratio_all() 函数
        if 'metric_name' in kwargs:
            metric_name = kwargs["metric_name"]
        else:
            # 从函数名中提取指标名称，去掉前缀 "cal_"
            metric_name = func.__name__.replace("cal_", "")

        # 如果指标已经存在于缓存中，则直接返回缓存中的值
        if metric_name in self.res_dict:
            return self.res_dict[metric_name]

        # VaR / CVaR 指标特殊处理
        if metric_name.startswith('VaR') or metric_name.startswith('CVaR'):
            metric_name = metric_name + '-' + str(int(kwargs["confidence_level"] * 100))
        # 分位数指标特殊处理
        elif metric_name.startswith('Percentile') or metric_name.startswith('TailRatio'):
            metric_name = metric_name + '-' + str(int(kwargs["tile"]))
        # 交叉乘积比率指标特殊处理
        elif metric_name.startswith('CrossProductRatio'):
            metric_name = metric_name + '-' + str(int(kwargs["days"]))

        # 调用被装饰函数计算结果，并将结果缓存到 `self.res_dict` 中
        result = func(self, *args, **kwargs)
        self.res_dict[metric_name] = result

        return result

    return wrapper


@njit
def calculate_hurst_exponent(portfolio_return):
    """
    计算hurst指数，用于判断时间序列的长期趋势是呈现自相似性还是anti-self-similarity。

    参数:
    portfolio_return: 一维ndarray，表示投资组合的回报序列。

    返回值:
    float，hurst指数。如果输入数据长度不足或含有过多nan值，则返回np.nan。
    """
    if len(portfolio_return) < 64:
        return np.nan

    # 清除含有 nan 的数据
    clean_data = portfolio_return[~np.isnan(portfolio_return)]
    if len(clean_data) < 64:
        return np.nan

    average_r_s = np.empty(6, dtype=np.float64)  # 用于存储不同尺度下r/s统计量的平均值
    size_list = np.empty(6, dtype=np.float64)  # 用于存储不同尺度下的子序列长度

    # 在不同尺度上计算r/s统计量
    for i in range(6):
        m = 2 ** i
        size = len(clean_data) // m
        size_list[i] = size
        r_s = np.empty(m, dtype=np.float64)

        for j in range(m):
            segment = clean_data[j * size:(j + 1) * size]
            s = np.std(segment)
            if s == 0:
                s = 0.00001
            deviation = segment - np.mean(segment)
            # 计算每个子序列的r/s统计量
            r_s[j] = (np.max(deviation) - np.min(deviation)) / s

        average_r_s[i] = np.mean(r_s)  # 计算该尺度下的平均r/s值

    log10_size_list = np.log10(size_list)
    log10_average_r_s = np.log10(average_r_s)
    # 手动计算线性回归，以确定hurst指数
    sxx = np.sum((log10_size_list - np.mean(log10_size_list)) ** 2)
    if sxx == 0:
        sxx = 0.00001
    sxy = np.sum((log10_size_list - np.mean(log10_size_list)) * (log10_average_r_s - np.mean(log10_average_r_s)))
    slope = sxy / sxx  # 计算线性回归的斜率，即hurst指数
    return slope


@njit(float64[:](float64[:, :]), parallel=True)
def compute_hurst_for_all_funds(funds_return):
    """
    计算所有基金的hurst指数

    参数:
        funds_return: 一个二维数组，代表所有基金的回报率，其中每一列是一个基金的所有回报率。

    返回值:
        一个一维数组，包含所有基金的hurst指数。
    """
    # 计算二维数组的列数
    n_cols = funds_return.shape[1]
    # 预分配一个一维数组，用于存储所有基金的hurst指数
    the_res = np.empty(n_cols, dtype=np.float64)

    # 并行计算每只基金的hurst指数
    for j in prange(n_cols):
        the_res[j] = calculate_hurst_exponent(funds_return[:, j])

    return the_res


@njit(float64[:, :](float64[:, :]), parallel=True)
def compute_k_ratios_combined(return_array):
    """
    输入:
        return_array: ndarray[float64] of shape (n_days, n_funds)
            每日对数收益率，允许存在 np.nan

    返回:
        result: ndarray[float64] of shape (n_funds, 2)
            第0列: 累计收益率的斜率 (slope)
            第1列: K-Ratio = slope / 标准误差
    """
    n_days, n_funds = return_array.shape
    result = np.full((n_funds, 2), np.nan)

    for i in prange(n_funds):
        r = return_array[:, i]
        t = np.empty(n_days, dtype=np.float64)
        y = np.empty(n_days, dtype=np.float64)

        cum_sum = 0.0
        idx = 0
        for j in range(n_days):
            if not np.isnan(r[j]):
                cum_sum += r[j]
                t[idx] = j
                y[idx] = cum_sum
                idx += 1

        if idx < 2:
            continue

        # 计算 slope 和 stderr
        t_valid = t[:idx]
        y_valid = y[:idx]

        t_mean = np.mean(t_valid)
        y_mean = np.mean(y_valid)
        cov = np.sum((t_valid - t_mean) * (y_valid - y_mean))
        var = np.sum((t_valid - t_mean) ** 2)

        if var == 0:
            continue

        slope = cov / var
        stderr = np.sqrt(np.sum((y_valid - (slope * t_valid + y_mean - slope * t_mean)) ** 2)
                         / (idx - 2)) / np.sqrt(var)

        result[i, 0] = slope
        result[i, 1] = slope / stderr if stderr != 0 else np.nan

    return result


@njit(float64[:, :](float64[:, :]), parallel=True)
def compute_net_equity_metrics(price_array):
    n_days, n_funds = price_array.shape
    result = np.full((n_funds, 2), np.nan, dtype=np.float64)  # [slope, r_squared]

    for i in prange(n_funds):
        y = price_array[:, i]
        valid_count = 0
        for j in range(n_days):
            if not np.isnan(y[j]):
                valid_count += 1

        if valid_count < 2:
            continue

        # 拿到有效点的索引和值
        t_valid = np.empty(valid_count, dtype=np.float64)
        y_valid = np.empty(valid_count, dtype=np.float64)

        idx = 0
        for j in range(n_days):
            if not np.isnan(y[j]):
                t_valid[idx] = j
                y_valid[idx] = np.log(y[j])  # log 净值
                idx += 1

        # 手动计算斜率
        t_mean = np.mean(t_valid)
        y_mean = np.mean(y_valid)

        t_centered = t_valid - t_mean
        y_centered = y_valid - y_mean

        numerator = np.sum(t_centered * y_centered)
        denominator = np.sum(t_centered ** 2)

        if denominator != 0:
            slope = numerator / denominator
            y_pred = slope * t_centered + y_mean
            ss_res = np.sum((y_valid - y_pred) ** 2)
            ss_tot = np.sum(y_centered ** 2)

            r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
            result[i, 0] = slope
            result[i, 1] = r_squared

    return result


# 金融指标计算器类
class CalMetrics:
    """
    用于计算和存储基金业绩评估指标的类。

    根据给定的基金代码、对数收益率数组、收盘价数组、时区代码、自然日数量和截止日期，
    以及是否将收益率转换为累计收益率的选项，初始化基金业绩评估指标计算。

    参数:
    - fund_codes (list / numpy.ndarray): 基金代码列表。
    - log_return_array (numpy.ndarray): 基金的对数收益率数组。
    - close_price_array (numpy.ndarray): 基金的收盘价数组。
    - time_zone_code (str): 时区代码。
    - nature_days_in_p (int): 自然日数量。
    - end_date (str): 结束日期。
    - min_data_required (int): 最少数据要求条数。
    - trans_to_cumulative_return (bool, optional): 是否将收益率转换为累计收益率，默认为False。
    """

    def __init__(self, fund_codes,
                 log_return_array,
                 close_price_array,
                 time_zone_code,
                 nature_days_in_p,
                 end_date,
                 min_data_required=5,
                 trans_to_cumulative_return=False):
        # 初始化基金代码列表
        self.fund_codes = fund_codes
        # 移除对数收益率数组中的全NaN行，并存储为返回数组
        self.return_array = log_return_array[~np.all(np.isnan(log_return_array), axis=1)]
        # 移除收盘价数组中的全NaN行，并存储为价格数组
        self.price_array = close_price_array[~np.all(np.isnan(close_price_array), axis=1)]
        # 存储时区代码
        self.time_zone_code = time_zone_code
        # 存储自然日数量
        self.nature_days = nature_days_in_p
        # 存储是否将收益率转换为累计收益率的选项
        self.cum_rtn = trans_to_cumulative_return
        # 存储结束日期
        self.end_date = end_date
        # 最少需要的数据量
        self.min_data_required = min_data_required
        # 计算并存储返回数组的行数和列数，分别代表天数和基金数量
        self.n_days, self.n_funds = self.return_array.shape
        # 初始化结果字典，用于存储后续计算的业绩评估指标
        self.res_dict = dict()

    @cache_metric  # 累积收益率
    def cal_TotalReturn(self, **kwargs):
        if not self.cum_rtn:  # 判断累计收益率的计算逻辑
            return np.nansum(self.return_array, axis=0)  # 计算对数收益率的累计值
        else:
            np.exp(np.nansum(self.return_array, axis=0)) - 1  # 计算普通收益率的累计值

    @cache_metric  # 年化收益率
    def cal_AnnualizedReturn(self, **kwargs):
        # 计算累计收益率
        total_rtn = self.cal_TotalReturn()

        if not self.cum_rtn:  # 判断累计收益率的计算逻辑
            return (total_rtn / self.nature_days) * return_ann_factor  # 使用对数收益率
        else:
            return (total_rtn + 1) ** (return_ann_factor / self.nature_days) - 1  # 使用简单收益率

    @cache_metric  # 每日回报率的平均值
    def cal_AverageDailyReturn(self, **kwargs):
        return np.nanmean(self.return_array, axis=0)

    @cache_metric  # 平均正收益率
    def cal_AvgPositiveReturn(self, **kwargs):
        return np.nanmean(np.where(self.return_array > 0, self.return_array, 0), axis=0)

    @cache_metric  # 平均负收益率
    def cal_AvgNegativeReturn(self, **kwargs):
        return np.nanmean(np.where(self.return_array < 0, self.return_array, 0), axis=0)

    @cache_metric  # 平均盈亏比
    def cal_AvgReturnRatio(self, **kwargs):
        ratio = self.cal_AvgPositiveReturn() / self.cal_AvgNegativeReturn()
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 总累计盈利 = 所有正收益的总和
    def cal_TotalPositiveReturn(self, **kwargs):
        return np.nansum(np.where(self.return_array > 0, self.return_array, 0), axis=0)

    @cache_metric  # 总累计亏损 = 所有负收益的总和
    def cal_TotalNegativeReturn(self, **kwargs):
        return np.nansum(np.where(self.return_array < 0, self.return_array, 0), axis=0)

    @cache_metric  # 盈利总和 / 亏损总和
    def cal_TotalReturnRatio(self, **kwargs):
        ratio = self.cal_TotalPositiveReturn() / np.abs(self.cal_TotalNegativeReturn())
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 每日回报率的中位数
    def cal_MedianDailyReturn(self, **kwargs):
        return np.nanmedian(self.return_array, axis=0)

    @cache_metric  # 每日回报率波动率
    def cal_Volatility(self, **kwargs):
        return np.nanstd(self.return_array, axis=0, ddof=1)

    @cache_metric  # 年化波动率
    def cal_AnnualizedVolatility(self, **kwargs):
        return self.cal_Volatility() * np.sqrt(risk_ann_factor)

    @cache_metric  # 收益率范围
    def cal_ReturnRange(self, **kwargs):
        return self.cal_MaxGain() - self.cal_MaxLoss()

    @cache_metric
    def cal_RescaledRange(self, **kwargs):
        return self.cal_ReturnRange() / self.cal_Volatility()

    @cache_metric  # 最大单日收益
    def cal_MaxGain(self, **kwargs):
        return np.nanmax(self.return_array, axis=0)

    @cache_metric  # 最大单日亏损
    def cal_MaxLoss(self, **kwargs):
        return np.nanmin(self.return_array, axis=0)

    @cache_metric  # 计算所有最大回撤相关的指标
    def cal_max_draw_down_for_all(self, metric_name: str, **kwargs) -> float:
        """
        计算所有基金的最大回撤及相关衍生指标

        :param metric_name: 需要返回的指标名称（如"MaxDrawDown", "MaxDrawDownDays" 等）
        :param kwargs: 其他可选参数（当前未被使用）
        :return: 指定指标对应的结果值
        """

        ''' (1) 计算 最大回撤：通过逐日跟踪历史最高净值，计算当前回撤率并取最小值 '''
        # 计算历史最大净值（逐日累计最大值）
        max_price = np.fmax.accumulate(np.where(np.isnan(self.price_array), -np.inf, self.price_array), axis=0)
        # 计算回撤序列（当前价格 / 历史最大值 - 1）
        draw_down = self.price_array / max_price - 1
        # 处理NaN值（无价格数据时回撤为0）
        draw_down = np.where(np.isnan(draw_down), 0, draw_down)
        # 取每只基金的最大回撤（最小回撤率即最大跌幅）
        max_dd = np.nanmin(draw_down, axis=0)

        ''' (2) 确定最大回撤结束位置：通过反转数组找到最后出现的最小值索引 '''
        # 反转数组后寻找各列最小值首次出现位置（对应原始数组的最后出现位置）
        reversed_argmin = np.argmin(draw_down[::-1, :], axis=0)
        # 转换为原始数组坐标系下的索引
        max_dd_idx = (self.n_days - 1) - reversed_argmin

        ''' (3) 计算最大回撤持续天数：通过构造有效掩码定位回撤起始与结束点 '''
        # 构建有效回撤区间掩码（满足回撤期间且未超过结束索引）
        mask_valid = (draw_down >= 0) & (np.arange(self.n_days)[:, None] <= max_dd_idx[None, :])
        # 反转掩码后寻找峰值索引（最后一次满足条件的位置）
        reversed_mask_valid = mask_valid[::-1, :]
        reversed_idx_peak = np.argmax(reversed_mask_valid, axis=0)
        # 转换为原始坐标系并计算持续天数
        peak_idx = (self.n_days - 1) - reversed_idx_peak
        self.res_dict['MaxDrawDownDays'] = max_dd_idx - peak_idx

        ''' (4) 计算回撤斜率：最大回撤绝对值与持续天数的比率 '''
        dd_slope = np.abs(max_dd) / self.res_dict['MaxDrawDownDays']
        self.res_dict['DrawDownSlope'] = np.where(np.isnan(dd_slope), 0, dd_slope)

        ''' (5) 计算收益回撤比及年化版本：包含普通和年化两种计算方式 '''
        # 转换对数累计收益率为普通收益率
        total_return = np.exp(self.cal_TotalReturn()) - 1
        # 计算收益回撤比（普通和年化版本）
        r_dd = total_return / np.abs(max_dd)
        ann_r_dd = (((total_return + 1) ** (return_ann_factor / self.nature_days) - 1) / np.abs(max_dd))
        # 处理无穷大情况
        self.res_dict['ReturnDrawDownRatio'] = np.where(r_dd == -np.inf, 0, r_dd)
        self.res_dict['AnnReturnDrawDownRatio'] = np.where(ann_r_dd == -np.inf, 0, ann_r_dd)
        self.res_dict["MaxDrawDown"] = max_dd

        ''' (6) 计算溃疡指数：衡量回撤波动性的指标（回撤平方均值的平方根） '''
        mean_dd_sq = np.mean(draw_down ** 2, axis=0)
        self.res_dict["UlcerIndex"] = np.sqrt(mean_dd_sq)

        ''' (7) 计算马丁比率, 年化收益率 / 溃疡指数 '''
        self.res_dict["MartinRatio"] = self.cal_AnnualizedReturn() / self.res_dict["UlcerIndex"]

        ''' (8) 返回指定指标结果 '''
        return self.res_dict[metric_name]

    @cache_metric  # 年化夏普比率
    def cal_AnnualizedSharpeRatio(self, **kwargs):
        ratio = (self.cal_AnnualizedReturn() - log_ann_return) / self.cal_AnnualizedVolatility()
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 夏普比率
    def cal_SharpeRatio(self, **kwargs):
        ratio = (self.cal_TotalReturn() - (log_daily_return * self.nature_days)) / self.cal_Volatility()
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 收益率波动率比
    def cal_ReturnVolatilityRatio(self, **kwargs):
        ratio = self.cal_TotalReturn() / self.cal_Volatility()
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 下行波动率
    def cal_DownsideVolatility(self, mar=log_daily_return, **kwargs):
        # 计算低于 MAR 的差值，否则为 0
        down_diff = np.where(self.return_array < mar, self.return_array - mar, 0.0)
        # 计算方差（1个自由度）
        downside_var = np.nanvar(down_diff, axis=0, ddof=1)
        # 返回标准差（即下行波动率）
        ratio = np.sqrt(downside_var)
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 上行波动率
    def cal_UpsideVolatility(self, mar=0, **kwargs):
        # 对于超过 MAR 的收益部分，计算偏差；其余为 0
        up_diff = np.where(self.return_array > mar, self.return_array - mar, 0.0)
        # 计算上行方差（忽略 nan，支持自由度）
        upside_var = np.nanvar(up_diff, axis=0, ddof=1)
        # 返回上行标准差
        ratio = np.sqrt(upside_var)
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 波动率偏度
    def cal_VolatilitySkew(self, **kwargs):
        ratio = (self.cal_UpsideVolatility() - self.cal_DownsideVolatility()) / self.cal_Volatility()
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 波动率比率
    def cal_VolatilityRatio(self, **kwargs):
        ratio = self.cal_UpsideVolatility() / self.cal_DownsideVolatility()
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 索提诺比率
    def cal_SortinoRatio(self, **kwargs):
        ratio = (self.cal_TotalReturn() - (log_daily_return * self.nature_days)) / self.cal_DownsideVolatility()
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 收益趋势一致性
    def cal_GainConsistency(self, **kwargs):
        # 筛选正收益
        gain_only = np.where(self.return_array > 0, self.return_array, np.nan)

        # 计算均值和标准差（忽略 nan）
        gain_mean = np.nanmean(gain_only, axis=0)  # shape = (N,)
        gain_std = np.nanstd(gain_only, axis=0)  # shape = (N,)

        # 避免除以 0 或 nan
        gain_consistency = gain_std / gain_mean

        return np.where(np.isnan(gain_consistency), 0, gain_consistency)

    @cache_metric  # 损失趋势一致性
    def cal_LossConsistency(self, **kwargs):
        # 筛选负收益
        loss_only = np.where(self.return_array < 0, self.return_array, np.nan)

        # 计算负收益的平均值和标准差
        loss_mean = np.nanmean(loss_only, axis=0)  # 为负值
        loss_std = np.nanstd(loss_only, axis=0)

        # 用 abs(loss_mean) 避免除以负值
        loss_consistency = loss_std / np.abs(loss_mean)

        return np.where(np.isnan(loss_consistency), 0, loss_consistency)

    @cache_metric  # 计算盈利率
    def cal_WinningRatio(self, **kwargs):
        # 计算胜率
        return (np.sum((self.return_array > 0) & ~np.isnan(self.return_array), axis=0)
                / np.sum(~np.isnan(self.return_array), axis=0))

    @cache_metric  # 计算亏损率
    def cal_LosingRatio(self, **kwargs):
        # 计算亏损率
        return (np.sum((self.return_array < 0) & ~np.isnan(self.return_array), axis=0)
                / np.sum(~np.isnan(self.return_array), axis=0))

    @cache_metric  # 平均绝对偏差
    def cal_MeanAbsoluteDeviation(self, **kwargs):
        return np.nanmean(np.abs(self.return_array - self.cal_AverageDailyReturn()), axis=0)

    @cache_metric  # 收益率偏度
    def cal_ReturnSkewness(self, **kwargs):
        return skew(self.return_array, axis=0, bias=False, nan_policy='omit')

    @cache_metric  # 收益率峰度
    def cal_ReturnKurtosis(self, excess=True, **kwargs):
        excess_kurtosis = kurtosis(self.return_array, axis=0, bias=False, nan_policy='omit')  # 返回的是超峰度
        if excess:
            return excess_kurtosis  # 正态分布的超峰度是 0
        else:
            return excess_kurtosis + 3  # 正态分布的总峰度是 3

    @cache_metric  # 计算在险价值 (参数法)
    def cal_VaR(self, confidence_level=0.99, **kwargs):
        mean_return = self.cal_AverageDailyReturn()
        std_return = self.cal_Volatility()
        alpha = 1.0 - confidence_level
        z_left = norm.ppf(alpha)
        var = - (mean_return + z_left * std_return)
        return np.maximum(var, 0.0)  # VaR 是损失，不小于0

    @cache_metric  # 基于 VaR 计算夏普比率
    def cal_VaRSharpe(self, confidence_level=0.99, **kwargs):
        var_name = 'VaR' + '-' + str(int(confidence_level * 100))
        ratio = (self.cal_TotalReturn() - (log_daily_return * self.nature_days)) / self.cal_metric(var_name)
        # 将 inf 转为 0
        ratio = np.where(np.isinf(ratio), 0, ratio)
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 计算 Cornish-Fisher修正后的在险价值 (参数法)
    def cal_VaRModified(self, confidence_level=0.99, **kwargs):
        mu = self.cal_AverageDailyReturn()
        sigma = self.cal_Volatility()
        skewness = self.cal_ReturnSkewness()
        k_excess = self.cal_ReturnKurtosis()

        alpha = 1.0 - confidence_level
        z = norm.ppf(alpha)

        z_mod = (z
                 + (skewness / 6.0) * (z ** 2 - 1.0)
                 + (k_excess / 24.0) * (z ** 3 - 3.0 * z)
                 - ((skewness ** 2) / 36.0) * (2.0 * z ** 3 - 5.0 * z)
                 )

        var_modified = - (mu + z_mod * sigma)
        return np.maximum(var_modified, 0.0)

    @cache_metric  # 基于 修正后的VaR 计算夏普比率
    def cal_VaRModifiedSharpe(self, confidence_level=0.99, **kwargs):
        var_name = 'VaRModified' + '-' + str(int(confidence_level * 100))
        ratio = (self.cal_TotalReturn() - (log_daily_return * self.nature_days)) / self.cal_metric(var_name)
        # 将 inf 转为 0
        ratio = np.where(np.isinf(ratio), 0, ratio)
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 计算期望损失ES (参数法)
    def cal_CVaR(self, confidence_level=0.99, **kwargs):
        mu = self.cal_AverageDailyReturn()
        sigma = self.cal_Volatility()
        alpha = 1.0 - confidence_level
        z = norm.ppf(alpha)
        phi_z = norm.pdf(z)
        cvar = - (mu - (sigma * phi_z) / alpha)
        return np.maximum(cvar, 0.0)  # CVaR 是损失，不应为负

    @cache_metric  # 基于 CVaR 计算夏普比率
    def cal_CVaRSharpe(self, confidence_level=0.99, **kwargs):
        cvar_name = 'CVaR' + '-' + str(int(confidence_level * 100))
        ratio = (self.cal_TotalReturn() - (log_daily_return * self.nature_days)) / self.cal_metric(cvar_name)
        # 将 inf 转为 0
        ratio = np.where(np.isinf(ratio), 0, ratio)
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 计算 Cornish-Fisher修正后的期望损失ES (参数法)
    def cal_CVaRModified(self, confidence_level=0.99, **kwargs):
        mu = self.cal_AverageDailyReturn()
        sigma = self.cal_Volatility()
        alpha = 1.0 - confidence_level
        skewness = self.cal_ReturnSkewness()
        k_excess = self.cal_ReturnKurtosis()

        alpha = 1.0 - confidence_level
        z = norm.ppf(alpha)

        z_mod = (z
                 + (skewness / 6.0) * (z ** 2 - 1.0)
                 + (k_excess / 24.0) * (z ** 3 - 3.0 * z)
                 - ((skewness ** 2) / 36.0) * (2.0 * z ** 3 - 5.0 * z))

        phi_z_mod = norm.pdf(z_mod)
        cvar_mod = - (mu - (sigma * phi_z_mod) / alpha)
        return np.maximum(cvar_mod, 0.0)

    @cache_metric  # 基于 修正后的CVaR 计算夏普比率
    def cal_CVaRModifiedSharpe(self, confidence_level=0.99, **kwargs):
        cvar_name = 'CVaRModified' + '-' + str(int(confidence_level * 100))
        ratio = (self.cal_TotalReturn() - (log_daily_return * self.nature_days)) / self.cal_metric(cvar_name)
        # 将 inf 转为 0
        ratio = np.where(np.isinf(ratio), 0, ratio)
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 计算收益率的分位数 (考虑所有收益率)
    def cal_Percentile(self, tile=5, **kwargs):
        perc_tile = np.nanpercentile(self.return_array, tile, axis=0)
        return perc_tile

    @cache_metric  # 计算收益率的分位数 (仅仅考虑正收益率)
    def cal_PercentileWin(self, tile=95, **kwargs):
        # 替换负值为 NaN，只保留正收益
        positive_only = np.where(self.return_array > 0, self.return_array, np.nan)
        # 计算分位（即最差的前5%盈利）
        perc_tile = np.nanpercentile(positive_only, tile, axis=0)
        # 任何负值或 nan 都设为0
        return np.where(np.isnan(perc_tile), 0.0, perc_tile)

    @cache_metric  # 计算收益率的分位数 (仅仅考虑负收益率, 并且负收益取绝对值)
    def cal_PercentileLoss(self, tile=95, **kwargs):
        # 替换正值为 NaN，只保留负收益
        negative_only = np.where(self.return_array < 0, self.return_array, np.nan)
        # 负收益率取绝对值
        negative_only = np.abs(negative_only)
        # 计算分位（即最严重的亏损）
        perc_tile = np.nanpercentile(negative_only, tile, axis=0)
        # 任何正值或 nan 都设为0
        return np.where(np.isnan(perc_tile), 0.0, perc_tile)

    @cache_metric  # 尾部比率, 极端正收益与极端负收益的比值
    def cal_TailRatio(self, tile=95, **kwargs):
        win_name = 'PercentileWin' + '-' + str(int(tile))
        loss_name = 'PercentileLoss' + '-' + str(int(tile))
        # 计算盈利和亏损的分位数
        win = self.cal_metric(win_name)
        loss = self.cal_metric(loss_name)

        # 设置极小值防止除以 0 或非常接近 0 导致爆炸
        eps = 1e-6
        safe_loss = np.where(np.abs(loss) < eps, eps, loss)

        return win / safe_loss

    @cache_metric  # 净值新高率 (净值创新高的天数 / 总非空天数)
    def cal_NewHighRatio(self, **kwargs):
        # 记录历史最大值（每列基金的滚动最高收盘价）
        rolling_max = np.maximum.accumulate(np.where(np.isnan(self.price_array), -np.inf, self.price_array), axis=0)

        # 是否创新高：当天价格 > 前一天的 rolling max（需移动一位）
        shifted_max = np.vstack([np.full((1, self.price_array.shape[1]), -np.inf), rolling_max[:-1]])
        is_new_high = (self.price_array > shifted_max) & (~np.isnan(self.price_array))

        # 统计创新高天数
        new_high_counts = np.sum(is_new_high, axis=0)

        # 总有效天数
        valid_counts = np.sum(~np.isnan(self.price_array), axis=0)

        # 新高率
        return new_high_counts / np.maximum(valid_counts, 1)

    # 工具函数: 计算 n 日收益率之和
    def tool_sum_return(self, n=2, **kwargs):
        """
        计算给定数组在不同时间段内的收益总和。

        该函数根据指定的时期数n，将return_array数组截断并分组，然后计算每个分组内的收益总和。
        如果一组内的所有值都是NaN，则该组的结果也为NaN。

        参数:
        - n: 指定每个时间段包含的天数，默认为2。
        - **kwargs: 允许接受任意额外的关键字参数，但在这个函数中不使用。

        返回:
        - summed: 一个二维数组，包含每个时间段的收益总和。
        """
        # 将n转换为整数类型
        n = int(n)
        # 获取原始数组
        arr = self.return_array
        # 获取数组的形状，n_days为天数，n_funds为基金数
        n_days, n_funds = arr.shape
        # 计算可以被n整除的部分的长度，去除尾部无法整除的部分
        usable_len = int((n_days // n) * n)
        # 截取可用部分的数组
        arr_trimmed = arr[:usable_len]
        # 将截取后的数组重塑为新的形状，以便后续处理
        arr_grouped = arr_trimmed.reshape(-1, n, n_funds)
        # 计算每组中是否所有元素都是NaN
        all_nan_mask = np.all(np.isnan(arr_grouped), axis=1)
        # 计算每组中非NaN元素的总和
        summed = np.nansum(arr_grouped, axis=1)
        # 将所有元素都是NaN的组的结果设置为NaN
        summed[all_nan_mask] = np.nan
        # 返回计算结果
        return summed

    @cache_metric  # 交叉乘积比率 = (WW * LL) / (WL * LW)
    def cal_CrossProductRatio(self, days=1, **kwargs):
        """
        计算交叉乘积比率（Cross Product Ratio）。

        参数:
        - days: int, 计算回报率所用的天数。默认为1天。
        - **kwargs: 允许接受的额外的关键字参数。

        返回:
        - ratio: numpy.ndarray, 形状为(n_funds,), 表示每个基金的交叉乘积比率。
                 若分母为0，则返回np.nan。
        """
        # 选择使用每日回报还是多日累积回报
        if days == 1:
            ra = self.return_array  # shape: (n_days, n_funds)
        else:
            ra = self.tool_sum_return(n=days)
        n_days, n_funds = ra.shape

        # 检查数据是否足够
        if n_days < self.min_data_required:
            return np.nan * np.ones(n_funds)

        # ------------------------------------------------
        # 1) 将各列的非 NaN 数据前移, NaN 后置
        # ------------------------------------------------
        # row_indices: 若是 NaN => n_days, 否则 => 本来的行号
        # 用于对每列做排序, 把所有 NaN (行号= n_days) 排到末尾
        row_indices = np.where(
            np.isnan(ra),
            n_days,  # 给 NaN 一个很大的行索引 => 排到最后
            np.arange(n_days)[:, None]  # 各行的真实索引
        )  # shape=(n_days, n_funds)

        # 对列方向做 argsort, 得到 "把哪几行排在前面"
        sorted_indices = np.argsort(row_indices, axis=0)  # shape=(n_days, n_funds)

        # 利用花式索引, 每列都按 sorted_indices 的顺序取数据
        # => new_ra: 每列上方全是非 NaN, 下方是 NaN
        new_return_array = ra[sorted_indices, np.arange(n_funds)]  # shape=(n_days, n_funds)

        # ------------------------------------------------
        # 2) 计算 W/L 标记, 并统计 WW/WL/LW/LL 次数(矢量化)
        # ------------------------------------------------
        # 定义 sign 矩阵: True表示W(>=0), False表示L(<0).
        # 对 NaN 元素, (NaN >= 0) => False, 但我们会用 mask 排除
        sign_mat = (new_return_array >= 0)

        # 再定义有效性 mask: 是否非 NaN
        valid_mask = ~np.isnan(new_return_array)

        # "昨日"与"今天"的组合 => 我们只比较相邻日 (行 i-1, i)
        yest_sign = sign_mat[:-1, :]  # => yest_sign = sign_mat[:-1, :], day_sign = sign_mat[1:, :]
        day_sign = sign_mat[1:, :]  # => valid_2day = valid_mask[:-1, :] & valid_mask[1:, :]
        valid_2day = valid_mask[:-1, :] & valid_mask[1:, :]  # 结果形状= (n_days-1, n_funds)

        # 组合编码 pair_code: 2*yest_sign + day_sign => {0,1,2,3}, 含义: 0=LL, 1=LW, 2=WL, 3=WW
        pair_code = 2 * yest_sign.astype(int) + day_sign.astype(int)  # shape=(n_days-1, n_funds)

        # 准备列索引
        col_index = np.arange(n_funds).reshape(1, -1)  # shape=(1, n_funds)
        col_index = np.broadcast_to(col_index, (n_days - 1, n_funds))  # shape=(n_days-1, n_funds)

        # 把 pair_code 和 col_index 扁平化, 并用 valid_2day 过滤
        valid_flat = valid_2day.ravel()  # 布尔数组
        code_flat = pair_code.ravel()[valid_flat]
        col_flat = col_index.ravel()[valid_flat]

        # 编成一个“单一数字” comb = code + 4*col => 对每列统计 code in {0,1,2,3}
        comb = code_flat + 4 * col_flat

        # bincount => hist长度 >= 4*n_funds
        hist = np.bincount(comb, minlength=4 * n_funds)
        hist_2d = hist.reshape(n_funds, 4)  # shape=(n_funds,4)

        # hist2D[col, 0]=LL, [1]=LW, [2]=WL, [3]=WW
        ll = hist_2d[:, 0]
        lw = hist_2d[:, 1]
        wl = hist_2d[:, 2]
        ww = hist_2d[:, 3]

        # 计算 ratio = (WW*LL)/(WL*LW), 若 WL*LW==0 => np.nan
        numerator = ww * ll
        denominator = wl * lw
        ratio = np.where(denominator == 0, np.nan, numerator / denominator)

        return ratio  # shape=(n_funds,)

    @cache_metric  # 近似 Hurst 指数
    def cal_HurstExponent(self, **kwargs):
        return compute_hurst_for_all_funds(self.return_array)

    @cache_metric  # 计算收益分布积分
    def cal_ReturnDistributionIntegral(self, **kwargs):
        """
        计算收益分布积分（Return Distribution Integral，RDI）。

        该函数用于量化基金收益超过最小接受回报（MAR）的部分的期望值。
        它通过计算所有超过MAR的每日收益与对应概率的乘积之和来实现。

        参数:
        - kwargs: 允许函数接受任意额外的关键字参数，这里未使用。

        返回:
        - rdi: 基金的收益分布积分值，表示超过MAR的收益的期望值。
        """

        # 排除全是 NaN 的基金列
        mask = ~np.isnan(self.return_array)
        n_days = np.sum(mask, axis=0)

        # 计算每个样本的概率
        p = 1.0 / n_days  # shape=(n_funds,)

        # 按行排序
        sorted_returns = np.sort(self.return_array, axis=0)

        # 构造权重矩阵 p_i（每一列都是常数向量）
        p_matrix = np.broadcast_to(p, sorted_returns.shape)

        # 选取大于 mar 的部分
        gain_mask = sorted_returns > log_daily_return
        gains = (sorted_returns - log_daily_return) * p_matrix
        gains[~gain_mask] = 0.0

        # 计算 RDI
        rdi = np.nansum(gains, axis=0)
        return rdi

    @cache_metric  # 计算Omega比率
    def cal_OmegaRatio(self, **kwargs):
        """
        计算Omega比率。

        Omega比率是通过计算收益超过某个阈值的概率加权平均值（上尾部分）与损失低于该阈值的概率加权平均值（下尾部分）的比值来衡量投资表现的指标。
        此函数主要用于评估给定收益序列的Omega比率，通过区分收益和损失部分来提供更细致的风险调整后收益分析。

        参数:
        - **kwargs: 允许函数接受可变关键字参数，但在这个上下文中未直接使用。

        返回:
        - omega: 计算得到的Omega比率。
        """
        # 创建一个掩码，用于排除不是数字的值
        mask = ~np.isnan(self.return_array)
        # 计算有效（非NaN）交易日的数量
        n_days = np.sum(mask, axis=0)
        # 每个样本的概率，即1除以有效交易日数
        p = 1.0 / n_days
        # 为避免除以零错误，引入一个极小值
        eps = 1e-8

        # 对收益进行排序，以便后续计算上尾和下尾
        sorted_returns = np.sort(self.return_array, axis=0)
        # 将概率p扩展成与sorted_returns相同形状的矩阵
        p_matrix = np.broadcast_to(p, sorted_returns.shape)

        # 计算上尾部分，即超过某个阈值（log_daily_return）的收益
        gain_mask = sorted_returns > log_daily_return
        # 对超过阈值的收益进行概率加权
        gains = (sorted_returns - log_daily_return) * p_matrix
        # 对不超过阈值的部分设为0，仅保留上尾部分的收益
        gains[~gain_mask] = 0.0
        # 计算上尾部分的总和，即收益部分
        rdi = np.nansum(gains, axis=0)

        # 计算下尾部分，即低于某个阈值（log_daily_return）的损失
        loss_mask = sorted_returns < log_daily_return
        # 对低于阈值的损失进行概率加权
        losses = (log_daily_return - sorted_returns) * p_matrix
        # 对超过阈值的部分设为0，仅保留下尾部分的损失
        losses[~loss_mask] = 0.0
        # 计算下尾部分的总和，即损失部分
        ldi = np.nansum(losses, axis=0)

        # 计算Omega比率，使用eps避免除以零
        omega = rdi / (ldi + eps)
        # 返回计算得到的Omega比率
        return omega

    @cache_metric  # 计算K比率的综合代码
    def cal_k_ratio_all(self, metric_name='KRatio', **kwargs):
        # 计算
        result = compute_k_ratios_combined(self.return_array)
        # 提取斜率和 K 比率
        slope_array = result[:, 0]  # 第一列为累计收益率斜率
        k_ratio_array = result[:, 1]  # 第二列为 K 比率
        self.res_dict["ReturnSlope"] = slope_array
        self.res_dict["KRatio"] = k_ratio_array
        # 返回指定指标结果
        return self.res_dict[metric_name]

    @cache_metric  # 索提诺偏度
    def cal_SortinoSkewness(self, mar=log_daily_return, **kwargs):
        """
        计算索提诺偏度（Sortino Skewness）。

        索提诺偏度是衡量投资组合收益分布的不对称性指标，专注于负收益（下行风险）。

        参数:
        - mar: 最低可接受回报率，默认为日对数收益率。
        - **kwargs: 其他关键字参数。

        返回:
        - skewness: 索提诺偏度数组，每个基金对应一个偏度值。
        """
        # 识别低于最低可接受回报率（mar）的收益
        downside_mask = self.return_array < mar

        # 对负收益部分计算均值（忽略 NaN）
        downside = np.where(downside_mask, self.return_array, np.nan)  # 非负收益部分置为 NaN

        mean_down = np.nanmean(downside, axis=0)  # shape=(n_funds,)
        std_down = self.cal_DownsideVolatility(mar=mar)

        # 三阶中心矩 numerator: mean((x - mean)^3)
        num = np.nanmean((downside - mean_down[np.newaxis, :]) ** 3, axis=0)

        # denominator: std^3，避免除以0
        denom = std_down ** 3
        denom[denom == 0] = np.nan  # 避免除0警告

        # 计算索提诺偏度
        skewness = num / denom
        return skewness

    @cache_metric  # 净值增长斜率
    def cal_nv_slope_all(self, metric_name='NetEquitySlope', **kwargs):
        """
        计算并返回净值曲线的斜率和光滑度指标。

        该方法通过调用compute_net_equity_metrics函数来计算给定价格数组的净值曲线斜率（NetEquitySlope）
        和净值曲线光滑度（EquitySmoothness），并将这些结果存储在结果字典（res_dict）中。

        参数:
        - metric_name (str): 指定返回的指标名称，默认为'NetEquitySlope'。
        - **kwargs: 允许传递任意额外的关键字参数，这里未使用。

        返回:
        - 返回指定名称的指标结果，类型为列表或数组。
        """
        # 计算净值曲线的斜率和光滑度
        result = compute_net_equity_metrics(self.price_array)

        # 提取净值曲线斜率和光滑度
        net_equity_slope = result[:, 0]
        equity_smoothness = result[:, 1]

        # 将计算结果存储到结果字典中
        self.res_dict["NetEquitySlope"] = net_equity_slope
        self.res_dict["EquitySmoothness"] = equity_smoothness

        # 返回指定的指标结果
        return self.res_dict[metric_name]

    # 根据指标名 计算相应的指标值
    def cal_metric(self, metric_name, **kwargs):
        """
        根据指标名 计算相应的指标值。

        参数:
        metric_name (str): 指标名称，如 'TotalReturn'

        返回值:
        numpy.ndarray: 计算得到的指标值。
        """

        # 最大回撤指标的统一处理
        if metric_name in ['MaxDrawDown', 'MaxDrawDownDays', 'ReturnDrawDownRatio', 'MartinRatio',
                           'AnnReturnDrawDownRatio', 'DrawDownSlope', 'UlcerIndex']:
            return self.cal_max_draw_down_for_all(metric_name=metric_name)
        # 收益率斜率和 K 比率的统一处理
        elif metric_name in ['ReturnSlope', 'KRatio']:
            return self.cal_k_ratio_all(metric_name=metric_name)
        # 净值曲线的斜率和光滑度指标的统一处理
        elif metric_name in ['NetEquitySlope', 'EquitySmoothness']:
            return self.cal_nv_slope_all(metric_name=metric_name)
        # VaR 指标的处理
        elif metric_name.startswith('VaR') or metric_name.startswith('CVaR'):
            kwargs["confidence_level"] = float(metric_name.split("-")[1]) / 100
            metric_name = metric_name.split("-")[0]
        # 分位数指标的处理
        elif metric_name.startswith('Percentile') or metric_name.startswith('TailRatio'):
            kwargs["tile"] = float(metric_name.split("-")[1])
            metric_name = metric_name.split("-")[0]
        # 交叉积比率相关的处理
        elif metric_name.startswith('CrossProductRatio'):
            kwargs["days"] = float(metric_name.split("-")[1])
            metric_name = metric_name.split("-")[0]

        method_name = f'cal_{metric_name}'
        try:
            return getattr(self, method_name)(**kwargs)
        except Exception as e:
            print(f"Error when calling '{method_name}': {e}")
            print(traceback.format_exc())

    # 根据指标名列表 计算相应的指标值
    def cal_metric_main(self, metric_name_list, **kwargs):
        # 检查数据是否足够
        if self.n_days < self.min_data_required:
            return pd.DataFrame()

        # 计算各个指标值
        final_df = list()
        for metric_name in metric_name_list:
            valid_counts = np.sum(~np.isnan(self.return_array), axis=0)
            res_array = self.cal_metric(metric_name, **kwargs)
            # 如果没有计算出任何指标值，则跳过
            if res_array is None:
                continue
            else:
                res_array = res_array.astype(float)
            res_array[valid_counts < self.min_data_required] = np.nan

            final_df.append(pd.DataFrame({
                "ts_code": self.fund_codes,
                "metric_value": res_array,
                "metric_name": metric_name,
                "date": self.end_date
            }))
        if len(final_df) == 0:
            raise ValueError("没有计算出任何指标")

        # 整理指标结果并返回
        final_df = pd.concat(final_df, axis=0)
        final_df = final_df.pivot(index=["ts_code", "date"], columns="metric_name", values="metric_value").reset_index()
        final_df.columns.name = None

        # 指标代码中加入时间区间
        columns = final_df.columns.tolist()
        columns = [col + ':' + self.time_zone_code if col not in ['ts_code', 'date'] else col for col in columns]
        final_df.columns = columns
        return final_df


if __name__ == '__main__':
    the_close_price_array = pd.read_parquet('close_df.parquet')
    the_log_return_df = pd.read_parquet('log_return_df.parquet')

    the_close_price_array = the_close_price_array.resample('D').asfreq()
    the_log_return_df = the_log_return_df.resample('D').asfreq()
    funds_codes = the_log_return_df.columns.values

    the_close_price_array = the_close_price_array.values
    the_log_return_df = the_log_return_df.values

    # the_close_price_array = the_close_price_array[:, :3]
    # the_log_return_df = the_log_return_df[:, :3]

    c_m = CalMetrics(funds_codes, the_log_return_df, the_close_price_array,
                     '1m', 61, pd.to_datetime('2025-04-11'))

    m_list = ['NetEquitySlope', 'EquitySmoothness']
    rr = c_m.cal_metric_main(m_list)
    print(rr)
