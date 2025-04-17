# -*- encoding: utf-8 -*-
"""
@File: factor_factory.py
@Modify Time: 2025/4/8 19:25       
@Author: Kevin-Chen
@Descriptions: 因子加工厂
"""
import gc
import re
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from multiprocessing import Pool, cpu_count, shared_memory

from functools import partial
from MetricsFactory.period_metrics_cal import CalMetrics
from MetricsFactory.metrics_cal_config import period_list, create_period_metrics_map

pd.set_option('display.max_columns', 1000)  # 显示字段的数量
pd.set_option('display.width', 1000)  # 表格不分段显示


# 根据给定区间 period 和结束日期 last_day, 返回该区间的开始日期
def get_start_date(last_day, period):
    """
    根据给定区间 period 和结束日期 last_day, 返回该区间的开始日期(“包含”逻辑可以配合 > start_date, <= last_day使用).
    返回一个 pd.Timestamp (如需 date, 可自行调用 .date()).

    参数:
    last_day (str or datetime): 区间的结束日期，可以是字符串或 datetime 对象。
    period (str): 区间类型，支持以下格式 (全部是不包含开始日期的模式)：
        - 'mtd': 从上个月尾开始
        - 'qtd': 从上个季度末开始
        - 'ytd': 从去年年末开始
        - 'Nd': 过去N天
        - 'Nw': 过去N周
        - 'Nm': 过去N个月
        - 'Ny': 过去N年

    返回:
    pd.Timestamp: 区间的开始日期。
    """
    # 将 last_day 转换为 pd.Timestamp 并只保留日期部分
    ld = pd.to_datetime(last_day).normalize()

    # 根据 period 类型计算开始日期
    if period.lower() == 'mtd':
        # Month-To-Date: 从当月1号开始，但为了满足 > start_date 的过滤条件，start_date 需要比真正区间早1天
        real_start = ld.replace(day=1)
        start = real_start - pd.Timedelta(days=1)

    elif period.lower() == 'qtd':
        # Quarter-To-Date: 从当季度1号开始，同样为了满足过滤条件，start_date 需要比真正区间早1天
        current_month = ld.month
        quarter_start_month = ((current_month - 1) // 3) * 3 + 1
        real_start = ld.replace(month=quarter_start_month, day=1)
        start = real_start - pd.Timedelta(days=1)

    elif period.lower() == 'ytd':
        # Year-To-Date: 从当年1号开始，为了满足过滤条件，start_date 需要比真正区间早1天
        real_start = ld.replace(month=1, day=1)
        start = real_start - pd.Timedelta(days=1)

    elif period.endswith('d'):
        # 过去N天: 从 last_day 减去 N 天
        num_days = int(period[:-1])
        start = ld - pd.Timedelta(days=num_days)

    elif period.endswith('w'):
        # 过去N周: 从 last_day 减去 N 周
        num_weeks = int(period[:-1])
        start = ld - pd.Timedelta(weeks=num_weeks)

    elif period.endswith('m'):
        # 过去N个月: 从 last_day 减去 N 个月
        num_months = int(period[:-1])
        start = ld - relativedelta(months=num_months)

    elif period.endswith('y'):
        # 过去N年: 从 last_day 减去 N 年
        num_years = int(period[:-1])
        start = ld - relativedelta(years=num_years)

    else:
        # 如果 period 类型不支持，抛出异常
        raise ValueError(f"不支持的区间类型: {period}")

    return pd.to_datetime(start)  # 返回开始日期，如果需要 date 类型, 可以 return start.date()


# 在给定的自然日数组中找到指定日期范围的索引
def find_date_range_indices(nature_days_array, start_date, end_date):
    """
    在给定的自然日数组中找到指定日期范围的索引。

    本函数使用二分查找算法来确定start_date和end_date在自然日数组中的位置，
    从而可以有效地找到指定日期范围内的所有日期。

    参数:
    nature_days_array (numpy.ndarray): 包含自然日的numpy数组，必须是升序排列。
    start_date (datetime.datetime): 指定日期范围的开始日期。
    end_date (datetime.datetime): 指定日期范围的结束日期。

    返回:
    tuple: 一个元组，包含两个整数，分别是指定日期范围在自然日数组中的开始和结束索引。
    """

    # 找到start_date在nature_days_array中的插入位置，这将是开始索引
    start_idx = np.searchsorted(nature_days_array, start_date, side='left')

    # 找到end_date在nature_days_array中的插入位置，然后减一，这将是结束索引
    end_idx = np.searchsorted(nature_days_array, end_date, side='right') - 1
    # 减一的原因是np.searchsorted找到的是end_date之后的位置，我们需要的是end_date的索引

    # 返回开始和结束索引
    return start_idx, end_idx


# 使用共享内存中的数据计算指标
def calc_metrics_shared_worker(args, shm_info):
    """
    使用共享内存中的数据计算指标。

    该函数从共享内存中获取收益数据和价格数据，然后使用这些数据和其它参数计算指定时期的指标。
    使用共享内存可以有效地减少数据复制和提高性能，适用于多进程环境中的数据共享。

    参数:
    args (tuple): 包含计算指标所需的各种参数和数据。
    shm_info (dict): 包含共享内存信息的字典，用于访问日志和价格数据。

    返回:
    pandas.DataFrame: 计算得到的指标数据框。
    """
    # 解包计算指标所需的参数
    (
        start_date,  # 当前计算周期的开始日期，表示指标计算的起始日期（pd.Timestamp 类型）。
        end_date,  # 当前计算周期的结束日期，表示指标计算的截止日期（pd.Timestamp 类型）。
        start_idx,  # 指定日期范围在自然日数组中的起始索引位置（整数类型）。
        end_idx,  # 指定日期范围在自然日数组中的结束索引位置（整数类型）。
        days_in_p,  # 当前计算周期内的自然日数量，用于时间相关的指标计算（整数类型）。
        funds_codes,  # 基金代码数组，包含所有需要计算指标的基金代码（numpy 数组类型）。
        period,  # 当前计算的时间区间类型，例如 '1m'（近一个月）、'qtd'（本季至今）等（字符串类型）。
        metrics_list,  # 需要计算的指标列表，包含具体的指标名称（列表类型）。
        min_data_required,  # 每个基金至少需要多少天的数据才能计算指标，默认为 2 天（整数类型）。
        first_valid_dates,  # 每个基金第一个非 NaN 值所对应的日期（numpy 数组类型，元素为 pd.Timestamp）。
    ) = args

    # 解包共享内存信息，包括共享内存名称、形状和数据类型
    log_shm_name, log_shape, log_dtype = shm_info["log"]
    price_shm_name, price_shape, price_dtype = shm_info["price"]

    # 创建共享内存的收益和价格数组视图
    shm_log = shared_memory.SharedMemory(name=log_shm_name)  # 收益
    log_arr = np.ndarray(log_shape, dtype=log_dtype, buffer=shm_log.buf)  # 价格

    # 创建共享内存的收益和价格数组视图
    shm_price = shared_memory.SharedMemory(name=price_shm_name)
    price_arr = np.ndarray(price_shape, dtype=price_dtype, buffer=shm_price.buf)

    # 通过切片获取指定时期的日志和价格数据，注意切片操作不会复制数据
    in_p_log = log_arr[start_idx + 1: end_idx + 1]
    in_p_price = price_arr[start_idx + 1: end_idx + 1]

    # 使用布尔索引得到保留的列索引
    keep_indices = np.where(first_valid_dates <= start_date)[0]
    # 使用切片保留对应基金（共享内存）
    filtered_funds_codes = funds_codes[keep_indices]
    filtered_log_return_array = in_p_log[:, keep_indices]
    filtered_close_price_array = in_p_price[:, keep_indices]

    # 实例化 CalMetrics 类，并使用切片后的数据计算指标
    c_m = CalMetrics(
        filtered_funds_codes,
        filtered_log_return_array,
        filtered_close_price_array,
        period,
        days_in_p,
        end_date,
        min_data_required=min_data_required,
    )
    sub_df = c_m.cal_metric_main(metrics_list)

    # 关闭共享内存视图，注意不要在此处释放共享内存，它将在主进程中释放
    shm_log.close()
    shm_price.close()

    # 返回计算得到的指标数据框
    return sub_df


# 创建共享内存对象
def create_shared_memory(arr: np.ndarray, name: str):
    """
    创建共享内存对象，并将给定数组的数据拷贝到共享内存中。

    参数:
    arr: np.ndarray - 需要共享的numpy数组。
    name: str - 共享内存的名称，用于标识共享内存段。

    返回:
    shm: shared_memory.SharedMemory - 创建的共享内存对象。
    shm_array: np.ndarray - 在共享内存上创建的numpy数组，与原始数组具有相同的形状和数据类型。
    """
    # 先尝试清除已有同名共享内存
    try:
        existing_shm = shared_memory.SharedMemory(name=name)
        existing_shm.close()
        existing_shm.unlink()
    except FileNotFoundError:
        pass  # 不存在，无需处理
    except Exception as e:
        print(f"清理旧共享内存时出错: {e}")

    # 创建一个新的共享内存段，名称为指定的名字，大小为数组所需的字节数
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes, name=name)
    # 在共享内存缓冲区上创建一个numpy数组，这样可以与原始数组具有相同的形状和数据类型
    shm_array = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    # 将原始数组的数据拷贝到共享内存数组中
    shm_array[:] = arr[:]  # 拷贝数据进去

    # 返回共享内存对象和在共享内存中的数组
    return shm, shm_array


# 遍历要计算的区间
def compute_metrics_for_period_initialize(log_return_df,
                                          close_price_df,
                                          high_price_df,
                                          low_price_df,
                                          volume_df,
                                          save_path,
                                          p_list=None,
                                          metrics_list=None,
                                          fund_list=None,
                                          spec_end_date=None,
                                          num_workers=None,
                                          multi_process=True,
                                          min_data_required=2
                                          ):
    """
    计算特定时间段内的基金业绩指标，并将结果保存为 Parquet 文件。

    :param log_return_df: 对数收益率 DataFrame，索引为日期，列为基金代码。
    :param close_price_df: 收盘价 DataFrame，索引为日期，列为基金代码。
    :param high_price_df: 最高价 DataFrame，索引为日期，列为基金代码。
    :param low_price_df: 最低价 DataFrame，索引为日期，列为基金代码。
    :param volume_df: 成交量 DataFrame，索引为日期，列为基金代码。
    :param save_path: 指标结果保存路径。
    :param p_list: 特定需要计算指标的时间段列表，如果为 None，则计算所有预定义的时间段。
    :param metrics_list: 特定需要计算的指标列表，如果为 None，则计算所有预定义的指标。
    :param fund_list: 特定需要计算的ETF列表，如果为 None，则计算所有ETF的指标。
    :param spec_end_date: 特定的区间结束日期 (必须是交易日才能计算)，如果为 None，则计算所有。
    :param num_workers: 并行处理时的进程数，如果为 None，则默认使用所有可用的 CPU 核心数。
    :param multi_process: 是否使用多进程并行计算指标，默认为 True。
    :param min_data_required: int, 计算指标的最少数据量要求，默认为 True。
    :return: 无返回值。
    """
    ''' 1) 数据预处理 '''
    if fund_list is not None:
        log_return_df = log_return_df[fund_list]
        close_price_df = close_price_df[fund_list]
        high_price_df = high_price_df[fund_list]
        low_price_df = low_price_df[fund_list]
        volume_df = volume_df[fund_list]
    # 获取交易日序列
    trading_days_array = pd.to_datetime(np.array(log_return_df.index))

    # 将索引日期扩充到自然日
    log_return_df = log_return_df.resample('D').asfreq()
    close_price_df = close_price_df.resample('D').asfreq()
    high_price_df = high_price_df.resample('D').asfreq()
    low_price_df = low_price_df.resample('D').asfreq()
    volume_df = volume_df.resample('D').asfreq()

    # 获取自然日序列
    nature_days_array = pd.to_datetime(np.array(log_return_df.index))
    # 得到区间与该区间要计算指标的映射
    period_metrics_map = create_period_metrics_map()
    # 找到每个产品第一个非nan值所对应的日期
    first_valid_dates = log_return_df.apply(lambda col: col.first_valid_index()).values

    # 将 dataframe 转换为 numpy 数组
    log_return_array = log_return_df.values
    close_price_array = close_price_df.values
    high_price_array = high_price_df.values
    low_price_array = low_price_df.values
    volume_array = volume_df.values

    # 得到基金代码
    funds_codes = log_return_df.columns.values
    # 设置进程数
    num_workers = cpu_count() if num_workers is None else num_workers

    # 释放内存
    del log_return_df, close_price_df, high_price_df, low_price_df, volume_df
    gc.collect()

    ''' 2) 遍历各个区间 '''
    for period in p_list if p_list else period_list:
        if period not in period_metrics_map:
            continue
        print(f"[开始] 计算区间: {period}")
        s_t = time.time()

        ''' 3.1) 遍历每一天的结束日期,滚动计算每天的指标 (进程池) '''
        if multi_process:
            # 创建共享内存
            shm_log, _ = create_shared_memory(log_return_array, "log_shm")
            shm_price, _ = create_shared_memory(close_price_array, "price_shm")

            # 创建共享内存信息字典，包含日志和价格的共享内存标识、形状和数据类型
            shm_info = {
                "log": ("log_shm", log_return_array.shape, log_return_array.dtype),
                "price": ("price_shm", close_price_array.shape, close_price_array.dtype),
            }

            # 构造任务列表
            task_args = []
            # 如果有特定结束日期，则只计算该日期
            days_array = np.array([pd.to_datetime(spec_end_date)]) if spec_end_date else trading_days_array
            for end_date in reversed(days_array):
                match = re.match(r'^(\d+).*d$', period)
                # 对于短区间 (2d, 3d, 5d), 我们使用交易日来划分 (避免非交易日导致的数据不完整)
                if match:
                    days = int(match.group(1))  # 获取天数
                    # 获取 end_date 在 trading_days_array 中的索引
                    end_idx_in_trading = np.searchsorted(trading_days_array, end_date)
                    # 检查是否越界
                    if end_idx_in_trading < days:
                        continue
                    # 获取 start_date
                    start_date = trading_days_array[end_idx_in_trading - days]

                # 对于1周以上的中长区间, 我们使用自然日来划分
                else:
                    # 计算开始日期
                    start_date = get_start_date(end_date, period)
                    # 如果 区间开始日期不在自然日序列中，则跳出循环 (区间不完整)
                    if start_date not in nature_days_array:
                        continue

                start_idx, end_idx = find_date_range_indices(nature_days_array, start_date, end_date)
                days_in_p = end_idx - start_idx
                # 将任务参数添加到任务列表中，每个任务包含以下信息：
                task_args.append((
                    start_date,  # 当前计算周期的开始日期，表示指标计算的起始日期。
                    end_date,  # 当前计算周期的结束日期，表示指标计算的截止日期。
                    start_idx,  # 指定日期范围在自然日数组中的起始索引位置。
                    end_idx,  # 指定日期范围在自然日数组中的结束索引位置。
                    days_in_p,  # 当前计算周期内的自然日数量，用于时间相关的指标计算。
                    funds_codes,  # 基金代码数组，包含所有需要计算指标的基金代码。
                    period,  # 当前计算的时间区间类型，例如'1m'(近一个月)、'qtd'(本季至今)等。
                    metrics_list if metrics_list else period_metrics_map[period],  # 需要计算的指标列表
                    min_data_required,  # 每个基金至少需要多少天的数据才能计算指标，默认为2天。
                    first_valid_dates,  # 每个基金第一个非nan值所对应的日期
                ))

            # 使用多进程执行任务
            with Pool(num_workers) as pool:
                results = list(tqdm(
                    pool.imap(partial(calc_metrics_shared_worker, shm_info=shm_info), task_args),
                    total=len(task_args),
                    desc=f"计算周期 {period}"
                ))

            # 清理共享内存
            shm_log.close()
            shm_log.unlink()
            shm_price.close()
            shm_price.unlink()

            # 合并结果
            results = [r for r in results if r is not None]
            final_df = pd.concat(results, axis=0)
            file_path = os.path.join(save_path, f"{period}.parquet")
            final_df.to_parquet(file_path)
            print(f"[完成] 区间{period}指标计算完成, 保存至 {file_path}，"
                  f"共 {len(final_df)} 条记录，耗时 {(time.time() - s_t):.2f} 秒")

        else:
            ''' 3.2) 遍历每一天的结束日期,滚动计算每天的指标 (单进程) '''
            final_df = list()
            days_array = np.array([pd.to_datetime(spec_end_date)]) if spec_end_date else trading_days_array
            for end_date in tqdm(list(reversed(days_array))):
                match = re.match(r'^(\d+).*d$', period)
                # 对于短区间 (2d, 3d, 5d), 我们使用交易日来划分 (避免非交易日导致的数据不完整)
                if match:
                    days = int(match.group(1))  # 获取天数
                    # 获取 end_date 在 trading_days_array 中的索引
                    end_idx_in_trading = np.searchsorted(trading_days_array, end_date)
                    # 检查是否越界
                    if end_idx_in_trading < days:
                        continue
                    # 获取 start_date
                    start_date = trading_days_array[end_idx_in_trading - days]

                # 对于1周以上的中长区间, 我们使用自然日来划分
                else:
                    # 计算开始日期
                    start_date = get_start_date(end_date, period)
                    # 如果 区间开始日期不在自然日序列中，则跳出循环 (区间不完整)
                    if start_date not in nature_days_array:
                        continue

                # 找到 开始日期 & 结束日期在 nature_days_array 的索引位置
                start_idx, end_idx = find_date_range_indices(nature_days_array, start_date, end_date)

                # 截取在区间内的数据
                in_p_log_return_array = log_return_array[start_idx + 1: end_idx + 1]  # 对数收益率
                in_p_close_price_array = close_price_array[start_idx + 1: end_idx + 1]  # 收盘价
                in_p_high_price_array = high_price_array[start_idx + 1: end_idx + 1]  # 最高价
                in_p_low_price_array = low_price_array[start_idx + 1: end_idx + 1]  # 最低价
                in_p_volume_array = volume_array[start_idx + 1: end_idx + 1]  # 成交量

                # 计算区间内有多少个自然日
                days_in_p = end_idx - start_idx

                # 找到那些在区间开始日期之前就有数据的基金, 用布尔索引得到这些基金的列索引 (我们只计算这些基金)
                keep_indices = np.where(first_valid_dates <= start_date)[0]
                # 使用切片保留对应基金
                filtered_funds_codes = funds_codes[keep_indices]  # 基金代码
                filtered_log_return_array = in_p_log_return_array[:, keep_indices]  # 对数收益率
                filtered_close_price_array = in_p_close_price_array[:, keep_indices]  # 收盘价
                filtered_high_price_array = in_p_high_price_array[:, keep_indices]  # 最高价
                filtered_low_price_array = in_p_low_price_array[:, keep_indices]  # 最低价
                filtered_volume_array = in_p_volume_array[:, keep_indices]  # 成交量

                # 创建CalMetrics类的实例，用于计算特定时间段内的基金指标
                c_m = CalMetrics(
                    filtered_funds_codes,  # 基金代码数组，包含所有需要计算指标的基金代码
                    filtered_log_return_array,  # 在指定日期范围内的对数收益率数据，numpy数组格式
                    filtered_close_price_array,  # 在指定日期范围内的收盘价数据，numpy数组格式
                    filtered_high_price_array,  # 在指定日期范围内的最高价数据，numpy数组格式
                    filtered_low_price_array,  # 在指定日期范围内的最低价数据，numpy数组格式
                    filtered_volume_array,  # 在指定日期范围内的交易量数据，numpy数组格式
                    period,  # 计算指标的时间区间类型，例如'1m'(近一个月)、'qtd'(本季至今)等
                    days_in_p,  # 指定日期范围内自然日的数量，用于计算某些时间相关的指标
                    end_date,  # 当前计算周期的结束日期，pd.Timestamp格式
                    min_data_required,  # 最小数据要求，表示计算指标时至少需要的数据点数量，默认为2
                )
                # 是否有指定的指标需要计算
                sub_df = c_m.cal_metric_main(metrics_list if metrics_list else period_metrics_map[period])
                final_df.append(sub_df)

            # 将所有区间的指标数据合并
            final_df = pd.concat(final_df, axis=0)
            file_path = os.path.join(save_path, f"{period}.parquet")
            final_df.to_parquet(file_path)
            print(f"[完成] 区间{period}指标计算完成, 保存至 {file_path}，"
                  f"共 {len(final_df)} 条记录，耗时 {(time.time() - s_t):.2f} 秒")


if __name__ == '__main__':
    pass
