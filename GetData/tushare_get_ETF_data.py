# -*- encoding: utf-8 -*-
"""
@File: tushare_getdata.py
@Modify Time: 2025/4/3 17:21       
@Author: Kevin-Chen
@Descriptions: 使用 Tushare 接口获取 ETF 数据
"""
from datetime import datetime, timedelta
from collections import deque
import pyarrow.parquet as pq
from set_tushare import pro
from functools import wraps
from tqdm import tqdm
import pyarrow as pa
import pandas as pd
import traceback
import time
import os

pd.set_option('display.max_columns', 1000)  # 显示字段的数量
pd.set_option('display.width', 1000)  # 表格不分段显示


# 获取ETF的基本信息
def get_etf_info(drop_dlist=True, save_parquet=False):
    """
    获取ETF的基本信息，并根据需求进行数据处理和保存。

    Parameters:
    drop_dlist (bool): 是否剔除已经清盘的ETF，默认为True。
    save_parquet (bool): 是否将结果保存为parquet格式文件，默认为True。

    Returns:
    DataFrame: ETF的基本信息数据框。
    """
    # 获取ETF基本信息
    max_retries = 5
    etf_info_df = None
    for attempt in range(max_retries):
        try:
            etf_info_df = pro.fund_basic(market='E')
        except Exception as e:
            print(f"[重试 {attempt + 1}/{max_retries}] 获取 ETF 基本信息失败: {e}")
            if attempt == max_retries - 1:
                print("[错误] 最多重试已达上限，仍然失败")
                print(traceback.format_exc())
                return None  # 或 raise e 根据你的策略

    # 将ETF按list_date升序排列
    etf_info_df = etf_info_df.sort_values(by='list_date', ascending=True)
    # 剔除已经清盘的ETF
    if drop_dlist:
        etf_info_df = etf_info_df[etf_info_df['delist_date'].isnull()]
        etf_info_df = etf_info_df[etf_info_df['due_date'].isnull()]
    # 重置数据框索引
    etf_info_df = etf_info_df.reset_index(drop=True)
    # 将数据保存为parquet格式
    if save_parquet:
        etf_info_df.to_parquet('etf_info.parquet')
    # 返回处理后的ETF信息数据框
    return etf_info_df


# 按年份分割日期, 得到每年的起始和结束日期二维列表
def split_dates_by_year(start_date, end_date):
    """
    根据年份拆分日期范围。

    将给定的日期范围按年份拆分成多个子范围，每个子范围表示一年的开始和结束日期。

    参数:
    start_date -- 字符串，表示日期范围的开始日期，格式为YYYYMMDD。
    end_date -- 字符串，表示日期范围的结束日期，格式为YYYYMMDD。

    返回值:
    一个列表，包含多个子列表，每个子列表包含两个字符串，分别表示一年的开始和结束日期。
    """
    # 将字符串转换为datetime对象
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')

    # 初始化结果列表
    result = []
    # 从起始年份开始遍历到结束年份
    current_year = start.year

    while current_year <= end.year:
        # 计算当前年份的第一天和最后一天
        year_start = datetime(current_year, 1, 1)
        year_end = datetime(current_year, 12, 31)

        # 如果是第一年，使用实际的起始日期
        if current_year == start.year:
            current_start = start
        else:
            current_start = year_start

        # 如果是最后一年，使用实际的结束日期
        if current_year == end.year:
            current_end = end
        else:
            current_end = year_end

        # 将日期格式化为字符串并添加到结果列表中
        result.append([
            current_start.strftime('%Y%m%d'),
            current_end.strftime('%Y%m%d')
        ])

        # 增加年份以继续下一年的处理
        current_year += 1

    # 返回结果列表
    return result


# 按5年分割日期, 得到每5年的起始和结束日期二维列表
def split_dates_by_5_years(start_date, end_date):
    """
    将给定的日期范围按每5年拆分为多个子范围。
    第一个和最后一个子范围会使用实际的起止日期。

    参数:
    start_date -- 字符串，格式为YYYYMMDD
    end_date -- 字符串，格式为YYYYMMDD

    返回值:
    一个列表，每个元素是 [子起始日期, 子结束日期]
    """
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')

    result = []

    current_start = start

    while current_start <= end:
        # 当前段的起始年份
        start_year = current_start.year
        # 计算5年后的最后一天（最多不超过 end）
        end_year = min(start_year + 4, end.year)
        current_end = datetime(end_year, 12, 31)

        # 如果这是最后一段，截断为实际 end 日期
        if current_end > end:
            current_end = end

        result.append([
            current_start.strftime('%Y%m%d'),
            current_end.strftime('%Y%m%d')
        ])

        # 下一段从下一年开始
        current_start = datetime(end_year + 1, 1, 1)

    return result


# 装饰器工厂，用于限制函数调用的速率
def rate_limit(calls_per_period=250, period=60):
    """
    一个装饰器工厂，用于限制函数调用的速率。

    :param calls_per_period: 在一个周期内允许的最大调用次数，默认为250次。
    :param period: 一个周期的时间长度（以秒为单位），默认为60秒。
    :return: 返回一个装饰器，用于应用速率限制到函数上。
    """
    call_timestamps = deque()  # 初始化一个双端队列，用于记录每个调用的时间戳。

    def decorator(func):
        """
        装饰器，用于应用速率限制到指定的函数。

        :param func: 被装饰的函数。
        :return: 返回包装后的函数。
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            包装函数，负责执行速率限制逻辑，并调用原始函数。

            :param args: 位置参数，传递给被装饰函数。
            :param kwargs: 关键字参数，传递给被装饰函数。
            :return: 返回被装饰函数的执行结果。
            """
            current = time.time()  # 获取当前时间戳。

            # 清理窗口外的时间戳
            while (call_timestamps  # 检查 call_timestamps 是否为空，若为空则直接跳过循环。
                   and current - call_timestamps[0] >= period  # 判断队列中最旧的时间戳是否已经超出了当前时间减去 period 的范围。
            ):
                call_timestamps.popleft()  # 如果条件满足，则移除队列中最旧的时间戳（即超出时间窗口的记录）。

            # 检查是否达到调用限制
            if len(call_timestamps) >= calls_per_period:
                sleep_time = period - (current - call_timestamps[0])  # 计算需要等待的时间。
                sleep_time = max(sleep_time, 1)  # 确保至少等待1秒
                # print(f"[限速] 超过调用限制，等待 {sleep_time:.2f} 秒...")  # 提示用户正在等待。
                time.sleep(sleep_time)  # 等待，以遵守速率限制。
                return wrapper(*args, **kwargs)  # 递归调用

            call_timestamps.append(time.time())  # 记录本次调用的时间戳。
            return func(*args, **kwargs)  # 调用被装饰的函数。

        return wrapper

    return decorator


@rate_limit(calls_per_period=249, period=60)
def safe_fund_daily(ts_code, start_date, end_date):
    """
    获取指定日期范围内的安全基金每日交易数据。

    该函数通过调用pro.fund_daily接口实现，使用了装饰器@rate_limit来限制函数调用的频率，
    以避免超过API提供商的限流限制。装饰器参数calls_per_period和period定义了每60秒最多调用240次。

    参数:
    ts_code (str): 基金的TS代码，用于标识特定的基金产品。
    start_date (str): 数据获取的开始日期，格式为YYYYMMDD。
    end_date (str): 数据获取的结束日期，格式为YYYYMMDD。

    返回:
    DataFrame: 包含指定基金和日期范围内每日交易数据的DataFrame对象。
    """
    return pro.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)


# 获取ETF的日线数据, 根据指定的开始和结束日期
def get_etf_daily_data(etf_code, start_date='20040101', end_date=datetime.now().strftime('%Y%m%d'), save_parquet=False):
    # print(f"正在获取 {etf_code} 的日线数据...")
    # 得到每年的起始和结束日期二维列表
    dates_list = split_dates_by_5_years(start_date, end_date)
    final_df = list()

    # 遍历每年的起始和结束日期, 获取日线数据
    for s_t, e_t in dates_list:
        sub_df = safe_fund_daily(ts_code=etf_code, start_date=s_t, end_date=e_t)
        final_df.append(sub_df)

    # 将所有数据合并为一个DataFrame, 并按日期升序排列
    final_df = [df for df in final_df if df is not None and not df.empty]
    if not final_df:
        print(f"没有获取到 {etf_code} 的日线数据")
        return pd.DataFrame()
    final_df = pd.concat(final_df, ignore_index=True)
    final_df = final_df.sort_values(by='trade_date', ascending=True)
    final_df = final_df.reset_index(drop=True)
    # print(f"{etf_code}的日线数据获取完成！")

    # 将数据保存为parquet格式
    if save_parquet:
        final_df.to_parquet(f'../data/{etf_code}_daily.parquet')
    return final_df


# 全量获取所有未清盘的ETF的每日数据，并将其保存到一个Parquet文件中
def get_etf_daily_data_all():
    """
    获取所有未清盘的ETF的每日数据，并将其保存到一个Parquet文件中。
    该函数首先获取所有ETF的代码和发行日期，然后为每个ETF获取每日数据，
    并将这些数据增量写入到一个Parquet文件中。如果目标Parquet文件已存在，
    则会在开始之前将其备份。
    :return: 无返回值
    """
    '''1) 获取所有未清盘的ETF的代码'''
    # 获取ETF信息，去除重复的代码，只保留首次出现的记录
    ts_issue_dict = get_etf_info().drop_duplicates(subset="ts_code", keep="first")
    # 将代码和发行日期制成字典
    ts_issue_dict = dict(
        zip(
            ts_issue_dict["ts_code"],
            ts_issue_dict["issue_date"].fillna("20040101").astype(str)
        )
    )
    # 打印ETF基金总数
    print(f"共有 {len(ts_issue_dict)} 只 ETF 基金")

    '''2) 启动前备份旧文件'''
    # 定义保存路径
    output_path = "../data/etf_daily.parquet"
    # 如果文件已存在，则进行备份
    if os.path.exists(output_path):
        today = datetime.today().strftime('%Y%m%d')
        backup_path = f"../data/etf_daily_{today}.parquet"
        os.rename(output_path, backup_path)
        print(f"[备份] 原 parquet 文件已保存为 {backup_path}")

    # 初始化 ParquetWriter（延迟初始化，直到写入第一块数据）
    writer = None

    '''3) 循环写入 ETF 数据'''

    # 定义一个安全获取ETF每日数据的函数，以处理可能的异常
    def safe_get_etf_daily_data(code, start_date):
        return get_etf_daily_data(etf_code=code, start_date=start_date)

    # 遍历所有ETF，获取并写入数据
    for etf_code, issue_date in tqdm(ts_issue_dict.items()):
        for attempt in range(5):
            try:
                df = safe_get_etf_daily_data(code=etf_code, start_date=issue_date)
                if df is not None and not df.empty:
                    # 将DataFrame转为Arrow表
                    table = pa.Table.from_pandas(df)

                    # 如果是第一次写入，初始化writer
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, table.schema)

                    # 进行增量写入
                    writer.write_table(table)
                    break  # 成功，退出重试循环

            except Exception as e:
                print(f"[重试 {attempt + 1}/5] 获取 {etf_code} 数据失败: {e}")
                if attempt == 4:  # 第5次仍失败，打印详细堆栈
                    print(f"获取 {etf_code} 数据失败: {e}")
                    print(traceback.format_exc())

    '''4) 最后关闭 writer'''
    # 确保所有数据已被写入，并关闭writer
    if writer:
        writer.close()
        print(f"[完成] 所有数据已写入 {output_path}")
    else:
        print("[警告] 没有任何数据被写入")


# 递增获取ETF每日数据并更新到parquet文件中
def get_etf_daily_data_increment():
    """
    递增获取ETF每日数据并更新到parquet文件中。

    该函数首先获取所有ETF的基本信息，
    然后遍历每只ETF, 调用API获取从六天前到今天的数据。
    如果获取数据成功, 则将数据追加到最终的数据集中。
    最后，将新数据与原有的parquet文件中的数据合并去重，并重新保存为parquet文件。
    """
    # 获取ETF基本信息，并去重
    ts_issue_list = get_etf_info().drop_duplicates(subset="ts_code", keep="first")["ts_code"].unique().tolist()
    final_df = list()

    # 遍历每只ETF，获取其每日数据（带重连机制）
    for ts_code in tqdm(ts_issue_list):
        for attempt in range(5):
            try:
                usb_df = safe_fund_daily(
                    ts_code=ts_code,
                    start_date=(datetime.today() - timedelta(days=6)).strftime('%Y%m%d'),
                    end_date=datetime.now().strftime('%Y%m%d')
                )
                if usb_df is not None and not usb_df.empty:
                    final_df.append(usb_df)
                    break  # 成功跳出重试
            except Exception as e:
                print(f"[重试 {attempt + 1}/5] 获取 {ts_code} 数据失败: {e}")
                if attempt == 4:  # 第3次还是失败
                    print(f"获取 {ts_code} 数据失败: {e}")
                    print(traceback.format_exc())

    # 2. 合并新数据
    if final_df is None or len(final_df) == 0:
        print("无新数据，程序结束")

    new_data = pd.concat(final_df, ignore_index=True)

    # 3. 读取原 parquet 数据（如果存在）
    parquet_path = "../data/etf_daily.parquet"
    if os.path.exists(parquet_path):
        old_data = pd.read_parquet(parquet_path)
        # 备份原文件（新增逻辑）
        today = datetime.today().strftime('%Y%m%d')
        backup_path = f"../data/etf_daily_{today}.parquet"
        os.rename(parquet_path, backup_path)
        print(f"[备份] 原文件已备份为 {backup_path}")

        # 合并新旧数据，并去重
        combined_df = pd.concat([old_data, new_data], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    else:
        combined_df = new_data.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")

    # 4. 重写 parquet 文件
    combined_df = combined_df.sort_values(by=["ts_code", "trade_date"]).reset_index(drop=True)
    combined_df.to_parquet(parquet_path)
    print(f"[完成] 已保存至 {parquet_path}，共 {len(combined_df)} 条记录")


if __name__ == '__main__':
    get_etf_daily_data_all()
    get_etf_daily_data_increment()
