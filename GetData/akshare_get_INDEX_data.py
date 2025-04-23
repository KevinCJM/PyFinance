# -*- encoding: utf-8 -*-
"""
@File: tushare_get_INDEX_data.py
@Modify Time: 2025/4/22 15:45       
@Author: Kevin-Chen
@Descriptions: 用Tushare接口获取指数数据
"""

import gc
import warnings
import akshare as ak
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

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 1000)  # 显示字段的数量
pd.set_option('display.width', 1000)  # 表格不分段显示
# 获取今日日期
today = datetime.now()


# 获取并保存指数信息
def get_index_info(save_path='../Data/Index'):
    """
    获取并保存指数信息。

    该函数会从不同的市场中获取指数基本信息，并保存到指定的路径中。
    使用重试机制来处理可能的请求失败情况。

    :param save_path: 保存指数信息文件的路径，默认为'../Data/Index'
    :return: 无
    """
    # 打印开始获取指数信息的消息
    print('开始获取指数信息...')

    # 定义市场字典，包含不同市场的名称和描述
    market_dict = {
        'MSCI': 'MSCI指数',
        'CSI': '中证指数',
        'SSE': '上交所指数',
        'SZSE': '深交所指数',
        'CICC': '中金指数',
        'SW': '申万指数',
        'OTH': '其他指数',
    }

    # 初始化一个空列表，用于存储所有市场的指数信息数据框
    final_df = list()
    # 定义最大重试次数常量
    max_retries = 5  # 最大重试次数

    # 遍历市场字典，获取每个市场的指数信息
    for market, desc in market_dict.items():
        # 打印当前市场的描述
        for attempt in range(max_retries):
            try:
                # 尝试获取指数信息
                print(f"\t正在获取 {desc}...")
                df = pro.index_basic(market=market)
                final_df.append(df)
                break
            except Exception as e:
                # 打印错误信息并重试
                print(f"[重试 {attempt + 1}/{max_retries}] 获取 {desc} 失败: {e}")
                if attempt == max_retries - 1:
                    # 如果达到最大重试次数，打印错误信息并返回None
                    print("[错误] 最多重试已达上限，仍然失败")
                    print(traceback.format_exc())
                    return None  # 或 raise e 根据你的策略

    # 合并所有数据
    final_df = pd.concat(final_df)
    # 重置索引
    final_df = final_df.reset_index(drop=True)
    # 打印合并后的数据
    print(final_df)
    # 保存数据到Excel和Parquet文件
    final_df.to_excel(os.path.join(save_path, 'index_info.xlsx'), index=False)
    final_df.to_parquet(os.path.join(save_path, 'index_info.parquet'), index=False)
    # 打印完成消息和保存路径
    print('指数信息获取完成, 保存路径:', os.path.join(save_path, 'index_info.xlsx & index_info.parquet'))


# 获取全球指数数据并保存为Parquet文件
def get_global_index_data(code_dict, func_obj, save_path='../Data/Index', parquet_name='xxx.parquet'):
    """
    获取全球指数数据并保存为Parquet文件。

    参数:
    code_dict (dict): 指数代码和名称的字典。
    func_obj (function): 用于获取指数数据的函数对象。
    save_path (str): 数据保存的路径，默认为'../Data/Index'。
    parquet_name (str): 保存的Parquet文件名，默认为'xxx.parquet'。

    返回:
    DataFrame: 合并后的全球指数数据。
    """
    # 打印开始获取数据的消息
    print('开始获取全球指数数据...')

    # 初始化一个空列表，用于存储所有指数的数据
    final_df = list()
    # 定义最大重试次数常量
    max_retries = 5  # 最大重试次数

    # 遍历指数字典，获取每个指数的历史数据
    for i_code, i_name in code_dict.items():
        # 尝试获取, 允许失败5次
        for attempt in range(max_retries):
            # 打印当前正在获取的指数名称
            print(f"\t获取{i_code}-{i_name}数据...")
            try:
                # 尝试获取指数数据
                sub_df = func_obj(symbol=i_name)
                if '代码' not in sub_df.columns:
                    sub_df['ts_code'] = i_name
                # 将获取到的数据添加到列表中
                final_df.append(sub_df)
                # 成功获取数据后，跳出循环
                break
            except Exception as e:
                # 打印错误信息并重试
                print(f"[重试 {attempt + 1}/{max_retries}] 获取 {i_name} 失败: {e}")
                if attempt == max_retries - 1:
                    # 如果达到最大重试次数，打印错误信息并返回None
                    print("[错误] 最多重试已达上限，仍然失败")
                    print(traceback.format_exc())

    # 合并所有数据
    final_df = pd.concat(final_df)
    # 重置索引
    final_df = final_df.reset_index(drop=True)
    # 变更字段名
    if '日期' in final_df.columns:
        final_df.rename(
            columns={'日期': 'trade_date', '代码': 'ts_code', '名称': 'name', '今开': 'open', '最新价': 'close',
                     '最高': 'high', '最低': 'low', '振幅': 'change'}, inplace=True)
        final_df.rename(
            columns={'指数': 'close'}, inplace=True)
    elif 'date' in final_df.columns:
        final_df.rename(columns={'date': 'trade_date'}, inplace=True)
    # 选择需要的列
    if 'open' in final_df.columns:
        final_df = final_df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close']]
    else:
        final_df = final_df[['ts_code', 'trade_date', 'close']]
    # 打印合并后的数据
    print(final_df)
    # 保存数据到Parquet文件
    final_df.to_parquet(os.path.join(save_path, parquet_name), index=False)
    # 打印数据获取完成的消息和保存路径
    print('全球指数数据获取完成, 保存路径:', os.path.join(save_path, parquet_name))
    # 返回合并后的数据
    return final_df


# 获取全球多个国家的利率决议数据
def get_global_interest_rate_data(save_path='../Data/Index', parquet_name='global_interest_daily.parquet'):
    """
    获取全球多个国家的利率决议数据，并保存为 Parquet 文件。

    参数:
    save_path (str): 数据保存的文件夹路径。默认为 '../Data/Index'。
    parquet_name (str): 保存的 Parquet 文件名。默认为 'global_interest_daily.parquet'。

    返回:
    pd.DataFrame: 合并后的全球利率数据 DataFrame。
    """
    # 定义一个字典，映射不同国家/地区的利率决议报告与其对应的获取数据的函数
    interest_rate_dict = {
        '美联储利率决议报告': ak.macro_bank_usa_interest_rate,
        '欧洲央行决议报告': ak.macro_bank_euro_interest_rate,
        '新西兰联储决议报告': ak.macro_bank_newzealand_interest_rate,
        '中国央行决议报告': ak.macro_bank_china_interest_rate,
        '瑞士央行利率决议报告': ak.macro_bank_switzerland_interest_rate,
        '英国央行决议报告': ak.macro_bank_english_interest_rate,
        '澳洲联储决议报告': ak.macro_bank_australia_interest_rate,
        '日本利率决议报告': ak.macro_bank_japan_interest_rate,
        '俄罗斯利率决议报告': ak.macro_bank_russia_interest_rate,
        '印度利率决议报告': ak.macro_bank_india_interest_rate,
        '巴西利率决议报告': ak.macro_bank_brazil_interest_rate,
    }
    # 打印开始获取数据的消息
    print('开始获取全球利率数据...')
    # 初始化一个空列表，用于存储所有数据
    final_df = list()
    # 定义最大重试次数为5次
    max_retries = 5  # 最大重试次数
    # 遍历字典中的每个国家/地区及其对应的函数
    for name, func in interest_rate_dict.items():
        # 打印当前获取的数据名称
        print(f"\t获取{name}数据...")
        # 尝试获取数据，允许失败5次
        for attempt in range(max_retries):
            try:
                # 调用函数获取利率数据
                interest_rate_df = func().copy()
                # 将日期列转换为日期格式
                interest_rate_df['日期'] = pd.to_datetime(interest_rate_df['日期'])
                # 获取当前日期（以“天”为单位截断）
                today = pd.Timestamp.today().normalize()
                # 如果 DataFrame 中没有今天的数据，添加一行空值
                if today not in interest_rate_df['日期'].values:
                    product = interest_rate_df['商品'].iloc[0]  # 因为只有一个商品
                    interest_rate_df = pd.concat([
                        interest_rate_df,
                        pd.DataFrame([{
                            '商品': product,
                            '日期': today,
                            '今值': pd.NA
                        }])
                    ], ignore_index=True)
                # 仅保留商品、日期和今值列
                interest_rate_df = interest_rate_df[['商品', '日期', '今值']]
                # 去除重复的行，保留最后一次出现的行
                interest_rate_df = interest_rate_df.drop_duplicates(subset=['日期', '商品'], keep='last')
                # 将数据透视，以便每个商品的今值成为列
                pivot_df = interest_rate_df.pivot(index='日期', columns='商品', values='今值')
                # 按天重采样数据，并使用前向填充方法填充缺失值
                pivot_df = pivot_df.resample('D').asfreq().ffill()
                # 重塑数据格式，将透视表变回长格式
                result_df = pivot_df.reset_index().melt(id_vars='日期', var_name='商品', value_name='今值')
                # 将处理后的数据添加到最终的数据列表中
                final_df.append(result_df)
                # 如果成功获取数据，跳出循环
                break
            except Exception as e:
                # 如果发生异常且重试次数未用尽，则打印失败消息并重试
                if attempt < max_retries - 1:
                    print(f"\t\t获取 {name} 数据失败，正在重试...")
                    time.sleep(1)  # 暂停1秒
                else:
                    # 如果重试次数用尽，打印最终失败消息
                    print(f"\t\t获取 {name} 数据失败:{e}，已重试{max_retries}次，不再重试。")
                    print(traceback.format_exc())

    # 合并所有数据，并重置索引
    final_df = pd.concat(final_df, ignore_index=True)
    # 重命名列名，以便后续处理
    final_df.rename(columns={'今值': 'close', '日期': 'trade_date', '商品': 'ts_code'}, inplace=True)
    # 打印合并后的数据
    print(final_df)
    # 构造保存文件的完整路径
    file_path = os.path.join(save_path, parquet_name)
    # 将数据保存为 Parquet 文件
    final_df.to_parquet(file_path, index=False)
    # 打印数据获取完成和保存路径的消息
    print('全球利率数据获取完成, 保存路径:', file_path)
    # 返回合并后的数据
    return final_df


# 获取中国VIX数据并保存为Parquet文件
def get_china_vix_data(save_path='../Data/Index', parquet_name='china_vix_daily.parquet'):
    """
    获取中国VIX数据，并保存为Parquet文件。

    :param save_path: 数据保存的路径，默认为'../Data/Index'
    :param parquet_name: 保存的Parquet文件名，默认为'china_vix_daily.parquet'
    :return: 无
    """
    print('开始获取中国VIX数据...')
    # 字典，包含不同ETF期权波动率指数的代码和对应的获取函数
    vix_dict = {
        '50etf_vix': ak.index_option_50etf_qvix,  # 50ETF 期权波动率指数 (又称中国版的恐慌指数)
        '300etf_vix': ak.index_option_300etf_qvix,  # 300ETF 期权波动率指数
        '500etf_vix': ak.index_option_500etf_qvix,  # 500ETF 期权波动率指数
        'cyb_vix': ak.index_option_cyb_qvix,  # 创业板 期权波动率指数
        'kcb_vix': ak.index_option_kcb_qvix,  # 科创板 期权波动率指数
    }

    # 定义最大重试次数常量
    max_retries = 5

    # 定义空列表，用于存储所有指数的数据
    final_df = list()

    # 遍历字典，获取每个ETF期权波动率指数的数据
    for code, func in vix_dict.items():
        for attempt in range(max_retries):
            try:
                print(f"\t获取{code}期权波动率指数数据...")
                # 调用函数获取数据，并添加ETF代码列
                vix_df = func().copy()
                vix_df['ts_code'] = code
                final_df.append(vix_df)
            except Exception as e:
                # 异常情况下打印错误信息，达到最大重试次数后退出
                print(f"\t[重试 {attempt + 1}/{max_retries}] 获取 {code} 期权波动率指数失败: {e}")
                if attempt == max_retries - 1:
                    print("\t[错误] 最多重试已达上限，仍然失败")
                    print(traceback.format_exc())
                    break
    # 合并结果
    final_df = pd.concat(final_df)
    # 重命名日期列为交易日期
    final_df.rename(columns={'date': 'trade_date'}, inplace=True)
    # 构造文件路径
    file_path = os.path.join(save_path, parquet_name)
    # 保存数据为Parquet文件
    final_df.to_parquet(file_path)
    print('中国VIX数据获取完成, 保存路径:', file_path)


# 获取中国银行间同业拆借利率数据
def get_china_interbank_interest_rate_data(save_path='../Data/Index', parquet_name='china_interbank_daily.parquet'):
    """
    获取全球银行间同业拆借利率数据，并保存为Parquet文件。

    参数:
    save_path (str): 保存文件的路径，默认为'../Data/Index'。
    parquet_name (str): 保存的Parquet文件名，默认为'global_interbank_daily.parquet'。

    返回:
    pd.DataFrame: 合并后的全球银行间同业拆借利率数据。
    """
    # 定义全球银行间同业拆借利率的字典，包含不同货币的拆借市场和指标
    rate_interbank_dict = {
        'Shibor人民币': ('上海银行同业拆借市场', '隔夜'),
        'Chibor人民币': ('中国银行同业拆借市场', '隔夜'),
        # 'Libor英镑': ('伦敦银行同业拆借市场', '隔夜'),
        # 'Libor美元': ('伦敦银行同业拆借市场', '隔夜'),
        # 'Libor欧元': ('伦敦银行同业拆借市场', '隔夜'),
        # 'Libor日元': ('伦敦银行同业拆借市场', '隔夜'),
        # 'Hibor港币': ('伦敦银行同业拆借市场', '隔夜'),
        # 'Hibor美元': ('香港银行同业拆借市场', '隔夜'),
        'Hibor人民币': ('香港银行同业拆借市场', '隔夜'),
    }
    # 打印开始获取数据的消息
    print(f"开始获取全球同业拆借利率数据...")
    # 设置最大重试次数
    max_retries = 5  # 最大重试次数
    # 初始化最终数据的列表
    final_df = list()
    # 遍历字典中的每个利率数据
    for symbol, (market, indicator) in rate_interbank_dict.items():
        # 打印当前获取的数据名称
        print(f"\t获取{symbol}数据...")
        # 尝试获取数据，允许失败5次
        for attempt in range(max_retries):
            try:
                # 调用接口获取数据
                rate_interbank_df = ak.rate_interbank(market=market, symbol=symbol, indicator=indicator)
                # 添加数据标识列
                rate_interbank_df['ts_code'] = symbol
                # 将获取的数据添加到最终数据列表中
                final_df.append(rate_interbank_df)
                # 成功获取数据后，跳出循环
                break
            except Exception as e:
                # 如果达到最大重试次数，打印错误信息并跳出循环
                if attempt == max_retries - 1:
                    print(f"\t\t获取 {symbol} 数据失败，已重试{max_retries}次，不再重试。")
                    print(traceback.format_exc())
                    break
                else:
                    # 未达到最大重试次数，打印重试信息并等待1秒后重试
                    print(f"\t\t获取 {symbol} 数据失败，正在重试...")
                    time.sleep(1)
    # 合并所有数据为一个DataFrame
    final_df = pd.concat(final_df, ignore_index=True)
    # 重命名列名以符合通用格式
    final_df.rename(columns={'报告日': 'trade_date', '利率': 'close'}, inplace=True)
    # 选择需要的列并按顺序排列
    final_df = final_df[['ts_code', 'trade_date', 'close']]
    # 将日期列转换为datetime类型
    final_df['trade_date'] = pd.to_datetime(final_df['trade_date'])
    # 保存数据为Parquet文件
    file_path = os.path.join(save_path, parquet_name)
    final_df.to_parquet(file_path, index=False)
    # 打印数据获取完成和保存路径的消息
    print('全球同业拆借利率数据获取完成, 保存路径:', file_path)
    # 返回合并后的数据
    return final_df


# 获取中国PMI相关数据并保存为Parquet文件
def get_china_pmi_data(save_path='../Data/Index', parquet_name='china_pmi_daily.parquet', fill_method='ffill'):
    """
    获取中国PMI相关数据并保存为Parquet文件。

    :param save_path: 数据保存路径，默认为'../Data/Index'
    :param parquet_name: 保存的Parquet文件名，默认为'china_pmi_daily.parquet'
    :param fill_method: 数据填充方法，默认为 'ffill'，可选值包括 'ffill', 'time', 'linear'
    :return: 无
    """
    # 定义PMI数据字典，键为数据名称，值为对应的API调用函数
    pmi_dict = {
        '集装箱指数WCI': ak.drewry_wci_index,
        '综合PMI': ak.index_pmi_com_cx,
        '制造业PMI': ak.index_pmi_man_cx,
        '服务业PMI': ak.index_pmi_ser_cx,
        '数字经济指数': ak.index_dei_cx,
        '产业指数': ak.index_ii_cx,
        '溢出指数': ak.index_si_cx,
        '融合指数': ak.index_fi_cx,
        '基础指数': ak.index_bi_cx,
        '中国新经济指数': ak.index_nei_cx,
        '劳动力投入指数': ak.index_li_cx,
        '资本投入指数': ak.index_ci_cx,
        '科技投入指数': ak.index_ti_cx,
        '新经济行业入职平均工资水平': ak.index_neaw_cx,
        '新经济入职工资溢价水平': ak.index_awpr_cx,
        '大宗商品指数': ak.index_cci_cx,
        '高质量因子': ak.index_qli_cx,
        'AI策略指数': ak.index_ai_cx,
        '基石经济指数': ak.index_bei_cx,
        '新动能指数': ak.index_neei_cx,
    }
    # 打印开始获取数据的消息
    print("开始获取PMI相关数据...")
    # 初始化最终数据列表
    final_df = list()
    # 设置最大重试次数为1次
    max_retries = 1

    # 遍历PMI数据字典，获取每个数据集
    for name, func in pmi_dict.items():
        # 打印当前获取的数据名称
        print(f"\t获取{name}数据...")
        # 尝试获取数据，允许失败5次
        for attempt in range(max_retries):
            try:
                # 调用函数获取利率数据
                sub_df = func().copy()

                # 统一日期字段名
                if '日期' in sub_df.columns:
                    sub_df.rename(columns={'日期': 'trade_date'}, inplace=True)
                elif 'date' in sub_df.columns:
                    sub_df.rename(columns={'date': 'trade_date'}, inplace=True)

                # 统一值字段名
                if 'wci' in sub_df.columns:
                    sub_df.rename(columns={'wci': 'close'}, inplace=True)
                elif ('综合PMI' or '制造业PMI' or '服务业PMI' or '数字经济指数' or '产业指数' or '溢出指数' or '融合指数'
                      or '基础指数' or '中国新经济指数' or '劳动力投入指数' or '资本投入指数' or '科技投入指数' or '大宗商品指数'
                      or '新经济行业入职平均工资水平' or '新经济入职工资溢价水平' or '高质量因子' or 'AI策略指数' or '基石经济指数'
                      or '新动能指数' in sub_df.columns):
                    sub_df.rename(columns={'综合PMI': 'close'}, inplace=True)
                    sub_df.rename(columns={'制造业PMI': 'close'}, inplace=True)
                    sub_df.rename(columns={'服务业PMI': 'close'}, inplace=True)
                    sub_df.rename(columns={'数字经济指数': 'close'}, inplace=True)
                    sub_df.rename(columns={'产业指数': 'close'}, inplace=True)
                    sub_df.rename(columns={'溢出指数': 'close'}, inplace=True)
                    sub_df.rename(columns={'融合指数': 'close'}, inplace=True)
                    sub_df.rename(columns={'基础指数': 'close'}, inplace=True)
                    sub_df.rename(columns={'中国新经济指数': 'close'}, inplace=True)
                    sub_df.rename(columns={'劳动力投入指数': 'close'}, inplace=True)
                    sub_df.rename(columns={'资本投入指数': 'close'}, inplace=True)
                    sub_df.rename(columns={'科技投入指数': 'close'}, inplace=True)
                    sub_df.rename(columns={'新经济行业入职平均工资水平': 'close'}, inplace=True)
                    sub_df.rename(columns={'新经济入职工资溢价水平': 'close'}, inplace=True)
                    sub_df.rename(columns={'大宗商品指数': 'close'}, inplace=True)
                    sub_df.rename(columns={'高质量因子指数': 'close'}, inplace=True)
                    sub_df.rename(columns={'AI策略指数': 'close'}, inplace=True)
                    sub_df.rename(columns={'基石经济指数': 'close'}, inplace=True)
                    sub_df.rename(columns={'新动能指数': 'close'}, inplace=True)

                # 将日期列转换为日期格式
                sub_df['trade_date'] = pd.to_datetime(sub_df['trade_date'])
                # 添加数据名称列
                sub_df['ts_code'] = name
                # 获取当前日期（以“天”为单位截断）
                today = pd.Timestamp.today().normalize()
                # 如果 DataFrame 中没有今天的数据，添加一行空值
                if today not in sub_df['trade_date'].values:
                    sub_df = pd.concat([
                        sub_df,
                        pd.DataFrame([{
                            'ts_code': name,
                            'trade_date': today,
                            'close': pd.NA
                        }])
                    ], ignore_index=True)
                # 仅保留商品、日期和今值列
                sub_df = sub_df[['ts_code', 'trade_date', 'close']]
                # 去除重复的行，保留最后一次出现的行
                sub_df = sub_df.drop_duplicates(subset=['trade_date', 'ts_code'], keep='last')
                sub_df.dropna(subset=['trade_date'], inplace=True)
                # 将数据透视，以便每个商品的今值成为列
                sub_df = sub_df.pivot(index='trade_date', columns='ts_code', values='close')
                # 按天重采样数据，并使用前向填充方法填充缺失值
                sub_df = sub_df.resample('D').asfreq()
                # 数据填充
                sub_df = sub_df.interpolate(method=fill_method)
                # 重塑数据格式，将透视表变回长格式
                sub_df = sub_df.reset_index().melt(id_vars='trade_date', var_name='ts_code', value_name='close')
                # 将处理后的数据添加到最终的数据列表中
                final_df.append(sub_df)
                # 如果成功获取数据，跳出循环
                break
            except Exception as e:
                # 如果发生异常且重试次数未用尽，则打印失败消息并重试
                if attempt < max_retries - 1:
                    print(f"\t\t获取 wci 数据失败，正在重试...")
                    time.sleep(1)  # 暂停1秒
                else:
                    # 如果重试次数用尽，打印最终失败消息
                    print(f"\t\t获取 wci 数据失败:{e}，已重试{max_retries}次，不再重试。")
                    print(traceback.format_exc())

    # 合并所有数据集
    final_df = pd.concat(final_df, ignore_index=True)
    # 打印合并后的数据
    print(final_df)
    # 构建文件路径
    file_path = os.path.join(save_path, parquet_name)
    # 保存数据为Parquet文件
    final_df.to_parquet(file_path, index=False)
    # 打印数据保存路径
    print('PMI指数相关数据获取完毕, 保存路径:', file_path)


def main():
    # 全球指数字典
    index_dict = {'MXX': '墨西哥BOLSA',
                  'JKSE': '印尼雅加达综合',
                  'ASE': '希腊雅典ASE',
                  'STI': '富时新加坡海峡时报',
                  'HSCCI': '红筹指数',
                  'SET': '泰国SET',
                  'WIG': '波兰WIG',
                  'HSI': '恒生指数',
                  'HSCEI': '国企指数',
                  'BFX': '比利时BFX',
                  'AXX': '富时AIM全股',
                  'RTS': '俄罗斯RTS',
                  'KSE100': '巴基斯坦卡拉奇',
                  '000001': '上证指数',
                  'SENSEX': '印度孟买SENSEX',
                  'IBEX': '西班牙IBEX35',
                  'PSI20': '葡萄牙PSI20',
                  'FTSE': '英国富时100',
                  'PSI': '菲律宾马尼拉',
                  'OMXSPI': '瑞典OMXSPI',
                  'UDI': '美元指数',
                  'HEX': '芬兰赫尔辛基',
                  'ISEQ': '爱尔兰综合',
                  '000300': '沪深300',
                  'AS51': '澳大利亚标普200',
                  'PX': '布拉格指数',
                  'KS11': '韩国KOSPI',
                  'ATX': '奥地利ATX',
                  'AORD': '澳大利亚普通股',
                  'N225': '日经225',
                  '399005': '中小100',
                  'MCX': '英国富时250',
                  'KOSPI200': '韩国KOSPI200',
                  'GDAXI': '德国DAX30',
                  'CSEALL': '斯里兰卡科伦坡',
                  'AEX': '荷兰AEX',
                  'SX5E': '欧洲斯托克50',
                  '399001': '深证成指',
                  'FCHI': '法国CAC40',
                  'MIB': '富时意大利MIB',
                  'OSEBX': '挪威OSEBX',
                  'TSX': '加拿大S&P/TSX',
                  '399006': '创业板指',
                  'SSMI': '瑞士SMI',
                  'VNINDEX': '越南胡志明',
                  'KLSE': '富时马来西亚KLCI',
                  'CRB': '路透CRB商品指数',
                  'TWII': '台湾加权',
                  'NZ50': '新西兰50',
                  'SPX': '标普500',
                  'DJIA': '道琼斯',
                  'NDX': '纳斯达克',
                  'OMXC20': 'OMX哥本哈根20'
                  }
    df = get_global_index_data(index_dict, ak.index_global_hist_em,
                               '../Data/Index', 'global_index_daily.parquet')
    # 人民币对全球其他货币汇率字典
    currency_map = {'100日元兑离岸人民币': 'JPYCNH', '纽元人民币中间价': 'NZDCNYC', '100日元人民币中间价': 'JPYCNYC',
                    '欧元人民币中间价': 'EURCNYC', '英镑人民币中间价': 'GBPCNYC', '瑞士法郎人民币中间价': 'CHFCNYC',
                    '澳元人民币中间价': 'AUDCNYC', '港币兑离岸人民币': 'HKDCNH', '人民币土耳其里拉中间价': 'CNYTRYC',
                    '人民币韩元中间价': 'CNYKRWC', '美元兑离岸人民币': 'USDCNH', '加元兑离岸人民币': 'CADCNH',
                    '新西兰元兑离岸人民币': 'NZDCNH', '英镑兑离岸人民币': 'GBPCNH', '新加坡元人民币中间价': 'SGDCNYC',
                    '欧元兑离岸人民币': 'EURCNH', '新加坡元兑离岸人民币': 'SGDCNH', '人民币墨西哥比索中间价': 'CNYMXNC',
                    '美元人民币中间价': 'USDCNYC', '港币人民币中间价': 'HKDCNYC', '人民币阿联酋迪拉姆中间价': 'CNYAEDC',
                    '澳元兑离岸人民币': 'AUDCNH', '人民币沙特里亚尔中间价': 'CNYSARC', '离岸人民币兑瑞士法郎': 'CNHCHF'
                    }
    df = get_global_index_data(currency_map, ak.forex_hist_em,
                               '../Data/Index', 'global_currency_daily.parquet')

    # 全球贵金属指数字典
    metal_code_dict = {
        "黄金Au99.99": "Au99.99",
        "黄金Au99.95": "Au99.95",
        "黄金Au100g": "Au100g",
        "铂金Pt99.95": "Pt99.95",
        "白银Ag(T+D)": "Ag(T+D)",
        "黄金Au(T+D)": "Au(T+D)",
        "迷你黄金mAu(T+D)": "mAu(T+D)",
        "黄金Au(T+N1)": "Au(T+N1)",
        "黄金Au(T+N2)": "Au(T+N2)",
        "白银Ag99.99": "Ag99.99",
        "国际黄金iAu99.99": "iAu99.99",
        "黄金Au99.5": "Au99.5",
        "国际黄金iAu100g": "iAu100g",
        "国际黄金iAu99.5": "iAu99.5",
        "个人账户金PGC30g": "PGC30g",
        "纽约金NYAuTN06": "NYAuTN06",
        "纽约金NYAuTN12": "NYAuTN12"
    }
    df = get_global_index_data(metal_code_dict, ak.spot_hist_sge,
                               '../Data/Index', 'global_metal_daily.parquet')

    # 全球商品现货价格指数
    spot_goods_dict = {
        "波罗的海干散货指数": "BDI",
        "钢坯价格指数": "GP",
        "澳大利亚粉矿价格": "PB",
    }
    df = get_global_index_data(spot_goods_dict, ak.spot_goods,
                               '../Data/Index', 'global_goods_daily.parquet')


if __name__ == '__main__':
    dd = get_china_pmi_data()
    print(dd)
    # index_ii_cx_df = ak.index_ii_cx()
    # print(index_ii_cx_df)
