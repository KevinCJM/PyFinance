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
    max_retries = 5

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
    return final_df


# 获取中国国证指数数据并保存为Parquet文件
def get_china_gzindex_data(save_path='../Data/Index', parquet_name='china_gzindex_daily.parquet'):
    """
    获取国证指数数据并保存为Parquet文件。

    本函数从指定的起始日期到今天，获取一系列国证指数的历史数据。
    数据获取失败时，允许重试多次，最后将数据保存到指定路径的Parquet文件中。

    参数:
    save_path (str): 数据保存的目录路径，默认为'../Data/Index'。
    parquet_name (str): 保存的Parquet文件名称，默认为'china_gzindex_daily.parquet'。

    返回:
    DataFrame: 获取到的国证指数数据。
    """
    gz_index_dict = {'399002': '深成指R', '399004': '深证100R', '39926401': '创业软件R', '399333': '中小100R',
                     '399344': '深证300R', '39936501': '国证粮食R', '39943901': '国证油气R', '399606': '创业板R',
                     '399611': '中创100R', '399679': '深证200R', '970046': '深证XR', '470026': '深指ESGR',
                     '470027': '100ESGR', '470028': '创指ESGR', '470029': '深指ESG领先R', '470030': '100ESG领先R',
                     '470031': '创指ESG领先R', '470032': '深指ESG增强R', '470033': '100ESG增强R',
                     '470034': '创指ESG增强R', '470035': '深证光伏R', '470036': '深证储能R', '470037': '深证高装R',
                     '470038': '生物经济R', '470039': '创新材料R', '470040': '创业高装R', '470041': '创新消费R',
                     '470042': '新材料50R', '470043': '新能装备R', '470046': '深证XR R', '470047': '数字文化R',
                     '470048': '新型显示R', '470049': '深证数字交通R', '470050': '深证卫星导航R',
                     '470051': '深证算力设施R', '470052': '深证数字安全R', '470053': '深证数据要素R',
                     '470054': '深证智能电网R', '470055': '深证国企ESGR', '470056': '深证民企ESGR',
                     '470057': '深证民企50R', '470058': '深证民企治理R', '470059': '深证民企科技R',
                     '470060': '深证ESG成长R', '470061': '深证ESG价值R', '470062': '深红利300R', '470063': '深证回购R',
                     '470064': '深证国企股东回报R', '470065': '深证优质信披R', '470066': '深证氢能R',
                     '470067': '深证专利R', '470068': '创业专利R', '470069': '深证绿色制造R',
                     '470070': '创业板人工智能R', '470071': '深证民企成长R', '470072': '深证民企价值R',
                     '470073': '深证绿色农牧R', '470074': '深证智能穿戴R', '470075': '深证AIGCR',
                     '470076': '深证绿色化工R', '470077': '深证绿色建材R', '470080': '科技50R', '471001': '深成地产R',
                     '471002': '中小地产R', '471004': '创业板战略科技R', '480015': '疫苗生科R', '480016': '医疗健康R',
                     '480017': '国证芯片R', '480018': '卫星通信R', '480021': '碳中和50R', '480026': '风光装备R',
                     '480027': '新能源电池R', '480028': '龙头家电R', '480030': '消费电子R', '480032': '新能电池R',
                     '480033': '自主科技R', '480034': '工业软件R', '480035': '化肥农药R', '480043': '工业金属R',
                     '480048': '国证建材R', '480055': '规模因子R', '480056': '价值因子R', '480057': '动量因子R',
                     '480058': '绿色治理R', '480059': '航空出行R', '480063': '国证回购R', '480065': '帮扶100R',
                     '480070': '国证专利R', '480071': '机器人龙头R', '480072': '生物农业R', '480073': '生猪指数R',
                     '480075': '数字交通R', '480076': '通用航空R', '480077': '充电设施R', '480078': 'ESG基准R',
                     '480079': 'ESG领先R', '480080': '成长100R', '480081': '价值100R', '480082': '央国企低碳科技R',
                     '480083': '央国企自主科技R', '480092': '自由现金流R', '480150': '单项冠军R',
                     '480532': '国证青岛海洋R', '98002201': '机器人产业R', 'AITCNYG': '中华陆股通行业龙头R',
                     'CN2008': '中小300R', 'CN2010': '深证700R', 'CN2011': '深证1000R', 'CN2012': '创业300R',
                     'CN2013': '深市精选R', 'CN2015': '中小创新R', 'CN2016': '深证创新R', 'CN2017': 'SME创新R',
                     'CN2018': '创业创新R', 'CN2019': '创业200R', 'CN2020': '创业小盘R', 'CN2030': '碳科技30R',
                     'CN2050': '创新引擎R', 'CN2060': '碳科技60R', 'CN2088': '深创100R', 'CN2258': '绿色低碳R',
                     'CN2259': '创业低碳R', 'CN2260': '先进制造R', 'CN2261': '创业制造R', 'CN2262': '数字经济R',
                     'CN2263': '创业数字R', 'CN2265': '创新药械R', 'CN2266': '创新能源R', 'CN2269': '创质量R',
                     'CN2274': '深新基建R', 'CN2275': '创医药R', 'CN2276': '创科技R', 'CN2277': '公共健康R',
                     'CN2278': '长江100R', 'CN2279': '云科技50R', 'CN2280': '生物50R', 'CN2281': '电子50R',
                     'CN2282': '大数据50R', 'CN2283': '机器人50R', 'CN2284': 'AI 50R', 'CN2285': '物联网50R',
                     'CN2286': '区块链50R', 'CN2291': '创精选88R', 'CN2292': '民企发展R', 'CN2293': '创业大盘R',
                     'CN2294': '中小创Q R', 'CN2295': '创价值R', 'CN2296': '创成长R', 'CN2297': '新浪100R',
                     'CN2303': '国证2000R', 'CN2311': '国证1000R', 'CN2312': '国证300R', 'CN2313': '巨潮100R',
                     'CN2314': '巨潮大盘R', 'CN2315': '巨潮中盘R', 'CN2316': '巨潮小盘R', 'CN2319': '资源优势R',
                     'CN2320': '国证服务R', 'CN2321': '国证红利R', 'CN2322': '国证治理R', 'CN2324': '深证红利R',
                     'CN2326': '成长40R', 'CN2328': '深证治理R', 'CN2335': '深证央企R', 'CN2337': '深证民营R',
                     'CN2339': '深证科技R', 'CN2341': '深证责任R', 'CN2346': '深证成长R', 'CN2348': '深证价值R',
                     'CN2350': '皖江30R', 'CN2351': '创新示范R', 'CN2353': '国证物流R', 'CN2354': '分析师指R',
                     'CN2355': '长三角R', 'CN2356': '珠三角R', 'CN2357': '环渤海R', 'CN2358': '国证环保R',
                     'CN2360': '新硬件R', 'CN2361': '在线消费R', 'CN2362': '民企100R', 'CN2363': '国证算力R',
                     'CN2366': '能源金属R', 'CN2367': '1000地产R', 'CN2368': '国证军工R', 'CN2369': '国证责任R',
                     'CN2370': '国证成长R', 'CN2371': '国证价值R', 'CN2372': '大盘成长R', 'CN2373': '大盘价值R',
                     'CN2374': '中盘成长R', 'CN2375': '中盘价值R', 'CN2376': '小盘成长R', 'CN2377': '小盘价值R',
                     'CN2378': 'ESG 300R', 'CN2381': '1000能源R', 'CN2382': '1000材料R', 'CN2383': '1000工业R',
                     'CN2384': '1000可选R', 'CN2385': '1000消费R', 'CN2386': '1000医药R', 'CN2387': '1000金融R',
                     'CN2388': '1000信息R', 'CN2389': '国证通信R', 'CN2390': '1000公用R', 'CN2391': '投资时钟R',
                     'CN2392': '国证新兴R', 'CN2393': '国证地产R', 'CN2394': '国证医药R', 'CN2395': '国证有色R',
                     'CN2396': '国证食品R', 'CN2397': '国证文化R', 'CN2398': '绩效指数R', 'CN2399': '中经GDPR',
                     'CN2400': '大中盘R', 'CN2401': '中小盘R', 'CN2402': '周期100R', 'CN2403': '防御100R',
                     'CN2404': '大盘低波R', 'CN2405': '大盘高贝R', 'CN2406': '中盘低波R', 'CN2407': '中盘高贝R',
                     'CN2408': '小盘低波R', 'CN2409': '小盘高贝R', 'CN2410': '苏州率先R', 'CN2411': '红利100R',
                     'CN2412': '国证新能R', 'CN2415': 'I100R', 'CN2416': 'I300R', 'CN2417': '新能源车R',
                     'CN2418': '数据要素R', 'CN2419': '国证高铁R', 'CN2420': '国证保证R', 'CN2422': '中关村AR',
                     'CN2423': '中关村50R', 'CN2428': '国证定增R', 'CN2429': '新丝路R', 'CN2431': '国证银行R',
                     'CN2432': '智能汽车R', 'CN2433': '国证交运R', 'CN2434': '数字传媒R', 'CN2435': '国证农牧R',
                     'CN2436': '绿色煤炭R', 'CN2437': '证券龙头R', 'CN2438': '绿色电力R', 'CN2440': '国证钢铁R',
                     'CN2441': '生物医药R', 'CN2550': '央视50R', 'CN2551': '央视创新R', 'CN2552': '央视成长R',
                     'CN2553': '央视回报R', 'CN2554': '央视治理R', 'CN2555': '央视责任R', 'CN2556': '央视生态R',
                     'CN2557': '央视文化R', 'CN2602': '中小成长R', 'CN2604': '中小价值R', 'CN2608': '科技100R',
                     'CN2610': 'TMT 50R', 'CN2613': '深证能源R', 'CN2614': '深证材料R', 'CN2615': '深证工业R',
                     'CN2616': '深证可选R', 'CN2617': '深证消费R', 'CN2618': '深证医药R', 'CN2619': '深证金融R',
                     'CN2620': '深证信息R', 'CN2621': '深证电信R', 'CN2622': '深证公用R', 'CN2623': '中小基础R',
                     'CN2624': '中创400R', 'CN2625': '中创500R', 'CN2626': '中创成长R', 'CN2627': '中创价值R',
                     'CN2628': '700成长R', 'CN2629': '700价值R', 'CN2630': '1000成长R', 'CN2631': '1000价值R',
                     'CN2632': '深100EWR', 'CN2633': '深300EWR', 'CN2634': '中小等权R', 'CN2635': '创业板EWR',
                     'CN2636': '深证装备R', 'CN2637': '深证地产R', 'CN2638': '深证环保R', 'CN2639': '深证大宗R',
                     'CN2640': '创业基础R', 'CN2641': '深证新兴R', 'CN2642': '中小新兴R', 'CN2643': '创业新兴R',
                     'CN2644': '深证时钟R', 'CN2645': '100低波R', 'CN2646': '深消费50R', 'CN2647': '深医药50R',
                     'CN2648': '深证GDPR', 'CN2649': '中小红利R', 'CN2650': '中小治理R', 'CN2651': '中小责任R',
                     'CN2652': '中创高新R', 'CN2653': '深证龙头R', 'CN2654': '深证文化R', 'CN2655': '深证绩效R',
                     'CN2656': '100绩效R', 'CN2657': '300绩效R', 'CN2658': '中小绩效R', 'CN2659': '深成指EWR',
                     'CN2660': '中创EWR', 'CN2661': '深证低波R', 'CN2662': '深证高贝R', 'CN2663': '中小低波R',
                     'CN2664': '中小高贝R', 'CN2665': '中创低波R', 'CN2666': '中创高贝R', 'CN2667': '创业成长R',
                     'CN2668': '创业板VR', 'CN2669': '深证农业R', 'CN2670': '深周期50R', 'CN2671': '深防御50R',
                     'CN2672': '深红利50R', 'CN2673': '创业板50R', 'CN2674': '深A医药R', 'CN2675': '深互联网R',
                     'CN2676': '深医药EWR', 'CN2677': '深互联EWR', 'CN2678': '深次新股R', 'CN2680': '深成能源R',
                     'CN2681': '深成材料R', 'CN2682': '深成工业R', 'CN2683': '深成可选R', 'CN2684': '深成消费R',
                     'CN2685': '深成医药R', 'CN2686': '深成金融R', 'CN2687': '深成信息R', 'CN2688': '深成电信R',
                     'CN2689': '深成公用R', 'CN2692': '创业低波R', 'CN2693': '安防产业R', 'CN2694': '创业高贝R',
                     'CN2695': '深证节能R', 'CN2696': '深证创投R', 'CN2697': '中关村60R', 'CN2698': '优势成长R',
                     'CN2699': '金融科技R', 'CN2750': '深主板50R', 'CN2850': '深证50R', 'CN6074': '财富管理R',
                     'CN6075': '国证信创R', 'CN6079': '国证中药R', 'CN6082': '国证航天R', 'CN6084': '电信服务R',
                     'CN6085': '服装纺织R', 'CN6086': '工业服务R', 'CN6090': '国证化工R', 'CN6093': '国证零售R',
                     'CN6094': '耐用消费R', 'CN6097': '消费服务R', 'CN6100': '白色家电R', 'CN6101': '国证百货R',
                     'CN6105': '路桥港口R', 'CN6106': '酒店餐饮R', 'CN6111': '定增综指R', 'CN6122': '中小能源R',
                     'CN6123': '中小材料R', 'CN6124': '中小工业R', 'CN6125': '中小可选R', 'CN6126': '中小消费R',
                     'CN6127': '中小医药R', 'CN6128': '中小金融R', 'CN6129': '中小信息R', 'CN6130': '中小电信R',
                     'CN6131': '中小公用R', 'CN6138': '大盘红利R', 'CN6139': '稳进价值R', 'CN6140': '先锋成长R',
                     'CN6141': '优势价值R', 'FACCNYG': '中华A股外资优先配置R', 'IBTCNYG': '中华创新医药R'
                     }
    # 设置最大重试次数
    max_retries = 5
    # 获取今天的日期，格式为YYYYMMDD
    today_str = datetime.today().strftime('%Y%m%d')
    # 初始化一个空列表，用于存储所有指数的数据
    final_df = list()
    # 打印获取数据开始的提示信息
    print(f"获取国证指数数据...")

    # 遍历国证指数字典，获取每个指数的历史数据
    for symbol, name in gz_index_dict.items():
        # 打印当前指数的名称
        print(f"\t获取{name}数据...")
        # 尝试获取数据，允许失败5次
        for attempt in range(max_retries):
            try:
                # 使用akshare获取指数历史数据
                gz_index_df = ak.index_hist_cni(symbol=symbol, start_date="20050101", end_date=today_str)
                # 重命名列名，以便后续处理
                gz_index_df.rename(columns={'日期': 'trade_date', '收盘价': 'close', '开盘价': 'open', '最高价': 'high',
                                            '最低价': 'low', '成交量': 'vol', '成交额': 'amount'}, inplace=True)
                gz_index_df = gz_index_df[['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']]
                # 将当前指数的数据添加到列表中
                final_df.append(gz_index_df)
                # 如果数据获取成功，跳出循环
                break
            except Exception as e:
                # 如果已经达到最大重试次数，打印错误信息并跳出循环
                if attempt == max_retries - 1:
                    print(f"获取{name}数据失败: {e}")
                    break
                else:
                    # 如果未达到最大重试次数，打印重试信息并等待1秒后重试
                    print(f"获取{name}数据失败，正在重试...")
                    time.sleep(1)

    # 将列表中的所有DataFrame合并为一个，忽略索引
    final_df = pd.concat(final_df, ignore_index=True)
    # 将日期列转换为datetime类型
    final_df['trade_date'] = pd.to_datetime(final_df['trade_date'])
    # 打印合并后的数据
    print(final_df)
    # 构造保存文件的完整路径
    file_path = os.path.join(save_path, parquet_name)
    # 将数据保存为Parquet文件
    final_df.to_parquet(file_path)
    # 打印数据获取和保存完成的提示信息
    print('国证指数数据获取完毕, 保存路径:', file_path)
    # 返回获取到的数据
    return final_df


def akshare_index_main(save_path='../Data/Index'):
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
    get_global_index_data(index_dict, ak.index_global_hist_em,
                          save_path, 'global_index_daily.parquet')
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
    get_global_index_data(currency_map, ak.forex_hist_em,
                          save_path, 'global_currency_daily.parquet')

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
    get_global_index_data(metal_code_dict, ak.spot_hist_sge,
                          save_path, 'global_metal_daily.parquet')

    # 全球商品现货价格指数
    spot_goods_dict = {
        "波罗的海干散货指数": "BDI",
        "钢坯价格指数": "GP",
        "澳大利亚粉矿价格": "PB",
    }
    get_global_index_data(spot_goods_dict, ak.spot_goods,
                          save_path, 'global_goods_daily.parquet')

    # 全球利率数据
    get_global_interest_rate_data(save_path, 'global_interest_daily.parquet')

    # 获取中国VIX数据
    get_china_vix_data(save_path, 'china_vix_daily.parquet')

    # 获取中国银行间同业拆借利率数据
    get_china_interbank_interest_rate_data(save_path, 'china_interbank_daily.parquet')

    # 获取中国PMI数据
    get_china_pmi_data(save_path, 'china_pmi_daily.parquet')

    # 获取中国国证指数数据
    get_china_gzindex_data(save_path, 'china_gzindex_daily.parquet')


if __name__ == '__main__':
    get_china_gzindex_data()
