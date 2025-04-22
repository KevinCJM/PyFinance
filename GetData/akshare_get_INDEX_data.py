# -*- encoding: utf-8 -*-
"""
@File: tushare_get_INDEX_data.py
@Modify Time: 2025/4/22 15:45       
@Author: Kevin-Chen
@Descriptions: 用Tushare接口获取指数数据
"""

import gc
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
    final_df.rename(columns={'日期': 'trade_date', '代码': 'ts_code', '名称': 'name', '今开': 'open', '最新价': 'close',
                             '最高': 'high', '最低': 'low', '振幅': 'change'}, inplace=True)
    # 选择需要的列
    final_df = final_df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close']]
    # 打印合并后的数据
    print(final_df)
    # 保存数据到Parquet文件
    final_df.to_parquet(os.path.join(save_path, parquet_name), index=False)
    # 打印数据获取完成的消息和保存路径
    print('全球指数数据获取完成, 保存路径:', os.path.join(save_path, parquet_name))
    # 返回合并后的数据
    return final_df



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
    print(df)


if __name__ == '__main__':
    # 人民币对全球其他货币的汇率


    # forex_hist_em_df = ak.forex_hist_em(symbol="JPYCNH")
    # print(forex_hist_em_df)
    # reversed_currency_map = {v: k for k, v in currency_map.items()}
    # print(reversed_currency_map)
    df = get_global_index_data(currency_map, ak.forex_hist_em,
                               '../Data/Index', 'global_currency_daily.parquet')