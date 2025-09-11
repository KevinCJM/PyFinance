# -*- encoding: utf-8 -*-
"""
@File: B01_category_brinsion_get_data.py
@Modify Time: 2025/9/8 10:51       
@Author: Kevin-Chen
@Descriptions: 大类Brinsion归因, 取数
"""
import time
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from tushare_config import pro
from A01_equity_brinsion_get_data import fetch_index_daily_return

pd.set_option('display.max_columns', 1000)  # 显示字段的数量
pd.set_option('display.width', 1000)  # 表格不分段显示


# 获取基金每日净值数据
def sub_fetch_fund_daily_return(symbol: str, max_retries: int = 5, retry_delay: float = 1.0) -> pd.DataFrame:
    for attempt in range(1, max_retries + 1):
        try:
            sub_df = pro.fund_nav(ts_code=f'{symbol}.OF')
            sub_df.rename(columns={'ts_code': 'fund_code', 'nav_date': 'date'}, inplace=True)
            sub_df = sub_df[['fund_code', 'date', 'unit_nav', 'accum_nav', 'adj_nav']]
            sub_df['fund_code'] = sub_df['fund_code'].str[:6]
            sub_df['date'] = pd.to_datetime(sub_df['date'], format='%Y%m%d')
            return sub_df
        except Exception as e:
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                # 超过重试次数，返回一个空 DataFrame，避免中断
                print(f"{symbol}基金的每日净值数据获取失败")
                return pd.DataFrame()


def fetch_fund_daily_return_parallel(codes: list[str], max_workers: int = 10, max_retries: int = 5,
                                     retry_delay: float = 1.0) -> pd.DataFrame:
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(sub_fetch_fund_daily_return, code, max_retries, retry_delay): code
            for code in codes
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="fetch_fund_daily_returns"):
            results.append(fut.result())

    return pd.concat(results, ignore_index=True).reset_index(drop=True) if results else pd.DataFrame()


# 生成按自然年拆分的区间
def split_years(start: str, end: str) -> list[tuple[str, str]]:
    start_dt = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")
    years = range(start_dt.year, end_dt.year + 1)

    intervals = []
    for y in years:
        y_start = datetime(y, 1, 1)
        y_end = datetime(y, 12, 31)
        if y == start_dt.year:
            y_start = start_dt
        if y == end_dt.year:
            y_end = end_dt
        intervals.append((y_start.strftime("%Y%m%d"), y_end.strftime("%Y%m%d")))
    return intervals


def main_fetch_index_daily_return(
        codes: list[str],
        start: str = "20230101",
        end: str = "20230905",
        max_workers: int = 10,
        max_retries: int = 5,
        retry_delay: float = 1.0
) -> pd.DataFrame:
    results = []
    intervals = split_years(start, end)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_index_daily_return, code, s, e, max_retries, retry_delay): (code, s, e)
            for code in codes
            for s, e in intervals
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="fetch_index_daily_return"):
            results.append(fut.result())

    return pd.concat(results, ignore_index=True)


if __name__ == '__main__':
    ''' 获取各个基金所属于大类 '''
    fund_info_df = pro.fund_basic(market='O')  # E场内 O场外
    fund_info_df['fund_code'] = fund_info_df['ts_code'].str[:6]
    fund_info_df.rename(columns={'name': 'fund_name', 'management': 'fund_company'}, inplace=True)
    fund_info_df = fund_info_df[['fund_code', 'fund_name', 'fund_type', 'fund_company']]
    _ = {'fund_code': '基金代码',
         'fund_name': '基金名称',
         'fund_type': '基金类型',  # '债券型', '商品型', '混合型', '股票型', '货币市场型'
         'fund_company': '基金公司'
         }
    fund_info_df.to_parquet('./data/fund_info.parquet', index=False)
    fund_info_df = pd.read_parquet('./data/fund_info.parquet')
    print(fund_info_df)
    print(set(fund_info_df['fund_type']))

    ''' FOF基金的持仓数据 '''
    # 示例数据, 权重和小于等于1
    fof_holding = {
        '217025': 0.12,  # 招商理财7天A (债券型)
        '018594': 0.07,  # 格林泓盈利率债 (债券型)
        '008930': 0.21,  # 中加安瑞平衡养老目标三年 (混合)
        '014008': 0.10,  # 华安制造升级一年持有期C (混合)
        '023863': 0.14,  # 东财中证A500联接A (股票型)
        '022908': 0.11,  # 国投瑞银沪深300量化增强Y (股票型)
        '017939': 0.15,  # 上银慧增利A (货币型)
    }
    fof_holding_df = pd.DataFrame(fof_holding.items(), columns=['fund_code', 'weight'])
    _ = {'fund_code': '基金代码',
         'weight': '权重'
         }
    fof_holding_df.to_parquet('./data/fof_holding.parquet', index=False)
    print(fof_holding_df)

    ''' 获取基金的收益率数据 '''
    # 取FOF基金与基准的持仓并集
    fund_codes = list(set(fof_holding.keys()))
    print(f"FOF基金与基准的持仓并集共{len(fund_codes)}只基金")
    # 获取基金每日净值数据
    fund_return_df = fetch_fund_daily_return_parallel(fund_codes)
    _ = {'fund_code': '基金代码',
         'date': '净值日期',
         'unit_nav': '单位净值',
         'accum_nav': '累计净值',
         'adj_nav': '复权净值'
         }
    fund_return_df.to_parquet('./data/fund_daily_return.parquet', index=False)
    fund_return_df = pd.read_parquet('./data/fund_daily_return.parquet')
    print(fund_return_df)

    ''' 基准的指数比例数据 '''
    # 示例数据, 权重和等于1, 基准使用指数构成
    benchmark_holding = {
        "H11021": 0.1,  # 中证股票型基金指数 -  权益
        "H11022": 0.2,  # 中证混合型基金指数 - 混合
        "H11023": 0.3,  # 中证债券型基金指数 - 债券
        "H11025": 0.2,  # 中证货币基金指数 - 货币
        "H30009": 0.2,  # 中证商品期货成份指数 - 商品
    }
    benchmark_holding_df = pd.DataFrame(benchmark_holding.items(), columns=['index_code', 'weight'])
    _ = {'index_code': '指数代码',
         'weight': '权重'
         }
    benchmark_holding_df.to_parquet('./data/benchmark_holding.parquet', index=False)
    print(benchmark_holding_df)

    ''' 指数对应大类 '''
    csi_index_codes = {
        "H11021": "股票型",
        "H11022": "混合型",
        "H11023": "债券型",
        "H11025": "货币市场型",
        "H30009": "商品型",
    }
    csi_index_df = pd.DataFrame(csi_index_codes.items(), columns=['index_code', 'index_type'])
    _ = {'index_code': '指数代码',
         'fund_type': '指数大类'
         }
    csi_index_df.to_parquet('./data/csi_index_type.parquet', index=False)
    print(csi_index_df)

    ''' 获取指数的收益率数据 '''
    index_daily = main_fetch_index_daily_return(list(csi_index_codes.keys()),
                                                start="20010101", end=pd.to_datetime('today').strftime('%Y%m%d'),
                                                max_workers=10, max_retries=5, retry_delay=1.0)
    # index_daily 的字段以及含义为:
    _ = {'date': '交易日期',
         'index_code': '指数代码',
         'open': '开盘价',
         'close': '收盘价',
         'high': '最高价',
         'low': '最低价',
         'volume': '成交量',
         'amount': '成交金额',
         'change': '涨跌幅',
         'change_amount': '涨跌金额',
         'pe_ratio': '滚动市盈率'}
    index_daily.to_parquet('data/index_daily_all.parquet', index=False)
    index_daily = pd.read_parquet('data/index_daily_all.parquet')
    print(index_daily.head())
    print(f"完成指数日行情数据获取, 共 {len(index_daily)} 条记录")
