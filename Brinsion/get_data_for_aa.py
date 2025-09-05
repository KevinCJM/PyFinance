# -*- encoding: utf-8 -*-
"""
@File: get_data_for_aa.py
@Modify Time: 2025/9/5 14:01       
@Author: Kevin-Chen
@Descriptions: 为基于公募基金的资产配置模型做数据准备
"""
import time
import pandas as pd
import akshare as ak
from tqdm import tqdm
from get_data_for_brinsion import fetch_index_daily_return
from concurrent.futures import ThreadPoolExecutor, as_completed

pd.set_option('display.max_columns', 1000)  # 显示字段的数量
pd.set_option('display.width', 1000)  # 表格不分段显示

csi_index_codes = {"930898": "中证可转债债券型基金指数",  # 债券类
                   "931153": "中证基金中基金指数",
                   "932047": "中证REITs全收益指数",  # 混合
                   "000300": "沪深300指数",  # 权益
                   "H11001": "中证全债指数",  # 债券类
                   "H11021": "中证股票型基金指数",  # 权益
                   "H11022": "中证混合型基金指数",  # 混合
                   "H11023": "中证债券型基金指数",  # 债券类
                   "H11025": "中证货币基金指数",  # 货币
                   "H11026": "中证QDII基金指数",  # 权益
                   "H30009": "中证商品期货成份指数",  # 商品
                   "H30072": "中证贵金属期货成份指数",  # 商品
                   }
index_weight = {
    "权益类": {"H11021": 0.3, "H11026": 0.3, "000300": 0.4},
    "债券类": {"930898": 0.3, "H11023": 0.3, "H11001": 0.4},
    "混合类": {"931153": 0.2, "H11023": 0.2, "H11022": 0.2, "H11021": 0.2, "932047": 0.2},
    "商品类": {"H30009": 0.5, "H30072": 0.5},
    "货币类": {"H11025": 1.0},
}


def main_fetch_index_daily_return(codes: list[str], start: str = "20230101", end: str = "20230905",
                                  max_workers: int = 10, max_retries: int = 5,
                                  retry_delay: float = 1.0) -> pd.DataFrame:
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_index_daily_return, code, start, end, max_retries, retry_delay): code
            for code in codes
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="fetch_index_daily_return"):
            results.append(fut.result())

    return pd.concat(results, ignore_index=True)


def sub_fetch_fund_info(symbol: str, max_retries=5, retry_delay=1.0):
    for attempt in range(1, max_retries + 1):
        try:
            sub_df = ak.stock_individual_info_em(symbol=symbol)
            if sub_df is None or sub_df.empty:
                raise ValueError("Empty response")
            sub_df = ak.fund_individual_basic_info_xq(symbol="000001")
            sub_df = sub_df.set_index("item").T
            sub_df.columns.name = None
            sub_df.reset_index(drop=True, inplace=True)
            return sub_df
        except Exception as e:
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                return pd.DataFrame()


def fetch_fund_basic_info(max_workers=10, max_retries=5, retry_delay=1.0):
    codes = ak.fund_name_em()['基金代码'].tolist()
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(sub_fetch_fund_info, code, max_retries, retry_delay): code
            for code in codes
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="fetch_fund_basic_info"):
            results.append(fut.result())

    return pd.concat(results, ignore_index=True)


if __name__ == '__main__':
    # index_daily = main_fetch_index_daily_return(list(csi_index_codes.keys()),
    #                                             start="20010101", end=pd.to_datetime('today').strftime('%Y%m%d'),
    #                                             max_workers=10, max_retries=5, retry_delay=1.0)
    # index_daily.to_parquet('data/index_daily_all.parquet', index=False)
    # print(index_daily.head())
    # # index_daily 的字段以及含义为:
    # _ = {'date': '交易日期',
    #      'index_code': '指数代码',
    #      'open': '开盘价',
    #      'close': '收盘价',
    #      'high': '最高价',
    #      'low': '最低价',
    #      'volume': '成交量',
    #      'amount': '成交金额',
    #      'change': '涨跌幅',
    #      'change_amount': '涨跌金额',
    #      'pe_ratio': '滚动市盈率'}
    # print(f"完成指数日行情数据获取, 共 {len(index_daily)} 条记录")

    fund_basic_info = fetch_fund_basic_info()
    fund_basic_info.to_parquet('data/fund_basic_info.parquet', index=False)
    print(fund_basic_info.head())
    print(f"完成基金基本信息数据获取, 共 {len(fund_basic_info)} 条记录")
