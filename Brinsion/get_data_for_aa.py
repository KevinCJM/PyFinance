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

from tushare_config import pro
from concurrent.futures import ThreadPoolExecutor, as_completed
from A01_equity_brinsion_get_data import fetch_index_daily_return

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


# 多线程获取指数日行情数据
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


# 获取公募基金基本信息
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


# 多线程获取公募基金基本信息
def fetch_fund_basic_info(max_workers=50, max_retries=5, retry_delay=1.0):
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
    ''' 获取指数日行情数据 '''
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

    ''' 获取公募基金基本信息 (AKshare) '''
    # fund_basic_info = fetch_fund_basic_info()
    # fund_basic_info.to_parquet('data/fund_basic_info.parquet', index=False)
    # print(fund_basic_info.head())
    # print(f"完成基金基本信息数据获取, 共 {len(fund_basic_info)} 条记录")

    ''' 获取公募基金基本信息 (Tushare) '''
    df = pro.fund_basic(market='E')   # E场内 O场外
    df.to_excel('data/fund_basic_info_E.xlsx')
    df.to_parquet('data/fund_basic_info_E.parquet', index=False)
    print(df.head())

    # ''' 获取公募基金净值数据 '''
    # fund_codes = pd.read_excel('fund_basic_info.xlsx')[['基金代码', '基金类型']].astype(str)
    # # 将基金代码补充至6位
    # fund_codes['基金代码'] = fund_codes['基金代码'].apply(lambda x: x.zfill(6))
    # # 将基金类型根据'-'分为一级分类和二级分类
    # fund_codes[['一级分类', '二级分类']] = fund_codes['基金类型'].str.split('-', expand=True)
    # fund_codes.rename(columns={'基金代码': 'fund_code', '基金类型': 'fund_type',
    #                            '一级分类': 'fund_type_1', '二级分类': 'fund_type_2'}, inplace=True)
    # # df = pro.fund_nav(ts_code='000001.OF')
    # # print(df)
    # print(fund_codes)
