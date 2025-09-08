# -*- encoding: utf-8 -*-
"""
@File: B01_category_brinsion_get_data.py
@Modify Time: 2025/9/8 10:51       
@Author: Kevin-Chen
@Descriptions: 
"""
import time
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from tushare_config import pro

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


if __name__ == '__main__':
    ''' 获取各个基金所属于大类 '''
    # fund_info_df = pro.fund_basic(market='O')  # E场内 O场外
    # fund_info_df['fund_code'] = fund_info_df['ts_code'].str[:6]
    # fund_info_df.rename(columns={'name': 'fund_name', 'management': 'fund_company'}, inplace=True)
    # fund_info_df = fund_info_df[['fund_code', 'fund_name', 'fund_type', 'fund_company']]
    # _ = {'fund_code': '基金代码',
    #      'fund_name': '基金名称',
    #      'fund_type': '基金类型',  # '债券型', '商品型', '混合型', '股票型', '货币市场型'
    #      'fund_company': '基金公司'
    #      }
    # fund_info_df.to_parquet('./data/fund_info.parquet', index=False)
    fund_info_df = pd.read_parquet('./data/fund_info.parquet')
    print(fund_info_df)

    ''' FOF基金的持仓数据 '''  # 示例数据, 权重和小于等于1
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

    ''' 基准的持仓数据 '''  # 示例数据, 权重和小于等于1
    benchmark_holding = {
        '000322': 0.11,  # 农银汇理金汇A (债券)
        '000674': 0.12,  # 中海中短债A (债券)
        '013166': 0.12,  # 东兴宸祥量化A (混合)
        '014600': 0.21,  # 博时回报严选A (混合)
        '019941': 0.11,  # 富国洞见价值A (股票)
        '020142': 0.22,  # 路博迈中国医疗健康A (股票)
        '018850': 0.05,  # 博时合晶货币A (货币)
        '970196': 0.06,  # 诚通天天利 (货币)
    }
    benchmark_holding_df = pd.DataFrame(benchmark_holding.items(), columns=['fund_code', 'weight'])
    _ = {'fund_code': '基金代码',
         'weight': '权重'
         }
    benchmark_holding_df.to_parquet('./data/benchmark_holding.parquet', index=False)
    print(benchmark_holding_df)

    ''' 获取基金的收益率数据 '''
    # # 取FOF基金与基准的持仓并集
    # fund_codes = list(set(fof_holding.keys()).union(set(benchmark_holding.keys())))
    # print(f"FOF基金与基准的持仓并集共{len(fund_codes)}只基金")
    # # 获取基金每日净值数据
    # fund_return_df = fetch_fund_daily_return_parallel(fund_codes)
    # _ = {'fund_code': '基金代码',
    #      'date': '净值日期',
    #      'unit_nav': '单位净值',
    #      'accum_nav': '累计净值',
    #      'adj_nav': '复权净值'
    #      }
    # fund_return_df.to_parquet('./data/fund_daily_return.parquet', index=False)
    fund_return_df = pd.read_parquet('./data/fund_daily_return.parquet')
    print(fund_return_df)
