# -*- encoding: utf-8 -*-
"""
@File: A01_get_data.py
@Modify Time: 2025/8/19 17:07       
@Author: Kevin-Chen
@Descriptions: 
"""

import akshare as ak
import pandas as pd
from pathlib import Path

# 展示df的所有列
pd.set_option('display.max_columns', None)
# df展示不换行
pd.set_option('display.width', 1000)


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_stock_data(code, s_date='20000101', e_date=pd.to_datetime("today").strftime("%Y%m%d")):
    """
    获取股票的日线数据

    :param code: 股票代码
    :return: 包含股票日线数据的 DataFrame
    """
    ''' 获取个股日频行情数据 '''
    stock_df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=s_date,
        end_date=e_date,
        adjust="qfq"
    )
    out_path = DATA_DIR / f"{code}.parquet"
    stock_df.to_parquet(out_path, index=False)
    return stock_df


def get_stock_info(code):
    """
    获取股票的简介信息

    :param code: 股票代码
    :return: 包含股票简介信息的 DataFrame
    """
    ''' 获取个股简介信息 '''
    stock_info_df = ak.stock_individual_info_em(symbol=code)
    return stock_info_df


if __name__ == '__main__':
    code_list = ['300008']

    for the_code in code_list:
        df_1 = get_stock_data(the_code)
        df_2 = get_stock_info(the_code)
        print(df_1.tail())
        print(df_2)
