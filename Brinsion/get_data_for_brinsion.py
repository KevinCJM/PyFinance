# -*- encoding: utf-8 -*-
"""
@File: get_data_for_brinsion.py
@Modify Time: 2025/9/4 15:26       
@Author: Kevin-Chen
@Descriptions: 取数
"""
import re
import time
import numpy as np
import pandas as pd
import akshare as ak
from tqdm import tqdm
from tushare_config import pro
from concurrent.futures import ThreadPoolExecutor, as_completed

pd.set_option('display.max_columns', 1000)  # 显示字段的数量
pd.set_option('display.width', 1000)  # 表格不分段显示


# 1) 工具：常用清洗函数
def _norm_code(code: str) -> str:
    """统一股票代码格式为6位数字字符串"""
    code = str(code)
    return code.zfill(6)


def _to_pct(x):
    """百分比字段转小数（若已是小数直接返回）"""
    try:
        x = float(str(x).replace('%', '').replace(',', ''))
        return x / 100 if x > 1 else x
    except Exception as e:
        print(f"[字段转换错误] {x}, {e}")
        return np.nan


def _check_nonempty(in_df: pd.DataFrame, name: str):
    if in_df is None or len(in_df) == 0:
        raise ValueError(f"[数据缺失] {name}")


# 2) 基金持仓（季报/半年报）
def _quarter_to_date(s: str) -> str:
    """将 '2024年1季度股票投资明细' 转换为 '20240331'"""
    m = re.match(r"(\d{4})年(\d)季度股票投资明细", str(s))
    if not m:
        return ''
    year, q = int(m.group(1)), int(m.group(2))
    quarter_end = {1: "0331", 2: "0630", 3: "0930", 4: "1231"}
    return f"{year}{quarter_end[q]}"


def _fetch_fund_single(fund_code: str, report_year: str, get_last_data: bool,
                       max_retries: int = 5, retry_delay: float = 1.0) -> pd.DataFrame:
    """
    单基金持仓获取（带重试）
    """
    for attempt in range(1, max_retries + 1):
        try:
            year = report_year[:4]
            df = ak.fund_portfolio_hold_em(symbol=fund_code, date=year)
            if df is None or df.empty:
                raise ValueError("Empty response")

            df['report_date'] = df['季度'].apply(_quarter_to_date)
            df['report_date'] = pd.to_datetime(df['report_date'], format='%Y%m%d', errors="coerce")

            if get_last_data:  # 取最新季度
                df = df[df["report_date"] == df["report_date"].max()]

            df['fund_code'] = fund_code
            df.rename(columns={"股票代码": "stock_code", "股票名称": "stock_name", "占净值比例": "weight",
                               "持股数": "num", "持仓市值": "market_value"}, inplace=True)

            # 权重归一
            df["weight"] = df["weight"] / df["weight"].sum()

            # 清理
            df = df.drop_duplicates(subset=["stock_code", "report_date"], keep="first")
            df = df.dropna(subset=["stock_code", "weight"])
            df = df[df['weight'] > 0.0]

            df = df.reset_index(drop=True)
            return df[['fund_code', 'report_date', 'stock_code', 'stock_name', 'weight', 'num', 'market_value']]

        except Exception as e:
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                # 超过重试次数，返回一个空 DataFrame，避免中断
                return pd.DataFrame(
                    columns=['fund_code', 'report_date', 'stock_code', 'stock_name', 'weight', 'num', 'market_value'])


def fetch_fund_holdings_multi(fund_codes: list[str], report_year: str = '2024',
                              get_last_data: bool = True, max_workers: int = 8,
                              max_retries: int = 5, retry_delay: float = 1.0) -> pd.DataFrame:
    """
    多基金持仓并发获取 + 合并

    Parameters
    ----------
    fund_codes : list[str]
        公募基金代码列表
    report_year : str
        年份，如 '2024'
    get_last_data : bool
        是否只取最新季度
    max_workers : int
        并发线程数
    max_retries : int
        最大重试次数
    retry_delay : float
        重试间隔秒数

    Returns
    -------
    pd.DataFrame
        所有基金持仓合并结果
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_fund_single, code, report_year, get_last_data, max_retries, retry_delay): code
            for code in fund_codes
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="fetch_fund_holdings"):
            results.append(fut.result())

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# 3) 指数成分及权重（CSIndex 中证指数）
def fetch_index_stock_cons_weight_index(index_code: str = '000300') -> pd.DataFrame:
    """
    index_code: 指数代码，如 '000300' 沪深300
    """
    index_hold_df = ak.index_stock_cons_weight_csindex(symbol=index_code)
    _check_nonempty(index_hold_df, f"指数权重 index={index_code}")

    index_hold_df.rename(columns={"指数代码": "index_code", "指数名称": "index_name", '成分券代码': "stock_code",
                                  '成分券名称': "stock_name", '权重': "weight", "日期": "report_date"}, inplace=True)
    index_hold_df['report_date'] = pd.to_datetime(index_hold_df['report_date'], format='%Y%m%d')
    index_hold_df = index_hold_df[index_hold_df["report_date"] == index_hold_df["report_date"].max()]

    # 归一
    index_hold_df["weight"] = index_hold_df["weight"] / index_hold_df["weight"].sum()
    # 清理
    index_hold_df.drop_duplicates(subset=["stock_code", "report_date"], keep='first', inplace=True)
    index_hold_df = index_hold_df.dropna(subset=["stock_code", "weight"])
    index_hold_df = index_hold_df[index_hold_df['weight'] > 0.0]
    index_hold_df = index_hold_df.reset_index(drop=True)
    # 规范列
    index_hold_df = index_hold_df[['index_code', 'index_name', 'report_date', 'stock_code', 'stock_name', 'weight']]
    return index_hold_df


# 4) 个股 → 行业映射（东财行业）
def sub_fetch_stock_info(symbol: str, max_retries: int = 5, retry_delay: float = 1.0) -> pd.DataFrame:
    """
    获取单只股票的完整信息（Eastmoney），带重试。
    返回: 单行 DataFrame，包含接口返回的所有字段
    """
    for attempt in range(1, max_retries + 1):
        try:
            sub_df = ak.stock_individual_info_em(symbol=symbol)
            if sub_df is None or sub_df.empty:
                raise ValueError("Empty response")
            sub_df = sub_df.set_index("item").T
            sub_df.columns.name = None
            sub_df.reset_index(drop=True, inplace=True)
            sub_df.rename(columns={"股票代码": "stock_code", "股票简称": "stock_name", '总股本': "total_share",
                                   '流通股': "float_share", '总市值': "total_market_value",
                                   '流通市值': "float_market_value",
                                   '行业': "industry", '上市时间': "listing_date"}, inplace=True)
            sub_df['stock_code'] = sub_df['stock_code'].str.zfill(6)
            sub_df['listing_date'] = pd.to_datetime(sub_df['listing_date'], format='%Y%m%d')
            sub_df = sub_df[['stock_code', 'stock_name', 'total_share', 'float_share', 'total_market_value',
                             'float_market_value', 'industry', 'listing_date']]
            return sub_df
        except Exception as e:
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                # 超过重试次数，返回一个仅含 code 的空行
                return pd.DataFrame([{"stock_name": symbol}])


def fetch_stock_info(codes: list[str], max_workers: int = 10, max_retries: int = 5,
                     retry_delay: float = 1.0) -> pd.DataFrame:
    """
    批量获取股票完整信息（含行业等所有字段），支持并发 + 重试。

    Parameters
    ----------
    codes : list[str]
        股票代码列表
    max_workers : int
        并发线程数
    max_retries : int
        单个代码最大重试次数
    retry_delay : float
        重试间隔（秒）

    Returns
    -------
    pd.DataFrame
        每只股票一行，包含 stock_individual_info_em 返回的全部字段
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(sub_fetch_stock_info, code, max_retries, retry_delay): code
            for code in codes
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="fetch_stock_info_em"):
            results.append(fut.result())

    return pd.concat(results, ignore_index=True)


# 5) 个股日行情（前复权）
def sub_fetch_stock_daily_return(symbol: str, start: str, end: str,
                                 max_retries: int = 5, retry_delay: float = 1.0) -> pd.DataFrame:
    """
    获取单只股票的区间日行情（带重试）

    Parameters
    ----------
    symbol : str
        股票代码（6位数字，如 '000001'）
    start : str
        开始日期 'YYYYMMDD'
    end : str
        结束日期 'YYYYMMDD'
    max_retries : int
        最大重试次数
    retry_delay : float
        每次重试的间隔秒数

    Returns
    -------
    pd.DataFrame
        ['date','stock_code','open','close','high','low','volume','amount','amplitude','change','change_amount','turnover_rate']
    """
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq")
            if df is None or df.empty:
                raise ValueError("Empty response")

            df.rename(columns={"日期": "date", "股票代码": "stock_code", "开盘": "open", "收盘": "close",
                               "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
                               "振幅": "amplitude", "涨跌幅": "change", "涨跌额": "change_amount",
                               "换手率": "turnover_rate"}, inplace=True)

            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df['stock_code'] = df['stock_code'].astype(str)
            float_cols = ['open', 'close', 'high', 'low', 'volume', 'amount',
                          'amplitude', 'change', 'change_amount', 'turnover_rate']
            df[float_cols] = df[float_cols].astype(float)

            return df
        except Exception as e:
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                # 超过重试次数，返回一个空 DataFrame，避免中断
                return pd.DataFrame(columns=["date", "stock_code", "open", "close", "high", "low",
                                             "volume", "amount", "amplitude", "change", "change_amount",
                                             "turnover_rate"])


def fetch_stock_daily_returns_parallel(codes: list[str], start: str, end: str,
                                       max_workers: int = 10, max_retries: int = 5,
                                       retry_delay: float = 1.0) -> pd.DataFrame:
    """
    并发抓取多个股票的日行情（带重试）

    Returns
    -------
    pd.DataFrame
        多只股票的拼接结果
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(sub_fetch_stock_daily_return, code, start, end, max_retries, retry_delay): code
            for code in codes
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="fetch_stock_daily_returns"):
            results.append(fut.result())

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# 6) 指数日行情（CSIndex 中证指数）
def fetch_index_daily_return(symbol: str, start: str, end: str, max_retries: int = 5,
                             retry_delay: float = 0.5) -> pd.DataFrame:
    """
    获取指定指数在指定时间范围内的日线行情数据

    参数:
        symbol (str): 指数代码
        start (str): 开始日期，格式为 'YYYYMMDD'
        end (str): 结束日期，格式为 'YYYYMMDD'

    返回:
        pd.DataFrame: 包含指数日线行情数据的DataFrame，包含以下列：
            - date: 交易日期
            - index_code: 指数代码
            - open: 开盘价
            - close: 收盘价
            - high: 最高价
            - low: 最低价
            - volume: 成交量
            - amount: 成交金额
            - change: 涨跌幅
            - change_amount: 涨跌金额
            - pe_ratio: 滚动市盈率
    """
    for attempt in range(1, max_retries + 1):
        try:
            # 获取指数历史行情数据
            index_df = ak.stock_zh_index_hist_csindex(symbol=symbol, start_date=start, end_date=end)

            # 重命名列名为英文
            index_df.rename(columns={"日期": "date", "指数代码": "index_code", "指数中文全称": "index_name",
                                     "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
                                     "成交量": "volume", "成交金额": "amount", "涨跌幅": "change",
                                     "涨跌": "change_amount", "滚动市盈率": "pe_ratio"}, inplace=True)

            # 数据类型转换
            index_df['date'] = pd.to_datetime(index_df['date'], format='%Y%m%d')
            index_df['index_code'] = index_df['index_code'].astype(str)
            index_df[['open', 'close', 'high', 'low', 'volume', 'amount', 'change', 'change_amount', 'pe_ratio']] = \
                index_df[
                    ['open', 'close', 'high', 'low', 'volume', 'amount', 'change', 'change_amount', 'pe_ratio']].astype(
                    float)

            # 返回所需的列数据
            return index_df[['date', 'index_code', 'index_name', 'open', 'close', 'high', 'low', 'volume', 'amount',
                             'change', 'change_amount', 'pe_ratio']]
        except Exception as e:
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                # 超过重试次数，返回一个空 DataFrame，避免中断
                return pd.DataFrame()


if __name__ == "__main__":
    fund_code_list = ['000082',  # 嘉实研究阿尔法股票A
                      '000309',  # 大摩品质生活精选股票A
                      '000628',  # 大成高鑫股票A
                      '000729',  # 建信中小盘先锋股票A
                      '000780',  # 鹏华医疗保健股票
                      '470888'  # 华宝制造股票
                      ]  # 可扩展多个基金
    index_code = '000985'  # 中证指数代码

    ''' 获取基金持仓 '''
    fund_hold = fetch_fund_holdings_multi(fund_code_list, '2025', get_last_data=True)
    fund_hold.to_parquet('data/fund_hold.parquet', index=False)
    # fund_hold 的字段以及含义为:
    _ = {'fund_code': '基金代码',
         'report_date': '报告日期',
         'stock_code': '股票代码',
         'stock_name': '股票名称',
         'weight': '占市值权重',
         'num': '持股数量',
         'market_value': '持仓市值'}
    print(f"完成基金持仓数据获取, 共 {len(fund_hold)} 条记录")

    ''' 获取指数成分及权重 '''
    index_hold = fetch_index_stock_cons_weight_index(index_code)
    index_hold.to_parquet('data/index_hold.parquet', index=False)
    # index_hold 的字段以及含义为:
    _ = {'index_code': '指数代码',
         'index_name': '指数名称',
         'report_date': '报告日期',
         'stock_code': '股票代码',
         'stock_name': '股票名称',
         'weight': '权重'}
    print(f"完成指数成分及权重数据获取, 共 {len(index_hold)} 条记录")

    ''' 获取股票信息 (含行业等) '''
    fund_stock_codes = fund_hold['stock_code'].unique().tolist()  # 基金股票代码
    index_stock_codes = index_hold['stock_code'].unique().tolist()  # 指数股票代码
    # fund_stock_codes = pd.read_parquet('data/fund_hold.parquet')['stock_code'].unique().tolist()  # 基金股票代码
    # index_stock_codes = pd.read_parquet('data/index_hold.parquet')['stock_code'].unique().tolist()  # 指数股票代码
    stock_codes = list(set(fund_stock_codes + index_stock_codes))  # 合并去重
    print(f"基金与指数下, 共 {len(stock_codes)} 只股票")
    # TUshare 接口
    stock_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    stock_info = stock_info[stock_info['symbol'].isin(stock_codes)][['symbol', 'name', 'industry', 'list_date']]
    stock_info.rename(columns={'symbol': 'stock_code', 'name': 'stock_name', 'list_date': 'listing_date'}, inplace=True)
    # # AKshare 接口
    # stock_info = fetch_stock_info(stock_codes, max_workers=20, max_retries=5, retry_delay=1.0)
    stock_info.to_parquet('data/stock_info.parquet', index=False)
    # stock_info 的字段以及含义为:
    _ = {'stock_code': '股票代码',
         'stock_name': '股票名称',
         'total_share': '总股本',
         'float_share': '流通股',
         'total_market_value': '总市值',
         'float_market_value': '流通市值',
         'industry': '行业',
         'listing_date': '上市时间'}
    print(f"完成股票信息数据获取, 共 {len(stock_info)} 条记录")

    # 获取股票日行情
    stock_daily = fetch_stock_daily_returns_parallel(stock_codes, '20240101',
                                                     pd.to_datetime('today').strftime('%Y%m%d'),
                                                     max_workers=20, max_retries=5, retry_delay=1.0)
    stock_daily.to_parquet('data/stock_daily.parquet', index=False)
    # stock_daily 的字段以及含义为:
    _ = {'date': '交易日期',
         'stock_code': '股票代码',
         'open': '开盘价',
         'close': '收盘价',
         'high': '最高价',
         'low': '最低价',
         'volume': '成交量',
         'amount': '成交额',
         'amplitude': '振幅',
         'change': '涨跌幅',
         'change_amount': '涨跌额',
         'turnover_rate': '换手率'}
    print(f"完成股票日行情数据获取, 共 {len(stock_daily)} 条记录")

    # 获取指数日行情
    index_daily = fetch_index_daily_return(index_code, '20240101',
                                           pd.to_datetime('today').strftime('%Y%m%d'))
    index_daily.to_parquet('data/index_daily.parquet', index=False)
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
    print(f"完成指数日行情数据获取, 共 {len(index_daily)} 条记录")
