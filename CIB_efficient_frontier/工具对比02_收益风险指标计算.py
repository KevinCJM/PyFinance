# -*- encoding: utf-8 -*-
"""
@File: 工具对比02_收益风险指标计算成.py
@Modify Time: 2025/9/19 10:29       
@Author: Kevin-Chen
@Descriptions: 收益风险指标计算
"""

import os
import time
import datetime
import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads
from multiprocessing import Pool, cpu_count
from 工具对比01_单纯形网格点生成 import generate_simplex_grid_numba


def data_prepare(excel_path, sheet, asset_list):
    def _read_excel_auto(path, sheet_name=None):
        """
        健壮的 Excel 读取：依据扩展名选择引擎；.xlsx/.xlsm 用 openpyxl；.xls 用 xlrd。
        若缺少对应引擎，会抛出带指引的异常。
        """
        ext = os.path.splitext(path)[1].lower()
        if ext in {'.xlsx', '.xlsm', '.xltx', '.xltm'}:
            engine = 'openpyxl'
        elif ext == '.xls':
            engine = 'xlrd'
        else:
            engine = None  # 让 pandas 自行判断
        try:
            return pd.read_excel(path, sheet_name=sheet_name, engine=engine)
        except ImportError as e:
            hint = (
                    "读取 Excel 失败：缺少引擎。对于 .xlsx/.xlsm 请安装 openpyxl；"
                    "对于 .xls 请安装 xlrd<2.0.0。原始错误: %r" % (e,)
            )
            raise RuntimeError(hint)

    hist_value = _read_excel_auto(excel_path, sheet)
    # 数据输入处理
    hist_value = hist_value.set_index('date')
    hist_value.index = pd.to_datetime(hist_value.index)
    hist_value = hist_value.dropna().sort_index(ascending=True)
    hist_value = hist_value.rename({
        "货基指数": "货币现金类",
        '固收类': '固定收益类',
        '混合类': '混合策略类',
        '权益类': '权益投资类',
        '另类': '另类投资类'
    }, axis=1)
    # 计算日收益率
    hist_value = hist_value.pct_change().dropna()
    return hist_value[asset_list]


# 加权求和计算指数组合每日收益数据
def generate_alloc_perf(portfolio_alloc, hist_value_r, p95=1.96):
    """
    计算投资组合的年度化收益率和年度化波动率

    参数:
        portfolio_alloc (dict): 投资组合权重字典，键为资产名称，值为权重
        hist_value_r (DataFrame): 历史收益率数据，包含各资产的历史日收益率

    返回:
        dict: 包含年度化收益率(ret_annual)和年度化波动率(vol_annual)的字典
    """
    port_daily = hist_value_r.copy()
    port_daily = port_daily[portfolio_alloc.keys()]
    # 计算投资组合每日收益率：各资产收益率按权重加权求和
    port_daily.loc[:, 'r'] = port_daily[portfolio_alloc.keys()].values @ np.array(list(portfolio_alloc.values()))
    # 计算累计净值
    port_daily['value'] = (1 + port_daily['r']).cumprod()
    # 设置初始净值为1
    port_daily.loc[port_daily.index[0] - datetime.timedelta(days=1), 'value'] = 1
    port_daily = port_daily.sort_index()
    # 计算年度化波动率：使用对数收益率，样本标准差乘以年化因子
    port_vol = np.std(port_daily['r'].apply(lambda r: np.log(r + 1)).dropna(), ddof=1) * np.sqrt(252)
    # 计算年度化收益率：基于净值变化的对数收益率 annualized
    port_ret = np.log(port_daily['value'].iloc[-1] / port_daily['value'].iloc[0]) / (len(port_daily) - 1) * 252
    var_annual = port_ret - port_vol * p95
    return {"ret_annual": float(port_ret), "vol_annual": float(port_vol), "var_annual": float(var_annual)}


def generate_alloc_perf_old(asset_list, return_df, p95=1.96):
    random_port_perf = []
    for w in weight_list:
        asset_alloc = dict(zip(asset_list, list(w)))
        alloc_perf = generate_alloc_perf(asset_alloc, return_df, p95)
        random_port_perf.append(dict(**asset_alloc, **alloc_perf))
    res_df = pd.DataFrame(random_port_perf)
    return res_df


def generate_alloc_perf_new(asset_list, return_df, weight_array: np.ndarray, p95=1.96,
                            trading_days: float = 252.0, ddof: int = 1) -> pd.DataFrame:
    """
    向量化批量计算多个权重组合的收益/风险指标，避免 Python for 循环。

    参数:
        asset_list: 资产列名顺序，对应权重列顺序
        return_df: 资产日收益率 DataFrame（index: 日期, columns: 资产列）
        weight_array: 权重矩阵，shape = (M, N)，M 为组合数，N 为资产数
        p95: 置信倍数（用于 VaR95 型指标）
        trading_days: 年化交易日数
        ddof: 标准差自由度（样本标准差=1）

    返回:
        DataFrame: 列 = 资产权重各列 + ret_annual + vol_annual + var_annual
    """
    # 依据权重矩阵列数对齐资产列（与旧版 zip 行为一致：多余列被截断）
    Nw = int(weight_array.shape[1])
    asset_cols = list(asset_list)[:Nw]

    # 提取并确保 dtype 为 float64
    R = return_df[asset_cols].values
    if R.dtype != np.float64:
        R = R.astype(np.float64, copy=False)
    W = np.asarray(weight_array, dtype=np.float64)

    # 组合日收益：一次矩阵乘法 (T,N) @ (N,M) -> (T,M);  注意这里需转置权重矩阵
    port_return_daily = R @ W.T

    # 对数日收益，以避免累计乘法的数值不稳定
    log_returns = np.log1p(port_return_daily)

    T = float(log_returns.shape[0])
    # 年化对数收益
    ret_annual = (log_returns.sum(axis=0) / T) * float(trading_days)
    # 年化波动
    vol_annual = log_returns.std(axis=0, ddof=ddof) * np.sqrt(float(trading_days))

    # 数值有效性过滤：若某列出现 -inf 或 nan
    valid = np.isfinite(ret_annual) & np.isfinite(vol_annual)
    if not np.all(valid):
        ret_annual = ret_annual[valid]
        vol_annual = vol_annual[valid]
        W = W[valid, :]

    var_annual = ret_annual - vol_annual * float(p95)

    # 组装结果，列名与旧函数保持一致（资产列名）
    weight_df = pd.DataFrame(W, columns=asset_cols)
    perf_df = pd.DataFrame({
        'ret_annual': ret_annual,
        'vol_annual': vol_annual,
        'var_annual': var_annual,
    })
    return pd.concat([weight_df, perf_df], axis=1)


# -------------------- Numba: 零拷贝并行计算 -------------------- #
@njit(parallel=True)
def _compute_perf_numba(R: np.ndarray, W: np.ndarray, trading_days: float, ddof: int, p95: float):
    """
    在不创建 (T,M) 大矩阵的前提下，逐组合并行计算：
      - 按天累积 log(1 + w·r_t) 的均值与方差（Welford）
      - 年化收益与年化波动

    参数:
        R: (T, N) 资产日收益矩阵（float64，C 连续）
        W: (M, N) 权重矩阵（float64，C 连续）
        trading_days: 年化交易日数
        ddof: 样本方差自由度
        p95: VaR 系数

    返回:
        (ret_annual, vol_annual, var_annual): 三个 shape=(M,) 的数组
    """
    T = R.shape[0]
    N = R.shape[1]
    M = W.shape[0]

    out_ret = np.empty(M, dtype=np.float64)
    out_vol = np.empty(M, dtype=np.float64)

    sqrt_td = np.sqrt(trading_days)

    for j in prange(M):
        mean = 0.0
        M2 = 0.0
        n = 0
        invalid = False

        # 遍历时间维度，按天增量更新均值与方差
        for t in range(T):
            s = 0.0
            # 点乘：w_j 与 r_t
            for k in range(N):
                s += R[t, k] * W[j, k]
            # 若出现 1+s <= 0（如 r=-100%），该组合无效
            if s <= -0.999999999:
                invalid = True
                break
            x = np.log1p(s)
            n += 1
            # Welford 增量更新
            delta = x - mean
            mean += delta / n
            M2 += delta * (x - mean)

        if invalid or n <= 1 or (n - ddof) <= 0:
            out_ret[j] = np.nan
            out_vol[j] = np.nan
            continue

        var = M2 / (n - ddof)
        if var < 0.0:
            var = 0.0
        out_ret[j] = mean * trading_days
        out_vol[j] = np.sqrt(var) * sqrt_td

    out_var = out_ret - out_vol * p95
    return out_ret, out_vol, out_var


def generate_alloc_perf_numba(asset_list, return_df, weight_array: np.ndarray, p95=1.96,
                              trading_days: float = 252.0, ddof: int = 1):
    """
    基于 numba 的零拷贝并行实现：
    - 不创建 (T,M) 中间矩阵，按组合并行逐日累积，内存占用 O(T*N + M*N + 3M)。
    - 返回三个一维数组，或按需拼装为 DataFrame（注意：M 很大时 DataFrame 可能占用数百 MB）。

    参数:
        asset_list: 资产列名顺序，对应权重列顺序
        return_df: 资产日收益率 DataFrame
        weight_array: 权重矩阵 (M, N)
        p95, trading_days, ddof: 指标参数
        return_dataframe: 为 True 时返回 DataFrame（含权重与指标），否则返回 (ret, vol, var) 元组

    返回:
        DataFrame: 列为资产权重各列(asset_cols) + ret_annual + vol_annual + var_annual
    """
    # 列对齐：按照权重矩阵的列数截断资产列
    Nw = int(weight_array.shape[1])
    asset_cols = list(asset_list)[:Nw]

    # 保证输入为 float64 且 C 连续
    R = return_df[asset_cols].values
    if R.dtype != np.float64:
        R = R.astype(np.float64)
    if not R.flags.c_contiguous:
        R = np.ascontiguousarray(R)

    W = np.asarray(weight_array, dtype=np.float64)
    if not W.flags.c_contiguous:
        W = np.ascontiguousarray(W)

    ret, vol, var = _compute_perf_numba(R, W, float(trading_days), int(ddof), float(p95))

    # 注意：当 M 很大时，构造 DataFrame 会占用较多内存
    weight_df = pd.DataFrame(W, columns=asset_cols)
    perf_df = pd.DataFrame({'ret_annual': ret, 'vol_annual': vol, 'var_annual': var})
    return pd.concat([weight_df, perf_df], axis=1)


if __name__ == '__main__':
    ''' 0) 数据准备 --------------------------------------------------------------------------------- '''
    e, s = '历史净值数据.xlsx', '历史净值数据'
    a_list = ['货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类']
    re_df = data_prepare(e, s, a_list)
    s_t_0 = time.time()
    weight_list = generate_simplex_grid_numba(len(a_list), 200)
    print(f"计算网格点数量: {weight_list.shape}, 耗时: {time.time() - s_t_0:.2f} 秒")  # 10%:1.87秒, 1%:1.97秒, 0.5%:2.0秒

    # ''' 1) 计算收益风险指标(老版本) ------------------------------------------------------------------- '''
    # s_t_1 = time.time()
    # res_1 = generate_alloc_perf_old(a_list, re_df)
    # print(res_1)
    # print(f"计算收益风险指标(老版本) 耗时: {time.time() - s_t_1:.2f} 秒")  # 10%:13.10秒, 1%:卡死
    #
    # ''' 2) 计算收益风险指标(向量化新版本) ------------------------------------------------------------ '''
    # s_t_2 = time.time()
    # res_2 = generate_alloc_perf_new(a_list, re_df, weight_list)
    # print(res_2)
    # print(f"计算收益风险指标(向量化) 耗时: {time.time() - s_t_2:.2f} 秒")  # 10%:0.08秒, 1%:SIGKILL

    ''' 3) 计算收益风险指标(Numba 并行零拷贝版本) ----------------------------------------------- '''
    s_t_3 = time.time()
    res_3 = generate_alloc_perf_numba(a_list, re_df, weight_list)
    print(res_3.head())
    print(f"计算收益风险指标(Numba) 耗时: {time.time() - s_t_3:.2f} 秒")  # 10%:0.94秒, 1%:30.84秒, 0.5%:SIGKILL
