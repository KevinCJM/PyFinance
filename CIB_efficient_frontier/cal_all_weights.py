# -*- encoding: utf-8 -*-
"""
@File: cal_all_weights.py
@Modify Time: 2025/9/19 14:31       
@Author: Kevin-Chen
@Descriptions: 
"""
import os
import time
import numpy as np
import pandas as pd
from numba import njit, prange, float64, int64, types
import plotly.graph_objects as go


# 使用 Numba 加速网格生成，用于构建资产配置的离散化组合空间
def generate_simplex_grid_numba(assets_num: int, resolution_ratio: int):
    """
    使用 Numba(njit+prange) 加速的网格生成函数。

    本函数用于在 n_assets 维空间中生成所有满足权重和为 1 的离散单纯形网格点。
    采用“星与杠”（Stars and Bars）方法枚举非负整数解，并映射为浮点权重向量。
    利用 Numba 的 njit 和并行 prange 加速组合生成过程，适用于高维、高分辨率场景。

    参数:
        assets_num (int): 资产维度，必须 ≥ 2。
        resolution_ratio (int): 权重划分精度，即将 1 划分为 resolution 份，步长 Δ = 1/resolution。

    返回:
        np.ndarray: shape 为 (C(resolution+n_assets-1, n_assets-1), n_assets) 的二维数组，
                    每一行表示一个权重向量，所有元素非负且和为 1。

    说明：
    - 旧实现基于 Python 的 itertools.combinations，生成 400 万级组合会非常慢；
    - 新实现预分配结果矩阵，使用组合字典序枚举，并在第一维上并行分块填充；
    - 需 numba==0.49.1（见 requirements.txt）。
    """

    @njit
    def _comb(n: int, k: int) -> np.int64:
        """
        计算组合数 C(n, k)，使用数值稳定的方式避免溢出。

        参数:
            n (int): 总数。
            k (int): 选取数。

        返回:
            np.int64: 组合数 C(n, k)。
        """
        if k < 0 or k > n:
            return np.int64(0)
        if k > n - k:
            k = n - k
        res = 1
        for i in range(1, k + 1):
            res = (res * (n - k + i)) // i
        return np.int64(res)

    @njit
    def _fill_block_given_first(out: np.ndarray, start_row: int,
                                first_bar: int, total_slots: int,
                                k: int, n_assets: int, resolution: int) -> int:
        """
        填充所有以 first_bar 作为第一个“杠”的组合块。

        参数:
            out (np.ndarray): 输出数组。
            start_row (int): 当前块起始行索引。
            first_bar (int): 第一个“杠”的位置。
            total_slots (int): 总槽位数（resolution + n_assets - 1）。
            k (int): 需要放置的“杠”数（n_assets - 1）。
            n_assets (int): 资产数。
            resolution (int): 分辨率。

        返回:
            int: 写入的行数。
        """
        # 处理只剩一个杠的情况（递归终止条件）
        k2 = k - 1
        if k2 == 0:
            prev = -1
            row = start_row
            out[row, 0] = (first_bar - prev - 1) / resolution
            prev = first_bar
            out[row, n_assets - 1] = (total_slots - prev - 1) / resolution
            for j in range(1, n_assets - 1):
                out[row, j] = 0.0
            return 1

        # 初始化剩余“杠”的位置索引
        base = first_bar + 1
        n2 = total_slots - base
        idx2 = np.empty(k2, dtype=np.int64)
        for i in range(k2):
            idx2[i] = i

        row = start_row
        while True:
            # 根据当前“杠”位置构造权重向量
            prev = -1
            out[row, 0] = (first_bar - prev - 1) / resolution
            prev = first_bar
            for j in range(k2):
                b = base + idx2[j]
                out[row, j + 1] = (b - prev - 1) / resolution
                prev = b
            out[row, n_assets - 1] = (total_slots - prev - 1) / resolution
            row += 1

            # 更新下一个组合索引（字典序递增）
            p = k2 - 1
            while p >= 0 and idx2[p] == p + n2 - k2:
                p -= 1
            if p < 0:
                break
            idx2[p] += 1
            for j in range(p + 1, k2):
                idx2[j] = idx2[j - 1] + 1

        return row - start_row

    @njit(parallel=True)
    def _generate(n_assets: int, resolution: int) -> np.ndarray:
        """
        并行生成整个单纯形网格。

        参数:
            n_assets (int): 资产维度。
            resolution (int): 权重划分精度。

        返回:
            np.ndarray: 生成的权重矩阵。
        """
        # 总槽位数和杠数
        total_slots = resolution + n_assets - 1
        k = n_assets - 1

        # 预分配输出数组
        M = _comb(total_slots, k)
        out = np.empty((M, n_assets), dtype=np.float64)

        # 计算每个 first_bar 对应的组合数和偏移量
        first_max = total_slots - k
        F = first_max + 1
        counts = np.empty(F, dtype=np.int64)
        for f in range(F):
            counts[f] = _comb(total_slots - (f + 1), k - 1)

        offsets = np.empty(F + 1, dtype=np.int64)
        offsets[0] = 0
        for i in range(F):
            offsets[i + 1] = offsets[i] + counts[i]

        # 并行填充每个 first_bar 对应的块
        for f in prange(F):
            start = offsets[f]
            _ = _fill_block_given_first(out, start, f, total_slots, k, n_assets, resolution)

        return out

    return _generate(assets_num, resolution_ratio)


@njit(
    types.UniTuple(float64[:], 4)(float64[:, :], float64[:, :], float64, int64, float64),
    parallel=True,
    nogil=True,
    fastmath=True,
    cache=True,
)
def _compute_perf_numba(r, w, trading_days, dof, p95):
    """
    在不创建 (T,M) 大矩阵的前提下，逐组合并行计算：
      - 按天累积 log(1 + w·r_t) 的均值与方差
      - 年化收益与年化波动

    参数:
        r: (T, N) 资产日收益矩阵（float64，C 连续）
        w: (M, N) 权重矩阵（float64，C 连续）
        trading_days: 年化交易日数
        dof: 样本方差自由度
        p95: VaR 系数

    返回:
        (ret_annual, vol_annual, var_annual): 三个 shape=(M,) 的数组
    """
    big_t = r.shape[0]
    big_n = r.shape[1]
    big_m = w.shape[0]

    out_ret = np.empty(big_m, dtype=np.float64)
    out_vol = np.empty(big_m, dtype=np.float64)
    out_sharpe = np.empty(big_m, dtype=np.float64)

    sqrt_td = np.sqrt(trading_days)

    # 并行遍历所有组合，计算每个组合的年化收益、波动率和VaR
    for j in prange(big_m):
        mean = 0.0
        m2 = 0.0
        n = 0
        invalid = False

        # 遍历时间维度，按天增量更新均值与方差
        for t in range(big_t):
            s = 0.0
            # 点乘：w_j 与 r_t
            for k in range(big_n):
                s += r[t, k] * w[j, k]
            # 若出现 1+s <= 0（如 r=-100%），该组合无效
            if s <= -0.999999999:
                invalid = True
                break
            x = np.log1p(s)
            n += 1
            delta = x - mean
            mean += delta / n
            m2 += delta * (x - mean)

        # 若组合无效或样本不足，则输出 NaN
        if invalid or n <= 1 or (n - dof) <= 0:
            out_ret[j] = np.nan
            out_vol[j] = np.nan
            out_sharpe[j] = np.nan
            continue

        # 计算年化收益与年化波动率
        var = m2 / (n - dof)
        if var < 0.0:
            var = 0.0
        annual_ret = mean * trading_days
        annual_vol = np.sqrt(var) * sqrt_td
        out_ret[j] = annual_ret
        out_vol[j] = annual_vol
        if annual_vol > 0.0:
            out_sharpe[j] = annual_ret / annual_vol
        else:
            out_sharpe[j] = np.nan

    # 计算 VaR：年化收益 - 年化波动 * VaR 系数
    out_var = out_ret - out_vol * p95
    return out_ret, out_vol, out_var, out_sharpe


# 计算资产组合的年度化收益,波动率,和在险价值
def generate_alloc_perf_numba(asset_list, return_df, weight_array: np.ndarray, p95=1.96,
                              trading_days: float = 252.0, dof: int = 1):
    """
    基于 numba 的零拷贝并行实现，用于计算资产组合的年度化收益、波动率和风险价值（VaR）等指标。
    该函数避免创建中间的 (T,M) 大小矩阵，从而降低内存占用。

    参数:
        asset_list (list): 资产列名列表，其顺序应与权重矩阵 weight_array 的列顺序一致。
        return_df (pd.DataFrame): 包含各资产日收益率的时间序列数据。
        weight_array (np.ndarray): 权重矩阵，形状为 (M, N)，其中 M 为组合数，N 为资产数。
        p95 (float): VaR 计算所用的分位数参数，默认为 1.96（即 95% 置信水平）。
        trading_days (float): 年化交易日数量，默认为 252。
        dof (int): 自由度参数，用于波动率调整，默认为 1。

    返回:
        pd.DataFrame: 包含每组权重及对应年度化收益(ret_annual)、波动率(vol_annual)、风险价值(var_annual)的结果表。
    """

    # 列对齐：按照权重矩阵的列数截断资产列，确保维度匹配
    nw = int(weight_array.shape[1])
    asset_cols = list(asset_list)[:nw]

    # 保证输入为 float64 且 C 连续，以提升 numba 执行效率
    r = return_df[asset_cols].values
    if r.dtype != np.float64:
        r = r.astype(np.float64)
    if not r.flags.c_contiguous:
        r = np.ascontiguousarray(r)

    w = np.asarray(weight_array, dtype=np.float64)
    if not w.flags.c_contiguous:
        w = np.ascontiguousarray(w)

    # 调用核心计算函数，获取年度化收益、波动率、VaR 与夏普比率
    ret, vol, var, sharpe = _compute_perf_numba(r, w, float(trading_days), int(dof), float(p95))

    # 构造结果 DataFrame，注意当 M 很大时，拼接 DataFrame 可能导致高内存占用
    weight_df = pd.DataFrame(w, columns=asset_cols)
    perf_df = pd.DataFrame({
        'ret_annual': ret,
        'vol_annual': vol,
        'var_annual': var,
        'sharpe_ratio': sharpe,
    })
    perf_df = pd.concat([weight_df, perf_df], axis=1)

    # 标记有效前沿上的点
    def _efficient_frontier_mask(ret_v: np.ndarray, vol_v: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        valid = np.isfinite(ret_v) & np.isfinite(vol_v)
        mask = np.zeros(ret_v.shape[0], dtype=bool)
        if not np.any(valid):
            return mask
        # 收益降序排列；对排序后的波动做累计最小
        idx = np.nonzero(valid)[0]
        order = np.argsort(ret_v[valid])[::-1]
        sorted_vol = vol_v[valid][order]
        cum_min = np.minimum.accumulate(sorted_vol)
        on_sorted = sorted_vol <= (cum_min + tol)
        mask[idx[order]] = on_sorted
        return mask

    perf_df['on_ef'] = _efficient_frontier_mask(perf_df['ret_annual'].values, perf_df['vol_annual'].values)
    return perf_df


def _build_hover_text(row, asset_cols):
    parts = [
        f"年化收益率: {row['ret_annual']:.2%}",
        f"年化波动率: {row['vol_annual']:.2%}",
        f"VaR95: {row['var_annual']:.2%}",
        f"Sharpe: {row.get('sharpe_ratio', np.nan):.3f}",
        "<br><b>资产权重</b>:"
    ]
    for col in asset_cols:
        v = row.get(col, np.nan)
        try:
            fv = float(v)
        except Exception:
            fv = np.nan
        if np.isfinite(fv) and fv > 1e-4:
            parts.append(f"<br>{col}: {fv:.1%}")
    return "".join(parts)


def plot_efficient_frontier_plotly(df: pd.DataFrame,
                                   asset_cols,
                                   title: str = '资产组合有效前沿',
                                   sample_non_ef: int = 50000,
                                   save_html=None) -> None:
    df = df.copy()
    ef_df = df[df['on_ef'] == True]
    non_ef_df = df[df['on_ef'] != True]

    if sample_non_ef is not None and len(non_ef_df) > sample_non_ef:
        non_ef_df = non_ef_df.sample(n=sample_non_ef, random_state=42)

    ef_df = ef_df.copy()
    non_ef_df = non_ef_df.copy()
    ef_df['hover_text'] = ef_df.apply(lambda r: _build_hover_text(r, asset_cols), axis=1)
    non_ef_df['hover_text'] = non_ef_df.apply(lambda r: _build_hover_text(r, asset_cols), axis=1)

    fig = go.Figure()
    if len(non_ef_df) > 0:
        fig.add_trace(go.Scatter(
            x=non_ef_df['vol_annual'],
            y=non_ef_df['ret_annual'],
            mode='markers',
            marker=dict(color='rgba(150,150,150,0.45)', size=3),
            hovertext=non_ef_df['hover_text'],
            hoverinfo='text',
            name='其他组合'
        ))
    if len(ef_df) > 0:
        fig.add_trace(go.Scatter(
            x=ef_df['vol_annual'],
            y=ef_df['ret_annual'],
            mode='markers',
            marker=dict(color='rgba(0, 102, 204, 0.9)', size=4),
            hovertext=ef_df['hover_text'],
            hoverinfo='text',
            name='有效前沿'
        ))

    fig.update_layout(
        title=title,
        xaxis_title='年化波动率 (vol_annual)',
        yaxis_title='年化收益率 (ret_annual)',
        legend_title='图例',
        hovermode='closest'
    )
    if save_html:
        fig.write_html(save_html)
        print(f"已保存 HTML 到 {save_html}")
    fig.show()


# 从指定的 Excel 文件中读取并处理大类资产的数据
def data_prepare(excel_path, sheet, asset_list):
    """
    从指定的 Excel 文件中读取并处理历史数据，返回指定资产的日收益率数据。

    参数:
        excel_path (str): Excel 文件的路径。
        sheet (str or int): 要读取的工作表名称或索引。
        asset_list (list of str): 需要保留的资产类别名称列表。

    返回:
        pandas.DataFrame: 包含指定资产日收益率的数据框，索引为日期，列名为资产名称。
    """

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

    # 读取原始数据
    hist_value = _read_excel_auto(excel_path, sheet)

    # 设置日期为索引，并转换为 datetime 类型，去除缺失值后按日期升序排列
    hist_value = hist_value.set_index('date')
    hist_value.index = pd.to_datetime(hist_value.index)
    hist_value = hist_value.dropna().sort_index(ascending=True)

    # 标准化列名，统一资产类别命名
    hist_value = hist_value.rename({
        "货基指数": "货币现金类",
        '固收类': '固定收益类',
        '混合类': '混合策略类',
        '权益类': '权益投资类',
        '另类': '另类投资类'
    }, axis=1)

    # 计算每日收益率，并去除计算后产生的空值
    hist_value = hist_value.pct_change().dropna()
    # 返回用户指定的资产列表对应的数据
    return hist_value[asset_list]


if __name__ == '__main__':
    ''' 0) 数据准备 --------------------------------------------------------------------------------- '''
    e, s = '历史净值数据.xlsx', '历史净值数据'
    a_list = ['货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类']
    re_df = data_prepare(e, s, a_list)

    ''' 1) 网格生成 --------------------------------------------------------------------------------- '''
    s_t_0 = time.time()
    weight_list = generate_simplex_grid_numba(len(a_list), 100)
    print(f"计算网格点数量: {weight_list.shape}, 耗时: {time.time() - s_t_0:.2f} 秒")  # 10%:1.87秒, 1%:1.97秒, 0.5%:2.0秒

    ''' 2) 指标计算 --------------------------------------------------------------------------------- '''
    s_t_1 = time.time()
    res_df = generate_alloc_perf_numba(a_list, re_df, weight_list)
    print(res_df.head())
    print(f"计算指标耗时: {time.time() - s_t_1:.2f} 秒")  # 10%:0.94秒, 1%:30.84秒, 0.5%:SIGKILL

    ''' 3) 画图 ------------------------------------------------------------------------------------- '''
    plot_efficient_frontier_plotly(
        res_df,
        asset_cols=a_list,
        title='资产组合有效前沿（抽样非前沿点）',
        sample_non_ef=50000,  # 抽样 N 个非前沿点以加快绘图速度
        save_html='efficient_frontier.html'
    )
