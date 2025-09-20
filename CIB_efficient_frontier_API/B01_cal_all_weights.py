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
from typing import List, Tuple, Optional
from datetime import date as _date

# 数据库/线程池工具
from T05_db_utils import (
    DatabaseConnectionPool,
    threaded_read_dataframe,
    get_active_db_url,
    read_dataframe,
    create_connection,
)

try:
    # 数据库配置仅包含参数
    from Y01_db_config import db_type, db_host, db_port, db_name, db_user, db_password  # type: ignore
except Exception:
    # 合理的默认值，便于本地快速尝试
    db_type, db_host, db_port, db_name, db_user, db_password = (
        'mysql', None, '3306', 'mysql', 'root', '112358'
    )


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
    parallel=True, nogil=True, fastmath=True, cache=True,
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


def plot_efficient_frontier_plotly(df: pd.DataFrame,
                                   asset_cols,
                                   title: str = '资产组合有效前沿',
                                   sample_non_ef: int = 50000,
                                   save_html=None) -> None:
    def _build_hover_text(row, a_cols):
        parts = [
            f"年化收益率: {row['ret_annual']:.2%}",
            f"年化波动率: {row['vol_annual']:.2%}",
            f"VaR95: {row['var_annual']:.2%}",
            f"Sharpe: {row.get('sharpe_ratio', np.nan):.3f}",
            "<br><b>资产权重</b>:"
        ]
        for col in a_cols:
            v = row.get(col, np.nan)
            try:
                fv = float(v)
            except Exception:
                fv = np.nan
            if np.isfinite(fv) and fv > 1e-4:
                parts.append(f"<br>{col}: {fv:.1%}")
        return "".join(parts)

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
def fetch_returns_from_db(mdl_ver_id: str,
                          start_dt: Optional[_date] = None,
                          end_dt: Optional[_date] = None) -> Tuple[List[str], pd.DataFrame]:
    """从数据库 iis_mdl_aset_pct_d 表读取指定模型的五大类日收益数据。

    返回:
        a_list: 资产大类名称列表（列名）
        re_df: 以日期为索引、资产大类为列、pct_yld 为值的 DataFrame
    """
    # 构造连接串（支持容器/本机自动主机判定；ENV: DB_URL 覆盖）
    db_url = get_active_db_url(
        db_type=db_type,
        db_user=db_user,
        db_password=db_password,
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
    )
    pool = DatabaseConnectionPool(url=db_url, pool_size=4)

    where_parts = ["mdl_ver_id = :mdl_ver_id"]
    params = {"mdl_ver_id": mdl_ver_id}
    if start_dt is not None:
        where_parts.append("pct_yld_date >= :start_dt")
        params["start_dt"] = start_dt
    if end_dt is not None:
        where_parts.append("pct_yld_date <= :end_dt")
        params["end_dt"] = end_dt
    where_sql = " WHERE " + " AND ".join(where_parts)

    sql_assets = (
        "SELECT DISTINCT aset_bclass_nm AS nm FROM iis_mdl_aset_pct_d" + where_sql + " ORDER BY nm"
    )
    sql_series = (
        "SELECT aset_bclass_nm AS nm, pct_yld_date AS dt, pct_yld AS yld FROM iis_mdl_aset_pct_d" + where_sql
    )

    df_assets, df_series = threaded_read_dataframe(
        pool,
        [
            (sql_assets, params),
            (sql_series, params),
        ],
        max_workers=2,
    )

    if df_assets.empty or df_series.empty:
        raise RuntimeError("数据库中未查询到模型 {} 的资产或收益数据".format(mdl_ver_id))

    a_list = df_assets["nm"].astype(str).tolist()

    df = df_series.copy()
    df["dt"] = pd.to_datetime(df["dt"])  # 保证为 datetime 索引
    df = df.pivot_table(index="dt", columns="nm", values="yld", aggfunc="first")
    df = df.sort_index().dropna(how='any')

    # reindex columns to a_list to keep stable ordering
    re_df = df.reindex(columns=a_list)
    return a_list, re_df


def fetch_default_mdl_ver_id() -> Tuple[str, Optional[_date], Optional[_date]]:
    """从 iis_wght_cfg_attc_mdl 表中获取第一条 mdl_ver_id、cal_strt_dt、cal_end_dt。

    若无数据则抛出异常。
    """
    db_url = get_active_db_url(
        db_type=db_type,
        db_user=db_user,
        db_password=db_password,
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
    )
    # 单次查询：取第一条记录（可按需调整排序口径）
    sql = "SELECT mdl_ver_id, cal_strt_dt, cal_end_dt FROM iis_wght_cfg_attc_mdl ORDER BY mdl_ver_id ASC LIMIT 1"
    conn = create_connection(db_url)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    if df.empty:
        raise RuntimeError("数据库中未找到可用的模型版本（iis_wght_cfg_attc_mdl 为空）")
    row = df.iloc[0]
    mdl = str(row["mdl_ver_id"]) if "mdl_ver_id" in df.columns else str(row[0])
    def _to_date(v):
        if pd.isna(v):
            return None
        if hasattr(v, 'date'):
            return v.date()
        return v
    s_dt = _to_date(row.get("cal_strt_dt")) if "cal_strt_dt" in row else None
    e_dt = _to_date(row.get("cal_end_dt")) if "cal_end_dt" in row else None
    return mdl, s_dt, e_dt


if __name__ == '__main__':
    ''' 0) 数据准备（从数据库） ------------------------------------------------------------------------ '''
    mdl_ver_id, cal_sta_dt, cal_end_dt = fetch_default_mdl_ver_id()
    print(f"使用大类资产指数配置模型版本: {mdl_ver_id} | 起止: {cal_sta_dt} ~ {cal_end_dt}")
    a_list, re_df = fetch_returns_from_db(mdl_ver_id, start_dt=cal_sta_dt, end_dt=cal_end_dt)
    print("使用模型: {} | 资产大类: {} | 日期范围: {} ~ {}".format(
        mdl_ver_id,
        ", ".join(a_list),
        re_df.index.min().date() if len(re_df.index) else None,
        re_df.index.max().date() if len(re_df.index) else None,
    ))

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
