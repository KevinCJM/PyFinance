# -*- coding: utf-8 -*-
"""
B02_construct_category_yield.py

根据模型配置汇总指数来源，并从 wind_cmfindexeod 抽取成分指数的净值数据。

步骤
1) 从 iis_wght_cnfg_attc_mdl 取第一条记录：mdl_ver_id, cal_strt_dt, cal_end_dt
2) 读取 iis_wght_cnfg_mdl 中该 mdl_ver_id 的配置：mdl_ver_id, aset_bclass_cd, indx_num, indx_nm, wght
3) 在 iis_fnd_indx_info 中查询这些 indx_num 的来源表：indx_num, src_tab_ennm
4) 如果存在 src_tab_ennm != 'wind_cmfindexeod' 的记录，抛出错误
5) 合并所有 src_tab_ennm 为 'wind_cmfindexeod' 的 indx_num，
   到 wind_cmfindexeod 中查询 s_info_windcode in (indx_num) 的数据（s_info_windcode, s_info_name, trade_dt, s_dq_close）

依赖
- T05_db_utils: DatabaseConnectionPool, read_dataframe, get_active_db_url
- Y01_db_config: 数据库连接参数
"""
import os
import sys

p_file = os.path.abspath("__file__")
pp_file = os.path.dirname(p_file)
sys.path.append(pp_file)
ppp_file = os.path.dirname(pp_file)
sys.path.append(ppp_file)

import time
import json
import traceback
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text
from typing import Dict, List, Tuple

try:
    from countus.efficient_frontier_API.T02_other_tools import log
    from countus.efficient_frontier_API.T05_db_utils import (DatabaseConnectionPool, read_dataframe, get_active_db_url,
                                                             threaded_upsert_dataframe_mysql, threaded_insert_dataframe)
    from countus.efficient_frontier_API.Y02_asset_id_map import aset_cd_nm_dict
    from countus.efficient_frontier_API.B01_cal_all_weights import fetch_standard_portfolios, generate_alloc_perf_numba
except ImportError:
    from efficient_frontier_API.T02_other_tools import log
    from efficient_frontier_API.T05_db_utils import (DatabaseConnectionPool, read_dataframe, get_active_db_url,
                                                     threaded_upsert_dataframe_mysql, threaded_insert_dataframe)
    from efficient_frontier_API.Y02_asset_id_map import aset_cd_nm_dict
    from efficient_frontier_API.B01_cal_all_weights import fetch_standard_portfolios, generate_alloc_perf_numba

try:
    try:
        from countus.efficient_frontier_API.Y01_db_config import (db_type, db_host, db_port, db_name, db_user,
                                                                  db_password)
    except ImportError:
        from efficient_frontier_API.Y01_db_config import (db_type, db_host, db_port, db_name, db_user, db_password)
except Exception:
    raise RuntimeError("请先在 Y01_db_config.py 中配置数据库连接参数")


def _db_pool() -> DatabaseConnectionPool:
    url = get_active_db_url(
        db_type=db_type,
        db_user=db_user,
        db_password=db_password,
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
    )
    return DatabaseConnectionPool(url=url, pool_size=4)


def _build_in_clause(name_prefix: str, values: List[str]) -> Tuple[str, Dict[str, str]]:
    """为 SQLAlchemy 1.2 生成 IN 子句占位符与参数映射。"""
    params: Dict[str, str] = {}
    holders: List[str] = []
    for i, v in enumerate(values):
        key = f"{name_prefix}{i}"
        holders.append(f":{key}")
        params[key] = v
    return "(" + ",".join(holders) + ")", params


def fetch_first_model(pool: DatabaseConnectionPool) -> pd.Series:
    log("读取 iis_wght_cnfg_attc_mdl 表的第一条记录")
    sql = (
        "SELECT mdl_ver_id, cal_strt_dt, cal_end_dt "
        "FROM iis_wght_cnfg_attc_mdl WHERE mdl_st = '2' ORDER BY mdl_ver_id ASC LIMIT 1"
    )
    df = read_dataframe(pool, sql)
    if df.empty:
        raise RuntimeError("iis_wght_cnfg_attc_mdl 表无数据")
    return df.iloc[0]


def fetch_model_config(pool: DatabaseConnectionPool, mdl_ver_id: str) -> pd.DataFrame:
    log(f"读取 iis_wght_cnfg_mdl 表中大类模型 {mdl_ver_id} 的配置")
    sql = (
        "SELECT mdl_ver_id, aset_bclass_cd, indx_num, indx_nm, wght "
        "FROM iis_wght_cnfg_mdl WHERE mdl_ver_id = :mdl"
    )
    df = read_dataframe(pool, sql, params={"mdl": mdl_ver_id})
    if df.empty:
        raise RuntimeError(f"iis_wght_cnfg_mdl 中无模型 {mdl_ver_id} 的配置")
    return df


def fetch_index_sources(pool: DatabaseConnectionPool, codes: List[str]) -> pd.DataFrame:
    log("读取 iis_fnd_indx_info 中的指数来源表")
    if not codes:
        return pd.DataFrame(columns=["indx_num", "src_tab_ennm"])  # 空
    in_clause, params = _build_in_clause("p", list(codes))
    sql = (
        "SELECT indx_num, src_tab_ennm FROM iis_fnd_indx_info "
        f"WHERE indx_num IN {in_clause}"
    )
    return read_dataframe(pool, sql, params=params)


def fetch_wind_index_eod(pool: DatabaseConnectionPool, codes: List[str]) -> pd.DataFrame:
    log("读取 wind_cmfindexeod 中抽取成分指数的净值数据")
    if not codes:
        raise RuntimeError("无可查询的指数代码")
    in_clause, params = _build_in_clause("w", list(codes))
    sql = (
        f"SELECT s_info_windcode, s_info_name, trade_dt, s_dq_close "
        f"FROM wind_cmfindexeod WHERE s_info_windcode IN {in_clause} and s_dq_close is not null"
    )
    return read_dataframe(pool, sql, params=params)


def calculate_and_insert_rsk_metrics(pool, mdl_ver_id, result_df):
    log("开始计算C1-C6标准组合性能指标...")

    # 1. 准备数据
    std_portfolios = fetch_standard_portfolios(pool, mdl_ver_id, aset_cd_nm_dict)
    if not std_portfolios:
        log("未能从数据库加载C1-C6标准组合权重，跳过指标计算。")
        return

    daily_returns_df = result_df.pivot_table(index="pct_yld_date", columns="aset_bclass_nm", values="pct_yld")
    asset_list = daily_returns_df.columns.tolist()

    weights_list = []
    rsk_lvl_list = []
    for lvl, wdict in std_portfolios:
        weights = [wdict.get(name, 0.0) for name in asset_list]
        weights_list.append(weights)
        rsk_lvl_list.append(lvl)

    weights_array = np.array(weights_list)

    # 2. 计算指标
    perf_df = generate_alloc_perf_numba(asset_list, daily_returns_df, weights_array)

    # 3. 准备待插入的数据
    perf_df['mdl_ver_id'] = mdl_ver_id
    perf_df['rsk_lvl'] = rsk_lvl_list

    db_df = perf_df.rename(columns={
        'ret_annual': 'pct_yld',
        'vol_annual': 'pct_std',
        'sharpe_ratio': 'shrp_prprtn',
        'var_annual': 'var_value'
    })

    final_cols = ['mdl_ver_id', 'rsk_lvl', 'pct_yld', 'pct_std', 'shrp_prprtn', 'var_value']
    db_df = db_df[final_cols]

    # 4. 执行数据库操作 (Delete + Insert)
    log(f"删除并重新插入 iis_wght_cnfg_mdl_rsk 表中模型 {mdl_ver_id} 的数据...")
    with pool.begin() as conn:
        try:
            conn.execute(
                text("DELETE FROM iis_wght_cnfg_mdl_rsk WHERE mdl_ver_id = :mdl_ver_id"),
                {"mdl_ver_id": mdl_ver_id}
            )
        except Exception as e:
            log(f"DELETE FROM iis_wght_cnfg_mdl_rsk 失败: {e}")
            raise

    threaded_insert_dataframe(pool, [{'dataframe': db_df, 'table': 'iis_wght_cnfg_mdl_rsk'}])
    log("iis_wght_cnfg_mdl_rsk 表写入完成。")


def run():
    s_t = time.time()
    pool = _db_pool()
    df_annual_stats = pd.DataFrame()

    ''' 1. 数据获取 ----------------------------------------------------------------------------- '''
    # 1) 模型头信息
    try:
        head = fetch_first_model(pool)
        mdl_ver_id = str(head["mdl_ver_id"]) if "mdl_ver_id" in head else str(head.iloc[0])
        cal_strt_dt = pd.to_datetime(head.get("cal_strt_dt")) if "cal_strt_dt" in head else None
        cal_end_dt = pd.to_datetime(head.get("cal_end_dt")) if "cal_end_dt" in head else None
    except Exception as e:
        return json.dumps({
            "code": 1,
            "msg": f"读取 iis_wght_cnfg_attc_mdl 表数据失败: {e}"
        }, ensure_ascii=False)

    # 2) 模型成分与权重
    try:
        cfg_df = fetch_model_config(pool, mdl_ver_id)
        codes = sorted(cfg_df["indx_num"].astype(str).unique().tolist())
    except Exception as e:
        return json.dumps({
            "code": 1,
            "msg": f"读取 iis_wght_cnfg_mdl 表中大类模型 {mdl_ver_id} 的配置失败: {e}"
        }, ensure_ascii=False)

    # 3) 指数来源表
    try:
        src_df = fetch_index_sources(pool, codes)
        if src_df.empty:
            raise RuntimeError("iis_fnd_indx_info 未返回任何指数来源信息")
        invalid_src = src_df["src_tab_ennm"].astype(str).unique().tolist()
        invalid = [s for s in invalid_src if s != 'wind_cmfindexeod']
        if invalid:
            return json.dumps({
                "code": 1,
                "msg": f"存在不支持的数据来源表: {invalid}，目前仅支持 'wind_cmfindexeod'"
            }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "code": 1,
            "msg": f"读取 iis_fnd_indx_info 中的指数来源表失败: {e}"
        }, ensure_ascii=False)

    # 4) 从 wind_cmfindexeod 抽取数据
    try:
        df_eod = fetch_wind_index_eod(pool, codes)
        if df_eod.empty:
            return json.dumps({
                "code": 1,
                "msg": f"wind_cmfindexeod 中未查询到对应指数的数据"
            }, ensure_ascii=False)

        # 统一数据类型与排序
        df_eod["trade_dt"] = pd.to_datetime(df_eod["trade_dt"])  # 日期
        df_eod_sorted = df_eod.sort_values(["s_info_windcode", "trade_dt"]).reset_index(drop=True)
    except Exception as e:
        return json.dumps({
            "code": 1,
            "msg": f"从 wind_cmfindexeod 抽取指数数据失败: {e}"
        }, ensure_ascii=False)

    ''' 2. 大类收益计算 ------------------------------------------------------------------------- '''
    try:
        # 1) 计算指数日收益（按代码分组后再透视）
        log("计算指数日收益 ... ")
        df_eod_sorted["ret"] = df_eod_sorted.groupby("s_info_windcode")["s_dq_close"].pct_change()
        # 去除首日 NaN 与非有限值（兼容旧版 pandas，避免 pd.NA 带来的歧义）
        ret_vals = pd.to_numeric(df_eod_sorted["ret"], errors="coerce").astype(float)
        mask_valid = np.isfinite(ret_vals.values)
        ret_df = df_eod_sorted.loc[mask_valid].copy()
        index_return_df = ret_df.pivot_table(index="trade_dt", columns="s_info_windcode", values="ret", aggfunc="first")
        index_return_df = index_return_df.sort_index().replace([np.inf, -np.inf], np.nan).dropna(how='any')

        # 2) 按 cal_strt_dt/cal_end_dt 截取区间
        if pd.notna(cal_strt_dt):
            index_return_df = index_return_df[index_return_df.index >= cal_strt_dt]
        if pd.notna(cal_end_dt):
            index_return_df = index_return_df[index_return_df.index <= cal_end_dt]
        else:
            cal_end_dt = pd.to_datetime("today")
            index_return_df = index_return_df[index_return_df.index <= cal_end_dt]
        if index_return_df.empty:
            return json.dumps({
                "code": 1,
                "msg": f"指定区间内无指数收益数据，无法构建大类收益"
            }, ensure_ascii=False)

        # 3) 按资产大类加权拟合收益
        log("按资产大类加权拟合收益与年度指标")
        out_frames: List[pd.DataFrame] = []
        annual_stats_list: List[Dict] = []

        for aset_cd, grp in cfg_df.groupby("aset_bclass_cd"):
            g = grp.copy()
            g["indx_num"] = g["indx_num"].astype(str)
            # 仅保留在收益矩阵中的成分
            cols = [c for c in g["indx_num"].tolist() if c in index_return_df.columns]
            if not cols:
                continue
            # 对应权重并归一化
            w_map = dict(zip(g["indx_num"], g["wght"].astype(float)))
            w = pd.Series([w_map[c] for c in cols], index=cols)
            s = float(w.sum())
            if s <= 0:
                continue
            w = w / s
            # 计算加权收益
            cat_ret = (index_return_df[cols] * w.values).sum(axis=1)

            if cat_ret.empty:
                continue

            asset_name = aset_cd_nm_dict.get(str(aset_cd), str(aset_cd))

            # 计算年化收益与波动
            trading_days = 252.0
            mu = cat_ret.mean()
            sigma = cat_ret.std(ddof=1)
            annual_return = ((1 + mu) ** trading_days) - 1
            annual_vol = sigma * np.sqrt(trading_days)
            annual_stats_list.append({
                "mdl_ver_id": mdl_ver_id,
                "aset_bclass_cd": str(aset_cd),
                "aset_bclass_nm": asset_name,
                "pct_yld": annual_return,
                "pct_std": annual_vol,
                "data_dt": cat_ret.index[-1].date(),
                "crt_tm": pd.to_datetime("now"),
            })

            # 计算累计净值
            acc_nav = (1 + cat_ret).cumprod()
            tmp = pd.DataFrame({
                "mdl_ver_id": mdl_ver_id,
                "aset_bclass_cd": str(aset_cd),
                "aset_bclass_nm": asset_name,
                "pct_yld_date": cat_ret.index,
                "pct_yld": cat_ret.values,
                "acc_value": acc_nav.values
            })

            # 构造并合并初始净值记录
            start_date = cat_ret.index[0] - timedelta(days=1)
            initial_nav_record = pd.DataFrame([{
                "mdl_ver_id": mdl_ver_id,
                "aset_bclass_cd": str(aset_cd),
                "aset_bclass_nm": asset_name,
                "pct_yld_date": start_date,
                "pct_yld": np.nan,
                "acc_value": 1.0
            }])

            tmp_with_initial_nav = pd.concat([initial_nav_record, tmp], ignore_index=True)
            out_frames.append(tmp_with_initial_nav)

        if not out_frames:
            return json.dumps({
                "code": 1,
                "msg": f"未能为任何大类生成收益序列"
            }, ensure_ascii=False)

        result_df = pd.concat(out_frames, ignore_index=True)
        if annual_stats_list:
            df_annual_stats = pd.DataFrame(annual_stats_list)
        else:
            log("未能为任何大类生成年度统计数据")
    except Exception as e:
        return json.dumps({
            "code": 1,
            "msg": f"拟合大类收益率数据失败: {e}"
        }, ensure_ascii=False)

    ''' 3. 结果入库 ----------------------------------------------------------------------------- '''
    try:
        log("Upserting 大类收益率和年度统计数据 ...")
        datasets = []

        # 清理 iis_mdl_aset_pct_d 的数据
        for aset_cd, sub in result_df.groupby("aset_bclass_cd"):
            sub_cleaned = sub.replace({np.nan: None})
            datasets.append({
                "dataframe": sub_cleaned,
                "table": "iis_mdl_aset_pct_d",
                "batch_size": 2000,
            })

        # 清理 iis_aset_allc_indx_rtrn 的数据
        if not df_annual_stats.empty:
            df_annual_stats_cleaned = df_annual_stats.replace({np.nan: None})
            datasets.append({
                "dataframe": df_annual_stats_cleaned,
                "table": "iis_aset_allc_indx_rtrn",
                "batch_size": 200,
            })

        if datasets:
            threaded_upsert_dataframe_mysql(pool, datasets, max_workers=4)
            log(f"大类收益率数据拟合与 Upsert 入 iis_aset_allc_indx_rtrn 和 iis_mdl_aset_pct_d 表完成，"
                f"耗时 {time.time() - s_t:.2f} 秒")
        else:
            log("没有数据需要入库。")
    except Exception as e:
        print(traceback.format_exc())
        return json.dumps({
            "code": 1,
            "msg": f"结果入库失败: {e}"
        }, ensure_ascii=False)

    ''' 4. 计算标准组合指标并入库 ----------------------------------------------------------------------------- '''
    try:
        # 计算并插入C1-C6风险指标
        calculate_and_insert_rsk_metrics(pool, mdl_ver_id, result_df)
    except Exception as e:
        print(traceback.format_exc())
        return json.dumps({
            "code": 1,
            "msg": f"C1-C6风险指标计算或入库失败: {e}"
        }, ensure_ascii=False)

    return json.dumps({
        "code": 0,
        "msg": ""
    }, ensure_ascii=False)


if __name__ == '__main__':
    result_json = run()
    print(result_json)
