# -*- coding: utf-8 -*-
"""
B02_construct_category_yield.py

根据模型配置汇总指数来源，并从 wind_cmfindexeod 抽取成分指数的净值数据。

步骤
1) 从 iis_wght_cfg_attc_mdl 取第一条记录：mdl_ver_id, cal_strt_dt, cal_end_dt
2) 读取 iis_wght_cnfg_mdl 中该 mdl_ver_id 的配置：mdl_ver_id, aset_bclass_cd, indx_num, indx_nm, wght
3) 在 iis_fnd_indx_info 中查询这些 indx_num 的来源表：indx_num, src_tab_ennm
4) 如果存在 src_tab_ennm != 'wind_cmfindexeod' 的记录，抛出错误
5) 合并所有 src_tab_ennm 为 'wind_cmfindexeod' 的 indx_num，
   到 wind_cmfindexeod 中查询 s_info_windcode in (indx_num) 的数据（s_info_windcode, s_info_name, trade_dt, s_dq_close）

依赖
- T05_db_utils: DatabaseConnectionPool, read_dataframe, get_active_db_url
- Y01_db_config: 数据库连接参数
"""
import time
import json
import numpy as np
import pandas as pd
from sqlalchemy import text
from typing import Dict, List, Tuple

from efficient_frontier_API.T02_other_tools import log
from efficient_frontier_API.T05_db_utils import (
    DatabaseConnectionPool,
    read_dataframe,
    get_active_db_url,
    threaded_insert_dataframe,
    create_connection,
)

try:
    from efficient_frontier_API.Y01_db_config import db_type, db_host, db_port, db_name, db_user, db_password
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
    """为 SQLAlchemy 1.2 生成 IN 子句占位符与参数映射。
    返回形如 "(:p0,:p1,...)" 与 {"p0": v0, ...}。
    """
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
        "FROM iis_wght_cnfg_attc_mdl ORDER BY mdl_ver_id ASC LIMIT 1"
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


def run():
    s_t = time.time()
    pool = _db_pool()
    df_annual_stats = pd.DataFrame()  # 新增：初始化年度统计DF

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
        annual_stats_list: List[Dict] = []  # 新增：年度统计列表

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

            # 新增：计算年化收益与波动
            if not cat_ret.empty:
                trading_days = 252.0
                mu = cat_ret.mean()
                sigma = cat_ret.std(ddof=1)
                annual_return = ((1 + mu) ** trading_days) - 1
                annual_vol = sigma * np.sqrt(trading_days)
                annual_stats_list.append({
                    "mdl_ver_id": mdl_ver_id,
                    "aset_bclass_cd": str(aset_cd),
                    "aset_bclass_nm": str(aset_cd),  # 与日收益表保持一致
                    "pct_yld": annual_return,
                    "pct_std": annual_vol,
                    "data_dt": cat_ret.index[-1].date(),
                    "crt_tm": pd.to_datetime("now"),
                })

            tmp = pd.DataFrame({
                "mdl_ver_id": mdl_ver_id,
                "aset_bclass_cd": str(aset_cd),
                "aset_bclass_nm": str(aset_cd),
                "pct_yld_date": cat_ret.index.date,
                "pct_yld": cat_ret.values.astype(float),
            })
            out_frames.append(tmp)

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
        # 1) 清空目标表
        log("清空目标表 iis_mdl_aset_pct_d, iis_aset_allc_indx_rtrn ...")
        # 使用单连接执行 TRUNCATE，无需连接池上下文
        single_conn = create_connection(pool.engine.url)
        try:
            single_conn.execute(text("TRUNCATE TABLE iis_mdl_aset_pct_d"))
            single_conn.execute(text("TRUNCATE TABLE iis_aset_allc_indx_rtrn"))
        finally:
            single_conn.close()

        # 2) 分批并发插入
        log("将大类收益率和年度统计数据分批并发插入 ...")
        datasets = []
        for aset_cd, sub in result_df.groupby("aset_bclass_cd"):
            datasets.append({
                "dataframe": sub,
                "table": "iis_mdl_aset_pct_d",
                "batch_size": 2000,
                "method": "multi",
            })

        if not df_annual_stats.empty:
            datasets.append({
                "dataframe": df_annual_stats,
                "table": "iis_aset_allc_indx_rtrn",
                "batch_size": 200,
                "method": "multi",
            })

        threaded_insert_dataframe(pool, datasets, max_workers=4)
        log(f"大类收益率数据拟合插入完成，耗时 {time.time() - s_t:.2f} 秒")
    except Exception as e:
        return json.dumps({
            "code": 1,
            "msg": f"结果入库失败: {e}"
        }, ensure_ascii=False)
    return json.dumps({
        "code": 0,
        "msg": ""
    }, ensure_ascii=False)


if __name__ == '__main__':
    result_json = run()
    print(result_json)
