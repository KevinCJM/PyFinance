# -*- coding: utf-8 -*-
"""
B02_construct_category_yield.py

根据模型配置汇总指数来源，并从 wind_cmfindexeod 抽取成分指数的净值数据。

步骤
1) 从 iis_wght_cfg_attc_mdl 取第一条记录：mdl_ver_id, on_ln_dt, off_ln_dt
2) 读取 iis_wght_cnfg_mdl 中该 mdl_ver_id 的配置：mdl_ver_id, aset_bclass_cd, indx_num, indx_nm, wght
3) 在 iis_fnd_indx_info 中查询这些 indx_num 的来源表：indx_num, src_tab_enmm
4) 如果存在 src_tab_enmm != 'wind_cmfindexeod' 的记录，抛出错误
5) 合并所有 src_tab_enmm 为 'wind_cmfindexeod' 的 indx_num，
   到 wind_cmfindexeod 中查询 s_info_windcode in (indx_num) 的数据（s_info_windcode, s_info_name, trade_dt, s_dq_close）

依赖
- T05_db_utils: DatabaseConnectionPool, read_dataframe, get_active_db_url
- Y01_db_config: 数据库连接参数
"""

from typing import Dict, List, Tuple

import pandas as pd

from T05_db_utils import DatabaseConnectionPool, read_dataframe, get_active_db_url

try:
    from Y01_db_config import db_type, db_host, db_port, db_name, db_user, db_password  # type: ignore
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
    sql = (
        "SELECT mdl_ver_id, on_ln_dt, off_ln_dt "
        "FROM iis_wght_cfg_attc_mdl ORDER BY mdl_ver_id ASC LIMIT 1"
    )
    df = read_dataframe(pool, sql)
    if df.empty:
        raise RuntimeError("iis_wght_cfg_attc_mdl 表无数据")
    return df.iloc[0]


def fetch_model_config(pool: DatabaseConnectionPool, mdl_ver_id: str) -> pd.DataFrame:
    sql = (
        "SELECT mdl_ver_id, aset_bclass_cd, indx_num, indx_nm, wght "
        "FROM iis_wght_cnfg_mdl WHERE mdl_ver_id = :mdl"
    )
    df = read_dataframe(pool, sql, params={"mdl": mdl_ver_id})
    if df.empty:
        raise RuntimeError(f"iis_wght_cnfg_mdl 中无模型 {mdl_ver_id} 的配置")
    return df


def fetch_index_sources(pool: DatabaseConnectionPool, codes: List[str]) -> pd.DataFrame:
    if not codes:
        return pd.DataFrame(columns=["indx_num", "src_tab_enmm"])  # 空
    in_clause, params = _build_in_clause("p", list(codes))
    sql = (
        "SELECT indx_num, src_tab_enmm FROM iis_fnd_indx_info "
        f"WHERE indx_num IN {in_clause}"
    )
    return read_dataframe(pool, sql, params=params)


def fetch_wind_index_eod(pool: DatabaseConnectionPool, codes: List[str]) -> pd.DataFrame:
    if not codes:
        raise RuntimeError("无可查询的指数代码")
    in_clause, params = _build_in_clause("w", list(codes))
    sql = (
        "SELECT s_info_windcode, s_info_name, trade_dt, s_dq_close "
        "FROM wind_cmfindexeod "
        f"WHERE s_info_windcode IN {in_clause}"
    )
    return read_dataframe(pool, sql, params=params)


def run() -> pd.DataFrame:
    pool = _db_pool()

    # 1) 模型头信息
    head = fetch_first_model(pool)
    mdl_ver_id = str(head["mdl_ver_id"]) if "mdl_ver_id" in head else str(head.iloc[0])

    # 2) 模型成分与权重
    cfg_df = fetch_model_config(pool, mdl_ver_id)
    codes = sorted(cfg_df["indx_num"].astype(str).unique().tolist())

    # 3) 指数来源表
    src_df = fetch_index_sources(pool, codes)
    if src_df.empty:
        raise RuntimeError("iis_fnd_indx_info 未返回任何指数来源信息")
    invalid_src = src_df["src_tab_enmm"].astype(str).unique().tolist()
    invalid = [s for s in invalid_src if s != 'wind_cmfindexeod']
    if invalid:
        raise RuntimeError(f"存在不支持的数据来源表: {invalid}，仅支持 'wind_cmfindexeod'")

    # 4) 从 wind_cmfindexeod 抽取数据
    df_eod = fetch_wind_index_eod(pool, codes)
    if df_eod.empty:
        raise RuntimeError("wind_cmfindexeod 中未查询到对应指数的数据")

    # 统一数据类型与排序
    df_eod["trade_dt"] = pd.to_datetime(df_eod["trade_dt"])  # 日期
    df_eod = df_eod.sort_values(["s_info_windcode", "trade_dt"]).reset_index(drop=True)
    return df_eod


if __name__ == '__main__':
    df = run()
    print(df.head())
    print(f"共返回 {len(df)} 行 | 指数数: {df['s_info_windcode'].nunique()} | "
          f"日期范围: {df['trade_dt'].min()} ~ {df['trade_dt'].max()}")

