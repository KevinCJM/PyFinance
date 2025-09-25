# -*- coding: utf-8 -*-
"""
Z03_insert_test_data.py

用途
- 通过 Python 模拟生成测试数据，并写入以下三张表（表结构见 Z02_crate_table_ddl.sql）：
  1) iis_wght_cnfg_attc_mdl（模型附件信息，插入 2 行）
  2) iis_wght_cnfg_mdl（指数拟合权重配置，插入 2 组不同模型的多行记录）
  3) iis_mdl_aset_pct_d（大类每日收益输入，插入 1 组模型的 2 年日度数据）

依赖
- 使用 T05_db_utils.DatabaseConnectionPool 与 insert_dataframe 完成批量写入。
- 需要 SQLAlchemy 及目标数据库驱动（如 MySQL 使用 PyMySQL）。

连接
- 通过环境变量 DB_URL 指定数据库连接串（SQLAlchemy 标准格式），例如：
  mysql+pymysql://user:password@127.0.0.1:3306/your_db?charset=utf8mb4

运行
- python Z03_insert_test_data.py  （请先在环境中配置 DB_URL）
"""

import os
import sys
import random
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple
from sqlalchemy import text

import numpy as np
import pandas as pd

from efficient_frontier_API.T05_db_utils import DatabaseConnectionPool, insert_dataframe, get_active_db_url

try:
    from efficient_frontier_API.Y01_db_config import db_type, db_host, db_port, db_name, db_user, db_password
except Exception:
    raise RuntimeError("请先在 Y01_db_config.py 中配置数据库连接参数")

ASSET_CLASSES = [
    ("EQ", "权益"),
    ("FI", "固收"),
    ("MM", "货币"),
    ("MIX", "混合"),
    ("ALT", "另类"),
]


def _today() -> date:
    return datetime.now().date()


def _mk_dates(start: date, end: date) -> List[date]:
    cur = start
    out: List[date] = []
    while cur <= end:
        out.append(cur)
        cur = cur + timedelta(days=1)
    return out


def build_cfg_attc_models() -> pd.DataFrame:
    """构造 iis_wght_cnfg_attc_mdl 的 2 行示例数据。

    要求字段：mdl_ver_id, mdl_nm, mdl_st, cal_strt_dt, cal_end_dt
    一行 cal_end_dt 非空；一行 cal_end_dt 为空。
    """
    start_dt = date(2023, 1, 1)
    on_dt = date(2023, 6, 1)
    off_dt = date(2024, 6, 1)
    rows = [
        {
            "mdl_ver_id": "MDL_SIM_001",
            "mdl_nm": "模拟模型-001",
            "mdl_st": "2",  # 1待审核；2已上线；3已下线
            "cal_strt_dt": start_dt,
            "cal_end_dt": date(2024, 12, 31),
            "on_ln_dt": on_dt,
            "off_ln_dt": off_dt,
        },
        {
            "mdl_ver_id": "MDL_SIM_002",
            "mdl_nm": "模拟模型-002",
            "mdl_st": "1",
            "cal_strt_dt": start_dt,
            "cal_end_dt": None,  # 按要求：一行结束日期为空
            "on_ln_dt": None,
            "off_ln_dt": None,
        },
    ]
    return pd.DataFrame(rows)


def _random_partition(n_parts: int, *, seed: int = 0) -> List[float]:
    """将 1.0 随机分割为 n_parts 份，返回各份权重，保留 7 位小数。

    用于生成同一大类下多个指数的权重，确保和为 1。
    """
    rng = random.Random(seed)
    cuts = sorted(rng.random() for _ in range(n_parts - 1))
    weights = []
    last = 0.0
    for c in cuts + [1.0]:
        w = max(0.0, c - last)
        weights.append(w)
        last = c
    s = sum(weights) or 1.0
    weights = [w / s for w in weights]
    # 四舍五入到 7 位小数，保证与表字段 (11,7) 对齐
    weights = [round(w, 7) for w in weights]
    # 调整舍入误差：令最后一项补齐到 1.0
    diff = round(1.0 - sum(weights), 7)
    weights[-1] = round(weights[-1] + diff, 7)
    return weights


def build_cnfg_mdl_rows(mdl_ver_id: str, seed: int = 0) -> pd.DataFrame:
    """构造 iis_wght_cnfg_mdl 的多行数据（覆盖 5 大类，每类 1~4 个指数）。"""
    rows = []
    ts_now = datetime.now()
    rng = random.Random(seed)
    for (code, name) in ASSET_CLASSES:
        n_idx = rng.randint(1, 4)
        weights = _random_partition(n_idx, seed=seed + hash(code) % 10000)
        for i in range(n_idx):
            rows.append({
                "mdl_ver_id": mdl_ver_id,
                "aset_bclass_cd": code,
                "indx_num": f"{code}_IDX_{i + 1:02d}",
                "indx_nm": f"{name}指数{i + 1}",
                "wght": float(weights[i]),
                "crt_tm": ts_now,
            })
    return pd.DataFrame(rows)


def build_pct_d_rows(mdl_ver_id: str, start: date, end: date, seed: int = 0) -> pd.DataFrame:
    """构造 iis_mdl_aset_pct_d 的日度收益数据（5 大类，各 2 年）。"""
    rng = np.random.RandomState(seed)
    dates = _mk_dates(start, end)
    # 为每个大类设置一个不同的均值/波动
    profile: Dict[str, Tuple[float, float]] = {
        "权益": (0.0003, 0.01),
        "固收": (0.0001, 0.002),
        "货币": (0.00005, 0.0005),
        "混合": (0.0002, 0.006),
        "另类": (0.00015, 0.004),
    }
    rows = []
    for code, name in ASSET_CLASSES:
        mu, sigma = profile.get(name, (0.0001, 0.003))
        # 生成日收益，控制在小数点后 6-8 位范围
        rets = rng.normal(loc=mu, scale=sigma, size=len(dates))
        # 防止极端值
        rets = np.clip(rets, -0.2, 0.2)
        for dt, r in zip(dates, rets):
            rows.append({
                "mdl_ver_id": mdl_ver_id,
                "aset_bclass_cd": code,
                "aset_bclass_nm": name,
                "pct_yld_date": dt,
                "pct_yld": float(round(float(r), 8)),
            })
    df = pd.DataFrame(rows)
    # 确保写入顺序稳定
    df = df.sort_values(["aset_bclass_cd", "pct_yld_date"]).reset_index(drop=True)
    return df


def main() -> None:
    # 连接串优先级：环境变量 DB_URL > 基于配置自动构造
    db_url = os.environ.get("DB_URL") or get_active_db_url(
        db_type=db_type,
        db_user=db_user,
        db_password=db_password,
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
    )
    if not db_url:
        print("[ERROR] 未找到数据库连接串。请设置环境变量 DB_URL 或在 Y01_db_config.py 中提供 db_url。")
        sys.exit(2)

    pool = DatabaseConnectionPool(url=db_url, pool_size=2)
    print("[INFO] Using DB URL from {}".format(
        "ENV(DB_URL)" if os.environ.get("DB_URL") else "config(auto-detect host)"
    ))

    # 清空相关表，避免主键重复/脏数据影响
    tables_to_truncate = [
        "iis_wght_cnfg_attc_mdl",
        "iis_wght_cnfg_mdl",
        "iis_fnd_indx_info",
        "wind_cmfindexeod",
        "iis_mdl_aset_pct_d",
    ]
    try:
        with pool.begin() as conn:
            for t in tables_to_truncate:
                try:
                    conn.execute(text(f"TRUNCATE TABLE {t}"))
                    print(f"[OK] TRUNCATE {t}")
                except Exception as e:
                    print(f"[WARN] TRUNCATE {t} 失败，尝试 DELETE: {e}")
                    conn.execute(text(f"DELETE FROM {t}"))
                    print(f"[OK] DELETE {t}")
    except Exception as e:
        print(f"[ERROR] 预清理表失败: {e}")
        sys.exit(2)

    # 1) iis_wght_cnfg_attc_mdl: 2 行
    try:
        df_attc = build_cfg_attc_models()
        insert_dataframe(pool, df_attc, table="iis_wght_cnfg_attc_mdl")
        print(f"[OK] 写入 iis_wght_cnfg_attc_mdl: {len(df_attc)} 行")
    except Exception as e:
        print(f"[ERR] 插入 iis_wght_cnfg_attc_mdl 失败: {e}")

    # 2) iis_wght_cnfg_mdl: 两个不同模型（每个模型覆盖 5 大类，每类 1~4 指数）
    try:
        df_cfg_1 = build_cnfg_mdl_rows("MDL_SIM_001", seed=123)
        df_cfg_2 = build_cnfg_mdl_rows("MDL_SIM_002", seed=456)
        insert_dataframe(pool, df_cfg_1, table="iis_wght_cnfg_mdl")
        insert_dataframe(pool, df_cfg_2, table="iis_wght_cnfg_mdl")
        print(f"[OK] 写入 iis_wght_cnfg_mdl: {len(df_cfg_1) + len(df_cfg_2)} 行")
    except Exception as e:
        print(f"[ERR] 插入 iis_wght_cnfg_mdl 失败: {e}")

    # 2.1) iis_fnd_indx_info: 基于配置插入指数来源信息
    try:
        df_cfg_all = pd.concat([df_cfg_1, df_cfg_2], ignore_index=True)
        uniq = df_cfg_all[["indx_num", "indx_nm"]].drop_duplicates().copy()
        df_indx_info = uniq.assign(
            indx_rmk='测试数据', src_tab_ennm='wind_cmfindexeod', src_tab_cnnm='中国共同基金指数行情'
        )
        insert_dataframe(pool, df_indx_info, table="iis_fnd_indx_info", batch_size=1000)
        print(f"[OK] 写入 iis_fnd_indx_info: {len(df_indx_info)} 行")
    except Exception as e:
        print(f"[ERR] 插入 iis_fnd_indx_info 失败: {e}")

    # 3) iis_mdl_aset_pct_d: 选择一个模型（MDL_SIM_001），五大类，各 2 年日度数据
    end_dt = _today() - timedelta(days=1)
    start_dt = date(end_dt.year - 2, end_dt.month, end_dt.day)

    # 3.1) wind_cmfindexeod: 构造指数行情
    try:
        codes = uniq.reset_index(drop=True)
        dates = _mk_dates(start_dt, end_dt)
        rows = []
        obj_id = 1
        rng = np.random.RandomState(2025)
        for _, r in codes.iterrows():
            code = str(r['indx_num'])
            name = str(r['indx_nm'])
            n = len(dates)
            rets = rng.normal(0.0002, 0.01, size=n)
            prices = 100.0 * np.cumprod(1.0 + rets)
            for i, dt in enumerate(dates):
                close = float(round(prices[i], 4))
                preclose = float(round(prices[i - 1] if i > 0 else prices[i], 4))
                spread1 = float(rng.normal(0, 0.002))
                spread2 = abs(float(rng.normal(0, 0.003)))
                open_px = float(round(close * (1.0 + spread1), 4))
                high_px = float(round(max(open_px, close) * (1.0 + spread2), 4))
                low_px = float(round(min(open_px, close) * (1.0 - spread2), 4))
                vol = int(rng.randint(1_000_000, 5_000_000))
                amt = float(round(close * vol, 2))
                rows.append({
                    'object_id': obj_id,
                    's_info_windcode': code,
                    's_info_name': name,
                    'trade_dt': dt,
                    's_dq_preclose': preclose,
                    's_dq_open': open_px,
                    's_dq_high': high_px,
                    's_dq_low': low_px,
                    's_dq_close': close,
                    's_dq_volume': vol,
                    's_dq_amount': amt,
                })
                obj_id += 1
        df_wind = pd.DataFrame(rows)
        # wind_cmfindexeod.trade_dt 多为 CHAR(8)/VARCHAR(8) 或 INT(8)，统一格式化为 'YYYYMMDD'
        df_wind['trade_dt'] = pd.to_datetime(df_wind['trade_dt']).dt.strftime('%Y%m%d')
        insert_dataframe(pool, df_wind, table="wind_cmfindexeod", batch_size=2000)
        print(f"[OK] 写入 wind_cmfindexeod: {len(df_wind)} 行 | 指数数={df_wind['s_info_windcode'].nunique()}")
    except Exception as e:
        print(f"[ERR] 插入 wind_cmfindexeod 失败: {e}")

    try:
        df_pct_d = build_pct_d_rows("MDL_SIM_001", start=start_dt, end=end_dt, seed=2024)
        insert_dataframe(pool, df_pct_d, table="iis_mdl_aset_pct_d", batch_size=2000)
        print(f"[OK] 写入 iis_mdl_aset_pct_d: {len(df_pct_d)} 行")
    except Exception as e:
        print(f"[ERR] 插入 iis_mdl_aset_pct_d 失败: {e}")

    print("[DONE] 测试数据插入完成。")


if __name__ == "__main__":
    main()
