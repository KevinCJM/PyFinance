# -*- encoding: utf-8 -*-
"""
@File: A06_cal_market_ef.py
@Modify Time: 2025/09/23 14:59       
@Author: Kevin-Chen
@Descriptions: 计算全市场有效前沿以及C1~C6的权重点位置
"""
import time
import json
import traceback
import numpy as np
import pandas as pd
from sqlalchemy import text
from datetime import date as _date
from typing import Optional, List, Tuple, Dict

try:
    from countus.efficient_frontier_API.T04_show_plt import plot_efficient_frontier
    from countus.efficient_frontier_API.T02_other_tools import log, compute_performance_numba
    from countus.efficient_frontier_API.T01_generate_random_weights import (multi_level_random_walk_config,
                                                                            compute_perf_arrays)
    from countus.efficient_frontier_API.Y02_asset_id_map import (asset_to_weight_column_map, aset_cd_nm_dict,
                                                                 rsk_level_code_dict)
    from countus.efficient_frontier_API.T05_db_utils import (DatabaseConnectionPool, get_active_db_url, read_dataframe,
                                                             create_connection, threaded_upsert_dataframe_mysql,
                                                             threaded_insert_dataframe)
    from countus.efficient_frontier_API.Y01_db_config import (db_type, db_host, db_port, db_name, db_user,
                                                              db_password)
except ImportError:
    from efficient_frontier_API.T04_show_plt import plot_efficient_frontier
    from efficient_frontier_API.T02_other_tools import log, compute_performance_numba
    from efficient_frontier_API.T01_generate_random_weights import (multi_level_random_walk_config, compute_perf_arrays)
    from efficient_frontier_API.Y02_asset_id_map import (asset_to_weight_column_map, aset_cd_nm_dict,
                                                         rsk_level_code_dict)
    from efficient_frontier_API.T05_db_utils import (DatabaseConnectionPool, get_active_db_url, read_dataframe,
                                                     create_connection, threaded_upsert_dataframe_mysql,
                                                     threaded_insert_dataframe)
    from efficient_frontier_API.Y01_db_config import (db_type, db_host, db_port, db_name, db_user, db_password)

# --- Constants ---
RANDOM_SEED = 12345
ROUNDS_CONFIG = {
    0: {"init_mode": "exploration", "samples": 300, "step_size": 0.99},
    1: {"samples_total": 1000, "step_size": 0.1, "vol_bins": 100, "parallel_workers": 100},
    2: {"samples_total": 2000, "step_size": 0.8, "vol_bins": 200, "parallel_workers": 100},
    3: {"samples_total": 3000, "step_size": 0.05, "vol_bins": 300, "parallel_workers": 100},
    4: {"samples_total": 4000, "step_size": 0.01, "vol_bins": 400, "parallel_workers": 100},
}
TRADING_DAYS = 252.0
DEDUP_DECIMALS = 2
EXTREME_SEED_CONFIG = {"enable": False}
RISK_METRIC = "var"
VAR_PARAMS = {
    "confidence": 0.95, "horizon_days": 1.0, "return_type": "log",
    "ddof": 1, "clip_non_negative": True,
}
PRECISION_CHOICE: Optional[str] = None
SLSQP_REFINE = {"enable": False, "n_grid": 1000}


# --- Helper functions ---

def _parse_series_from_dict(d):
    """将 {date->nav} 字典解析为 Series，日期自动解析，按时间升序，值为 float。"""
    s = pd.Series(d)
    try:
        idx = pd.to_datetime(s.index, format="%Y%m%d")
    except Exception:
        idx = pd.to_datetime(s.index)
    s.index = idx
    s = pd.to_numeric(s, errors='coerce').astype(float)
    return s.dropna().sort_index()


def _load_returns_from_nv(nv_dict, asset_list):
    """从净值数据字典加载收益率"""
    ser_list = []
    for asset in asset_list:
        if asset not in nv_dict:
            raise ValueError(f"缺少大类 '{asset}' 的净值数据(nv)")
        v = nv_dict[asset]
        if not isinstance(v, dict):
            raise ValueError(f"nv['{asset}'] 的格式应为 {{date->nav}} 字典")
        ser_list.append(_parse_series_from_dict(v).rename(asset))
    df_nv = pd.concat(ser_list, axis=1, join='inner')
    if df_nv.empty or len(df_nv) < 2:
        raise ValueError("无法从净值数据计算收益率，可能是日期无重叠或数据点不足")

    last_nav_date = df_nv.index.max().date()
    df_ret = df_nv.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how='any')
    return df_ret, last_nav_date


def fetch_default_mdl_ver_id() -> Tuple[str, Optional[_date], Optional[_date]]:
    """从 iis_wght_cnfg_attc_mdl 表中获取第一个 mdl_ver_id、cal_strt_dt、cal_end_dt。"""
    db_url = get_active_db_url(
        db_type=db_type, db_user=db_user, db_password=db_password,
        db_host=db_host, db_port=db_port, db_name=db_name,
    )
    sql = text("SELECT mdl_ver_id, cal_strt_dt, cal_end_dt FROM iis_wght_cnfg_attc_mdl "
               "WHERE mdl_st = '2' ORDER BY mdl_ver_id ASC LIMIT 1")
    conn = create_connection(db_url)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    if df.empty:
        raise RuntimeError("数据库中未找到可用的模型版本（iis_wght_cnfg_attc_mdl 为空）")
    row = df.iloc[0]
    mdl = str(row["mdl_ver_id"]) if "mdl_ver_id" in df.columns else str(row[0])

    def _to_date(v):
        if pd.isna(v):
            return None
        return v.date() if hasattr(v, 'date') else v

    s_dt = _to_date(row.get("cal_strt_dt"))
    e_dt = _to_date(row.get("cal_end_dt"))
    return mdl, s_dt, e_dt


def fetch_returns_from_db(mdl_ver_id: str, start_dt: Optional[_date] = None, end_dt: Optional[_date] = None) -> Tuple[
    List[str], pd.DataFrame, Dict[str, str]]:
    """从数据库 iis_mdl_aset_pct_d 表读取指定模型的五大类日收益数据。"""
    db_url = get_active_db_url(
        db_type=db_type, db_user=db_user, db_password=db_password,
        db_host=db_host, db_port=db_port, db_name=db_name,
    )
    pool = DatabaseConnectionPool(url=db_url, pool_size=4)
    where_parts = ["mdl_ver_id = :mdl_ver_id"]
    params = {"mdl_ver_id": mdl_ver_id}
    if start_dt:
        where_parts.append("pct_yld_date >= :start_dt")
        params["start_dt"] = start_dt
    if end_dt:
        where_parts.append("pct_yld_date <= :end_dt")
        params["end_dt"] = end_dt
    where_sql = " WHERE " + " AND ".join(where_parts)
    sql_series = text(
        "SELECT aset_bclass_cd AS cd, aset_bclass_nm AS nm, pct_yld_date AS dt, pct_yld AS yld FROM iis_mdl_aset_pct_d" + where_sql)
    df = read_dataframe(pool, sql_series, params=params)
    if df.empty:
        raise RuntimeError(f"数据库中未查询到模型 {mdl_ver_id} 的收益数据")

    df['nm'] = df['nm'].replace('', np.nan)
    missing_nm_mask = df['nm'].isna()
    if missing_nm_mask.any():
        log(f"发现 {missing_nm_mask.sum()} 行的资产名称为空，将使用代码进行映射。")
        df.loc[missing_nm_mask, 'nm'] = df.loc[missing_nm_mask, 'cd'].map(aset_cd_nm_dict)

    df["nm"] = df["nm"].astype(str)
    df["cd"] = df["cd"].astype(str)
    code_to_name_map = df[["cd", "nm"]].dropna().drop_duplicates().set_index("cd")["nm"].to_dict()
    asset_list = sorted(df["nm"].unique().tolist())
    df["dt"] = pd.to_datetime(df["dt"])
    return_df = df.pivot_table(index="dt", columns="nm", values="yld", aggfunc="first").sort_index().dropna(how='any')
    return_df = return_df.reindex(columns=asset_list)
    return asset_list, return_df, code_to_name_map


def fetch_standard_portfolios(pool: DatabaseConnectionPool, mdl_ver_id: str, code_to_name_map: Dict[str, str]) -> List[
    Tuple[int, Dict[str, float]]]:
    """从数据库读取C1-C6标准组合的权重。"""
    log(f"从数据库为模型 {mdl_ver_id} 获取标准组合权重...")
    sql = "SELECT aset_bclass_cd, rsk_lvl, wght FROM iis_wght_cnfg_mdl_ast_rsk_rel WHERE mdl_ver_id = :mdl"
    df = read_dataframe(pool, sql, params={"mdl": mdl_ver_id})
    if df.empty:
        log(f"警告: 在 iis_wght_cnfg_mdl_ast_rsk_rel 中未找到模型 {mdl_ver_id} 的标准组合权重。")
        return []

    portfolios = {}
    for _, row in df.iterrows():
        try:
            rsk_lvl, asset_code, weight = int(row['rsk_lvl']), row['aset_bclass_cd'], float(row['wght'])
        except (ValueError, TypeError) as e:
            log(f"警告: 解析标准组合权重行时出错，已跳过: {row}, 错误: {e}")
            continue
        if rsk_lvl not in portfolios:
            portfolios[rsk_lvl] = {}
        asset_name = code_to_name_map.get(asset_code)
        if asset_name:
            portfolios[rsk_lvl][asset_name] = weight
        else:
            log(f"警告: 无法为资产代码 '{asset_code}' 找到对应的资产名称。")
    std_portfolios = sorted(portfolios.items())
    log(f"成功从数据库加载 {len(std_portfolios)} 个标准组合。")
    return std_portfolios


def main(json_input: str) -> str:
    """
    主函数，用于计算全市场有效前沿及C1-C6标准组合点位, 并选择性绘图和入库。
    :param json_input: str, 包含资产配置和分析参数的JSON字符串
    :return: str, 包含计算结果或错误信息的JSON字符串
    """
    try:
        log("步骤 1: 解析输入参数...")
        if isinstance(json_input, dict):
            input_data = json_input.get('in_data', json_input)
        else:
            input_data = json.loads(json_input).get('in_data', json.loads(json_input))

        insert_table = input_data.get("insert_table", False)
        mdl_ver_id = None

        if insert_table:
            log("落表模式: 从数据库加载数据...")
            mdl_ver_id, cal_sta_dt, cal_end_dt = fetch_default_mdl_ver_id()
            log(f"使用默认在线模型: {mdl_ver_id}")

            asset_list, return_df, code_to_name_map = fetch_returns_from_db(mdl_ver_id, start_dt=cal_sta_dt,
                                                                            end_dt=cal_end_dt)
            returns = return_df.values
            last_nav_date = return_df.index.max().date()

            db_url = get_active_db_url(db_type=db_type, db_user=db_user, db_password=db_password, db_host=db_host,
                                       db_port=db_port, db_name=db_name)
            pool = DatabaseConnectionPool(url=db_url)
            std_portfolios_from_db = fetch_standard_portfolios(pool, mdl_ver_id, code_to_name_map)

            standard_proportion = {}
            lvl_to_name_map = {v: k for k, v in rsk_level_code_dict.items()}
            for lvl, wdict in std_portfolios_from_db:
                rsk_name = lvl_to_name_map.get(lvl)
                if rsk_name:
                    standard_proportion[rsk_name] = wdict
        else:
            log("API模式: 从JSON输入加载数据...")
            asset_list_codes = input_data["asset_list"]
            nv_data = input_data["nv"]
            standard_proportion_codes = input_data["StandardProportion"]

            asset_list = [aset_cd_nm_dict.get(code, code) for code in asset_list_codes]
            nv_data_named = {aset_cd_nm_dict.get(code, code): v for code, v in nv_data.items()}

            standard_proportion = {}
            for rsk_name, w_dict_codes in standard_proportion_codes.items():
                standard_proportion[rsk_name] = {aset_cd_nm_dict.get(code, code): w for code, w in w_dict_codes.items()}

            return_df, last_nav_date = _load_returns_from_nv(nv_data_named, asset_list)
            returns = return_df.values

        log("步骤 3: 计算无约束的市场组合有效前沿...")
        single_limit = [(0.0, 1.0)] * len(asset_list)
        (W_unc, ret_annual_unc, risk_arr_unc, ef_mask_unc) = multi_level_random_walk_config(
            port_daily_returns=returns,
            single_limits=single_limit,
            multi_limits={},
            rounds_config=ROUNDS_CONFIG,
            dedup_decimals=DEDUP_DECIMALS,
            annual_trading_days=TRADING_DAYS,
            global_seed=RANDOM_SEED,
            extreme_seed_config=EXTREME_SEED_CONFIG,
            risk_metric=RISK_METRIC,
            var_params=VAR_PARAMS,
            precision_choice=PRECISION_CHOICE,
            slsqp_refine_config=SLSQP_REFINE
        )
        log(f"无约束市场组合的随机权重计算完成. 权重数: {W_unc.shape[0]}")

        log("步骤 4: 计算C1-C6标准组合的点位...")
        standard_points = []
        for name, w_dict in standard_proportion.items():
            weights = np.array([w_dict.get(asset, 0.0) for asset in asset_list])
            w_2d = np.asarray(weights, dtype=np.float64).reshape(1, -1)
            ret, vol, var, sharpe = compute_performance_numba(returns, w_2d, TRADING_DAYS, 1, 1.645)
            point_data = {
                'name': name,
                'weights': weights.tolist(),
                'ret_annual': float(ret[0]),
                'vol_annual': float(vol[0]),
                'var_value': float(var[0]),
                'sharpe_ratio': float(sharpe[0])
            }
            standard_points.append(point_data)
            log(f"计算完成: {name} - 收益率: {ret[0]:.4f}, 波动率: {vol[0]:.4f}, VaR: {var[0]:.4f}, 夏普比率: {sharpe[0]:.4f}")

        log("步骤 5: 格式化数据...")
        ef_mask = ef_mask_unc
        ef_weights = W_unc[ef_mask]
        ef_ret = ret_annual_unc[ef_mask]
        ef_risk = risk_arr_unc[ef_mask]

        _, ef_vol = compute_perf_arrays(returns, ef_weights, TRADING_DAYS, return_type="log")
        ef_sharpe = np.divide(ef_ret, ef_vol, out=np.zeros_like(ef_ret), where=ef_vol != 0)

        ef_df = pd.DataFrame(ef_weights, columns=asset_list)
        ef_df['rate'] = ef_ret
        ef_df['shrp_prprtn'] = ef_sharpe
        ef_df['var95_b'] = ef_risk
        ef_df['var95'] = np.clip(ef_risk, 0, None)
        ef_df['liquid'] = ef_vol  # 添加年化波动率

        ef_df = ef_df.sort_values(by='rate', ascending=True).reset_index(drop=True)
        ef_df['rsk_lvl'] = np.arange(11, 11 + len(ef_df))

        std_df_list = []
        for point in standard_points:
            rsk_name = point['name']
            rsk_lvl = rsk_level_code_dict.get(rsk_name)
            if rsk_lvl is None: continue

            row = {
                'rsk_lvl': rsk_lvl,
                'rate': point['ret_annual'],
                'shrp_prprtn': point['sharpe_ratio'],
                'var95_b': point['var_value'],
                'var95': max(0, point['var_value']),
                'liquid': point['vol_annual'],  # 添加年化波动率
            }
            for i, asset_name in enumerate(asset_list):
                row[asset_name] = point['weights'][i]
            std_df_list.append(row)
        std_df = pd.DataFrame(std_df_list)

        final_df = pd.concat([std_df, ef_df], ignore_index=True)

        rename_map = {}
        for code, name in aset_cd_nm_dict.items():
            target_col = asset_to_weight_column_map.get(name)
            if target_col:
                rename_map[name] = target_col
        final_df.rename(columns=rename_map, inplace=True)

        output_cols = [
            'rsk_lvl', 'rate', 'liquid', 'csh_mgt_typ_pos', 'fx_yld_pos',
            'mix_strg_typ_pos', 'eqty_invst_typ_pos', 'altnt_invst_pos',
            'shrp_prprtn', 'var95', 'var95_b'
        ]
        for col in output_cols:
            if col not in final_df.columns:
                final_df[col] = 0.0
        result_df = final_df[output_cols]

        if insert_table:
            log("步骤 6: 执行数据落表操作...")
            db_df = result_df.copy()  # 使用已经整理好列的 result_df
            db_df['mdl_ver_id'] = mdl_ver_id
            db_df['is_efct_font'] = '1'
            db_df['dt_dt'] = last_nav_date
            db_df['crt_tm'] = pd.to_datetime('now')

            db_url = get_active_db_url(db_type=db_type, db_user=db_user, db_password=db_password,
                                       db_host=db_host, db_port=db_port, db_name=db_name)
            pool = DatabaseConnectionPool(url=db_url)

            log(f"删除并重新插入 iis_ef_rndm_srch_wght 表中模型 {mdl_ver_id} 的数据...")
            with pool.begin() as conn:
                try:
                    conn.execute(
                        text("DELETE FROM iis_ef_rndm_srch_wght WHERE mdl_ver_id = :mdl_ver_id"),
                        {"mdl_ver_id": mdl_ver_id}
                    )
                except Exception as e:
                    log(f"DELETE FROM iis_ef_rndm_srch_wght 失败: {e}")
                    raise
            log("\t iis_ef_rndm_srch_wght 表的数据已经删除。")
            threaded_insert_dataframe(pool, [{'dataframe': db_df, 'table': 'iis_ef_rndm_srch_wght'}])
            log("\t 数据已写入 iis_ef_rndm_srch_wght 表。")

            return json.dumps({"code": 0, "msg": ""}, ensure_ascii=False)

        result_data = result_df.to_dict(orient='records')
        success_response = {
            "code": 0,
            "msg": "",
            "data": result_data
        }
        return json.dumps(success_response, ensure_ascii=False, indent=2)

    except Exception as e:
        log(f"{traceback.format_exc()}")
        error_response = {
            "code": 1,
            "msg": f"INTERNAL_SERVER_ERROR, 计算过程中发生错误: {type(e).__name__} - {e}"
        }
        return json.dumps(error_response, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    df = pd.read_excel("历史净值数据.xlsx")
    df = df[['date', '货基指数', '固收类', '混合类', '权益类', '另类']]
    df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")
    df['date'] = df['date'].dt.strftime("%Y%m%d")
    df = df.set_index('date')

    in_dict = {
        "in_data": {
            # "insert_table": True,
            "asset_list": [
                "01", "02", "03", "04", "05"
            ],
            "nv": {
                "01": df['货基指数'].to_dict(),
                "02": df['固收类'].to_dict(),
                "03": df['混合类'].to_dict(),
                "04": df['权益类'].to_dict(),
                "05": df['另类'].to_dict()
            },
            "StandardProportion": {
                "C1": {"01": 1.00, "02": 0.00, "03": 0.00, "04": 0.00, "05": 0.00},
                "C2": {"01": 0.20, "02": 0.80, "03": 0.00, "04": 0.00, "05": 0.00},
                "C3": {"01": 0.10, "02": 0.55, "03": 0.35, "04": 0.00, "05": 0.00},
                "C4": {"01": 0.05, "02": 0.40, "03": 0.30, "04": 0.20, "05": 0.05},
                "C5": {"01": 0.05, "02": 0.20, "03": 0.25, "04": 0.40, "05": 0.10},
                "C6": {"01": 0.05, "02": 0.10, "03": 0.15, "04": 0.60, "05": 0.10}
            }
        }}
    json_str_input = json.dumps(in_dict, ensure_ascii=False)
    with open("sample_A06_input.json", "w", encoding="utf-8") as f:
        f.write(json_str_input)

    s_t = time.time()
    res_json = main(json_str_input)
    print(res_json)
    log(f"计算总耗时: {time.time() - s_t:.2f} 秒")

    with open("sample_A06_output.json", "w", encoding="utf-8") as f:
        f.write(res_json)
