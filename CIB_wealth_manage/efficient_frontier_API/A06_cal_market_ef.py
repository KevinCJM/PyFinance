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
from typing import Optional

try:
    from countus.efficient_frontier_API.T04_show_plt import plot_efficient_frontier
    from countus.efficient_frontier_API.T02_other_tools import log, compute_performance_numba
    from countus.efficient_frontier_API.T01_generate_random_weights import (multi_level_random_walk_config,
                                                                            compute_perf_arrays)
    from countus.efficient_frontier_API.Y02_asset_id_map import (asset_to_weight_column_map, aset_cd_nm_dict,
                                                                 rsk_level_code_dict)
except ImportError:
    from efficient_frontier_API.T04_show_plt import plot_efficient_frontier
    from efficient_frontier_API.T02_other_tools import log, compute_performance_numba
    from efficient_frontier_API.T01_generate_random_weights import (multi_level_random_walk_config, compute_perf_arrays)
    from efficient_frontier_API.Y02_asset_id_map import (asset_to_weight_column_map, aset_cd_nm_dict,
                                                         rsk_level_code_dict)

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
    df_ret = df_nv.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how='any')
    return df_ret.values.astype(np.float64, copy=False)


def main(json_input: str) -> str:
    """
    主函数，用于计算全市场有效前沿及C1-C6标准组合点位, 并选择性绘图。
    :param json_input: str, 包含资产配置和分析参数的JSON字符串
    :return: str, 包含计算结果或错误信息的JSON字符串
    """
    try:
        log("步骤 1: 解析输入参数...")
        if isinstance(json_input, dict):
            input_data = json_input.get('in_data', json_input)
        else:
            input_data = json.loads(json_input).get('in_data', json.loads(json_input))

        asset_list = input_data["asset_list"]
        nv_data = input_data["nv"]
        standard_proportion = input_data["StandardProportion"]

        log("步骤 2: 从净值数据计算日收益率...")
        returns = _load_returns_from_nv(nv_data, asset_list)

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

        log("步骤 5: 格式化返回结果...")

        # a) 处理有效前沿数据
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

        ef_df = ef_df.sort_values(by='rate', ascending=True).reset_index(drop=True)
        ef_df['rsk_lvl'] = np.arange(11, 11 + len(ef_df))

        # b) 处理标准组合数据
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
            }
            for i, asset_code in enumerate(asset_list):
                row[asset_code] = point['weights'][i]
            std_df_list.append(row)
        std_df = pd.DataFrame(std_df_list)

        # c) 合并并重命名列
        final_df = pd.concat([std_df, ef_df], ignore_index=True)

        rename_map = {}
        for code, name in aset_cd_nm_dict.items():
            if code in final_df.columns:
                target_col = asset_to_weight_column_map.get(name)
                if target_col:
                    rename_map[code] = target_col
        final_df.rename(columns=rename_map, inplace=True)

        # d) 选择并排序最终列
        output_cols = [
            'rsk_lvl', 'rate', 'csh_mgt_typ_pos', 'fx_yld_pos',
            'mix_strg_typ_pos', 'eqty_invst_typ_pos', 'altnt_invst_pos',
            'shrp_prprtn', 'var95', 'var95_b'
        ]
        for col in output_cols:
            if col not in final_df.columns:
                final_df[col] = 0.0

        final_df = final_df[output_cols]

        # e) 转换为JSON
        result_data = final_df.to_dict(orient='records')

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
