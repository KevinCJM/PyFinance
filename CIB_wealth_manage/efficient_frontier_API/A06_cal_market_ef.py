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
from typing import Optional, Dict, Any, List

from efficient_frontier_API.T02_other_tools import log
from efficient_frontier_API.T04_show_plt import plot_efficient_frontier
from efficient_frontier_API.T01_generate_random_weights import multi_level_random_walk_config, compute_perf_arrays, \
    compute_var_parametric_arrays

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


# --- Helper functions copied from A01_main_api.py ---

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
    return df_ret.values.astype(np.float32, copy=False)


def calculate_single_port_metrics(weights_array, daily_returns, annual_trading_days, risk_metric, var_params):
    """计算单个组合的指标"""
    w_2d = np.asarray(weights_array).reshape(1, -1)

    ret_annual_arr, vol_annual_arr = compute_perf_arrays(
        port_daily=daily_returns,
        portfolio_allocs=w_2d,
        trading_days=annual_trading_days,
        return_type="log"
    )
    annual_ret = ret_annual_arr[0]

    if (risk_metric or "vol").lower() == "var":
        vp = var_params or {}
        risk_arr = compute_var_parametric_arrays(
            port_daily=daily_returns,
            portfolio_allocs=w_2d,
            confidence=float(vp.get("confidence", 0.95)),
            horizon_days=float(vp.get("horizon_days", 1.0)),
            return_type=str(vp.get("return_type", "log")),
            ddof=int(vp.get("ddof", 1)),
            clip_non_negative=bool(vp.get("clip_non_negative", True)),
        )
        annual_risk = risk_arr[0]
    else:
        annual_risk = vol_annual_arr[0]

    return annual_ret, annual_risk


# --- Plotting helper functions ---

def format_hover_text(row, title, a_list):
    text = f"<b>{title}</b><br>年化收益率: {row['ret_annual']:.2%}<br>年化风险: {row['risk_arr']:.2%}"
    text += "<br><br><b>--权重--</b><br>"
    for asset in a_list:
        if asset in row and pd.notna(row[asset]):
            text += f"{asset}: {row[asset]:.2%}<br>"
    return text.strip('<br>')


def create_plot_data(asset_list, market_ef_data, standard_points):
    """准备绘图所需的数据结构"""
    scatter_points_data = []

    # 1. 无约束的市场组合
    W_unc = np.array(market_ef_data['weights'])
    df_unc = pd.DataFrame(W_unc, columns=asset_list)
    df_unc['ret_annual'] = market_ef_data['ret_annual']
    df_unc['risk_arr'] = market_ef_data['risk_arr']
    df_unc['hover_text'] = df_unc.apply(
        lambda r: format_hover_text(r, "无约束组合", asset_list), axis=1)

    ef_mask_unc = np.array(market_ef_data['ef_mask'])

    # 无约束 - 随机点
    scatter_points_data.append({
        "data": df_unc,
        "name": "无约束-随机点", "color": "black", "size": 2, "opacity": 0.2,
    })
    # 无约束 - 有效前沿
    scatter_points_data.append({
        "data": df_unc[ef_mask_unc],
        "name": "无约束-有效前沿", "color": "blue", "size": 3, "opacity": 0.8,
    })

    # 2. 标准组合点 (C1-C6)
    if standard_points:
        df_standard_points = pd.DataFrame(standard_points)
        df_standard_points = df_standard_points.rename(columns={'risk_value': 'risk_arr'})

        weights_df = pd.DataFrame(df_standard_points['weights'].tolist(), columns=asset_list,
                                  index=df_standard_points.index)
        df_standard_points = pd.concat([df_standard_points, weights_df], axis=1)

        df_standard_points['hover_text'] = df_standard_points.apply(
            lambda r: format_hover_text(r, f"标准-{r['name']}", asset_list), axis=1)

        scatter_points_data.append({
            "data": df_standard_points,
            "name": "标准组合点", "color": "gold", "size": 12, "opacity": 1.0, "symbol": "star",
            "marker_line": dict(width=1, color='black')
        })

    return scatter_points_data


# --- Main API function ---

def main(json_input: str) -> str:
    """
    主函数，用于计算全市场有效前沿及C1-C6标准组合点位, 并选择性绘图。
    :param json_input: str, 包含资产配置和分析参数的JSON字符串
    :return: str, 包含计算结果或错误信息的JSON字符串
    """
    try:
        # 1) 解析输入参数
        log("步骤 1: 解析输入参数...")
        # 判断入参是字典还是字符串
        if isinstance(json_input, dict):
            input_data = json_input['in_data']
        else:
            # Json转字典
            input_data = json.loads(json_input)['in_data']
        asset_list = input_data["asset_list"]
        nv_data = input_data["nv"]
        standard_proportion = input_data["StandardProportion"]
        draw_plt = input_data.get("draw_plt", False)
        draw_plt_filename = input_data.get("draw_plt_filename", None)

        # 2) 从净值数据计算日收益率
        log("步骤 2: 从净值数据计算日收益率...")
        returns = _load_returns_from_nv(nv_data, asset_list)

        # 3) 计算无约束的市场组合有效前沿
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

        market_ef_data = {
            'weights': W_unc.tolist(),
            'ret_annual': ret_annual_unc.tolist(),
            'risk_arr': risk_arr_unc.tolist(),
            'ef_mask': ef_mask_unc.tolist(),
        }

        # 4) 计算C1-C6标准组合的点位
        log("步骤 4: 计算C1-C6标准组合的点位...")
        standard_points = []
        for name, w_dict in standard_proportion.items():
            weights = np.array([w_dict.get(asset, 0.0) for asset in asset_list])
            ret, risk = calculate_single_port_metrics(weights, returns, TRADING_DAYS, RISK_METRIC, VAR_PARAMS)
            point_data = {
                'name': name,
                'weights': weights.tolist(),
                'ret_annual': float(ret),
                'risk_value': float(risk)
            }
            standard_points.append(point_data)
            log(f"计算完成: {name} - 收益率: {ret:.4f}, 风险: {risk:.4f}")

        # 5) 绘图
        if draw_plt:
            log("步骤 5: 准备绘图数据并生成图表...")
            plot_data = create_plot_data(asset_list, market_ef_data, standard_points)
            plot_efficient_frontier(
                plot_data,
                title="全市场有效前沿与标准组合点位",
                x_axis_title=f"年化风险 ({RISK_METRIC.upper()})",
                y_axis_title="年化收益率",
                x_col="risk_arr",
                y_col="ret_annual",
                hover_text_col="hover_text",
                output_filename=draw_plt_filename
            )

        # 6) 封装并返回结果
        log("步骤 6: 封装并返回结果...")
        success_response = {
            "success": True,
            "data": {
                "market_efficient_frontier": market_ef_data,
                "standard_portfolios": standard_points
            }
        }
        return json.dumps(success_response, ensure_ascii=False, indent=2)

    except Exception as e:
        log(traceback.format_exc())
        error_response = {
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": f"计算过程中发生未知错误: {type(e).__name__} - {e}"
        }
        return json.dumps(error_response, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # 读取测试数据
    df = pd.read_excel("历史净值数据.xlsx")
    df = df[['date', '货基指数', '固收类', '混合类', '权益类', '另类']]
    df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")
    df['date'] = df['date'].dt.strftime("%Y%m%d")
    df = df.set_index('date')

    # 模拟输入数据
    in_dict = {
        "in_data": {
            "asset_list": [
                "货币现金类", "固定收益类", "混合策略类", "权益投资类", "另类投资类"
            ],
            "nv": {
                "货币现金类": df['货基指数'].to_dict(),
                "固定收益类": df['固收类'].to_dict(),
                "混合策略类": df['混合类'].to_dict(),
                "权益投资类": df['权益类'].to_dict(),
                "另类投资类": df['另类'].to_dict()
            },
            "StandardProportion": {
                "C1": {"货币现金类": 1.0, "固定收益类": 0.0, "混合策略类": 0.0, "权益投资类": 0.0, "另类投资类": 0.0},
                "C2": {"货币现金类": 0.2, "固定收益类": 0.8, "混合策略类": 0.0, "权益投资类": 0.0, "另类投资类": 0.0},
                "C3": {"货币现金类": 0.1, "固定收益类": 0.55, "混合策略类": 0.35, "权益投资类": 0.0, "另类投资类": 0.0},
                "C4": {"货币现金类": 0.05, "固定收益类": 0.4, "混合策略类": 0.3, "权益投资类": 0.2, "另类投资类": 0.05},
                "C5": {"货币现金类": 0.05, "固定收益类": 0.2, "混合策略类": 0.25, "权益投资类": 0.4, "另类投资类": 0.1},
                "C6": {"货币现金类": 0.05, "固定收益类": 0.1, "混合策略类": 0.15, "权益投资类": 0.6, "另类投资类": 0.1}
            }
        }}
    json_str_input = json.dumps(in_dict, ensure_ascii=False)
    with open("sample_A06_input.json", "w", encoding="utf-8") as f:
        f.write(json_str_input)

    s_t = time.time()
    res_json = main(json_str_input)
    print(res_json)
    log(f"计算总耗时: {time.time() - s_t:.2f} 秒")
