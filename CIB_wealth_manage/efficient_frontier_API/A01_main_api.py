# -*- encoding: utf-8 -*-
"""
@File: A01_main_api.py
@Modify Time: 2025/9/16 11:05       
@Author: Kevin-Chen
@Descriptions: 
"""
import time
import json
import traceback
import numpy as np
import pandas as pd
from typing import Optional

from T04_show_plt import plot_efficient_frontier
from T02_other_tools import load_returns_from_excel, log
from T01_generate_random_weights import compute_var_parametric_arrays
from T03_weight_limit_cal import level_weight_limit_cal, hold_weight_limit_cal
from T01_generate_random_weights import multi_level_random_walk_config, compute_perf_arrays

''' 0) 准备工作: 配置一些预定义的参数 ----------------------------------------------------------------------------- '''
# 随机游走与指标参数（仅字典方式）
RANDOM_SEED = 12345
# 字典式多轮配置（参数含义见 multi_level_random_walk_config 注释）
ROUNDS_CONFIG = {
    # 第0轮：初始化方式二选一：
    0: {
        "init_mode": "exploration",  # "exploration" 随机探索 或 "solver" 求解器
        # exploration 参数（当 init_mode=="exploration" 生效）：
        "samples": 300,
        "step_size": 0.99,
        # solver 参数（当 init_mode=="solver" 生效）：
        "solver_params": {
            "n_grid": 1000,
            "solver": "SLSQP",  # ECOS/SCS/MOSEK/SLSQP
            "ridge": 1e-8,
        },
    },
    1: {"samples_total": 1000, "step_size": 0.1, "vol_bins": 100, "parallel_workers": 100},
    2: {"samples_total": 2000, "step_size": 0.8, "vol_bins": 200, "parallel_workers": 100},
    3: {"samples_total": 3000, "step_size": 0.05, "vol_bins": 300, "parallel_workers": 100},
    4: {"samples_total": 4000, "step_size": 0.01, "vol_bins": 400, "parallel_workers": 100},
}
TRADING_DAYS = 252.0  # 年化换算用交易天数
DEDUP_DECIMALS = 2  # 在“权重去重”时对每行权重进行四舍五入保留的小数位数
# 是否启用“极端权重”种子，以及每个种子生成的数量
EXTREME_SEED_CONFIG = {
    "enable": False,  # 是否启用“极端权重”种子: True表示启用
    "samples_per_seed": 100,  # 每个极端种子（例如 5 个大类 -> 5 个种子）生成多少权重
    "step_size": 0.3,  # 可选步长
}
# 风险度量与 VaR 参数
RISK_METRIC = "var"  # 可选："vol"（波动率）或 "var"（参数法 VaR）
VAR_PARAMS = {
    "confidence": 0.95,
    "horizon_days": 1.0,
    "return_type": "log",  # 或 "simple"
    "ddof": 1,
    "clip_non_negative": True,  # 对“无下跌”情形，VaR 取 0
}
# 权重精度（量化）选择：'0.1%'、'0.2%'、'0.5%' 或 None（不量化）
PRECISION_CHOICE: Optional[str] = None
# SLSQP 最终精炼参数
SLSQP_REFINE = {"enable": False, "n_grid": 1000}


# 帮助函数：生成悬停文本
def format_hover_text(row, title, a_list):
    text = f"<b>{title}</b><br>年化收益率: {row['ret_annual']:.2%}<br>年化风险: {row['risk_arr']:.2%}"
    text += "<br><br><b>--权重--</b><br>"
    for asset in a_list:
        if asset in row and pd.notna(row[asset]):
            text += f"{asset}: {row[asset]:.2%}<br>"
    return text.strip('<br>')


# 帮助函数：计算单个组合的指标
def calculate_single_port_metrics(weights_array, daily_returns, annual_trading_days, risk_metric, var_params):
    w_2d = weights_array.reshape(1, -1)

    # 计算年化收益率 (口径与主流程保持一致，使用对数收益)
    ret_annual_arr, vol_annual_arr = compute_perf_arrays(
        port_daily=daily_returns,
        portfolio_allocs=w_2d,
        trading_days=annual_trading_days,
        return_type="log"
    )
    annual_ret = ret_annual_arr[0]

    # 根据风险度量选择计算方式
    if (risk_metric or "vol").lower() == "var":
        vp = var_params or {}
        risk_arr = compute_var_parametric_arrays(
            port_daily=daily_returns,
            portfolio_allocs=w_2d,
            confidence=float(vp.get("confidence", 0.95)),
            horizon_days=float(vp.get("horizon_days", 1.0)),
            return_type=str(vp.get("return_type", "log")),  # 使用 VAR_PARAMS 中定义的收益类型
            ddof=int(vp.get("ddof", 1)),
            clip_non_negative=bool(vp.get("clip_non_negative", True)),
        )
        annual_risk = risk_arr[0]
    else:  # 'vol'
        annual_risk = vol_annual_arr[0]

    return annual_ret, annual_risk


def create_scatter_point_data(asset_list, W_unc, ret_annual_unc, risk_arr_unc, ef_mask_unc, level_colors,
                              random_weight_dict, W_hold, ret_annual_hold, risk_arr_hold, standard_proportion,
                              ef_mask_hold, user_holding, returns, trading_days, var_params, risk_metric):
    scatter_points_data = list()  # 重新初始化用于绘图的点云数据

    # a) 处理无约束组合
    if W_unc is not None:
        df_unc = pd.DataFrame(W_unc, columns=asset_list)
        df_unc['ret_annual'] = ret_annual_unc
        df_unc['risk_arr'] = risk_arr_unc
        df_unc['hover_text'] = df_unc.apply(
            lambda r: format_hover_text(r, "无约束组合", asset_list), axis=1)

        # 无约束 - 随机点 (抽样以避免浏览器卡顿)
        scatter_points_data.append({
            "data": df_unc,
            "name": "无约束-随机点", "color": "black", "size": 2, "opacity": 0.2,
        })
        # 无约束 - 有效前沿
        scatter_points_data.append({
            "data": df_unc[ef_mask_unc],
            "name": "无约束-有效前沿", "color": "blue", "size": 3, "opacity": 0.8,
        })
    else:
        log("没有无约束组合的权重数据，跳过无约束组合的绘图处理。")

    # b) 处理各标准组合的 frontier
    if random_weight_dict is not None and len(random_weight_dict) > 0:
        for k, v in random_weight_dict.items():
            color = level_colors.get(k, 'black')
            df_level = pd.DataFrame(v['weights'], columns=asset_list)
            df_level['ret_annual'] = v['ret_annual']
            df_level['risk_arr'] = v['risk_arr']
            df_level['hover_text'] = df_level.apply(
                lambda r: format_hover_text(r, f"{k} 组合", asset_list), axis=1)
            level_ef_mask = v['ef_mask']
            scatter_points_data.append({
                "data": df_level,
                "name": f"{k}-随机点", "color": color, "size": 3, "opacity": 0.5,
            })
            scatter_points_data.append({
                "data": df_level[level_ef_mask],
                "name": f"{k}-有效前沿", "color": color, "size": 3, "opacity": 0.9,
            })
    else:
        log("没有标准组合的权重数据，跳过标准组合的绘图处理。")

    # c) 处理客户持仓组合的 frontier
    if W_hold is not None:
        df_hold = pd.DataFrame(W_hold, columns=asset_list)
        df_hold['ret_annual'] = ret_annual_hold
        df_hold['risk_arr'] = risk_arr_hold
        df_hold['hover_text'] = df_hold.apply(
            lambda r: format_hover_text(r, "客户持仓约束", asset_list), axis=1)
        scatter_points_data.append({
            "data": df_hold,
            "name": "客户持仓-随机点", "color": "purple", "size": 2, "opacity": 0.2,
        })
        scatter_points_data.append({
            "data": df_hold[ef_mask_hold],
            "name": "客户持仓-有效前沿", "color": "red", "size": 3, "opacity": 0.8,
        })
    else:
        log("没有客户持仓组合的权重数据，跳过客户持仓组合的绘图处理。")

    # d) 处理标准组合点
    if standard_proportion is not None:
        standard_points = []
        for name, w_dict in standard_proportion.items():
            weights = np.array([w_dict.get(asset, 0.0) for asset in asset_list])
            ret, risk = calculate_single_port_metrics(weights, returns,
                                                      trading_days, risk_metric, var_params)
            point_data = {'name': name, 'ret_annual': ret, 'risk_arr': risk}
            for asset, w in w_dict.items():
                point_data[asset] = w
            standard_points.append(point_data)
        df_standard_points = pd.DataFrame(standard_points)
        df_standard_points['hover_text'] = df_standard_points.apply(
            lambda r: format_hover_text(r, f"标准-{r['name']}", asset_list), axis=1)
        scatter_points_data.append({
            "data": df_standard_points,
            "name": "标准组合点", "color": "gold", "size": 12, "opacity": 1.0, "symbol": "star",
            "marker_line": dict(width=1, color='black')
        })
    else:
        log("没有标准组合的权重数据，跳过标准组合点的绘图处理。")

    # e) 处理客户当前持仓点
    if user_holding is not None:
        user_weights_dict = user_holding['StandardProportion']
        user_weights_arr = np.array([user_weights_dict.get(asset, 0.0) for asset in asset_list])
        user_ret, user_risk = calculate_single_port_metrics(user_weights_arr, returns,
                                                            trading_days, risk_metric,
                                                            var_params)
        df_user_point_data = {'ret_annual': user_ret, 'risk_arr': user_risk, **user_weights_dict}
        df_user_point = pd.DataFrame([df_user_point_data])
        df_user_point['hover_text'] = df_user_point.apply(
            lambda r: format_hover_text(r, "客户当前持仓", asset_list), axis=1)
        scatter_points_data.append({
            "data": df_user_point,
            "name": "客户当前持仓", "color": "cyan", "size": 12, "opacity": 1.0, "symbol": "diamond",
            "marker_line": dict(width=1, color='black')
        })
    else:
        log("没有客户当前持仓的权重数据，跳过客户当前持仓点的绘图处理。")

    return scatter_points_data


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
    ser_list = []
    for asset in asset_list:
        if asset not in nv_dict:
            raise ValueError("缺少大类 '%s' 的净值数据(nv)" % asset)
        v = nv_dict[asset]
        if not isinstance(v, dict):
            raise ValueError("nv['%s'] 的格式应为 {date->nav} 字典" % asset)
        ser_list.append(_parse_series_from_dict(v).rename(asset))
    df_nv = pd.concat(ser_list, axis=1, join='inner')
    df_ret = df_nv.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how='any')
    return df_ret.values.astype(np.float32, copy=False)


# 解析Json参数 & 读取大类收益率
def analysis_json_and_read_data(json_input, excel_name, sheet_name):
    # Json转字典
    json_dict = json.loads(json_input)
    # 分解参数
    asset_list = json_dict['asset_list']  # 大类列表
    cal_market_ef = json_dict.get('cal_market_ef', None)  # 是否计算无约束的市场组合
    draw_plt = json_dict.get('draw_plt', None)  # 是否绘图展示
    draw_plt_filename = json_dict.get('draw_plt_filename', None)  # 绘图保存文件名, None表示不保存直接显示
    weight_range = json_dict.get('WeightRange', None)  # 标准组合约束
    standard_proportion = json_dict.get('StandardProportion', None)  # 标准组合
    user_holding = json_dict.get('user_holding', None)  # 客户持仓组合
    # 优先使用 JSON 内嵌的净值数据 nv；否则读 Excel；都没有则报错
    nv_data = json_dict.get('nv', None)
    if nv_data is not None:
        log("使用 JSON 内嵌的净值数据(nv)计算收益率")
        returns = _load_returns_from_nv(nv_data, asset_list)
    else:
        if excel_name and sheet_name:
            log("使用 Excel 读取的净值数据计算收益率")
            returns, _ = load_returns_from_excel(excel_name, sheet_name, asset_list)
        else:
            log("未提供净值数据(nv)，且缺少 Excel 读取参数(excel_name/sheet_name)")
            raise ValueError("未提供净值数据(nv)，且缺少 Excel 读取参数(excel_name/sheet_name)")
    return (asset_list, cal_market_ef, draw_plt, draw_plt_filename,
            weight_range, standard_proportion, user_holding, returns)


def main(json_input, excel_name, sheet_name):
    """
    主函数，用于执行完整的投资组合风险分析流程。包括解析JSON配置、读取Excel数据、计算权重约束、
    寻找最小和最大风险组合，并可选地绘制风险收益图。

    :param json_input: str, 包含资产配置和分析参数的JSON字符串
    :param excel_name: str, Excel文件路径，包含历史收益率等数据
    :param sheet_name: str, Excel工作表名称，用于读取数据
    :return: str, 包含计算结果或错误信息的JSON字符串
    """
    try:
        # 存储结果
        res_dict = dict()

        ''' 1) 解析Json参数 & 读取大类收益率 ----------------------------------------------------------------------------- '''
        (asset_list, cal_market_ef, draw_plt, draw_plt_filename, weight_range, standard_proportion,
         user_holding, returns) = analysis_json_and_read_data(json_input, excel_name, sheet_name)

        ''' 2) 计算约束 ----------------------------------------------------------------------------------------------- '''
        level_weight_limit = None
        if weight_range and standard_proportion:
            level_weight_limit = dict()
            for k, v in weight_range.items():
                single_limit, multi_limit = level_weight_limit_cal(asset_list, v)
                level_weight_limit[k] = {'single_limit': single_limit, 'multi_limit': multi_limit}
            log("标准组合的约束：", level_weight_limit)
        else:
            log("无标准组合的约束输入，跳过约束计算。")

        single_limit_hold, multi_limit_hold = (None, None)
        if user_holding:
            single_limit_hold, multi_limit_hold = hold_weight_limit_cal(asset_list, user_holding)
            hold_weight_limit = {'single_limit': single_limit_hold, 'multi_limit': multi_limit_hold}
            log("客户持仓的约束：", hold_weight_limit)
        else:
            log("无客户持仓的约束输入，跳过约束计算。")

        ''' 3) 计算无约束的市场组合的随机权重和有效前沿 --------------------------------------------------------------------- '''
        W_unc, ret_annual_unc, risk_arr_unc, ef_mask_unc = (None, None, None, None)
        if cal_market_ef:
            single_limit = [(0.0, 1.0)] * len(asset_list)
            log(f"计算无约束的市场组合随机权重. 单资产约束: {single_limit}")
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
            )
            log(f"无约束市场组合的随机权重计算完成. 权重数: {W_unc.shape[0]}")
            res_dict['market'] = {
                'weights': W_unc.tolist(),
                'ret_annual': ret_annual_unc.tolist(),
                'risk_arr': risk_arr_unc.tolist(),
                'ef_mask': ef_mask_unc.tolist(),
            }
        else:
            log("无需计算无约束市场组合的有效前沿和配置空间")

        ''' 4) 计算标准组合的随机权重和有效前沿 ------------------------------------------------------------------------ '''
        random_weight_dict = dict()
        if level_weight_limit:
            for k, v in level_weight_limit.items():
                single_limit = v['single_limit']
                multi_limit = v['multi_limit']
                log(f"计算标准组合 {k} 的随机权重. 单资产约束: {single_limit}; 多资产约束: {multi_limit}")
                (W, ret_annual, risk_arr, ef_mask) = multi_level_random_walk_config(
                    port_daily_returns=returns,
                    single_limits=single_limit,
                    multi_limits=multi_limit,
                    rounds_config=ROUNDS_CONFIG,
                    dedup_decimals=DEDUP_DECIMALS,
                    annual_trading_days=TRADING_DAYS,
                    global_seed=RANDOM_SEED,
                    extreme_seed_config=EXTREME_SEED_CONFIG,
                    risk_metric=RISK_METRIC,
                    var_params=VAR_PARAMS,
                    precision_choice=PRECISION_CHOICE,
                )
                random_weight_dict[k] = {
                    'weights': W.tolist(),
                    'ret_annual': ret_annual.tolist(),
                    'risk_arr': risk_arr.tolist(),
                    'ef_mask': ef_mask.tolist()
                }
                log(f"标准组合 {k} 的随机权重计算完成. 权重数: {W.shape[0]}")
            res_dict['standard'] = random_weight_dict
        else:
            log("无标准组合的相关输入，跳过计算有效前沿和可配置空间。")

        ''' 5) 计算客户持仓的随机权重和有效前沿 ------------------------------------------------------------------------ '''
        W_hold, ret_annual_hold, risk_arr_hold, ef_mask_hold = (None, None, None, None)
        if user_holding and single_limit_hold:
            log(f"计算客户持仓组合的随机权重. 单资产约束: {single_limit_hold}; 多资产约束: {multi_limit_hold}")
            (W_hold, ret_annual_hold, risk_arr_hold, ef_mask_hold) = multi_level_random_walk_config(
                port_daily_returns=returns,
                single_limits=single_limit_hold,
                multi_limits=multi_limit_hold,
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
            log(f"客户持仓组合的随机权重计算完成. 权重数: {W_hold.shape[0]}")
            res_dict['user'] = {
                'weights': W_hold.tolist(),
                'ret_annual': ret_annual_hold.tolist(),
                'risk_arr': risk_arr_hold.tolist(),
                'ef_mask': ef_mask_hold.tolist(),
            }
        else:
            log("无客户持仓的相关输入，跳过计算有效前沿和可配置空间。")

        ''' 6) 绘图展示 -------------------------------------------------------------------------------------------- '''
        if draw_plt:
            log("\n开始准备绘图数据...")
            level_colors = {'C1': '#1f77b4', 'C2': '#ff7f0e', 'C3': 'grey', 'C4': 'green', 'C5': 'blue', 'C6': 'red'}
            scatter_points_data = create_scatter_point_data(asset_list, W_unc, ret_annual_unc,
                                                            risk_arr_unc, ef_mask_unc,
                                                            level_colors, random_weight_dict,
                                                            W_hold, ret_annual_hold,
                                                            risk_arr_hold, standard_proportion,
                                                            ef_mask_hold, user_holding,
                                                            returns, TRADING_DAYS,
                                                            VAR_PARAMS, RISK_METRIC)
            if len(scatter_points_data) > 0:
                plot_efficient_frontier(
                    scatter_points_data,
                    title="各约束下的有效前沿对比",
                    x_axis_title=f"年化风险 ({RISK_METRIC.upper()})",
                    y_axis_title="年化收益率",
                    x_col="risk_arr",
                    y_col="ret_annual",
                    hover_text_col="hover_text",
                    output_filename=draw_plt_filename
                )
            else:
                log("无有效前沿数据，无法绘图。")
        else:
            log("无需绘图展示。")

        # 封装成功响应
        success_response = {
            "success": True,
            "data": res_dict
        }
        return json.dumps(success_response, ensure_ascii=False)

    except json.JSONDecodeError as e:
        log(traceback.format_exc())
        error_response = {
            "success": False,
            "error_code": "INVALID_JSON_INPUT",
            "message": f"请求的JSON格式错误: {e}"
        }
        return json.dumps(error_response, ensure_ascii=False)

    except (KeyError, TypeError) as e:
        log(traceback.format_exc())
        error_response = {
            "success": False,
            "error_code": "MISSING_OR_INVALID_FIELD",
            "message": f"输入数据中缺少必需的字段或字段类型错误: {e}"
        }
        return json.dumps(error_response, ensure_ascii=False)

    except FileNotFoundError as e:
        log(traceback.format_exc())
        error_response = {
            "success": False,
            "error_code": "DATA_FILE_NOT_FOUND",
            "message": f"服务器端数据文件未找到: {e.filename}"
        }
        return json.dumps(error_response, ensure_ascii=False)

    except ValueError as e:
        log(traceback.format_exc())
        error_response = {
            "success": False,
            "error_code": "INVALID_DATA_OR_CONFIG",
            "message": f"数据或配置无效: {e}"
        }
        return json.dumps(error_response, ensure_ascii=False)

    except Exception as e:
        log(traceback.format_exc())
        error_response = {
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": f"计算过程中发生未知错误: {type(e).__name__} - {e}"
        }
        return json.dumps(error_response, ensure_ascii=False)


if __name__ == '__main__':
    ''' 准备工作: 模拟json参数输入 ------------------------------------------------------------------------------- '''
    with open('sample_A01_input_market.json', 'r', encoding='utf-8') as f:
        json_str = f.read()
    # excel信息
    excel_path = None
    sheet = '历史净值数据'

    ''' 开始计算, 调用主程序 ------------------------------------------------------------------------------------- '''
    s_t = time.time()
    json_res = main(json_str, excel_path, sheet)
    # log("最终返回的结果 Json 字符串为：\n", json_res)
    log(f"计算总耗时: {time.time() - s_t:.2f} 秒")
