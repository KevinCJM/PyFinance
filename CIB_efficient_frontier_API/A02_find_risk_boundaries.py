# -*- encoding: utf-8 -*-
"""
@File: A02_find_risk_boundaries.py
@Modify Time: 2025/9/16 19:02       
@Author: Kevin-Chen
@Descriptions: 寻找在指定大类约束下的风险边界
"""
import json
import pandas as pd
import numpy as np

from T04_show_plt import plot_efficient_frontier
from T02_other_tools import load_returns_from_excel
from T01_generate_random_weights import compute_perf_arrays, compute_var_parametric_arrays
from T03_weight_limit_cal import level_weight_limit_cal


# 解析Json参数 & 读取大类收益率
def analysis_json_and_read_data(json_input, excel_name, sheet_name):
    # Json转字典
    json_dict = json.loads(json_input)
    # 分解参数
    asset_list = json_dict['asset_list']  # 大类列表
    draw_plt = json_dict.get('draw_plt', None)  # 是否绘图展示
    draw_plt_filename = json_dict.get('draw_plt_filename', None)  # 绘图保存文件名, None表示不保存直接显示
    weight_range = json_dict.get('WeightRange', None)  # 标准组合约束
    # 读取excel，生成日收益二维数组
    returns, assets = load_returns_from_excel(excel_name, sheet_name, asset_list)
    return asset_list, draw_plt, draw_plt_filename, weight_range, returns


if __name__ == '__main__':
    ''' 准备工作: 模拟json参数输入 -------------------------------------------------------------------------------
    '''
    # 字典格式入参
    dict_input = {
        'asset_list': [
            '货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类'
        ],
        'WeightRange': {  # 大类权重上下限
            '货币现金类': [0.05 * 0.8, 0.05 * 1.2], '固定收益类': [0.10 * 0.8, 0.10 * 1.2],
            '混合策略类': [0.15 * 0.8, 0.15 * 1.2], '权益投资类': [0.60 * 0.8, 0.60 * 1.2],
            '另类投资类': [0.10 * 0.8, 0.10 * 1.2]
        },
        'draw_plt': True,  # 是否绘图展示
        'draw_plt_filename': None,  # 绘图保存文件名，None表示不保存直接显示
    }
    # excel信息
    excel_path = '历史净值数据_万得指数.xlsx'
    excel_sheet = '历史净值数据'
    # 字典转Json, 模拟输入的Json参数
    json_str = json.dumps(dict_input, ensure_ascii=False)
    print(json_str)

    ''' 0. 配置计算参数 ---------------------------------------------------------------------------------------- '''
    RANDOM_SEED = 12345
    NUM_RANDOM_SAMPLES = 5000  # 用于热启动的随机样本数
    RANDOM_WALK_STEP_SIZE = 0.05  # 随机游走步长
    RISK_METRIC = "var"  # 计算风险边界所用的指标, 可选 "vol" 或 "var"
    TRADING_DAYS = 252.0
    VAR_PARAMS = {
        "confidence": 0.95, "horizon_days": 1.0, "return_type": "log",
        "ddof": 1, "clip_non_negative": True,
    }

    ''' 1. 处理Json & 读取数据 ----------------------------------------------------------------------------------- '''
    (asset_list, draw_plt, draw_plt_filename, weight_range,
     returns) = analysis_json_and_read_data(json_str, excel_path, excel_sheet)

    ''' 2) 计算约束 ---------------------------------------------------------------------------------------------- '''
    single_limit, multi_limit = level_weight_limit_cal(asset_list, weight_range)
    print(f"单层约束: {single_limit}; 多层约束: {multi_limit}")

    ''' 3) 计算两端风险水平 --------------------------------------------------------------------------------------- '''
    from T01_generate_random_weights import generate_weights_random_walk
    from scipy.optimize import minimize

    # 步骤 1: 生成随机权重用于热启动
    print(f"\n步骤 1: 生成 {NUM_RANDOM_SAMPLES} 个随机权重用于热启动...")
    W_random = generate_weights_random_walk(
        N=len(asset_list),
        single_limits=single_limit,
        multi_limits=multi_limit,
        seed=RANDOM_SEED,
        num_samples=NUM_RANDOM_SAMPLES,
        step_size=RANDOM_WALK_STEP_SIZE,
    )
    print(f"生成了 {W_random.shape[0]} 个有效的随机权重。")

    # 步骤 2: 从随机权重中寻找风险最大/最小的组合作为热启动点
    print("\n步骤 2: 从随机权重中寻找风险最大/最小的组合作为热启动点...")
    ret_annual_random, risk_arr_random_vol = compute_perf_arrays(
        port_daily=returns,
        portfolio_allocs=W_random,
        trading_days=TRADING_DAYS,
        return_type="log"
    )
    if RISK_METRIC == 'vol':
        risk_arr_random = risk_arr_random_vol
    else:  # 'var'
        risk_arr_random = compute_var_parametric_arrays(
            port_daily=returns,
            portfolio_allocs=W_random,
            **VAR_PARAMS
        )

    w0_min_risk = W_random[np.argmin(risk_arr_random)]
    w0_max_risk = W_random[np.argmax(risk_arr_random)]
    print(f"最小风险热启动点 (风险={np.min(risk_arr_random):.4f})")
    print(f"最大风险热启动点 (风险={np.max(risk_arr_random):.4f})")

    # 步骤 3: 使用SLSQP优化器精确寻找风险边界
    print("\n步骤 3: 使用SLSQP优化器精确寻找风险边界...")


    def _risk_func_optimizer(w: np.ndarray) -> float:
        w = np.asarray(w, dtype=np.float64)
        if RISK_METRIC == 'vol':
            port_log_ret = np.log1p(returns @ w)
            return float(port_log_ret.std(ddof=1)) * np.sqrt(TRADING_DAYS)
        else:
            from statistics import NormalDist
            vp = VAR_PARAMS or {}
            confidence = float(vp.get("confidence", 0.95))
            horizon_days = float(vp.get("horizon_days", 1.0))
            return_type = str(vp.get("return_type", "log"))
            z_score = NormalDist().inv_cdf(1.0 - confidence)
            if return_type == "log":
                X = np.log1p(returns @ w)
            else:
                X = returns @ w
            mu = X.mean()
            sigma = X.std(ddof=1)
            h = float(horizon_days)
            mu_h = mu * h
            sigma_h = sigma * np.sqrt(h)
            var = -(mu_h + z_score * sigma_h)
            return max(0.0, var) if vp.get("clip_non_negative", True) else var


    bounds = single_limit
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    for idx_tuple, (lo, hi) in multi_limit.items():
        idx = list(idx_tuple)
        if idx:
            cons.append({'type': 'ineq', 'fun': lambda w, i=idx, l=lo: np.sum(w[i]) - l})
            cons.append({'type': 'ineq', 'fun': lambda w, i=idx, h=hi: h - np.sum(w[i])})

    res_min = minimize(_risk_func_optimizer, w0_min_risk, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'ftol': 1e-12, 'maxiter': 1000})
    res_max = minimize(lambda w: -_risk_func_optimizer(w), w0_max_risk, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'ftol': 1e-12, 'maxiter': 1000})

    if res_min.success:
        min_risk_val = res_min.fun
        min_risk_weights = res_min.x
        print("\n--- 最小风险组合 (已找到) ---")
        print(f"风险值 ({RISK_METRIC}): {min_risk_val:.6f}")
        for asset, weight in zip(asset_list, min_risk_weights):
            print(f"  {asset}: {weight:.4%}")
    else:
        print("\n--- 最小风险组合优化失败 ---")
        print(res_min.message)

    if res_max.success:
        max_risk_val = -res_max.fun
        max_risk_weights = res_max.x
        print("\n--- 最大风险组合 (已找到) ---")
        print(f"风险值 ({RISK_METRIC}): {max_risk_val:.6f}")
        for asset, weight in zip(asset_list, max_risk_weights):
            print(f"  {asset}: {weight:.4%}")
    else:
        print("\n--- 最大风险组合优化失败 ---")
        print(res_max.message)

    ''' 4) 绘图展示 ----------------------------------------------------------------------------------------------- '''
    if draw_plt:
        print("\n步骤 4: 准备绘图数据并生成图表...")

        df_random = pd.DataFrame(W_random, columns=asset_list)
        df_random['ret_annual'] = ret_annual_random
        df_random['risk_arr'] = risk_arr_random

        extreme_points_data = []
        if res_min.success:
            ret, risk = compute_perf_arrays(returns, res_min.x.reshape(1, -1), TRADING_DAYS)
            point_data = {'name': '最小风险点', 'ret_annual': ret[0], 'risk_arr': res_min.fun}
            for i, asset in enumerate(asset_list):
                point_data[asset] = res_min.x[i]
            extreme_points_data.append(point_data)

        if res_max.success:
            ret, risk = compute_perf_arrays(returns, res_max.x.reshape(1, -1), TRADING_DAYS)
            point_data = {'name': '最大风险点', 'ret_annual': ret[0], 'risk_arr': -res_max.fun}
            for i, asset in enumerate(asset_list):
                point_data[asset] = res_max.x[i]
            extreme_points_data.append(point_data)

        df_extreme = pd.DataFrame(extreme_points_data)


        def format_hover_text_local(row, title):
            text = f"<b>{title}</b><br>年化收益率: {row['ret_annual']:.2%}<br>年化风险: {row['risk_arr']:.2%}"
            text += "<br><br><b>--权重--</b><br>"
            for asset in asset_list:
                if asset in row and pd.notna(row[asset]):
                    text += f"{asset}: {row[asset]:.2%}<br>"
            return text.strip('<br>')


        df_random['hover_text'] = df_random.apply(lambda r: format_hover_text_local(r, "随机组合"), axis=1)
        if not df_extreme.empty:
            df_extreme['hover_text'] = df_extreme.apply(lambda r: format_hover_text_local(r, r['name']), axis=1)

        scatter_points_data = []
        scatter_points_data.append({
            "data": df_random,
            "name": "可配置空间 (随机权重)",
            "color": "lightblue",
            "size": 5,
            "opacity": 0.5,
        })
        if not df_extreme.empty:
            scatter_points_data.append({
                "data": df_extreme,
                "name": "风险边界组合",
                "color": "red",
                "size": 12,
                "opacity": 1.0,
                "symbol": "star",
                "marker_line": dict(width=1, color='black')
            })

        plot_efficient_frontier(
            scatter_points_data,
            title="风险边界探索",
            x_axis_title=f"年化风险 ({RISK_METRIC.upper()})",
            y_axis_title="年化收益率",
            x_col="risk_arr",
            y_col="ret_annual",
            hover_text_col="hover_text",
            output_filename=draw_plt_filename
        )
        print(f"绘图完成。")
