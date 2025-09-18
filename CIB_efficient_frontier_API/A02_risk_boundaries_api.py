# -*- encoding: utf-8 -*-
"""
@File: A02_risk_boundaries_api.py
@Modify Time: 2025/9/16 19:02       
@Author: Kevin-Chen
@Descriptions: 寻找在指定大类约束下的风险边界
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from T04_show_plt import plot_efficient_frontier
from T02_other_tools import load_returns_from_excel
from T01_generate_random_weights import compute_perf_arrays, compute_var_parametric_arrays
from T03_weight_limit_cal import level_weight_limit_cal
from T01_generate_random_weights import generate_weights_random_walk

''' 预设计算参数 ---------------------------------------------------------------------------------------- '''
RANDOM_SEED = 12345
NUM_RANDOM_SAMPLES = 1000  # 用于热启动的随机样本数
RANDOM_WALK_STEP_SIZE = 0.1  # 随机游走步长
RISK_METRIC = "var"  # 计算风险边界所用的指标, 可选 "vol" 或 "var"
TRADING_DAYS = 252.0
VAR_PARAMS = {
    "confidence": 0.95, "horizon_days": 1.0, "return_type": "log",
    "ddof": 1, "clip_non_negative": True,
}


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


# 计算随机投资组合中的最小风险和最大风险组合
def find_min_max_risk(w_random, returns, risk_metric, trading_days, var_params):
    """
    计算随机投资组合中的最小风险和最大风险组合

    参数:
        w_random: 随机生成的投资组合权重矩阵
        returns: 投资组合的日收益率数据
        risk_metric: 风险度量指标，可选 'vol'(波动率) 或 'var'(风险价值)
        trading_days: 年交易日数量，用于年化计算
        var_params: VaR计算所需的参数字典

    返回值:
        ret_annual_random: 年化收益率数组
        risk_arr_random: 风险指标数组
        w0_min_risk: 最小风险对应的投资组合权重
        w0_max_risk: 最大风险对应的投资组合权重
    """
    # 计算投资组合的性能指标数组
    ret_annual_random, risk_arr_random_vol = compute_perf_arrays(
        port_daily=returns,
        portfolio_allocs=w_random,
        trading_days=trading_days,
        return_type="log"
    )

    # 根据风险度量指标选择相应的风险计算方法
    if risk_metric == 'vol':
        risk_arr_random = risk_arr_random_vol
    else:  # 'var'
        risk_arr_random = compute_var_parametric_arrays(
            port_daily=returns,
            portfolio_allocs=w_random,
            **var_params
        )

    # 找到最小风险和最大风险对应的投资组合权重
    w0_min_risk = w_random[np.argmin(risk_arr_random)]
    w0_max_risk = w_random[np.argmax(risk_arr_random)]
    return ret_annual_random, risk_arr_random, w0_min_risk, w0_max_risk


# 风险函数优化器，用于计算投资组合的风险值
def _risk_func_optimizer(w, returns, risk_metric, var_params, trading_days) -> float:
    """
    风险函数优化器，用于计算投资组合的风险值

    参数:
        w (np.ndarray): 投资组合权重向量
        returns (np.ndarray): 各资产的历史收益率矩阵
        risk_metric (str): 风险度量指标，'vol' 表示波动率，其他值表示VaR
        var_params (dict or None): VaR计算参数字典，包含置信水平、时间跨度等参数
        trading_days (int): 年化交易日数量，用于年化波动率计算

    返回:
        float: 计算得到的风险值，可能是年化波动率或VaR值
    """
    w = np.asarray(w, dtype=np.float64)

    # 如果风险度量指标是波动率，则计算年化波动率
    if risk_metric == 'vol':
        port_log_ret = np.log1p(returns @ w)
        return float(port_log_ret.std(ddof=1)) * np.sqrt(trading_days)
    else:
        # 计算风险价值(VaR)指标
        from statistics import NormalDist
        vp = var_params or {}
        confidence = float(vp.get("confidence", 0.95))
        horizon_days = float(vp.get("horizon_days", 1.0))
        return_type = str(vp.get("return_type", "log"))
        z_score = NormalDist().inv_cdf(1.0 - confidence)

        # 根据收益率类型选择计算方式
        if return_type == "log":
            x = np.log1p(returns @ w)
        else:
            x = returns @ w

        # 计算收益率的均值和标准差
        mu = x.mean()
        sigma = x.std(ddof=1)
        h = float(horizon_days)

        # 计算多期的均值和标准差
        mu_h = mu * h
        sigma_h = sigma * np.sqrt(h)

        # 计算VaR值
        var = -(mu_h + z_score * sigma_h)
        return max(0.0, var) if vp.get("clip_non_negative", True) else var


# 使用优化器计算最小风险和最大风险的投资组合权重
def use_optimizer(bounds, multi_limit, returns, risk_metric, var_params, trading_days, w0_min_risk, w0_max_risk,
                  asset_list):
    """
    使用优化器计算最小风险和最大风险的投资组合权重。

    参数:
        bounds (list of tuple): 每个资产的权重边界，格式为 [(low, high), ...]。
        multi_limit (dict): 多资产组合的限制条件，键为资产索引元组，值为 (下限, 上限)。
        returns (np.ndarray): 资产的历史收益率矩阵。
        risk_metric (str): 风险度量指标名称，例如 'var', 'cvar' 等。
        var_params (dict): 风险计算所需的参数字典。
        trading_days (int): 一年中的交易日数量，用于年化风险。
        w0_min_risk (np.ndarray): 最小风险优化的初始权重。
        w0_max_risk (np.ndarray): 最大风险优化的初始权重。
        asset_list (list): 资产名称列表。

    返回:
        无返回值。直接打印最小风险和最大风险组合的优化结果。
    """

    # 定义约束条件：权重总和为1
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    # 添加多资产组合的上下限约束
    for idx_tuple, (lo, hi) in multi_limit.items():
        idx = list(idx_tuple)
        if idx:
            cons.append({'type': 'ineq', 'fun': lambda w, i=idx, l=lo: np.sum(w[i]) - l})
            cons.append({'type': 'ineq', 'fun': lambda w, i=idx, h=hi: h - np.sum(w[i])})

    # 求解最小风险组合
    res_min = minimize(
        lambda w: _risk_func_optimizer(w, returns, risk_metric, var_params, trading_days),
        w0_min_risk, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-12, 'maxiter': 1000})

    # 求解最大风险组合（通过最小化负风险实现）
    res_max = minimize(
        lambda w: -_risk_func_optimizer(w, returns, risk_metric, var_params, trading_days),
        w0_max_risk, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-12, 'maxiter': 1000})

    # 输出最小风险组合结果
    if res_min.success:
        min_risk_val = res_min.fun
        min_risk_weights = res_min.x
        print("\n--- 最小风险组合 (已找到) ---")
        print(f"风险值 ({risk_metric}): {min_risk_val:.6f}")
        for asset, weight in zip(asset_list, min_risk_weights):
            print(f"  {asset}: {weight:.4%}")
    else:
        print("\n--- 最小风险组合优化失败 ---")
        print(res_min.message)

    # 输出最大风险组合结果
    if res_max.success:
        max_risk_val = -res_max.fun
        max_risk_weights = res_max.x
        print("\n--- 最大风险组合 (已找到) ---")
        print(f"风险值 ({risk_metric}): {max_risk_val:.6f}")
        for asset, weight in zip(asset_list, max_risk_weights):
            print(f"  {asset}: {weight:.4%}")
    else:
        print("\n--- 最大风险组合优化失败 ---")
        print(res_max.message)
    return res_min, res_max


# 绘制投资组合的有效前沿图，包括随机生成的投资组合和极值点
def draw_plt_func(res_min, res_max, returns, w_random, ret_annual_random, risk_arr_random, asset_list,
                  trading_days, risk_metric, draw_plt_filename):
    """
    绘制投资组合的有效前沿图，包括随机生成的投资组合和极值点（最小/最大风险点）。

    参数:
        res_min (OptimizeResult): 最小化优化结果对象，包含最小风险点的权重等信息。
        res_max (OptimizeResult): 最大化优化结果对象，包含最大风险点的权重等信息。
        returns (np.ndarray): 资产的历史收益率矩阵。
        w_random (np.ndarray): 随机生成的投资组合权重矩阵。
        ret_annual_random (np.ndarray): 每个随机投资组合的年化收益率。
        risk_arr_random (np.ndarray): 每个随机投资组合的年化风险。
        asset_list (list): 资产名称列表。
        trading_days (int): 每年的交易日数量，用于年化计算。
        risk_metric (str): 风险度量指标名称，例如 'volatility'。
        draw_plt_filename (str): 输出图表保存的文件路径。

    返回:
        无返回值。生成并保存一张有效前沿图。
    """
    print("\n步骤 4: 准备绘图数据并生成图表...")

    # 构建随机投资组合的数据框
    df_random = pd.DataFrame(w_random, columns=asset_list)
    df_random['ret_annual'] = ret_annual_random
    df_random['risk_arr'] = risk_arr_random

    # 构建极值点（最小/最大风险点）数据
    extreme_points_data = []
    if res_min.success:
        ret, risk = compute_perf_arrays(returns, res_min.x.reshape(1, -1), trading_days)
        point_data = {'name': '最小风险点', 'ret_annual': ret[0], 'risk_arr': res_min.fun}
        for i, asset in enumerate(asset_list):
            point_data[asset] = res_min.x[i]
        extreme_points_data.append(point_data)

    if res_max.success:
        ret, risk = compute_perf_arrays(returns, res_max.x.reshape(1, -1), trading_days)
        point_data = {'name': '最大风险点', 'ret_annual': ret[0], 'risk_arr': -res_max.fun}
        for i, asset in enumerate(asset_list):
            point_data[asset] = res_max.x[i]
        extreme_points_data.append(point_data)

    df_extreme = pd.DataFrame(extreme_points_data)

    # 定义局部函数：格式化鼠标悬停时显示的文本
    def format_hover_text_local(row, title):
        text = f"<b>{title}</b><br>年化收益率: {row['ret_annual']:.2%}<br>年化风险: {row['risk_arr']:.2%}"
        text += "<br><br><b>--权重--</b><br>"
        for asset in asset_list:
            if asset in row and pd.notna(row[asset]):
                text += f"{asset}: {row[asset]:.2%}<br>"
        return text.strip('<br>')

    # 为数据框添加悬停文本列
    df_random['hover_text'] = df_random.apply(lambda r: format_hover_text_local(r, "随机组合"), axis=1)
    if not df_extreme.empty:
        df_extreme['hover_text'] = df_extreme.apply(lambda r: format_hover_text_local(r, r['name']), axis=1)

    # 准备绘图所需的数据结构
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

    # 调用绘图函数生成图表
    plot_efficient_frontier(
        scatter_points_data,
        title="风险边界探索",
        x_axis_title=f"年化风险 ({risk_metric.upper()})",
        y_axis_title="年化收益率",
        x_col="risk_arr",
        y_col="ret_annual",
        hover_text_col="hover_text",
        output_filename=draw_plt_filename
    )
    print(f"绘图完成。")


# 主函数
def main(json_str, excel_path, excel_sheet):
    """
    主函数，用于执行完整的投资组合风险分析流程。包括解析JSON配置、读取Excel数据、计算权重约束、
    寻找最小和最大风险组合，并可选地绘制风险收益图。

    :param json_str: str, 包含资产配置和分析参数的JSON字符串
    :param excel_path: str, Excel文件路径，包含历史收益率等数据
    :param excel_sheet: str, Excel工作表名称，用于读取数据
    :return: str, 包含最小风险和最大风险优化结果的JSON字符串
    """

    ''' 1. 处理Json & 读取数据 ----------------------------------------------------------------------------------- '''
    (asset_list, draw_plt, draw_plt_filename, weight_range,
     returns) = analysis_json_and_read_data(json_str, excel_path, excel_sheet)

    ''' 2) 计算约束 ---------------------------------------------------------------------------------------------- '''
    single_limit, multi_limit = level_weight_limit_cal(asset_list, weight_range)
    print(f"单层约束: {single_limit}; 多层约束: {multi_limit}")

    ''' 3) 计算两端风险水平 --------------------------------------------------------------------------------------- '''
    print(f"\n步骤 1: 生成 {NUM_RANDOM_SAMPLES} 个随机权重用于热启动...")
    w_random = generate_weights_random_walk(
        N=len(asset_list),
        single_limits=single_limit,
        multi_limits=multi_limit,
        seed=RANDOM_SEED,
        num_samples=NUM_RANDOM_SAMPLES,
        step_size=RANDOM_WALK_STEP_SIZE,
    )
    print(f"生成了 {w_random.shape[0]} 个有效的随机权重。")

    print("\n步骤 2: 从随机权重中寻找风险最大/最小的组合作为热启动点...")
    ret_annual_random, risk_arr_random, w0_min_risk, w0_max_risk = find_min_max_risk(
        w_random, returns, RISK_METRIC, TRADING_DAYS, VAR_PARAMS)
    print(
        f"最小风险热启动点 (风险={np.min(risk_arr_random):.4f}); 最大风险热启动点 (风险={np.max(risk_arr_random):.4f})")

    print("\n步骤 3: 使用SLSQP优化器精确寻找风险边界...")
    res_min, res_max = use_optimizer(single_limit, multi_limit, returns,
                                     RISK_METRIC, VAR_PARAMS, TRADING_DAYS,
                                     w0_min_risk, w0_max_risk, asset_list)

    ''' 4) 绘图展示 ----------------------------------------------------------------------------------------------- '''
    if draw_plt:
        draw_plt_func(res_min, res_max, returns, w_random, ret_annual_random,
                      risk_arr_random, asset_list, TRADING_DAYS, RISK_METRIC,
                      draw_plt_filename)

    # 构建可序列化的返回结果字典
    result_to_serialize = {}
    if res_min:
        result_to_serialize['min_risk'] = {'risk_value': res_min.fun, 'weights': res_min.x.tolist()}
    else:
        result_to_serialize['min_risk'] = {
            'success': False,
            'message': "Optimization not run or failed."
        }

    if res_max:
        result_to_serialize['max_risk'] = {'risk_value': -res_max.fun, 'weights': res_min.x.tolist()}
    else:
        result_to_serialize['max_risk'] = {
            'success': False,
            'message': "Optimization not run or failed."
        }

    return json.dumps(result_to_serialize, ensure_ascii=False)


if __name__ == '__main__':
    ''' 准备工作: 模拟json参数输入 ------------------------------------------------------------------------------ '''
    with open('sample_A02_input.json', 'r', encoding='utf-8') as f:
        json_str = f.read()
    # excel信息
    excel_path = '历史净值数据_万得指数.xlsx'
    excel_sheet = '历史净值数据'

    ''' 调用主函数进行计算 -------------------------------------------------------------------------------------- '''
    result_json = main(json_str, excel_path, excel_sheet)
    print("\n最终返回的结果 Json 字符串为：\n", result_json)
