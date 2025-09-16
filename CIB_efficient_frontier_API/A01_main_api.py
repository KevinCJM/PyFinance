# -*- encoding: utf-8 -*-
"""
@File: A01_main_api.py
@Modify Time: 2025/9/16 11:05       
@Author: Kevin-Chen
@Descriptions: 
"""
import json
import pandas as pd
import plotly.colors

from T04_show_plt import plot_efficient_frontier
from T02_other_tools import load_returns_from_excel
from T01_generate_random_weights import multi_level_random_walk_config
from T03_weight_limit_cal import level_weight_limit_cal, hold_weight_limit_cal


# 帮助函数：生成悬停文本
def format_hover_text(row, title, a_list):
    text = f"<b>{title}</b><br>年化收益率: {row['ret_annual']:.2%}<br>年化风险: {row['risk_arr']:.2%}"
    text += "<br><br><b>--权重--</b><br>"
    for asset in a_list:
        text += f"{asset}: {row[asset]:.2%}<br>"
    return text.strip('<br>')


if __name__ == '__main__':
    ''' 0) 准备工作: 模拟json参数输入 ------------------------------------------------------------------------------- '''
    # 字典格式入参
    dict_input = {
        'asset_list': [
            '货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类'
        ],
        'WeightRange': {
            'C1': {'货币现金类': [1.0, 1.0], '固定收益类': [0.0, 0.0], '混合策略类': [0.0, 0.0],
                   '权益投资类': [0.0, 0.0], '另类投资类': [0.0, 0.0]},
            'C2': {'货币现金类': [0.0, 1.0], '固定收益类': [0.0, 1.0], '混合策略类': [0.0, 0.0],
                   '权益投资类': [0.0, 0.0], '另类投资类': [0.0, 0.0]},
            'C3': {'货币现金类': [0.0, 1.0], '固定收益类': [0.0, 1.0], '混合策略类': [0.0, 0.5],
                   '权益投资类': [0.0, 0.0], '另类投资类': [0.0, 0.0]},
            'C4': {'货币现金类': [0.0, 1.0], '固定收益类': [0.0, 1.0], '混合策略类': [0.0, 0.6],
                   '权益投资类': [0.0, 0.2], '另类投资类': [0.0, 0.1]},
            'C5': {'货币现金类': [0.0, 1.0], '固定收益类': [0.0, 1.0], '混合策略类': [0.0, 0.8],
                   '权益投资类': [0.0, 0.5], '另类投资类': [0.0, 0.3]},
            'C6': {'货币现金类': [0.0, 1.0], '固定收益类': [0.0, 1.0], '混合策略类': [0.0, 1.0],
                   '权益投资类': [0.0, 0.7], '另类投资类': [0.0, 0.5]}
        },
        'StandardProportion': {
            'C1': {'货币现金类': 1.0, '固定收益类': 0.0, '混合策略类': 0.0, '权益投资类': 0.0, '另类投资类': 0.0},
            'C2': {'货币现金类': 0.2, '固定收益类': 0.8, '混合策略类': 0.0, '权益投资类': 0.0, '另类投资类': 0.0},
            'C3': {'货币现金类': 0.1, '固定收益类': 0.55, '混合策略类': 0.35, '权益投资类': 0.0, '另类投资类': 0.0},
            'C4': {'货币现金类': 0.05, '固定收益类': 0.4, '混合策略类': 0.3, '权益投资类': 0.2, '另类投资类': 0.05},
            'C5': {'货币现金类': 0.05, '固定收益类': 0.2, '混合策略类': 0.25, '权益投资类': 0.4, '另类投资类': 0.1},
            'C6': {'货币现金类': 0.05, '固定收益类': 0.1, '混合策略类': 0.15, '权益投资类': 0.6, '另类投资类': 0.1}
        },
        'user_holding': {
            'WeightRange': {
                '货币现金类': [0.0, 1.0], '固定收益类': [0.0, 1.0], '混合策略类': [0.0, 0.4],
                '权益投资类': [0.0, 0.7], '另类投资类': [0.0, 0.12]
            },
            'StandardProportion': {
                '货币现金类': 0.1, '固定收益类': 0.2, '混合策略类': 0.2, '权益投资类': 0.4, '另类投资类': 0.1
            },
            'can_sell': {
                '货币现金类': False, '固定收益类': True, '混合策略类': False, '权益投资类': True, '另类投资类': True
            },
            'can_buy': {
                '货币现金类': True, '固定收益类': True, '混合策略类': False, '权益投资类': True, '另类投资类': False
            }
        }
    }
    # 字典转Json, 模拟输入的Json参数
    json_str = json.dumps(dict_input, ensure_ascii=False)
    print(json_str)

    ''' 0) 准备工作: 配置一些预定义的参数 ----------------------------------------------------------------------------- '''
    # 随机游走与指标参数（仅字典方式）
    RANDOM_SEED = 12345
    # 字典式多轮配置（参数含义见 multi_level_random_walk_config 注释）
    ROUNDS_CONFIG = {
        # 第0轮：初始化方式二选一：
        0: {
            "init_mode": "solver",  # "exploration" 随机探索 或 "solver" 求解器
            # exploration 参数（当 init_mode=="exploration" 生效）：
            "samples": 300,
            "step_size": 0.99,
            # solver 参数（当 init_mode=="solver" 生效）：
            "solver_params": {
                "n_grid": 5000,
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
    PRECISION_CHOICE: str | None = None

    ''' 1) 解析Json参数 & 读取大类收益率 ----------------------------------------------------------------------------- '''
    # Json转字典
    dict_input = json.loads(json_str)
    # 分解参数
    asset_list = dict_input['asset_list']  # 大类列表
    weight_range = dict_input['WeightRange']  # 标准组合约束
    standard_proportion = dict_input['StandardProportion']  # 标准组合
    user_holding = dict_input['user_holding']  # 客户持仓组合
    # 读取excel，生成日收益二维数组
    excel_path = '历史净值数据_万得指数.xlsx'
    excel_sheet = '历史净值数据'
    returns, assets = load_returns_from_excel(excel_path, excel_sheet, asset_list)
    # 用于绘图的点云数据
    scatter_points_data = list()

    ''' 2) 计算约束 ----------------------------------------------------------------------------------------------- '''
    # 计算标准组合的约束
    level_weight_limit = dict()
    for k, v in weight_range.items():
        single_limit, multi_limit = level_weight_limit_cal(asset_list, v)
        level_weight_limit[k] = {'single_limit': single_limit, 'multi_limit': multi_limit}
    print("标准组合的约束：", level_weight_limit)
    # 计算客户持仓的约束
    single_limit_hold, multi_limit_hold = hold_weight_limit_cal(asset_list, user_holding)
    hold_weight_limit = {'single_limit': single_limit_hold, 'multi_limit': multi_limit_hold}
    print("客户持仓的约束：", hold_weight_limit)

    ''' 3) 计算无约束的市场组合的随机权重和有效前沿 --------------------------------------------------------------------- '''
    single_limit = [(0.0, 1.0)] * len(asset_list)
    print(f"计算无约束的市场组合随机权重. 单资产约束: {single_limit}")
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
    print(f"无约束市场组合的随机权重计算完成. 权重数: {W_unc.shape[0]}")

    ''' 4) 计算标准组合的随机权重和有效前沿 ---------------------------------------------------------------------------- '''
    # 循环计算各个标准组合的随机权重以及权重对应的收益率和波动率
    random_weight_dict = dict()
    for k, v in level_weight_limit.items():
        single_limit = v['single_limit']
        multi_limit = v['multi_limit']
        print(f"计算标准组合 {k} 的随机权重. 单资产约束: {single_limit}; 多资产约束: {multi_limit}")
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
            'weights': W,
            'ret_annual': ret_annual,
            'risk_arr': risk_arr,
            'ef_mask': ef_mask
        }
        print(f"标准组合 {k} 的随机权重计算完成. 权重数: {W.shape[0]}")

    ''' 5) 绘图展示 ------------------------------------------------------------------------------------------------- '''
    print("\n开始准备绘图数据...")
    scatter_points_data = list()  # 重新初始化用于绘图的点云数据

    # a) 处理无约束组合
    df_unc = pd.DataFrame(W_unc, columns=asset_list)
    df_unc['ret_annual'] = ret_annual_unc
    df_unc['risk_arr'] = risk_arr_unc
    df_unc['hover_text'] = df_unc.apply(
        lambda r: format_hover_text(r, "无约束组合", asset_list), axis=1)

    # 无约束 - 随机点 (抽样以避免浏览器卡顿)
    scatter_points_data.append({
        "data": df_unc[~ef_mask_unc].sample(n=min(2000, len(df_unc[~ef_mask_unc]))),
        "name": "无约束-随机点", "color": "black", "size": 3, "opacity": 0.5,
    })
    # 无约束 - 有效前沿
    scatter_points_data.append({
        "data": df_unc[ef_mask_unc],
        "name": "无约束-有效前沿", "color": "black", "size": 3, "opacity": 0.9,
    })

    # b) 处理各标准组合
    level_colors = {
        'C1': '#1f77b4', 'C2': '#ff7f0e', 'C3': 'grey',
        'C4': 'green', 'C5': 'blue', 'C6': 'red'
    }
    for i, (k, v) in enumerate(random_weight_dict.items()):
        color = level_colors[k]
        df_level = pd.DataFrame(v['weights'], columns=asset_list)
        df_level['ret_annual'] = v['ret_annual']
        df_level['risk_arr'] = v['risk_arr']
        df_level['hover_text'] = df_level.apply(
            lambda r: format_hover_text(r, f"{k} 组合", asset_list), axis=1)
        level_ef_mask = v['ef_mask']

        # 标准组合 - 随机点 (抽样)
        scatter_points_data.append({
            "data": df_level[~level_ef_mask].sample(n=min(2000, len(df_level[~level_ef_mask]))),
            "name": f"{k}-随机点", "color": color, "size": 3, "opacity": 0.5,
        })
        # 标准组合 - 有效前沿
        scatter_points_data.append({
            "data": df_level[level_ef_mask],
            "name": f"{k}-有效前沿", "color": color, "size": 3, "opacity": 0.9,
        })

    # c) 调用绘图函数
    plot_efficient_frontier(
        scatter_points_data,
        title="各约束下的有效前沿对比",
        x_axis_title=f"年化风险 ({RISK_METRIC.upper()})",
        y_axis_title="年化收益率",
        x_col="risk_arr",
        y_col="ret_annual",
        hover_text_col="hover_text",
        output_filename=None
    )
