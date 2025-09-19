# -*- encoding: utf-8 -*-
"""
@File: A03_ideal_portfolio_api.py
@Modify Time: 2025/9/17 08:05
@Author: Kevin-Chen
@Descriptions: 根据已有持仓寻找组合配置：
  3.1 计算客户当前持仓的风险和收益
  3.2 同等收益下，风险最小的有效前沿点（SLSQP热启动）
  3.3 同等风险下，收益最大的有效前沿点（SLSQP热启动）
  3.4 与客户当前持仓换仓最小的有效前沿点（按风险锚点用SLSQP求前沿点后挑选）
"""

import time
import json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from T04_show_plt import plot_efficient_frontier
from T02_other_tools import load_returns_from_excel, log, ann_log_return, ann_log_vol
from T01_generate_random_weights import (
    compute_perf_arrays,
    compute_var_parametric_arrays,
)
from T03_weight_limit_cal import hold_weight_limit_cal

# 全局参数（与其他模块保持一致）
TRADING_DAYS = 252.0
RISK_METRIC = "var"  # "vol" 或 "var"
VAR_PARAMS = {
    "confidence": 0.95,
    "horizon_days": 1.0,
    "return_type": "log",
    "ddof": 1,
    "clip_non_negative": True,
}


def _parse_series_from_dict(d):
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


def analysis_json_and_read_data(json_input: str, excel_name: str, sheet_name: str):
    """
    解析JSON输入并从Excel文件中读取数据

    参数:
        json_input (str): 包含配置信息的JSON字符串
        excel_name (str): Excel文件名
        sheet_name (str): Excel工作表名

    返回:
        tuple: 包含以下元素的元组:
            - asset_list: 资产列表
            - draw_plt: 是否绘制图表的标志
            - draw_plt_filename: 图表保存文件名
            - user_holding: 用户持仓数据
            - ef_data: 有效前沿数据，需包含"weights"字段
            - returns: 从Excel加载的收益率数据
            - refine_ef_before_select: 是否精炼有效前沿点集的标志
    """
    # 解析JSON输入参数
    params = json.loads(json_input)
    asset_list = params["asset_list"]
    draw_plt = params.get("draw_plt", True)
    draw_plt_filename = params.get("draw_plt_filename", None)
    user_holding = params["user_holding"]
    ef_data = params["ef_data"]  # 需包含 "weights": List[List[float]]
    # 开关：是否精炼有效前沿点集
    refine_ef_before_select = bool(params.get("refine_ef_before_select", False))

    # 加载收益率：优先使用 JSON 中的 nv；否则 Excel；都无则报错
    nv_data = params.get("nv")
    if nv_data is not None:
        returns = _load_returns_from_nv(nv_data, asset_list)
    else:
        if excel_name and sheet_name:
            returns, _ = load_returns_from_excel(excel_name, sheet_name, asset_list)
        else:
            raise ValueError("未提供净值数据(nv)，且缺少 Excel 读取参数(excel_name/sheet_name)")

    return asset_list, draw_plt, draw_plt_filename, user_holding, ef_data, returns, refine_ef_before_select


def _ann_log_ret_func(returns_daily: np.ndarray, w: np.ndarray, annual_trading_days: float = TRADING_DAYS) -> float:
    return ann_log_return(returns_daily, w, annual_trading_days)


def _risk_func(
        returns_daily: np.ndarray,
        w: np.ndarray,
        risk_metric: str = RISK_METRIC,
        var_params: Optional[Dict[str, Any]] = None,
        annual_trading_days: float = TRADING_DAYS,
) -> float:
    risk_metric = (risk_metric or "vol").lower()
    if risk_metric == "vol":
        return ann_log_vol(returns_daily, w, annual_trading_days, ddof=1)
    # 参数法VaR
    vp = var_params or {}
    try:
        from scipy.stats import norm
        z_score = float(norm.ppf(1.0 - float(vp.get("confidence", 0.95))))
    except Exception:
        # 退化为常用近似值（95% 左尾约 -1.645）
        z_score = -1.645
    confidence = float(vp.get("confidence", 0.95))
    horizon_days = float(vp.get("horizon_days", 1.0))
    return_type = str(vp.get("return_type", "log"))

    if return_type == "log":
        X = np.log1p(returns_daily @ w)
    else:
        X = returns_daily @ w
    mu = float(X.mean())
    sigma = float(X.std(ddof=int(vp.get("ddof", 1))))
    h = horizon_days
    mu_h = mu * h
    sigma_h = sigma * np.sqrt(h)
    var = -(mu_h + z_score * sigma_h)
    return max(0.0, var) if bool(vp.get("clip_non_negative", True)) else var


def _build_constraints(single_limit: List[Tuple[float, float]],
                       multi_limit: Dict[Tuple[int, ...], Tuple[float, float]]):
    bounds = list(single_limit)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    for idx_tuple, (lo, hi) in (multi_limit or {}).items():
        idx = list(idx_tuple)
        if idx:
            cons.append({"type": "ineq", "fun": lambda w, i=idx, l=lo: np.sum(w[i]) - l})
            cons.append({"type": "ineq", "fun": lambda w, i=idx, h=hi: h - np.sum(w[i])})
    return bounds, cons


def _solve_efficient_given_risk_cap(
        returns_daily: np.ndarray,
        bounds: List[Tuple[float, float]],
        base_cons: List[Dict[str, Any]],
        r_cap: float,
        w0: np.ndarray,
) -> np.ndarray:
    """在固定风险上限 r_cap 下最大化年化收益，返回权重；失败则返回 None。"""
    cons = list(base_cons) + [
        {"type": "ineq",
         "fun": lambda w, t=r_cap: t - _risk_func(returns_daily, w, risk_metric=RISK_METRIC, var_params=VAR_PARAMS)}
    ]
    res = minimize(
        lambda w: -_ann_log_ret_func(returns_daily, w),
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 800, "disp": False},
    )
    if res.success:
        return res.x
    return None


# 根据目标值选择最优的权重向量作为热启动点
def _select_hot_start_by_target(
        W: np.ndarray,
        target: float,
        returns_daily: np.ndarray,
        mode: str = "ret",
) -> np.ndarray:
    """
    根据目标值选择最优的权重向量作为热启动点

    该函数通过比较不同权重向量对应的风险或收益指标与目标值的差异，
    选择差异最小的权重向量作为热启动的起始点。

    参数:
        W: 权重矩阵，每一行代表一个权重向量
        target: 目标值，用于匹配最优权重向量
        returns_daily: 日收益率数组
        mode: 选择模式，"ret"表示按收益匹配，其他值表示按风险匹配

    返回:
        np.ndarray: 与目标值最匹配的权重向量，如果W为空则返回None
    """
    # 检查权重矩阵是否为空
    if W.size == 0:
        return None

    # 按收益模式进行匹配
    if mode == "ret":
        ret_ann, _ = compute_perf_arrays(returns_daily, W, trading_days=TRADING_DAYS, return_type="log")
        idx = int(np.argmin(np.abs(ret_ann - target)))
    else:
        # 按风险模式进行匹配
        if RISK_METRIC == "vol":
            # 使用波动率作为风险指标
            _, vol_ann = compute_perf_arrays(returns_daily, W, trading_days=TRADING_DAYS, return_type="log")
            risk_arr = vol_ann
        else:
            # 使用VaR作为风险指标
            risk_arr = compute_var_parametric_arrays(
                returns_daily, W, confidence=VAR_PARAMS["confidence"], horizon_days=VAR_PARAMS["horizon_days"],
                return_type=VAR_PARAMS["return_type"], ddof=VAR_PARAMS["ddof"],
                clip_non_negative=VAR_PARAMS["clip_non_negative"],
            )
        idx = int(np.argmin(np.abs(risk_arr - target)))

    # 返回与目标值最匹配的权重向量
    return W[idx]


# 计算单个组合的年化收益、风险与换手率
def compute_point_metrics(
        returns_daily: np.ndarray,
        w: np.ndarray,
        w_user: Optional[np.ndarray] = None,
        *,
        risk_metric: str = RISK_METRIC,
        var_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    计算单个组合的年化收益、风险与换手率。

    参数:
        returns_daily: 日收益率数组
        w: 组合权重数组
        w_user: 用户指定的权重数组，用于计算换手率，默认为None
        risk_metric: 风险度量指标名称，默认为RISK_METRIC
        var_params: VaR参数字典，包含置信水平、时间跨度等参数，默认为None

    返回:
        包含年化收益、风险值和换手率的字典，键包括：
        - ret_annual: 年化收益率
        - risk_value: 风险值（波动率或VaR）
        - turnover_l1_half: 换手率（L1/2范数）
    """
    # 计算组合的收益和波动率数组
    ret_arr, vol_arr = compute_perf_arrays(returns_daily, w.reshape(1, -1), trading_days=TRADING_DAYS,
                                           return_type="log")
    ret = float(ret_arr[0])

    # 根据风险度量类型选择计算方式
    if (risk_metric or "vol").lower() == "vol":
        risk = float(vol_arr[0])
    else:
        # 计算参数化VaR风险值
        va = compute_var_parametric_arrays(
            returns_daily,
            w.reshape(1, -1),
            confidence=(var_params or VAR_PARAMS)["confidence"],
            horizon_days=(var_params or VAR_PARAMS)["horizon_days"],
            return_type=(var_params or VAR_PARAMS)["return_type"],
            ddof=(var_params or VAR_PARAMS)["ddof"],
            clip_non_negative=(var_params or VAR_PARAMS)["clip_non_negative"],
        )
        risk = float(va[0])

    # 计算换手率（L1/2范数）
    turnover = 0.0
    if w_user is not None:
        turnover = 0.5 * float(np.abs(w - w_user).sum())

    return {"ret_annual": ret, "risk_value": risk, "turnover_l1_half": turnover}


# 批量计算点集的年化收益、风险与换手率
def compute_array_metrics(
        returns_daily: np.ndarray,
        W: np.ndarray,
        w_user: Optional[np.ndarray] = None,
        *,
        risk_metric: str = RISK_METRIC,
        var_params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """批量计算点集的年化收益、风险与(可选)换手率(L1/2)。

    参数
    ----------
    returns_daily : np.ndarray
        日收益率数组
    W : np.ndarray
        权重矩阵，每行代表一个权重向量
    w_user : Optional[np.ndarray], optional
        用户权重向量，用于计算换手率，默认为None
    risk_metric : str, optional
        风险度量指标，可选"vol"(波动率)或其他VaR指标，默认为RISK_METRIC
    var_params : Dict[str, Any] | None, optional
        VaR计算参数字典，包含confidence、horizon_days、return_type、ddof、clip_non_negative等参数，默认为None

    返回值
    -------
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
        包含年化收益数组、风险数组和换手率数组的元组
    """
    # 计算年化收益和波动率数组
    ret_arr, vol_arr = compute_perf_arrays(returns_daily, W, trading_days=TRADING_DAYS, return_type="log")

    # 根据风险度量指标选择风险计算方式
    if (risk_metric or "vol").lower() == "vol":
        risk_arr = vol_arr
    else:
        risk_arr = compute_var_parametric_arrays(
            returns_daily,
            W,
            confidence=(var_params or VAR_PARAMS)["confidence"],
            horizon_days=(var_params or VAR_PARAMS)["horizon_days"],
            return_type=(var_params or VAR_PARAMS)["return_type"],
            ddof=(var_params or VAR_PARAMS)["ddof"],
            clip_non_negative=(var_params or VAR_PARAMS)["clip_non_negative"],
        )

    # 计算换手率(L1/2距离)
    turn_arr = None
    if w_user is not None:
        turn_arr = 0.5 * np.abs(W - w_user.reshape(1, -1)).sum(axis=1)

    return ret_arr, risk_arr, turn_arr


# 查找同等收益下风险最小的组合
def solve_same_return_min_risk(
        returns_daily: np.ndarray,
        single_limit: List[Tuple[float, float]],
        multi_limit: Dict[Tuple[int, ...], Tuple[float, float]],
        target_ret: float,
        w0: np.ndarray,
) -> np.ndarray:
    """
    同等收益下风险最小：min risk(w) s.t. ann_ret(w)=target_ret, constraints。失败返回 w0。

    该函数通过优化求解在给定目标收益率约束下使风险最小化的资产权重分配问题。

    参数:
        returns_daily: np.ndarray - 资产的日收益率矩阵，形状为(交易日数, 资产数量)
        single_limit: List[Tuple[float, float]] - 各资产权重的单边限制条件，每个元组表示(最小权重, 最大权重)
        multi_limit: Dict[Tuple[int, ...], Tuple[float, float]] - 多资产组合的联合限制条件，
                    键为资产索引元组，值为(最小权重和, 最大权重和)
        target_ret: float - 目标年化收益率
        w0: np.ndarray - 初始权重向量

    返回:
        np.ndarray - 优化后的权重向量，如果优化失败则返回初始权重w0
    """
    # 构建优化问题的边界约束和基础约束条件
    bounds, base_cons = _build_constraints(single_limit, multi_limit)

    # 添加目标收益率等式约束：年化收益率等于目标收益率
    cons = base_cons + [{"type": "eq", "fun": lambda w, r=target_ret: _ann_log_ret_func(returns_daily, w) - r}]

    # 调用优化器求解最小风险问题
    res = minimize(
        lambda w: _risk_func(returns_daily, w, risk_metric=RISK_METRIC, var_params=VAR_PARAMS),
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 800, "disp": False},
    )

    # 返回优化结果，失败时返回初始权重
    return res.x if res.success else w0


# 查找同等风险下收益最大的组合
def solve_same_risk_max_return(
        returns_daily: np.ndarray,
        single_limit: List[Tuple[float, float]],
        multi_limit: Dict[Tuple[int, ...], Tuple[float, float]],
        target_risk: float,
        w0: np.ndarray,
) -> np.ndarray:
    """同等风险下收益最大：max ann_ret(w) s.t. risk(w)=target_risk, constraints。失败返回 w0。

    该函数通过优化求解在给定风险水平下使年化收益最大化的资产配置权重。

    参数:
        returns_daily: 日收益率矩阵，形状为 (交易日数量, 资产数量)
        single_limit: 单个资产的权重约束列表，每个元素为 (下限, 上限) 的元组
        multi_limit: 多个资产组合的权重约束字典，键为资产索引元组，值为 (下限, 上限) 的元组
        target_risk: 目标风险水平
        w0: 初始权重向量

    返回:
        np.ndarray: 最优化后的权重向量，如果优化失败则返回初始权重 w0
    """
    # 构建优化问题的边界约束和基础约束条件
    bounds, base_cons = _build_constraints(single_limit, multi_limit)

    # 添加风险等于目标风险的等式约束
    cons = base_cons + [{"type": "eq",
                         "fun": lambda w, t=target_risk: _risk_func(returns_daily, w, risk_metric=RISK_METRIC,
                                                                    var_params=VAR_PARAMS) - t}]

    # 求解优化问题：最小化负的年化对数收益率（即最大化年化收益率）
    res = minimize(
        lambda w: -_ann_log_ret_func(returns_daily, w),
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 800, "disp": False},
    )

    # 返回优化结果，如果失败则返回初始权重
    return res.x if res.success else w0


def find_min_turnover_on_ef(
        returns_daily: np.ndarray,
        W_ef: np.ndarray,
        risk_seed: np.ndarray,
        single_limit: List[Tuple[float, float]],
        multi_limit: Dict[Tuple[int, ...], Tuple[float, float]],
        w_user: np.ndarray,
        *,
        refine_before_select: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    在“有效前沿上”寻找相对 w_user 换手率(L1/2)最小的组合。

    - refine_before_select=False：直接在传入前沿点集中按 L1/2 选择最小者（不做再优化）。
    - refine_before_select=True：对每个传入点的风险 r_cap，用 SLSQP 解“risk(w) ≤ r_cap 且收益最大”的高效点，
      组成精炼前沿，再在其中按 L1/2 选择最小者。

    返回: (w_min_turnover, W_refined 或 None)
    """
    if not refine_before_select:
        W_scan = W_ef
        W_refined = None
    else:
        bounds2, base_cons2 = _build_constraints(single_limit, multi_limit)
        solutions: List[np.ndarray] = []
        for i in range(W_ef.shape[0]):
            r_cap_i = float(risk_seed[i])
            w0_i = W_ef[i]
            w_sol = _solve_efficient_given_risk_cap(returns_daily, bounds2, base_cons2, r_cap_i, w0_i)
            if w_sol is None:
                w_sol = w0_i
            solutions.append(w_sol)
        W_refined = np.vstack(solutions)
        W_scan = W_refined

    l1_all = np.abs(W_scan - w_user.reshape(1, -1)).sum(axis=1)
    idx_min = int(np.argmin(l1_all))
    w_min_turnover = W_scan[idx_min]
    return w_min_turnover, W_refined


def _make_scatter_data(
        asset_list: List[str],
        W_ef: np.ndarray,
        returns_daily: np.ndarray,
        user_point: Dict[str, Any],
        p_same_ret: Dict[str, Any],
        p_same_risk: Dict[str, Any],
        p_min_turnover: Dict[str, Any],
        W_ef_refined: Optional[np.ndarray] = None,
        p_mid_better: Optional[Dict[str, Any]] = None,
):
    """
    构造用于前端散点图展示的资产组合数据，包括有效前沿、当前持仓、特定优化目标点等。

    参数:
        asset_list (List[str]): 资产名称列表。
        W_ef (np.ndarray): 原始有效前沿上的资产权重矩阵，形状为 (N, M)，N为组合数，M为资产数。
        returns_daily (np.ndarray): 每日收益率矩阵，形状为 (T, M)，T为时间步数，M为资产数。
        user_point (Dict[str, Any]): 当前用户持仓信息，包含 "weights" 键。
        p_same_ret (Dict[str, Any]): 与当前持仓收益相同、风险最小的组合点。
        p_same_risk (Dict[str, Any]): 与当前持仓风险相同、收益最大的组合点。
        p_min_turnover (Dict[str, Any]): 在有效前沿上与当前持仓换手率最小的组合点。
        W_ef_refined (Optional[np.ndarray]): 可选的精炼后的有效前沿权重矩阵。

    返回:
        List[Dict]: 用于绘图的散点图数据列表，每个元素为一个字典，包含绘图所需的数据和样式信息。
    """
    # EF 曲线
    df_ef = pd.DataFrame(W_ef, columns=asset_list)
    ret_ann, vol_ann = compute_perf_arrays(returns_daily, W_ef, trading_days=TRADING_DAYS, return_type="log")
    if RISK_METRIC == "vol":
        risk_arr = vol_ann
    else:
        risk_arr = compute_var_parametric_arrays(
            returns_daily, W_ef, confidence=VAR_PARAMS["confidence"], horizon_days=VAR_PARAMS["horizon_days"],
            return_type=VAR_PARAMS["return_type"], ddof=VAR_PARAMS["ddof"],
            clip_non_negative=VAR_PARAMS["clip_non_negative"],
        )
    df_ef["ret_annual"] = ret_ann
    df_ef["risk_arr"] = risk_arr
    # 计算相对当前持仓的单边换手率（L1/2）
    w_user_arr = np.asarray(user_point.get("weights", [0] * len(asset_list)), dtype=float).reshape(1, -1)
    df_ef["turnover_l1_half"] = (np.abs(W_ef - w_user_arr).sum(axis=1) * 0.5)

    def _format_hover_row(r: pd.Series) -> str:
        """
        格式化鼠标悬停时显示的文本信息。
        """
        parts = [
            "<b>有效前沿</b>",
            f"年化收益率: {r['ret_annual']:.2%}",
            f"年化风险: {r['risk_arr']:.2%}",
            f"换手率(单边 L1/2): {r['turnover_l1_half']:.2%}",
            "<br><b>--权重--</b>",
        ]
        for a in asset_list:
            if a in r and pd.notna(r[a]):
                parts.append(f"{a}: {r[a]:.2%}")
        return "<br>".join(parts)

    df_ef["hover_text"] = df_ef.apply(_format_hover_row, axis=1)

    def _point_df(name: str, p: Dict[str, Any], color: str, symbol: str):
        """
        构造一个特定点（如当前持仓、优化目标点）的绘图数据结构。

        参数:
            name (str): 点的名称。
            p (Dict[str, Any]): 点的信息，包括权重、年化收益、风险等。
            color (str): 点的颜色。
            symbol (str): 点的形状。

        返回:
            Dict: 包含绘图所需数据和样式的字典。
        """
        row = {"ret_annual": p["ret_annual"], "risk_arr": p["risk_value"]}
        for a, w in zip(asset_list, p["weights"]):
            row[a] = w
        df = pd.DataFrame([row])
        weights_text = "<br>".join([f"{a}: {w:.2%}" for a, w in zip(asset_list, p["weights"])])
        # 若未提供换手率，按当前持仓计算
        if "turnover_l1_half" not in p:
            w_user_arr = np.asarray(user_point.get("weights", [0] * len(asset_list)), dtype=float)
            p["turnover_l1_half"] = float(np.abs(np.asarray(p["weights"]) - w_user_arr).sum() * 0.5)
        turnover_line = f"<br>换手率(单边 L1/2): {p['turnover_l1_half']:.2%}"
        df["hover_text"] = (
            f"<b>{name}</b><br>年化收益率: {p['ret_annual']:.2%}<br>年化风险: {p['risk_value']:.2%}"
            f"{turnover_line}" \
            f"<br><b>--权重--</b><br>{weights_text}"
        )
        return {
            "data": df,
            "name": name,
            "color": color,
            "size": 12,
            "opacity": 1.0,
            "symbol": symbol,
            "marker_line": dict(width=1, color="black"),
        }

    scatter = [
        {"data": df_ef, "name": "有效前沿(传入)", "color": "blue", "size": 3, "opacity": 0.9},
        _point_df("当前持仓", user_point, "cyan", "diamond"),
        _point_df("同收益最小风险", p_same_ret, "red", "star"),
        _point_df("同风险最大收益", p_same_risk, "orange", "star"),
        _point_df("换仓最小(在EF上)", p_min_turnover, "purple", "cross"),
    ]
    if p_mid_better is not None:
        scatter.append(_point_df("收益↑ 风险↓ (中点)", p_mid_better, "goldenrod", "triangle-up"))
    # 可选：展示精炼的前沿点集
    if W_ef_refined is not None and W_ef_refined.size > 0:
        df_ref = pd.DataFrame(W_ef_refined, columns=asset_list)
        ret_ref, vol_ref = compute_perf_arrays(returns_daily, W_ef_refined, trading_days=TRADING_DAYS,
                                               return_type="log")
        if RISK_METRIC == "vol":
            risk_ref = vol_ref
        else:
            risk_ref = compute_var_parametric_arrays(
                returns_daily, W_ef_refined, confidence=VAR_PARAMS["confidence"],
                horizon_days=VAR_PARAMS["horizon_days"],
                return_type=VAR_PARAMS["return_type"], ddof=VAR_PARAMS["ddof"],
                clip_non_negative=VAR_PARAMS["clip_non_negative"],
            )
        df_ref["ret_annual"] = ret_ref
        df_ref["risk_arr"] = risk_ref
        w_user_arr = np.asarray(user_point.get("weights", [0] * len(asset_list)), dtype=float).reshape(1, -1)
        df_ref["turnover_l1_half"] = (np.abs(W_ef_refined - w_user_arr).sum(axis=1) * 0.5)

        def _fmt_ref(r: pd.Series) -> str:
            """
            格式化精炼前沿点的悬停文本。
            """
            parts = [
                "<b>精炼前沿</b>",
                f"年化收益率: {r['ret_annual']:.2%}",
                f"年化风险: {r['risk_arr']:.2%}",
                f"换手率(单边 L1/2): {r['turnover_l1_half']:.2%}",
                "<br><b>--权重--</b>",
            ]
            for a in asset_list:
                if a in r and pd.notna(r[a]):
                    parts.append(f"{a}: {r[a]:.2%}")
            return "<br>".join(parts)

        df_ref["hover_text"] = df_ref.apply(_fmt_ref, axis=1)
        scatter.insert(1, {"data": df_ref, "name": "有效前沿(精炼)", "color": "green", "size": 3, "opacity": 0.9})
    return scatter


def main(json_input: str, excel_name: str, sheet_name: str) -> str:
    try:
        # 1) 解析参数 & 读取数据 -------------------------------------------------------------------------------------
        (asset_list, draw_plt, draw_plt_filename, user_holding, ef_data,
         returns, refine_ef_before_select) = analysis_json_and_read_data(
            json_input, excel_name, sheet_name)

        # 2) 计算约束 -----------------------------------------------------------------------------------------------
        single_limit, multi_limit = hold_weight_limit_cal(asset_list, user_holding)

        # 3) 基于EF输入与SLSQP热启动计算点 -----------------------------------------------------------------------------
        W_seed = np.asarray(ef_data.get("weights", []), dtype=float)
        if W_seed.ndim != 2 or W_seed.shape[1] != len(asset_list):
            raise ValueError("ef_data['weights'] 维度不正确或与资产列表不匹配")

        ret_seed, vol_seed = compute_perf_arrays(
            returns, W_seed, trading_days=TRADING_DAYS, return_type="log")
        risk_seed = vol_seed if RISK_METRIC == "vol" else compute_var_parametric_arrays(
            returns, W_seed,
            confidence=VAR_PARAMS["confidence"], horizon_days=VAR_PARAMS["horizon_days"],
            return_type=VAR_PARAMS["return_type"], ddof=VAR_PARAMS["ddof"],
            clip_non_negative=VAR_PARAMS["clip_non_negative"],
        )
        W_ef = W_seed
        # 当前持仓
        w_user = np.array([user_holding["StandardProportion"].get(a, 0.0) for a in asset_list], dtype=float)
        met_user = compute_point_metrics(returns, w_user)
        user_point = {"weights": w_user.tolist(), **met_user}

        # 3.1 同收益最小风险 -----------------------------------------------------------------------------------------
        w0_same_ret = _select_hot_start_by_target(W_ef, met_user["ret_annual"], returns, mode="ret")
        res_same_ret_x = solve_same_return_min_risk(returns, single_limit, multi_limit,
                                                    met_user["ret_annual"], w0_same_ret)
        met_same_ret = compute_point_metrics(returns, res_same_ret_x, w_user)
        p_same_ret = {"weights": res_same_ret_x.tolist(), **met_same_ret}

        # 3.2 同风险最大收益 -----------------------------------------------------------------------------------------
        w0_same_risk = _select_hot_start_by_target(W_ef, met_user["risk_value"], returns, mode="risk")
        res_same_risk_x = solve_same_risk_max_return(
            returns, single_limit, multi_limit, met_user["risk_value"], w0_same_risk)
        met_same_risk = compute_point_metrics(returns, res_same_risk_x, w_user)
        p_same_risk = {"weights": res_same_risk_x.tolist(), **met_same_risk}

        # 3.3 在EF上换仓最小 -----------------------------------------------------------------------------------------
        w_min_turnover, W_refined = find_min_turnover_on_ef(
            returns, W_ef, risk_seed, single_limit, multi_limit,
            w_user, refine_before_select=refine_ef_before_select,
        )
        met_min_turn = compute_point_metrics(returns, w_min_turnover, w_user)
        p_min_turnover = {"weights": w_min_turnover.tolist(), **met_min_turn}

        # 3.4 收益↑风险↓ (EF中点)：在 EF 上选择靠近（同收益最小风险点 与 同风险最大收益点）中点的位置 --------------------------
        ef_for_mid = W_refined if (refine_ef_before_select and W_refined is not None and W_refined.size) else W_ef
        ret_arr_mid, risk_arr_mid, _ = compute_array_metrics(returns, ef_for_mid, None,
                                                             risk_metric=RISK_METRIC, var_params=VAR_PARAMS)
        target_ret = 0.5 * (p_same_ret["ret_annual"] + p_same_risk["ret_annual"])
        target_risk = 0.5 * (p_same_ret["risk_value"] + p_same_risk["risk_value"])
        d2 = (ret_arr_mid - target_ret) ** 2 + (risk_arr_mid - target_risk) ** 2
        mid_idx = int(np.argmin(d2))
        w_mid = ef_for_mid[mid_idx]
        met_mid = compute_point_metrics(returns, w_mid, w_user)
        p_mid_better = {"weights": w_mid.tolist(), **met_mid}

        # 4) 可选绘图 ------------------------------------------------------------------------------------------------
        if draw_plt:
            scatter_points_data = _make_scatter_data(
                asset_list, W_ef, returns, user_point, p_same_ret,
                p_same_risk, p_min_turnover, W_refined, p_mid_better
            )
            plot_efficient_frontier(
                scatter_points_data,
                title="持仓对比与推荐点(在EF上)",
                x_axis_title=f"年化风险 ({RISK_METRIC.upper()})",
                y_axis_title="年化收益率",
                x_col="risk_arr",
                y_col="ret_annual",
                hover_text_col="hover_text",
                output_filename=draw_plt_filename,
            )

        # 5) 返回 JSON 结果
        result = {
            "success": True,
            "same_return_min_risk": p_same_ret,
            "same_risk_max_return": p_same_risk,
            "min_turnover_on_ef": p_min_turnover,
            "better_return_lower_risk": p_mid_better
        }
        if refine_ef_before_select and W_refined is not None and W_refined.size:
            result["ef_refined"] = {"weights": W_refined.tolist()}
        return json.dumps(result, ensure_ascii=False)

    except FileNotFoundError as e:
        return json.dumps({
            "success": False,
            "error_code": "DATA_FILE_NOT_FOUND",
            "message": f"服务器端数据文件未找到: {getattr(e, 'filename', str(e))}"
        }, ensure_ascii=False)
    except ValueError as e:
        return json.dumps({
            "success": False,
            "error_code": "INVALID_DATA_OR_CONFIG",
            "message": f"数据或配置无效: {e}"
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": f"计算过程中发生未知错误: {type(e).__name__} - {e}"
        }, ensure_ascii=False)


if __name__ == '__main__':
    ''' 准备工作: 模拟json参数输入 ------------------------------------------------------------------------------ '''
    with open('sample_A03_input.json', 'r', encoding='utf-8') as f:
        json_str = f.read()
    excel_path = '历史净值数据_万得指数.xlsx'
    excel_sheet = '历史净值数据'

    ''' 计算并输出结果 ------------------------------------------------------------------------------------------ '''
    s_t = time.time()
    str_res = main(json_str, excel_path, excel_sheet)
    print("\n最终返回的结果 Json 字符串为：\n", str_res)
    print(f"\n总计算耗时: {time.time() - s_t:.3f} 秒")
