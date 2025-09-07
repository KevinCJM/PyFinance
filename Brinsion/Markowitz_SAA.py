# -*- encoding: utf-8 -*-
"""
@File: Markowitz_SAA.py
@Modify Time: 2025/9/6 17:56       
@Author: Kevin-Chen
@Descriptions: 马科维兹 - 战略资产配置 SAA
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


# 获取大类资产收益率
def get_category_ret(index_weight_dict):
    """
    根据指数权重字典计算大类资产收益率

    参数:
        index_weight_dict (dict): 包含大类资产分类及其对应指数代码和权重的字典
                                格式为 {category: {index_code: weight, ...}, ...}

    返回:
        pd.DataFrame: 包含各行业分类收益率的时间序列数据，索引为日期，列为资产分类名称
    """
    # 读取用于资产配置的指数数据
    index_codes = [code for category in index_weight_dict.values() for code in category.keys()]

    # 从 /data/index_daily_all.parquet 获取指定指数的日收益率数据
    index_daily = pd.read_parquet("./data/index_daily_all.parquet")
    index_daily = index_daily[index_daily["index_code"].isin(index_codes)].reset_index(drop=True)
    print(index_daily)

    # 按权重构建大类资产的收益率
    index_ret = index_daily.pivot(index="date", columns="index_code", values="change")

    # 按 index_weight_dict 聚合为大类资产收益率
    category_ret = pd.DataFrame(index=index_ret.index)

    # 遍历每个行业分类，计算加权收益率
    for category, codes_weights in index_weight_dict.items():
        # 计算当前分类的加权收益率：各成分股收益率乘以其权重后求和
        weighted_ret = sum(index_ret[code] * w for code, w in codes_weights.items())
        category_ret[category] = weighted_ret

    # 删除收益率为空的行业分类数据
    category_ret = category_ret.dropna(axis=0)

    # 返回各行业分类的收益率序列
    return category_ret


def optimize_saa(category_ret: pd.DataFrame,
                 mode: str = "max_sharpe",
                 *,
                 rf: float = 0.0,
                 target_vol: float = None,
                 target_ret: float = None,
                 bounds=None,
                 w0=None,
                 cov_ridge: float = 1e-10):
    """
    mode 取值：
      - "max_sharpe"                  最大化夏普
      - "max_return"                  最大化收益率
      - "min_vol"                     最小化波动率
      - "max_return_at_risk"          指定(上限)风险下最大化收益率，需要 target_vol
      - "min_vol_at_return"           指定(下限)收益下最小化风险，需要 target_ret
    其它：
      - rf: 无风险日利率（默认0）
      - cov_ridge: 对协方差对角线加微小脊值，增强数值稳定性
    """
    assert isinstance(category_ret, pd.DataFrame) and category_ret.shape[1] >= 2
    mu = category_ret.mean().values.astype(float)  # 日度期望收益向量
    cov = category_ret.cov().values.astype(float)  # 日度协方差矩阵
    n = mu.shape[0]
    cov = cov + cov_ridge * np.eye(n)  # 数值稳定化

    # 默认边界与起点：长仓、等权
    if bounds is None:
        bounds = [(0.0, 1.0)] * n
    if w0 is None:
        w0 = np.full(n, 1.0 / n, dtype=float)

    # 便捷函数：组合收益/波动
    def port_ret(w):
        return float(w @ mu)

    def port_var(w):
        return float(w @ (cov @ w))

    def port_vol(w):
        v = port_var(w)
        return np.sqrt(v) if v > 0.0 else 0.0

    eps = 1e-12  # 防止除零

    # ========== 目标函数（最小化） ==========
    if mode == "max_sharpe":
        def obj(w):
            r = port_ret(w) - rf
            s = port_vol(w)
            return -(r / (s + eps))

        extra_constraints = []
    elif mode == "max_return":
        def obj(w):
            return -port_ret(w)

        extra_constraints = []
    elif mode == "min_vol":
        def obj(w):
            return port_vol(w)

        extra_constraints = []
    elif mode == "max_return_at_risk":
        assert target_vol is not None and target_vol > 0, "需要指定 target_vol > 0"

        def obj(w):
            return -port_ret(w)

        # SLSQP 的 'ineq' 约束要求 fun(w) >= 0
        extra_constraints = [{"type": "ineq", "fun": lambda w: target_vol - port_vol(w)}]
    elif mode == "min_vol_at_return":
        assert target_ret is not None, "需要指定 target_ret"

        def obj(w):
            return port_vol(w)

        extra_constraints = [{"type": "ineq", "fun": lambda w: port_ret(w) - target_ret}]
    else:
        raise ValueError(f"未知 mode: {mode}")

    # ========== 通用约束：资金守恒 + 长仓 ==========
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    constraints.extend(extra_constraints)

    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 200, "ftol": 1e-12})

    w = np.clip(res.x, 0.0, 1.0)  # 数值边界清理
    # 归一化，确保和为1（避免数值误差）
    s = w.sum()
    if s > 0:
        w /= s

    report = {
        "success": bool(res.success),
        "message": res.message,
        "weights": pd.Series(w, index=category_ret.columns),
        "ret": port_ret(w),
        "vol": port_vol(w),
        "sharpe": (port_ret(w) - rf) / (port_vol(w) + eps)
    }
    return report


if __name__ == '__main__':
    ''' 数据读取与预处理 '''
    index_weight = {
        "权益类": {"H11021": 1.0},
        "债券类": {"H11023": 1.0},
        "混合类": {"H11022": 1.0},
        "商品类": {"H30009": 1.0},
        "货币类": {"H11025": 1.0},
    }
    category_ret_df = get_category_ret(index_weight)

    ''' 构建 SAA 战略层配置 '''
    # 1) 最大化夏普
    r1 = optimize_saa(category_ret_df, mode="max_sharpe", rf=0.0)
    print(r1)
    # 2) 最大化收益率
    r2 = optimize_saa(category_ret_df, mode="max_return")
    print(r2)
    # 3) 最小化波动率
    r3 = optimize_saa(category_ret_df, mode="min_vol")
    print(r3)
    # 4) 指定风险(例：日波动率≤0.6%)下最大化收益
    r4 = optimize_saa(category_ret_df, mode="max_return_at_risk", target_vol=0.08)
    print(r4)
    # 5) 指定收益(例：日收益≥0.05%)下最小化风险
    r5 = optimize_saa(category_ret_df, mode="min_vol_at_return", target_ret=0.05)
    print(r5)
