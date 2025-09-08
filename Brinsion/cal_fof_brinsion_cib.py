# -*- encoding: utf-8 -*-
"""
@File: cal_fund_attribution.py
@Modify Time: 2025/9/5 11:00
@Author: Kevin-Chen
@Descriptions: 支持BHB和BF两种归因模型的基金业绩归因分析
"""

import numpy as np
import pandas as pd
from typing import Dict, Literal, Optional, Tuple

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

brinson_result = Dict[str, object]  # 约定：同 perform_attribution 的 results(dict)


def _normalize_series_weight(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    total = s.sum()
    if total <= 0:
        raise ValueError("FOF 权重之和<=0；请检查输入。")
    return s / total


def _ensure_industry_columns(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """确保存在 AR_k / SR_k / IR_k 列（BF 无 IR，则补0）"""
    need_cols = ["industry", "AR_k", "SR_k"]
    if method == "BHB":
        need_cols += ["IR_k"]
    out = df.copy()
    for c in need_cols:
        if c not in out.columns:
            if c == "industry":
                raise ValueError("industry_table 缺少 'industry' 列。")
            out[c] = 0.0
    return out[need_cols]


def _pick_totals_from_single_result(res: brinson_result, method: str) -> pd.Series:
    """抽取单只基金的总因子（标量）"""
    if method == "BHB":
        cols = ["Total_AR", "Total_SR", "Total_IR", "ER", "TR"]
    elif method == "BF":
        cols = ["Total_AR", "Total_SR", "ER", "TR"]
    else:
        raise ValueError("method 必须是 'BHB' 或 'BF'")
    return pd.Series({c: float(res.get(c, 0.0)) for c in cols})


def _stack_industry_table(fund_code: str, res: brinson_result, Wj: float, method: str) -> pd.DataFrame:
    """将单基金行业明细乘以 FOF 权重后堆叠，返回列: [fund_code, industry, AR, SR, (IR)]"""
    ind = res["industry_table"]
    ind = _ensure_industry_columns(ind, method)
    # 乘以 FOF 权重
    out = ind.copy()
    out["AR"] = Wj * out["AR_k"]
    out["SR"] = Wj * out["SR_k"]
    if method == "BHB":
        out["IR"] = Wj * out["IR_k"]
    out.insert(0, "fund_code", fund_code)
    return out.drop(columns=[c for c in out.columns if c.endswith("_k")])


def aggregate_fof_from_funds(
        fof_hold: Dict[str, float],
        fund_results: Dict[str, brinson_result],
        method: Literal["BHB", "BF"] = "BHB",
        er_actual: Optional[float] = None,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    基于 单基金 Brinson 因子 → 聚合 的 FOF 归因。
    参数
    ----
    fof_hold: {fund_code: raw_weight}  —— FOF 对各基金的原始权重（未必归一）
    fund_results: {fund_code: results_dict} —— 单基金 Brinson 结果（同 perform_attribution 返回的 results）
    method: 'BHB' 或 'BF'
    er_actual: 若提供，则用它替换聚合出的 ER（用于产生交易/残差效应）。

    返回
    ----
    totals: dict —— FOF 总体因子（AR, SR, [IR], ER, TR）
    ind_table: DataFrame —— FOF 分行业因子（industry, AR, SR, [IR]）
    weights_table: DataFrame —— 聚合时使用的基金权重（含丢失基金提示）
    """
    # 1) 对齐基金清单并归一化权重（只保留有结果的基金）
    fof_w = pd.Series(fof_hold, name="W_raw").astype(float)
    available = set(fund_results.keys())
    used = fof_w.index.intersection(available)
    dropped = fof_w.index.difference(available)
    if len(used) == 0:
        raise ValueError("fof_hold 与 fund_results 无交集，无法聚合。")

    W = _normalize_series_weight(fof_w.loc[used]).rename("W_norm")
    weights_table = pd.DataFrame({"fund_code": used, "W_raw": fof_w.loc[used].values, "W_norm": W.values})
    if len(dropped) > 0:
        warn_df = pd.DataFrame({"fund_code": list(dropped), "W_raw": fof_w.loc[dropped].values})
        warn_df["note"] = "在 fund_results 中缺失，已从聚合中剔除并对剩余权重归一化"
        weights_table = pd.concat([weights_table, warn_df], ignore_index=True)

    # 2) 聚合“总因子”（标量）：Σ_j W_j * Total_x_j
    tot_rows = []
    for j, Wj in W.items():
        s = _pick_totals_from_single_result(fund_results[j], method)
        s = s * Wj
        s.name = j
        tot_rows.append(s)
    tot_df = pd.DataFrame(tot_rows)
    totals_weighted = tot_df.sum(axis=0)

    # 3) 可选：用真实 ER 覆盖（以产生 TR）
    if er_actual is not None:
        ER = float(er_actual)
    else:
        ER = float(totals_weighted["ER"])

    if method == "BHB":
        AR = float(totals_weighted["Total_AR"])
        SR = float(totals_weighted["Total_SR"])
        IR = float(totals_weighted["Total_IR"])
        TR = ER - (AR + SR + IR)
        totals = {"AR": AR, "SR": SR, "IR": IR, "ER": ER, "TR": TR}
    else:  # BF
        AR = float(totals_weighted["Total_AR"])
        SR = float(totals_weighted["Total_SR"])
        TR = ER - (AR + SR)
        totals = {"AR": AR, "SR": SR, "ER": ER, "TR": TR}

    # 4) 聚合“分行业因子”：对每只基金的行业因子乘以 W_j 再行业汇总
    ind_blocks = []
    for j, Wj in W.items():
        blk = _stack_industry_table(j, fund_results[j], Wj, method)
        ind_blocks.append(blk)
    ind_all = pd.concat(ind_blocks, ignore_index=True)

    # 汇总为 FOF 行业因子
    group_cols = ["industry"]
    agg_dict = {"AR": "sum", "SR": "sum"}
    if method == "BHB":
        agg_dict["IR"] = "sum"
    ind_table = ind_all.groupby(group_cols, as_index=False).agg(agg_dict)

    # 5) 一致性自检（闭合）
    # 分行业合计 vs 总体（允许微小数值误差）
    if method == "BHB":
        chk = (abs(ind_table["AR"].sum() - totals["AR"]) +
               abs(ind_table["SR"].sum() - totals["SR"]) +
               abs(ind_table["IR"].sum() - totals["IR"]))
    else:
        chk = (abs(ind_table["AR"].sum() - totals["AR"]) +
               abs(ind_table["SR"].sum() - totals["SR"]))
    if chk > 1e-10:
        raise AssertionError(f"行业合计与总体不闭合，差值={chk}")

    # 返回
    return totals, ind_table.sort_values("industry").reset_index(drop=True), weights_table.sort_values("fund_code")


if __name__ == '__main__':
    # 组合FOF持仓
    fof_hold = {'000628': 0.21,  # 大成高鑫股票A
                '000729': 0.22,  # 建信中小盘先锋股票A
                '000780': 0.11,  # 鹏华医疗保健股票
                '470888': 0.16,  # 华宝制造股票
                }
