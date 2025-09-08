# -*- encoding: utf-8 -*-
"""
@File: cal_fund_attribution.py
@Modify Time: 2025/9/5 11:00
@Author: Kevin-Chen
@Descriptions: 权益类Brinsion归因, 计算FOF组合的因子, 支持BHB和BF两种归因模型的基金业绩归因分析
"""
import re
import numpy as np
import pandas as pd
from typing import Dict, Literal, Optional, Tuple


# ============ 1) 工具：把“长表 df”还原为 {fund_code: results_dict} ============
def _pivot_single_fund_from_summary(df_one: pd.DataFrame, method: str) -> Dict[str, object]:
    """
    将某只基金的长表记录（含总体&行业行）还原成 results(dict)：
      keys: 'Total_AR','Total_SR','ER','TR','Rp_eq','Rb_eq', [BHB 还有 'Total_IR'], 'industry_table'
      industry_table: columns=['industry','AR_k','SR_k', ('IR_k' for BHB)]
    """
    # 总体指标
    total_keys = {"Total_AR", "Total_SR", "ER", "TR", "Rp_eq", "Rb_eq"}
    if method == "BHB":
        total_keys.add("Total_IR")

    tot_map = (
        df_one.loc[df_one["index_code"].isin(total_keys), ["index_code", "index_value"]]
        .drop_duplicates(subset=["index_code"])
        .set_index("index_code")["index_value"]
        .to_dict()
    )
    # 行业明细：解析 “行业名_因子” 结构
    df_ind = df_one.loc[~df_one["index_code"].isin(total_keys)].copy()
    df_ind["index_code"] = df_ind["index_code"].str.strip()

    m = df_ind["index_code"].str.extract(r"^(?P<industry>.+)_(?P<kind>AR|SR|IR)$")
    ind_df = pd.concat([m, df_ind[["index_value"]]], axis=1).dropna(subset=["industry", "kind"])
    ind_piv = (
        ind_df.pivot_table(index="industry", columns="kind", values="index_value", aggfunc="sum")
        .rename(columns={"AR": "AR_k", "SR": "SR_k", "IR": "IR_k"})
        .reset_index()
    )
    if method == "BF":
        if "IR_k" not in ind_piv.columns:
            ind_piv["IR_k"] = 0.0  # BF 无交互项，补 0
        ind_piv = ind_piv[["industry", "AR_k", "SR_k"]]  # BF 不输出 IR_k

    results = {**tot_map, "industry_table": ind_piv}
    return results


def build_fund_results_map_from_summary(summary_df: pd.DataFrame, method: Literal["BHB", "BF"]) -> Dict[str, Dict]:
    """
    输入：你保存的汇总长表（含多只基金）；
    输出：{fund_code: results_dict}
    """
    req_cols = {"fund_code", "index_code", "index_value"}
    missing = req_cols - set(summary_df.columns)
    if missing:
        raise ValueError(f"summary_df 缺少必要列: {missing}")

    results_map = {}
    for fc, g in summary_df.groupby("fund_code"):
        results_map[str(fc)] = _pivot_single_fund_from_summary(g, method)
    return results_map


# ============ 2) 核心：按 FOF 权重聚合（分层聚合法） ============

def _normalize_series_weight(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    total = s.sum()
    if total <= 0:
        raise ValueError("FOF 权重之和<=0；请检查输入。")
    return s / total


def _ensure_industry_columns(df: pd.DataFrame, method: str) -> pd.DataFrame:
    need = ["industry", "AR_k", "SR_k"] + (["IR_k"] if method == "BHB" else [])
    out = df.copy()
    for c in need:
        if c not in out.columns:
            out[c] = 0.0
    return out[need]


def _pick_totals(res: Dict[str, object], method: str) -> pd.Series:
    cols = ["Total_AR", "Total_SR", "ER", "TR"] + (["Total_IR"] if method == "BHB" else [])
    return pd.Series({c: float(res.get(c, 0.0)) for c in cols})


def _stack_industry(fund_code: str, res: Dict[str, object], Wj: float, method: str) -> pd.DataFrame:
    ind = _ensure_industry_columns(res["industry_table"], method).copy()
    ind["AR"] = Wj * ind["AR_k"]
    ind["SR"] = Wj * ind["SR_k"]
    if method == "BHB":
        ind["IR"] = Wj * ind["IR_k"]
    ind.insert(0, "fund_code", fund_code)
    return ind[["fund_code", "industry"] + (["AR", "SR", "IR"] if method == "BHB" else ["AR", "SR"])]


def aggregate_fof_from_summary_df(
        fof_hold: Dict[str, float],
        summary_df: pd.DataFrame,
        method: Literal["BHB", "BF"] = "BHB",
        er_actual: Optional[float] = None,
) -> pd.DataFrame:
    """
    直接用“长表 df + FOF 权重”做 FOF 归因聚合。
    返回：一个 DataFrame, 包含 index_code, index_value
    """
    # 1) 还原每只基金的 results 结构
    res_map = build_fund_results_map_from_summary(summary_df, method=method)

    # 2) 归一化 FOF 权重，只用有结果的基金
    W_raw = pd.Series(fof_hold, name="W_raw").astype(float)
    used = W_raw.index.intersection(res_map.keys())
    if len(used) == 0:
        raise ValueError("fof_hold 与 summary_df 无交集。")
    W = _normalize_series_weight(W_raw.loc[used]).rename("W_norm")

    # 3) 聚合总体因子
    rows = []
    for j, wj in W.items():
        s = _pick_totals(res_map[j], method) * wj
        s.name = j
        rows.append(s)
    sums = pd.DataFrame(rows).sum()

    ER = float(er_actual) if er_actual is not None else float(sums["ER"])
    if method == "BHB":
        AR, SR, IR = float(sums["Total_AR"]), float(sums["Total_SR"]), float(sums["Total_IR"])
        TR = ER - (AR + SR + IR)
        totals = {"Total_AR": AR, "Total_SR": SR, "Total_IR": IR, "ER": ER, "TR": TR}
    else:
        AR, SR = float(sums["Total_AR"]), float(sums["Total_SR"])
        TR = ER - (AR + SR)
        totals = {"Total_AR": AR, "Total_SR": SR, "ER": ER, "TR": TR}

    # 4) 聚合行业因子
    blocks = []
    for j, wj in W.items():
        blk = _stack_industry(j, res_map[j], wj, method)
        blocks.append(blk)
    ind_all = pd.concat(blocks, ignore_index=True)
    agg = {"AR": "sum", "SR": "sum"}
    if method == "BHB":
        agg["IR"] = "sum"
    ind_table = ind_all.groupby("industry", as_index=False).agg(agg)

    # 5) 格式化输出 —— 合并总体和行业
    out_rows = []
    for k, v in totals.items():
        out_rows.append({"index_code": k, "index_value": v})

    for _, r in ind_table.iterrows():
        out_rows.append({"index_code": f"{r['industry']}_AR", "index_value": r["AR"]})
        out_rows.append({"index_code": f"{r['industry']}_SR", "index_value": r["SR"]})
        if method == "BHB":
            out_rows.append({"index_code": f"{r['industry']}_IR", "index_value": r["IR"]})

    result_df = pd.DataFrame(out_rows)
    return result_df


if __name__ == '__main__':
    # FOF持有基金的权重 (示例数据)
    fof_hold = {
        '000082': 0.18,
        '000309': 0.22,
        '000628': 0.11,
        '000729': 0.12,
    }

    # 1) BHB 聚合
    summary_df = pd.read_parquet("output/bhb_summary.parquet")  # BHB 口径
    result_bhb = aggregate_fof_from_summary_df(
        fof_hold=fof_hold,
        summary_df=summary_df,
        method="BHB",
        er_actual=None  # 若有 FOF 实际超额收益，可传入一个浮点数
    )
    print(result_bhb)

    # 2) BF 聚合
    summary_df = pd.read_parquet("output/bf_summary.parquet")  # BHB 口径
    result_bhb = aggregate_fof_from_summary_df(
        fof_hold=fof_hold,
        summary_df=summary_df,
        method="BF",
        er_actual=None  # 若有 FOF 实际超额收益，可传入一个浮点数
    )
    print(result_bhb)
