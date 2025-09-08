# -*- encoding: utf-8 -*-
"""
@File: C02_campisi_for_fund.py
@Modify Time: 2025/9/8 15:29       
@Author: Kevin-Chen
@Descriptions: Campisi归因, 计算FOF组合的因子
"""

import numpy as np
import pandas as pd


# ===== 聚合函数 =====
def aggregate_fof_campisi(single_df: pd.DataFrame,
                          holdings_df: pd.DataFrame,
                          fund_info_df: pd.DataFrame,
                          end_date: str,
                          bond_only: bool = True,
                          reclose_total: bool = True) -> pd.DataFrame:
    """
    将单基金 Campisi 结果聚合到 FOF 组合层面（权重归一化在“债券型”内部进行）。

    Parameters
    ----------
    single_df : 长表，列包含 ['fund_code','date','index_code','index_value']
    holdings_df : 列包含 ['fund_code','weight']（原始权重，可能含非债基）
    fund_info_df : 列包含 ['fund_code','fund_type']，用于筛选 '债券型'
    end_date : str，与 single_df['date'] 匹配
    bond_only : bool，仅在“债券型”内部做归一化聚合
    reclose_total : bool，是否用 coupon+capital 重建 total，保证加总闭环

    Returns
    -------
    fof_long : 长表，列 ['date','index_code','index_value']
    """

    # 1) 过滤单基金到目标日期
    m_date = single_df["date"].astype(str) == str(end_date)
    sd = single_df.loc[m_date, ["fund_code", "index_code", "index_value"]].copy()

    if sd.empty:
        raise ValueError("single_df 在指定日期没有数据。")

    # 2) 只保留在持仓里的基金（避免无持仓的噪音）
    m_in_hold = sd["fund_code"].isin(holdings_df["fund_code"])
    sd = sd.loc[m_in_hold].copy()
    if sd.empty:
        raise ValueError("single_df 与 FOF 持仓无交集。")

    # 3) 若 bond_only=True，则先筛出债券型基金，再做权重归一化
    if bond_only:
        mask_bond = fund_info_df["fund_type"] == "债券型"
        bond_codes = set(fund_info_df.loc[mask_bond, "fund_code"])
        # 持仓里保留债基
        h = holdings_df.loc[holdings_df["fund_code"].isin(bond_codes), ["fund_code", "weight"]].copy()
    else:
        h = holdings_df[["fund_code", "weight"]].copy()

    # 只保留同时出现在 single_df 的基金
    h = h.loc[h["fund_code"].isin(sd["fund_code"])].copy()
    if h.empty:
        raise ValueError("FOF 持仓（债券型筛选后）与 single_df 无交集。")

    # 4) 权重归一化（在当前集合上）
    wsum = float(h["weight"].sum())
    if wsum <= 0:
        raise ValueError("FOF 债券部分权重和为 0。")
    h["w_norm"] = h["weight"] / wsum

    # 5) 合并权重到单基金 Campisi 结果
    sd = sd.merge(h[["fund_code", "w_norm"]], on="fund_code", how="inner")
    # 6) 加权汇总到 FOF 因子
    sd["weighted"] = sd["index_value"] * sd["w_norm"]
    fof = (sd.groupby("index_code", as_index=False)["weighted"].sum()
           .rename(columns={"weighted": "index_value"}))

    # 7) 可选：用 coupon+capital 重建 total，提高闭环稳定性
    if reclose_total:
        pivot = fof.set_index("index_code")["index_value"]
        total_rebuilt = float(pivot.get("coupon_return", 0.0) + pivot.get("capital_return", 0.0))
        # 若原本存在 total_return，则覆盖；否则追加
        if "total_return" in fof["index_code"].values:
            fof.loc[fof["index_code"] == "total_return", "index_value"] = total_rebuilt
        else:
            fof = pd.concat([fof,
                             pd.DataFrame([{"index_code": "total_return",
                                            "index_value": total_rebuilt}])],
                            ignore_index=True)

    # 8) 加上日期列并整理输出
    fof["date"] = str(end_date)
    fof = fof[["date", "index_code", "index_value"]].sort_values(["index_code"]).reset_index(drop=True)
    return fof


if __name__ == '__main__':
    # ===== 区间期末报告日期 =====
    end = "20250630"

    # ===== 读取 FOF 权重 & 基金信息 =====
    fof_holding_df = pd.read_parquet("./data/fof_holding_campisi.parquet")
    fund_info_df = pd.read_parquet("./data/fund_info.parquet")
    single_df = pd.read_parquet("./data/campisi_result_for_fund.parquet")

    # ======== 调用 ========
    fof_result = aggregate_fof_campisi(single_df, fof_holding_df, fund_info_df, end)
    print(fof_result)
    fof_result.to_parquet("./data/campisi_for_fof.parquet", index=False)
