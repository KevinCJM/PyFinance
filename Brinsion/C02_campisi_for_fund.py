# -*- encoding: utf-8 -*-
"""
@File: C02_campisi_for_fund.py
@Modify Time: 2025/9/8 15:29       
@Author: Kevin-Chen
@Descriptions: Campisi归因, 计算单个公募基金的因子
"""

import numpy as np
import pandas as pd


# 线性插值
def _build_interp(yield_df, date_str):
    sub = yield_df[yield_df["date"].astype(str) == str(date_str)].copy()
    # 兼容 '3月'/'5年' 这类字符串
    if sub["duration"].dtype == object:
        def _to_year(x):
            s = str(x)
            if "月" in s:
                return float(s.replace("月", "")) / 12.0
            if "年" in s:
                return float(s.replace("年", ""))
            return float(s)

        sub["duration"] = sub["duration"].map(_to_year)
    sub = sub.sort_values("duration")

    x = sub["duration"].astype(float).to_numpy()
    y = sub["yield"].astype(float).to_numpy()
    # 若是百分数口径（如 1.68 表示 1.68%），转成小数
    if y.max() > 1.0:
        y = y / 100.0

    def interp_vec(maturities):
        # maturities 为 ndarray（单位：年）
        m = np.asarray(maturities, dtype=float)
        m = np.clip(m, x.min(), x.max())
        return np.interp(m, x, y)

    return interp_vec


def campisi_for_fund(fund_info_df, start_p_data_df, in_p_data_df, yield_df, start, end, target_funds):
    # 1) ===== 仅保留债券型 & 目标基金 =====
    mask_bond = fund_info_df["fund_type"] == "债券型"
    mask_funds = fund_info_df["fund_code"].isin(target_funds)
    bond_funds = fund_info_df.loc[mask_bond & mask_funds, ["fund_code", "fund_name"]]

    # 2) ===== 期初市值/久期（报告日 = start） =====
    m_start_date = start_p_data_df["date"].astype(str) == str(start)
    m_start_code = start_p_data_df["fund_code"].isin(bond_funds["fund_code"])
    start_side = start_p_data_df.loc[m_start_date & m_start_code, ["fund_code", "start_bond_mv", "start_fund_duration"]]
    start_side = start_side.rename(columns={"start_bond_mv": "MV_start", "start_fund_duration": "D_start"})

    # 3) ===== 区间内收益（报告日 = end） =====
    m_end_date = in_p_data_df["date"].astype(str) == str(end)
    m_end_code = in_p_data_df["fund_code"].isin(bond_funds["fund_code"])
    in_period = in_p_data_df.loc[
        m_end_date & m_end_code, ["fund_code", "interest_income", "bond_invest_income", "fair_value_change"]]

    # 4) ===== 合并基金面板 =====
    fund_panel = (bond_funds
                  .merge(start_side, on="fund_code", how="inner")
                  .merge(in_period, on="fund_code", how="inner"))
    if fund_panel.empty:
        raise ValueError("没有匹配到可计算的债券基金，请检查 fund_code / fund_type / 报告日期。")

    # 5) ===== 构造两日的国债插值（线性插值） =====
    interp_start = _build_interp(yield_df, start)
    interp_end = _build_interp(yield_df, end)
    # 在两条曲线上取 期初久期 D_start 对应的利率
    y_start = interp_start(fund_panel["D_start"].to_numpy())
    y_end = interp_end(fund_panel["D_start"].to_numpy())
    # 计算 delta_y
    dy = y_end - y_start

    # 6) ===== 计算各分项收益率（全用 /MV_start 的收益率口径） =====
    MV = fund_panel["MV_start"].astype(float).to_numpy()
    D0 = fund_panel["D_start"].astype(float).to_numpy()

    R_coupon = fund_panel["interest_income"].astype(float).to_numpy() / MV
    R_capital = (fund_panel["bond_invest_income"].astype(float).to_numpy()
                 + fund_panel["fair_value_change"].astype(float).to_numpy()) / MV
    R_duration = - D0 * dy
    R_spread = R_capital - R_duration
    R_total = R_coupon + R_capital

    # 7) ===== 组装DF结果 =====
    result_long = pd.DataFrame({
        "fund_code": np.repeat(fund_panel["fund_code"].to_numpy(), 5),
        "date": end,
        "index_code":
            np.tile(["total_return", "coupon_return", "capital_return", "duration_return", "spread_return"],
                    len(fund_panel)),
        "index_value": np.hstack([R_total, R_coupon, R_capital, R_duration, R_spread])
    })

    # 按 fund_code, index_code 排序
    result_long = result_long.sort_values(["fund_code", "index_code"]).reset_index(drop=True)
    return result_long


if __name__ == '__main__':
    # ===== 参数 =====
    start_date = "20241231"
    end_date = "20250630"
    target_fund_list = ['016295', '217025', '012240', '018959']  # 只计算这4只

    # ===== 读数据 =====
    fund_info = pd.read_parquet("./data/fund_info.parquet")
    in_p_data = pd.read_parquet("./data/in_p_data_campisi.parquet")
    start_p_data = pd.read_parquet("./data/start_p_data_campisi.parquet")
    t_bound_yield = pd.read_parquet("./data/yield_curve_campisi.parquet")

    # ==== 计算 =====
    result = campisi_for_fund(fund_info, start_p_data, in_p_data,
                              t_bound_yield, start_date, end_date, target_fund_list)
    print(result)
