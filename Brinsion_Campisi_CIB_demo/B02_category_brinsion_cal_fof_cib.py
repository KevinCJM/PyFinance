# -*- encoding: utf-8 -*-
"""
@File: B02_category_brinsion_cal_fof_cib.py
@Modify Time: 2025/9/8 13:39       
@Author: Kevin-Chen
@Descriptions: 大类Brinsion归因, 计算FOF组合的因子
"""
import pandas as pd
import numpy as np
from typing import Literal, Dict, Iterable


# ---------------------------
# 工具：区间收益（端点法）
# ---------------------------
def _period_return_from_series(
        df: pd.DataFrame,
        code_col: str,
        date_col: str,
        px_col: str,
        start: str,
        end: str
) -> pd.DataFrame:
    """
    区间收益计算：每只基金在[start,end]区间内的首个非空价格和最后一个非空价格
    支持避免 NaN 导致的丢失。
    """
    d = df.loc[(df[date_col] >= pd.to_datetime(start)) & (df[date_col] <= pd.to_datetime(end)),
    [date_col, code_col, px_col]].copy()

    if d.empty:
        raise ValueError(f"没有位于 {start}~{end} 的行情数据：{px_col}")

    # 转宽表：索引=日期, 列=基金, 值=价格
    wide = d.pivot_table(index=date_col, columns=code_col, values=px_col).sort_index()

    # 对每个基金，找到区间内第一条非空净值和最后一条非空净值
    px_start = wide.apply(lambda s: s.loc[s.first_valid_index()] if s.first_valid_index() is not None else np.nan)
    px_end = wide.apply(lambda s: s.loc[s.last_valid_index()] if s.last_valid_index() is not None else np.nan)

    # 过滤掉首尾为 NaN 或 <=0 的情况
    mask = (px_start > 0) & (px_end > 0)
    px_start = px_start[mask]
    px_end = px_end[mask]

    out = pd.DataFrame({
        code_col: px_start.index.astype(str),
        "R": px_end.values / px_start.values - 1.0
    })
    return out.reset_index(drop=True)


# ---------------------------
# 主函数：FOF 大类 Brinson（BHB / BF）
# ---------------------------
def calc_asset_class_brinson_fof(
        fof_holding_df: pd.DataFrame,  # ['fund_code','weight']  —— FOF持有的子基金及权重（可不等于1，函数会归一化）
        fund_info_df: pd.DataFrame,  # ['fund_code','fund_type'] —— 子基金所属大类（股票型/混合型/债券型/货币市场型/商品型...）
        fund_return_daily: pd.DataFrame,  # ['fund_code','date','adj_nav'] —— 子基金的日净值（复权净值）
        benchmark_holding_df: pd.DataFrame,  # ['index_code','weight'] —— 基准由若干“基金大类指数”构成的权重（建议权重和=1）
        index_type_df: pd.DataFrame,  # ['index_code','index_type'] —— 指数映射到基金大类（要与 fund_type 同一套命名）
        index_daily: pd.DataFrame,  # ['index_code','date','close'] —— 指数日行情
        start: str, end: str,  # 区间首尾端点，格式 'YYYYMMDD'
        method: Literal["BHB", "BF"] = "BF",  # 归因方法 （BHB 或 BF）
        fof_code_label: str = "FOF"  # 归因结果中，FOF的代码标签
) -> pd.DataFrame:
    """
    计算FOF基金的大类业绩归因分析 (股票,债券,货币,商品,混合)

    :param fof_holding_df: FOF基金持仓数据DataFrame，包含基金代码、持仓比例、持仓日期等信息
    :param fund_info_df: 基金信息数据DataFrame，包含基金基础信息如基金类型、成立日期等
    :param fund_return_daily: 基金日收益率数据DataFrame，包含基金代码、交易日期、日收益率等字段
    :param benchmark_holding_df: 基准持仓数据DataFrame，包含基准指数的成分股及权重信息
    :param index_type_df: 指数类型数据DataFrame，定义不同指数的分类信息
    :param index_daily: 指数日行情数据DataFrame，包含指数代码、交易日期、收盘价等信息
    :param start: 分析起始日期，格式为YYYY-MM-DD字符串
    :param end: 分析结束日期，格式为YYYY-MM-DD字符串
    :param method: 业绩归因方法，可选值包括'brinson'、'carino'等模型方法
    :param fof_code_label: FOF基金代码标识，用于筛选特定基金进行分析
    :return: 业绩归因分析结果，包含超额收益分解、风格贡献度等指标的DataFrame
    """

    ''' ---- 1) 计算区间收益 ---- '''
    # 子基金收益
    fund_R = _period_return_from_series(
        df=fund_return_daily, code_col="fund_code", date_col="date", px_col="adj_nav", start=start, end=end
    )
    # 指数收益
    index_R = _period_return_from_series(
        df=index_daily, code_col="index_code", date_col="date", px_col="close", start=start, end=end
    )

    ''' ---- 2) FOF 侧：子基金 → 大类聚合 ---- '''
    # 归一化 FOF 权重
    fof = fof_holding_df[["fund_code", "weight"]].copy()
    fof["weight"] = fof["weight"].astype(float)
    w_sum = fof["weight"].sum()
    if w_sum <= 0:
        raise ValueError("FOF 权重和<=0")
    fof["W_pj"] = fof["weight"] / w_sum

    # 贴上大类 + 收益
    info = fund_info_df[["fund_code", "fund_type"]].drop_duplicates()
    fof = fof.merge(info, on="fund_code", how="left").merge(fund_R, on="fund_code", how="left")
    # 少量基金可能缺收益或缺分类，剔除后归一化
    fof = fof.dropna(subset=["fund_type", "R", "W_pj"]).copy()
    w_sum = fof["W_pj"].sum()
    if w_sum <= 0:
        raise ValueError("FOF 有效子基金权重和<=0（可能都缺收益/分类）")
    fof["W_pj"] = fof["W_pj"] / w_sum

    # 大类权重与大类收益（权重平均）
    grp = fof.groupby("fund_type", as_index=False)["W_pj"].sum().rename(columns={"W_pj": "W_p_c"})
    tmp = fof.assign(wr=lambda x: x.W_pj * x.R).groupby("fund_type", as_index=False)["wr"].sum()
    grp = grp.merge(tmp, on="fund_type", how="left")
    grp["R_p_c"] = np.where(grp["W_p_c"] > 0, grp["wr"] / grp["W_p_c"], 0.0)
    grp = grp[["fund_type", "W_p_c", "R_p_c"]]

    ''' ---- 3) 基准侧：指数 → 大类聚合 ---- '''
    bm = benchmark_holding_df[["index_code", "weight"]].copy()
    bm["weight"] = bm["weight"].astype(float)
    w_sum_b = bm["weight"].sum()
    if w_sum_b <= 0:
        raise ValueError("基准权重和<=0")
    bm["W_bi"] = bm["weight"] / w_sum_b
    # 指数→大类映射 + 指数收益
    idxmap = index_type_df.rename(columns={"index_type": "fund_type"})[["index_code", "fund_type"]]
    bm = bm.merge(idxmap, on="index_code", how="left").merge(index_R, on="index_code", how="left")
    bm = bm.dropna(subset=["fund_type", "R", "W_bi"]).copy()

    W_b_c = bm.groupby("fund_type", as_index=False)["W_bi"].sum().rename(columns={"W_bi": "W_b_c"})
    num_b = bm.assign(num=lambda x: x.W_bi * x.R).groupby("fund_type", as_index=False)["num"].sum()
    bm_grp = W_b_c.merge(num_b, on="fund_type", how="left")
    bm_grp["R_b_c"] = np.where(bm_grp["W_b_c"] > 0, bm_grp["num"] / bm_grp["W_b_c"], 0.0)
    bm_grp = bm_grp[["fund_type", "W_b_c", "R_b_c"]]

    ''' ---- 4) 对齐并集大类 ---- '''
    C = pd.DataFrame({"fund_type": sorted(set(grp["fund_type"]) | set(bm_grp["fund_type"]))})
    df = (C.merge(grp, on="fund_type", how="left")
          .merge(bm_grp, on="fund_type", how="left")
          .fillna(0.0))

    ''' ---- 5) 整体收益 & 超额收益 ---- '''
    R_p_FOF = float((df["W_p_c"] * df["R_p_c"]).sum())
    R_b_FOF = float((df["W_b_c"] * df["R_b_c"]).sum())
    ER = R_p_FOF - R_b_FOF

    ''' ---- 6) 归因分解 ---- '''
    if method == "BHB":
        # 行项
        df["AR_c"] = (df["W_p_c"] - df["W_b_c"]) * df["R_b_c"]
        df["SR_c"] = df["W_b_c"] * (df["R_p_c"] - df["R_b_c"])
        df["IR_c"] = (df["W_p_c"] - df["W_b_c"]) * (df["R_p_c"] - df["R_b_c"])
        Total_AR = float(df["AR_c"].sum())
        Total_SR = float(df["SR_c"].sum())
        Total_IR = float(df["IR_c"].sum())
        TR = ER - (Total_AR + Total_SR + Total_IR)

        rows = [
            {"fund_code": fof_code_label, "index_code": "Total_AR", "index_value": Total_AR},
            {"fund_code": fof_code_label, "index_code": "Total_SR", "index_value": Total_SR},
            {"fund_code": fof_code_label, "index_code": "Total_IR", "index_value": Total_IR},
            {"fund_code": fof_code_label, "index_code": "ER", "index_value": ER},
            {"fund_code": fof_code_label, "index_code": "TR", "index_value": TR},
        ]
        for _, r in df.iterrows():
            rows.append({"fund_code": fof_code_label, "index_code": f"{r['fund_type']}_AR", "index_value": r["AR_c"]})
            rows.append({"fund_code": fof_code_label, "index_code": f"{r['fund_type']}_SR", "index_value": r["SR_c"]})
            rows.append({"fund_code": fof_code_label, "index_code": f"{r['fund_type']}_IR", "index_value": r["IR_c"]})

    elif method == "BF":
        # 注意：BF 配置项使用 (R_b_c - R_b^FOF)
        df["AR_c"] = (df["W_p_c"] - df["W_b_c"]) * (df["R_b_c"] - R_b_FOF)
        df["SR_c"] = df["W_p_c"] * (df["R_p_c"] - df["R_b_c"])
        Total_AR = float(df["AR_c"].sum())
        Total_SR = float(df["SR_c"].sum())
        TR = ER - (Total_AR + Total_SR)

        rows = [
            {"fund_code": fof_code_label, "index_code": "Total_AR", "index_value": Total_AR},
            {"fund_code": fof_code_label, "index_code": "Total_SR", "index_value": Total_SR},
            {"fund_code": fof_code_label, "index_code": "ER", "index_value": ER},
            {"fund_code": fof_code_label, "index_code": "TR", "index_value": TR},
        ]
        for _, r in df.iterrows():
            rows.append({"fund_code": fof_code_label, "index_code": f"{r['fund_type']}_AR", "index_value": r["AR_c"]})
            rows.append({"fund_code": fof_code_label, "index_code": f"{r['fund_type']}_SR", "index_value": r["SR_c"]})
    else:
        raise ValueError("method 仅支持 'BHB' 或 'BF'")

    ''' ---- 7) 闭合性自检 (可选) '''
    out = pd.DataFrame(rows)

    # 行业级别因子（排除掉 Total_xx）
    chk_AR = out.loc[
        out["index_code"].str.endswith("_AR") & ~out["index_code"].str.startswith("Total"), "index_value"].sum()
    chk_SR = out.loc[
        out["index_code"].str.endswith("_SR") & ~out["index_code"].str.startswith("Total"), "index_value"].sum()

    if method == "BHB":
        chk_IR = out.loc[
            out["index_code"].str.endswith("_IR") & ~out["index_code"].str.startswith("Total"), "index_value"].sum()
        tot_AR = float(out.loc[out["index_code"] == "Total_AR", "index_value"].sum())
        tot_SR = float(out.loc[out["index_code"] == "Total_SR", "index_value"].sum())
        tot_IR = float(out.loc[out["index_code"] == "Total_IR", "index_value"].sum())
        err = abs(chk_AR - tot_AR) + abs(chk_SR - tot_SR) + abs(chk_IR - tot_IR)
    else:
        tot_AR = float(out.loc[out["index_code"] == "Total_AR", "index_value"].sum())
        tot_SR = float(out.loc[out["index_code"] == "Total_SR", "index_value"].sum())
        err = abs(chk_AR - tot_AR) + abs(chk_SR - tot_SR)
    if err > 1e-10:
        # 不抛错，给个提醒（也可以改成 raise）
        print(f"[WARN] 大类归因闭合误差={err:.3e}")

    return out.sort_values("index_code").reset_index(drop=True)


if __name__ == '__main__':
    # 读取已保存的文件
    fund_info_df = pd.read_parquet('./data/fund_info.parquet')  # fund_code, fund_type
    fof_holding_df = pd.read_parquet('./data/fof_holding.parquet')  # fund_code, weight
    benchmark_holding_df = pd.read_parquet('./data/benchmark_holding.parquet')  # index_code, weight
    index_type_df = pd.read_parquet('./data/csi_index_type.parquet')  # index_code, index_type
    fund_return_daily = pd.read_parquet('./data/fund_daily_return.parquet')  # fund_code, date, adj_nav
    index_daily = pd.read_parquet('./data/index_daily_all.parquet')  # index_code, date, close

    # 约定区间
    start, end = "2024-09-04", "2025-09-04"

    # 计算 BF（与你图里“BF方法(FOF组合)”一致）
    res_bf = calc_asset_class_brinson_fof(
        fof_holding_df=fof_holding_df,
        fund_info_df=fund_info_df,
        fund_return_daily=fund_return_daily,
        benchmark_holding_df=benchmark_holding_df,
        index_type_df=index_type_df,
        index_daily=index_daily,
        start=start, end=end,
        method="BF",
        fof_code_label="FOF"
    )
    print("BF 结果：")
    print(res_bf.head(20))

    # 如需 BHB：
    res_bhb = calc_asset_class_brinson_fof(
        fof_holding_df=fof_holding_df,
        fund_info_df=fund_info_df,
        fund_return_daily=fund_return_daily,
        benchmark_holding_df=benchmark_holding_df,
        index_type_df=index_type_df,
        index_daily=index_daily,
        start=start, end=end,
        method="BHB",
        fof_code_label="FOF"
    )
    print("BHB 结果：")
    print(res_bhb.head(20))
