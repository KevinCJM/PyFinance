# -*- encoding: utf-8 -*-
"""
@File: cal_fund_attribution.py
@Modify Time: 2025/9/5 11:00
@Author: Kevin-Chen
@Descriptions: 支持BHB和BF两种归因模型的基金业绩归因分析
"""

import os
import traceback
from typing import Literal
from datetime import datetime

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------ 工具函数 ------------

def _norm_code_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.zfill(6)


def _pick_industry_column(df: pd.DataFrame) -> str:
    candidates = ["行业", "所属行业", "行业名称", "行业板块", "申万行业", "申万行业名称", "东财行业", "所属东财行业",
                  "板块"]
    cols = [c for c in candidates if c in df.columns]
    return cols[0] if cols else "industry"


def _period_stock_return_from_close(stock_daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    sd = stock_daily.loc[(stock_daily["date"] >= start) & (stock_daily["date"] <= end), ["date", "stock_code", "close"]]
    if sd.empty:
        raise ValueError(f"股票无{start}~{end}的区间数据")

    idx_first = sd.groupby("stock_code")["date"].idxmin()
    idx_last = sd.groupby("stock_code")["date"].idxmax()
    start_px = sd.loc[idx_first, ["stock_code", "close"]].rename(columns={"close": "px_start"})
    end_px = sd.loc[idx_last, ["stock_code", "close"]].rename(columns={"close": "px_end"})
    px = start_px.merge(end_px, on="stock_code", how="inner")
    px = px[(px["px_start"] > 0) & (px["px_end"] > 0)]
    if px.empty:
        raise ValueError(f"股票无{start}~{end}的区间数据")

    px["R_i"] = px["px_end"] / px["px_start"] - 1.0
    return px[["stock_code", "R_i"]]


def _normalize_weights(df: pd.DataFrame, code_col: str, weight_col: str) -> pd.DataFrame:
    gsum = df.groupby(code_col)[weight_col].transform("sum")
    df = df.loc[gsum > 0].copy()
    df[weight_col] = df[weight_col] / gsum
    return df


def _get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ------------ 归因模型核心计算 ------------

def _calculate_attribution_inputs(
        fund_code: str, fund_hold: pd.DataFrame, index_hold: pd.DataFrame,
        stock_info: pd.DataFrame, stock_daily: pd.DataFrame,
        start_date: str, end_date: str
):
    """为所有归因模型准备通用的输入数据"""
    # 1. 基金持仓处理
    fh = fund_hold.loc[fund_hold["fund_code"] == fund_code].copy()
    if fh.empty:
        raise ValueError(f"[{fund_code}] 无基金持仓数据")
    fh["stock_code"] = _norm_code_series(fh["stock_code"])
    fh = fh.dropna(subset=["stock_code", "weight"]).copy()
    fund_rep_date = pd.to_datetime(fh["report_date"]).max()
    fh = fh.loc[pd.to_datetime(fh["report_date"]) == fund_rep_date].copy()
    fh = _normalize_weights(fh, code_col="fund_code", weight_col="weight")
    fh.rename(columns={"weight": "w_p_i"}, inplace=True)
    print(f"{_get_time()} [通用数据准备] {fund_code}基金持仓数据处理完成, 持有股票数: {len(fh)}")

    # 2. 指数成分处理
    ih = index_hold.copy()
    index_code_val = ih["index_code"].iloc[0] if "index_code" in ih.columns and not ih.empty else "UNKNOWN"
    index_rep_date = pd.to_datetime(ih["report_date"]).max()
    ih["stock_code"] = _norm_code_series(ih["stock_code"])
    ih = ih.loc[pd.to_datetime(ih["report_date"]) == index_rep_date].copy()
    ih = _normalize_weights(ih, code_col="index_code", weight_col="weight")
    ih.rename(columns={"weight": "w_b_i"}, inplace=True)
    print(f"{_get_time()} [通用数据准备] {fund_code}基准指数({index_code_val})成分数据处理完成, 成分股数: {len(ih)}")

    # 3. 行业映射与个股收益
    industry_col = _pick_industry_column(stock_info)
    si = stock_info[["stock_code", industry_col]].rename(columns={industry_col: "industry"})
    si["stock_code"] = _norm_code_series(si["stock_code"])
    print(f"{_get_time()} [通用数据准备] {fund_code}行业映射与个股收益数据处理完成, 行业数: {len(si)}")
    r = _period_stock_return_from_close(stock_daily, pd.to_datetime(start_date), pd.to_datetime(end_date))

    # 4. 数据合并与聚合
    pf = fh.merge(si, on="stock_code", how="left").merge(r, on="stock_code", how="left").dropna(subset=["R_i"])
    bm = ih.merge(si, on="stock_code", how="left").merge(r, on="stock_code", how="left").dropna(subset=["R_i"])
    pf["industry"] = pf["industry"].fillna("未知")
    bm["industry"] = bm["industry"].fillna("未知")
    print(f"{_get_time()} [通用数据准备] 将 持仓数据+行业映射+个股收益 数据进行合并")

    # 5. 计算行业权重与收益率
    w_p_k = pf.groupby("industry", as_index=False)["w_p_i"].sum().rename(columns={"w_p_i": "w_p_k"})
    w_b_k = bm.groupby("industry", as_index=False)["w_b_i"].sum().rename(columns={"w_b_i": "w_b_k"})
    num_p = pf.assign(num=lambda x: x.w_p_i * x.R_i).groupby("industry", as_index=False)["num"].sum()
    num_b = bm.assign(num=lambda x: x.w_b_i * x.R_i).groupby("industry", as_index=False)["num"].sum()
    print(f"{_get_time()} [通用数据准备] 完成基准与基金的'行业平均收益'")
    ind = pd.DataFrame({"industry": pd.Index(sorted(set(w_p_k["industry"]) | set(w_b_k["industry"])))})
    # 添加行业权重与行业平均收益
    ind = ind.merge(w_p_k, on="industry", how="left") \
        .merge(w_b_k, on="industry", how="left") \
        .merge(num_p, on="industry", how="left").rename(columns={"num": "num_p"}) \
        .merge(num_b, on="industry", how="left").rename(columns={"num": "num_b"})
    ind.fillna(0.0, inplace=True)
    # 计算行业的收益贡献
    ind["R_p_k"] = np.where(ind["w_p_k"] > 0, ind["num_p"] / ind["w_p_k"], 0.0)
    ind["R_b_k"] = np.where(ind["w_b_k"] > 0, ind["num_b"] / ind["w_b_k"], 0.0)
    print(f"{_get_time()} [通用数据准备] 完成基准与基金的'行业收益贡献'")

    # 6. 计算总体收益
    Rp_eq = float((pf["w_p_i"] * pf["R_i"]).sum())
    Rb_eq = float((bm["w_b_i"] * bm["R_i"]).sum())
    print(f"{_get_time()} [通用数据准备] 完成基准与基金的'总体收益率'计算: Rp_eq={Rp_eq:.4%}, Rb_eq={Rb_eq:.4%}")

    meta = {
        "stock_price_start_date": pd.to_datetime(start_date),
        "stock_price_end_date": pd.to_datetime(end_date),
        "fund_report_date": fund_rep_date,
        "index_report_date": index_rep_date,
        "index_code_val": index_code_val
    }
    print(f"{_get_time()} [通用数据准备] 通用数据准备完成")
    return ind[["industry", "w_p_k", "w_b_k", "R_p_k", "R_b_k"]], Rp_eq, Rb_eq, meta


def _calculate_bhb_attribution(ind_df: pd.DataFrame, Rp_eq: float, Rb_eq: float):
    """使用BHB模型计算归因"""
    ind_df["AR_k"] = (ind_df["w_p_k"] - ind_df["w_b_k"]) * ind_df["R_b_k"]
    ind_df["SR_k"] = ind_df["w_b_k"] * (ind_df["R_p_k"] - ind_df["R_b_k"])
    ind_df["IR_k"] = (ind_df["w_p_k"] - ind_df["w_b_k"]) * (ind_df["R_p_k"] - ind_df["R_b_k"])

    ER = Rp_eq - Rb_eq
    Total_AR = ind_df["AR_k"].sum()
    Total_SR = ind_df["SR_k"].sum()
    Total_IR = ind_df["IR_k"].sum()
    TR = ER - (Total_AR + Total_SR + Total_IR)
    print(f"{_get_time()} [BHB模型] 基金归因结果: ER={ER:.4%}, Total_AR={Total_AR:.4%}, "
          f"Total_SR={Total_SR:.4%}, Total_IR={Total_IR:.4%}, TR={TR:.4%}")
    return {
        "Rp_eq": Rp_eq, "Rb_eq": Rb_eq, "ER": ER,
        "Total_AR": Total_AR, "Total_SR": Total_SR, "Total_IR": Total_IR, "TR": TR,
        "industry_table": ind_df
    }


def _calculate_bf_attribution(ind_df: pd.DataFrame, Rp_eq: float, Rb_eq: float):
    """使用BF模型计算归因"""
    ind_df["AR_k"] = (ind_df["w_p_k"] - ind_df["w_b_k"]) * (ind_df["R_b_k"] - Rb_eq)
    ind_df["SR_k"] = ind_df["w_p_k"] * (ind_df["R_p_k"] - ind_df["R_b_k"])  # 核心区别
    ind_df["IR_k"] = 0.0  # BF模型无交互项

    ER = Rp_eq - Rb_eq
    Total_AR = ind_df["AR_k"].sum()
    Total_SR = ind_df["SR_k"].sum()
    Total_IR = 0.0
    TR = ER - (Total_AR + Total_SR)
    print(f"{_get_time()} [BF模型] 基金归因结果: ER={ER:.4%}, Total_AR={Total_AR:.4%}, "
          f"Total_SR={Total_SR:.4%}, TR={TR:.4%}")
    return {
        "Rp_eq": Rp_eq, "Rb_eq": Rb_eq, "ER": ER,
        "Total_AR": Total_AR, "Total_SR": Total_SR, "Total_IR": Total_IR, "TR": TR,
        "industry_table": ind_df
    }


def perform_attribution(
        fund_code: str, fund_hold: pd.DataFrame, index_hold: pd.DataFrame,
        stock_info: pd.DataFrame, stock_daily: pd.DataFrame,
        start_date: str, end_date: str,
        method: Literal["BHB", "BF"] = "BHB"
):
    """
    执行业绩归因分析的调度函数
    """
    # 准备通用的输入数据

    ind_df, Rp_eq, Rb_eq, meta = _calculate_attribution_inputs(
        fund_code,  # 基金代码，用于标识当前需要分析的基金产品
        fund_hold,  # 基金持仓数据，包含基金持有的所有股票及其权重信息
        index_hold,  # 指数持仓数据，用作基准比较的指数成分股及其权重信息
        stock_info,  # 股票基本信息数据，包含股票的行业分类等静态信息
        stock_daily,  # 股票日线行情数据，用于计算分析期间的股票收益率
        start_date,  # 分析开始日期，用于确定计算收益率的时间区间起点
        end_date  # 分析结束日期，用于确定计算收益率的时间区间终点
    )
    # 调用模型
    if method == "BHB":
        results = _calculate_bhb_attribution(ind_df, Rp_eq, Rb_eq)
    elif method == "BF":
        results = _calculate_bf_attribution(ind_df, Rp_eq, Rb_eq)
    else:
        raise ValueError(f"不支持的归因方法: {method}. 请选择 'BHB' 或 'BF'.")

    return results, meta


# ------------ 批量执行与结果格式化 ------------

def run_all_funds(start_date: str, end_date: str, method: Literal["BHB", "BF"] = "BHB"):
    # 读取数据
    try:
        fund_hold = pd.read_parquet(os.path.join(DATA_DIR, "fund_hold.parquet"))
        index_hold = pd.read_parquet(os.path.join(DATA_DIR, "index_hold.parquet"))
        stock_info = pd.read_parquet(os.path.join(DATA_DIR, "stock_info.parquet"))
        stock_daily = pd.read_parquet(os.path.join(DATA_DIR, "stock_daily.parquet"))
    except FileNotFoundError as e:
        print(f"数据文件缺失: {e}. 请先运行 get_data.py 脚本。")
        return

    # 统一类型
    fund_hold["fund_code"] = fund_hold["fund_code"].astype(str)
    fund_hold["report_date"] = pd.to_datetime(fund_hold["report_date"])
    index_hold["report_date"] = pd.to_datetime(index_hold["report_date"])
    stock_daily["date"] = pd.to_datetime(stock_daily["date"])

    fund_list = sorted(fund_hold["fund_code"].unique().tolist())
    final_rows = []
    print(f"--- 开始使用 {method} 模型进行归因分析 ---")
    for fc in fund_list:
        try:
            results, meta = perform_attribution(
                fc, fund_hold, index_hold, stock_info, stock_daily,
                start_date=start_date, end_date=end_date, method=method
            )
            base_info = {
                "fund_code": fc,
                "stock_price_start_date": meta["stock_price_start_date"],
                "stock_price_end_date": meta["stock_price_end_date"],
                "fund_report_date": meta["fund_report_date"],
                "index_report_date": meta["index_report_date"],
            }

            # 添加总体指标 (根据方法过滤IR)
            total_metrics = ["Total_AR", "Total_SR", "Total_IR", "ER", "TR", "Rp_eq", "Rb_eq"]
            if method == "BF":
                total_metrics.remove("Total_IR")

            for metric in total_metrics:
                row = {**base_info, "index_code": metric, "index_value": results[metric]}
                final_rows.append(row)

            # 添加分行业指标 (根据方法过滤IR)
            industry_df = results["industry_table"]
            for _, r in industry_df.iterrows():
                industry_name = r["industry"]
                final_rows.append({**base_info, "index_code": f"{industry_name}_AR", "index_value": r["AR_k"]})
                final_rows.append({**base_info, "index_code": f"{industry_name}_SR", "index_value": r["SR_k"]})
                # 只在BHB模式下添加IR行
                if method == "BHB":
                    final_rows.append({**base_info, "index_code": f"{industry_name}_IR", "index_value": r["IR_k"]})

            print(f"{_get_time()} ======================================== "
                  f"基金 {fc} 使用 {method} 方法计算完成 ========================================")
        except Exception as e:
            print(f"基金 {fc} 使用 {method} 方法计算失败: {e}")
            traceback.print_exc()

    if not final_rows:
        print(f"使用 {method} 模型没有成功计算任何基金。")
        return pd.DataFrame()

    summary_df = pd.DataFrame(final_rows)
    final_cols = ['fund_code', 'stock_price_start_date', 'stock_price_end_date',
                  'fund_report_date', 'index_report_date', 'index_code', 'index_value']
    summary_df = summary_df[final_cols]

    summary_path = os.path.join(OUTPUT_DIR, f"{method.lower()}_summary.parquet")
    summary_df.to_parquet(summary_path, index=False)
    print(f"[完成] 使用 {method} 模型分析了 {len(fund_list)} 只基金。汇总已保存：{summary_path}")
    return summary_df


if __name__ == "__main__":
    start = "2024-09-04"
    end = "2025-09-04"

    # 1. 使用 BHB 模型进行计算
    res_bhb = run_all_funds(start_date=start, end_date=end, method='BHB')
    if res_bhb is not None and not res_bhb.empty:
        print("\n--- BHB 模型结果预览 ---")
        print(res_bhb.head(20))

    print("\n" + "=" * 50 + "\n")

    # 2. 使用 BF 模型进行计算
    res_bf = run_all_funds(start_date=start, end_date=end, method='BF')
    if res_bf is not None and not res_bf.empty:
        print("\n--- BF 模型结果预览 ---")
        print(res_bf.head(20))
