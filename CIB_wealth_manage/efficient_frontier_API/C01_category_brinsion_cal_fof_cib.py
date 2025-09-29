# -*- encoding: utf-8 -*-
"""
@File: C01_category_brinsion_cal_fof_cib.py
@Modify Time: 2025/9/8 13:39       
@Author: Kevin-Chen
@Descriptions: 大类Brinsion归因, 计算FOF组合的因子
"""
import json
import pandas as pd
import numpy as np
from typing import Literal, Dict


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
        fof_holding_df: pd.DataFrame,  # ['fund_code','weight']
        fund_info_df: pd.DataFrame,  # ['fund_code','fund_type']
        fund_return_daily: pd.DataFrame,  # ['fund_code','date','adj_nav']
        benchmark_holding_df: pd.DataFrame,  # ['index_code','weight']
        index_type_df: pd.DataFrame,  # ['index_code','index_type']
        index_daily: pd.DataFrame,  # ['index_code','date','close']
        start: str, end: str,  # 区间首尾端点
        method: Literal["BHB", "BF"] = "BF",
        fof_code_label: str = "FOF"
) -> pd.DataFrame:

    ''' ---- 1) 计算区间收益 ---- '''
    fund_R = _period_return_from_series(
        df=fund_return_daily, code_col="fund_code", date_col="date", px_col="adj_nav", start=start, end=end
    )
    index_R = _period_return_from_series(
        df=index_daily, code_col="index_code", date_col="date", px_col="close", start=start, end=end
    )

    ''' ---- 2) FOF 侧：子基金 → 大类聚合 ---- '''
    fof = fof_holding_df[["fund_code", "weight"]].copy()
    fof["weight"] = fof["weight"].astype(float)
    w_sum = fof["weight"].sum()
    if w_sum <= 0: raise ValueError("FOF 权重和<=0")
    fof["W_pj"] = fof["weight"] / w_sum

    info = fund_info_df[["fund_code", "fund_type"]].drop_duplicates()
    fof = fof.merge(info, on="fund_code", how="left").merge(fund_R, on="fund_code", how="left")
    fof = fof.dropna(subset=["fund_type", "R", "W_pj"]).copy()
    w_sum = fof["W_pj"].sum()
    if w_sum <= 0: raise ValueError("FOF 有效子基金权重和<=0")
    fof["W_pj"] = fof["W_pj"] / w_sum

    grp = fof.groupby("fund_type", as_index=False)["W_pj"].sum().rename(columns={"W_pj": "W_p_c"})
    tmp = fof.assign(wr=lambda x: x.W_pj * x.R).groupby("fund_type", as_index=False)["wr"].sum()
    grp = grp.merge(tmp, on="fund_type", how="left")
    grp["R_p_c"] = np.where(grp["W_p_c"] > 0, grp["wr"] / grp["W_p_c"], 0.0)
    grp = grp[["fund_type", "W_p_c", "R_p_c"]]

    ''' ---- 3) 基准侧：指数 → 大类聚合 ---- '''
    bm = benchmark_holding_df[["index_code", "weight"]].copy()
    bm["weight"] = bm["weight"].astype(float)
    w_sum_b = bm["weight"].sum()
    if w_sum_b <= 0: raise ValueError("基准权重和<=0")
    bm["W_bi"] = bm["weight"] / w_sum_b
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
    df = C.merge(grp, on="fund_type", how="left").merge(bm_grp, on="fund_type", how="left").fillna(0.0)

    ''' ---- 5) 整体收益 & 超额收益 ---- '''
    R_p_FOF = (df["W_p_c"] * df["R_p_c"]).sum()
    R_b_FOF = (df["W_b_c"] * df["R_b_c"]).sum()
    ER = R_p_FOF - R_b_FOF

    ''' ---- 6) 归因分解 ---- '''
    if method == "BHB":
        df["AR_c"] = (df["W_p_c"] - df["W_b_c"]) * df["R_b_c"]
        df["SR_c"] = df["W_b_c"] * (df["R_p_c"] - df["R_b_c"])
        df["IR_c"] = (df["W_p_c"] - df["W_b_c"]) * (df["R_p_c"] - df["R_b_c"])
        Total_AR, Total_SR, Total_IR = df["AR_c"].sum(), df["SR_c"].sum(), df["IR_c"].sum()
        TR = ER - (Total_AR + Total_SR + Total_IR)
        rows = [
            {"fund_code": fof_code_label, "index_code": "Total_AR", "index_value": Total_AR},
            {"fund_code": fof_code_label, "index_code": "Total_SR", "index_value": Total_SR},
            {"fund_code": fof_code_label, "index_code": "Total_IR", "index_value": Total_IR},
            {"fund_code": fof_code_label, "index_code": "ER", "index_value": ER},
            {"fund_code": fof_code_label, "index_code": "TR", "index_value": TR},
        ]
        for _, r in df.iterrows():
            rows.extend([
                {"fund_code": fof_code_label, "index_code": f"{r['fund_type']}_AR", "index_value": r["AR_c"]},
                {"fund_code": fof_code_label, "index_code": f"{r['fund_type']}_SR", "index_value": r["SR_c"]},
                {"fund_code": fof_code_label, "index_code": f"{r['fund_type']}_IR", "index_value": r["IR_c"]}
            ])
    elif method == "BF":
        df["AR_c"] = (df["W_p_c"] - df["W_b_c"]) * (df["R_b_c"] - R_b_FOF)
        df["SR_c"] = df["W_p_c"] * (df["R_p_c"] - df["R_b_c"])
        Total_AR, Total_SR = df["AR_c"].sum(), df["SR_c"].sum()
        TR = ER - (Total_AR + Total_SR)
        rows = [
            {"fund_code": fof_code_label, "index_code": "Total_AR", "index_value": Total_AR},
            {"fund_code": fof_code_label, "index_code": "Total_SR", "index_value": Total_SR},
            {"fund_code": fof_code_label, "index_code": "ER", "index_value": ER},
            {"fund_code": fof_code_label, "index_code": "TR", "index_value": TR},
        ]
        for _, r in df.iterrows():
            rows.extend([
                {"fund_code": fof_code_label, "index_code": f"{r['fund_type']}_AR", "index_value": r["AR_c"]},
                {"fund_code": fof_code_label, "index_code": f"{r['fund_type']}_SR", "index_value": r["SR_c"]}
            ])
    else:
        raise ValueError("method 仅支持 'BHB' 或 'BF'")

    return pd.DataFrame(rows).sort_values("index_code").reset_index(drop=True)


def main(json_input: str) -> str:
    """
    接收一个包含所有输入数据的JSON字符串，执行Brinsion归因分析，并返回JSON结果。
    """
    try:
        # 1. 解析JSON
        input_data = json.loads(json_input)

        # 2. 从层级式JSON结构重建DataFrames
        fof_holding_df = pd.DataFrame(input_data['fof_holding'].items(), columns=['fund_code', 'weight'])
        fund_info_df = pd.DataFrame(input_data['fund_info'].items(), columns=['fund_code', 'fund_type'])
        benchmark_holding_df = pd.DataFrame(input_data['benchmark_holding'].items(), columns=['index_code', 'weight'])
        index_type_df = pd.DataFrame(input_data['index_type'].items(), columns=['index_code', 'index_type'])

        def unnest_time_series(ts_dict, code_col, val_col):
            records = []
            for code, date_val_map in ts_dict.items():
                for date, value in date_val_map.items():
                    records.append({code_col: code, 'date': date, val_col: value})
            return pd.DataFrame(records)

        fund_return_daily = unnest_time_series(input_data['fund_return_daily'], 'fund_code', 'adj_nav')
        index_daily = unnest_time_series(input_data['index_daily'], 'index_code', 'close')

        # 确保日期列是datetime类型
        fund_return_daily['date'] = pd.to_datetime(fund_return_daily['date'])
        index_daily['date'] = pd.to_datetime(index_daily['date'])

        # 3. 获取其他参数
        start = input_data['start_date']
        end = input_data['end_date']
        method = input_data.get('method', 'BF')
        fof_code_label = input_data.get('fof_code_label', 'FOF')

        # 4. 调用核心计算函数
        result_df = calc_asset_class_brinson_fof(
            fof_holding_df=fof_holding_df,
            fund_info_df=fund_info_df,
            fund_return_daily=fund_return_daily,
            benchmark_holding_df=benchmark_holding_df,
            index_type_df=index_type_df,
            index_daily=index_daily,
            start=start,
            end=end,
            method=method,
            fof_code_label=fof_code_label
        )

        # 5. 格式化并返回结果
        result_data = result_df.to_dict(orient='records')
        response = {"code": 0, "msg": "Success", "data": result_data}
        return json.dumps(response, ensure_ascii=False, indent=2)

    except Exception as e:
        # 6. 统一错误处理
        response = {"code": 1, "msg": f"Error: {type(e).__name__} - {str(e)}"}
        return json.dumps(response, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # 1. 从文件加载数据 (仅用于演示和测试)
    fund_info_df = pd.read_parquet('./data/fund_info.parquet')
    fof_holding_df = pd.read_parquet('./data/fof_holding.parquet')
    benchmark_holding_df = pd.read_parquet('./data/benchmark_holding.parquet')
    index_type_df = pd.read_parquet('./data/csi_index_type.parquet')
    fund_return_daily = pd.read_parquet('./data/fund_daily_return.parquet')
    index_daily = pd.read_parquet('./data/index_daily_all.parquet')

    # 2. 为JSON序列化准备数据
    fund_return_daily['date'] = fund_return_daily['date'].dt.strftime('%Y%m%d')
    index_daily['date'] = index_daily['date'].dt.strftime('%Y%m%d')

    def df_to_nested_dict(df, code_col, date_col, val_col):
        nested_dict = {}
        for code, group in df.groupby(code_col):
            # 确保在转换为字典前处理NaN
            clean_series = group.set_index(date_col)[val_col].replace({np.nan: None})
            nested_dict[code] = clean_series.to_dict()
        return nested_dict

    # 3. 构建JSON输入负载 (处理NaN为None)
    required_funds = fof_holding_df['fund_code'].unique().tolist()
    filtered_fund_info_df = fund_info_df[fund_info_df['fund_code'].isin(required_funds)]

    json_payload = {
        "fof_holding": fof_holding_df.replace({np.nan: None}).set_index('fund_code')['weight'].to_dict(),
        "fund_info": filtered_fund_info_df.replace({np.nan: None}).set_index('fund_code')['fund_type'].to_dict(),
        "benchmark_holding": benchmark_holding_df.replace({np.nan: None}).set_index('index_code')['weight'].to_dict(),
        "index_type": index_type_df.replace({np.nan: None}).set_index('index_code')['index_type'].to_dict(),
        "fund_return_daily": df_to_nested_dict(fund_return_daily, 'fund_code', 'date', 'adj_nav'),
        "index_daily": df_to_nested_dict(index_daily, 'index_code', 'date', 'close'),
        "start_date": "2025-08-04",
        "end_date": "2025-09-04",
        "method": "BF",
        "fof_code_label": "FOF_JSON_TEST"
    }

    json_input_str = json.dumps(json_payload, ensure_ascii=False)
    print(json_input_str)

    # 4. 调用新的main函数并打印结果
    print("--- Calling main function with JSON input ---")
    result_json = main(json_input_str)
    with open('sample_C01_output.json', 'w', encoding='utf-8') as f:
        f.write(result_json)
    print(result_json)
