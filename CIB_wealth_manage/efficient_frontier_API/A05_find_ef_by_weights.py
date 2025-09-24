# -*- encoding: utf-8 -*-
"""
@File: A05_find_ef_by_weights.py
@Modify Time: 2025/9/22 16:06       
@Author: Kevin-Chen
@Descriptions: 基于400万个1%精度的权重点，寻找在约束下的有效前沿
"""
import os
import json
import traceback
import numpy as np
import pandas as pd

# 复用现有工具函数，避免冗余
from efficient_frontier_API.T02_other_tools import log
from efficient_frontier_API.T04_show_plt import plot_efficient_frontier
from efficient_frontier_API.T01_generate_random_weights import cal_ef_mask

# df展示1000列
pd.set_option('display.max_columns', 1000)
# df展示不换行
pd.set_option('expand_frame_repr', False)
# 绘图开关
draw_plt = True


def draw_plt_func(a_list, weight_df, ef_points, the_folder_path):
    try:
        # 准备 hover 文本（原始 VaR 为负值，展示时取负）
        def _format_hover_text(row, title):
            text = f"<b>{title}</b><br>年化收益: {row['ret_annual']:.4f}<br>年化风险(var): {(-row['var_annual']):.4f}"
            text += "<br><br><b>--权重--</b><br>"
            for asset in a_list:
                if asset in row and pd.notna(row[asset]):
                    text += f"{asset}: {row[asset]:.2%}<br>"
            return text.strip('<br>')

        # 为了渲染性能，对散点做抽样
        scatter = weight_df.copy()
        max_points = 50000
        if len(scatter) > max_points:
            scatter = scatter.sample(n=max_points, random_state=123)
        scatter = scatter.copy()
        scatter["hover_text"] = scatter.apply(lambda r: _format_hover_text(r, "候选点"), axis=1)
        # 画图使用正向风险：var_plot = -var_annual
        scatter["var_plot"] = -scatter["var_annual"]

        ef_plot = ef_points.copy()
        ef_plot["hover_text"] = ef_plot.apply(lambda r: _format_hover_text(r, "有效前沿"), axis=1)
        ef_plot["var_plot"] = -ef_plot["var_annual"]

        scatter_points_data = [
            {"data": scatter, "name": "候选点", "color": "rgba(0,0,0,0.5)", "size": 3, "opacity": 0.4},
            {"data": ef_plot, "name": "有效前沿", "color": "red", "size": 6, "opacity": 1.0},
        ]

        out_html = os.path.join(the_folder_path, "efficient_frontier_filtered.html")
        plot_efficient_frontier(
            scatter_points_data,
            title="有效前沿 (基于权重点)",
            x_axis_title="年化风险 (Var)",
            y_axis_title="年化收益",
            x_col="var_plot",
            y_col="ret_annual",
            hover_text_col="hover_text",
            output_filename=out_html,
        )
        log(f"已输出前沿可视化: {out_html}")
    except Exception as e:
        log(f"绘图失败: {e}")
        log(traceback.format_exc())


def main(json_input) -> str:
    # json转字典
    try:
        try:
            input_data = json.loads(json_input)['in_data']
        except Exception as e:
            input_data = json_input['in_data']
        log(input_data)
        # 提取资产列表和约束条件
        asset_list = input_data["asset_list"]
        weight_range = input_data["weight_range"]
        # 读取本地400w个权重点的 pickle 文件
        folder_path = os.path.dirname(os.path.abspath("__file__"))
        alloc_results = pd.read_pickle(os.path.join(folder_path, "alloc_results_400w.pkl"))
        log(f"\n{alloc_results.head()}")
    except Exception as e:
        return json.dumps({
            "code": 1,
            "msg": f"json 解析失败: {e}"
        }, ensure_ascii=False)

    # ========== 1) 剔除不符合约束条件的权重点 ==========
    try:
        key_cols = asset_list + ["ret_annual", "var_annual"]
        alloc_results = alloc_results.dropna(subset=key_cols)  # 清理缺失值（权重或绩效为空则丢弃）

        # 按权重上下限过滤（闭区间）
        mask = np.ones(len(alloc_results), dtype=bool)
        for a in asset_list:
            lo, hi = weight_range.get(a, [0.0, 1.0])
            mask &= (alloc_results[a] >= float(lo)) & (alloc_results[a] <= float(hi))
        filtered = alloc_results.loc[mask].copy()
        log(f"过滤后样本数: {len(filtered):,} / 原始 {len(alloc_results):,}")

        # 若没有满足条件的点，直接退出
        if len(filtered) == 0:
            return json.dumps({
                "code": 1,
                "msg": f"没有找到满足条件的点, 所有的权重点都超出了约束条件"
            }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "code": 1,
            "msg": f"剔除不符合约束条件的权重点失败: {e}"
        }, ensure_ascii=False)

    # ========== 2) 从剩余的权重点中找到有效前沿 ==========
    try:
        cols_needed = asset_list + ["ret_annual", "var_annual"]
        df = filtered[cols_needed].copy()
        # 注意：原始 VaR 为负值（损失为负）。为了使用 cal_ef_mask（最小化“正向风险”），需取负号转为正值。
        risk_positive = (-df["var_annual"].to_numpy())
        mask_ef = cal_ef_mask(df["ret_annual"].to_numpy(), risk_positive)
        ef = df.loc[mask_ef].copy()
        log(f"有效前沿点数: {len(ef):,}")
        log(f"\n{ef.head()}")
    except Exception as e:
        return json.dumps({
            "code": 1,
            "msg": f"从剩余的权重点中找到有效前沿失败: {e}"
        }, ensure_ascii=False)

    # # ========== 3) 用 plotly 画出有效前沿 ==========
    # if draw_plt:
    #     draw_plt_func(asset_list, df, ef, folder_path)

    # ========== 4) 将有效前沿结果以 JSON 格式输出 ==========
    # 输出每个前沿点的 风险/收益/权重（按 asset_list 顺序）
    out_records = []
    for _, row in ef.iterrows():
        rec = {
            "var_annual": float(row["var_annual"]),
            "ret_annual": float(row["ret_annual"]),
            "weights": {a: float(row[a]) for a in asset_list},
        }
        out_records.append(rec)

    res = {
        "code": 0,
        "msg": "",
        "data": {
            "asset_list": asset_list,
            "count": len(out_records),
            "frontier": out_records,
        }
    }
    return json.dumps(res, ensure_ascii=False)


if __name__ == '__main__':
    input_dict = {
        "in_data": {
            "asset_list": ["MM", "FI", "MIX", "EQ", "ALT"],
            # 大类资产列表, 必须要和 iis_mdl_aset_pct_d 表的 aset_bclass_cd 一致
            "weight_range": {
                "MM": [  # 必须要和 iis_mdl_aset_pct_d 表的 aset_bclass_cd 一致
                    0.0,  # 代表权重下限%
                    1.0  # 代表权重上限%
                ],
                "FI": [0.0, 1.0],
                "MIX": [0.0, 0.5],
                "EQ": [0.0, 0.0],
                "ALT": [0.0, 0.0]
            }
        }}
    with open("sample_A05_input.json", "w") as f:
        f.write(json.dumps(input_dict, ensure_ascii=False, indent=4))

    output_json = main(input_dict)
    log(output_json)
