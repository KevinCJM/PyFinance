# -*- encoding: utf-8 -*-
"""
@File: A04_construct_category_yield.py
@Modify Time: 2025/9/18 15:15
@Author: Kevin-Chen
@Descriptions: 通过指数收益率数据与指定权重，拟合/构建大类资产的日收益序列。

输入 JSON 结构（示例）：
{
  "权益投资类": {
    "index_weight": [0.5, 0.5],
    "index_nv": [ {"20250101": 100, ...}, {"20250101": 99, ...} ]
  },
  "另类投资类": {
    "index_weight": [0.2, 0.8],
    "index_nv": [ ... ]
  }
}

返回：各大类按日期对齐的日收益序列（JSON）。
"""

import json
import time
import traceback
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from efficient_frontier_API.T02_other_tools import read_excel_compat


def _parse_series_from_dict(d: Dict[str, Any]) -> pd.Series:
    s = pd.Series(d)
    # 支持两种日期格式：YYYYMMDD 或 ISO 日期
    try:
        idx = pd.to_datetime(s.index, format="%Y%m%d")
    except Exception:
        idx = pd.to_datetime(s.index)
    s.index = idx
    s = s.astype(float).sort_index()
    return s


def _build_returns_from_nv_list(nv_list: List[Dict[str, Any]]) -> pd.DataFrame:
    # 将每个净值字典转为 Series 并按内连接对齐
    ser_list = [_parse_series_from_dict(d) for d in nv_list]
    df_nv = pd.concat(ser_list, axis=1, join="inner")
    df_nv.columns = list(range(len(ser_list)))
    df_ret = df_nv.pct_change().dropna()
    return df_ret


def analysis_json_and_read_data(json_input, excel_name=None, sheet_name=None):
    """修订：不再使用 load_returns_from_excel。支持两种来源：index_nv 或 index_code。

    - index_nv: 列表，每个元素是 {date->nav} 字典；按日期对齐后 pct_change 得到日收益。
    - index_code: 列表，需同时提供 excel_name 与 sheet_name；从 Excel 读取相应列的净值并转收益。

    输出：在原结构中添加 params_dict[k]["returns"]（DataFrame，行=日期，列=成分）。
    """
    params_dict = json.loads(json_input)
    for k, params in params_dict.items():
        index_code = params.get("index_code")
        index_nv = params.get("index_nv")

        if index_nv is None and index_code is None:
            raise ValueError("必须提供指数净值数据(index_nv)，或者提供指数代码(index_code)参数读取数据")

        if index_nv is not None:
            # 由净值推导成分日收益
            df_ret = _build_returns_from_nv_list(index_nv)
        else:
            # 按代码从 Excel 中取出净值并转收益
            if excel_name is None or sheet_name is None:
                raise ValueError("使用 index_code 时需提供 excel_name 与 sheet_name")
            if not isinstance(index_code, list) or not index_code:
                raise ValueError("index_code 必须是非空列表")
            df_all = read_excel_compat(excel_name, sheet_name)
            if 'date' not in df_all.columns:
                raise ValueError("Excel 中缺少 'date' 列")
            df_all = df_all.set_index('date')
            df_all.index = pd.to_datetime(df_all.index)
            df_all = df_all.sort_index().dropna(how='all')
            missing = [c for c in index_code if c not in df_all.columns]
            if missing:
                raise ValueError(f"Excel 中缺少列: {missing}")
            df_nv = df_all[index_code].copy()
            for col in df_nv.columns:
                df_nv[col] = pd.to_numeric(df_nv[col].astype(str).str.replace(",", "").str.strip(), errors='coerce')
            df_nv = df_nv.dropna(how='all')
            df_ret = df_nv.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how='any')

        params_dict[k]["returns"] = df_ret
    return params_dict


def parse_inputs_from_json(json_input: str, excel_name: Optional[str] = None, sheet_name: Optional[str] = None):
    """新增解析：更健壮地构造每个大类的成分日收益 DataFrame（_index_returns_df）。"""
    params_dict = json.loads(json_input)
    for cat, params in params_dict.items():
        index_nv = params.get("index_nv")
        index_returns = params.get("index_returns")
        index_code = params.get("index_code")

        if index_returns is not None:
            ser_list = [_parse_series_from_dict(d) for d in index_returns]
            df_ret = pd.concat(ser_list, axis=1, join="inner").astype(float).dropna()
        elif index_nv is not None:
            df_ret = _build_returns_from_nv_list(index_nv)
        elif index_code is not None:
            if excel_name is None or sheet_name is None:
                raise ValueError(f"[{cat}] 使用 index_code 时需提供 excel_name 与 sheet_name")
            if not isinstance(index_code, list) or not index_code:
                raise ValueError(f"[{cat}] index_code 必须是非空列表")
            df_all = read_excel_compat(excel_name, sheet_name)
            if 'date' not in df_all.columns:
                raise ValueError("Excel 中缺少 'date' 列")
            df_all = df_all.set_index("date")
            df_all.index = pd.to_datetime(df_all.index)
            df_all = df_all.sort_index().dropna(how="all")
            missing = [c for c in index_code if c not in df_all.columns]
            if missing:
                raise ValueError(f"[{cat}] Excel 中缺少列: {missing}")
            # 仅处理目标列，并进行字符串清理与数值化
            df_nv = df_all[index_code].copy()
            for col in df_nv.columns:
                df_nv[col] = pd.to_numeric(df_nv[col].astype(str).str.replace(",", "").str.strip(), errors='coerce')
            df_nv = df_nv.dropna(how='all')
            df_ret = df_nv.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how='any')
        else:
            raise ValueError(f"[{cat}] 必须提供 index_nv 或 index_returns 或 index_code 之一")

        params_dict[cat]["_index_returns_df"] = df_ret
    return params_dict


def construct_category_returns(params_dict: Dict[str, Any]) -> pd.DataFrame:
    """根据每个大类的指数日收益和权重，构建大类日收益序列，并在各大类之间按日期内连接对齐。"""
    cat_series = []
    cat_names = []
    for cat, params in params_dict.items():
        df_ret: pd.DataFrame = params["_index_returns_df"]
        w: np.ndarray = np.asarray(params.get("index_weight", []), dtype=float)
        if df_ret.shape[1] != w.size:
            raise ValueError(f"[{cat}] 权重长度({w.size})与指数数量({df_ret.shape[1]})不一致")
        # 线性加权（可选是否归一化，以输入为准）
        ser = pd.Series(df_ret.values @ w, index=df_ret.index, name=cat)
        cat_series.append(ser)
        cat_names.append(cat)

    # 按日期对齐（内连接）
    if not cat_series:
        return pd.DataFrame()
    cat_df = pd.concat(cat_series, axis=1, join="inner")
    cat_df.columns = cat_names
    return cat_df


def main(json_input: str, excel_name: Optional[str] = None, sheet_name: Optional[str] = None) -> str:
    """入口：解析输入、构建大类日收益，并返回 JSON。"""
    try:
        # 1) 解析入参与取数
        params_all = parse_inputs_from_json(json_input, excel_name, sheet_name)

        # 2) 构建大类收益
        cat_df = construct_category_returns(params_all)
        if cat_df.empty:
            return json.dumps({
                "success": False,
                "error_code": "EMPTY_RESULT",
                "message": "未能构建任何大类收益序列。"
            }, ensure_ascii=False)

        # 3) 序列化为 JSON（按大类输出，日期为 ISO 字符串）
        out: Dict[str, Any] = {"success": True, "category_returns": {}}
        for cat in cat_df.columns:
            ser = cat_df[cat]
            out["category_returns"][cat] = [
                {"date": d.strftime("%Y-%m-%d"), "return": float(v)} for d, v in ser.items()
            ]
        return json.dumps(out, ensure_ascii=False)

    except FileNotFoundError as e:
        print(traceback.format_exc())
        return json.dumps({
            "success": False,
            "error_code": "DATA_FILE_NOT_FOUND",
            "message": f"数据文件未找到: {getattr(e, 'filename', str(e))}"
        }, ensure_ascii=False)
    except ValueError as e:
        print(traceback.format_exc())
        return json.dumps({
            "success": False,
            "error_code": "INVALID_DATA_OR_CONFIG",
            "message": f"数据或配置无效: {e}"
        }, ensure_ascii=False)
    except Exception as e:
        print(traceback.format_exc())
        return json.dumps({
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": f"计算过程中发生未知错误: {type(e).__name__} - {e}"
        }, ensure_ascii=False)


if __name__ == '__main__':
    ''' 准备工作: 模拟json参数输入 ------------------------------------------------------------------------------ '''
    with open("sample_A04_input.json", "r", encoding="utf-8") as f:
        json_str = f.read()
    # 指数数据相关的 Excel 文件路径与工作表名
    excel_path = "万得指数数据.xlsx"
    sheet = "万得原始数据"

    ''' 计算并输出结果 ------------------------------------------------------------------------------------------ '''
    s_t = time.time()
    json_res = main(json_str, excel_path, sheet)
    print(json_res)
    print(f"总耗时: {time.time() - s_t:.2f} 秒")
