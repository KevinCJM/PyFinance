# -*- encoding: utf-8 -*-
"""
@File: A04_construct_category_yield.py
@Modify Time: 2025/9/18 15:15       
@Author: Kevin-Chen
@Descriptions: 构建大类资产收益率
"""

import json
import pandas as pd
from T02_other_tools import load_returns_from_excel, log, ann_log_return, ann_log_vol


def analysis_json_and_read_data(json_input, excel_name=None, sheet_name=None):
    params_dict = json.loads(json_input)
    for k, params in params_dict.items():
        index_code = params.get("index_code", None)
        index_nv = params.get("index_nv", None)
        index_returns = None
        if index_nv is None and index_code is None:
            raise ValueError("必须提供指数净值数据，或者提供指数代码参数读取数据")
        if index_code is not None and index_nv is None and excel_name is None:
            raise ValueError("必须提供excel_name参数读取数据")
        if index_code is not None and index_nv is None and excel_name is not None:
            # 读取excel数据
            index_returns, _ = load_returns_from_excel(excel_name, sheet_name, asset_list)
        if index_nv is not None:
            # 二维列表, 反转, 转DataFrame
            index_nv = pd.DataFrame(index_nv).T
            index_nv.index = pd.to_datetime(index_nv.index, format='%Y%m%d')
            index_returns = index_nv.pct_change()
        params_dict[k]["returns"] = index_returns
    return params_dict


if __name__ == '__main__':
    dict_input = {
        "权益投资类": {
            "index_weight": [0.5, 0.5],
            "index_code": ['000300.SH', '000905.SH'],
            "index_nv": [
                {'20250101': 100, '20250102': 102, '20250103': 101, '20250104': 105},
                {'20250101': 99, '20250102': 98, '20250103': 97, '20250104': 96}
            ]
        },
        "另类投资类": {
            "index_weight": [0.2, 0.8],
            "index_code": ['000300.SH', '000905.SH'],
            "index_nv": [
                {'20250101': 100, '20250102': 102, '20250103': 101, '20250104': 105},
                {'20250101': 99, '20250102': 98, '20250103': 97, '20250104': 96}
            ]
        }
    }
    json_input = json.dumps(dict_input, ensure_ascii=False)

    # 1) 解析参数 & 读取数据 -------------------------------------------------------------------------------------
    ll = analysis_json_and_read_data(json_input, None, None)
    print(ll)
