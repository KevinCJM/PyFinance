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

# df展示1000列
pd.set_option('display.max_columns', 1000)
# df展示不换行
pd.set_option('expand_frame_repr', False)

if __name__ == '__main__':
    input_dict = {
        "asset_list": ['货币', '固收', '混合', '权益', '另类'],  # 大类资产列表, 必须要和 iis_mdl_aset_pct_d 表的 aset_bclass_nm 一致
        "weight_range": {
            "货币": [  # 必须要和 iis_mdl_aset_pct_d 表的 aset_bclass_nm 一致
                0.0,  # 代表权重下限%
                1.0  # 代表权重上限%
            ],
            "固收": [0.0, 1.0],
            "混合": [0.0, 0.5],
            "权益": [0.0, 0.0],
            "另类": [0.0, 0.0]
        }
    }
    # 字典转json
    input_json = json.dumps(input_dict, ensure_ascii=False)
    print(input_json)

    # json转字典
    input_data = json.loads(input_json)
    print(input_data)
    # 提取资产列表和约束条件
    asset_list = input_data["asset_list"]
    weight_range = input_data["weight_range"]
    # 读取本地400w个权重点的parquet文件
    folder_path = os.path.dirname(os.path.abspath(__file__))
    alloc_results = pd.read_parquet(os.path.join(folder_path, "alloc_results_400w.parquet"))
    print(alloc_results.head())
    # 剔除不符合约束条件的权重点
    # pass
    # # 从剩余的权重点中找到有效前沿
    # pass
    # # 用plotly画出有效前沿
    # pass
    # # 将有效前沿结果以json格式print出来
    # pass
