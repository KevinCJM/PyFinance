# -*- encoding: utf-8 -*-
"""
@File: A05_find_ef_by_weights.py
@Modify Time: 2025/9/22 16:06       
@Author: Kevin-Chen
@Descriptions: 基于400万个1%精度的权重点，寻找在约束下的有效前沿
"""

import traceback
import numpy as np
import pandas as pd

if __name__ == '__main__':
    input_dict = {
        "asset_list": ['货币', '固收', '混合', '权益', '另类'],  # 大类资产列表, 必须要和 iis_mdl_aset_pct_d 表的 aset_bclass_nm 一致
        "C3": {
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
