# -*- encoding: utf-8 -*-
"""
@File: Y02_asset_id_map.py.py
@Modify Time: 2025/9/20 21:02       
@Author: Kevin-Chen
@Descriptions: 
"""

# iis_mdl_aset_pct_d 的 aset_bclass_cd 字段与 iis_aset_allc_indx_wght 表的什么字段对应
asset_to_weight_column_map = {
    '货币': 'csh_mgt_typ_pos',  # 现金管理类持仓
    '固收': 'fx_yld_pos',  # 固定持仓
    '混合': 'mix_strg_typ_pos',  # 混合持仓
    '权益': 'eqty_invst_pos',  # 权益混合持仓
    '另类': 'altrnt_invst_pos',  # 另类投资持仓
}

# C1 ~ C6 的 rsk_level 代码
rsk_level_code_dict = {
    'C1': 1,
    'C2': 2,
    'C3': 3,
    'C4': 4,
    'C5': 5,
    'C6': 6,
}
