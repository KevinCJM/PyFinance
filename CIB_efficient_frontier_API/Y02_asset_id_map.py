# -*- encoding: utf-8 -*-
"""
@File: Y02_asset_id_map.py.py
@Modify Time: 2025/9/20 21:02       
@Author: Kevin-Chen
@Descriptions: 
"""

# iis_mdl_aset_pct_d 的 aset_bclass_cd 字段与 iis_aset_allc_indx_wght 表的什么字段对应
asset_to_weight_column_map = {
    '另类': 'altrnt_invst_pos',  # 另类投资持仓
    '固收': 'fx_yld_pos',  # 固定收益类持仓
    '权益': 'eqty_invst_pos',  # 权益策略类持仓
    '混合': 'mix_strg_typ_pos',  # 混合策略类持仓
    '货币': 'csh_mgt_typ_pos'  # 现金管理类持仓
}
