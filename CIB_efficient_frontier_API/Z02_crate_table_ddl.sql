CREATE TABLE iis_wght_cfg_attc_mdl
(
    mdl_ver_id       varchar(64) NOT NULL COMMENT '模型编号',
    mdl_nm           varchar(100)  DEFAULT NULL COMMENT '模型名称',
    aset_desc        varchar(4000) DEFAULT NULL COMMENT '大类资产说明',
    mdl_st           varchar(1)    DEFAULT NULL COMMENT '模型状态，1待审核；2已上线；3已下线',
    cal_strt_dt      date          DEFAULT NULL COMMENT '测算开始日期',
    cal_end_dt       date          DEFAULT NULL COMMENT '测算结束日期',
    on_ln_dt         date          DEFAULT NULL COMMENT '上线日期',
    off_ln_dt        date          DEFAULT NULL COMMENT '下线日期',
    addtn_psn_id     varchar(6)    DEFAULT NULL COMMENT '新增人员ID',
    addtn_psn_fll_nm varchar(60)   DEFAULT NULL COMMENT '新增人员姓名',
    addtn_tm         datetime      DEFAULT NULL COMMENT '新增日期时间',
    mod_psn_id       varchar(6)    DEFAULT NULL COMMENT '修改人员ID',
    mod_psn_fll_nm   varchar(60)   DEFAULT NULL COMMENT '修改人员姓名',
    mod_tm           datetime      DEFAULT NULL COMMENT '修改日期时间',
    chk_psn_id       varchar(6)    DEFAULT NULL COMMENT '复核人员ID',
    chk_psn_fll_nm   varchar(60)   DEFAULT NULL COMMENT '复核人员姓名',
    chk_tm           datetime      DEFAULT NULL COMMENT '复核日期时间',
    PRIMARY KEY (mdl_ver_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='权重配置模型参数附件表';

CREATE TABLE iis_wght_cnfg_mdl
(
    pk_id          int(10) NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    mdl_ver_id     varchar(64)    DEFAULT NULL COMMENT '模型编号',
    aset_bclass_cd varchar(10)    DEFAULT NULL COMMENT '资产大类',
    indx_num       varchar(100)   DEFAULT NULL COMMENT '指数编码',
    indx_nm        varchar(500)   DEFAULT NULL COMMENT '指数名称',
    wght           decimal(11, 7) DEFAULT NULL COMMENT '权重',
    crt_tm         datetime       DEFAULT NULL COMMENT '新增时间',
    PRIMARY KEY (pk_id)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COMMENT='权重配置模型参数表';

CREATE TABLE iis_mdl_aset_pct_d
(
    mdl_ver_id     varchar(64) NOT NULL COMMENT '模型版本id',
    aset_bclass_cd varchar(10) NOT NULL COMMENT '资产大类编号',
    aset_bclass_nm varchar(50)     DEFAULT NULL COMMENT '资产大类名称',
    pct_yld_date   date        NOT NULL COMMENT '收益率日期',
    pct_yld        decimal(25, 20) DEFAULT NULL COMMENT '资产大类每日收益（百分比）',
    PRIMARY KEY (mdl_ver_id, aset_bclass_cd, pct_yld_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='模型资产收益率表';

CREATE TABLE iis_aset_allc_indx_wght
(
    mdl_ver_id       varchar(64) NOT NULL COMMENT '模型版本id',
    rsk_lvl          varchar(10) NOT NULL COMMENT '风险等级 (标准组合为1至6,其他组合从11开始递增)',
    rate             decimal(11, 7) DEFAULT NULL COMMENT '预期收益率',
    liquid           decimal(11, 7) DEFAULT NULL COMMENT '预期成功性',
    csh_mgt_typ_pos  decimal(11, 7) DEFAULT NULL COMMENT '现金管理类持仓',
    fx_yld_pos       decimal(11, 7) DEFAULT NULL COMMENT '固定收益类持仓',
    mix_strg_typ_pos decimal(11, 7) DEFAULT NULL COMMENT '混合策略类持仓',
    eqty_invst_pos   decimal(11, 7) DEFAULT NULL COMMENT '权益策略类持仓',
    altrnt_invst_pos decimal(11, 7) DEFAULT NULL COMMENT '另类投资持仓',
    shrp_prprtn      decimal(11, 7) DEFAULT NULL COMMENT '夏普比例',
    gamma_val        decimal(11, 7) DEFAULT NULL COMMENT 'GAMMA值',
    VaR99            decimal(11, 7) DEFAULT NULL COMMENT 'VaR(99%)',
    VaR95            decimal(11, 7) DEFAULT NULL COMMENT 'VaR(95%)',
    VaR975           decimal(11, 7) DEFAULT NULL COMMENT 'VaR(97.5%)',
    VaR99_           decimal(11, 7) DEFAULT NULL COMMENT 'VaR(99%)',
    VaR95_b          decimal(11, 7) DEFAULT NULL COMMENT 'var95备份',
    isefct_fond      decimal(11, 7) DEFAULT NULL COMMENT '是否有效前沿',
    rsvn_fld1        decimal(11, 7) DEFAULT NULL COMMENT '预留字段1',
    rsvn_fld2        decimal(11, 7) DEFAULT NULL COMMENT '预留字段2',
    crt_tm           datetime       DEFAULT NULL COMMENT '创建时间',
    PRIMARY KEY (mdl_ver_id, rsk_lvl)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='资产配置权重盯市信息表';

CREATE TABLE iis_aset_allc_indx_pub
(
    mdl_ver_id       varchar(64) NOT NULL COMMENT '模型版本id',
    rsk_lvl          varchar(10) NOT NULL COMMENT '风险等级 (标准组合为1至6,其他组合从11开始递增)',
    rate             decimal(11, 7) DEFAULT NULL COMMENT '预期收益率',
    liquid           decimal(11, 7) DEFAULT NULL COMMENT '预期成功性',
    csh_mgt_typ_pos  decimal(11, 7) DEFAULT NULL COMMENT '现金管理类持仓',
    fx_yld_pos       decimal(11, 7) DEFAULT NULL COMMENT '固定收益类持仓',
    mix_strg_typ_pos decimal(11, 7) DEFAULT NULL COMMENT '混合策略类持仓',
    eqty_invst_pos   decimal(11, 7) DEFAULT NULL COMMENT '权益策略类持仓',
    altrnt_invst_pos decimal(11, 7) DEFAULT NULL COMMENT '另类投资持仓',
    shrp_prprtn      decimal(11, 7) DEFAULT NULL COMMENT '夏普比例',
    gamma_val        decimal(11, 7) DEFAULT NULL COMMENT 'GAMMA值',
    VaR99            decimal(11, 7) DEFAULT NULL COMMENT 'VaR(99%)',
    VaR95            decimal(11, 7) DEFAULT NULL COMMENT 'VaR(95%)',
    VaR975           decimal(11, 7) DEFAULT NULL COMMENT 'VaR(97.5%)',
    VaR99_           decimal(11, 7) DEFAULT NULL COMMENT 'VaR(99%)',
    rsvn_fld1        decimal(11, 7) DEFAULT NULL COMMENT '预留字段1',
    rsvn_fld2        decimal(11, 7) DEFAULT NULL COMMENT '预留字段2',
    crt_tm           datetime       DEFAULT NULL COMMENT '创建时间',
    VaR95_b          decimal(11, 7) DEFAULT NULL COMMENT 'var95备份',
    PRIMARY KEY (mdl_ver_id, rsk_lvl)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='资产指数公共参数表';

CREATE TABLE iis_aset_allc_indx_rtrn
(
    mdl_ver_id     varchar(64) NOT NULL COMMENT '模型版本id',
    aset_bclass_cd varchar(10) NOT NULL COMMENT '资产大类编号',
    aset_bclass_nm varchar(50)     DEFAULT NULL COMMENT '资产大类名称',
    pct_yld        decimal(25, 20) DEFAULT NULL COMMENT '资产大类收益率（百分比）',
    pct_std        decimal(25, 20) DEFAULT NULL COMMENT '资产大类波动率（百分比）',
    data_dt        date            DEFAULT NULL COMMENT '数据日期',
    crt_tm         datetime        DEFAULT NULL COMMENT '创建时间',
    PRIMARY KEY (mdl_ver_id, aset_bclass_cd)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='资产指数收益信息表';

COMMIT;
