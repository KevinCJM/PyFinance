CREATE TABLE iis_wght_cnfg_attc_mdl
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
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4 COMMENT ='权重配置模型参数附件表';

CREATE TABLE iis_aset_allc_indx_wght
(
    mdl_ver_id       varchar(64) NOT NULL COMMENT '模型版本id',
    rsk_lvl          varchar(10) NOT NULL COMMENT '风险等级',
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
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4 COMMENT ='资产配置权重盯市信息表';

CREATE TABLE iis_aset_allc_indx_pub
(
    mdl_ver_id          varchar(64) NOT NULL COMMENT '模型版本id',
    rsk_lvl             varchar(10) NOT NULL COMMENT '风险等级',
    rate                decimal(11, 7) DEFAULT NULL COMMENT '预期收益率',
    liquid              decimal(11, 7) DEFAULT NULL COMMENT '预期成功性',
    csh_mgt_typ_pos     decimal(11, 7) DEFAULT NULL COMMENT '现金管理类持仓',
    fx_yld_pos          decimal(11, 7) DEFAULT NULL COMMENT '固定收益类持仓',
    mix_strg_typ_pos    decimal(11, 7) DEFAULT NULL COMMENT '混合策略类持仓',
    eqty_invst_typ_pos  decimal(11, 7) DEFAULT NULL COMMENT '权益策略类持仓',
    altnt_invst_pos     decimal(11, 7) DEFAULT NULL COMMENT '另类投资持仓',
    shrp_prprtn         decimal(11, 7) DEFAULT NULL COMMENT '夏普比例',
    gamma_val           decimal(11, 7) DEFAULT NULL COMMENT 'GAMMA值',
    var99               decimal(11, 7) DEFAULT NULL COMMENT 'VaR(99%)',
    var95               decimal(11, 7) DEFAULT NULL COMMENT 'VaR(95%)',
    var975              decimal(11, 7) DEFAULT NULL COMMENT 'VaR(97.5%)',
    var99_              decimal(11, 7) DEFAULT NULL COMMENT 'VaR(99%)',
    rsvn_fld1           decimal(11, 7) DEFAULT NULL COMMENT '预留字段1',
    rsvn_fld2           decimal(11, 7) DEFAULT NULL COMMENT '预留字段2',
    crt_tm              datetime       DEFAULT NULL COMMENT '创建时间',
    var95_b             decimal(11, 7) DEFAULT NULL COMMENT 'var95备份',
    PRIMARY KEY (mdl_ver_id, rsk_lvl)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4 COMMENT ='资产指数公共参数表';

CREATE TABLE iis_ef_grid_srch_wght (
  mdl_ver_id varchar(64) NOT NULL COMMENT '模型版本id',
  rsk_lvl varchar(10) NOT NULL COMMENT '风险等级',
  rate decimal(11,7) DEFAULT NULL COMMENT '预期收益率',
  liquid decimal(11,7) DEFAULT NULL COMMENT '预期流动性',
  csh_mgt_typ_pos decimal(11,7) DEFAULT NULL COMMENT '现金管理类持仓',
  fx_yld_pos decimal(11,7) DEFAULT NULL COMMENT '固定收益率持仓',
  mix_strg_typ_pos decimal(11,7) DEFAULT NULL COMMENT '混合策略类持仓',
  eqty_invst_typ_pos decimal(11,7) DEFAULT NULL COMMENT '权益投资类持仓',
  altnt_invst_pos decimal(11,7) DEFAULT NULL COMMENT '另类投资持仓',
  shrp_prprtn decimal(11,7) DEFAULT NULL COMMENT '夏普比例',
  var95 decimal(11,7) DEFAULT NULL COMMENT 'var(95%)',
  var95_b decimal(11,7) DEFAULT NULL COMMENT 'var95备份',
  is_efct_font varchar(1) DEFAULT NULL COMMENT '是否有效前沿',
  dt_dt date DEFAULT NULL COMMENT '数据日期',
  crt_tm datetime DEFAULT NULL COMMENT '创建时间',
  PRIMARY KEY (mdl_ver_id, rsk_lvl)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='有效前沿网格搜索权重点';

CREATE TABLE iis_ef_rndm_srch_wght (
  mdl_ver_id varchar(64) NOT NULL COMMENT '模型版本id',
  rsk_lvl varchar(10) NOT NULL COMMENT '风险等级',
  rate decimal(11,7) DEFAULT NULL COMMENT '预期收益率',
  liquid decimal(11,7) DEFAULT NULL COMMENT '预期流动性',
  csh_mgt_typ_pos decimal(11,7) DEFAULT NULL COMMENT '现金管理类持仓',
  fx_yld_pos decimal(11,7) DEFAULT NULL COMMENT '固定收益率持仓',
  mix_strg_typ_pos decimal(11,7) DEFAULT NULL COMMENT '混合策略类持仓',
  eqty_invst_typ_pos decimal(11,7) DEFAULT NULL COMMENT '权益投资类持仓',
  altnt_invst_pos decimal(11,7) DEFAULT NULL COMMENT '另类投资持仓',
  shrp_prprtn decimal(11,7) DEFAULT NULL COMMENT '夏普比例',
  var95 decimal(11,7) DEFAULT NULL COMMENT 'var(95%)',
  var95_b decimal(11,7) DEFAULT NULL COMMENT 'var95备份',
  is_efct_font varchar(1) DEFAULT '1' COMMENT '是否有效前沿',
  dt_dt date DEFAULT NULL COMMENT '数据日期',
  crt_tm datetime DEFAULT NULL COMMENT '创建时间',
  PRIMARY KEY (mdl_ver_id, rsk_lvl)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='有效前沿随机搜索权重点';

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
) ENGINE = InnoDB
  AUTO_INCREMENT = 2
  DEFAULT CHARSET = utf8mb4 COMMENT ='权重配置模型参数表';

CREATE TABLE iis_mdl_aset_pct_d
(
    mdl_ver_id     varchar(64) NOT NULL COMMENT '模型版本id',
    aset_bclass_cd varchar(10) NOT NULL COMMENT '资产大类编号',
    aset_bclass_nm varchar(50) DEFAULT NULL COMMENT '资产大类名称',
    pct_yld_date   date        NOT NULL COMMENT '收益率日期',
    pct_yld        decimal(25, 20) DEFAULT NULL COMMENT '资产大类每日收益（百分比）',
    acc_value      decimal(25, 20) DEFAULT NULL COMMENT '累计净值',
    PRIMARY KEY (mdl_ver_id, aset_bclass_cd, pct_yld_date)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4 COMMENT ='模型资产收益率表';

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
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4 COMMENT ='资产指数收益信息表';

CREATE TABLE iis_wght_cnfg_mdl_ast_rsk_rel (
  pk_id bigint(20) NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  mdl_ver_id varchar(64) DEFAULT NULL COMMENT '模型编号',
  aset_bclass_cd varchar(10) DEFAULT NULL COMMENT '资产大类',
  rsk_lvl varchar(1) DEFAULT NULL COMMENT '风险等级',
  wght decimal(11,7) DEFAULT NULL COMMENT '权重',
  PRIMARY KEY (pk_id)
) ENGINE=InnoDB AUTO_INCREMENT=1971108220271980548 DEFAULT CHARSET=utf8mb4 COMMENT='权重配置模型资产大类风险映射表';

CREATE TABLE iis_wght_cnfg_mdl_rsk (
  pk_id bigint(20) NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  mdl_ver_id varchar(64) DEFAULT NULL COMMENT '模型编号',
  rsk_lvl varchar(1) DEFAULT NULL COMMENT '风险等级',
  pct_yld decimal(25,20) DEFAULT NULL COMMENT '年化收益率',
  pct_std decimal(25,20) DEFAULT NULL COMMENT '年化波动率',
  shrp_prprtn decimal(25,20) DEFAULT NULL COMMENT '夏普比例',
  var_value decimal(25,20) DEFAULT NULL COMMENT 'VAR值',
  PRIMARY KEY (pk_id)
) ENGINE=InnoDB AUTO_INCREMENT=1971108220192288772 DEFAULT CHARSET=utf8mb4 COMMENT='权重配置模型风险表';

CREATE TABLE iis_fnd_indx_info
(
    indx_num             varchar(40) NOT NULL COMMENT '指数编码',
    indx_nm              varchar(100)  DEFAULT NULL COMMENT '指数名称',
    indx_enm             varchar(200)  DEFAULT NULL COMMENT '指数英文名称',
    brs_nm               varchar(40)   DEFAULT NULL COMMENT '交易所名称',
    publisher            varchar(100)  DEFAULT NULL COMMENT '发布人',
    publish_dt           date          DEFAULT NULL COMMENT '发布日期',
    publish_end_dt       date          DEFAULT NULL COMMENT '终止发布日期',
    indx_sty             varchar(40)   DEFAULT NULL COMMENT '指数风格',
    weight_typ           varchar(100)  DEFAULT NULL COMMENT '权重类型',
    component_stocks_num decimal(5, 0) DEFAULT NULL COMMENT '成份股数量',
    indx_rgn_cd          varchar(9)    DEFAULT NULL COMMENT '指数区域代码',
    indx_scale_cd        varchar(9)    DEFAULT NULL COMMENT '指数规模代码',
    weight_typ_cd        varchar(9)    DEFAULT NULL COMMENT '权重类型代码',
    own_mkt_cd           varchar(9)    DEFAULT NULL COMMENT '所属市场代码',
    income_way_cd        varchar(9)    DEFAULT NULL COMMENT '收益处理方式代码',
    indx_typ_cd          varchar(9)    DEFAULT NULL COMMENT '指数类别代码',
    indx_typ_nm          varchar(40)   DEFAULT NULL COMMENT '指数类别名称',
    indx_rmk             text COMMENT '指数备注',
    src_tab_ennm         varchar(100)  DEFAULT NULL COMMENT '来源表英文名',
    src_tab_cnnm         varchar(100)  DEFAULT NULL COMMENT '来源表中文名',
    PRIMARY KEY (indx_num)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='基金指数信息表';

CREATE TABLE wind_cmfindexeod
(
    object_id       varchar(100) NOT NULL COMMENT '对象ID',
    s_info_windcode varchar(40)    DEFAULT NULL COMMENT 'Wind代码',
    s_info_name     varchar(100)   DEFAULT NULL COMMENT '指数简称',
    trade_dt        varchar(8)     DEFAULT NULL COMMENT '交易日期',
    crncy_code      varchar(10)    DEFAULT NULL COMMENT '货币代码',
    s_dq_preclose   decimal(20, 4) DEFAULT NULL COMMENT '昨收盘价',
    s_dq_open       decimal(20, 4) DEFAULT NULL COMMENT '开盘价',
    s_dq_high       decimal(20, 4) DEFAULT NULL COMMENT '最高价',
    s_dq_low        decimal(20, 4) DEFAULT NULL COMMENT '最低价',
    s_dq_close      decimal(20, 4) DEFAULT NULL COMMENT '最新价',
    s_dq_volume     decimal(20, 4) DEFAULT NULL COMMENT '成交量手',
    s_dq_amount     decimal(20, 4) DEFAULT NULL COMMENT '成交金额千元',
    sec_id          varchar(10)    DEFAULT NULL COMMENT '证券ID',
    s_dq_change     decimal(20, 4) DEFAULT NULL COMMENT '涨跌点',
    s_dq_pctchange  decimal(20, 4) DEFAULT NULL COMMENT '涨跌幅',
    opdate          datetime       DEFAULT NULL,
    opmode          varchar(1)     DEFAULT NULL,
    PRIMARY KEY (object_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='中国共同基金指数行情';

CREATE TABLE wind_cmfindexeod_alt
(
    object_id       varchar(100) NOT NULL COMMENT '对象ID',
    s_info_windcode varchar(40)    DEFAULT NULL COMMENT 'Wind代码',
    s_info_name     varchar(100)   DEFAULT NULL COMMENT '指数简称',
    trade_dt        varchar(8)     DEFAULT NULL COMMENT '交易日期',
    crncy_code      varchar(10)    DEFAULT NULL COMMENT '货币代码',
    s_dq_preclose   decimal(20, 4) DEFAULT NULL COMMENT '昨收盘价',
    s_dq_open       decimal(20, 4) DEFAULT NULL COMMENT '开盘价',
    s_dq_high       decimal(20, 4) DEFAULT NULL COMMENT '最高价',
    s_dq_low        decimal(20, 4) DEFAULT NULL COMMENT '最低价',
    s_dq_close      decimal(20, 4) DEFAULT NULL COMMENT '最新价',
    s_dq_volume     decimal(20, 4) DEFAULT NULL COMMENT '成交量手',
    s_dq_amount     decimal(20, 4) DEFAULT NULL COMMENT '成交金额千元',
    sec_id          varchar(10)    DEFAULT NULL COMMENT '证券ID',
    s_dq_change     decimal(20, 4) DEFAULT NULL COMMENT '涨跌点',
    s_dq_pctchange  decimal(20, 4) DEFAULT NULL COMMENT '涨跌幅',
    opdate          datetime       DEFAULT NULL,
    opmode          varchar(1)     DEFAULT NULL,
    PRIMARY KEY (object_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='中国共同基金指数行情(扩展)';

COMMIT;
