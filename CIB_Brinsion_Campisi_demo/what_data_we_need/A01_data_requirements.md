# FOF 业绩归因分析所需数据说明

本文档详细说明了执行本项目中三种核心业绩归因模型（权益Brinsion、大类Brinsion、债券Campisi）所需的数据表、字段及其用途。所有示例数据均已从项目中的 `.parquet` 文件导出为 Excel 格式，存放于当前文件夹下，方便查阅。

---

## 1. 权益类Brinsion归因

**目标**：分析FOF组合在**股票投资**上的超额收益来源，分解为**行业配置**和**个股选择**。

| 所需数据表 | 对应Excel示例 | 用途说明 |
| :--- | :--- | :--- |
| **FOF持仓权重** | `fof_holding_equity.xlsx` | **聚合权重**：定义了FOF组合中各子基金的权重，是最后一步加权聚合单基金归因结果、得出FOF整体归因的核心。 |
| **基金持仓数据** | `fund_hold.xlsx` | **组合端核心**：定义了FOF所投子基金的股票持仓及权重，是计算组合在各行业、各股票上权重分配的基础。 |
| **基准指数成分股** | `index_hold.xlsx` | **基准端核心**：定义了业绩比较基准的构成，用于与组合持仓进行对比，是计算超额收益的“标尺”。 |
| **股票基本信息** | `stock_info.xlsx` | **连接个股与行业**：提供从股票代码到其所属行业的映射关系，是进行“行业配置”分析的前提。 |
| **股票日行情** | `stock_daily.xlsx` | **业绩来源（个股）**：提供计算分析区间内每一只股票收益率的基础数据。 |
| **指数日行情** | `index_daily.xlsx` | **业绩来源（基准）**：提供计算基准指数自身在区间内的整体收益率，用于确定总超额收益。 |

### 字段明细:
- **`fof_holding_equity.xlsx`**:
    - `fund_code`: 股票基金代码
    - `weight`: 占FOF组合权重
- **`fund_hold.xlsx`**:
    - `fund_code`: 基金代码
    - `report_date`: 持仓报告期
    - `stock_code`: 股票代码
    - `weight`: 占基金净值权重
- **`index_hold.xlsx`**:
    - `index_code`: 指数代码
    - `report_date`: 成分股报告期
    - `stock_code`: 股票代码
    - `weight`: 占指数权重
- **`stock_info.xlsx`**:
    - `stock_code`: 股票代码
    - `industry`: 所属行业
- **`stock_daily.xlsx`**:
    - `date`: 交易日期
    - `stock_code`: 股票代码
    - `close`: 前复权收盘价（用于计算收益率）
- **`index_daily.xlsx`**:
    - `date`: 交易日期
    - `index_code`: 指数代码
    - `close`: 收盘价（用于计算收益率）

---

## 2. 大类Brinsion归因

**目标**：从宏观视角分析FOF在**大类资产**（如股票、债券、商品等）上的配置表现。

| 所需数据表 | 对应Excel示例 | 用途说明 |
| :--- | :--- | :--- |
| **FOF持仓权重** | `fof_holding.xlsx` | **组合构成**：定义了FOF投资了哪些子基金，以及每只子基金的投资权重。 |
| **基金基本信息** | `fund_info.xlsx` | **大类资产映射（组合端）**：提供从基金代码到其所属大类资产（`fund_type`）的映射，用于将FOF持仓聚合到大类资产层面。 |
| **基金日净值** | `fund_daily_return.xlsx` | **业绩来源（组合端）**：提供计算分析区间内每一只子基金收益率的基础数据。 |
| **基准配置权重** | `benchmark_holding.xlsx` | **基准构成**：定义了业绩基准由哪些大类资产指数构成，以及各自的权重。 |
| **指数大类映射** | `csi_index_type.xlsx` | **大类资产映射（基准端）**：提供从基准的指数代码到其所属大类资产的映射，确保与组合端口径一致。 |
| **指数日行情** | `index_daily_all.xlsx` | **业绩来源（基准端）**：提供计算分析区间内每一个大类资产指数收益率的基础数据。 |

### 字段明细:
- **`fof_holding.xlsx`**:
    - `fund_code`: 基金代码
    - `weight`: 占FOF组合权重
- **`fund_info.xlsx`**:
    - `fund_code`: 基金代码
    - `fund_type`: 基金类型（如：股票型、债券型）
- **`fund_daily_return.xlsx`**:
    - `fund_code`: 基金代码
    - `date`: 净值日期
    - `adj_nav`: 复权单位净值（用于计算收益率）
- **`benchmark_holding.xlsx`**:
    - `index_code`: 大类资产指数代码
    - `weight`: 占基准权重
- **`csi_index_type.xlsx`**:
    - `index_code`: 大类资产指数代码
    - `index_type`: 指数所属大类
- **`index_daily_all.xlsx`**:
    - `index_code`: 指数代码
    - `date`: 交易日期
    - `close`: 收盘价（用于计算收益率）

---

## 3. 债券类Campisi归因

**目标**：深入分析FOF组合中**债券部分**的收益来源，分解为票息、久期和利差等驱动因素。

| 所需数据表 | 对应Excel示例 | 用途说明 |
| :--- | :--- | :--- |
| **FOF债券持仓** | `fof_holding_campisi.xlsx` | **组合构成**：定义了FOF投资了哪些债券基金及权重，用于最终的加权聚合。 |
| **基金基本信息** | `fund_info.xlsx` | **筛选债券基金**：用于从FOF全部持仓中识别出“债券型”基金，确保归因对象正确。 |
| **期初财务数据** | `start_p_data_campisi.xlsx` | **久期效应基础**：提供期初的债券市值（作为收益率计算分母）和**期初久期**（计算久期效应的核心参数）。 |
| **区间财务数据** | `in_p_data_campisi.xlsx` | **收益来源**：提供区间内的**利息收入**（用于计算票息收益）和**投资收入/公允价值变动**（用于计算资本利得）。 |
| **国债收益率曲线** | `yield_curve_campisi.xlsx` | **利率变动衡量**：提供期初和期末的无风险利率水平，用于计算利率变动(`dy`)，是计算久期效应的关键。 |

### 字段明细:
- **`fof_holding_campisi.xlsx`**:
    - `fund_code`: 债券基金代码
    - `weight`: 占FOF组合权重
- **`start_p_data_campisi.xlsx`**:
    - `fund_code`: 基金代码
    - `date`: 报告日期（期初）
    - `start_bond_mv`: 期初债券市值
    - `start_fund_duration`: 基金期初久期
- **`in_p_data_campisi.xlsx`**:
    - `fund_code`: 基金代码
    - `date`: 报告日期（期末）
    - `interest_income`: 利息收入
    - `bond_invest_income`: 债券投资收入
    - `fair_value_change`: 债券公允价值变动
- **`yield_curve_campisi.xlsx`**:
    - `date`: 日期
    - `duration`: 期限（年）
    - `yield`: 收益率(%)
