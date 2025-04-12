
# PyFinance

一个用于获取和处理中国市场ETF日频数据，并计算各类ETF指标的金融数据分析工具库。支持自动从 Tushare 拉取数据、清洗处理、宽表构建、指标批量计算等功能，适用于量化分析、因子研究等场景。

---

## 📁 项目结构说明

```plaintext
PyFinance/
├── Data/                         # 数据目录
│   ├── etf_daily.parquet         # 原始ETF日频数据
│   ├── wide_*.parquet            # 预处理后的各类宽表数据（按字段划分）
│   └── Metrics/                  # 计算得到的指标数据
│       ├── 2d.parquet
│       ├── 3d.parquet
│       └── ...更多指标结果
├── GetData/                      # 数据获取与预处理模块
│   ├── data_prepare.py
│   └── tushare_get_ETF_data.py
├── MetricsFactory/               # 指标计算模块
│   ├── metrics_cal.py
│   ├── metrics_cal_config.py
│   └── metrics_factory.py
├── main_get_data.py              # 主程序：数据获取与预处理
├── main_cal_metrics.py           # 主程序：指标计算
├── set_tushare.py                # 设置 Tushare Token 的脚本
├── requirements.txt              # Python 依赖列表
├── LICENSE
└── README.md                     # 本文件
```

---

## 🚀 快速开始

### 1️⃣ 安装依赖

请先确保安装了所需的Python依赖：

```bash
pip install -r requirements.txt
```

### 2️⃣ 设置 Tushare Token

打开 `set_tushare.py`，将你的 Tushare Token 写入环境变量或脚本中。

```python
import tushare as ts
ts.set_token("your_token_here")
```

---

## 📥 获取并预处理ETF数据

执行主程序：

```bash
python main_get_data.py
```

程序逻辑如下：

- 若 `Data/etf_daily.parquet` 不存在：
  - 调用 **全量拉取函数** `get_etf_daily_data_all()` 获取所有ETF历史数据；
- 若文件已存在：
  - 调用 **增量更新函数** `get_etf_daily_data_increment()` 获取最新数据；

无论哪种方式，随后都会执行：

- `data_prepare()` 对数据进行清洗、过滤与宽表构建；
- 最终输出多个以 `wide_*.parquet` 命名的宽格式DataFrame，分别存储开盘价、收盘价、成交量、对数收益率等。

---

## 📊 计算ETF指标

```bash
python main_cal_metrics.py
```

该模块将基于 ETF 日频对数收益率数据与收盘价数据，计算多个区间（如1日、3日、1月、3月等）的指标结果。

你可以在 `MetricsFactory/metrics_cal_config.py` 文件中：

- 自定义指标类型
- 定义计算周期
- 设置指标公式

实际的指标逻辑代码写在 `metrics_cal.py` 中，支持矢量化加速和多进程共享内存。

对于目前支持的指标, 以及各个指标的含义, 计算逻辑, 支持的周期等信息, 请查看 `MetricsFactory/metrics_cal_config.py` 文件 或 `MetricsFactory/指标说明.xlsx` 文件。

结果将保存于 `Data/Metrics/` 目录下。

---

## ✅ 增加自定义指标

若希望扩展自定义指标：

1. 在 `metrics_cal_config.py` 中添加配置项（名称、周期、公式说明）；
2. 在 `metrics_cal.py` 中添加计算逻辑；
3. 无需修改主程序，会自动加载并执行。

---

## 📌 示例字段解释（以原始ETF数据为例）

| 字段名        | 含义 |
|--------------|------|
| `ts_code`     | ETF代码 |
| `trade_date`  | 交易日期 |
| `open`        | 开盘价 |
| `close`       | 收盘价 |
| `vol`         | 成交量 |
| `amount`      | 成交金额 |
| `pct_chg`     | 涨跌幅 |
| `change`      | 涨跌额 |
| `log_return`  | 对数收益率（后续计算生成） |

---

## 📜 LICENSE

本项目使用 MIT License 发布。

---

## 📮 联系方式

如果你在使用过程中遇到问题，欢迎提出, 可以添加微信: Kevin-CJM。

---
