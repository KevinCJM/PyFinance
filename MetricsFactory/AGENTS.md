# Purpose

Machine-first routing protocol for downstream agents operating from the current working directory.

# Scope Boundary

- Treat `.` as the writable project boundary unless higher-priority instructions say otherwise.
- Keep routing facts in JSON files under `docs/`.
- Record uncertain external behavior as `unknown` or `out_of_scope` instead of implementation truth.

# Required Read Order

1. `AGENTS.md`
2. `docs/repo_map.json`
3. `docs/task_routes.json`
4. `docs/pitfalls.json`
5. Routed code, tests, and configs

# Hard Rules

- `docs/task_routes.json` owns task matching, module expansion, and operational-list merge policy.
- `docs/repo_map.json` owns module facts and operational lists.
- `docs/pitfalls.json` owns hidden contracts, recurring pitfalls, and safe checks.
- Do not duplicate module-level file, test, config, or regression lists in `docs/task_routes.json`.
- Verify claims from code, tests, configs, or command output before promoting them to routing memory.

# AI Routing Self-Evolution

- Treat `docs/ai_routing_evolution_policy.json` as governance only; routing facts belong in `docs/task_routes.json`, `docs/repo_map.json`, and `docs/pitfalls.json`.
- Update `AGENTS.md` only when protocol, required read order, scope rules, or tool workflow changes.
- Promote verified hidden contracts and recurring pitfalls to the correct JSON owner.
- Use `skills/ai-hermes-self-evolve/scripts/evolve_ai_routing.py` after code, test, config, tool, or routing changes to check coverage.
- For routing-only work, run `skills/ai-hermes-self-evolve/scripts/evolve_ai_routing.py --routing-only` with explicit changed paths.
- Re-run `skills/ai-hermes-self-evolve/scripts/validate_ai_routing.py` after routing file changes.

# MetricsFactory Usage Notes

## Purpose

本目录是金融产品指标计算工具，核心入口在 `metrics_factory.py`：

- `compute_metrics_for_period_initialize(...)`: 按指定结束日和区间计算区间绩效/风险/净值/成交量指标，按区间保存为 `{period}.parquet`。
- `compute_all_rolling_metrics(...)`: 计算技术分析类滚动指标，保存为 `rolling_metrics.parquet`。
- 实际公式在 `period_metrics_cal.py` 和 `rolling_metrics_cal.py`；可用指标和默认区间在 `metrics_cal_config.py`。

## Runtime Environment

- 本项目不保留项目内 `skills/metrics-factory`；MetricsFactory skill 已安装为 Codex 全局技能。
- 指标运行、运行时导入和 `run_metrics_job.py` 必须使用通过 `${CODEX_HOME:-$HOME/.codex}/skills/metrics-factory/scripts/check_runtime.py --project-root .` 的 Python。
- 不要默认使用系统 `python3`；它可能与 NumPy/Pandas native wheel 架构不一致。
- 可用 `${CODEX_HOME:-$HOME/.codex}/skills/metrics-factory/scripts/setup_runtime.py --project-root .` 创建项目局部 `.metricsfactory-venv`。
- 已存在的项目 `.venv` 或外部 venv 也可以使用，但必须先通过 runtime check。

## Data Format

- 所有输入都是宽表 `pandas.DataFrame`：index 为升序日期，columns 为产品代码，values 为对应数值。
- 区间入口需要 `log_return_df`, `close_price_df`, `high_price_df`, `low_price_df`, `volume_df`。
- 滚动入口需要 `open_price_df`, `close_price_df`, `high_price_df`, `low_price_df`, `volume_df`。
- 输出长表字段固定含 `ts_code` 和 `date`；指标列名为 `指标名:区间` 或 `指标名:滚动天数`。
- `save_path` 必须已存在；函数只写 parquet，不返回结果。

## Strong Data Recommendation

强烈建议用复权净值/复权价格计算：

- `close_price_df` 应使用复权净值，并由它生成 `log_return_df = log(close_t / close_{t-1})`。
- `open/high/low` 也应与 `close` 同一复权口径；只有净值产品没有 OHLC 时，可用复权净值临时代替 OHLC，但 AR/BR/DKX/CCI 等价格形态指标解释性会下降。
- 不建议混用未复权价格、单位净值和累计净值；这会污染收益、回撤、净值斜率、新高率和技术指标。
- `volume_df` 只对成交量类指标有意义；无成交量产品应避免解读 `Vol*`, `OBV`, `PVT`, `VR` 等指标。

## Period Metrics

默认 `period_list` 为：
`2d, 3d, 5d, 6d, 7d, 10d, 15d, 20d, 25d, 50d, 75d, 5m, 6m, 9m, 12m, 2y, 3y, 5y, mtd, qtd, ytd, max`。

实际配置了指标的区间为：
`2d, 3d, 5d, 6d, 7d, 10d, 15d, 20d, 25d, 30d, 35d, 50d, 70d, 75d, 5m, 6m, 9m, 12m, 2y`。

注意：

- 默认运行只会计算 `period_list` 和指标映射的交集；`3y, 5y, mtd, qtd, ytd, max` 当前会被跳过。
- `30d, 35d, 70d` 有指标映射，但不在默认 `period_list`，要通过 `p_list` 显式传入。
- 所有 `Nd` 区间按交易日数量截取；`Nm/Ny/mtd/qtd/ytd` 由 `get_start_date()` 按自然日边界计算。
- `spec_end_date` 必须是交易日；`fund_list` 会筛选列；`min_data_required` 默认 2。

区间指标共 85 个，主要包括：

- 收益类：`TotalReturn`, `AnnualizedReturn`, `AverageDailyReturn`, `AvgPositiveReturn`, `AvgNegativeReturn`, `AvgReturnRatio`, `TotalPositiveReturn`, `TotalNegativeReturn`, `TotalReturnRatio`, `MedianDailyReturn`, `MaxGain`, `MaxLoss`。
- 波动和风险类：`Volatility`, `AnnualizedVolatility`, `MeanAbsoluteDeviation`, `ReturnRange`, `RescaledRange`, `DownsideVolatility`, `UpsideVolatility`, `VolatilitySkew`, `VolatilityRatio`, `ReturnVolatilityRatio`。
- 回撤类：`MaxDrawDown`, `MaxDrawDownDays`, `ReturnDrawDownRatio`, `DrawDownSlope`, `UlcerIndex`, `MartinRatio`。
- 风险调整收益：`SharpeRatio`, `AnnualizedSharpeRatio`, `SortinoRatio`, `GainConsistency`, `LossConsistency`, `WinningRatio`, `LosingRatio`。
- 分布和尾部：`ReturnSkewness`, `ReturnKurtosis`, `VaR-99`, `VaR-95`, `VaR-90`, `VaRSharpe-95`, `VaRModified-*`, `VaRModifiedSharpe-95`, `CVaR-*`, `CVaRModified-*`, `CVaRSharpe-95`, `CVaRModifiedSharpe-95`, `Percentile-*`, `PercentileWin-*`, `PercentileLoss-*`, `TailRatio-*`。
- 序列形态：`NewHighRatio`, `CrossProductRatio-1`, `CrossProductRatio-5`, `CrossProductRatio-10`, `HurstExponent`, `OmegaRatio`, `ReturnDistributionIntegral`, `ReturnSlope`, `KRatio`, `SortinoSkewness`, `NetEquitySlope`, `EquitySmoothness`。
- 价量区间：`VolAvg`, `VolSlope`, `VolVolatility`, `MaxHigh`, `MinLow`, `HLDiff`, `AvgHigh`, `AvgLow`。

特殊区间限制：

- `HurstExponent` 只配置在 `70d, 5m, 6m, 9m, 12m, 2y`。
- `CrossProductRatio-5` 从 `50d` 起；`CrossProductRatio-10` 从 `75d` 起。
- `VolSlope` 只配置在 `3d, 5d, 10d, 15d, 25d, 50d, 75d, 5m, 6m`。

## Rolling Metrics

滚动天数支持：
`0, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 19, 20, 22, 25, 26, 30, 35, 60, 99`。

- `0` 只用于无需窗口的 `OBV`, `PVT`, `TR`。
- 其他滚动天数默认各计算 93 个滚动指标。
- `roll_list` 可显式指定滚动天数；`num_workers` 参数当前未参与并行。

滚动指标共 96 个，主要包括：

- 均线/布林：`PriceSigma`, `CloseMA`, `CloseMADiff`, `BollUp-*`, `BollUpDiff-*`, `BollDo-*`, `BollDoDiff-*`, `BollUpDo-*`。
- KDJ/EMA/RSI：`L`, `H`, `RSV`, `KDJ-K-3`, `KDJ-D-3`, `KDJ-J-3`, `KDJ-KD-3`, `KDJ-KJ-3`, `KDJ-DJ-3`, `EMA`, `EMADiff`, `RSI`。
- 量价：`VolMA`, `VolMADiff`, `OBV`, `MAOBV`, `MAOBVDiff`, `PVT`, `MAPVT`, `MAPVTDiff`, `VR`, `MAVR-*`, `MAVRDiff-*`。
- 动量/趋势：`MTM`, `MTMMA-*`, `MTMMADiff-*`, `TRIX`, `MATRIX-*`, `MATRIXDiff-*`, `PSY`, `MAPSY-*`, `MAPSYDiff-*`。
- 价格形态：`CCI`, `CR`, `MACR-*`, `MACRDiff-*`, `AR`, `BR`, `BRARDiff`, `ARDiff-*`, `BRDiff-*`, `BIAS`, `MABIAS-*`, `MABIASDiff-*`。
- DMI/DKX：`TR`, `PDI`, `MDI`, `PDIMDIDiff`, `ADX-6`, `ADXR-6-6`, `ADXRDiff-6-6`, `ADXR-6-14`, `ADXRDiff-6-14`, `DKX`, `DKXDiff`, `MADKX-*`, `MADKXDiff-*`。

重要风险：`compute_all_rolling_metrics()` 调用 `CalRollingMetrics` 时传入顺序是 `open_price_array, close_price_array`，但构造函数签名写的是 `close_price_array, open_price_array`。当前实现可能把 open/close 互换，使用滚动指标前应先修正或按现状做对照验证。

## Not Wired By Current Entrypoints

- `log_return_relative_metrics_dict` 当前没有接入任何入口，并且部分字符串注释会和指标 key 发生字面量拼接，不应视为可直接运行指标。
- `long_short_metrics` 中的 `BBI`, `DMA`, `DIF`, `DEA`, `MACD`, `ADXSlop-6` 当前没有进入 `create_rolling_metrics_map()`，计算类也没有公开分派。
