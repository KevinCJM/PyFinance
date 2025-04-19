import pandas as pd

res = pd.read_parquet("/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/Metrics/rolling_metrics.parquet")
res = res[res['ts_code'] == '510050.SH']
print(res)
