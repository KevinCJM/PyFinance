import pandas as pd
import glob
import os

# 定义输出目录路径，用于存放转换后的Excel文件
output_dir = '/Users/chenjunming/Desktop/KevinGit/PyFinance/CIB_Brinsion_Campisi_demo/what_data_we_need'
# 创建输出目录，如果目录已存在则不会报错
os.makedirs(output_dir, exist_ok=True)

# 定义数据源目录路径，用于查找需要转换的parquet文件
data_dir = '/Users/chenjunming/Desktop/KevinGit/PyFinance/CIB_Brinsion_Campisi_demo/data'
# 使用glob模式匹配查找所有.parquet文件
parquet_files = glob.glob(os.path.join(data_dir, '*.parquet'))

# 打印找到的parquet文件数量
print(f"找到 {len(parquet_files)} 个 parquet 文件进行转换。")

# 遍历所有找到的parquet文件，逐个转换为Excel格式
for file_path in parquet_files:
    try:
        # 读取parquet文件到pandas DataFrame
        df = pd.read_parquet(file_path)

        # 获取文件的基本名称（不含路径）
        base_name = os.path.basename(file_path)
        # 构造Excel文件名，将.parquet扩展名替换为.xlsx
        excel_name = os.path.splitext(base_name)[0] + '.xlsx'
        # 构造完整的Excel文件输出路径
        excel_path = os.path.join(output_dir, excel_name)

        # 将DataFrame保存为Excel文件，不包含行索引
        df.to_excel(excel_path, index=False)
        # 打印转换成功的消息
        print(f"成功将 {base_name} 转换为 {excel_name}")
    except Exception as e:
        # 捕获转换过程中可能出现的异常并打印错误信息
        print(f"无法转换 {file_path}: {e}")

# Data from A03_equity_brinsion_cal_fof_cib.py
fof_hold = {
    '000082': 0.18,
    '000309': 0.22,
    '000628': 0.11,
    '000729': 0.12,
}

# Convert to DataFrame
df = pd.DataFrame(fof_hold.items(), columns=['fund_code', 'weight'])

# Define output path
output_path = '/Users/chenjunming/Desktop/KevinGit/PyFinance/CIB_Brinsion_Campisi_demo/what_data_we_need/fof_holding_equity.xlsx'

# Save to Excel
df.to_excel(output_path, index=False)

print(f"Successfully created {output_path}")

# 打印文件转换流程结束的提示信息
print("文件转换流程结束。")
