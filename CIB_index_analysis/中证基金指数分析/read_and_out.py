# -*- encoding: utf-8 -*-
"""
@File: read_and_out.py
@Modify Time: 2025/09/11
@Author: Kevin-Chen
@Descriptions:
    - 读取指定的指数 Excel 文件
    - 提取核心字段
    - 输出到单个 Excel 文件
"""
import os
import pandas as pd
from typing import List


def read_and_combine_excel(file_list: List[str]) -> pd.DataFrame:
    """
    读取指定的 Excel 文件列表，并统一列名 -> 纵表：date, index_code, index_name, close
    """
    cols_map = {
        '日期Date': 'date',
        '指数代码Index Code': 'index_code',
        '指数中文全称Index Chinese Name(Full)': 'index_name',
        '收盘Close': 'close',
    }
    dfs = []
    for fp in file_list:
        if not os.path.exists(fp):
            print(f"[WARN] 文件未找到，已跳过: {fp}")
            continue

        try:
            df = pd.read_excel(fp)
        except Exception as e:
            print(f"[ERROR] 读取文件失败 {fp}: {e}")
            continue

        df = df.rename(columns=cols_map)
        required_cols = ['date', 'index_code', 'index_name', 'close']

        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f'[WARN] 文件 {os.path.basename(fp)} 缺少必需列 {missing_cols}，已跳过。')
            continue

        df = df[required_cols].copy()
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
        df['index_code'] = df['index_code'].astype(str)
        df['index_name'] = df['index_name'].astype(str)
        df['close'] = pd.to_numeric(df['close'], errors='coerce')

        # 移除日期或收盘价为空的无效行
        df.dropna(subset=['date', 'close'], inplace=True)

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    all_data = pd.concat(dfs, ignore_index=True)
    return all_data


def main(identifiers_to_include):
    """
    主函数
    """
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 输出文件名
    output_filename = 'filtered_index_data.xlsx'
    # =================================================

    # 1. 自动查找目录下的所有 .xlsx 文件
    try:
        all_excel_files = [f for f in os.listdir(current_dir) if f.lower().endswith('.xlsx')]
    except FileNotFoundError:
        print(f"[ERROR] 无法访问目录: {current_dir}")
        return

    # 2. 根据标识符筛选文件
    target_files = []
    for identifier in identifiers_to_include:
        found_for_id = False
        for filename in all_excel_files:
            # 检查标识符是否是文件名的一部分（可匹配代码或名称）
            if identifier in filename:
                full_path = os.path.join(current_dir, filename)
                if full_path not in target_files:
                    target_files.append(full_path)
                found_for_id = True
        if not found_for_id:
            print(f"[WARN] 未找到与标识符 '{identifier}' 匹配的 .xlsx 文件。")

    if not target_files:
        print("[INFO] 未根据提供的标识符找到任何要处理的文件。程序退出。")
        return

    print(f"根据您的输入，将处理以下 {len(target_files)} 个文件:")
    for f_path in target_files:
        print(f"  - {os.path.basename(f_path)}")
    print("-" * 30)

    # 读取并合并数据
    combined_data = read_and_combine_excel(target_files)

    if combined_data.empty:
        print("\n[INFO] 未能处理任何数据，程序退出。")
        return

    # 定义输出文件的完整路径
    output_path = os.path.join(current_dir, output_filename)

    # 保存到 Excel
    try:
        combined_data.to_excel(output_path, index=False, engine='openpyxl')
        print(f"\n[SUCCESS] 数据已成功合并并保存至: {output_path}")
        print("\n输出数据预览 (前5行):")
        print(combined_data.head())
    except Exception as e:
        print(f"\n[ERROR] 保存 Excel 文件失败: {e}")


if __name__ == '__main__':
    # ===================== 参数配置 =====================
    # 在这里指定您想要包含的指数代码 (str) 或名称关键字
    index_list = [
        '中证商品期货成份指数',  # 示例: 通过指数代码
        '中证债券型基金指数',  # 示例: 通过名称关键字
        '中证QDII基金指数',  # 示例: 通过完整名称
        '中证股票型基金指数',
        '中证货币基金指数',
        # 您可以在此添加更多代码或名称
    ]
    main(index_list)
