# -*- encoding: utf-8 -*-
"""
@File: vectorize_tables.py
@Modify Time: 2025/7/5
@Author: KevinChen
@Descriptions: 本脚本用于读取 table_definitions.json 文件,
将其中的每个表的结构信息转换为文本块, 然后使用 sentence-transformers
模型生成向量嵌入, 并最终使用 FAISS 库将这些向量保存到本地索引文件中,
为后续的 RAG (Retrieval-Augmented Generation) 应用做准备。
"""
import json
import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# --- 配置中心 ---
CONFIG = {
    "json_path": "/Users/chenjunming/Desktop/KevinGit/PyFinance/table_definitions.json",
    "output_folder": "/Users/chenjunming/Desktop/KevinGit/PyFinance/rag_index",
    "embedding_model": 'paraphrase-multilingual-MiniLM-L12-v2',
    "faiss_index_file": "faiss_index.bin",
    "table_mapping_file": "table_mapping.json"
}


def create_text_chunks_from_json(json_data: dict) -> (list, list):
    """
    从JSON数据中为每个表创建文本块。

    :param json_data: 包含表定义的字典。
    :return: 一个元组，包含 (所有表的名称列表, 所有表的描述性文本块列表)。
    """
    table_names = []
    text_chunks = []

    print("正在为每个表创建文本描述...")
    for table_name, table_info in json_data.items():
        if not isinstance(table_info, dict):
            print(f"警告: 跳过无效的表定义 '{table_name}'。")
            continue

        table_names.append(table_name)

        # 构建一个详细的、对LLM友好的文本描述
        description = f"Table Name: {table_info.get('tableName', 'N/A')}\n"
        description += f"Chinese Name: {table_info.get('tableChiName', 'N/A')}\n"
        description += f"Description: {table_info.get('description', 'N/A')}\n"
        description += f"Update Time: {table_info.get('tableUpdateTime', 'N/A')}\n"
        description += f"Primary Key: {table_info.get('key', 'N/A')}\n"
        description += "Columns:\n"

        columns_data = table_info.get('columns', [])
        if isinstance(columns_data, list):
            for col in columns_data:
                # 检查列是否是有效的字典
                if isinstance(col, dict):
                    col_desc = ", ".join([f"{k}: {v}" for k, v in col.items()])
                    description += f"  - {col_desc}\n"
        else:
            print(f"警告: 表 '{table_name}' 的列数据格式不正确。")

        text_chunks.append(description.strip())

    print(f"成功为 {len(text_chunks)} 个表创建了文本块。")
    return table_names, text_chunks


def main():
    """
    主执行函数。
    """
    print("--- 开始向量化流程 ---")

    # 确保输出目录存在
    output_folder = CONFIG["output_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建输出目录: {output_folder}")

    # 1. 加载JSON文件
    try:
        with open(CONFIG["json_path"], 'r', encoding='utf-8') as f:
            table_definitions = json.load(f)
    except FileNotFoundError:
        print(f"错误: JSON文件未找到于路径 {CONFIG['json_path']}")
        return
    except json.JSONDecodeError:
        print(f"错误: 无法解析JSON文件 {CONFIG['json_path']}")
        return

    # 2. 创建文本块
    table_names, text_chunks = create_text_chunks_from_json(table_definitions)
    if not text_chunks:
        print("错误: 未能从JSON文件中创建任何文本块。")
        return

    # 3. 加载Embedding模型
    print(f"正在加载Embedding模型: {CONFIG['embedding_model']}...")
    model = SentenceTransformer(CONFIG['embedding_model'],
                                cache_folder=None,
                                use_auth_token=None,
                                revision=None,
                                trust_remote_code=True,
                                )
    print("模型加载完成。")

    # 4. 生成向量嵌入
    print("正在为所有文本块生成向量嵌入... (这可能需要一些时间)")
    embeddings = model.encode(text_chunks, show_progress_bar=True)
    print(f"向量生成完成，维度为: {embeddings.shape}")

    # 5. 构建并保存FAISS索引
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)  # 使用L2距离的精确索引
    index.add(embeddings.astype('float32'))  # FAISS需要float32类型

    faiss_index_path = os.path.join(output_folder, CONFIG["faiss_index_file"])
    faiss.write_index(index, faiss_index_path)
    print(f"FAISS索引已保存至: {faiss_index_path}")

    # 6. 保存表名与索引的映射关系
    table_mapping_path = os.path.join(output_folder, CONFIG["table_mapping_file"])
    with open(table_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(table_names, f, ensure_ascii=False, indent=4)
    print(f"表名映射已保存至: {table_mapping_path}")

    print("--- 向量化流程全部完成 ---")


if __name__ == '__main__':
    main()
