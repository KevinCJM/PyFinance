# rag_sql.py
import faiss, json, os, numpy as np
import argparse
from sentence_transformers import SentenceTransformer
from openai import OpenAI  # 也可换成 deepseek 等

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CONFIG = {
    "json_path": "/Users/chenjunming/Desktop/KevinGit/PyFinance/table_definitions.json",
    "index_path": "/Users/chenjunming/Desktop/KevinGit/PyFinance/rag_index/faiss_index.bin",
    "mapping_path": "/Users/chenjunming/Desktop/KevinGit/PyFinance/rag_index/table_mapping.json",
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "top_k": 4,
    "openai_key": "sk-a221b0c62c8a460693fe00a627d4598e",
    "openai_base": "https://api.deepseek.com",  # 如果走 DeepSeek
    "llm_model": "deepseek-chat"  # GPT-4o / gpt-3.5-turbo 亦可
}


def load_resources():
    index = faiss.read_index(CONFIG["index_path"])
    with open(CONFIG["mapping_path"], "r", encoding="utf-8") as f:
        table_names = json.load(f)
    embedder = SentenceTransformer(CONFIG["embedding_model"])
    return index, table_names, embedder


def search_tables(query, index, embedder, table_names, top_k=4):
    q_emb = embedder.encode([query])  # shape (1, dim)
    D, I = index.search(np.asarray(q_emb, dtype="float32"), top_k)
    hits = [(table_names[i], float(D[0][rank])) for rank, i in enumerate(I[0])]
    return hits  # 返回 (表名, 距离) 列表


def build_prompt(query, hits, table_definitions, db_dialect):
    context_blocks = []
    for tbl, _ in hits:
        info = table_definitions[tbl]
        # 只放关键信息，避免 token 过大
        block = (
                f"表名: {tbl}（{info['tableChiName']}）\n"
                f"说明: {info['description']}（{info['description_en']}）\n"
                f"主键: {info['key']}\n"
                f"字段:\n" + "\n".join([
            f"  - 列名: {c.get('列名', 'N/A')}, 数据类型: {c.get('数据类型', 'N/A')}, 备注: {c.get('备注', '无')}"
            for c in info['columns'][:15]
        ]) + "\n"
        )
        context_blocks.append(block)
    context = "\n\n".join(context_blocks)

    prompt = f"""
你是一名资深数据工程师，请根据以下数据库元数据，为用户编写高质量 SQL：

### 数据库元数据
{context}

### 用户需求
{query}

### 要求
- 生成的SQL必须兼容 {db_dialect} 语法。
- 请在SQL中添加注释，清晰地解释关键部分的逻辑，例如JOIN条件、WHERE子句的目的或复杂的函数用法。
- 必要时请合理 JOIN，JOIN 条件优先使用主键/外键 InnerCode 等。
- 只返回 SQL 代码（包含注释），不要附带任何额外的解释性文字。
"""
    return prompt.strip()


def build_full_prompt(query, table_definitions, db_dialect):
    """将所有表定义作为上下文构建prompt。"""
    context_blocks = []
    for tbl, info in table_definitions.items():
        block = (
                f"表名: {tbl}（{info.get('tableChiName', 'N/A')}）\n"
                f"说明: {info.get('description', 'N/A')}（{info.get('description_en', 'N/A')}）\n"
                f"主键: {info.get('key', 'N/A')}\n"
                f"字段:\n" + "\n".join([
            f"  - 列名: {c.get('列名', 'N/A')}, 数据类型: {c.get('数据类型', 'N/A')}, 备注: {c.get('备注', '无')}"
            for c in info.get('columns', [])
        ]) + "\n"
        )
        context_blocks.append(block)
    context = "\n\n".join(context_blocks)

    prompt = f"""
你是一名资深数据工程师，请根据以下完整的数据库元数据，为用户编写高质量 SQL：

### 数据库元数据
{context}

### 用户需求
{query}

### 要求
- 生成的SQL必须兼容 {db_dialect} 语法。
- 请在SQL中添加注释，清晰地解释关键部分的逻辑，例如JOIN条件、WHERE子句的目的或复杂的函数用法。
- 必要时请合理 JOIN，JOIN 条件优先使用主键/外键 InnerCode 等。
- 只返回 SQL 代码（包含注释），不要附带任何额外的解释性文字。
"""
    return prompt.strip()


def call_llm(prompt):
    client = OpenAI(api_key=CONFIG["openai_key"], base_url=CONFIG["openai_base"])
    resp = client.chat.completions.create(
        model=CONFIG["llm_model"],
        messages=[
            {"role": "system", "content": """
你是一名资深数据工程师，精通 SQL 编写。请严格遵循以下原则：
1.  **数据类型和值映射：** 如果字段备注中提到与 `CT_SystemConst` 表关联，或明确给出值到描述的映射（例如：`7-指数型, 8-优化指数型, 16-非指数型`），请务必将用户查询中的中文描述转换为对应的数字代码或英文缩写进行过滤。例如，如果用户查询“QDII类型”，而备注中说明 `InvestmentType` 字段 `7-QDII`，则应使用 `InvestmentType = 7`。
2.  **表选择：** 优先选择包含用户所需信息的表。对于日期相关的查询（如清盘日期），请优先考虑 `MF_FundArchives` 表中的 `ExpireDate` 或 `LastOperationDate` 字段，而不是 `MF_Transformation` 等不包含此类信息的表。
3.  **JOIN 条件：** 必要时请合理 JOIN，JOIN 条件优先使用主键/外键 InnerCode 等。
4.  **仔细遵循用户在 ### 要求 部分提供的所有格式化和内容指令。**
"""},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    return resp.choices[0].message.content.strip()


def main(query, mode, db_dialect):
    with open(CONFIG["json_path"], "r", encoding="utf-8") as f:
        table_defs = json.load(f)

    if mode == 'rag':
        print("🚀 正在使用 RAG 模式...")
        index, names, embedder = load_resources()
        hits = search_tables(query, index, embedder, names, CONFIG["top_k"])
        print("🔍 Top-k 检索结果:", hits)
        prompt = build_prompt(query, hits, table_defs, db_dialect)
    elif mode == 'direct':
        print("🚀 正在使用直接模式...")
        prompt = build_full_prompt(query, table_defs, db_dialect)
    else:
        raise ValueError("无效的模式，请选择 'rag' 或 'direct'")

    # print("\n--- 生成的 Prompt ---\n", prompt)  # 可选：取消注释以调试prompt
    sql = call_llm(prompt)
    print("\n=== 生成的 SQL ===\n", sql)


if __name__ == "__main__":
    # 交互式选择模式
    mode_choice = ''
    while mode_choice not in ['1', '2']:
        mode_choice = input("请选择运行模式:\n1. RAG 模式 (推荐)\n2. Direct 模式 (完整上下文)\n\n请输入选项 (1 或 2): ")
    mode = 'rag' if mode_choice == '1' else 'direct'

    # 交互式选择数据库类型
    db_choice = ''
    while db_choice not in ['1', '2']:
        db_choice = input("\n请选择数据库类型:\n1. MySQL\n2. Oracle\n\n请输入选项 (1 或 2): ")
    db_dialect = 'MySQL' if db_choice == '1' else 'Oracle'

    # 获取用户查询
    query = input(f"\n您选择了 {db_dialect} 数据库。请输入您的 SQL 查询需求: ")

    if query:
        main(query, mode, db_dialect)
    else:
        print("❌ 查询不能为空。")
