# rag_sql.py
import faiss, json, os, numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI  # 也可换成 deepseek 等
import os

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


def build_prompt(query, hits, table_definitions):
    context_blocks = []
    for tbl, _ in hits:
        info = table_definitions[tbl]
        # 只放关键信息，避免 token 过大
        block = (
            f"表名: {tbl}（{info['tableChiName']}）\n"
            f"说明: {info['description']}\n"
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
- SQL 必须使用 ANSI SQL-92 语法。
- 必要时请合理 JOIN，JOIN 条件优先使用主键/外键 InnerCode 等。
- 只返回 SQL 代码，不要附带解释。
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
3.  **SQL 语法：** 生成的 SQL 必须是 ANSI SQL-92 兼容的。
4.  **JOIN 条件：** 必要时请合理 JOIN，JOIN 条件优先使用主键/外键 InnerCode 等。
5.  **输出格式：** 只返回 SQL 代码，不要附带任何解释或额外文字。
"""},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    return resp.choices[0].message.content.strip()


def main(query=None):
    index, names, embedder = load_resources()
    with open(CONFIG["json_path"].replace("table_definitions", "table_definitions"), "r", encoding="utf-8") as f:
        table_defs = json.load(f)

    if query is None:
        query = input("请输入 SQL 查询需求: ")
    hits = search_tables(query, index, embedder, names, CONFIG["top_k"])
    print("🔍 Top-k 检索结果:", hits)

    prompt = build_prompt(query, hits, table_defs)
    sql = call_llm(prompt)
    print("\n=== 生成的 SQL ===\n", sql)


if __name__ == "__main__":
    main()
