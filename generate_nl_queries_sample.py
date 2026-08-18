#!/usr/bin/env python3
"""
生成自然语言查询 - 示例版本
从combined_queries.json中选择10个查询生成NL描述
"""

import json
import random
from openai import OpenAI

# API配置
API_KEY = "sk-VTOqNlWBMvz6Hg5CK6uwUhctIRGpA5TZ0eGJ0KaVIyLtZwTD"
BASE_URL = "https://www.dmxapi.com/v1"
MODEL = "gpt-4o"

# 路径配置
QUERIES_FILE = '/data2/liujinqi/Revision/SQL_generation/combined_queries.json'
OUTPUT_FILE = '/data2/liujinqi/Revision/SQL_generation/nl_queries_sample.json'

# 初始化API客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def sql_to_nl(sql: str, table_name: str, selected_columns: list) -> str:
    """使用LLM将SQL转换为自然语言查询"""

    prompt = f"""You are a data analyst. Convert the following SQL query into a natural language question that a user might ask.

Table: {table_name}
SQL Query:
{sql}

Selected Columns: {', '.join(selected_columns)}

Requirements:
1. Generate a clear, natural English question that this SQL query answers
2. The question should be specific and mention the key information being retrieved
3. Keep it concise (1-2 sentences)
4. Don't mention technical SQL terms like JOIN, SELECT, etc.
5. Focus on what business question is being answered

Natural Language Question:"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful data analyst who converts SQL queries to natural language questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )

        nl_query = response.choices[0].message.content.strip()
        return nl_query

    except Exception as e:
        print(f"  Error calling LLM: {e}")
        return f"Error: {str(e)}"


def select_diverse_queries(all_queries_dict, num_samples=10):
    """选择多样化的查询样本"""
    selected = []

    # 按表分组
    table_queries = {}
    for table_name, table_data in all_queries_dict.items():
        queries = table_data.get('queries', [])
        if queries:
            table_queries[table_name] = queries

    # 从不同表中选择
    table_names = list(table_queries.keys())
    random.seed(42)  # 固定随机种子以便复现

    # 随机选择表
    sampled_tables = random.sample(table_names, min(num_samples, len(table_names)))

    for table_name in sampled_tables:
        queries = table_queries[table_name]
        # 从每个表随机选一个查询
        query = random.choice(queries)
        selected.append({
            'table_name': table_name,
            'query': query
        })

    return selected[:num_samples]


def main():
    print("=" * 80)
    print("生成自然语言查询 - 示例版本")
    print("=" * 80)

    # 加载查询
    print("\n加载查询数据...")
    with open(QUERIES_FILE, 'r') as f:
        all_queries = json.load(f)

    total_queries = sum(len(v['queries']) for v in all_queries.values())
    print(f"✓ 加载了 {len(all_queries)} 个表，共 {total_queries} 个查询")

    # 选择10个样本
    print("\n选择10个多样化的查询样本...")
    samples = select_diverse_queries(all_queries, num_samples=10)
    print(f"✓ 选择了 {len(samples)} 个样本")

    # 生成NL查询
    print("\n生成自然语言查询...")
    results = []

    for i, sample in enumerate(samples, 1):
        table_name = sample['table_name']
        query = sample['query']
        sql = query['sql']
        selected_columns = query['selected_columns']

        print(f"\n[{i}/{len(samples)}] {table_name}")
        print(f"  SQL: {sql[:80]}...")

        nl_query = sql_to_nl(sql, table_name, selected_columns)
        print(f"  NL:  {nl_query}")

        results.append({
            'id': i,
            'table_name': table_name,
            'query_id': query['query_id'],
            'template_source': query['template_source'],
            'template_difficulty': query['template_difficulty'],
            'selected_columns': selected_columns,
            'sql': sql,
            'nl_query': nl_query
        })

    # 保存结果
    print(f"\n保存结果到 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("完成！")
    print(f"生成了 {len(results)} 个自然语言查询")
    print(f"输出文件: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == '__main__':
    main()
