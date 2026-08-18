#!/usr/bin/env python3
"""
将SQL转换为自然语言查询（NL Query）
使用LLM分析SQL和执行结果，生成自然、完整的人类查询语句
"""

import json
import os
import pandas as pd
from openai import OpenAI
import time
from typing import Dict, Any

# API配置 - 使用dmxapi
client = OpenAI(
    api_key="sk-VTOqNlWBMvz6Hg5CK6uwUhctIRGpA5TZ0eGJ0KaVIyLtZwTD",
    base_url="https://www.dmxapi.com/v1"
)
MODEL = "gpt-5.1"

# 路径配置
FILTERED_QUERIES_FILE = '/data2/liujinqi/Revision/SQL_generation/filtered_queries.json'
FILTERED_RESULTS_DIR = '/data2/liujinqi/Revision/SQL_generation/filtered_results'
ANALYSIS_REPORT_FILE = '/data2/liujinqi/Revision/SQL_generation/column_analysis_report.json'
OUTPUT_FILE = '/data2/liujinqi/Revision/SQL_generation/nl_queries.json'


def sample_result_data(csv_path: str, num_rows: int = 3) -> str:
    """从CSV结果中采样数据"""
    try:
        df = pd.read_csv(csv_path)

        if len(df) == 0:
            return "Empty result set"

        # 采样前几行
        sample_size = min(num_rows, len(df))
        sample_df = df.head(sample_size)

        # 格式化为字符串
        result = f"Result has {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n"
        result += f"Sample data (first {sample_size} rows):\n"
        result += sample_df.to_string(index=False, max_colwidth=50)

        return result
    except Exception as e:
        return f"Error reading result: {e}"


def get_column_meanings(table_name: str, query_id: int, analysis_report: list) -> Dict[str, str]:
    """从分析报告中获取列的含义"""
    for item in analysis_report:
        if item['table_name'] == table_name and item['query_id'] == query_id:
            return item.get('column_analyses', {})
    return {}


def sql_to_nl_query(
    sql: str,
    result_sample: str,
    column_meanings: Dict[str, str],
    selected_columns: list,
    reasoning: str
) -> str:
    """使用LLM将SQL转换为自然语言查询"""

    # 分析SQL的实际操作
    sql_upper = sql.upper()
    has_where = 'WHERE' in sql_upper
    has_group_by = 'GROUP BY' in sql_upper
    has_order_by = 'ORDER BY' in sql_upper
    has_limit = 'LIMIT' in sql_upper
    has_join = 'JOIN' in sql_upper

    # 提取WHERE条件（如果有）
    where_clause = ""
    if has_where:
        import re
        where_match = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1).strip()[:200]

    # 简化列名为通俗英语
    simplified_columns = []
    for col in selected_columns:
        meaning = column_meanings.get(col, col)
        # 提取关键词，去掉冗长描述
        if len(meaning) > 50:
            meaning = meaning.split('.')[0].split(',')[0]
        simplified_columns.append(meaning.strip())

    columns_list = ', '.join(simplified_columns)

    prompt = f"""Given the columns below, write a natural English query that covers all of them.

- Sound like a real person asking for data, not a machine
- Vary question types, connectors, and sentence structures
- Simplify column names to plain English
- Keep it brief and conversational

Columns: {columns_list}

Return ONLY the question."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert at writing natural, specific user questions. Be concise and concrete."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=100
        )

        nl_query = response.choices[0].message.content.strip()
        nl_query = nl_query.strip('"\'')

        return nl_query

    except Exception as e:
        print(f"  Error generating NL query: {e}")
        return f"Error: {e}"


def process_single_query(
    table_name: str,
    query_info: Dict[str, Any],
    analysis_report: list
) -> Dict[str, Any]:
    """处理单个查询，生成NL query"""

    query_id = query_info['query_id']
    template_source = query_info['template_source']
    template_difficulty = query_info['template_difficulty']

    print(f"\n  Query {query_id} ({template_source} - {template_difficulty})")

    # 构建CSV文件路径
    csv_filename = f"{table_name}_{template_source}_{template_difficulty}_q{query_id}.csv"
    csv_path = os.path.join(FILTERED_RESULTS_DIR, csv_filename)

    if not os.path.exists(csv_path):
        print(f"    跳过：CSV文件不存在")
        return None

    # 读取结果样本
    result_sample = sample_result_data(csv_path)

    # 获取列含义
    column_meanings = get_column_meanings(table_name, query_id, analysis_report)

    # 生成NL query
    nl_query = sql_to_nl_query(
        query_info['sql'],
        result_sample,
        column_meanings,
        query_info['selected_columns'],
        query_info['reasoning']
    )

    print(f"    NL Query: {nl_query[:80]}...")

    time.sleep(0.3)  # 避免API限流

    return {
        'table_name': table_name,
        'query_id': query_id,
        'template_source': template_source,
        'template_difficulty': template_difficulty,
        'sql': query_info['sql'],
        'selected_columns': query_info['selected_columns'],
        'nl_query': nl_query,
        'csv_filename': csv_filename
    }


def main():
    """主函数"""
    print("=" * 80)
    print("SQL转自然语言查询（NL Query）")
    print("=" * 80)
    print(f"\nAPI: dmxapi ({MODEL})")

    # 读取数据
    print("\n加载数据...")
    with open(FILTERED_QUERIES_FILE, 'r') as f:
        filtered_queries = json.load(f)

    with open(ANALYSIS_REPORT_FILE, 'r') as f:
        analysis_report = json.load(f)

    total_tables = len(filtered_queries)
    print(f"总表数: {total_tables}")

    # 自动选择：处理所有表
    print("\n自动选择: 处理所有表")
    choice = '1'

    tables_to_process = []

    if choice == '1':
        tables_to_process = list(filtered_queries.keys())
    elif choice == '2':
        n = int(input("请输入要处理的表数量: "))
        tables_to_process = list(filtered_queries.keys())[:n]
    elif choice == '3':
        table_name = input("请输入表名: ").strip()
        if table_name in filtered_queries:
            tables_to_process = [table_name]
        else:
            print(f"错误: 表 {table_name} 不存在")
            return
    else:
        print("无效选择")
        return

    print(f"\n将处理 {len(tables_to_process)} 个表")
    # 自动确认
    confirm = 'y'
    if confirm != 'y':
        print("已取消")
        return

    # 处理每个查询
    all_results = []
    successful = 0
    failed = 0

    for i, table_name in enumerate(tables_to_process, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(tables_to_process)}] 表: {table_name}")
        print(f"{'='*80}")

        table_info = filtered_queries[table_name]

        for query_info in table_info['queries']:
            try:
                result = process_single_query(table_name, query_info, analysis_report)
                if result:
                    all_results.append(result)
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"    错误: {e}")
                failed += 1
                continue

    # 保存结果
    print(f"\n{'='*80}")
    print("保存结果...")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"✓ NL查询已保存到: {OUTPUT_FILE}")

    # 打印统计
    print(f"\n{'='*80}")
    print("处理统计:")
    print(f"{'='*80}")
    print(f"成功: {successful} 个查询")
    print(f"失败: {failed} 个查询")
    print(f"\n输出文件: {OUTPUT_FILE}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
