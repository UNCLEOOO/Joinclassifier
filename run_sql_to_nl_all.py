#!/usr/bin/env python3
"""
非交互式：处理所有表，生成NL Query
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import pandas as pd
from openai import OpenAI
import time

# API配置
client = OpenAI(
    api_key="sk-q9Qhk8KvRqkBQ6Ti6d971dBbCbCd491b8b995325918cA471",
    base_url="https://aihubmix.com/v1"
)

# 导入主脚本的函数
import sql_to_nl_query as main_script

# 更新全局client
main_script.client = client

def main_non_interactive():
    """非交互式处理所有表"""
    print("=" * 80)
    print("SQL转自然语言查询（NL Query）- 处理所有表")
    print("=" * 80)
    print(f"\nAPI: AiHubMix (gpt-4o)")

    # 读取数据
    print("\n加载数据...")
    with open(main_script.FILTERED_QUERIES_FILE, 'r') as f:
        filtered_queries = json.load(f)

    with open(main_script.ANALYSIS_REPORT_FILE, 'r') as f:
        analysis_report = json.load(f)

    tables_to_process = list(filtered_queries.keys())
    print(f"将处理 {len(tables_to_process)} 个表 (约 {len(tables_to_process) * 5} 个查询)")

    # 处理所有查询
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
                result = main_script.process_single_query(table_name, query_info, analysis_report)
                if result:
                    all_results.append(result)
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"    错误: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
                continue

        # 每10个表保存一次
        if i % 10 == 0:
            print(f"\n*** 中间保存 (已处理 {i} 个表) ***")
            with open(main_script.OUTPUT_FILE, 'w') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 保存最终结果
    print(f"\n{'='*80}")
    print("保存最终结果...")

    with open(main_script.OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"✓ NL查询已保存到: {main_script.OUTPUT_FILE}")

    # 打印统计
    print(f"\n{'='*80}")
    print("处理统计:")
    print(f"{'='*80}")
    print(f"成功: {successful} 个查询")
    print(f"失败: {failed} 个查询")
    print(f"\n输出文件: {main_script.OUTPUT_FILE}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main_non_interactive()
