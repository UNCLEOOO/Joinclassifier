"""
POLYJOIN 推理 + 评测

支持功能：
1）纯推理（不评测，只输出每个 query 的排序结果）
2）使用单独的 ground truth 文件进行评测
3）或者使用 query 文件中每个表自带的 "ground_truth" 字段

使用示例:

1）只算 joinability 分数并输出排序结果:
    python inference.py \
        --checkpoint ./checkpoints/best_model.pt \
        --model_path all-mpnet-base-v2-local \
        --query_file eval_benchmark/queries.json \
        --lake_file eval_benchmark/data_lake.json \
        --output_file eval_benchmark/results.json

2）指定 ground truth 文件和多种 top-k 做评测:
    python inference.py \
        --checkpoint ./checkpoints/best_model.pt \
        --model_path all-mpnet-base-v2-local \
        --query_file eval_benchmark/queries.json \
        --lake_file eval_benchmark/data_lake.json \
        --ground_truth_file eval_benchmark/ground_truth.json \
        --device cuda \
        --top_k_list 1,5,10,30
"""

import os
import json
import argparse
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from polyjoin_model import POLYJOINModel


# ----------------------------------------------------------------------
# 数据加载 & 预处理
# ----------------------------------------------------------------------

def load_tables_from_json(path: str) -> List[Dict]:
    """从 JSON 文件加载表格列表"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 兼容 {"tables": [...]} 或直接是列表
    if isinstance(data, dict) and "tables" in data:
        return data["tables"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"无法识别的 JSON 格式: {path}")


def load_ground_truth(path: str) -> Dict[str, List[str]]:
    """
    从单独的 ground truth JSON 文件中加载映射:
      { "query_id_1": ["pos_id_1", "pos_id_2", ...],
        "query_id_2": [...],
        ... }

    也兼容 list 格式，例如:
      [
        {"query_id": "Q1", "positives": ["T1", "T2"]},
        {"query_id": "Q2", "positives": ["T3"]},
        ...
      ]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"ground truth 文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_map: Dict[str, List[str]] = {}

    if isinstance(data, dict):
        # 典型形式: { "query_id": [pos_ids...] }
        for qid, pos in data.items():
            if isinstance(pos, list):
                gt_map[str(qid)] = [str(x) for x in pos]
    elif isinstance(data, list):
        # 列表形式: [{"query_id": "...", "positives": [...]}, ...]
        for item in data:
            if not isinstance(item, dict):
                continue
            qid = str(item.get("query_id", ""))
            pos = item.get("positives", [])
            if qid and isinstance(pos, list):
                gt_map[qid] = [str(x) for x in pos]
    else:
        raise ValueError(f"无法识别的 ground truth 格式: {path}")

    return gt_map


def column_major_to_rows(cells: List[List[str]], col_indices: List[int]) -> List[List[str]]:
    """
    将列优先的 cells 转成行列表（只保留 col_indices 指定的列）。

    输入:
        cells: List[列]，每列是该列所有值的列表
        col_indices: 要使用的列索引列表

    返回:
        rows: List[行]，每行是 [val_col_i1, val_col_i2, ...]
    """
    if not cells or not col_indices:
        return []

    # 计算这一批 key 列的最大行数
    max_len = 0
    for idx in col_indices:
        if idx < len(cells):
            max_len = max(max_len, len(cells[idx]))

    rows: List[List[str]] = []
    for r in range(max_len):
        row_vals = []
        for idx in col_indices:
            if idx < len(cells) and r < len(cells[idx]):
                row_vals.append(cells[idx][r])
            else:
                row_vals.append("")  # 缺失值用空字符串填充
        rows.append(row_vals)
    return rows


def determine_key_indices(table: Dict, override: Optional[List[int]] = None) -> List[int]:
    """
    决定一张表用哪些列做 multi-key。

    优先级：
      1) 命令行 --key_indices 显式指定（override）
      2) 表里自带的 "join_key_indices"
      3) 默认使用所有列 [0..n_cols-1]
    """
    columns = table.get("columns", [])
    n_cols = len(columns)
    if n_cols == 0:
        return []

    if override is not None:
        idx = [i for i in override if 0 <= i < n_cols]
        return idx

    jk = table.get("join_key_indices", None)
    if isinstance(jk, list) and jk:
        idx = [int(i) for i in jk if 0 <= int(i) < n_cols]
        if idx:
            return idx

    return list(range(n_cols))


# ----------------------------------------------------------------------
# 单表编码 + 预计算缓存（GPU 上）
# ----------------------------------------------------------------------

@torch.no_grad()
def compute_table_embedding(
    model: POLYJOINModel,
    table: Dict,
    key_indices_override: Optional[List[int]],
    batch_size: int,
) -> torch.Tensor:
    """
    为单张表计算行级嵌入：
      cells 在推理数据中是【按行】存储的：
        cells[r] = [val_col0, val_col1, ..., val_colN]
    我们按 use_indices 取子列，得到行列表 rows，再交给 encode_table_for_search。
    """
    columns: List[str] = table.get("columns", [])
    cells: List[List[str]] = table.get("cells", [])
    if not columns or not cells:
        return torch.empty(0, 0, device=model.device)

    # 选 key 列索引
    use_indices = determine_key_indices(table, override=key_indices_override)
    if not use_indices:
        return torch.empty(0, 0, device=model.device)

    key_column_names = [columns[i] for i in use_indices]

    # ✅ 关键修正：cells 是“行列表”，直接按列索引取值
    rows: List[List[str]] = []
    for row in cells:  # row = [val_col0, val_col1, ..., val_colN]
        if not row:
            continue
        row_vals = []
        for idx in use_indices:
            if idx < len(row):
                row_vals.append(row[idx])
            else:
                row_vals.append("")  # 行太短时补空
        rows.append(row_vals)

    if not rows:
        # 这里就返回 0×0 就行
        return torch.empty(0, 0, device=model.device)

    # 按论文 3.3：对每个 cell 编码，再在向量维拼接
    emb = model.encode_table_for_search(key_column_names, rows, batch_size=batch_size)
    return emb


@torch.no_grad()
def precompute_all_embeddings(
    model: POLYJOINModel,
    query_tables: List[Dict],
    lake_tables: List[Dict],
    key_indices_override: Optional[List[int]],
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    """
    对 query + data lake 里出现的所有表，预计算一次嵌入并缓存（GPU 上）。

    同时在这里就做 F.normalize，后面 joinability 直接 matmul 即可。
    """
    cache: Dict[str, torch.Tensor] = {}

    all_tables: List[Dict] = []
    all_tables.extend(query_tables)
    all_tables.extend(lake_tables)

    print("\n开始预计算所有表的嵌入（只做一次，保存在 GPU 上）...")
    for tbl in tqdm(all_tables, desc="Precompute embeddings"):
        tid = str(tbl.get("table_id", "unknown"))
        if tid in cache:
            continue

        emb = compute_table_embedding(
            model=model,
            table=tbl,
            key_indices_override=key_indices_override,
            batch_size=batch_size,
        )
        if emb.numel() == 0:
            cache[tid] = emb
        else:
            cache[tid] = F.normalize(emb, dim=1)  # 提前单位化

    print(f"预计算完成，共缓存 {len(cache)} 张表的嵌入\n")
    return cache


# ----------------------------------------------------------------------
# joinability 计算 & 排序（使用缓存 + GPU）
# ----------------------------------------------------------------------

@torch.no_grad()
def compute_joinability_score_gpu(
    query_emb: torch.Tensor,
    cand_emb: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    在 GPU 上计算 joinability 分数（论文定义）

    j(T_D, T_Q) = |R_match| / |R_Q|
    其中 R_match = {x ∈ R_Q | ∃y ∈ R_D s.t. cos(h_x, h_y) > α}
    """
    if query_emb.numel() == 0 or cand_emb.numel() == 0:
        return 0.0

    # query_emb: [n_query, hidden_size], 已经 normalized
    # cand_emb: [n_cand, hidden_size], 已经 normalized
    similarity = torch.matmul(query_emb, cand_emb.T)  # [n_query, n_cand]
    max_sim = similarity.max(dim=1)[0]  # [n_query]
    
    n_matches = (max_sim > threshold).sum().item()
    n_query = query_emb.size(0)
    
    return n_matches / n_query if n_query > 0 else 0.0


def rank_joinable_tables_cached(
    emb_cache: Dict[str, torch.Tensor],
    query_table: Dict,
    lake_tables: List[Dict],
    threshold: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    使用预计算的嵌入缓存，对 data lake 中的表按 joinability 分数排序
    """
    qid = str(query_table.get("table_id", "unknown"))
    query_emb = emb_cache.get(qid)
    
    if query_emb is None or query_emb.numel() == 0:
        return []

    results: List[Tuple[str, float]] = []
    
    for lake_tbl in lake_tables:
        tid = str(lake_tbl.get("table_id", "unknown"))
        if tid == qid:
            continue
        
        cand_emb = emb_cache.get(tid)
        if cand_emb is None or cand_emb.numel() == 0:
            continue
        
        score = compute_joinability_score_gpu(query_emb, cand_emb, threshold)
        results.append((tid, score))

    # 按分数降序排序
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ----------------------------------------------------------------------
# 评测指标
# ----------------------------------------------------------------------

def evaluate_single_query(
    results: List[Tuple[str, float]],
    ground_truth: List[str],
    k_list: List[int] = [10, 30],
) -> Dict[str, float]:
    """
    对单个 query 计算 P@K, R@K, MAP@K, MRR@K
    
    论文使用 MAP@30 和 R@30 作为主要评测指标
    """
    if not ground_truth:
        return {}

    gt_set = set(str(x) for x in ground_truth)
    n_relevant = len(gt_set)
    
    if n_relevant == 0:
        return {}

    pred_ids = [str(tid) for tid, _ in results]
    max_k = len(pred_ids)
    
    if max_k == 0:
        return {f"P@{K}": 0.0 for K in k_list} | \
               {f"R@{K}": 0.0 for K in k_list} | \
               {f"MAP@{K}": 0.0 for K in k_list} | \
               {f"MRR@{K}": 0.0 for K in k_list}

    metrics: Dict[str, float] = {}

    # 预计算累积相关数和累积精度
    prefix_rel_cnt = [0] * (max_k + 1)
    p_prefix = [0.0] * (max_k + 1)
    
    rel_so_far = 0
    for i in range(1, max_k + 1):
        if pred_ids[i - 1] in gt_set:
            rel_so_far += 1
        prefix_rel_cnt[i] = rel_so_far
        p_prefix[i] = rel_so_far / i

    # 累积精度和
    p_prefix_cumsum = [0.0] * (max_k + 1)
    running = 0.0
    for i in range(1, max_k + 1):
        running += p_prefix[i]
        p_prefix_cumsum[i] = running

    for K in k_list:
        if K <= 0:
            continue
        K_eff = min(K, max_k)

        # P@K, R@K
        rel_in_topK = prefix_rel_cnt[K_eff]
        p_at_k = p_prefix[K_eff]
        r_at_k = rel_in_topK / n_relevant

        # MAP@K：论文定义 Eq.(10)
        map_at_k = p_prefix_cumsum[K_eff] / K_eff

        # MRR@K
        rr = 0.0
        for rank in range(1, K_eff + 1):
            if pred_ids[rank - 1] in gt_set:
                rr = 1.0 / rank
                break

        metrics[f"P@{K}"] = p_at_k
        metrics[f"R@{K}"] = r_at_k
        metrics[f"MAP@{K}"] = map_at_k
        metrics[f"MRR@{K}"] = rr

    return metrics


def merge_metrics(
    agg: Dict[str, float],
    curr: Dict[str, float],
) -> Dict[str, float]:
    """把单个 query 的指标累加到聚合字典中"""
    for k, v in curr.items():
        agg[k] = agg.get(k, 0.0) + float(v)
    return agg


def average_metrics(
    agg: Dict[str, float],
    n: int,
) -> Dict[str, float]:
    """对聚合后的指标除以 query 数，得到平均指标"""
    if n <= 0:
        return {}
    return {k: v / n for k, v in agg.items()}


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="POLYJOIN 推理与评测（修复版）")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="训练好的模型检查点路径 (例如 ./checkpoints/best_model.pt)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="SentenceTransformer 模型路径 (例如 all-mpnet-base-v2-local)")

    parser.add_argument("--query_file", type=str, required=True,
                        help="查询表 JSON 文件路径")
    parser.add_argument("--lake_file", type=str, required=True,
                        help="数据湖表 JSON 文件路径")
    parser.add_argument("--output_file", type=str, default="inference_results.json",
                        help="结果保存路径 (JSON)")

    # 单独的 ground truth 文件
    parser.add_argument("--ground_truth_file", type=str, default=None,
                        help="ground truth JSON 文件路径，如果提供则按此进行评测")

    parser.add_argument("--device", type=str, default=None,
                        help="设备: cuda / cpu，默认自动检测")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="预计算嵌入时 encode_table_for_search 的内部 batch 大小")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="joinability 阈值 (论文中 α=0.5)")

    # 评测系统支持多个 top-k
    parser.add_argument("--top_k_list", type=str, default="30",
                        help="逗号分隔的多个 K 值，例如 '1,5,10,30'")

    # 兼容旧用法
    parser.add_argument("--top_k", type=int, default=None,
                        help="[可选] 仅用于打印时的最大 K，默认使用 top_k_list 中的最大值")

    # 可选：如果只想用部分列作为 join key
    parser.add_argument("--key_indices", type=str, default=None,
                        help="用作 multi-key 的列索引列表，例如 '0,1,2'")

    return parser.parse_args()


def main():
    args = parse_args()

    # 设备
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 解析 key_indices
    if args.key_indices is not None and args.key_indices.strip():
        key_indices = [int(x) for x in args.key_indices.split(",")]
    else:
        key_indices = None

    # 解析 top_k_list
    k_list = [int(x) for x in args.top_k_list.split(",") if x.strip()]
    k_list = sorted(set(k_list))
    if not k_list:
        k_list = [30]
    k_max = max(k_list)
    print_top_k = args.top_k if args.top_k is not None else k_max

    # 加载数据
    print(f"\n加载查询表: {args.query_file}")
    query_tables = load_tables_from_json(args.query_file)
    print(f"查询表数量: {len(query_tables)}")

    print(f"\n加载数据湖表: {args.lake_file}")
    lake_tables = load_tables_from_json(args.lake_file)
    print(f"数据湖表数量: {len(lake_tables)}")

    # 加载 ground truth
    gt_map: Dict[str, List[str]] = {}
    if args.ground_truth_file:
        print(f"\n加载 ground truth 文件: {args.ground_truth_file}")
        gt_map = load_ground_truth(args.ground_truth_file)
        print(f"ground truth 中包含 {len(gt_map)} 个 query 条目")
    else:
        print("\n未提供 ground truth 文件，如 query JSON 内有 'ground_truth' 字段仍会尝试使用。")

    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    model = POLYJOINModel(
        model_path=args.model_path,
        device=str(device),
    )
    model.to(device)

    print(f"从检查点加载权重: {args.checkpoint}")
    _ = model.load_checkpoint(args.checkpoint, device=str(device), load_clustering=False)
    model.eval()

    # 预计算所有表的嵌入
    emb_cache = precompute_all_embeddings(
        model=model,
        query_tables=query_tables,
        lake_tables=lake_tables,
        key_indices_override=key_indices,
        batch_size=args.batch_size,
    )

    all_results: Dict[str, List[Tuple[str, float]]] = {}
    per_query_metrics: Dict[str, Dict[str, float]] = {}
    agg_metrics: Dict[str, float] = {}
    n_queries_with_gt = 0

    print("\n开始对每个查询表计算 joinability 排序...")

    for q_tbl in tqdm(query_tables, desc="Queries"):
        qid = str(q_tbl.get("table_id", "unknown"))

        # 使用缓存的嵌入计算
        results = rank_joinable_tables_cached(
            emb_cache=emb_cache,
            query_table=q_tbl,
            lake_tables=lake_tables,
            threshold=args.threshold,
        )
        all_results[qid] = results

        # 获取 ground truth
        gt: Optional[List[str]] = gt_map.get(qid)
        if gt is None:
            gt = q_tbl.get("ground_truth", None)

        if gt:
            metrics = evaluate_single_query(results, gt, k_list=k_list)
            if metrics:
                per_query_metrics[qid] = metrics
                agg_metrics = merge_metrics(agg_metrics, metrics)
                n_queries_with_gt += 1

        # 打印前若干结果
        print(f"\nQuery table: {qid}")
        print(f"Top-{min(print_top_k, len(results))} 结果:")
        for i, (tid, score) in enumerate(results[:print_top_k], 1):
            print(f"  {i}. {tid}: {score:.4f}")

        # 打印本 query 的评测指标
        if gt and qid in per_query_metrics:
            print("评估指标:")
            for k in k_list:
                p = per_query_metrics[qid].get(f"P@{k}", 0.0)
                r = per_query_metrics[qid].get(f"R@{k}", 0.0)
                m = per_query_metrics[qid].get(f"MAP@{k}", 0.0)
                rr = per_query_metrics[qid].get(f"MRR@{k}", 0.0)
                print(
                    f"  P@{k}: {p:.4f}, R@{k}: {r:.4f}, "
                    f"MAP@{k}: {m:.4f}, MRR@{k}: {rr:.4f}"
                )

    # 计算全局平均指标
    avg_metrics = average_metrics(agg_metrics, n_queries_with_gt)
    if n_queries_with_gt > 0:
        print("\n================ 全局平均指标 ================")
        print(f"参与评测的 query 数量: {n_queries_with_gt}")
        for k in k_list:
            p = avg_metrics.get(f"P@{k}", 0.0)
            r = avg_metrics.get(f"R@{k}", 0.0)
            m = avg_metrics.get(f"MAP@{k}", 0.0)
            rr = avg_metrics.get(f"MRR@{k}", 0.0)
            print(
                f"  P@{k}: {p:.4f}, R@{k}: {r:.4f}, "
                f"MAP@{k}: {m:.4f}, MRR@{k}: {rr:.4f}"
            )
        print("=============================================")
    else:
        print("\n[提示] 没有找到任何 query 的 ground truth，未计算评测指标。")

    # 保存结果
    output = {
        "results": {
            qid: [(tid, float(s)) for tid, s in v]
            for qid, v in all_results.items()
        },
        "per_query_metrics": per_query_metrics,
        "avg_metrics": avg_metrics,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n结果保存到: {args.output_file}")


if __name__ == "__main__":
    main()
