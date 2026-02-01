import os
import pandas as pd
import numpy as np
# 注释掉单卡限制，允许使用所有可用GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# 禁用 tokenizer 并行以避免多进程冲突
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from typing import List, Tuple, Dict, Optional, Any
import time
import logging
import argparse
import json
import random
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import heapq

# ------------------------------
# 日志配置
# ------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
torch.use_deterministic_algorithms(True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('JoinClassifier')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")

# ------------------------------
# 数据结构定义
# ------------------------------

class Table:
    def __init__(self, table_name: str, columns: Dict[str, List], unique_id: str = None,
                 base_id: str = None, partition: str = None, join_key_indices: List[int] = None):
        self.table_name = table_name
        self.unique_id = unique_id if unique_id else table_name
        self.table_id = self.unique_id  # Alias for compatibility with JOINnow format
        self.base_id = base_id if base_id else self._infer_base_id(self.unique_id)
        self.partition = partition if partition else self._infer_partition(self.unique_id)
        self.columns = columns
        self.col_names = list(columns.keys())
        self.columns_list = self.col_names  # Alias for compatibility
        self.join_key_indices = join_key_indices if join_key_indices else []

        # Build cells structure for compatibility
        self.cells = self._build_cells()

    def _infer_base_id(self, table_id: str) -> str:
        """Infer base_id from table_id"""
        if "-" in table_id:
            return table_id.split("-")[0]
        if "_" in table_id:
            return table_id.split("_")[0]
        return table_id

    def _infer_partition(self, table_id: str) -> str:
        """Infer partition from table_id"""
        parts = table_id.split("-")
        if len(parts) >= 2 and parts[1] in ("top", "middle", "bottom"):
            return parts[1]
        return "unknown"

    def _build_cells(self) -> List[List[Any]]:
        """Build cells structure from columns dict"""
        if not self.columns:
            return []
        num_rows = len(next(iter(self.columns.values())))
        cells = []
        for i in range(num_rows):
            row = [self.columns[col][i] if i < len(self.columns[col]) else None
                   for col in self.col_names]
            cells.append(row)
        return cells

    def get_column_values_by_name(self, name: str) -> List[Any]:
        """Get column values by name"""
        return self.columns.get(name, [])

    def get_column_values_by_idx(self, idx: int) -> List[Any]:
        """Get column values by index"""
        if 0 <= idx < len(self.col_names):
            return self.columns[self.col_names[idx]]
        return []

    def __repr__(self):
        row_count = len(next(iter(self.columns.values()))) if self.columns else 0
        return f"Table({self.unique_id}, {len(self.columns)}列, {row_count}行)"

class BIModel:
    def __init__(self, tables: List[Table], joins: List[Tuple[Tuple[Table, str], Tuple[Table, str]]]):
        self.tables = tables
        self.joins = joins
    def __repr__(self):
        return f"BIModel({len(self.tables)}表, {len(self.joins)}连接)"

# ------------------------------
# 数据加载 - JSON格式支持
# ------------------------------

def load_tables_json(path: str) -> Dict[str, Table]:
    """
    从JSON文件加载表（JOINnow格式）

    Args:
        path: JSON文件路径

    Returns:
        表字典 {table_id: Table}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tables = {}
    for t in data:
        table_id = t["table_id"]
        columns_list = t["columns"]
        cells = t["cells"]
        join_key_indices = list(t.get("join_key_indices", []))
        base_id = t.get("base_id", None)
        partition = t.get("partition", None)

        # Convert cells to columns dict
        columns = {}
        for col_idx, col_name in enumerate(columns_list):
            col_values = []
            for row in cells:
                if col_idx < len(row):
                    col_values.append(row[col_idx])
                else:
                    col_values.append(None)
            columns[col_name] = col_values

        tables[table_id] = Table(
            table_name=table_id,
            columns=columns,
            unique_id=table_id,
            base_id=base_id,
            partition=partition,
            join_key_indices=join_key_indices
        )

    logger.info(f"从JSON加载了 {len(tables)} 张表")
    return tables


def load_ground_truth_json(path: str) -> Dict[str, List[str]]:
    """
    从JSON文件加载ground truth（JOINnow格式）

    Args:
        path: JSON文件路径

    Returns:
        Ground truth字典 {query_key: [candidate_keys]}
    """
    with open(path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    return {str(k): [str(x) for x in v] for k, v in gt.items()}


def parse_column_pair_from_key(key: str) -> Tuple[str, str]:
    """
    从key解析表ID和列名
    格式: "table_id::column_name"

    Args:
        key: 列对key

    Returns:
        (table_id, column_name)
    """
    if "::" in key:
        parts = key.split("::", 1)
        return parts[0], parts[1]
    else:
        # 兼容旧格式
        return key, ""


def build_training_samples_from_json(
    ground_truth: Dict[str, List[str]],
    tables: Dict[str, Table]
) -> List[Tuple[Tuple[str, str], Tuple[str, str], int]]:
    """
    从JSON格式的ground truth构建训练样本（表级别）

    Ground truth 格式: {"query_table_id": ["candidate_table_id1", ...]}
    只使用 join_key_indices 指定的列进行配对

    Args:
        ground_truth: Ground truth字典（表级别）
        tables: 表字典

    Returns:
        训练样本列表 [((table_id1, col1), (table_id2, col2), label)]
    """
    samples = []

    for query_table_id, candidate_table_ids in ground_truth.items():
        if query_table_id not in tables:
            continue

        query_table = tables[query_table_id]

        # 获取查询表的连接键列
        query_join_cols = []
        if query_table.join_key_indices:
            for idx in query_table.join_key_indices:
                if idx < len(query_table.col_names):
                    query_join_cols.append(query_table.col_names[idx])

        if not query_join_cols:
            continue

        # 对每个候选表
        for cand_table_id in candidate_table_ids:
            if cand_table_id not in tables:
                continue

            cand_table = tables[cand_table_id]

            # 获取候选表的连接键列
            cand_join_cols = []
            if cand_table.join_key_indices:
                for idx in cand_table.join_key_indices:
                    if idx < len(cand_table.col_names):
                        cand_join_cols.append(cand_table.col_names[idx])

            if not cand_join_cols:
                continue

            # 创建连接键列之间的配对样本（只配对序号一致的列）
            # 例如：query[0] ↔ candidate[0], query[1] ↔ candidate[1]
            min_len = min(len(query_join_cols), len(cand_join_cols))
            for i in range(min_len):
                query_col = query_join_cols[i]
                cand_col = cand_join_cols[i]
                samples.append(((query_table_id, query_col), (cand_table_id, cand_col), 1))

    logger.info(f"从JSON ground truth构建了 {len(samples)} 个正样本")
    return samples


# ------------------------------
# 数据加载 - CSV格式支持（保持向后兼容）
# ------------------------------

def normalize_table_name(name: str) -> str:
    return name.replace('\\', '/')

def parse_table_identifier(name: str) -> Tuple[str, str]:
    name = normalize_table_name(name)
    if '/' in name:
        parts = name.rsplit('/', 1)
        return (parts[0], parts[1])
    return (None, name)

def load_tables_from_dir(table_dir: str) -> Dict[str, Table]:
    tables = {}
    table_path = Path(table_dir)
    if not table_path.exists():
        logger.error(f"表目录不存在: {table_path}")
        return tables
    csv_files = list(table_path.glob("*.csv"))
    logger.info(f"找到 {len(csv_files)} 个CSV文件")
    for filepath in csv_files:
        try:
            df = pd.read_csv(filepath)
            table_name = filepath.stem
            columns = {col.strip(): df[col].tolist() for col in df.columns}
            tables[table_name] = Table(table_name, columns, unique_id=table_name)
        except Exception as e:
            logger.error(f"加载表{filepath.name}失败: {e}")
    logger.info(f"加载完成: {len(tables)} 张表")
    return tables

def load_tables_from_folder_structure(table_dir: str) -> Dict[str, Table]:
    tables = {}
    table_path = Path(table_dir)
    if not table_path.exists():
        logger.error(f"表目录不存在: {table_path}")
        return tables

    subfolders = [f for f in table_path.iterdir() if f.is_dir()]
    logger.info(f"找到 {len(subfolders)} 个子文件夹")

    total_csv_count = 0
    for subfolder in subfolders:
        folder_name = subfolder.name
        csv_files = list(subfolder.glob("*.csv"))
        for filepath in csv_files:
            try:
                df = pd.read_csv(filepath)
                csv_name = filepath.stem
                unique_id = f"{folder_name}/{csv_name}"
                columns = {col: df[col].tolist() for col in df.columns}
                tables[unique_id] = Table(table_name=csv_name, columns=columns, unique_id=unique_id)
                total_csv_count += 1
            except Exception as e:
                logger.error(f"加载表 {filepath} 失败: {e}")
    logger.info(f"加载完成: 共 {total_csv_count} 张表")
    return tables

def load_tables(table_dir: str, isfolder: bool = False) -> Dict[str, Table]:
    if isfolder:
        return load_tables_from_folder_structure(table_dir)
    else:
        return load_tables_from_dir(table_dir)

def find_table_by_name(tables: Dict[str, Table], name: str) -> Optional[Table]:
    name = normalize_table_name(name)
    if name in tables: return tables[name]
    if name.endswith('.csv'):
        name_no_ext = name[:-4]
        if name_no_ext in tables: return tables[name_no_ext]
    folder_name, csv_name = parse_table_identifier(name)
    if folder_name:
        candidate_key = f"{folder_name}/{csv_name}"
        if candidate_key in tables: return tables[candidate_key]
        if csv_name.endswith('.csv'):
            candidate_key = f"{folder_name}/{csv_name[:-4]}"
            if candidate_key in tables: return tables[candidate_key]
    return None

def load_links_csv(links_file: str, tables: Dict[str, Table]) -> BIModel:
    try:
        links_path = Path(links_file)
        if not links_path.exists():
            return BIModel([], [])

        links_df = pd.read_csv(links_path, header=None)
        tables_in_model = set()
        joins = []

        for idx, row in links_df.iterrows():
            try:
                t1n = normalize_table_name(str(row[0]).strip())
                t2n = normalize_table_name(str(row[1]).strip())
                if t1n.endswith('.csv'): t1n = t1n[:-4]
                if t2n.endswith('.csv'): t2n = t2n[:-4]

                c1n = str(row[2]).strip()
                c2n = str(row[3]).strip()

                t1 = find_table_by_name(tables, t1n)
                t2 = find_table_by_name(tables, t2n)

                if t1 and t2:
                    tables_in_model.add(t1.unique_id)
                    tables_in_model.add(t2.unique_id)
                    if c1n in t1.columns and c2n in t2.columns:
                        joins.append(((t1, c1n), (t2, c2n)))
            except Exception as e:
                logger.error(f"处理连接关系文件第{idx+1}行时出错: {e}")
                continue

        table_objs = [tables[name] for name in tables_in_model if name in tables]
        return BIModel(table_objs, joins)
    except Exception as e:
        logger.error(f"加载连接关系失败: {e}")
        return BIModel([], [])

# ------------------------------
# JoinClassifier 实现
# ------------------------------

class JoinClassifier:
    """
    JoinClassifier: 基于预训练语言模型的可连接表发现
    升级自 DeepJoin (VLDB 2023)
    """

    def __init__(self):
        """
        初始化 JoinClassifier 分类器
        使用本地预训练的 MPNet 模型，自动检测并使用多GPU
        """
        # 自动检测设备
        if torch.cuda.is_available():
            device = 'cuda'
            self.num_gpus = torch.cuda.device_count()
            logger.info(f"检测到 {self.num_gpus} 个可用GPU")
        else:
            device = 'cpu'
            self.num_gpus = 0
            logger.info("未检测到GPU，使用CPU")

        # 尝试加载本地模型
        local_model_path = '/home/zhaohangyu/HuaweiJoin/polyjointest/all-mpnet-base-v2-local'
        try:
            self.model = SentenceTransformer(local_model_path, device=device)
            logger.info(f"已加载本地模型: {local_model_path}")
        except Exception:
            logger.warning("本地模型加载失败，从HuggingFace加载模型")
            self.model = SentenceTransformer(local_model_path, device=device)

        # 设置最大序列长度并强制截断
        self.max_seq_length = 384
        self.model.max_seq_length = self.max_seq_length

        # 强制设置 tokenizer 的截断参数
        if hasattr(self.model, 'tokenizer'):
            self.model.tokenizer.model_max_length = self.max_seq_length

        # 设置所有 transformer 模块的最大序列长度
        for module in self.model.modules():
            if hasattr(module, 'max_seq_length'):
                module.max_seq_length = self.max_seq_length

        logger.info(f"模型最大序列长度: {self.max_seq_length} (已强制截断)")

        if self.num_gpus > 1:
            logger.info(f"多GPU训练已启用，将使用 {self.num_gpus} 个GPU进行并行训练")

        self.logger = logging.getLogger('JoinClassifier')
        self.column_embeddings = {}  # 存储列嵌入
        self.column_text_cache = {}  # 存储列序列化文本缓存 {(table_id, col_name, freq_hash): text}

    def build_global_frequency_from_json(self, all_tables_json_path: str) -> Dict[str, int]:
        """
        从所有表的 JSON 文件构建全局值频率字典（仅统计连接键列）

        Args:
            all_tables_json_path: 所有表的 JSON 文件路径

        Returns:
            值频率字典 {value: frequency}
        """
        logger.info(f"从 {all_tables_json_path} 构建全局值频率字典（仅统计连接键列）...")

        # 加载所有表
        all_tables = load_tables_json(all_tables_json_path)

        freq_dict = Counter()

        # 只统计连接键列的值频率
        for table in all_tables.values():
            if table.join_key_indices:
                for idx in table.join_key_indices:
                    if idx < len(table.col_names):
                        col_name = table.col_names[idx]
                        col_data = table.columns[col_name]
                        # 统计值频率
                        str_values = [str(x) for x in col_data if pd.notna(x)]
                        freq_dict.update(str_values)

        logger.info(f"全局值频率字典构建完成，共 {len(freq_dict)} 个不同值")
        return dict(freq_dict)

    def _column_to_text(self, table: Table, column_name: str, pattern='title-colname-stat-col',
                       datalake_freq: Dict[str, int] = None) -> str:
        """
        将列转换为文本序列（带缓存优化）

        Args:
            table: 表对象
            column_name: 列名
            pattern: 转换模式
            datalake_freq: 数据湖值频率字典（用于排序采样）

        Returns:
            文本序列
        """
        # 构建缓存键
        # 使用 id(datalake_freq) 作为频率字典的标识（同一个字典对象在内存中的地址）
        freq_id = id(datalake_freq) if datalake_freq is not None else None
        cache_key = (table.unique_id, column_name, pattern, freq_id)

        # 检查缓存
        if cache_key in self.column_text_cache:
            return self.column_text_cache[cache_key]

        # 缓存未命中，计算文本
        column_data = table.columns[column_name]

        # 采样频繁值（可以基于数据湖频率）
        sampled_values = self._sample_frequent_values(column_data, datalake_freq=datalake_freq)

        # 计算统计信息
        n = len(set(column_data))
        value_lengths = [len(str(x)) for x in column_data if pd.notna(x)]
        max_len = max(value_lengths) if value_lengths else 0
        min_len = min(value_lengths) if value_lengths else 0
        avg_len = sum(value_lengths) / len(value_lengths) if value_lengths else 0

        # 根据模式生成文本
        if pattern == 'col':
            text = ', '.join(sampled_values)
        elif pattern == 'colname-col':
            text = f"{column_name}: {', '.join(sampled_values)}"
        elif pattern == 'colname-stat-col':
            text = f"{column_name} contains {n} values ({max_len}, {min_len}, {avg_len:.1f}): {', '.join(sampled_values)}"
        elif pattern == 'title-colname-col':
            text = f"{table.table_name}. {column_name}: {', '.join(sampled_values)}"
        elif pattern == 'title-colname-stat-col':
            text = f"{column_name} contains {n} values ({max_len}, {min_len}, {avg_len:.1f}): {', '.join(sampled_values)}"
        else:
            # 默认使用最佳模式
            text = f"{table.unique_id}. {column_name} contains {n} values ({max_len}, {min_len}, {avg_len:.1f}): {', '.join(sampled_values)}"

        # 存入缓存
        self.column_text_cache[cache_key] = text
        return text

    def _sample_frequent_values(self, column: List, max_tokens=None, datalake_freq: Dict[str, int] = None) -> List[str]:
        """
        采样最频繁的单元格值

        Args:
            column: 列数据
            max_tokens: 最大token数
            datalake_freq: 数据湖值频率字典（如果提供，按数据湖频率排序）

        Returns:
            采样的值列表
        """
        if max_tokens is None:
            # 预留更多空间给表名、列名、统计信息（约150 tokens）
            # 这样可以确保最终序列不超过 384 tokens
            max_tokens = self.max_seq_length - 150

        # 统计频率
        str_values = [str(x) for x in column if pd.notna(x)]
        if not str_values:
            return []

        value_freq = Counter(str_values)

        # 根据是否提供数据湖频率，选择排序策略
        if datalake_freq is not None:
            # 按数据湖频率排序（频率越高越靠前）
            distinct_values = sorted(
                list(value_freq.keys()),
                key=lambda x: datalake_freq.get(x, 0),
                reverse=True
            )
        else:
            # 按列内部频率排序
            distinct_values = sorted(
                list(value_freq.keys()),
                key=lambda x: value_freq[x],
                reverse=True
            )

        sampled_values = []
        current_tokens = 0
        tokenizer = self.model.tokenizer

        for value in distinct_values:
            # 限制单个值长度（更保守）
            if len(value) > 200:
                value = value[:200]

            tokens = tokenizer.tokenize(value)
            token_count = len(tokens)

            # 如果单个值就超过限制，截断并停止
            if token_count > max_tokens:
                if not sampled_values:  # 如果还没有任何值，至少添加一个截断的值
                    sampled_values.append(value[:max_tokens * 2])
                break

            if sampled_values:
                token_count += 1  # 逗号分隔符

            if current_tokens + token_count <= max_tokens:
                sampled_values.append(value)
                current_tokens += token_count
            else:
                break

        return sampled_values

    def _shuffle_column(self, column: List) -> List:
        """随机打乱列中的单元格顺序（用于数据增强）"""
        shuffled = column.copy()
        random.shuffle(shuffled)
        return shuffled

    def train(self, bimodels: List[BIModel] = None, batch_size=8, epochs=10, shuffle_rate=0.3,
              training_samples: List[Tuple[Tuple[str, str], Tuple[str, str], int]] = None,
              tables_dict: Dict[str, Table] = None,
              all_tables_json_path: str = None):
        """
        训练 JoinClassifier 模型（支持CSV和JSON格式，使用in-batch负样本）

        Args:
            bimodels: BI模型列表（包含正样本）- CSV格式
            batch_size: 批大小（默认8，因为Multiple Negatives Ranking Loss显存占用为O(batch_size²)）
            epochs: 训练轮数
            shuffle_rate: 数据增强中打乱列的比例
            training_samples: 训练样本列表 - JSON格式 [((table_id1, col1), (table_id2, col2), label)]
            tables_dict: 表字典 - JSON格式
            all_tables_json_path: 所有表的JSON文件路径 - 用于构建全局值频率字典
        """
        try:
            logger.info("开始准备训练数据...")

            # 构建全局值频率字典（如果提供了 all_tables.json 路径）
            global_freq = None
            if all_tables_json_path is not None:
                global_freq = self.build_global_frequency_from_json(all_tables_json_path)

            positive_pairs = []

            # JSON格式输入
            if training_samples is not None and tables_dict is not None:
                logger.info("使用JSON格式训练数据")
                for (t1_id, c1), (t2_id, c2), label in training_samples:
                    if label == 1:  # 只使用正样本
                        if t1_id in tables_dict and t2_id in tables_dict:
                            t1 = tables_dict[t1_id]
                            t2 = tables_dict[t2_id]
                            if c1 in t1.columns and c2 in t2.columns:
                                positive_pairs.append(((t1, c1), (t2, c2)))

            # CSV格式输入（向后兼容）
            elif bimodels is not None:
                logger.info("使用CSV格式训练数据")
                for bm in bimodels:
                    for (t1, c1), (t2, c2) in bm.joins:
                        positive_pairs.append(((t1, c1), (t2, c2)))
            else:
                raise ValueError("必须提供 training_samples+tables_dict 或 bimodels")

            logger.info(f"正样本数量: {len(positive_pairs)}")

            # 数据增强：随机打乱部分列
            augmented_pairs = []
            num_to_shuffle = int(len(positive_pairs) * shuffle_rate)
            if num_to_shuffle > 0:
                shuffle_indices = random.sample(range(len(positive_pairs)), num_to_shuffle)

                for idx, ((t1, c1), (t2, c2)) in enumerate(positive_pairs):
                    if idx in shuffle_indices:
                        # 打乱第一列
                        shuffled_col = self._shuffle_column(t1.columns[c1])
                        # 创建临时表
                        temp_columns = t1.columns.copy()
                        temp_columns[c1] = shuffled_col
                        temp_table = Table(t1.table_name, temp_columns, t1.unique_id + "_shuffled")
                        augmented_pairs.append(((temp_table, c1), (t2, c2)))

            logger.info(f"数据增强后新增样本: {len(augmented_pairs)}")
            all_positive_pairs = positive_pairs + augmented_pairs

            # 清空文本缓存（开始新的训练）
            cache_size_before = len(self.column_text_cache)
            self.column_text_cache.clear()
            if cache_size_before > 0:
                logger.info(f"已清空旧的文本缓存 ({cache_size_before} 条)")

            # 准备训练样本（使用 InputExample）
            logger.info("开始序列化列文本（带缓存优化）...")
            train_examples = []
            for (t1, c1), (t2, c2) in all_positive_pairs:
                # 使用全局频率字典进行序列化（带缓存）
                text1 = self._column_to_text(t1, c1, datalake_freq=global_freq)
                text2 = self._column_to_text(t2, c2, datalake_freq=global_freq)
                # InputExample 的 label 用于 Multiple Negatives Ranking Loss
                train_examples.append(InputExample(texts=[text1, text2], label=1.0))

            # 输出缓存统计
            cache_size = len(self.column_text_cache)
            total_serializations = len(all_positive_pairs) * 2
            cache_hit_rate = (1 - cache_size / total_serializations) * 100 if total_serializations > 0 else 0
            logger.info(f"总训练样本数: {len(train_examples)}")
            logger.info(f"文本缓存统计: 缓存条目={cache_size}, 总序列化请求={total_serializations}, 缓存命中率={cache_hit_rate:.1f}%")

            # 多GPU训练时调整batch_size
            effective_batch_size = batch_size
            if self.num_gpus > 1:
                # 多GPU时，每个GPU处理 batch_size 个样本
                # 总的有效batch_size = batch_size * num_gpus
                effective_batch_size = batch_size * self.num_gpus
                logger.info(f"多GPU训练: 每个GPU batch_size={batch_size}, 有效batch_size={effective_batch_size}")
            else:
                logger.info(f"单GPU/CPU训练: batch_size={batch_size}")

            # 创建 DataLoader（使用多进程加速数据加载）
            num_workers = min(4, os.cpu_count() or 1)  # 使用4个worker或CPU核心数（取较小值）
            train_dataloader = DataLoader(
                train_examples,
                shuffle=True,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True if torch.cuda.is_available() else False  # GPU训练时使用pin_memory加速
            )
            logger.info(f"DataLoader配置: num_workers={num_workers}, pin_memory={torch.cuda.is_available()}")

            # 使用 Multiple Negatives Ranking Loss (自动使用in-batch负样本)
            train_loss = losses.MultipleNegativesRankingLoss(self.model)

            # 微调模型
            logger.info("开始微调预训练语言模型 (使用in-batch负样本)...")
            warmup_steps = int(len(train_dataloader) * epochs * 0.1)

            # SentenceTransformer 会自动检测并使用多GPU
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                warmup_steps=warmup_steps,
                optimizer_params={'lr': 2e-5},
                show_progress_bar=True,
                use_amp=True  # 使用混合精度训练以节省显存
            )

            logger.info("训练完成!")

        except Exception as e:
            logger.error(f"训练过程中出错: {e}")
            logger.exception("详细错误堆栈:")
            raise

    def build_embeddings(self, tables: List[Table]):
        """
        为所有列生成嵌入（不再使用FAISS索引）
        支持多GPU加速

        Args:
            tables: 表列表
        """
        logger.info("开始生成所有列的嵌入...")

        # 生成所有列的嵌入
        all_columns = []
        all_texts = []

        for table in tables:
            for col_name in table.col_names:
                all_columns.append((table, col_name))
                text = self._column_to_text(table, col_name)
                all_texts.append(text)

        logger.info(f"总列数: {len(all_columns)}")

        # 批量编码 - 多GPU时使用更大的batch_size
        encode_batch_size = 32
        if self.num_gpus > 1:
            encode_batch_size = 32 * self.num_gpus
            logger.info(f"多GPU编码: batch_size={encode_batch_size}")

        logger.info("批量生成列嵌入...")
        embeddings = self.model.encode(
            all_texts,
            batch_size=encode_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # 存储嵌入
        for i, (table, col_name) in enumerate(all_columns):
            key = (table.unique_id, col_name)
            self.column_embeddings[key] = embeddings[i]

        logger.info(f"列嵌入生成完成! 嵌入维度: {embeddings.shape[1]}")

    def get_column_embedding(self, table: Table, column_name: str) -> np.ndarray:
        """
        获取列的嵌入向量（如果没有则现场计算）

        Args:
            table: 表对象
            column_name: 列名

        Returns:
            列的嵌入向量
        """
        key = (table.unique_id, column_name)
        if key not in self.column_embeddings:
            # 现场计算嵌入
            text = self._column_to_text(table, column_name)
            embedding = self.model.encode([text], convert_to_numpy=True)[0]
            self.column_embeddings[key] = embedding
        return self.column_embeddings[key]

    def compute_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        计算两个嵌入向量的余弦相似度

        Args:
            emb1: 第一个嵌入向量
            emb2: 第二个嵌入向量

        Returns:
            余弦相似度 (0-1)
        """
        # 归一化向量
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        # 计算余弦相似度
        similarity = np.dot(emb1_norm, emb2_norm)
        return float(similarity)

    def save_model(self, model_dir: str):
        """保存模型"""
        try:
            os.makedirs(model_dir, exist_ok=True)

            # 保存 sentence-transformer 模型
            model_path = os.path.join(model_dir, "sentence_transformer")
            self.model.save(model_path)
            logger.info(f"模型已保存到: {model_path}")

            # 保存配置
            config = {
                "max_seq_length": self.max_seq_length,
            }
            config_path = os.path.join(model_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f)
            logger.info(f"配置已保存到: {config_path}")

        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            raise

    def load_model(self, model_dir: str):
        """加载模型"""
        try:
            # 加载 sentence-transformer 模型
            model_path = os.path.join(model_dir, "sentence_transformer")
            self.model = SentenceTransformer(model_path, device=device)
            logger.info(f"模型已从 {model_path} 加载")

            # 加载配置
            config_path = os.path.join(model_dir, "config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.max_seq_length = config["max_seq_length"]
            self.model.max_seq_length = self.max_seq_length
            logger.info(f"配置已加载")

        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise

# ------------------------------
# 全局预测（针对多表）
# ------------------------------

class GlobalOptimizer:
    """全局优化器：对多个表进行批量预测（使用余弦相似度遍历所有列对）"""

    def __init__(self, join_clf: JoinClassifier, top_k_edges=6):
        self.join_clf = join_clf
        self.top_k_edges = top_k_edges

    def predict_bi_model(self, tables: List[Table]) -> List[Tuple]:
        """
        预测表之间的所有可连接关系
        遍历所有表对的所有列对，计算余弦相似度，保留Top-K

        Returns:
            [(表1, 表2, 列1, 列2, 概率), ...]
        """
        logger.info(f"开始预测 {len(tables)} 张表之间的连接关系...")

        joins = []

        # 对每对表进行预测
        for i, t1 in tqdm(enumerate(tables), desc="预测连接概率", total=len(tables)):
            for j, t2 in enumerate(tables):
                if i >= j:  # 避免重复和自连接
                    continue

                # 使用堆保留Top-K列对（性能优化）
                heap = []

                # 遍历 t1 和 t2 的所有列对
                for col1 in t1.col_names:
                    # 获取 col1 的嵌入
                    emb1 = self.join_clf.get_column_embedding(t1, col1)

                    for col2 in t2.col_names:
                        # 获取 col2 的嵌入
                        emb2 = self.join_clf.get_column_embedding(t2, col2)

                        # 计算余弦相似度
                        similarity = self.join_clf.compute_cosine_similarity(emb1, emb2)

                        # 使用最小堆保留Top-K（效率更高）
                        if similarity > 0:
                            if len(heap) < self.top_k_edges:
                                heapq.heappush(heap, (similarity, col1, col2))
                            elif similarity > heap[0][0]:
                                heapq.heapreplace(heap, (similarity, col1, col2))

                # 将Top-K列对添加到结果中
                for similarity, col1, col2 in heap:
                    joins.append((t1, t2, col1, col2, similarity))

        logger.info(f"预测完成，共找到 {len(joins)} 个潜在连接")
        return joins

    def predict_for_query_tables(
        self,
        query_tables: List[Table],
        datalake_tables: List[Table],
        top_k_per_query: int = 30
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        为查询表预测与数据湖表的连接关系（表级别）

        只比较 join_key_indices 指定的列，返回表级别的匹配结果

        Args:
            query_tables: 查询表列表
            datalake_tables: 数据湖表列表
            top_k_per_query: 每个查询表返回的top-k候选表

        Returns:
            {query_table_id: [(candidate_table_id, score), ...]}
        """
        logger.info(f"开始预测 {len(query_tables)} 个查询表与 {len(datalake_tables)} 个数据湖表的连接...")

        predictions = {}

        for query_table in tqdm(query_tables, desc="预测查询表"):
            query_table_id = query_table.table_id

            # 获取查询表的连接键列
            query_join_cols = []
            if query_table.join_key_indices:
                for idx in query_table.join_key_indices:
                    if idx < len(query_table.col_names):
                        query_join_cols.append(query_table.col_names[idx])

            if not query_join_cols:
                # 如果没有指定连接键，跳过
                continue

            # 使用堆保留Top-K候选表
            heap = []

            for dl_table in datalake_tables:
                dl_table_id = dl_table.table_id

                # 获取数据湖表的连接键列
                dl_join_cols = []
                if dl_table.join_key_indices:
                    for idx in dl_table.join_key_indices:
                        if idx < len(dl_table.col_names):
                            dl_join_cols.append(dl_table.col_names[idx])

                if not dl_join_cols:
                    continue

                # 为每个查询列找到最佳匹配的数据湖列
                best_match_scores = []
                for query_col in query_join_cols:
                    query_emb = self.join_clf.get_column_embedding(query_table, query_col)

                    # 找到与当前查询列最相似的数据湖列
                    max_similarity = 0.0
                    for dl_col in dl_join_cols:
                        dl_emb = self.join_clf.get_column_embedding(dl_table, dl_col)
                        similarity = self.join_clf.compute_cosine_similarity(query_emb, dl_emb)
                        max_similarity = max(max_similarity, similarity)

                    best_match_scores.append(max_similarity)

                if not best_match_scores:
                    continue

                # 表对得分：最佳匹配列对相似度的平均值
                table_pair_score = sum(best_match_scores) / len(best_match_scores)

                # 使用最小堆保留Top-K表对
                if table_pair_score > 0:
                    if len(heap) < top_k_per_query:
                        heapq.heappush(heap, (table_pair_score, dl_table_id))
                    elif table_pair_score > heap[0][0]:
                        heapq.heapreplace(heap, (table_pair_score, dl_table_id))

            # 按相似度降序排序
            candidates = sorted(heap, key=lambda x: x[0], reverse=True)
            predictions[query_table_id] = [(cand_id, score) for score, cand_id in candidates]

        logger.info(f"预测完成，共 {len(predictions)} 个查询表")
        return predictions

# ------------------------------
# 评估指标计算
# ------------------------------

def compute_metrics_for_query(predictions: List[str], ground_truth: List[str], k: int) -> Dict[str, float]:
    """
    计算单个查询的评估指标

    Args:
        predictions: 预测的候选列表 (按相似度排序)
        ground_truth: Ground truth 列表
        k: Top-K 值

    Returns:
        包含 P@K, R@K, AP@K 的字典
    """
    gold_set = set(ground_truth)

    # 计算命中数和所有位置的precision
    hits = 0
    precisions_sum = 0.0

    for rank, pred in enumerate(predictions[:k], start=1):
        if pred in gold_set:
            hits += 1
        # 累加每个位置的precision值 (无论是否命中)
        precisions_sum += hits / rank

    # P@K: Precision at K
    p_at_k = hits / max(k, 1)

    # R@K: Recall at K
    r_at_k = hits / max(len(gold_set), 1)

    # MAP@K: Average of P@1, P@2, ..., P@k
    ap = precisions_sum / max(k, 1)

    return {
        f"P@{k}": float(p_at_k),
        f"R@{k}": float(r_at_k),
        f"AP@{k}": float(ap),
        "gold_size": int(len(gold_set)),
        "hits": int(hits),
    }


def compute_overall_metrics(all_metrics: List[Dict[str, float]], k: int) -> Dict[str, float]:
    """
    计算全局平均指标

    Args:
        all_metrics: 所有查询的指标列表
        k: Top-K 值

    Returns:
        全局指标字典
    """
    if not all_metrics:
        return {}

    q_count = len(all_metrics)
    sum_p = sum(m[f"P@{k}"] for m in all_metrics)
    sum_r = sum(m[f"R@{k}"] for m in all_metrics)
    sum_ap = sum(m[f"AP@{k}"] for m in all_metrics)

    total_hits = sum(m["hits"] for m in all_metrics)
    total_gold = sum(m["gold_size"] for m in all_metrics)

    # Macro 指标：对每个查询的指标取平均
    macro_p = sum_p / q_count
    macro_r = sum_r / q_count
    macro_map = sum_ap / q_count

    # Micro 指标：全局计算
    micro_p = total_hits / max(q_count * k, 1)
    micro_r = total_hits / max(total_gold, 1)

    # Weighted MAP：按 gold_size 加权
    sum_ap_weighted = sum(m[f"AP@{k}"] * m["gold_size"] for m in all_metrics)
    weighted_map = sum_ap_weighted / max(total_gold, 1.0)

    return {
        f"macro_P@{k}": float(macro_p),
        f"macro_R@{k}": float(macro_r),
        f"MAP@{k}": float(macro_map),
        f"weighted_MAP@{k}": float(weighted_map),
        f"micro_P@{k}": float(micro_p),
        f"micro_R@{k}": float(micro_r),
        "num_queries_with_gt": int(q_count),
        "total_hits": int(total_hits),
        "total_gold": int(total_gold),
    }


def print_overall_metrics(overall: Dict[str, Any], k: int) -> None:
    """打印全局平均指标（终端友好格式）"""
    n_q = overall.get("num_queries_with_gt", 0)
    macro_p = overall.get(f"macro_P@{k}", 0.0)
    macro_r = overall.get(f"macro_R@{k}", 0.0)
    macro_map = overall.get(f"MAP@{k}", 0.0)

    print()
    print("============== 全局平均指标 ===============")
    print(f"参与评测的 query 数量: {n_q}")
    print(f"P@{k}: {macro_p:.4f}, R@{k}: {macro_r:.4f}, MAP@{k}: {macro_map:.4f}")
    print("==========================================")
    print()


# ------------------------------
# 主流程
# ------------------------------

def JoinClassify(mode, tables, model_dir, links=None, output=None, top_k=30, isfolder=False,
                 query_tables=None, datalake_tables=None, ground_truth=None, all_tables=None):
    """
    JoinClassifier 主函数（支持CSV和JSON格式）

    Args:
        mode: 'train', 'predict', 'train_json', 'predict_json'
        tables: 表目录路径 (CSV模式)
        model_dir: 模型目录
        links: 正样本文件 (CSV模式)
        output: 输出文件
        top_k: 每对表保留的最佳连接数
        isfolder: 是否使用文件夹结构 (CSV模式)
        query_tables: 查询表JSON文件 (JSON模式)
        datalake_tables: 数据湖表JSON文件 (JSON模式)
        ground_truth: Ground truth JSON文件 (JSON模式训练)
        all_tables: 所有表JSON文件 (用于构建全局值频率)
    """
    try:
        logger.info(f"=== 开始执行 {mode} 模式 ===")

        if mode == 'train_json':
            logger.info("加载JSON格式训练数据...")
            if not query_tables or not ground_truth:
                raise ValueError("train_json模式需要提供 --query_tables 和 --ground_truth 参数")

            # 加载查询表
            query_tables_dict = load_tables_json(query_tables)
            logger.info(f"加载了 {len(query_tables_dict)} 个查询表")

            # 加载数据湖表（如果提供）
            datalake_tables_dict = {}
            if datalake_tables:
                datalake_tables_dict = load_tables_json(datalake_tables)
                logger.info(f"加载了 {len(datalake_tables_dict)} 个数据湖表")

            # 合并所有表
            all_tables_dict = {**query_tables_dict, **datalake_tables_dict}

            # 加载ground truth
            gt = load_ground_truth_json(ground_truth)
            logger.info(f"加载了 {len(gt)} 个ground truth条目")

            # 构建训练样本
            training_samples = build_training_samples_from_json(gt, all_tables_dict)

            # 创建 JoinClassifier 分类器
            clf = JoinClassifier()

            # 训练模型
            clf.train(
                training_samples=training_samples,
                tables_dict=all_tables_dict,
                batch_size=16,
                epochs=1,
                shuffle_rate=0.3,
                all_tables_json_path=all_tables
            )

            # 保存模型
            clf.save_model(model_dir)
            logger.info("=== 训练完成 ===")

        elif mode == 'predict_json':
            logger.info("加载JSON格式预测数据...")
            if not query_tables or not datalake_tables or not output:
                raise ValueError("predict_json模式需要提供 --query_tables, --datalake_tables 和 --output 参数")

            # 加载查询表和数据湖表
            query_tables_dict = load_tables_json(query_tables)
            datalake_tables_dict = load_tables_json(datalake_tables)
            logger.info(f"加载了 {len(query_tables_dict)} 个查询表和 {len(datalake_tables_dict)} 个数据湖表")

            # 加载 ground truth（如果提供）
            gt = {}
            if ground_truth:
                gt = load_ground_truth_json(ground_truth)
                logger.info(f"加载了 {len(gt)} 个 ground truth 条目")

            # 加载模型
            clf = JoinClassifier()
            clf.load_model(model_dir)

            # 生成所有表的列嵌入
            all_tables = list(query_tables_dict.values()) + list(datalake_tables_dict.values())
            clf.build_embeddings(all_tables)

            # 全局预测
            opt = GlobalOptimizer(clf, top_k)
            logger.info(f"开始预测，Top-K={top_k}")

            predictions = opt.predict_for_query_tables(
                query_tables=list(query_tables_dict.values()),
                datalake_tables=list(datalake_tables_dict.values()),
                top_k_per_query=top_k
            )

            logger.info(f"预测完成，共 {len(predictions)} 个查询表")

            # 构建结果（JOINnow 兼容格式）
            results = []
            all_metrics = []

            for query_table_id, candidates in predictions.items():
                # 候选表列表（只保留表ID）
                candidate_list = [cand_id for cand_id, _ in candidates]

                # 计算评估指标（如果有 ground truth）
                metrics = None
                if query_table_id in gt:
                    metrics = compute_metrics_for_query(candidate_list, gt[query_table_id], top_k)
                    all_metrics.append(metrics)

                results.append({
                    "query_table_id": query_table_id,
                    "k": int(top_k),
                    "metrics": metrics,
                    "candidates": candidate_list,
                })

            # 计算全局指标
            overall_metrics = None
            if all_metrics:
                overall_metrics = compute_overall_metrics(all_metrics, top_k)
                logger.info(f"全局指标: {overall_metrics}")

            # 构建输出对象（JOINnow 兼容格式）
            output_obj = {
                "inference_config": {
                    "k": int(top_k),
                    "model_dir": model_dir,
                    "query_tables": query_tables,
                    "datalake_tables": datalake_tables,
                },
                "overall_metrics": overall_metrics,
                "results": results,
            }

            # 保存结果
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(output_obj, f, indent=2, ensure_ascii=False)
            logger.info(f"结果已保存到: {output}")

            # 打印全局指标
            if overall_metrics:
                print_overall_metrics(overall_metrics, top_k)

            logger.info("=== 预测完成 ===")

        elif mode == 'train':
            logger.info("加载CSV格式训练数据...")
            if not links:
                raise ValueError("训练模式需要提供 --links 参数")

            tables_dict = load_tables(tables, isfolder)
            if not tables_dict:
                raise ValueError(f"未能加载任何表，请检查路径: {tables}")

            bimodel = load_links_csv(links, tables_dict)
            if not bimodel.joins:
                logger.warning("未找到有效的连接关系")

            # 创建 JoinClassifier 分类器
            clf = JoinClassifier()

            # 训练模型（不再需要负样本）
            clf.train(bimodels=[bimodel], batch_size=32, epochs=10, shuffle_rate=0.3)

            # 保存模型
            clf.save_model(model_dir)
            logger.info("=== 训练完成 ===")

        elif mode == 'predict':
            logger.info("加载CSV格式预测数据...")
            if not output:
                raise ValueError("预测模式需要提供 --output 参数")

            tables_dict = load_tables(tables, isfolder)
            if not tables_dict:
                raise ValueError(f"未能加载任何表，请检查路径: {tables}")
            logger.info(f"将对 {len(tables_dict)} 张表进行预测")

            # 加载模型
            clf = JoinClassifier()
            clf.load_model(model_dir)

            # 生成列嵌入
            clf.build_embeddings(list(tables_dict.values()))

            # 全局预测
            opt = GlobalOptimizer(clf, top_k)
            logger.info(f"开始预测，Top-K={top_k}")

            results = []
            for t1, t2, c1, c2, prob in opt.predict_bi_model(list(tables_dict.values())):
                results.append({
                    "table1": t1.unique_id,
                    "table2": t2.unique_id,
                    "column1": c1,
                    "column2": c2,
                    "prob": prob
                })

            logger.info(f"预测完成，共找到 {len(results)} 个连接关系")

            # 保存结果
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"结果已保存到: {output}")
            logger.info("=== 预测完成 ===")

        else:
            raise ValueError(f"不支持的模式: {mode}，支持的模式: train, predict, train_json, predict_json")

    except Exception as e:
        logger.error(f"执行失败: {e}")
        logger.exception("详细错误堆栈:")
        raise

# ------------------------------
# 命令行入口
# ------------------------------

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description='JoinClassifier - 基于预训练语言模型的表连接关系预测工具',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
使用示例:

CSV格式训练:
  python joinclassifier.py --mode train --tables ./tables --links ./links.csv --model_dir ./model

CSV格式预测:
  python joinclassifier.py --mode predict --tables ./tables --model_dir ./model --output ./results.json --top_k 10

JSON格式训练:
  python joinclassifier.py --mode train_json --query_tables ./traindata/queries.json --datalake_tables ./traindata/data_lake.json --ground_truth ./traindata/ground_truth.json --model_dir ./joinclassifier --all_tables ./traindata/all_tables.json

JSON格式预测:
  python joinclassifier.py --mode predict_json --query_tables ./5%/queries.json --datalake_tables ./5%/data_lake.json --model_dir ./joinclassifier --output ./joinclassifier-predictions.json --top_k 30 --ground_truth ./5%/ground_truth.json --all_tables ./5%/all_tables.json
            """
        )

        parser.add_argument('--mode', required=True,
                            choices=['train', 'predict', 'train_json', 'predict_json'],
                            help='运行模式: train(CSV训练), predict(CSV预测), train_json(JSON训练), predict_json(JSON预测)')

        # CSV格式参数
        parser.add_argument('--tables', help='CSV模式: 表数据目录路径')
        parser.add_argument('--links', help='CSV训练模式: 正样本连接关系文件路径(CSV格式)')
        parser.add_argument('--isfolder', action='store_true',
                            help='CSV模式: 如果设置，表示tables目录包含子文件夹结构')

        # JSON格式参数
        parser.add_argument('--query_tables', help='JSON模式: 查询表JSON文件路径')
        parser.add_argument('--datalake_tables', help='JSON模式: 数据湖表JSON文件路径')
        parser.add_argument('--ground_truth', help='JSON模式: Ground truth JSON文件路径 (train_json必需, predict_json可选用于评估)')
        parser.add_argument('--all_tables', help='JSON训练模式: 所有表JSON文件路径 (用于构建全局值频率，可选)')

        # 通用参数
        parser.add_argument('--model_dir', required=True, help='模型保存/加载目录')
        parser.add_argument('--output', help='预测模式: 结果输出文件路径(JSON格式)')
        parser.add_argument('--top_k', type=int, default=30,
                            help='预测时每个查询列返回的候选数量(默认30)')

        args = parser.parse_args()

        # 参数验证
        if args.mode == 'train':
            if not args.tables or not args.links:
                parser.error("CSV训练模式(--mode train)必须提供 --tables 和 --links 参数")
        elif args.mode == 'predict':
            if not args.tables or not args.output:
                parser.error("CSV预测模式(--mode predict)必须提供 --tables 和 --output 参数")
        elif args.mode == 'train_json':
            if not args.query_tables or not args.ground_truth:
                parser.error("JSON训练模式(--mode train_json)必须提供 --query_tables 和 --ground_truth 参数")
        elif args.mode == 'predict_json':
            if not args.query_tables or not args.datalake_tables or not args.output:
                parser.error("JSON预测模式(--mode predict_json)必须提供 --query_tables, --datalake_tables 和 --output 参数")

        logger.info("=" * 60)
        logger.info("JoinClassifier 启动")
        logger.info("=" * 60)

        JoinClassify(
            mode=args.mode,
            tables=args.tables,
            model_dir=args.model_dir,
            links=args.links,
            output=args.output,
            top_k=args.top_k,
            isfolder=args.isfolder,
            query_tables=args.query_tables,
            datalake_tables=args.datalake_tables,
            ground_truth=args.ground_truth,
            all_tables=args.all_tables
        )

        logger.info("=" * 60)
        logger.info("程序执行成功完成")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("\n程序被用户中断")
        exit(1)
    except Exception as e:
        logger.error("=" * 60)
        logger.error("程序执行失败")
        logger.error("=" * 60)
        logger.error(f"错误: {e}")
        logger.exception("详细错误堆栈:")
        exit(1)
