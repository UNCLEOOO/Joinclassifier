import random
from typing import List, Dict, Tuple, Optional
import logging

from torch.utils.data import Dataset

from polyjoin_model import serialize_multi_key_columns, serialize_descriptions


class POLYJOINDataset(Dataset):
    """
    POLYJOIN 训练数据集
    
    从表格数据中生成 (row_text, desc_text) 训练对
    
    数据格式说明:
    - columns: 列名列表
    - cells: 按列组织的值，cells[i] 是第i列的所有值
    - descriptions: 列描述列表
    """
    
    def __init__(
        self,
        tables: List[Dict],
        n_columns_range: Tuple[int, int] = (2, 4),
        samples_per_table: int = 5,
        max_values_per_column: int = 10,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            tables: 表格列表
            n_columns_range: 多键列数范围 (min, max)
            samples_per_table: 每个表生成的样本数
            max_values_per_column: 每列最多取的值数量
            logger: 日志记录器
        """
        self.samples = []
        self.logger = logger or logging.getLogger(__name__)
        self.n_columns_range = n_columns_range
        self.max_values_per_column = max_values_per_column
        
        self._build_samples(tables, samples_per_table)
    
    def _build_samples(self, tables: List[Dict], samples_per_table: int):
        """构建训练样本"""
        total_tables = len(tables)
        valid_tables = 0
        skipped_tables = 0
        
        for table in tables:
            table_id = table.get("table_id", "unknown")
            columns = table.get("columns", [])
            
            # 支持两种格式: cells (按列) 或 rows (按行)
            if "cells" in table:
                # cells 格式: cells[i] 是第i列的所有值
                cells = table.get("cells", [])
            elif "rows" in table:
                # rows 格式: rows[i] 是第i行的所有值，需要转置
                rows = table.get("rows", [])
                if rows and len(rows) > 0:
                    n_cols = len(columns)
                    cells = [[] for _ in range(n_cols)]
                    for row in rows:
                        for col_idx, val in enumerate(row):
                            if col_idx < n_cols:
                                cells[col_idx].append(val)
                else:
                    cells = []
            else:
                cells = []
            
            descriptions = table.get("descriptions", [""] * len(columns))
            
            # 确保长度匹配
            while len(descriptions) < len(columns):
                descriptions.append("")
            while len(cells) < len(columns):
                cells.append([])
            
            # 检查表格是否有效
            if len(columns) < self.n_columns_range[0]:
                skipped_tables += 1
                self.logger.debug(f"跳过表格 {table_id}: 列数不足 ({len(columns)} < {self.n_columns_range[0]})")
                continue
            
            # 检查是否有数据
            has_data = any(len(col_vals) > 0 for col_vals in cells)
            if not has_data:
                skipped_tables += 1
                self.logger.debug(f"跳过表格 {table_id}: 无数据")
                continue
            
            valid_tables += 1
            
            # 为每个表生成多个样本
            for _ in range(samples_per_table):
                sample = self._create_sample(
                    table_id, columns, cells, descriptions
                )
                if sample:
                    self.samples.append(sample)
        
        self.logger.info(
            f"数据集构建完成: {valid_tables}/{total_tables} 有效表格, "
            f"{len(self.samples)} 训练样本, {skipped_tables} 表格被跳过"
        )
    
    def _create_sample(
        self,
        table_id: str,
        columns: List[str],
        cells: List[List[str]],
        descriptions: List[str]
    ) -> Optional[Dict]:
        """创建单个训练样本"""
        # 随机选择列数
        n_cols = min(
            random.randint(self.n_columns_range[0], self.n_columns_range[1]),
            len(columns)
        )
        
        # 随机选择列索引
        selected_indices = random.sample(range(len(columns)), n_cols)
        
        # 提取选中列的名称、值和描述
        selected_names = [columns[i] for i in selected_indices]
        selected_values = [cells[i] if i < len(cells) else [] for i in selected_indices]
        selected_descs = [descriptions[i] if i < len(descriptions) else "" for i in selected_indices]
        
        # 检查是否有足够的值
        if not any(selected_values):
            return None
        
        # 序列化
        row_text = serialize_multi_key_columns(
            selected_names, selected_values, self.max_values_per_column
        )
        desc_text = serialize_descriptions(selected_names,selected_descs)
        
        # 跳过无效样本
        if row_text == "empty" or desc_text == "No description available":
            return None
        # print(desc_text)
        # print(row_text)
        # print("-----------------------------------")
        return {
            "table_id": table_id,
            "row_text": row_text,
            "desc_text": desc_text,
            "column_names": selected_names,
            "n_columns": n_cols,
            "sample_idx": len(self.samples)  # 【修复】记录该样本在数据集中的全局索引
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        # 确保返回时 sample_idx 是当前的实际索引（以防数据集被过滤等情况）
        sample["sample_idx"] = idx
        return sample


class DescriptionDataset(Dataset):
    """
    描述数据集（用于初始聚类阶段）

    直接复用 POLYJOINDataset 中的 desc_text，
    保证用于聚类的描述和用于对比学习的描述是同一批。
    
    """
    def __init__(
        self,
        train_dataset: POLYJOINDataset,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.samples: List[Dict] = []
        
        # 【关键】保持与 train_dataset 完全相同的索引顺序
        for idx, sample in enumerate(train_dataset.samples):
            desc_text = sample.get("desc_text", "")
            self.samples.append({
                "original_idx": idx,  # 保存在 train_dataset 中的原始索引
                "table_id": sample.get("table_id", "unknown"),
                "desc_text": desc_text if desc_text and desc_text.strip() and desc_text != "No description available" else ""
            })

        self.logger.info(f"描述数据集: 从训练集复用 {len(self.samples)} 个描述样本")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def train_collate_fn(batch: List[Dict]) -> Dict[str, List]:
    """训练数据集的collate函数"""
    return {
        "row_texts": [item["row_text"] for item in batch],
        "desc_texts": [item["desc_text"] for item in batch],
        "table_ids": [item["table_id"] for item in batch],
        "n_columns": [item.get("n_columns", 0) for item in batch],
        "sample_indices": [item["sample_idx"] for item in batch]  # 【修复】添加全局索引
    }


def desc_collate_fn(batch: List[Dict]) -> Dict[str, List]:
    """描述数据集的collate函数"""
    return {
        "desc_texts": [item["desc_text"] for item in batch],
        "table_ids": [item["table_id"] for item in batch],
        "original_indices": [item["original_idx"] for item in batch]  # 保留原始索引信息
    }


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试用户的数据格式
    test_tables = [
        {
            "table_id": "28et-rv7b",
            "columns": ["identifier", "record_title", "object_type", "resource_identifier"],
            "cells": [
                ["REC0052", "REC0008", "REC0014-04-1"],  # identifier列的值
                ["Bodies in Transit registers", "Almshouse ledgers collection", "Mayor William O'Dwyer photographic prints"],  # record_title列的值
                ["Text", "Text", "Still Image"],  # object_type列的值
                ["REC.0052", "REC.0008", "REC.0014"]  # resource_identifier列的值
            ],
            "descriptions": [
                "A unique identifier for the digital object as a whole.",
                "A title expression for the digital object.",
                "A generic term indicating the basic content type of the digital object.",
                "The identification number assigned to the each discrete Resource within a Repository"
            ]
        }
    ]
    
    print("=" * 70)
    print("POLYJOIN 数据集测试 (cells格式)")
    print("=" * 70)
    
    # 测试训练数据集
    dataset = POLYJOINDataset(
        tables=test_tables,
        n_columns_range=(2, 3),
        samples_per_table=5
    )
    
    print(f"\n训练数据集大小: {len(dataset)}")
    
    if len(dataset) > 0:
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\n示例样本 {i}:")
            print(f"  table_id: {sample['table_id']}")
            print(f"  sample_idx: {sample['sample_idx']}")
            print(f"  row_text: {sample['row_text'][:100]}...")
            print(f"  desc_text: {sample['desc_text'][:100]}...")
            print(f"  n_columns: {sample['n_columns']}")
    
    # 测试描述数据集
    desc_dataset = DescriptionDataset(
        train_dataset=dataset
    )
    
    print(f"\n描述数据集大小: {len(desc_dataset)}")
    
    # 测试 collate_fn
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, collate_fn=train_collate_fn)
    batch = next(iter(loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"sample_indices in batch: {batch['sample_indices']}")
    
    print("\n✓ 数据集测试通过!")
