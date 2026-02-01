"""
POLYJOIN 训练入口

使用方法:
    # 使用本地BERT模型
    python train.py --data_json your_data.json --model_path /path/to/your/bert/model
    
    # 使用在线模型
    python train.py --data_json datatest.json --model_path all-mpnet-base-v2-local

参数说明:
    --data_json     训练数据JSON文件路径
    --model_path    BERT/SentenceTransformer模型路径（支持本地路径）
    --save_dir      检查点保存目录
    --batch_size    批次大小
    --epochs        训练轮数
    --lr            学习率
    --device        设备 (cuda/cpu)
"""

import os
import json
import argparse
import torch
from torch.utils.data import DataLoader

from polyjoin_model import POLYJOINModel
from polyjoin_dataset import (
    POLYJOINDataset, 
    DescriptionDataset,
    train_collate_fn, 
    desc_collate_fn
)
from polyjoin_trainer import POLYJOINTrainer, setup_logging


def load_tables_from_json(json_path: str):
    """从JSON文件加载表格数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果是单个表格，转换为列表
    if isinstance(data, dict):
        return [data]
    return data


def main():
    """主训练函数"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="POLYJOIN 训练脚本")
    # 必需参数
    parser.add_argument(
        "--data_json", 
        type=str, 
        required=True,
        help="训练数据JSON文件路径"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="BERT/SentenceTransformer模型路径（支持本地路径，如 /path/to/bert-base-chinese）"
    )
    
    # 可选参数
    parser.add_argument("--save_dir", type=str, default="./checkpoints1", help="检查点保存目录")
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志保存目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=160, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)，默认自动检测")
    
    # 模型超参数
    parser.add_argument("--temperature", type=float, default=0.05, help="对比学习温度参数 τ")
    parser.add_argument("--momentum", type=float, default=0.9, help="动量更新系数 z")
    parser.add_argument("--n_clustering_layers", type=int, default=3, help="聚类层数 L")
    parser.add_argument("--alpha", type=float, default=1.0, help="L_key 权重")
    parser.add_argument("--beta", type=float, default=1.0, help="L_cent 权重")
    
    # 数据参数
    parser.add_argument("--n_columns_min", type=int, default=2, help="多键最小列数")
    parser.add_argument("--n_columns_max", type=int, default=4, help="多键最大列数")
    parser.add_argument("--samples_per_table", type=int, default=3, help="每个表生成的样本数")
    parser.add_argument("--max_values_per_column", type=int, default=10, help="每列最多取的值数量")
    
    # 训练参数
    parser.add_argument("--early_stopping", type=int, default=10, help="早停耐心值")
    parser.add_argument("--save_every", type=int, default=10, help="每隔多少epoch保存")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device:
        device = args.device
    else:
        device = "cuda:2" if torch.cuda.is_available() else "cpu"
    
    # 设置日志
    logger = setup_logging(args.log_dir)
    
    logger.info("=" * 70)
    logger.info("POLYJOIN 训练（修复版）")
    logger.info("=" * 70)
    logger.info(f"数据文件: {args.data_json}")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"保存目录: {args.save_dir}")
    logger.info(f"设备: {device}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"训练轮数: {args.epochs}")
    logger.info(f"学习率: {args.lr}")
    logger.info(f"温度参数: {args.temperature}")
    logger.info(f"动量系数: {args.momentum}")
    logger.info(f"聚类层数: {args.n_clustering_layers}")
    logger.info("=" * 70)
    
    # 加载数据
    logger.info("\n加载数据...")
    tables = load_tables_from_json(args.data_json)
    logger.info(f"加载了 {len(tables)} 个表格")
    
    # 创建数据集
    logger.info("\n创建数据集...")
    
    n_columns_range = (args.n_columns_min, args.n_columns_max)
    
    train_dataset = POLYJOINDataset(
        tables=tables,
        n_columns_range=n_columns_range,
        samples_per_table=args.samples_per_table,
        max_values_per_column=args.max_values_per_column,
        logger=logger
    )
    # 【重要】DescriptionDataset 直接从 train_dataset 构建，保持索引对应
    desc_dataset = DescriptionDataset(
        train_dataset=train_dataset,
        logger=logger
    )

    if len(train_dataset) == 0:
        logger.error("没有可用的训练样本!")
        logger.error("请检查数据格式是否正确，确保每个表格至少有2列且有描述")
        return
    
    # 创建DataLoader
    # 【注意】train_loader 使用 shuffle=True，每个 epoch 打乱顺序
    # 但 sample_idx 记录的是在原始数据集中的索引，与聚类分配对应
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=train_collate_fn,
        drop_last=True
    )
    
    # 【重要】desc_loader 使用 shuffle=False，保证收集嵌入的顺序与数据集索引一致
    # 这样聚类得到的 cluster_assignments[i] 对应 train_dataset[i]
    desc_loader = DataLoader(
        desc_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 【关键】不打乱，保持索引顺序
        num_workers=0,
        collate_fn=desc_collate_fn
    )
    
    logger.info(f"训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
    logger.info(f"描述集: {len(desc_dataset)} 样本")
    
    # 验证索引对应关系
    logger.info("\n验证数据集索引对应关系...")
    if len(train_dataset) == len(desc_dataset):
        logger.info(f"✓ 训练集和描述集大小一致: {len(train_dataset)}")
    else:
        logger.warning(f"⚠ 训练集 ({len(train_dataset)}) 和描述集 ({len(desc_dataset)}) 大小不一致!")
    
    # 初始化模型
    logger.info("\n初始化模型...")
    logger.info(f"加载模型: {args.model_path}")
    
    model = POLYJOINModel(
        model_path=args.model_path,
        temperature=args.temperature,
        momentum=args.momentum,
        n_clustering_layers=args.n_clustering_layers,
        alpha=args.alpha,
        beta=args.beta,
        device=device,
        logger=logger
    )
    
    logger.info(f"模型隐藏维度: {model.hidden_size}")
    logger.info(f"可训练参数: {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad):,}")
    
    # 创建训练器
    trainer = POLYJOINTrainer(
        model=model,
        train_loader=train_loader,
        device=device,
        save_dir=args.save_dir,
        lr=args.lr,
        logger=logger
    )
    
    # 开始训练
    trainer.train(
        n_epochs=args.epochs,
        desc_loader=desc_loader,
        save_every=args.save_every,
        early_stopping_patience=args.early_stopping
    )
    
    logger.info(f"\n检查点保存在: {args.save_dir}")
    logger.info(f"日志保存在: {args.log_dir}")


if __name__ == "__main__":
    main()
