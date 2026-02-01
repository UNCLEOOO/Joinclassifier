import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from polyjoin_model import POLYJOINModel
from polyjoin_dataset import (
    POLYJOINDataset, 
    DescriptionDataset,
    train_collate_fn, 
    desc_collate_fn
)


def setup_logging(log_dir: str = "./logs") -> logging.Logger:
    """配置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"polyjoin_train_{timestamp}.log")
    
    logger = logging.getLogger("POLYJOIN")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class POLYJOINTrainer:
    """POLYJOIN 训练器"""
    
    def __init__(
        self,
        model: POLYJOINModel,
        train_loader: DataLoader,
        device: str,
        save_dir: str,
        lr: float = 1e-5,
        weight_decay: float = 0.0,
        grad_clip: float = 1.0,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.save_dir = save_dir
        self.grad_clip = grad_clip
        self.logger = logger or logging.getLogger("POLYJOIN")
        
        # 【新增】保存 desc_loader 的引用，用于每个 epoch 后重新收集完整嵌入
        self.desc_loader: Optional[DataLoader] = None
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.encoder.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 训练历史
        self.history = {
            'epochs': [],
            'train_loss': [],
            'key_loss': [],
            'cent_loss': []
        }
        
        self.best_loss = float('inf')
    
    def initial_clustering(self, desc_loader: DataLoader) -> bool:
        """
        阶段一: 初始聚类
        
        收集所有描述嵌入，执行层次聚类，得到每个样本的聚类分配。
        聚类分配的索引与训练数据集的索引一一对应。
        """
        self.logger.info("=" * 70)
        self.logger.info("阶段一: 初始聚类")
        self.logger.info("=" * 70)
        
        self.model.eval()
        all_desc_embeddings = []
        
        # 按顺序收集所有描述嵌入（保持索引对应）
        with torch.no_grad():
            for batch in tqdm(desc_loader, desc="收集描述嵌入"):
                desc_texts = batch["desc_texts"]
                desc_emb = self.model.encode_descriptions(desc_texts)
                all_desc_embeddings.append(desc_emb.cpu())
        
        all_desc_embeddings = torch.cat(all_desc_embeddings, dim=0)
        self.logger.info(f"收集了 {len(all_desc_embeddings)} 个描述嵌入")
        
        # 执行聚类
        self.logger.info("执行层次聚类...")
        self.model.update_centroids(all_desc_embeddings.to(self.device))
        
        if not self.model.centroids_per_layer:
            self.logger.error("聚类失败")
            return False
        
        # 打印聚类统计
        for layer_idx, assignments in enumerate(self.model.cluster_assignments_per_layer):
            n_clusters = len(set(assignments))
            self.logger.info(f"  Layer {layer_idx + 1}: {n_clusters} 个聚类, "
                           f"{len(assignments)} 个样本分配")
        
        # 保存聚类结果
        self._save_clustering_checkpoint()
        return True
    
    def train_epoch(self, epoch: int) -> Dict:
        """训练一个epoch"""
        self.model.train()
        
        running_total = 0.0
        running_key = 0.0
        running_cent = 0.0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            row_texts = batch["row_texts"]
            desc_texts = batch["desc_texts"]
            sample_indices = batch["sample_indices"]  # 【修复】获取全局索引
            
            # 前向传播
            row_emb, desc_emb = self.model(row_texts, desc_texts)
            
            # 计算损失（传入 sample_indices）
            loss, loss_dict = self.model.compute_loss(
                row_emb, 
                desc_emb, 
                sample_indices  # 【修复】传入索引用于查找聚类分配
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            
            # 动量更新
            with torch.no_grad():
                self.model._momentum_update()
            
            running_total += loss_dict['total_loss']
            running_key += loss_dict['key_loss']
            running_cent += loss_dict['cent_loss']
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f"{running_total/n_batches:.4f}",
                'key': f"{running_key/n_batches:.4f}",
                'cent': f"{running_cent/n_batches:.4f}"
            })
        
        # 【修复】使用完整的 desc_loader 重新收集所有描述嵌入，保证索引对应
        if self.desc_loader is not None:
            self.logger.info("更新聚类中心（使用完整数据集）...")
            self.model.eval()
            all_desc_embeddings = []
            with torch.no_grad():
                for batch in self.desc_loader:
                    desc_texts = batch["desc_texts"]
                    desc_emb = self.model.encode_descriptions(desc_texts)
                    all_desc_embeddings.append(desc_emb.cpu())
            all_desc_emb = torch.cat(all_desc_embeddings, dim=0)
            self.model.update_centroids(all_desc_emb.to(self.device))
            self.model.train()
        else:
            self.logger.warning("desc_loader 未设置，跳过聚类更新")
        
        return {
            'avg_total_loss': running_total / max(n_batches, 1),
            'avg_key_loss': running_key / max(n_batches, 1),
            'avg_cent_loss': running_cent / max(n_batches, 1),
            'n_batches': n_batches
        }

    def train(
        self, 
        n_epochs: int = 20,
        desc_loader: Optional[DataLoader] = None,
        save_every: int = 5,
        early_stopping_patience: int = 10
    ):
        """完整训练流程"""
        self.logger.info("=" * 70)
        self.logger.info("POLYJOIN 训练开始")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"训练轮数: {n_epochs}")
        self.logger.info("=" * 70)
        
        # 【新增】保存 desc_loader 引用，供 train_epoch 使用
        self.desc_loader = desc_loader
        
        # 阶段一: 初始聚类
        if desc_loader is not None:
            if not self.initial_clustering(desc_loader):
                return
        
        # 阶段二: 自监督微调
        self.logger.info("=" * 70)
        self.logger.info("阶段二: 自监督微调")
        self.logger.info("=" * 70)
        
        patience_counter = 0
        
        for epoch in range(1, n_epochs + 1):
            self.logger.info(f"\nEpoch {epoch}/{n_epochs}")
            
            metrics = self.train_epoch(epoch)
            
            self.history['epochs'].append(epoch)
            self.history['train_loss'].append(metrics['avg_total_loss'])
            self.history['key_loss'].append(metrics['avg_key_loss'])
            self.history['cent_loss'].append(metrics['avg_cent_loss'])
            
            self.logger.info(f"  总损失: {metrics['avg_total_loss']:.4f}")
            self.logger.info(f"  Key损失: {metrics['avg_key_loss']:.4f}")
            self.logger.info(f"  Cent损失: {metrics['avg_cent_loss']:.4f}")
            
            if metrics['avg_total_loss'] < self.best_loss:
                self.best_loss = metrics['avg_total_loss']
                self._save_checkpoint(epoch, metrics, is_best=True)
                self.logger.info(f"  ★ 新最佳模型 (loss={self.best_loss:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"\n早停: {early_stopping_patience} 轮无改进")
                break
            
            if epoch % save_every == 0:
                self._save_checkpoint(epoch, metrics)
        
        self._save_checkpoint(epoch, metrics, is_final=True)
        self._save_history()
        
        self.logger.info("=" * 70)
        self.logger.info(f"训练完成! 最佳损失: {self.best_loss:.4f}")
        self.logger.info("=" * 70)
    
    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False, is_final: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'metrics': metrics,
            'model_state_dict': self.model.state_dict(),
            'encoder_state_dict': self.model.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'centroids': [c.cpu().numpy() if isinstance(c, torch.Tensor) else c 
                         for c in self.model.centroids_per_layer],
            'cluster_assignments': self.model.cluster_assignments_per_layer,
        }
        
        if is_best:
            path = os.path.join(self.save_dir, "best_model.pt")
        elif is_final:
            path = os.path.join(self.save_dir, "final_model.pt")
        else:
            path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt")
        
        torch.save(checkpoint, path)
    
    def _save_clustering_checkpoint(self):
        """保存初始聚类结果"""
        checkpoint = {
            'centroids': [c.cpu().numpy() if isinstance(c, torch.Tensor) else c 
                         for c in self.model.centroids_per_layer],
            'cluster_assignments': self.model.cluster_assignments_per_layer,
        }
        path = os.path.join(self.save_dir, "initial_clustering.pt")
        torch.save(checkpoint, path)
        self.logger.info(f"保存初始聚类结果: {path}")
    
    def _save_history(self):
        """保存训练历史"""
        path = os.path.join(self.save_dir, "training_history.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)