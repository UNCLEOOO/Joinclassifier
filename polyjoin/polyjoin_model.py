import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AffinityPropagation
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
import time


# ============================================================================
# 序列化函数 (论文 Section 3.1, Equation 1)
# ============================================================================

def serialize_multi_key_columns(
    column_names: List[str], 
    column_values: List[List[str]],
    max_values_per_column: int = 10
) -> str:
    """
    序列化多键列数据
    
    格式: col1_name : val1, val2, ... [SEP] col2_name : val1, val2, ...
    
    Args:
        column_names: 列名列表，如 ["Country", "City"]
        column_values: 每列的值列表，如 [["USA", "UK"], ["NYC", "London"]]
        max_values_per_column: 每列最多取多少个值
    
    Returns:
        序列化字符串
    """
    parts = []
    for col_name, values in zip(column_names, column_values):
        # 去重并限制数量
        unique_values = list(dict.fromkeys(values))[:max_values_per_column]
        values_str = ", ".join(str(v) for v in unique_values if v)  # 过滤空值
        if values_str:
            parts.append(f"{col_name} : {values_str}")
    
    return " [SEP] ".join(parts) if parts else "empty"


def serialize_descriptions(column_names: List[str],descriptions: List[str]) -> str:
    """
    序列化列描述
    
    格式: desc1 [SEP] desc2 [SEP] ...
    
    Args:
        descriptions: 描述列表
    
    Returns:
        序列化字符串
    """
    valid_descs = [f"{col} : {d.strip()}" for col, d in zip(column_names, descriptions) if d and d.strip()]
    if not valid_descs:
        return "No description available"
    return " [SEP] ".join(valid_descs)


def serialize_row(column_names: List[str], row_values: List[str]) -> str:
    """
    序列化单行数据（用于推理阶段）
    
    格式: col1 : val1 [SEP] col2 : val2 [SEP] ...
    """
    parts = [f"{col} : {str(val)}" for col, val in zip(column_names, row_values) if val]
    return " [SEP] ".join(parts) if parts else "empty"


# ============================================================================
# 自适应层次聚类模块 (论文 Section 3.2)
# ============================================================================

class AdaptiveHierarchicalClustering:
    """
    自适应层次聚类模块
    
    使用 Affinity Propagation 发现数据中的层次聚类结构。
    通过调整 preference 参数实现多层聚类。
    
    层次聚类的 preference 计算 (论文公式):
    preference_l = min(s_ij) + (median(s_ij) - min(s_ij))/(L-1) * (l-1)
    """
    
    def __init__(
        self, 
        n_layers: int = 3, 
        max_iter: int = 500,
        convergence_iter: int = 50,
        damping: float = 0.7,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            n_layers: 聚类层数 L
            max_iter: 最大迭代次数
            convergence_iter: 收敛检测的迭代窗口
            damping: 阻尼因子 (0.5-1.0)，越大收敛越慢但更稳定
            logger: 日志记录器
        """
        self.n_layers = n_layers
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.damping = damping
        self.logger = logger or logging.getLogger(__name__)
        
        # 存储结果
        self.centroids_per_layer: List[np.ndarray] = []
        self.cluster_assignments_per_layer: List[np.ndarray] = []
        self.stats: List[Dict] = []
    
    def fit(self, embeddings: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        执行层次聚类
        
        Args:
            embeddings: [N, hidden_size] 嵌入矩阵
        
        Returns:
            centroids_per_layer: 每层的聚类中心列表
            cluster_assignments_per_layer: 每层的聚类分配列表
        """
        self.centroids_per_layer = []
        self.cluster_assignments_per_layer = []
        self.stats = []
        
        n_samples, n_dims = embeddings.shape
        self.logger.info(f"开始层次聚类: {n_samples} 样本, {n_dims} 维, {self.n_layers} 层")
        
        if n_samples < 2:
            self.logger.warning("样本数量不足，无法聚类")
            # 返回单一聚类
            for _ in range(self.n_layers):
                self.centroids_per_layer.append(embeddings.mean(axis=0, keepdims=True))
                self.cluster_assignments_per_layer.append(np.zeros(n_samples, dtype=np.int32))
            return self.centroids_per_layer, self.cluster_assignments_per_layer
        
        # 计算相似度矩阵 (负欧氏距离平方)
        similarity_matrix = -np.sum(
            (embeddings[:, None, :] - embeddings[None, :, :]) ** 2, 
            axis=2
        )
        
        min_sim = similarity_matrix.min()
        median_sim = np.median(similarity_matrix)
        
        self.logger.debug(f"相似度: min={min_sim:.4f}, median={median_sim:.4f}")
        
        # 对每一层执行聚类
        for layer in range(self.n_layers):
            # 计算 preference (论文公式)
            if self.n_layers > 1:
                preference = min_sim + (median_sim - min_sim) / (self.n_layers - 1) * layer
            else:
                preference = median_sim
            
            self.logger.debug(f"Layer {layer+1}: preference={preference:.4f}")
            
            start_time = time.time()
            
            try:
                ap = AffinityPropagation(
                    preference=preference,
                    max_iter=self.max_iter,
                    convergence_iter=self.convergence_iter,
                    damping=self.damping,
                    random_state=42
                )
                
                labels = ap.fit_predict(embeddings)
                elapsed = time.time() - start_time
                
                # 获取聚类中心
                if ap.cluster_centers_indices_ is not None and len(ap.cluster_centers_indices_) > 0:
                    centroids = ap.cluster_centers_
                    n_clusters = centroids.shape[0]
                    converged = ap.n_iter_ < self.max_iter
                else:
                    # 回退策略
                    self.logger.warning(f"Layer {layer+1}: 未找到聚类中心，使用回退策略")
                    centroids = embeddings.mean(axis=0, keepdims=True)
                    labels = np.zeros(n_samples, dtype=np.int32)
                    n_clusters = 1
                    converged = True
                
                # 记录统计信息
                stat = {
                    "layer": layer + 1,
                    "n_clusters": n_clusters,
                    "n_iterations": getattr(ap, 'n_iter_', 0),
                    "converged": converged,
                    "elapsed": elapsed,
                    "preference": preference
                }
                self.stats.append(stat)
                
                status = "✓ 收敛" if converged else "✗ 未收敛"
                self.logger.info(
                    f"  Layer {layer+1}: {n_clusters} 聚类, "
                    f"{stat['n_iterations']} 迭代, "
                    f"{elapsed:.2f}s {status}"
                )
                
                self.centroids_per_layer.append(centroids)
                self.cluster_assignments_per_layer.append(labels)
                
            except Exception as e:
                self.logger.error(f"Layer {layer+1} 聚类失败: {e}")
                centroids = embeddings.mean(axis=0, keepdims=True)
                labels = np.zeros(n_samples, dtype=np.int32)
                self.centroids_per_layer.append(centroids)
                self.cluster_assignments_per_layer.append(labels)
        
        return self.centroids_per_layer, self.cluster_assignments_per_layer
    
    def get_stats(self) -> List[Dict]:
        """获取聚类统计信息"""
        return self.stats


# ============================================================================
# POLYJOIN 主模型 (论文 Figure 2)
# ============================================================================

class POLYJOINModel(nn.Module):
    """
    POLYJOIN 模型
    
    结构:
    - 主encoder f_θ: 编码多键列数据 (可训练)
    - 动量encoder f_θ': 编码列描述 (动量更新)
    - 层次聚类模块: 发现描述的语义层次结构
    
    损失函数 (论文 Section 3.3):
    L_overall = α·L_key + β·L_cent
    """
    
    def __init__(
        self,
        model_path: str,
        temperature: float = 0.05,
        momentum: float = 0.9,
        n_clustering_layers: int = 3,
        alpha: float = 1.0,
        beta: float = 1.0,
        device: str = 'cuda',
        logger: Optional[logging.Logger] = None
    ):
        super().__init__()
        
        self.device = device
        self.temperature = temperature
        self.momentum = momentum
        self.alpha = alpha
        self.beta = beta
        self.logger = logger or logging.getLogger(__name__)
        # 加载主 encoder（相当于预训练 BERT）
        self.logger.info(f"加载模型: {model_path}")
        self.encoder = SentenceTransformer(model_path)
        self.encoder = self.encoder.to(device)
        self.hidden_size = self.encoder.get_sentence_embedding_dimension()
        # 创建动量 encoder（深拷贝）
        self.momentum_encoder = SentenceTransformer(model_path)
        self.momentum_encoder = self.momentum_encoder.to(device)
        self._init_momentum_encoder()
        # 聚类模块
        self.clustering = AdaptiveHierarchicalClustering(
            n_layers=n_clustering_layers,
            logger=self.logger
        )
        # 存储聚类结果
        self.centroids_per_layer: List[torch.Tensor] = []
        self.cluster_assignments_per_layer: List[np.ndarray] = []
        self.logger.info(f"模型初始化完成: hidden_size={self.hidden_size}")
    
    # ------------------- 动量 encoder 初始化 & 更新 -------------------
    def _init_momentum_encoder(self):
        """初始化动量encoder的参数，使其与主encoder一致，并冻结梯度。"""
        self.momentum_encoder.load_state_dict(self.encoder.state_dict())
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update(self):
        """
        动量更新 (论文 Equation 8)
        θ' ← z·θ' + (1-z)·θ
        """
        for param_q, param_k in zip(
            self.encoder.parameters(),
            self.momentum_encoder.parameters()
        ):
            param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data

    # ------------------- 统一的编码函数 -------------------
    def _encode_texts(
        self,
        texts: List[str],
        encoder: SentenceTransformer,
        enable_grad: bool = True
    ) -> torch.Tensor:
        """
        使用给定的 SentenceTransformer 对文本进行编码。
        enable_grad=True 时参与反向传播，False 时使用 no_grad。
        """
        if enable_grad:
            features = encoder.tokenize(texts)
            for k, v in features.items():
                if isinstance(v, torch.Tensor):
                    features[k] = v.to(self.device)
            outputs = encoder(features)
        else:
            with torch.no_grad():
                features = encoder.tokenize(texts)
                for k, v in features.items():
                    if isinstance(v, torch.Tensor):
                        features[k] = v.to(self.device)
                outputs = encoder(features)
        
        # SentenceTransformer 通常返回 dict，包含 "sentence_embedding"
        if isinstance(outputs, dict):
            if "sentence_embedding" in outputs:
                embeddings = outputs["sentence_embedding"]
            elif "pooler_output" in outputs:
                embeddings = outputs["pooler_output"]
            else:
                tensors = [v for v in outputs.values() if isinstance(v, torch.Tensor)]
                if not tensors:
                    raise RuntimeError("Encoder 输出中没有找到嵌入张量")
                embeddings = tensors[0]
        else:
            embeddings = outputs
        
        return embeddings
    
    # ------------------- 推理接口（不带梯度） -------------------
    @torch.no_grad()
    def encode_rows(self, texts: List[str]) -> torch.Tensor:
        """使用主 encoder 编码行数据（推理用）"""
        return self._encode_texts(texts, self.encoder, enable_grad=False)
    
    @torch.no_grad()
    def encode_descriptions(self, texts: List[str]) -> torch.Tensor:
        """使用动量 encoder 编码描述（推理/聚类用）"""
        return self._encode_texts(texts, self.momentum_encoder, enable_grad=False)
    
    # ------------------- 训练前向 -------------------
    def forward(self, row_texts: List[str], desc_texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数

        训练时：
        - row_emb: 来自主 encoder，允许梯度，用于更新 θ
        - desc_emb: 来自动量 encoder，在 no_grad 下计算，仅作为 teacher
        """
        # 主分支：带梯度（类似微调 BERT）
        row_emb = self._encode_texts(row_texts, self.encoder, enable_grad=True)
        # 动量分支：不回传梯度
        desc_emb = self._encode_texts(desc_texts, self.momentum_encoder, enable_grad=False)
        return row_emb, desc_emb
    
    def compute_key_wise_loss(
        self,
        row_embeddings: torch.Tensor,
        desc_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Key-wise 对比损失 (论文 Equation 5)
        
        L_key = Σᵢ -log( exp(hᵢ·h'ᵢ/τ) / Σⱼ exp(hᵢ·h'ⱼ/τ) )
        """
        row_emb = F.normalize(row_embeddings, dim=1)
        desc_emb = F.normalize(desc_embeddings, dim=1)
        
        batch_size = row_emb.size(0)
        logits = torch.matmul(row_emb, desc_emb.T) / self.temperature
        labels = torch.arange(batch_size, device=row_emb.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def compute_centroid_wise_loss(
        self,
        row_embeddings: torch.Tensor,
        sample_indices: List[int],
        centroids_per_layer: Optional[List[torch.Tensor]] = None,
        assignments_per_layer: Optional[List[np.ndarray]] = None
    ) -> torch.Tensor:
        """
        计算 Centroid-wise 对比损失 (论文 Equation 6)
        
        L_cent = -Σᵢ (1/L) Σₗ log( exp(hᵢ·eₗⱼ/τ) / Σₘ exp(hᵢ·eₗₘ/τ) )
        
        
        Args:
            row_embeddings: [B, hidden_size] 行嵌入
            sample_indices: batch 中每个样本在原始数据集中的全局索引
            centroids_per_layer: 每层的聚类中心
            assignments_per_layer: 每层的聚类分配（全数据集）
        """
        if centroids_per_layer is None:
            centroids_per_layer = self.centroids_per_layer
        if assignments_per_layer is None:
            assignments_per_layer = self.cluster_assignments_per_layer
        
        # 检查是否有有效的聚类结果
        if not centroids_per_layer or not assignments_per_layer:
            return torch.tensor(0.0, device=row_embeddings.device, requires_grad=True)
        
        if len(centroids_per_layer) != len(assignments_per_layer):
            self.logger.warning("centroids 和 assignments 层数不匹配")
            return torch.tensor(0.0, device=row_embeddings.device, requires_grad=True)
        
        row_emb = F.normalize(row_embeddings, dim=1)
        batch_size = row_emb.size(0)
        
        total_loss = torch.tensor(0.0, device=row_embeddings.device)
        valid_layers = 0
        
        for layer_idx, centroids in enumerate(centroids_per_layer):
            if centroids is None or len(centroids) == 0:
                continue
            
            # 转换 centroids 为 tensor
            if not isinstance(centroids, torch.Tensor):
                centroids = torch.from_numpy(centroids).float()
            centroids = centroids.to(row_embeddings.device)
            centroids = F.normalize(centroids, dim=1)
            
            # 计算与所有聚类中心的相似度
            similarities = torch.matmul(row_emb, centroids.T)  # [B, C_l]
            logits = similarities / self.temperature
            
            # 【关键修复】使用预计算的聚类分配作为标签
            layer_assignments = assignments_per_layer[layer_idx]  # 整个数据集的分配 [N]
            
            # 根据 sample_indices 获取当前 batch 中每个样本的真实聚类标签
            batch_labels = []
            for idx in sample_indices:
                if idx < len(layer_assignments):
                    batch_labels.append(int(layer_assignments[idx]))
                else:
                    # 索引越界时使用 argmax 作为回退
                    self.logger.warning(f"索引 {idx} 越界，使用 argmax 回退")
                    batch_labels.append(0)
            
            batch_labels = torch.tensor(
                batch_labels,
                dtype=torch.long,
                device=row_embeddings.device
            )
            
            # 确保标签在有效范围内
            n_clusters = centroids.size(0)
            batch_labels = torch.clamp(batch_labels, 0, n_clusters - 1)
            
            layer_loss = F.cross_entropy(logits, batch_labels)
            total_loss = total_loss + layer_loss
            valid_layers += 1
        
        if valid_layers > 0:
            return total_loss / valid_layers
        else:
            return torch.tensor(0.0, device=row_embeddings.device, requires_grad=True)
    
    def compute_loss(
        self,
        row_embeddings: torch.Tensor,
        desc_embeddings: torch.Tensor,
        sample_indices: List[int],
        centroids_per_layer: Optional[List[torch.Tensor]] = None,
        assignments_per_layer: Optional[List[np.ndarray]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失 (论文 Equation 7)
        
        L_overall = α·L_key + β·L_cent
        
        Args:
            row_embeddings: 行嵌入
            desc_embeddings: 描述嵌入
            sample_indices: batch 中每个样本的全局索引（用于查找聚类分配）
        """
        key_loss = self.compute_key_wise_loss(row_embeddings, desc_embeddings)
        cent_loss = self.compute_centroid_wise_loss(
            row_embeddings, 
            sample_indices,
            centroids_per_layer, 
            assignments_per_layer
        )
        
        total_loss = self.alpha * key_loss + self.beta * cent_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'key_loss': key_loss.item(),
            'cent_loss': cent_loss.item()
        }
        
        return total_loss, loss_dict
    
    def update_centroids(self, desc_embeddings: torch.Tensor):
        """更新聚类中心 (每个epoch后调用)"""
        embeddings_np = desc_embeddings.detach().cpu().numpy()
        centroids_list, assignments_list = self.clustering.fit(embeddings_np)
        
        self.centroids_per_layer = [
            torch.from_numpy(c).float().to(self.device)
            for c in centroids_list
        ]
        self.cluster_assignments_per_layer = assignments_list
        
        self.logger.info(f"聚类更新完成: {len(self.centroids_per_layer)} 层, "
                        f"聚类数: {[c.shape[0] for c in self.centroids_per_layer]}")
    
    # ------------------- 推理：与训练一致的序列化方式 -------------------
    @torch.no_grad()
    def encode_table_for_search(
        self,
        column_names: List[str],
        rows: List[List[str]],
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        为 joinability 搜索编码表格 (推理阶段)
            对每一行 x 的第 i 个 key 值 x_i，先分别编码得到 e_{x_i}，
            再在向量维度上拼接 h_x = [e_{x_1}; ...; e_{x_n}].

        这里我们把“key 值”具体实现为每个 cell 的文本：
            "<col_name> : <cell_value>"

        Args:
            column_names: 参与多键的列名列表，长度为 n (key 的个数)
            rows: 行列表，每行是这些列上的取值（长度同 column_names）
            batch_size: 编码时的批次大小

        Returns:
            row_embeddings: [n_rows, n_keys * hidden_size]
        """
        if not rows or not column_names:
            # 没有行或没有列，直接返回空
            return torch.empty(0, 0, device=self.device)

        n_rows = len(rows)
        n_cols = len(column_names)

        # 1) 为每个 (row, col) 构造一条句子： "col_name : cell_value"
        cell_texts: List[str] = []
        for r in range(n_rows):
            row_vals = rows[r]
            for c in range(n_cols):
                val = row_vals[c] if c < len(row_vals) else ""
                val = "" if val is None else str(val).strip()

                if val:
                    text = f"{column_names[c]} : {val}"
                else:
                    # 空值的情况，只保留列名，也可以根据需要改成特殊 token
                    text = f"{column_names[c]} :"
                cell_texts.append(text)

        if not cell_texts:
            return torch.empty(0, 0, device=self.device)
        
        # 2) 批量编码所有 cell，得到 e_{x_i}
        all_embs = []
        for i in range(0, len(cell_texts), batch_size):
            batch = cell_texts[i:i + batch_size]
            emb = self.encode_rows(batch)  # [B, hidden_size]
            all_embs.append(emb)

        if not all_embs:
            return torch.empty(0, 0, device=self.device)

        cell_embs = torch.cat(all_embs, dim=0)  # [n_rows * n_cols, hidden_size]

        # 3) reshape 成 [n_rows, n_cols, hidden_size]
        try:
            cell_embs = cell_embs.view(n_rows, n_cols, self.hidden_size)
        except RuntimeError as e:
            # 一般说明 rows 的长度和 column_names 不一致
            raise RuntimeError(
                f"encode_table_for_search: 期望 {n_rows * n_cols} 个 cell embedding，"
                f"实际得到 {cell_embs.size(0)} 个，请检查 rows 与 column_names 是否对齐。"
            ) from e

        # 4) 在特征维度上拼接：h_x = [e_{x_1}; ...; e_{x_n}]
        row_embs = cell_embs.reshape(n_rows, n_cols * self.hidden_size)  # [n_rows, n_cols * hidden_size]
        return row_embs
    
    @torch.no_grad()
    def compute_joinability_score(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        threshold: float = 0.5
    ) -> float:
        """
        计算语义多键 joinability 分数 (论文 Section 2)

        对每个 query 行 x:
            计算它与 candidate 表所有行 y 的 cos 相似度，
            取最大值 max_y cos(h_x, h_y)，与阈值 α 比较，
            统计"通过"的比例。
        """
        if query_embeddings.numel() == 0 or candidate_embeddings.numel() == 0:
            return 0.0

        query_emb = F.normalize(query_embeddings, dim=1)
        candidate_emb = F.normalize(candidate_embeddings, dim=1)
        
        similarity = torch.matmul(query_emb, candidate_emb.T)  # [n_query, n_cand]
        max_sim = similarity.max(dim=1)[0]
        
        n_matches = (max_sim > threshold).sum().item()
        n_query = query_emb.size(0)
        
        return n_matches / n_query if n_query > 0 else 0.0
    
    # ------------------- 保存 / 加载 -------------------
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict = None):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'metrics': metrics,
            'model_state_dict': self.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'centroids': [c.cpu().numpy() if isinstance(c, torch.Tensor) else c 
                         for c in self.centroids_per_layer],
            'cluster_assignments': self.cluster_assignments_per_layer,
            'config': {
                'temperature': self.temperature,
                'momentum': self.momentum,
                'alpha': self.alpha,
                'beta': self.beta,
                'hidden_size': self.hidden_size
            }
        }
        torch.save(checkpoint, path)
        self.logger.info(f"保存检查点: {path}")
    
    def load_checkpoint(self, path: str, device: str = None, load_clustering: bool = True):
        """加载模型检查点"""
        if device is None:
            device = self.device

        # 对新版本 PyTorch：显式设 weights_only=False
        # 对旧版本：这个参数不存在，会抛 TypeError，再退回老用法
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            # 旧版本 PyTorch 没有 weights_only 参数
            checkpoint = torch.load(path, map_location=device)

        self.load_state_dict(checkpoint['model_state_dict'])
        
        if load_clustering:
            if 'centroids' in checkpoint:
                self.centroids_per_layer = [
                    torch.from_numpy(c).float().to(device) if isinstance(c, np.ndarray) else c.to(device)
                    for c in checkpoint['centroids']
                ]
            if 'cluster_assignments' in checkpoint:
                self.cluster_assignments_per_layer = checkpoint['cluster_assignments']
        else:
            # 推理时显式清空，避免误用
            self.centroids_per_layer = []
            self.cluster_assignments_per_layer = []

        self.logger.info(f"加载检查点: {path}, epoch={checkpoint.get('epoch', 'unknown')}")
        return checkpoint


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("POLYJOIN 模型测试")
    print("=" * 70)
    
    # 测试序列化函数
    column_names = ["Country", "City"]
    column_values = [["USA", "UK", "France"], ["NYC", "London", "Paris"]]
    
    row_text = serialize_multi_key_columns(column_names, column_values)
    print(f"\n序列化多键列: {row_text}")
    
    descriptions = ["The name of the country", "The name of the city"]
    desc_text = serialize_descriptions(descriptions)
    print(f"序列化描述: {desc_text}")
    
    row = serialize_row(column_names, ["USA", "NYC"])
    print(f"序列化单行: {row}")
    
    print("\n✓ 序列化测试通过!")
