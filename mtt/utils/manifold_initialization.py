import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging

logger = logging.getLogger(__name__)


class ManifoldInitializer:
    """
    Manifold distribution initializer that preserves original semantic features through low-dimensional manifold analysis
    
    Implements a manifold learning-based initialization strategy to preserve semantic relationships
    between student and teacher models during knowledge distillation, ensuring compressed models
    maintain the original semantic structure.
    """
    
    def __init__(self, config):
        """
        Initialize the manifold distribution initializer
        
        Args:
            config: Configuration object containing manifold initialization parameters
        """
        self.config = config
        self.manifold_method = getattr(config, 'manifold_method', 'pca')  # 'pca' or 'tsne'
        self.latent_dim = getattr(config, 'latent_dim', 128)
        self.n_components = getattr(config, 'n_components', 10)
        self.perplexity = getattr(config, 'tsne_perplexity', 30.0)
        
        # 初始化流形学习器
        if self.manifold_method == 'pca':
            self.manifold_learner = PCA(n_components=self.n_components)
        elif self.manifold_method == 'tsne':
            self.manifold_learner = TSNE(
                n_components=self.n_components, 
                perplexity=self.perplexity,
                random_state=config.seed
            )
        else:
            raise ValueError(f"Unknown manifold method: {self.manifold_method}")
        
        logger.info(f"Initialized {self.manifold_method.upper()} manifold initializer with {self.n_components} components")
    
    def learn_manifold(self, embeddings, attention_mask=None):
        """
        Learn manifold structure from embedding vectors
        
        Args:
            embeddings: Input embedding vectors with shape [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask with shape [batch_size, seq_len] for filtering padded parts
            
        Returns:
            Learned manifold model and feature vectors
        """
        batch_size, seq_len, hidden_dim = embeddings.shape
        
        # 将嵌入向量化并应用掩码
        if attention_mask is not None:
            # 展平嵌入并过滤掉掩码为0的部分
            flat_embeddings = []
            for i in range(batch_size):
                valid_positions = attention_mask[i] > 0
                valid_embeddings = embeddings[i][valid_positions]
                flat_embeddings.append(valid_embeddings)
            
            if not flat_embeddings:
                logger.warning("No valid embeddings found after masking")
                return None
            
            # 合并所有有效嵌入
            flat_embeddings = torch.cat(flat_embeddings, dim=0).cpu().numpy()
        else:
            # 如果没有掩码，直接展平所有嵌入
            flat_embeddings = embeddings.view(-1, hidden_dim).cpu().numpy()
        
        # 检查数据是否足够进行流形学习
        if len(flat_embeddings) < self.n_components:
            logger.warning(f"Not enough samples for manifold learning: {len(flat_embeddings)} < {self.n_components}")
            # 减少组件数量
            adjusted_components = min(self.n_components, len(flat_embeddings) - 1)
            if adjusted_components < 1:
                adjusted_components = 1
                
            logger.info(f"Adjusting to {adjusted_components} components")
            
            if self.manifold_method == 'pca':
                self.manifold_learner = PCA(n_components=adjusted_components)
            elif self.manifold_method == 'tsne':
                # 对于t-SNE，调整困惑度以匹配样本数量
                adjusted_perplexity = min(self.perplexity, len(flat_embeddings) // 3)
                self.manifold_learner = TSNE(
                    n_components=adjusted_components, 
                    perplexity=adjusted_perplexity,
                    random_state=self.config.seed
                )
        
        # 应用流形学习
        logger.info(f"Learning {self.manifold_method} manifold from {len(flat_embeddings)} samples")
        manifold_features = self.manifold_learner.fit_transform(flat_embeddings)
        
        return {
            'manifold_learner': self.manifold_learner,
            'manifold_features': manifold_features,
            'original_shape': embeddings.shape,
            'attention_mask': attention_mask
        }
    
    def project_to_manifold(self, embeddings, manifold_model):
        """
        Project embedding vectors onto the learned manifold
        
        Args:
            embeddings: Embedding vectors to project
            manifold_model: Learned manifold model
            
        Returns:
            Projected manifold features
        """
        batch_size, seq_len, hidden_dim = embeddings.shape
        
        # 展平嵌入
        flat_embeddings = embeddings.view(-1, hidden_dim).cpu().numpy()
        
        # 投影到流形
        manifold_features = manifold_model['manifold_learner'].transform(flat_embeddings)
        
        # 重塑为批次格式
        manifold_features_reshaped = manifold_features.reshape(batch_size, seq_len, -1)
        
        return torch.tensor(manifold_features_reshaped, dtype=torch.float32, device=embeddings.device)
    
    def reconstruct_from_manifold(self, manifold_features, manifold_model, target_dim):
        """
        Reconstruct embedding vectors from manifold features
        
        Args:
            manifold_features: Manifold features
            manifold_model: Learned manifold model
            target_dim: Target embedding dimension
            
        Returns:
            Reconstructed embedding vectors
        """
        batch_size, seq_len, manifold_dim = manifold_features.shape
        
        # 展平流形特征
        flat_manifold_features = manifold_features.view(-1, manifold_dim).cpu().numpy()
        
        # 如果使用PCA，可以直接逆变换
        if self.manifold_method == 'pca':
            reconstructed = manifold_model['manifold_learner'].inverse_transform(flat_manifold_features)
        else:
            # 对于t-SNE等非线性方法，我们使用伪逆变换（近似）
            # 注意：这只是一个近似，t-SNE不保证可逆
            logger.warning("Reconstruction from t-SNE is an approximation only")
            
            # 获取学习器的嵌入映射矩阵（如果可用）
            if hasattr(manifold_model['manifold_learner'], 'embedding_'):
                # 使用嵌入和原始数据计算伪逆矩阵
                X_original = manifold_model['manifold_features']
                X_embedded = manifold_model['manifold_learner'].embedding_
                
                # 使用最小二乘求解逆变换
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression().fit(X_embedded, X_original)
                reconstructed = reg.predict(flat_manifold_features)
            else:
                # 如果无法获取嵌入矩阵，使用随机初始化
                logger.warning("Using random initialization for reconstruction")
                reconstructed = np.random.randn(*flat_manifold_features.shape[:1], target_dim)
        
        # 重塑为批次格式
        reconstructed_reshaped = reconstructed.reshape(batch_size, seq_len, -1)
        
        return torch.tensor(reconstructed_reshaped, dtype=torch.float32)
    
    def initialize_student_weights(self, teacher_weights, manifold_model):
        """
        Initialize student model weights using manifold features
        
        Args:
            teacher_weights: Teacher model weights
            manifold_model: Learned manifold model
            
        Returns:
            Initialized student model weights
        """
        # 提取教师模型的嵌入矩阵
        if hasattr(teacher_weights, 'weight'):
            teacher_embed = teacher_weights.weight.data
        else:
            teacher_embed = teacher_weights
        
        # 学习教师嵌入的流形
        teacher_manifold = self.learn_manifold(teacher_embed.unsqueeze(0))
        if teacher_manifold is None:
            logger.warning("Failed to learn teacher manifold, using random initialization")
            return torch.randn(*teacher_embed.shape[:-1], self.latent_dim, device=teacher_embed.device)
        
        # 投影到流形
        projected = self.project_to_manifold(teacher_embed.unsqueeze(0), teacher_manifold)
        
        # 将流形特征映射到学生模型的潜在空间
        # 这里我们直接使用投影特征作为学生模型的初始化权重
        student_weights = projected.squeeze(0)
        
        # 如果需要调整维度
        if student_weights.shape[-1] != self.latent_dim:
            # 使用线性投影调整维度
            projection_matrix = torch.nn.Linear(
                student_weights.shape[-1], self.latent_dim, bias=False
            ).weight.data.to(student_weights.device)
            student_weights = torch.matmul(student_weights, projection_matrix.t())
        
        logger.info(f"Student weights initialized from manifold features with shape {student_weights.shape}")
        return student_weights


def apply_manifold_initialization(config, embeddings, attention_mask=None):
    """
    Convenience function to apply manifold distribution initialization
    
    Args:
        config: Configuration object
        embeddings: Embedding vectors with shape [batch_size, seq_len, hidden_dim]
        attention_mask: Attention mask with shape [batch_size, seq_len]
        
    Returns:
        Manifold initialization result containing the learned manifold model and transformed features
    """
    try:
        # 创建流形初始化器
        initializer = ManifoldInitializer(config)
        
        # 学习流形
        manifold_result = initializer.learn_manifold(embeddings, attention_mask)
        
        if manifold_result is None:
            logger.error("Manifold learning failed")
            return None
        
        # 投影到流形
        manifold_features = initializer.project_to_manifold(
            embeddings, manifold_result
        )
        
        logger.info(f"Successfully applied manifold initialization")
        
        return {
            'manifold_model': manifold_result,
            'manifold_features': manifold_features,
            'initializer': initializer
        }
    
    except Exception as e:
        logger.error(f"Error during manifold initialization: {str(e)}", exc_info=True)
        raise
