import argparse
import json
import os

class Config:
    """
    Configuration class for managing all parameters for model training and knowledge distillation
    
    Supports loading parameters from command line arguments and configuration files, with command line arguments having higher priority.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='BERT Knowledge Distillation with Embeddings')
        self._add_arguments()
        self.args = None
        self.model_pairs = []
        # TTM framework specific parameter defaults
        self.mode = "train"  # Running mode: train, evaluate, distill, mask_and_fill, manifold_init
        
        # Trajectory matching parameters
        self.trajectory_matching = True  # Whether to use trajectory matching
        self.trajectory_weight = 0.3  # Trajectory matching loss weight
        self.trajectory_points = 3  # Number of trajectory sampling points
        self.trajectory_distance = "mse"  # Trajectory distance metric: mse, cosine, lpips
        
        # Mask-and-Fill parameters
        self.mask_ratio = 0.15  # Mask ratio
        self.fill_candidates = 3  # Number of fill candidates
        self.mask_strategy = "uniform"  # Mask strategy: uniform, random, important
        self.fill_strategy = "gaussian"  # Fill strategy: gaussian, uniform, attention
        self.num_trajectories = 3  # Number of generated trajectories
        self.sample_size = 10  # Sample size (for initialization)
        
        # Manifold distribution initialization parameters
        self.manifold_method = "pca"  # Manifold learning method: pca, tsne
        self.latent_dim = 128  # Latent space dimension
        self.n_components = 10  # Number of manifold components
        self.tsne_perplexity = 30.0  # t-SNE perplexity parameter
        self.initialize_student_weights = True  # Whether to initialize student weights using manifold
        
        # Subset matching strategy parameters
        self.subset_strategy = "importance"  # Subset selection strategy: random, importance, diversity, hybrid
        self.subset_ratio = 0.5  # Subset ratio
        self.importance_metric = "attention"  # Importance metric: attention, gradient, activation
        self.diversity_metric = "cosine"  # Diversity metric: cosine, euclidean, manhattan
        self.layer_selection = "all"  # Layer selection strategy: all, skip, top
        self.skip_interval = 2  # Interval when layer_selection is 'skip'
        self.use_subset_matching = True  # Whether to use subset matching
    
    def _add_arguments(self):
        """
        Add command line arguments
        """
        # Basic parameters
        self.parser.add_argument('--config_file', type=str, default=None, help='Configuration file path')
        self.parser.add_argument('--output_dir', type=str, default='./results', help='Output results path')
        self.parser.add_argument('--log_dir', type=str, default='./logs', help='Log saving path')
        self.parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
        self.parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'distill', 'mask_and_fill', 'manifold_init'], default='train', help='Running mode')
        
        # Data parameters
        self.parser.add_argument('--dataset_name', type=str, default='glue', help='Dataset name')
        self.parser.add_argument('--dataset_config', type=str, default='sst2', help='Dataset configuration')
        self.parser.add_argument('--cache_dir', type=str, default='./cache_dataset', help='Dataset cache path')
        self.parser.add_argument('--num_positive_samples', type=int, default=50, help='Number of positive samples')
        self.parser.add_argument('--num_negative_samples', type=int, default=50, help='Number of negative samples')
        self.parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
        
        # Model parameters
        self.parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased', help='Tokenizer name')
        self.parser.add_argument('--tokenizer_cache_dir', type=str, default='./cache_model', help='Tokenizer cache path')
        self.parser.add_argument('--initial_model_path', type=str, default=None, help='Initial model path')
        
        # Training parameters
        self.parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
        self.parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
        self.parser.add_argument('--logging_steps', type=int, default=10, help='Logging steps')
        self.parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
        self.parser.add_argument('--distillation_steps', type=int, default=2, help='Number of distillation steps')
        
        # Distillation parameters
        self.parser.add_argument('--num_distillation_epochs', type=int, default=200, help='Number of distillation epochs')
        self.parser.add_argument('--layers_to_distill', type=int, default=12, help='Number of layers to distill')
        self.parser.add_argument('--save_embeddings_path', type=str, default='./emb_all_looptrained.pt', help='Path to save embeddings')
        
        # TTM framework specific parameters
        # Trajectory matching parameters
        self.parser.add_argument('--trajectory_matching', type=bool, default=True, help='Whether to use trajectory matching')
        self.parser.add_argument('--trajectory_weight', type=float, default=0.3, help='Trajectory matching loss weight')
        self.parser.add_argument('--trajectory_points', type=int, default=3, help='Number of trajectory sampling points')
        self.parser.add_argument('--trajectory_distance', type=str, choices=['mse', 'cosine', 'lpips'], default='mse', help='Trajectory distance metric')
        
        # Mask-and-Fill parameters
        self.parser.add_argument('--mask_ratio', type=float, default=0.15, help='Mask ratio')
        self.parser.add_argument('--fill_candidates', type=int, default=3, help='Number of fill candidates')
        self.parser.add_argument('--mask_strategy', type=str, choices=['uniform', 'random', 'important'], default='uniform', help='Mask strategy')
        self.parser.add_argument('--fill_strategy', type=str, choices=['gaussian', 'uniform', 'attention'], default='gaussian', help='Fill strategy')
        self.parser.add_argument('--num_trajectories', type=int, default=3, help='Number of generated trajectories')
        self.parser.add_argument('--sample_size', type=int, default=10, help='Sample size (for initialization)')
        
        # Manifold distribution initialization parameters
        self.parser.add_argument('--manifold_method', type=str, choices=['pca', 'tsne'], default='pca', help='Manifold learning method')
        self.parser.add_argument('--latent_dim', type=int, default=128, help='Latent space dimension')
        self.parser.add_argument('--n_components', type=int, default=10, help='Number of manifold components')
        self.parser.add_argument('--tsne_perplexity', type=float, default=30.0, help='t-SNE perplexity parameter')
        self.parser.add_argument('--initialize_student_weights', type=bool, default=True, help='Whether to initialize student weights using manifold')
        
        # Subset matching strategy parameters
        self.parser.add_argument('--subset_strategy', type=str, choices=['random', 'importance', 'diversity', 'hybrid'], default='importance', help='Subset selection strategy')
        self.parser.add_argument('--subset_ratio', type=float, default=0.5, help='Subset ratio')
        self.parser.add_argument('--importance_metric', type=str, choices=['attention', 'gradient', 'activation'], default='attention', help='Importance metric')
        self.parser.add_argument('--diversity_metric', type=str, choices=['cosine', 'euclidean', 'manhattan'], default='cosine', help='Diversity metric')
        self.parser.add_argument('--layer_selection', type=str, choices=['all', 'skip', 'top'], default='all', help='Layer selection strategy')
        self.parser.add_argument('--skip_interval', type=int, default=2, help='Interval when layer_selection is skip')
        self.parser.add_argument('--use_subset_matching', type=bool, default=True, help='Whether to use subset matching')
    
    def parse_args(self):
        """
        Parse command line arguments
        
        Returns:
            Config: The configuration object itself
        """
        self.args = self.parser.parse_args()
        
        # 如果指定了配置文件，加载配置文件中的参数
        if self.args.config_file and os.path.exists(self.args.config_file):
            with open(self.args.config_file, 'r') as f:
                config_dict = json.load(f)
                # 将配置文件中的参数更新到args中，但不覆盖命令行参数
                for key, value in config_dict.items():
                    if key != 'model_pairs' and not hasattr(self.args, key) or getattr(self.args, key) is None:
                        setattr(self.args, key, value)
                # 加载模型对列表
                if 'model_pairs' in config_dict:
                    self.model_pairs = config_dict['model_pairs']
        
        # 将args中的参数复制到self中
        for key, value in vars(self.args).items():
            setattr(self, key, value)
        
        return self
    
    def to_dict(self):
        """
        Convert configuration to dictionary
        
        Returns:
            dict: Configuration dictionary
        """
        config_dict = vars(self.args).copy()
        config_dict['model_pairs'] = self.model_pairs
        return config_dict
    
    def save(self, output_path):
        """
        Save configuration to file
        
        Args:
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

# 需要导入torch
import torch
