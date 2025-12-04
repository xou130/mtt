#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Textual Trajectory Matching (TTM) - Unified Entry Script

This is the unified entry point for the TTM framework, supporting the Textual Trajectory Matching knowledge distillation method.
Training parameters can be specified through command line arguments or configuration files, supporting multiple operation modes.

Usage examples:
  python train.py --mode train --config configs/example_config.json
  python train.py --mode evaluate --model_path ./results/model.pt --dataset_name glue/sst2
  python train.py --mode distill --teacher_model bert-base-uncased --student_model ./student_model
"""

import sys
import os
import logging

# Add project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mtt.configs import Config

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Set up logging configuration
    
    Args:
        log_level: Logging level
        log_file: Log file path, if None only output to console
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def run_training(config):
    """
    Run training mode
    
    Args:
        config: Configuration object
    """
    from mtt.scripts.train_stu_loop_all_hid_cls import main as train_main
    train_main(config)

def run_evaluation(config):
    """
    Run evaluation mode
    
    Args:
        config: Configuration object
    """
    logging.info("Running evaluation mode...")
    # Evaluation logic can be implemented here
    # TODO: Implement evaluation functionality
    logging.warning("Evaluation mode not yet implemented")

def run_distillation(config):
    """
    Run distillation mode (TTM framework core)
    
    Args:
        config: Configuration object
    """
    logging.info("Running Textual Trajectory Matching distillation...")
    # Use the modified training script
    from mtt.scripts.train_stu_loop_all_hid_cls import main as distill_main
    distill_main(config)

def run_mask_and_fill(config):
    """
    Run Mask-and-Fill initialization
    
    Args:
        config: Configuration object
    """
    logging.info("Running Mask-and-Fill initialization...")
    try:
        # Import necessary modules
        from mtt.utils import apply_mask_and_fill
        from transformers import BertTokenizer
        from datasets import load_dataset
        import torch
        
        # Create output directory
        output_dir = getattr(config, 'output_dir', './results_mask_fill')
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据集和分词器
        logging.info(f"Loading dataset {config.dataset_name}")
        dataset = load_dataset(config.dataset_name, config.dataset_config, cache_dir=config.cache_dir)
        tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name, cache_dir=config.tokenizer_cache_dir)
        
        # 准备示例数据
        sample_size = getattr(config, 'sample_size', 10)
        samples = dataset['train'].select(range(min(sample_size, len(dataset['train']))))
        
        # 预处理文本并获取嵌入
        def tokenize_function(examples):
            return tokenizer(examples['sentence'] if 'sentence' in examples else examples['text'], 
                            padding='max_length', truncation=True, max_length=config.max_length)
        
        tokenized_samples = samples.map(tokenize_function, batched=True)
        
        # 从预训练模型获取嵌入（这里简化处理，实际应该使用预训练模型的嵌入层）
        # 为了演示，我们使用随机嵌入
        batch_size = len(tokenized_samples)
        embeddings = torch.randn(batch_size, config.max_length, 768).to(config.device)  # 假设BERT-base嵌入维度
        attention_mask = torch.tensor(tokenized_samples['attention_mask']).to(config.device)
        
        # 应用Mask-and-Fill生成多样化轨迹
        num_trajectories = getattr(config, 'num_trajectories', 3)
        logging.info(f"Generating {num_trajectories} diverse trajectories")
        trajectories = apply_mask_and_fill(config, embeddings, attention_mask, num_trajectories)
        
        # 保存结果
        output_path = os.path.join(output_dir, 'mask_fill_trajectories.pt')
        torch.save({
            'base_embeddings': embeddings,
            'trajectories': trajectories,
            'attention_mask': attention_mask
        }, output_path)
        
        logging.info(f"Mask-and-Fill initialization completed successfully")
        logging.info(f"Generated trajectories saved to {output_path}")
        
    except Exception as e:
        logging.error(f"Error during Mask-and-Fill initialization: {str(e)}", exc_info=True)
        raise

def run_manifold_initialization(config):
    """
    Run manifold distribution initialization
    
    Args:
        config: Configuration object
    """
    logging.info("Running manifold distribution initialization...")
    try:
        # Import necessary modules
        from mtt.utils import apply_manifold_initialization
        from transformers import BertTokenizer, BertModel
        from datasets import load_dataset
        import torch
        
        # Create output directory
        output_dir = getattr(config, 'output_dir', './results_manifold')
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据集和分词器
        logging.info(f"Loading dataset {config.dataset_name}")
        dataset = load_dataset(config.dataset_name, config.dataset_config, cache_dir=config.cache_dir)
        tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name, cache_dir=config.tokenizer_cache_dir)
        
        # 准备示例数据
        sample_size = getattr(config, 'sample_size', 50)
        samples = dataset['train'].select(range(min(sample_size, len(dataset['train']))))
        
        # 预处理文本
        def tokenize_function(examples):
            return tokenizer(examples['sentence'] if 'sentence' in examples else examples['text'], 
                            padding='max_length', truncation=True, max_length=config.max_length)
        
        tokenized_samples = samples.map(tokenize_function, batched=True)
        
        # 加载预训练模型获取真实嵌入
        device = config.device
        model = BertModel.from_pretrained(config.model_name_or_path, cache_dir=config.model_cache_dir).to(device)
        model.eval()
        
        # 提取嵌入
        input_ids = torch.tensor(tokenized_samples['input_ids']).to(device)
        attention_mask = torch.tensor(tokenized_samples['attention_mask']).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state
        
        # 应用流形分布初始化
        logging.info(f"Applying {config.manifold_method} manifold initialization with {config.n_components} components")
        manifold_result = apply_manifold_initialization(config, embeddings, attention_mask)
        
        if manifold_result is None:
            raise RuntimeError("Manifold initialization failed")
        
        # 保存结果
        output_path = os.path.join(output_dir, 'manifold_features.pt')
        torch.save({
            'original_embeddings': embeddings,
            'manifold_features': manifold_result['manifold_features'],
            'attention_mask': attention_mask,
            'manifold_method': config.manifold_method,
            'n_components': config.n_components
        }, output_path)
        
        # If needed, demonstrate student model weight initialization
        if hasattr(config, 'initialize_student_weights') and config.initialize_student_weights:
            # Get embedding layer weights from teacher model
            teacher_embedding = model.embeddings.word_embeddings.weight.data
            
            # 使用流形初始化学生模型权重
            student_weights = manifold_result['initializer'].initialize_student_weights(
                teacher_embedding, manifold_result['manifold_model']
            )
            
            # 保存学生模型初始化权重
            student_weights_path = os.path.join(output_dir, 'student_initial_weights.pt')
            torch.save(student_weights, student_weights_path)
            logging.info(f"Student model weights initialized and saved to {student_weights_path}")
        
        logging.info(f"Manifold distribution initialization completed successfully")
        logging.info(f"Manifold features saved to {output_path}")
        
    except Exception as e:
        logging.error(f"Error during manifold distribution initialization: {str(e)}", exc_info=True)
        raise

def main():
    """
    Main function, parses arguments and runs corresponding functionality based on mode
    """
    # Create configuration object and add additional command line arguments
    config = Config()
    config.parser.add_argument('--mode', type=str, default='distill', 
                             choices=['train', 'evaluate', 'distill', 'mask_and_fill', 'manifold_init'],
                             help='Operation mode: train, evaluate, distill, mask_and_fill (Mask-and-Fill initialization), manifold_init (manifold distribution initialization)')
    
    # Parse arguments
    config.parse_args()
    
    # Set up logging
    log_file = os.path.join(config.log_dir, f"{config.mode}_log.txt") if hasattr(config, 'log_dir') else None
    setup_logging(log_level=logging.INFO, log_file=log_file)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting TTM framework in {config.mode} mode")
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Run corresponding functionality based on mode
    mode_handlers = {
        'train': run_training,
        'evaluate': run_evaluation,
        'distill': run_distillation,
        'mask_and_fill': run_mask_and_fill,
        'manifold_init': run_manifold_initialization
    }
    
    if config.mode in mode_handlers:
        mode_handlers[config.mode](config)
    else:
        logger.error(f"Unknown mode: {config.mode}")
        sys.exit(1)
    
    logger.info("TTM framework execution completed")

if __name__ == "__main__":
    main()