# Textual Trajectory Matching (TTM) 框架

这是一个专为工业环境设计的文本数据集蒸馏框架，旨在解决资源受限的边缘设备上的模型部署问题。TTM通过对齐学生模型轨迹与从工业数据训练中派生的专家轨迹，实现高效的知识蒸馏和模型压缩。

## 项目概述

TTM（Textual Trajectory Matching）框架实现了一种创新的文本数据集蒸馏方法，专为工业环境中的边缘设备设计。现代工业环境生成大量离散文本日志，记录设备状态、错误代码和操作事件。在资源受限的边缘设备上部署模型需要极致的数据压缩和快速推理。

### 核心创新点

- **Mask-and-Fill初始化**：通过掩码和填充技术增强轨迹多样性，扩展语义表示
- **流形分布初始化**：通过低维流形分析保留原始语义特征
- **子集匹配策略**：通过对齐关键结构组件减少计算成本
- **轨迹匹配训练**：捕捉长期训练动态，而不仅是单步梯度或分布匹配

主要特点包括：
- 支持直接使用嵌入向量作为输入进行训练
- 实现了多层次的知识蒸馏（隐藏状态、分类器权重、层权重）
- 灵活的配置系统，支持命令行参数和配置文件
- 模块化设计，便于扩展和维护
- 统一的入口脚本，简化使用流程

## 项目结构

```
bert-mtt/
├── mtt/                        # 主包目录
│   ├── models/                 # 模型定义
│   │   ├── __init__.py
│   │   └── custom_bert.py      # 自定义BERT分类模型
│   ├── datasets/               # 数据集相关
│   │   ├── __init__.py
│   │   └── embedding_dataset.py # 嵌入数据集和数据收集器
│   ├── utils/                  # 工具函数
│   │   ├── __init__.py
│   │   ├── distillation_utils.py # 蒸馏相关工具函数
│   │   ├── mask_and_fill.py    # Mask-and-Fill初始化
│   │   ├── manifold_initialization.py # 流形分布初始化
│   │   ├── subset_matching.py  # 子集匹配策略
│   │   └── trajectory_matching.py # 轨迹匹配功能
│   ├── configs/                # 配置文件
│   │   ├── __init__.py
│   │   ├── config.py           # 配置类定义
│   │   └── example_config.json # 示例配置文件
│   └── scripts/                # 脚本文件
│       └── train_stu_loop_all_hid_cls.py # 主要训练脚本
├── run.py                      # 统一主入口脚本
├── train.py                    # 原始训练入口脚本
├── requirements.txt            # 依赖列表
└── README.md                   # 项目文档
```

## 安装指南

### 环境要求

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.10+
- Datasets 2.0+
- NumPy, SciPy, scikit-learn
- CUDA支持（推荐用于加速训练）

### 安装步骤

1. 克隆项目代码：

```bash
git clone [repository_url]
cd bert-mtt
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

TTM框架提供了统一的入口脚本`run.py`，支持多种运行模式。

### 基本用法

#### 1. 执行标准训练

```bash
python run.py --mode train \
    --dataset_name glue \
    --dataset_config sst2 \
    --model_name_or_path bert-base-uncased \
    --output_dir ./results/train \
    --batch_size 32 \
    --num_train_epochs 3
```

#### 2. 执行知识蒸馏

```bash
python run.py --mode distill \
    --dataset_name glue \
    --dataset_config sst2 \
    --model_name_or_path bert-base-uncased \
    --teacher_model_name_or_path roberta-base \
    --output_dir ./results/distill \
    --alpha 0.5 \
    --temperature 2.0 \
    --trajectory_matching true \
    --trajectory_weight 0.3 \
    --trajectory_points 3
```

#### 3. 执行Mask-and-Fill初始化

```bash
python run.py --mode mask_and_fill \
    --dataset_name glue \
    --dataset_config sst2 \
    --model_name_or_path bert-base-uncased \
    --output_dir ./results/mask_and_fill \
    --mask_ratio 0.15 \
    --fill_candidates 3 \
    --num_trajectories 3 \
    --sample_size 1000
```

#### 4. 执行流形分布初始化

```bash
python run.py --mode manifold_init \
    --dataset_name glue \
    --dataset_config sst2 \
    --model_name_or_path bert-base-uncased \
    --output_dir ./results/manifold_init \
    --manifold_method pca \
    --latent_dim 128 \
    --n_components 10 \
    --initialize_student_weights true
```

### 传统用法（兼容旧版）

```bash
python train.py --config_file mtt/configs/example_config.json
```

### 配置选项

TTM框架使用统一的配置系统，支持命令行参数和配置文件两种配置方式。

### 基本参数
- `--mode`: 运行模式（train/distill/mask_and_fill/manifold_init）
- `--config_file`: 配置文件路径
- `--output_dir`: 输出结果路径
- `--log_dir`: 日志保存路径
- `--device`: 使用的设备（cuda/cpu）

### 数据参数
- `--dataset_name`: 数据集名称（默认：glue）
- `--dataset_config`: 数据集配置（默认：sst2）
- `--cache_dir`: 数据集缓存路径
- `--max_length`: 最大序列长度（默认：128）
- `--batch_size`: 批量大小（默认：32）
- `--sample_size`: 样本数量（用于初始化过程）

### 模型参数
- `--model_name_or_path`: 模型名称或路径
- `--teacher_model_name_or_path`: 教师模型名称或路径
- `--tokenizer_name`: 分词器名称

### 训练参数
- `--num_train_epochs`: 训练轮数
- `--learning_rate`: 学习率
- `--logging_steps`: 日志记录步数
- `--max_grad_norm`: 梯度裁剪阈值

### TTM特有参数

#### 轨迹匹配参数
- `--trajectory_matching`: 是否使用轨迹匹配（默认true）
- `--trajectory_weight`: 轨迹匹配损失权重（默认0.3）
- `--trajectory_points`: 轨迹采样点数量（默认3）
- `--trajectory_distance`: 轨迹距离度量（mse/cosine/lpips）

#### Mask-and-Fill参数
- `--mask_ratio`: 掩码比例（默认0.15）
- `--fill_candidates`: 填充候选数（默认3）
- `--mask_strategy`: 掩码策略（uniform/random/important）
- `--fill_strategy`: 填充策略（gaussian/uniform/attention）
- `--num_trajectories`: 生成的轨迹数量（默认3）

#### 流形初始化参数
- `--manifold_method`: 流形学习方法（pca/tsne）
- `--latent_dim`: 潜在空间维度（默认128）
- `--n_components`: 流形组件数量（默认10）
- `--tsne_perplexity`: t-SNE的困惑度参数（默认30.0）
- `--initialize_student_weights`: 是否初始化学生模型权重

#### 子集匹配参数
- `--subset_strategy`: 子集选择策略（random/importance/diversity/hybrid）
- `--subset_ratio`: 子集比例（默认0.5）
- `--importance_metric`: 重要性度量（attention/gradient/activation）
- `--layer_selection`: 层选择策略（all/skip/top）

## 配置文件格式

配置文件使用JSON格式，示例：

```json
{
    "output_dir": "./results",
    "log_dir": "./logs",
    "batch_size": 4,
    "learning_rate": 0.05,
    "num_distillation_epochs": 100,
    "model_pairs": [
        ["/path/to/student_model_1", "/path/to/teacher_model_1"],
        ["/path/to/student_model_2", "/path/to/teacher_model_2"]
    ]
}
```

## 知识蒸馏原理

TTM框架实现了多层次的知识蒸馏，同时添加了轨迹匹配和优化的初始化方法：

1. **隐藏状态蒸馏**：使学生模型的中间层隐藏状态与教师模型相似
2. **分类器权重蒸馏**：使学生模型的分类器权重与教师模型相似
3. **层权重蒸馏**：蒸馏注意力查询权重和输出层权重
4. **轨迹匹配蒸馏**：对齐学生和教师模型在训练过程中的状态轨迹

通过优化的初始化方法（Mask-and-Fill和流形初始化）和子集匹配策略，实现更高效的知识迁移和计算资源利用。

## 核心功能模块

### 1. Mask-and-Fill初始化

增强轨迹多样性的初始化技术，通过随机掩码和智能填充扩展语义表示空间。

### 2. 流形分布初始化

通过低维流形分析保留原始语义特征的初始化方法。

### 3. 子集匹配策略

通过对齐关键结构组件减少计算成本的策略。

### 4. 轨迹匹配训练

TTM框架的核心功能，通过对齐学生轨迹和专家轨迹优化知识蒸馏。

## 注意事项

1. 确保有足够的GPU内存用于模型训练
2. 模型路径需要根据实际环境修改
3. 训练过程中会生成大量日志，建议定期清理
4. 嵌入向量文件可能会很大，请注意存储空间

## 扩展与自定义

可以通过以下方式扩展项目：

1. 在`mtt/models/`中添加新的模型定义
2. 在`mtt/datasets/`中添加新的数据集处理类
3. 在`mtt/utils/`中添加新的工具函数，特别是可以扩展：
   - `TrajectoryMatcher`类添加新的轨迹距离度量
   - `MaskAndFillInitializer`类添加新的掩码或填充策略
   - `ManifoldInitializer`类添加新的流形学习方法
   - `SubsetMatchingStrategy`类添加新的子集选择策略
4. 修改`mtt/configs/config.py`添加新的配置选项
5. 在`run.py`中添加新的运行模式

## 性能基准

在SST-2、MNLI-m和AGNews数据集上，使用BERT-BASE、RoBERTa-BASE、XLNet-BASE和LLaMA 3模型进行评估，TTM框架比现有方法实现了：

- **3.6%更高的准确率**
- **1.2倍更快的训练速度**
- **更低的计算资源消耗**

## 引用

如果您在研究中使用TTM框架，请引用我们的论文：

```
@article{ttm2024,
  title={Textual Trajectory Matching: Efficient Dataset Distillation for Industrial Edge Devices},
  author={Author Name and Co-authors},
  journal={Journal Name},
  year={2024},
  volume={XX},
  number={XX},
  pages={XXXX--XXXX}
}
```

## 故障排除

- **CUDA内存不足**：减小批量大小或使用更小的模型
- **模型加载错误**：检查模型路径是否正确，确保模型文件完整
- **数据集加载失败**：检查网络连接，或手动下载数据集到缓存目录

## 许可证

[在此添加许可证信息]

## 致谢

本项目基于以下开源库：
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets

---

# Textual Trajectory Matching (TTM) Framework

This is a text dataset distillation framework designed for industrial environments to solve model deployment problems on resource-constrained edge devices. TTM achieves efficient knowledge distillation and model compression by aligning student model trajectories with expert trajectories derived from industrial data training.

## Project Overview

The TTM (Textual Trajectory Matching) framework implements an innovative text dataset distillation method designed for edge devices in industrial environments. Modern industrial environments generate a large amount of discrete text logs recording device status, error codes, and operational events. Deploying models on resource-constrained edge devices requires extreme data compression and fast inference.

### Core Innovations

- **Mask-and-Fill Initialization**: Enhances trajectory diversity through masking and filling techniques, expanding semantic representation
- **Manifold Distribution Initialization**: Preserves original semantic features through low-dimensional manifold analysis
- **Subset Matching Strategy**: Reduces computational costs by aligning key structural components
- **Trajectory Matching Training**: Captures long-term training dynamics, not just single-step gradients or distribution matching

Key features include:
- Support for direct training with embedding vectors as input
- Implementation of multi-level knowledge distillation (hidden states, classifier weights, layer weights)
- Flexible configuration system supporting command-line parameters and configuration files
- Modular design for easy extension and maintenance
- Unified entry script for simplified usage

## Project Structure

```
bert-mtt/
├── mtt/                        # Main package directory
│   ├── models/                 # Model definitions
│   │   ├── __init__.py
│   │   └── custom_bert.py      # Custom BERT classification model
│   ├── datasets/               # Dataset related
│   │   ├── __init__.py
│   │   └── embedding_dataset.py # Embedding dataset and data collector
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── distillation_utils.py # Distillation related utility functions
│   │   ├── mask_and_fill.py    # Mask-and-Fill initialization
│   │   ├── manifold_initialization.py # Manifold distribution initialization
│   │   ├── subset_matching.py  # Subset matching strategy
│   │   └── trajectory_matching.py # Trajectory matching functionality
│   ├── configs/                # Configuration files
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration class definition
│   │   └── example_config.json # Example configuration file
│   └── scripts/                # Script files
│       └── train_stu_loop_all_hid_cls.py # Main training script
├── run.py                      # Unified main entry script
├── train.py                    # Original training entry script
├── requirements.txt            # Dependency list
└── README.md                   # Project documentation
```

## Installation Guide

### Environment Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.10+
- Datasets 2.0+
- NumPy, SciPy, scikit-learn
- CUDA support (recommended for accelerated training)

### Installation Steps

1. Clone the project code:

```bash
git clone [repository_url]
cd bert-mtt
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The TTM framework provides a unified entry script `run.py` that supports multiple operation modes.

### Basic Usage

#### 1. Execute Standard Training

```bash
python run.py --mode train \
    --dataset_name glue \
    --dataset_config sst2 \
    --model_name_or_path bert-base-uncased \
    --output_dir ./results/train \
    --batch_size 32 \
    --num_train_epochs 3
```

#### 2. Execute Knowledge Distillation

```bash
python run.py --mode distill \
    --dataset_name glue \
    --dataset_config sst2 \
    --model_name_or_path bert-base-uncased \
    --teacher_model_name_or_path roberta-base \
    --output_dir ./results/distill \
    --alpha 0.5 \
    --temperature 2.0 \
    --trajectory_matching true \
    --trajectory_weight 0.3 \
    --trajectory_points 3
```

#### 3. Execute Mask-and-Fill Initialization

```bash
python run.py --mode mask_and_fill \
    --dataset_name glue \
    --dataset_config sst2 \
    --model_name_or_path bert-base-uncased \
    --output_dir ./results/mask_and_fill \
    --mask_ratio 0.15 \
    --fill_candidates 3 \
    --num_trajectories 3 \
    --sample_size 1000
```

#### 4. Execute Manifold Distribution Initialization

```bash
python run.py --mode manifold_init \
    --dataset_name glue \
    --dataset_config sst2 \
    --model_name_or_path bert-base-uncased \
    --output_dir ./results/manifold_init \
    --manifold_method pca \
    --latent_dim 128 \
    --n_components 10 \
    --initialize_student_weights true
```

### Legacy Usage (Backward Compatibility)

```bash
python train.py --config_file mtt/configs/example_config.json
```

### Configuration Options

The TTM framework uses a unified configuration system that supports both command-line parameters and configuration files.

### Basic Parameters
- `--mode`: Operation mode (train/distill/mask_and_fill/manifold_init)
- `--config_file`: Configuration file path
- `--output_dir`: Output result path
- `--log_dir`: Log saving path
- `--device`: Device to use (cuda/cpu)

### Data Parameters
- `--dataset_name`: Dataset name (default: glue)
- `--dataset_config`: Dataset configuration (default: sst2)
- `--cache_dir`: Dataset cache path
- `--max_length`: Maximum sequence length (default: 128)
- `--batch_size`: Batch size (default: 32)
- `--sample_size`: Number of samples (for initialization processes)

### Model Parameters
- `--model_name_or_path`: Model name or path
- `--teacher_model_name_or_path`: Teacher model name or path
- `--tokenizer_name`: Tokenizer name

### Training Parameters
- `--num_train_epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--logging_steps`: Logging steps
- `--max_grad_norm`: Gradient clipping threshold

### TTM-Specific Parameters

#### Trajectory Matching Parameters
- `--trajectory_matching`: Whether to use trajectory matching (default true)
- `--trajectory_weight`: Trajectory matching loss weight (default 0.3)
- `--trajectory_points`: Number of trajectory sampling points (default 3)
- `--trajectory_distance`: Trajectory distance metric (mse/cosine/lpips)

#### Mask-and-Fill Parameters
- `--mask_ratio`: Mask ratio (default 0.15)
- `--fill_candidates`: Number of fill candidates (default 3)
- `--mask_strategy`: Mask strategy (uniform/random/important)
- `--fill_strategy`: Fill strategy (gaussian/uniform/attention)
- `--num_trajectories`: Number of generated trajectories (default 3)

#### Manifold Initialization Parameters
- `--manifold_method`: Manifold learning method (pca/tsne)
- `--latent_dim`: Latent space dimension (default 128)
- `--n_components`: Number of manifold components (default 10)
- `--tsne_perplexity`: t-SNE perplexity parameter (default 30.0)
- `--initialize_student_weights`: Whether to initialize student model weights

#### Subset Matching Parameters
- `--subset_strategy`: Subset selection strategy (random/importance/diversity/hybrid)
- `--subset_ratio`: Subset ratio (default 0.5)
- `--importance_metric`: Importance metric (attention/gradient/activation)
- `--layer_selection`: Layer selection strategy (all/skip/top)

## Configuration File Format

Configuration files use JSON format, example:

```json
{
    "output_dir": "./results",
    "log_dir": "./logs",
    "batch_size": 4,
    "learning_rate": 0.05,
    "num_distillation_epochs": 100,
    "model_pairs": [
        ["/path/to/student_model_1", "/path/to/teacher_model_1"],
        ["/path/to/student_model_2", "/path/to/teacher_model_2"]
    ]
}
```

## Knowledge Distillation Principles

The TTM framework implements multi-level knowledge distillation, along with trajectory matching and optimized initialization methods:

1. **Hidden State Distillation**: Makes the intermediate hidden states of the student model similar to those of the teacher model
2. **Classifier Weight Distillation**: Makes the classifier weights of the student model similar to those of the teacher model
3. **Layer Weight Distillation**: Distills attention query weights and output layer weights
4. **Trajectory Matching Distillation**: Aligns the state trajectories of student and teacher models during training

Through optimized initialization methods (Mask-and-Fill and manifold initialization) and subset matching strategies, more efficient knowledge transfer and computational resource utilization are achieved.

## Core Function Modules

### 1. Mask-and-Fill Initialization

An initialization technique that enhances trajectory diversity by extending semantic representation space through random masking and intelligent filling.

### 2. Manifold Distribution Initialization

An initialization method that preserves original semantic features through low-dimensional manifold analysis.

### 3. Subset Matching Strategy

A strategy that reduces computational costs by aligning key structural components.

### 4. Trajectory Matching Training

The core functionality of the TTM framework, optimizing knowledge distillation by aligning student trajectories with expert trajectories.

## Notes

1. Ensure sufficient GPU memory for model training
2. Model paths need to be modified according to the actual environment
3. A large number of logs will be generated during training, regular cleaning is recommended
4. Embedding vector files may be large, please pay attention to storage space

## Extension and Customization

The project can be extended in the following ways:

1. Add new model definitions in `mtt/models/`
2. Add new dataset processing classes in `mtt/datasets/`
3. Add new utility functions in `mtt/utils/`, especially you can extend:
   - `TrajectoryMatcher` class to add new trajectory distance metrics
   - `MaskAndFillInitializer` class to add new masking or filling strategies
   - `ManifoldInitializer` class to add new manifold learning methods
   - `SubsetMatchingStrategy` class to add new subset selection strategies
4. Modify `mtt/configs/config.py` to add new configuration options
5. Add new operation modes in `run.py`

## Performance Benchmarks

On SST-2, MNLI-m, and AGNews datasets, using BERT-BASE, RoBERTa-BASE, XLNet-BASE, and LLaMA 3 models for evaluation, the TTM framework achieves:

- **3.6% higher accuracy**
- **1.2x faster training speed**
- **Lower computational resource consumption**

## Citation

If you use the TTM framework in your research, please cite our paper:

```
@article{ttm2024,
  title={Textual Trajectory Matching: Efficient Dataset Distillation for Industrial Edge Devices},
  author={Author Name and Co-authors},
  journal={Journal Name},
  year={2024},
  volume={XX},
  number={XX},
  pages={XXXX--XXXX}
}
```

## Troubleshooting

- **CUDA out of memory**: Reduce batch size or use a smaller model
- **Model loading error**: Check if the model path is correct and ensure model files are complete
- **Dataset loading failure**: Check network connection, or manually download the dataset to the cache directory

## License

[Add license information here]

## Acknowledgments

This project is based on the following open-source libraries:
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets