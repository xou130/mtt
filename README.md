# Textual Trajectory Matching (TTM) Framework

This is a text dataset distillation framework designed for industrial environments to solve model deployment problems on resource-constrained edge devices. TTM achieves efficient knowledge distillation and model compression by aligning student model trajectories with expert trajectories derived from industrial data training.

## Project Overview

The TTM (Textual Trajectory Matching) framework implements an innovative text dataset distillation method designed for edge devices in industrial environments. Modern industrial environments generate a large amount of discrete text logs recording device status, error codes, and operational events. Deploying models on resource-constrained edge devices requires extreme data compression and fast inference.

## Framework Visualization

The following figure illustrates the performance comparison and trajectory matching visualization of the TTM framework:

![TTM Framework](figures/Figure2-ind.pdf)

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
mtt/
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
