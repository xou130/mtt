import torch
import torch.nn.functional as F
import random

def setup_optimizer(params, learning_rate=0.1):
    """
    Set up optimizer
    
    Args:
        params: List of parameters or parameter groups to optimize
        learning_rate: Learning rate
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    return torch.optim.Adam(params, lr=learning_rate)

def compute_distillation_loss(output_1, output_2, model_stu, model_tea, layers_to_distill=12):
    """
    Compute knowledge distillation loss, including hidden state loss, classifier loss, and layer weight loss
    
    Args:
        output_1: Student model output
        output_2: Teacher model output
        model_stu: Student model
        model_tea: Teacher model
        layers_to_distill: Number of layers to distill
        
    Returns:
        torch.Tensor: Total loss
    """
    hidden_states_1 = output_1.hidden_states
    hidden_states_2 = output_2.hidden_states
    
    # 计算隐藏状态损失
    loss_hidden_states = sum(
        [F.mse_loss(hidden_states_1[i], hidden_states_2[i]) for i in range(layers_to_distill)]
    )
    
    # 计算分类器损失
    classifier_1 = model_stu.classifier
    classifier_2 = model_tea.classifier
    loss_classifier = F.mse_loss(classifier_1.weight, classifier_2.weight)
    
    # 计算层权重损失
    loss_layer = 0
    for i in range(layers_to_distill):
        layer_weights_1 = model_stu.bert.encoder.layer[i].attention.self.query.weight
        layer_weights_2 = model_tea.bert.encoder.layer[i].attention.self.query.weight
        output_weights_1 = model_stu.bert.encoder.layer[i].output.dense.weight
        output_weights_2 = model_tea.bert.encoder.layer[i].output.dense.weight
        loss_layer += F.mse_loss(layer_weights_1, layer_weights_2) + F.mse_loss(output_weights_1, output_weights_2)
    
    # 总损失
    total_loss = loss_hidden_states + loss_classifier + loss_layer
    return total_loss

def create_training_schedule(num_epochs, num_models):
    """
    Create training schedule, randomly selecting model pairs to use
    
    Args:
        num_epochs: Number of training epochs
        num_models: Number of available model pairs
        
    Returns:
        list: List of model indices selected for each epoch
    """
    schedule = []
    for _ in range(num_epochs):
        select_trace = random.randint(0, num_models - 1)
        schedule.append(select_trace)
    return schedule

def load_model_pairs(config):
    """
    Load model pairs list from configuration
    
    Args:
        config: Configuration object
        
    Returns:
        list: List of model pairs, each element is [student model path, teacher model path]
    """
    # Load model path array from configuration
    if hasattr(config, 'model_pairs'):
        return config.model_pairs
    else:
        # Default model paths (extracted from original code)
        return [
            ["/home/dc/bert-mtt/dataset-distillation-with-attention-labels/src/bert-base-uncased-finetuned-sst2-dev1/checkpoint-2105","/home/dc/bert-mtt/dataset-distillation-with-attention-labels/src/bert-base-uncased-finetuned-sst2-dev1/checkpoint-12630"],
            ["/home/dc/bert-mtt/dataset-distillation-with-attention-labels/src/bert-base-uncased-finetuned-sst2-dev1/checkpoint-4210","/home/dc/bert-mtt/dataset-distillation-with-attention-labels/src/bert-base-uncased-finetuned-sst2-dev1/checkpoint-14735"],
            # 这里可以添加更多默认模型路径
        ]
