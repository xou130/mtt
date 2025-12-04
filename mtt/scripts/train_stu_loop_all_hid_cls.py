import os
import torch
import logging
from datasets import load_dataset
from transformers import BertTokenizer, Trainer, TrainingArguments, BertForSequenceClassification
import sys
import os

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtt.models import CustomBertForSequenceClassification
from mtt.datasets import EmbeddingDataset, CustomDataCollator
from mtt.utils import (
    setup_optimizer, 
    compute_distillation_loss, 
    create_training_schedule, 
    load_model_pairs,
    create_subset_matcher,
    create_trajectory_matcher,
    compute_combined_loss,
    collect_trajectories
)
from mtt.configs import Config

# Get logger
logger = logging.getLogger(__name__)

def prepare_embeddings(config):
    """
    Prepare embeddings for training
    
    Args:
        config: Configuration object
        
    Returns:
        torch.Tensor: Prepared embedding vectors
        list: Corresponding label list
    """
    logger.info("Loading dataset and preparing embeddings...")
    
    # Load SST-2 dataset
    dataset = load_dataset(config.dataset_name, config.dataset_config, cache_dir=config.cache_dir)
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name, cache_dir=config.tokenizer_cache_dir)
    
    # Sample by category
    positive_samples = dataset['train'].filter(lambda x: x['label'] == 1)
    negative_samples = dataset['train'].filter(lambda x: x['label'] == 0)
    
    # Randomly select samples
    positive_texts = positive_samples.shuffle(seed=42).select(range(config.num_positive_samples))
    negative_texts = negative_samples.shuffle(seed=42).select(range(config.num_negative_samples))
    
    # Extract texts and labels
    texts = positive_texts['sentence'] + negative_texts['sentence']
    labels = positive_texts['label'] + negative_texts['label']
    
    # If initial model path is not provided, use default path
    if config.initial_model_path is None:
        # Use the first model pair's student model as initial model
        model_pairs = load_model_pairs(config)
        config.initial_model_path = model_pairs[0][0]
    
    # Load initial model to get embeddings
    logger.info(f"Loading initial model from {config.initial_model_path}...")
    from transformers import BertConfig
    config_bert = BertConfig.from_pretrained(config.initial_model_path)
    stu_model = CustomBertForSequenceClassification.from_pretrained(config.initial_model_path, config=config_bert)
    
    # Get embeddings through tokenizer
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=config.max_length, return_tensors='pt')
    
    # Get BERT word embeddings
    with torch.no_grad():
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        embeddings = stu_model.bert.embeddings(input_ids)  # Get embeddings
    
    # Clean memory
    del stu_model
    torch.cuda.empty_cache()
    
    # Set embeddings as trainable
    embeddings.requires_grad_()
    
    logger.info(f"Embeddings prepared with shape: {embeddings.shape}")
    return embeddings, labels, tokenizer

def train_with_distillation(config, embeddings, labels, tokenizer):
    """
    Execute training process with knowledge distillation, integrating TTM framework's trajectory matching functionality
    
    Args:
        config: Configuration object
        embeddings: Embedding vectors
        labels: List of labels
        tokenizer: Tokenizer
    """
    logger.info("Starting training with knowledge distillation...")
    
    # Ensure output directories exist
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Load model pairs list
    model_pairs = load_model_pairs(config)
    num_model_pairs = len(model_pairs)
    logger.info(f"Loaded {num_model_pairs} model pairs for distillation")
    
    # Create training schedule
    schedule = create_training_schedule(config.num_distillation_epochs, num_model_pairs)
    logger.info(f"Training schedule created for {config.num_distillation_epochs} epochs")
    
    # Initialize subset matcher (if enabled)
    subset_matcher = None
    if config.use_subset_matching:
        subset_matcher = create_subset_matcher(config)
        logger.info(f"Initialized subset matcher with strategy: {config.subset_strategy}")
    
    # Initialize trajectory matcher (if enabled)
    if config.trajectory_matching:
        logger.info(f"Trajectory matching enabled with {config.trajectory_points} points")
    
    # Iterative training
    for epoch in range(config.num_distillation_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.num_distillation_epochs}")
        
        # Select model pair for current epoch
        select_trace = schedule[epoch]
        model_dir = model_pairs[select_trace]
        logger.info(f"Selected model pair: Student={model_dir[0]}, Teacher={model_dir[1]}")
        
        # Load student model
        model_stu = CustomBertForSequenceClassification.from_pretrained(model_dir[0])
        model_stu.to(config.device)
        
        # Create dataset and Trainer
        train_dataset = EmbeddingDataset(embeddings, labels)
        
        # Define training parameters
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            save_strategy="no",
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.batch_size,
            logging_dir=config.log_dir,
            logging_steps=config.logging_steps,
            report_to="none",
            fp16=torch.cuda.is_available(),  # Enable mixed precision training
            gradient_accumulation_steps=1
        )
        
        # Create Trainer
        trainer = Trainer(
            model=model_stu,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=CustomDataCollator(),
        )
        
        # Train student model
        trainer.train()
        
        # Save trained student model
        temp_model_dir = os.path.join(config.output_dir, "temp_student_model")
        trainer.save_model(temp_model_dir)
        
        # Clean memory
        del model_stu, trainer
        torch.cuda.empty_cache()
        
        # Load trained student model and teacher model
        model_stu = BertForSequenceClassification.from_pretrained(
            temp_model_dir, 
            output_hidden_states=True
        ).to(config.device)
        model_tea = BertForSequenceClassification.from_pretrained(
            model_dir[1], 
            output_hidden_states=True
        ).to(config.device)
        
        # Freeze model parameters
        for param in model_tea.parameters():
            param.requires_grad = False
        for param in model_stu.parameters():
            param.requires_grad = False
        
        # Set up optimizer (only optimize embeddings)
        optimizer = setup_optimizer([embeddings], lr=config.learning_rate)
        
        # Prepare attention mask (if not provided)
        batch_size, seq_len, _ = embeddings.shape
        attention_mask = torch.ones(batch_size, seq_len).to(config.device)
        
        # Execute distillation steps
        for step in range(config.distillation_steps):
            optimizer.zero_grad()
            
            # Forward pass
            output_1 = model_stu(
                inputs_embeds=embeddings.to(config.device),
                attention_mask=attention_mask
            )
            output_2 = model_tea(
                inputs_embeds=embeddings.to(config.device),
                attention_mask=attention_mask
            )
            
            # Calculate base distillation loss
            base_distillation_loss = compute_distillation_loss(
                output_1, output_2, model_stu, model_tea, config.layers_to_distill
            )
            
            # Initialize trajectory loss
            trajectory_loss = None
            
            # If trajectory matching is enabled, collect trajectories and calculate trajectory loss
            if config.trajectory_matching:
                # Collect trajectories
                trajectories = collect_trajectories(
                    model_stu,
                    model_tea,
                    embeddings.to(config.device),
                    attention_mask,
                    config,
                    subset_matcher
                )
                
                # Calculate trajectory loss
                from mtt.utils import TrajectoryMatcher
                matcher = TrajectoryMatcher(config)
                trajectory_loss = matcher.compute_trajectory_loss(
                    trajectories['student_trajectories'],
                    trajectories['teacher_trajectories']
                )
            
            # Calculate combined loss
            total_loss = compute_combined_loss(
                output_1[0],  # ce_loss
                base_distillation_loss,
                trajectory_loss,
                alpha=config.alpha,
                use_trajectory=config.trajectory_matching
            )
            
            # Backward pass and optimization
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([embeddings], config.max_grad_norm)
            
            optimizer.step()
            
            # Log loss information
            loss_info = f"Distillation step {step + 1}/{config.distillation_steps}, "
            loss_info += f"Total Loss: {total_loss.item():.6f}, "
            loss_info += f"Distillation Loss: {base_distillation_loss.item():.6f}"
            
            if config.trajectory_matching and trajectory_loss is not None:
                loss_info += f", Trajectory Loss: {trajectory_loss.item():.6f}"
            
            logger.info(loss_info)
        
        # Clean memory
        del model_tea, model_stu, optimizer
        torch.cuda.empty_cache()
    
    # Save final embeddings
    os.makedirs(os.path.dirname(config.save_embeddings_path), exist_ok=True)
    torch.save(embeddings, config.save_embeddings_path)
    logger.info(f"Final embeddings saved to {config.save_embeddings_path}")

def main(config=None):
    """
    Main function to execute the entire training process
    
    Args:
        config: Configuration object, creates a new one if None
    """
    try:
        # Create new configuration object if not provided
        if config is None:
            config = Config().parse_args()
        logger.info("Configuration prepared successfully")
        
        # Prepare embeddings
        embeddings, labels, tokenizer = prepare_embeddings(config)
        
        # Execute training and distillation
        train_with_distillation(config, embeddings, labels, tokenizer)
        
        logger.info("Training and distillation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()