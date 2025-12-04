import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
import random
import logging

logger = logging.getLogger(__name__)

class MaskAndFillInitializer:
    """
    Mask-and-Fill initializer for enhancing trajectory diversity and expanding semantic representation
    
    This module implements the Mask-and-Fill strategy in the Textual Trajectory Matching (TTM) framework through:
    1. Randomly masking parts of the embedding vectors
    2. Using pre-trained language models to fill masked positions
    3. Generating diverse semantic representations
    """
    
    def __init__(self, config):
        """
        Initialize the Mask-and-Fill module
        
        Args:
            config: Configuration object containing necessary parameters
        """
        self.config = config
        self.mask_ratio = getattr(config, 'mask_ratio', 0.3)  # Mask ratio
        self.fill_top_k = getattr(config, 'fill_top_k', 3)   # Top-k candidates for filling
        self.tokenizer_cache_dir = getattr(config, 'tokenizer_cache_dir', './cache_model')
        
        self.tokenizer = None
        self.mask_filler = None
        self._load_mask_filler()
    
    def _load_mask_filler(self):
        """
        Load pre-trained model for filling masks
        """
        try:
            model_name = getattr(self.config, 'mask_filler_model', 'bert-base-uncased')
            logger.info(f"Loading mask filler model: {model_name}")
            
            self.tokenizer = BertTokenizer.from_pretrained(
                model_name, 
                cache_dir=self.tokenizer_cache_dir
            )
            self.mask_filler = BertForMaskedLM.from_pretrained(
                model_name, 
                cache_dir=self.tokenizer_cache_dir
            ).to(self.config.device)
            self.mask_filler.eval()
            logger.info("Mask filler model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load mask filler model: {str(e)}")
            raise
    
    def mask_embeddings(self, embeddings, attention_mask=None):
        """
        Apply masking to embedding vectors
        
        Args:
            embeddings: Input embedding vectors with shape [batch_size, seq_length, hidden_size]
            attention_mask: Attention mask with shape [batch_size, seq_length]
            
        Returns:
            tuple: (masked_embeddings, mask_indices) - Masked embeddings and mask positions
        """
        batch_size, seq_length, _ = embeddings.shape
        masked_embeddings = embeddings.clone()
        mask_indices = []
        
        mask_token_id = self.tokenizer.mask_token_id
        mask_embedding = self.mask_filler.get_input_embeddings()(
            torch.tensor([mask_token_id], device=embeddings.device)
        ).repeat(batch_size, seq_length, 1)
        
        for i in range(batch_size):
            if attention_mask is not None:
                valid_length = attention_mask[i].sum().item()
                valid_indices = list(range(1, valid_length - 1))
            else:
                valid_indices = list(range(seq_length))
            
            num_to_mask = max(1, int(len(valid_indices) * self.mask_ratio))
            selected_indices = random.sample(valid_indices, num_to_mask)
            mask_indices.append(selected_indices)
            
            # 应用掩码
            for idx in selected_indices:
                masked_embeddings[i, idx] = mask_embedding[i, idx]
        
        return masked_embeddings, mask_indices
    
    def fill_masked_positions(self, masked_embeddings, mask_indices, attention_mask=None):
        """
        Fill masked positions using pre-trained model
        
        Args:
            masked_embeddings: Masked embedding vectors
            mask_indices: List of mask positions for each sample
            attention_mask: Attention mask
            
        Returns:
            torch.Tensor: Filled embedding vectors
        """
        batch_size, seq_length, hidden_size = masked_embeddings.shape
        filled_embeddings = masked_embeddings.clone()
        
        with torch.no_grad():
            for i in range(batch_size):
                if not mask_indices[i]:
                    continue
                
                sample_embedding = masked_embeddings[i:i+1]
                sample_mask = attention_mask[i:i+1] if attention_mask is not None else None
                
                outputs = self.mask_filler(
                    inputs_embeds=sample_embedding,
                    attention_mask=sample_mask,
                    output_hidden_states=True
                )
                
                final_hidden = outputs.hidden_states[-1][0]
                
                for idx in mask_indices[i]:
                    filled_embeddings[i, idx] = final_hidden[idx]
        
        return filled_embeddings
    
    def generate_diverse_trajectories(self, base_embeddings, num_trajectories=3, attention_mask=None):
        """
        Generate diverse trajectory embeddings
        
        Args:
            base_embeddings: Base embedding vectors
            num_trajectories: Number of trajectories to generate
            attention_mask: Attention mask
            
        Returns:
            list: List of diverse trajectory embeddings
        """
        trajectories = []
        
        for i in range(num_trajectories):
            masked_embeddings, mask_indices = self.mask_embeddings(base_embeddings, attention_mask)
            filled_embeddings = self.fill_masked_positions(masked_embeddings, mask_indices, attention_mask)
            trajectories.append(filled_embeddings)
        
        logger.info(f"Generated {num_trajectories} diverse trajectories using Mask-and-Fill")
        return trajectories
    
    def __call__(self, embeddings, attention_mask=None, num_trajectories=3):
        """
        Call interface to generate diverse trajectory embeddings
        
        Args:
            embeddings: Input embedding vectors
            attention_mask: Attention mask
            num_trajectories: Number of trajectories
            
        Returns:
            list: List of diverse trajectory embeddings
        """
        return self.generate_diverse_trajectories(embeddings, num_trajectories, attention_mask)

def apply_mask_and_fill(config, embeddings, attention_mask=None, num_trajectories=3):
    """
    Convenience function to apply Mask-and-Fill initialization
    
    Args:
        config: Configuration object
        embeddings: Input embedding vectors
        attention_mask: Attention mask
        num_trajectories: Number of trajectories to generate
        
    Returns:
        list: List of diverse trajectory embeddings
    """
    initializer = MaskAndFillInitializer(config)
    return initializer(embeddings, attention_mask, num_trajectories)