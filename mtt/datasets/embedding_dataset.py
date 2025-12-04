from torch.utils.data import Dataset
import torch

class EmbeddingDataset(Dataset):
    """
    Custom dataset class for handling embedding vectors
    
    This dataset class is used to store and access pre-generated word embedding vectors and their corresponding labels,
    suitable for training with pre-generated embeddings during knowledge distillation.
    """
    def __init__(self, embeddings, labels):
        """
        Initialize the dataset
        
        Args:
            embeddings: Pre-generated word embedding vectors, shape [num_samples, seq_length, hidden_size]
            labels: Corresponding label list, shape [num_samples]
        """
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        """
        Return the number of samples in the dataset
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get the sample at the specified index
        
        Args:
            idx: Sample index
            
        Returns:
            dict: Dictionary containing inputs_embeds, attention_mask, and labels
        """
        return {
            'inputs_embeds': self.embeddings[idx],  # Return embeddings instead of input_ids
            'attention_mask': torch.tensor([1] * self.embeddings.shape[1]),  # Assume all tokens are valid
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class CustomDataCollator:
    """
    Custom data collator class
    
    Used to combine multiple samples into a batch, handling embedding vectors instead of input_ids.
    This custom collator is required when using Hugging Face's Trainer.
    """
    def __call__(self, features):
        """
        Combine multiple samples into a batch
        
        Args:
            features: List of samples, each sample is a dictionary containing inputs_embeds, attention_mask, and labels
            
        Returns:
            dict: Dictionary containing batched data
        """
        
        batch = {
            'inputs_embeds': torch.stack([x['inputs_embeds'] for x in features]),  
            'attention_mask': torch.stack([x['attention_mask'] for x in features]),
            'labels': torch.stack([x['labels'] for x in features])
        }
        return batch
