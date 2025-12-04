import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from scipy.spatial import distance

logger = logging.getLogger(__name__)


class SubsetMatchingStrategy:
    """
    Subset matching strategy module for reducing computational costs by aligning key structural components
    
    Implements various subset selection and matching strategies to optimize computational efficiency
    during knowledge distillation while maintaining model performance. Supports subset selection methods
    based on importance, diversity, and structural similarity.
    """
    
    def __init__(self, config):
        """
        Initialize subset matching strategy
        
        Args:
            config: Configuration object containing subset matching parameters
        """
        self.config = config
        self.strategy_type = getattr(config, 'subset_strategy', 'random')  # 'random', 'importance', 'diversity', 'hybrid'
        self.subset_ratio = getattr(config, 'subset_ratio', 0.5)  # Subset ratio
        self.importance_metric = getattr(config, 'importance_metric', 'attention')  # 'attention', 'gradient', 'activation'
        self.diversity_metric = getattr(config, 'diversity_metric', 'cosine')  # 'cosine', 'euclidean', 'manhattan'
        self.layer_selection = getattr(config, 'layer_selection', 'all')  # 'all', 'skip', 'top'
        self.skip_interval = getattr(config, 'skip_interval', 2)  # Interval when layer_selection is 'skip'
        
        logger.info(f"Initialized subset matching strategy: {self.strategy_type} with ratio {self.subset_ratio}")
    
    def select_subset(self, 
                     teacher_hidden_states: List[torch.Tensor],
                     student_hidden_states: List[torch.Tensor],
                     attention_masks: Optional[torch.Tensor] = None,
                     importance_scores: Optional[List[torch.Tensor]] = None
                    ) -> Dict[str, List[int]]:
        """
        Select key subset based on chosen strategy
        
        Args:
            teacher_hidden_states: List of teacher model hidden states
            student_hidden_states: List of student model hidden states
            attention_masks: Attention masks
            importance_scores: Optional precomputed importance scores
            
        Returns:
            Dictionary containing selected layer and position indices
        """
        selected_layers = self._select_layers(teacher_hidden_states)
        
        selected_positions = {}
        for layer_idx in selected_layers:
            if layer_idx < len(teacher_hidden_states) and layer_idx < len(student_hidden_states):
                positions = self._select_positions(
                    teacher_hidden_states[layer_idx],
                    student_hidden_states[layer_idx],
                    attention_masks,
                    importance_scores[layer_idx] if importance_scores and layer_idx < len(importance_scores) else None
                )
                selected_positions[layer_idx] = positions
        
        return {
            'layers': selected_layers,
            'positions': selected_positions
        }
    
    def _select_layers(self, hidden_states: List[torch.Tensor]) -> List[int]:
        """
        Select layers to consider
        
        Args:
            hidden_states: List of hidden states
            
        Returns:
            List of selected layer indices
        """
        num_layers = len(hidden_states)
        
        if self.layer_selection == 'all':
            selected = list(range(num_layers))
        elif self.layer_selection == 'skip':
            selected = list(range(0, num_layers, self.skip_interval))
        elif self.layer_selection == 'top':
            num_top_layers = max(1, int(num_layers * self.subset_ratio))
            selected = list(range(num_layers - num_top_layers, num_layers))
        else:
            num_selected = max(1, int(num_layers * self.subset_ratio))
            selected = np.random.choice(range(num_layers), num_selected, replace=False).tolist()
            selected.sort()
        
        logger.info(f"Selected layers: {selected} out of {num_layers}")
        return selected
    
    def _select_positions(self,
                         teacher_hidden: torch.Tensor,
                         student_hidden: torch.Tensor,
                         attention_masks: Optional[torch.Tensor] = None,
                         importance_scores: Optional[torch.Tensor] = None
                        ) -> List[int]:
        """
        Select key positions within each layer
        
        Args:
            teacher_hidden: Teacher model hidden states with shape [batch_size, seq_len, hidden_dim]
            student_hidden: Student model hidden states with shape [batch_size, seq_len, hidden_dim]
            attention_masks: Attention masks with shape [batch_size, seq_len]
            importance_scores: Optional precomputed importance scores with shape [batch_size, seq_len]
            
        Returns:
            List of selected position indices
        """
        batch_size, seq_len, _ = teacher_hidden.shape
        
        if self.strategy_type == 'random':
            num_selected = max(1, int(seq_len * self.subset_ratio))
            selected = np.random.choice(range(seq_len), num_selected, replace=False).tolist()
            selected.sort()
            
        elif self.strategy_type == 'importance':
            if importance_scores is not None:
                position_scores = importance_scores.mean(dim=0).detach().cpu().numpy()
            else:
                position_scores = self._compute_position_importance(teacher_hidden, student_hidden, attention_masks)
            
            num_selected = max(1, int(seq_len * self.subset_ratio))
            top_indices = np.argsort(-position_scores)[:num_selected]
            selected = sorted(top_indices.tolist())
            
        elif self.strategy_type == 'diversity':
            selected = self._select_diverse_positions(teacher_hidden, student_hidden, attention_masks)
            
        elif self.strategy_type == 'hybrid':
            if importance_scores is not None:
                position_scores = importance_scores.mean(dim=0).detach().cpu().numpy()
            else:
                position_scores = self._compute_position_importance(teacher_hidden, student_hidden, attention_masks)
            
            num_important = max(1, int(seq_len * self.subset_ratio * 0.5))
            important_indices = np.argsort(-position_scores)[:num_important].tolist()
            
            remaining_indices = [i for i in range(seq_len) if i not in important_indices]
            num_diverse = max(1, int(seq_len * self.subset_ratio) - num_important)
            
            remaining_mask = torch.zeros(seq_len, dtype=torch.bool)
            remaining_mask[remaining_indices] = True
            
            diverse_indices = self._select_diverse_positions(
                teacher_hidden, student_hidden, attention_masks, 
                allowed_positions=remaining_indices, num_selected=num_diverse
            )
            
            selected = sorted(important_indices + diverse_indices)
            
        else:
            num_selected = max(1, int(seq_len * self.subset_ratio))
            selected = np.random.choice(range(seq_len), num_selected, replace=False).tolist()
            selected.sort()
        
        if attention_masks is not None:
            valid_positions = (attention_masks.sum(dim=0) > 0).nonzero().squeeze().tolist()
            if isinstance(valid_positions, int):
                valid_positions = [valid_positions]
            
            selected = [pos for pos in selected if pos in valid_positions]
            
            if not selected and valid_positions:
                selected = [valid_positions[0]]
        
        logger.debug(f"Selected {len(selected)} positions out of {seq_len}")
        return selected
    
    def _compute_position_importance(self,
                                   teacher_hidden: torch.Tensor,
                                   student_hidden: torch.Tensor,
                                   attention_masks: Optional[torch.Tensor] = None
                                  ) -> np.ndarray:
        """
        Compute importance scores for each position
        
        Args:
            teacher_hidden: Teacher model hidden states
            student_hidden: Student model hidden states
            attention_masks: Attention masks
            
        Returns:
            Importance scores for each position
        """
        batch_size, seq_len, hidden_dim = teacher_hidden.shape
        
        if self.importance_metric == 'attention':
            diff = (teacher_hidden - student_hidden).norm(dim=-1)  # [batch_size, seq_len]
            
            if attention_masks is not None:
                diff = diff * attention_masks
                
            position_scores = diff.mean(dim=0).detach().cpu().numpy()
            
        elif self.importance_metric == 'activation':
            activation_strength = teacher_hidden.norm(dim=-1)  # [batch_size, seq_len]
            
            if attention_masks is not None:
                activation_strength = activation_strength * attention_masks
                
            position_scores = activation_strength.mean(dim=0).detach().cpu().numpy()
            
        else:
            position_scores = np.random.rand(seq_len)
        
        return position_scores
    
    def _select_diverse_positions(self,
                                teacher_hidden: torch.Tensor,
                                student_hidden: torch.Tensor,
                                attention_masks: Optional[torch.Tensor] = None,
                                allowed_positions: Optional[List[int]] = None,
                                num_selected: Optional[int] = None
                               ) -> List[int]:
        """
        Select diverse positions
        
        Args:
            teacher_hidden: Teacher model hidden states
            student_hidden: Student model hidden states
            attention_masks: Attention masks
            allowed_positions: List of allowed positions for selection
            num_selected: Number of positions to select
            
        Returns:
            List of selected diverse positions
        """
        batch_size, seq_len, hidden_dim = teacher_hidden.shape
        
        if allowed_positions is None:
            allowed_positions = list(range(seq_len))
        
        if num_selected is None:
            num_selected = max(1, int(len(allowed_positions) * self.subset_ratio))
        
        if len(allowed_positions) <= num_selected:
            return allowed_positions
        
        avg_representations = []
        for pos in allowed_positions:
            pos_rep = teacher_hidden[:, pos, :]
            
            if attention_masks is not None:
                valid_mask = attention_masks[:, pos] > 0
                if valid_mask.any():
                    pos_rep = pos_rep[valid_mask]
                else:
                    pos_rep = pos_rep.mean(dim=0, keepdim=True)
            
            avg_rep = pos_rep.mean(dim=0).detach().cpu().numpy()
            avg_representations.append(avg_rep)
        
        selected_indices = []
        
        first_idx = np.random.choice(range(len(allowed_positions)))
        selected_indices.append(first_idx)
        
        while len(selected_indices) < num_selected:
            max_min_distance = -1
            best_idx = 0
            
            for i in range(len(allowed_positions)):
                if i not in selected_indices:
                    min_dist = float('inf')
                    for sel_idx in selected_indices:
                        if self.diversity_metric == 'cosine':
                            dist = 1 - np.dot(avg_representations[i], avg_representations[sel_idx]) / \
                                   (np.linalg.norm(avg_representations[i]) * np.linalg.norm(avg_representations[sel_idx]) + 1e-8)
                        elif self.diversity_metric == 'euclidean':
                            dist = np.linalg.norm(avg_representations[i] - avg_representations[sel_idx])
                        elif self.diversity_metric == 'manhattan':
                            dist = np.sum(np.abs(avg_representations[i] - avg_representations[sel_idx]))
                        else:
                            dist = np.linalg.norm(avg_representations[i] - avg_representations[sel_idx])
                        
                        if dist < min_dist:
                            min_dist = dist
                    
                    if min_dist > max_min_distance:
                        max_min_distance = min_dist
                        best_idx = i
            
            selected_indices.append(best_idx)
        
        selected_positions = [allowed_positions[i] for i in selected_indices]
        
        return selected_positions
    
    def compute_subset_loss(self,
                          teacher_hidden_states: List[torch.Tensor],
                          student_hidden_states: List[torch.Tensor],
                          subset_selection: Dict[str, List[int]],
                          loss_fn: Optional[callable] = None
                         ) -> torch.Tensor:
        """
        Compute loss on selected subset
        
        Args:
            teacher_hidden_states: List of teacher model hidden states
            student_hidden_states: List of student model hidden states
            subset_selection: Selected subset
            loss_fn: Loss function, MSE by default
            
        Returns:
            Subset loss
        """
        if loss_fn is None:
            loss_fn = torch.nn.MSELoss()
        
        total_loss = 0.0
        layer_losses = []
        
        for layer_idx in subset_selection['layers']:
            if layer_idx < len(teacher_hidden_states) and layer_idx < len(student_hidden_states):
                positions = subset_selection['positions'].get(layer_idx, [])
                
                if positions:
                    teacher_subset = teacher_hidden_states[layer_idx][:, positions, :]
                    student_subset = student_hidden_states[layer_idx][:, positions, :]
                    
                    layer_loss = loss_fn(student_subset, teacher_subset)
                    layer_losses.append(layer_loss)
                    
        if layer_losses:
            total_loss = torch.stack(layer_losses).mean()
        
        return total_loss


def create_subset_matcher(config):
    """
    Create subset matcher instance
    
    Args:
        config: Configuration object
        
    Returns:
        SubsetMatchingStrategy instance
    """
    return SubsetMatchingStrategy(config)


def apply_subset_matching(config,
                         teacher_hidden_states,
                         student_hidden_states,
                         attention_masks=None,
                         importance_scores=None):
    """
    Convenience function to apply subset matching strategy
    
    Args:
        config: Configuration object
        teacher_hidden_states: List of teacher model hidden states
        student_hidden_states: List of student model hidden states
        attention_masks: Attention masks
        importance_scores: Optional precomputed importance scores
        
    Returns:
        Subset selection result and corresponding loss
    """
    try:
        matcher = create_subset_matcher(config)
        
        subset_selection = matcher.select_subset(
            teacher_hidden_states,
            student_hidden_states,
            attention_masks,
            importance_scores
        )
        
        subset_loss = matcher.compute_subset_loss(
            teacher_hidden_states,
            student_hidden_states,
            subset_selection
        )
        
        logger.info(f"Applied subset matching with {len(subset_selection['layers'])} layers and \
                    {sum(len(pos) for pos in subset_selection['positions'].values())} positions total")
        
        return {
            'subset_selection': subset_selection,
            'subset_loss': subset_loss,
            'matcher': matcher
        }
    
    except Exception as e:
        logger.error(f"Error during subset matching: {str(e)}", exc_info=True)
        raise
