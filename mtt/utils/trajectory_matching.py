import torch
import numpy as np
import logging
from typing import List, Dict, Optional, Union
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TrajectoryMatcher:
    """
    Trajectory matcher for calculating trajectory similarity between student and teacher models
    
    Implements the core functionality of Text Trajectory Matching (TTM) framework, optimizing
    knowledge distillation process by aligning student trajectories with expert trajectories
    to improve model compression effectiveness.
    """
    
    def __init__(self, config):
        """
        Initialize trajectory matcher
        
        Args:
            config: Configuration object containing trajectory matching parameters
        """
        self.config = config
        self.trajectory_points = config.trajectory_points
        self.trajectory_weight = config.trajectory_weight
        self.distance_metric = config.trajectory_distance
        
        # Validate distance metric parameter
        if self.distance_metric not in ['mse', 'cosine', 'lpips']:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        logger.info(f"Initialized TrajectoryMatcher with {self.trajectory_points} points, "
                   f"weight {self.trajectory_weight}, metric {self.distance_metric}")
    
    def sample_trajectory_points(self, epoch: int, total_epochs: int) -> List[float]:
        """
        Sample trajectory points, returning time points during training process
        
        Args:
            epoch: Current training epoch
            total_epochs: Total number of training epochs
            
        Returns:
            List[float]: List of trajectory sampling points, values between [0, 1]
        """
        if self.trajectory_points <= 1:
            return [1.0]  # Only sample final state
        
        # Uniform sampling, ensuring start and end points are included
        points = []
        for i in range(self.trajectory_points):
            # Non-linear sampling, more concentrated in later training stages
            ratio = (i / (self.trajectory_points - 1)) ** 1.5
            points.append(ratio)
        
        return points
    
    def compute_trajectory_distance(self, 
                                  student_trajectory: List[List[torch.Tensor]],
                                  teacher_trajectory: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Compute distance between student trajectory and teacher trajectory
        
        Args:
            student_trajectory: Student model trajectories at various time points
            teacher_trajectory: Teacher model trajectories at various time points
            
        Returns:
            torch.Tensor: Trajectory distance
        """
        if len(student_trajectory) != len(teacher_trajectory):
            raise ValueError("Trajectory lengths do not match")
        
        distances = []
        
        # Iterate through trajectories at each time point
        for i in range(len(student_trajectory)):
            student_states = student_trajectory[i]
            teacher_states = teacher_trajectory[i]
            
            # Calculate distance for each layer
            layer_distances = []
            for s_state, t_state in zip(student_states, teacher_states):
                if self.distance_metric == 'mse':
                    dist = F.mse_loss(s_state, t_state)
                elif self.distance_metric == 'cosine':
                    # Calculate cosine similarity as distance
                    s_normalized = F.normalize(s_state.view(-1), dim=0)
                    t_normalized = F.normalize(t_state.view(-1), dim=0)
                    dist = 1.0 - torch.dot(s_normalized, t_normalized)
                elif self.distance_metric == 'lpips':
                    # Simple implementation of LPIPS-like perceptual distance
                    s_flat = s_state.view(-1)
                    t_flat = t_state.view(-1)
                    dist = torch.mean(torch.abs(s_flat - t_flat) ** 2)
                
                layer_distances.append(dist)
            
            # Calculate average distance for this time point
            time_point_distance = torch.mean(torch.stack(layer_distances))
            distances.append(time_point_distance)
        
        # Calculate total distance across all time points
        total_distance = torch.mean(torch.stack(distances))
        return total_distance
    
    def extract_model_trajectory(self, 
                               model: nn.Module,
                               inputs_embeds: torch.Tensor,
                               attention_mask: Optional[torch.Tensor] = None,
                               layers_to_extract: Optional[List[int]] = None) -> List[torch.Tensor]:
        """
        Extract trajectory states from model
        
        Args:
            model: Model to extract trajectory from
            inputs_embeds: Input embedding vectors
            attention_mask: Attention mask
            layers_to_extract: List of layer indices to extract
            
        Returns:
            List[torch.Tensor]: Hidden states of the model at various layers
        """
        # Forward pass to get hidden states
        with torch.no_grad():
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # Get hidden states from all layers
        hidden_states = outputs.hidden_states
        
        # If specific layers are specified, only extract those layers
        if layers_to_extract is not None:
            hidden_states = [hidden_states[i] for i in layers_to_extract]
        
        return hidden_states
    
    def compute_trajectory_loss(self, 
                             student_trajectories: List[List[List[torch.Tensor]]],
                             teacher_trajectories: List[List[List[torch.Tensor]]]) -> torch.Tensor:
        """
        Compute trajectory matching loss
        
        Args:
            student_trajectories: Student model trajectories across multiple samples
            teacher_trajectories: Teacher model trajectories across multiple samples
            
        Returns:
            torch.Tensor: Trajectory matching loss
        """
        if len(student_trajectories) != len(teacher_trajectories):
            raise ValueError("Number of sample trajectories do not match")
        
        losses = []
        
        # Calculate trajectory loss for each sample
        for s_traj, t_traj in zip(student_trajectories, teacher_trajectories):
            traj_distance = self.compute_trajectory_distance(s_traj, t_traj)
            losses.append(traj_distance)
        
        # Calculate average loss
        avg_loss = torch.mean(torch.stack(losses))
        
        # Apply weight
        weighted_loss = avg_loss * self.trajectory_weight
        
        return weighted_loss


def create_trajectory_matcher(config) -> TrajectoryMatcher:
    """
    Create trajectory matcher instance
    
    Args:
        config: Configuration object
        
    Returns:
        TrajectoryMatcher: Trajectory matcher instance
    """
    return TrajectoryMatcher(config)


def compute_combined_loss(
    ce_loss: torch.Tensor,
    distillation_loss: torch.Tensor,
    trajectory_loss: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
    use_trajectory: bool = True
) -> torch.Tensor:
    """
    Compute combined loss including cross-entropy loss, distillation loss, and trajectory loss
    
    Args:
        ce_loss: Cross-entropy loss
        distillation_loss: Distillation loss
        trajectory_loss: Trajectory loss
        alpha: Distillation loss weight
        use_trajectory: Whether to use trajectory loss
        
    Returns:
        torch.Tensor: Combined loss
    """
    # Base loss: cross-entropy + distillation
    base_loss = (1 - alpha) * ce_loss + alpha * distillation_loss
    
    # If using trajectory loss, add trajectory loss
    if use_trajectory and trajectory_loss is not None:
        total_loss = base_loss + trajectory_loss
    else:
        total_loss = base_loss
    
    return total_loss


def collect_trajectories(
    model_stu: nn.Module,
    model_tea: nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    config,
    subset_matcher = None
) -> Dict[str, Union[List[List[List[torch.Tensor]]], torch.Tensor]]:
    """
    Collect trajectories of student model and teacher model
    
    Args:
        model_stu: Student model
        model_tea: Teacher model
        inputs_embeds: Input embedding vectors
        attention_mask: Attention mask
        config: Configuration object
        subset_matcher: Subset matcher instance (optional)
        
    Returns:
        Dict: Dictionary containing student and teacher trajectories
    """
    # Create trajectory matcher
    trajectory_matcher = create_trajectory_matcher(config)
    
    # Determine layers to extract
    layers_to_extract = list(range(config.layers_to_distill))
    
    # If using subset matching, adjust layers to extract
    subset_selection = None
    if config.use_subset_matching and subset_matcher is not None:
        # Get teacher model hidden states for subset selection
        with torch.no_grad():
            tea_outputs = model_tea(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        tea_hidden_states = tea_outputs.hidden_states
        
        # Select subset
        subset_selection = subset_matcher.select_subset(tea_hidden_states)
        layers_to_extract = subset_selection['layers']
    
    # Extract teacher model trajectory
    teacher_trajectory = trajectory_matcher.extract_model_trajectory(
        model_tea,
        inputs_embeds,
        attention_mask,
        layers_to_extract
    )
    
    # Extract student model trajectory
    student_trajectory = trajectory_matcher.extract_model_trajectory(
        model_stu,
        inputs_embeds,
        attention_mask,
        layers_to_extract
    )
    
    # Package into batch format
    student_trajectories = [[student_trajectory]]
    teacher_trajectories = [[teacher_trajectory]]
    
    return {
        'student_trajectories': student_trajectories,
        'teacher_trajectories': teacher_trajectories,
        'subset_selection': subset_selection
    }
