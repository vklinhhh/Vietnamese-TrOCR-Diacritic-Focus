import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import re
from tqdm.auto import tqdm
import wandb
import matplotlib.pyplot as plt

class CurriculumDataset(Dataset):
    """Dataset wrapper that implements curriculum learning strategy"""
    
    def __init__(self, dataset, curriculum_strategy="length", curriculum_stages=3, 
                 min_examples_per_stage=100):
        """
        Initialize curriculum dataset wrapper.
        
        Args:
            dataset: Base dataset to wrap
            curriculum_strategy: Strategy for determining difficulty ('length', 'complexity', or 'combined')
            curriculum_stages: Number of curriculum stages
            min_examples_per_stage: Minimum number of examples required per stage
        """
        self.dataset = dataset
        self.curriculum_strategy = curriculum_strategy
        self.curriculum_stages = curriculum_stages
        self.min_examples_per_stage = min_examples_per_stage
        
        # Current curriculum stage (0-indexed)
        self.current_stage = 0
        
        # Calculate difficulty scores for all examples
        self.difficulty_scores = self._calculate_difficulty_scores()
        
        # Compute stage thresholds
        self._compute_stage_thresholds()
        
        # Get indices for current stage
        self.current_indices = self._get_indices_for_stage(self.current_stage)
        
        print(f"Initialized curriculum with {curriculum_stages} stages.")
        print(f"Stage 1 (current): {len(self.current_indices)} examples")

    def _calculate_difficulty_scores(self):
        """Calculate difficulty score for each example based on the chosen strategy."""
        difficulty_scores = []
        
        print("Calculating difficulty scores for curriculum...")
        for idx in tqdm(range(len(self.dataset.dataset))):  # Access the underlying HuggingFace dataset
            # Get example directly from the HuggingFace dataset to avoid image processing
            example = self.dataset.dataset[idx]
            
            # Get the word or character sequence
            if 'word' in example:
                text = example['word']
            elif 'full_character' in example:
                # Handle when full_character is a list or a string representation of a list
                if isinstance(example['full_character'], list):
                    text = ''.join(example['full_character'])
                elif isinstance(example['full_character'], str) and example['full_character'].startswith('[') and example['full_character'].endswith(']'):
                    try:
                        import ast
                        chars = ast.literal_eval(example['full_character'])
                        text = ''.join(chars)
                    except:
                        text = example['full_character']
                else:
                    text = example['full_character']
            else:
                text = ""  # Fallback
            
            # Calculate score based on strategy
            if self.curriculum_strategy == 'length':
                # Simple length-based scoring
                score = len(text)
            
            elif self.curriculum_strategy == 'complexity':
                # Count diacritics as indicator of complexity
                diacritic_count = 0
                if 'diacritic_type' in example:
                    # Parse diacritics if they're in a string representation of a list
                    diacritics = example['diacritic_type']
                    if isinstance(diacritics, str) and diacritics.startswith('[') and diacritics.endswith(']'):
                        try:
                            import ast
                            diacritics = ast.literal_eval(diacritics)
                        except:
                            diacritics = [diacritics]
                    elif not isinstance(diacritics, list):
                        diacritics = [diacritics]
                    
                    # Count non-empty, non-'none' diacritics
                    diacritic_count = sum(1 for d in diacritics if d and d.lower() != 'none')
                
                score = diacritic_count
            
            elif self.curriculum_strategy == 'combined':
                # Combine length and diacritic complexity
                text_length = len(text)
                
                diacritic_count = 0
                if 'diacritic_type' in example:
                    # Parse diacritics if they're in a string representation of a list
                    diacritics = example['diacritic_type']
                    if isinstance(diacritics, str) and diacritics.startswith('[') and diacritics.endswith(']'):
                        try:
                            import ast
                            diacritics = ast.literal_eval(diacritics)
                        except:
                            diacritics = [diacritics]
                    elif not isinstance(diacritics, list):
                        diacritics = [diacritics]
                    
                    # Count non-empty, non-'none' diacritics
                    diacritic_count = sum(1 for d in diacritics if d and d.lower() != 'none')
                
                # Combine scores with appropriate weighting
                score = text_length + diacritic_count * 2  # Weight diacritics more
            
            else:
                raise ValueError(f"Unknown curriculum strategy: {self.curriculum_strategy}")
            
            difficulty_scores.append(score)
        
        return np.array(difficulty_scores)

    def _compute_stage_thresholds(self):
        """Compute difficulty thresholds for each curriculum stage."""
        # Get unique difficulty scores and their counts
        unique_scores, counts = np.unique(self.difficulty_scores, return_counts=True)
        
        # Sort scores
        sorted_indices = np.argsort(unique_scores)
        unique_scores = unique_scores[sorted_indices]
        counts = counts[sorted_indices]
        
        # Calculate cumulative counts
        cumulative_counts = np.cumsum(counts)
        total_examples = cumulative_counts[-1]
        
        # Calculate target examples per stage
        target_per_stage = total_examples / self.curriculum_stages
        
        # Find thresholds that divide examples most evenly
        thresholds = []
        for stage in range(self.curriculum_stages - 1):
            target_count = (stage + 1) * target_per_stage
            idx = np.argmin(np.abs(cumulative_counts - target_count))
            thresholds.append(unique_scores[idx])
        
        # Add maximum threshold
        thresholds.append(float('inf'))
        
        self.thresholds = thresholds
        
        # Print stage distribution
        prev_count = 0
        for i, thresh in enumerate(thresholds):
            # Count examples below this threshold
            count = np.sum(self.difficulty_scores <= thresh)
            stage_count = count - prev_count
            prev_count = count
            
            print(f"Stage {i+1}: {stage_count} examples (difficulty <= {thresh})")
            
            # Check if any stage has too few examples
            if stage_count < self.min_examples_per_stage:
                print(f"Warning: Stage {i+1} has only {stage_count} examples, "
                      f"which is less than minimum {self.min_examples_per_stage}")

    def _get_indices_for_stage(self, stage):
        """Get dataset indices for a specific curriculum stage."""
        assert 0 <= stage < self.curriculum_stages, f"Invalid stage: {stage}"
        
        lower_threshold = 0 if stage == 0 else self.thresholds[stage - 1]
        upper_threshold = self.thresholds[stage]
        
        # Get indices where difficulty score is in the appropriate range
        indices = np.where(
            (self.difficulty_scores > lower_threshold) & 
            (self.difficulty_scores <= upper_threshold)
        )[0]
        
        # Convert numpy int64 to python int to avoid issues with HuggingFace datasets
        indices = [int(idx) for idx in indices]
        
        return indices

    def set_stage(self, stage):
        """Set curriculum to a specific stage."""
        assert 0 <= stage < self.curriculum_stages, f"Invalid stage: {stage}"
        
        self.current_stage = stage
        self.current_indices = self._get_indices_for_stage(stage)
        
        # Log the stage change
        print(f"Switched to curriculum stage {stage + 1}/{self.curriculum_stages}")
        print(f"Stage {stage + 1}: {len(self.current_indices)} examples")
        
        return len(self.current_indices)

    def advance_stage(self):
        """Advance to the next curriculum stage if possible."""
        if self.current_stage < self.curriculum_stages - 1:
            return self.set_stage(self.current_stage + 1)
        else:
            print("Already at final curriculum stage.")
            return len(self.current_indices)

    def __len__(self):
        """Return the number of examples in the current stage."""
        return len(self.current_indices)

    def __getitem__(self, idx):
        """Get item from the current stage."""
        # Map the local index to the correct dataset index
        dataset_idx = self.current_indices[idx]
        # Ensure we're using a Python int, not numpy.int64
        dataset_idx = int(dataset_idx)
        # Return the actual item
        return self.dataset[dataset_idx]

    def get_all_data(self):
        """Return a dataset with all examples (for evaluation)."""
        return self.dataset

    def get_current_stage_dataset(self):
        """Return a Subset dataset for the current stage (useful for DataLoader)."""
        return Subset(self.dataset, self.current_indices)

    def log_curriculum_stats(self):
        """Log curriculum statistics to wandb."""
        try:
            import matplotlib.pyplot as plt
            import wandb
            import numpy as np
            
            # Create histogram of difficulty scores
            plt.figure(figsize=(10, 6))
            
            # Plot histogram
            plt.hist(self.difficulty_scores, bins=30)
            
            # Add threshold lines
            for i, thresh in enumerate(self.thresholds[:-1]):  # Skip the last threshold (inf)
                plt.axvline(x=thresh, color='r', linestyle='--', 
                         label=f'Stage {i+1}/{i+2} threshold' if i == 0 else None)
            
            # Add labels
            plt.xlabel('Difficulty Score')
            plt.ylabel('Number of Examples')
            plt.title('Curriculum Learning Difficulty Distribution')
            plt.legend()
            
            # Save figure to a temporary file that wandb can handle
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plt.savefig(tmp.name)
                wandb.log({"curriculum/difficulty_distribution": wandb.Image(tmp.name)})
            
            plt.close()
            
            # Log stage statistics as a table
            stage_stats = []
            prev_count = 0
            
            for i, thresh in enumerate(self.thresholds):
                count = np.sum(self.difficulty_scores <= thresh)
                stage_count = count - prev_count
                prev_count = count
                
                stage_stats.append({
                    "stage": i+1,
                    "example_count": int(stage_count),
                    "threshold": float(thresh) if thresh != float('inf') else None,
                    "percent_of_total": float(stage_count / len(self.difficulty_scores) * 100)
                })
            
            # Create a table to log
            columns = ["Stage", "Example Count", "Max Difficulty", "% of Total"]
            data = [[s["stage"], s["example_count"], 
                   "inf" if s["threshold"] is None else f"{s['threshold']:.1f}", 
                   f"{s['percent_of_total']:.1f}%"] 
                  for s in stage_stats]
            
            wandb.log({"curriculum/stage_statistics": wandb.Table(columns=columns, data=data)})
            
        except Exception as e:
            print(f"Warning: Error logging curriculum stats to wandb: {e}")
            print("Training will continue, but wandb logging may be incomplete.")
        
        # Log stage statistics
        stage_stats = []
        prev_count = 0
        
        for i, thresh in enumerate(self.thresholds):
            count = np.sum(self.difficulty_scores <= thresh)
            stage_count = count - prev_count
            prev_count = count
            
            stage_stats.append({
                "stage": i+1,
                "example_count": int(stage_count),
                "threshold": float(thresh) if thresh != float('inf') else None,
                "percent_of_total": float(stage_count / len(self.difficulty_scores) * 100)
            })
        
        # Log as table
        wandb.log({"curriculum/stage_statistics": wandb.Table(
            columns=["Stage", "Example Count", "Max Difficulty", "% of Total"],
            data=[[s["stage"], s["example_count"], s["threshold"], s["percent_of_total"]] 
                 for s in stage_stats]
        )})

def create_curriculum_datasets(train_dataset, val_dataset, curriculum_strategy="length", 
                              curriculum_stages=3, min_examples_per_stage=100):
    """
    Create curriculum learning wrapper datasets for training and validation.
    
    Args:
        train_dataset: Training dataset to wrap
        val_dataset: Validation dataset to wrap (will not be curriculum filtered)
        curriculum_strategy: Strategy for determining difficulty
        curriculum_stages: Number of curriculum stages
        min_examples_per_stage: Minimum number of examples required per stage
    
    Returns:
        train_curriculum: CurriculumDataset for training
        val_dataset: Original validation dataset (unchanged)
    """
    # Create curriculum wrapper for training data
    train_curriculum = CurriculumDataset(
        train_dataset,
        curriculum_strategy=curriculum_strategy,
        curriculum_stages=curriculum_stages,
        min_examples_per_stage=min_examples_per_stage
    )
    
    # Log curriculum statistics
    train_curriculum.log_curriculum_stats()
    
    return train_curriculum, val_dataset