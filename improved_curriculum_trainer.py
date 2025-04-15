from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils import custom_collate_fn, WandbImageLogger, log_model_architecture
from dataset import VietnameseCharacterDataset
from improved_models import ImprovedVietnameseOCRModel
from curriculum import create_curriculum_datasets
from focal_loss import FocalLoss
import torch
import torch.nn as nn
from transformers import TrOCRProcessor
import wandb
import gc
import os
import matplotlib.pyplot as plt
import io
import numpy as np
import math
from improved_utils import ImprovedWandbImageLogger

# Helper function to make objects JSON serializable
def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    else:
        return obj

class CosineWarmupScheduler:
    """
    Learning rate scheduler with warm-up and cosine annealing
    """
    def __init__(self, optimizer, warmup_steps, max_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self._step = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            # Linear warmup
            lr_scale = self._step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (self._step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.base_lrs[i] * lr_scale
            
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

def calculate_class_weights(diacritic_indices, num_classes):
    """
    Calculate class weights for imbalanced diacritic classes
    """
    # Use reshape instead of view to handle non-contiguous tensors
    # Also, make sure the tensor is contiguous before reshaping
    all_diacritics = diacritic_indices.contiguous().reshape(-1).cpu().numpy()
    
    # Count occurrences of each class
    class_counts = np.bincount(all_diacritics, minlength=num_classes)
    
    # Add small epsilon to avoid division by zero
    class_counts = class_counts + 1e-5
    
    # Calculate weights as inverse of frequency
    weights = 1.0 / class_counts
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum() * num_classes
    
    # Convert to tensor
    return torch.tensor(weights, dtype=torch.float32)

def train_improved_vietnamese_ocr_with_curriculum(
    model, train_dataset, val_dataset, epochs=5, batch_size=8, lr=5e-6, device=None, processor=None,
    project_name="vietnamese-trocr-diacritics-curriculum", run_name=None, log_interval=10,
    curriculum_strategy="combined", curriculum_stages=3, stage_epochs=None, patience=3,
    early_stopping_patience=5, focal_loss_gamma=2.0
):
    """
    Train Vietnamese OCR model with curriculum learning and improved architecture.
    
    Args:
        model: The OCR model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Total number of epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        processor: TrOCR processor
        project_name: W&B project name
        run_name: W&B run name
        log_interval: How often to log metrics
        curriculum_strategy: Strategy for curriculum ('length', 'complexity', 'combined')
        curriculum_stages: Number of curriculum stages
        stage_epochs: List specifying how many epochs to spend on each stage.
                    If None, will automatically determine based on loss plateaus.
        patience: How many epochs to wait for improvement before advancing curriculum stage
        early_stopping_patience: How many epochs to wait for improvement before early stopping
        focal_loss_gamma: Gamma parameter for focal loss (higher = more focus on hard examples)
    
    Returns:
        Trained model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set up wandb
    wandb.init(project=project_name, name=run_name, config={
        "model_type": "ImprovedVietnameseOCRModel",
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "base_model": "microsoft/trocr-base-handwritten",
        "device": str(device),
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
        "curriculum_strategy": curriculum_strategy,
        "curriculum_stages": curriculum_stages,
        "stage_epochs": stage_epochs,
        "early_stopping_patience": early_stopping_patience,
        "focal_loss_gamma": focal_loss_gamma,
    })
    
    # Initialize image logger and log model architecture
    base_char_vocab = train_dataset.base_char_vocab
    diacritic_vocab = train_dataset.diacritic_vocab
    # image_logger = WandbImageLogger(processor, base_char_vocab, diacritic_vocab, log_interval=log_interval)
    image_logger = ImprovedWandbImageLogger(processor, base_char_vocab, diacritic_vocab, log_interval=log_interval)

    log_model_architecture(model)
    
    # Create curriculum datasets
    print("Setting up curriculum learning...")
    train_curriculum, val_dataset = create_curriculum_datasets(
        train_dataset,
        val_dataset,
        curriculum_strategy=curriculum_strategy,
        curriculum_stages=curriculum_stages,
        min_examples_per_stage=batch_size * 5  # Ensure at least 5 batches per stage
    )
    
    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Calculate total training steps for learning rate scheduler
    total_steps = epochs * len(train_curriculum) // batch_size
    warmup_steps = total_steps // 10  # 10% of steps for warmup
    
    # Initialize learning rate scheduler with warmup
    lr_scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        max_steps=total_steps
    )
    
    # Define loss functions
    # Regular cross entropy for text prediction
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='mean')
    
    # Focal loss for diacritics to handle class imbalance
    focal_loss = FocalLoss(gamma=focal_loss_gamma, reduction='mean')
    
    # Track best validation loss and best model
    best_val_loss = float('inf')
    best_model_state = None
    no_improvement_count = 0
    
    # Track training history for each stage
    stage_histories = []
    current_history = {
        "stage": 0,
        "epoch_losses": [],
        "val_losses": [],
        "word_accs": [],
        "char_accs": [],
        "base_char_accs": [],
        "diacritic_accs": []
    }
    
    # If stage_epochs is specified, calculate epoch ranges for each stage
    stage_start_epochs = [0]
    if stage_epochs is not None:
        assert len(stage_epochs) == curriculum_stages, "Must specify epochs for each stage"
        for i in range(len(stage_epochs) - 1):
            stage_start_epochs.append(stage_start_epochs[-1] + stage_epochs[i])
    
    # Training loop
    global_step = 0
    total_epochs_completed = 0
    
    while total_epochs_completed < epochs:
        # Check if we should advance to the next curriculum stage based on stage_epochs
        if stage_epochs is not None:
            # Find which stage we should be in based on the current epoch
            current_epoch_stage = 0
            for i in range(1, len(stage_start_epochs)):
                if total_epochs_completed >= stage_start_epochs[i]:
                    current_epoch_stage = i
            
            # Only advance stage if we need to go to a higher stage than current
            if train_curriculum.current_stage < current_epoch_stage:
                # Time to advance
                print(f"Advancing curriculum stage after {total_epochs_completed} epochs "
                      f"(stage plan: {stage_epochs})")
                
                # Save current stage history and create new one
                stage_histories.append(current_history)
                current_history = {
                    "stage": current_epoch_stage,
                    "epoch_losses": [],
                    "val_losses": [],
                    "word_accs": [],
                    "char_accs": [],
                    "base_char_accs": [],
                    "diacritic_accs": []
                }
                
                # Update the curriculum - advance to exactly the stage we need
                while train_curriculum.current_stage < current_epoch_stage:
                    train_curriculum.advance_stage()
        
        # Get current stage dataset
        current_train_dataset = train_curriculum.get_current_stage_dataset()
        
        # Create data loader for current stage
        train_loader = DataLoader(
            current_train_dataset, batch_size=batch_size, shuffle=True, 
            collate_fn=custom_collate_fn
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
        
        # Training epoch
        model.train()
        train_loss = 0
        train_text_loss = 0
        train_base_char_loss = 0
        train_diacritic_loss = 0
        step = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {total_epochs_completed+1}, Stage {train_curriculum.current_stage+1}"):
            step += 1
            global_step += 1
            
            # Set up tokenizer config
            tokenizer = processor.tokenizer
            model.config.decoder_start_token_id = tokenizer.bos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
            model.config.eos_token_id = tokenizer.eos_token_id
            
            # Move batch to device
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            base_char_labels = batch['base_character_indices'].to(device).long()
            diacritic_labels = batch['diacritic_indices'].to(device).long()
            
            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            
            # Calculate losses
            text_loss = outputs['loss']  # TrOCR's built-in loss
            
            # Calculate the minimum sequence length between labels and logits
            seq_length = min(
                base_char_labels.size(1),
                outputs['base_char_logits'].size(1)
            )
            
            base_char_loss = torch.zeros(1, device=device)
            diacritic_loss = torch.zeros(1, device=device)
            
            # Calculate class weights for this batch to handle imbalance
            num_diacritic_classes = len(diacritic_vocab)
            diacritic_weights = calculate_class_weights(diacritic_labels[:, :seq_length], num_diacritic_classes).to(device)
            
            for pos in range(seq_length):
                # Cross entropy for base characters - more balanced
                base_char_loss += ce_loss(outputs['base_char_logits'][:, pos, :], base_char_labels[:, pos])
                
                # Focal loss for diacritics - handles imbalance better
                diacritic_loss += focal_loss(outputs['diacritic_logits'][:, pos, :], diacritic_labels[:, pos])
            
            base_char_loss /= seq_length
            diacritic_loss /= seq_length
            
            # Weight the component losses based on curriculum stage
            # Early stages: Focus more on base character recognition
            # Later stages: Balance more toward diacritics
            stage_ratio = train_curriculum.current_stage / (curriculum_stages - 1)
            text_weight = 0.5 + 0.5 * stage_ratio  # 0.5 -> 1.0
            base_char_weight = 1.0  # Always important
            diacritic_weight = 0.5 + 1.5 * stage_ratio  # 0.5 -> 2.0 (increased weight for diacritics in later stages)
            
            # Combine losses with dynamic weights
            total_loss = (
                text_weight * text_loss + 
                base_char_weight * base_char_loss + 
                diacritic_weight * diacritic_loss
            )
            
            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            # Update learning rate schedule
            lr_scheduler.step()
            
            # Accumulate losses
            train_loss += total_loss.item()
            train_text_loss += text_loss.item()
            train_base_char_loss += base_char_loss.item()
            train_diacritic_loss += diacritic_loss.item()
            
            # Log image predictions occasionally
            if step % (log_interval * 5) == 0:
                image_logger.log_predictions(batch, outputs, phase="train")
            
            # Log training metrics at intervals
            if step % log_interval == 0:
                wandb.log({
                    "train/step": global_step,
                    "train/batch_loss": total_loss.item(),
                    "train/batch_text_loss": text_loss.item(),
                    "train/batch_base_char_loss": base_char_loss.item(),
                    "train/batch_diacritic_loss": diacritic_loss.item(),
                    "curriculum/current_stage": train_curriculum.current_stage + 1,
                    "curriculum/stage_weight_text": text_weight,
                    "curriculum/stage_weight_base_char": base_char_weight,
                    "curriculum/stage_weight_diacritic": diacritic_weight,
                    "train/learning_rate": optimizer.param_groups[0]['lr']
                })
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader) if train_loader else 0
        avg_train_text_loss = train_text_loss / len(train_loader) if train_loader else 0
        avg_train_base_char_loss = train_base_char_loss / len(train_loader) if train_loader else 0
        avg_train_diacritic_loss = train_diacritic_loss / len(train_loader) if train_loader else 0
        
        # Add to history
        current_history["epoch_losses"].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_text_loss = 0
        val_base_char_loss = 0
        val_diacritic_loss = 0
        word_acc = 0
        char_acc = 0
        base_char_acc = 0
        diacritic_acc = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Set up tokenizer config
                tokenizer = processor.tokenizer
                model.config.decoder_start_token_id = tokenizer.bos_token_id
                model.config.pad_token_id = tokenizer.pad_token_id
                model.config.eos_token_id = tokenizer.eos_token_id
                
                # Move batch to device
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                base_char_labels = batch['base_character_indices'].to(device).long()
                diacritic_labels = batch['diacritic_indices'].to(device).long()
                
                # Forward pass
                outputs = model(pixel_values=pixel_values, labels=labels)
                
                # Calculate losses
                text_loss = outputs['loss']  # TrOCR's built-in loss
                
                # Calculate the minimum sequence length between labels and logits
                seq_length = min(
                    base_char_labels.size(1),
                    outputs['base_char_logits'].size(1)
                )
                
                base_char_loss = torch.zeros(1, device=device)
                diacritic_loss = torch.zeros(1, device=device)
                
                for pos in range(seq_length):
                    base_char_loss += ce_loss(outputs['base_char_logits'][:, pos, :], 
                                            base_char_labels[:, pos])
                    diacritic_loss += focal_loss(outputs['diacritic_logits'][:, pos, :], 
                                              diacritic_labels[:, pos])
                
                base_char_loss /= seq_length
                diacritic_loss /= seq_length
                
                # Always use balanced weights during validation (for consistent tracking)
                total_loss = text_loss + base_char_loss + diacritic_loss
                
                # Accumulate validation losses
                val_loss += total_loss.item()
                val_text_loss += text_loss.item()
                val_base_char_loss += base_char_loss.item()
                val_diacritic_loss += diacritic_loss.item()
                
                # Calculate accuracies
                # Text accuracy (word and character level)
                predicted_ids = outputs['logits'].argmax(-1)
                word_correct = 0
                char_correct = 0
                char_total = 0
                
                for i, (pred, label) in enumerate(zip(predicted_ids, labels)):
                    valid_indices = label != -100
                    if valid_indices.any():
                        valid_pred = pred[valid_indices]
                        valid_label = label[valid_indices]
                        word_correct += torch.all(valid_pred == valid_label).item()
                        char_correct += (valid_pred == valid_label).sum().item()
                        char_total += valid_indices.sum().item()
                
                batch_word_acc = word_correct / len(labels)
                batch_char_acc = char_correct / char_total if char_total > 0 else 0
                
                word_acc += batch_word_acc
                char_acc += batch_char_acc
                
                # Base character accuracy
                base_char_preds = outputs['base_char_logits'].argmax(-1)
                base_char_matches = 0
                base_char_total = 0
                
                for pos in range(seq_length):
                    # Only count positions with valid labels (not padding)
                    valid_indices = base_char_labels[:, pos] != 0  # Assuming 0 is padding
                    matches = (base_char_preds[:, pos][valid_indices] == 
                             base_char_labels[:, pos][valid_indices]).sum().item()
                    base_char_matches += matches
                    base_char_total += valid_indices.sum().item()
                
                batch_base_char_acc = base_char_matches / base_char_total if base_char_total > 0 else 0
                base_char_acc += batch_base_char_acc
                
                # Diacritic accuracy
                diacritic_preds = outputs['diacritic_logits'].argmax(-1)
                diacritic_matches = 0
                diacritic_total = 0
                
                for pos in range(seq_length):
                    # Only count positions with valid labels (not padding)
                    valid_indices = diacritic_labels[:, pos] != 0  # Assuming 0 is padding
                    matches = (diacritic_preds[:, pos][valid_indices] == 
                             diacritic_labels[:, pos][valid_indices]).sum().item()
                    diacritic_matches += matches
                    diacritic_total += valid_indices.sum().item()
                
                batch_diacritic_acc = diacritic_matches / diacritic_total if diacritic_total > 0 else 0
                diacritic_acc += batch_diacritic_acc
                
                # Log image predictions
                image_logger.log_predictions(batch, outputs, phase="val")
        
        # Calculate average validation metrics
        num_val_batches = len(val_loader) if val_loader else 1
        avg_val_loss = val_loss / num_val_batches
        avg_val_text_loss = val_text_loss / num_val_batches
        avg_val_base_char_loss = val_base_char_loss / num_val_batches
        avg_val_diacritic_loss = val_diacritic_loss / num_val_batches
        avg_word_acc = word_acc / num_val_batches
        avg_char_acc = char_acc / num_val_batches
        avg_base_char_acc = base_char_acc / num_val_batches
        avg_diacritic_acc = diacritic_acc / num_val_batches
        
        # Add to history
        current_history["val_losses"].append(avg_val_loss)
        current_history["word_accs"].append(avg_word_acc)
        current_history["char_accs"].append(avg_char_acc)
        current_history["base_char_accs"].append(avg_base_char_acc)
        current_history["diacritic_accs"].append(avg_diacritic_acc)
        
        # Log epoch metrics to wandb
        wandb.log({
            "epoch": total_epochs_completed + 1,
            "train/loss": avg_train_loss,
            "train/text_loss": avg_train_text_loss,
            "train/base_char_loss": avg_train_base_char_loss,
            "train/diacritic_loss": avg_train_diacritic_loss,
            "val/loss": avg_val_loss,
            "val/text_loss": avg_val_text_loss,
            "val/base_char_loss": avg_val_base_char_loss,
            "val/diacritic_loss": avg_val_diacritic_loss,
            "val/word_accuracy": avg_word_acc,
            "val/char_accuracy": avg_char_acc,
            "val/base_char_accuracy": avg_base_char_acc,
            "val/diacritic_accuracy": avg_diacritic_acc,
            "curriculum/stage": train_curriculum.current_stage + 1,
        })
        
        # Print epoch results
        print(f'Epoch {total_epochs_completed + 1}/{epochs}, '
              f'Curriculum Stage: {train_curriculum.current_stage + 1}/{curriculum_stages}')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'Word Acc: {avg_word_acc:.4f}, Char Acc: {avg_char_acc:.4f}, '
              f'Base Char Acc: {avg_base_char_acc:.4f}, Diacritic Acc: {avg_diacritic_acc:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.7f}')
        
        # Check for best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {
                'epoch': total_epochs_completed,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'curriculum_stage': train_curriculum.current_stage,
                'val_loss': avg_val_loss,
                'val_word_acc': avg_word_acc,
                'val_char_acc': avg_char_acc,
                'val_base_char_acc': avg_base_char_acc,
                'val_diacritic_acc': avg_diacritic_acc
            }
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        # Check for early stopping
        if no_improvement_count >= early_stopping_patience:
            print(f"Early stopping triggered after {total_epochs_completed + 1} epochs")
            break
            
        # Consider advancing curriculum stage based on plateau (if not using fixed schedule)
        if stage_epochs is None and no_improvement_count >= patience:
            # If we've been on this stage for a while with no improvement, try advancing
            if train_curriculum.current_stage < curriculum_stages - 1:
                print(f"No improvement for {patience} epochs. Advancing curriculum stage.")
                
                # Save current stage history and create new one
                stage_histories.append(current_history)
                
                # Advance to next stage
                next_stage = train_curriculum.current_stage + 1
                train_curriculum.set_stage(next_stage)
                
                # Reset counters
                no_improvement_count = 0
                
                # Create new history tracker for this stage
                current_history = {
                    "stage": next_stage,
                    "epoch_losses": [],
                    "val_losses": [],
                    "word_accs": [],
                    "char_accs": [],
                    "base_char_accs": [],
                    "diacritic_accs": []
                }
                
                # Log stage transition
                wandb.log({
                    "curriculum/stage_transition": next_stage,
                    "curriculum/stage_transition_epoch": total_epochs_completed + 1
                })
            
        # Optional: Save model checkpoint
        if (total_epochs_completed + 1) % 5 == 0 or (total_epochs_completed == epochs - 1):
            checkpoints_dir = "checkpoints"
            os.makedirs(checkpoints_dir, exist_ok=True)

            model_checkpoint_path = f"{checkpoints_dir}/checkpoint-epoch-{total_epochs_completed+1}-stage-{train_curriculum.current_stage+1}"
            processor.save_pretrained(checkpoints_dir)
            model.save_pretrained(model_checkpoint_path)
            wandb.save(f"{model_checkpoint_path}/*")
            
            # Also save curriculum metadata
            curriculum_metadata = {
                "current_stage": train_curriculum.current_stage,
                "curriculum_strategy": curriculum_strategy,
                "thresholds": [float(t) if t != float('inf') else "inf" for t in train_curriculum.thresholds],
                "stage_histories": make_json_serializable(stage_histories + [current_history])
            }
            
            import json
            with open(f"{model_checkpoint_path}/curriculum_metadata.json", "w") as f:
                json.dump(curriculum_metadata, f, indent=2)
        
        # Increment epoch counter
        total_epochs_completed += 1
        
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()
    
    # Training complete
    print(f"Training completed after {total_epochs_completed} epochs")
    
    # Add the final stage history if not already added
    stage_histories.append(current_history)
    
    # Log curriculum learning curves
    log_curriculum_learning_curves(stage_histories)
    
    # Restore best model if we have one
    if best_model_state is not None:
        print(f"Restoring best model from epoch {best_model_state['epoch'] + 1} "
              f"with validation loss {best_model_state['val_loss']:.4f}")
        model.load_state_dict(best_model_state['model_state_dict'])
    
    # Save final model
    final_model_path = f"vietnamese-ocr-curriculum-final"
    os.makedirs(final_model_path, exist_ok=True)
    
    processor.save_pretrained(final_model_path)
    model.save_pretrained(final_model_path)
    
    
    # Save curriculum metadata
    final_curriculum_metadata = {
        "curriculum_strategy": curriculum_strategy,
        "curriculum_stages": curriculum_stages,
        "thresholds": [float(t) if t != float('inf') else "inf" for t in train_curriculum.thresholds],
        "final_stage": train_curriculum.current_stage,
        "best_val_loss": best_val_loss,
        "best_epoch": best_model_state['epoch'] + 1 if best_model_state else None,
        "total_epochs": total_epochs_completed,
        "stage_histories": make_json_serializable(stage_histories)
    }
    
    import json
    with open(f"{final_model_path}/curriculum_metadata.json", "w") as f:
        json.dump(final_curriculum_metadata, f, indent=2)
    
    # Finish wandb run
    wandb.save("*.py")  # Track all Python scripts
    wandb.save("*.json")  # Track all JSON files
    wandb.save("*.log")  # Track log files
    wandb.finish()
    
    return model

def log_curriculum_learning_curves(stage_histories):
    """
    Log learning curves for each curriculum stage to wandb.
    
    Args:
        stage_histories: List of dictionaries with stage training history
    """
    # Create a figure showing how metrics evolved across stages
    plt.figure(figsize=(15, 10))
    
    # Create 2x2 subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Colors for different stages
    stage_colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot training loss
    ax = axes[0, 0]
    for i, history in enumerate(stage_histories):
        color = stage_colors[i % len(stage_colors)]
        epochs = range(1, len(history["epoch_losses"]) + 1)
        ax.plot(epochs, history["epoch_losses"], '-', color=color, label=f"Stage {i+1}")
    ax.set_title('Training Loss by Curriculum Stage')
    ax.set_xlabel('Epochs within Stage')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Plot validation loss
    ax = axes[0, 1]
    for i, history in enumerate(stage_histories):
        color = stage_colors[i % len(stage_colors)]
        epochs = range(1, len(history["val_losses"]) + 1)
        ax.plot(epochs, history["val_losses"], '-', color=color, label=f"Stage {i+1}")
    ax.set_title('Validation Loss by Curriculum Stage')
    ax.set_xlabel('Epochs within Stage')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Plot word accuracy
    ax = axes[1, 0]
    for i, history in enumerate(stage_histories):
        color = stage_colors[i % len(stage_colors)]
        epochs = range(1, len(history["word_accs"]) + 1)
        ax.plot(epochs, history["word_accs"], '-', color=color, label=f"Stage {i+1}")
    ax.set_title('Word Accuracy by Curriculum Stage')
    ax.set_xlabel('Epochs within Stage')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    
    # Plot character-level accuracies
    ax = axes[1, 1]
    for i, history in enumerate(stage_histories):
        color = stage_colors[i % len(stage_colors)]
        epochs = range(1, len(history["char_accs"]) + 1)
        ax.plot(epochs, history["char_accs"], '-', color=color, label=f"Char Stage {i+1}")
        
        if "base_char_accs" in history and len(history["base_char_accs"]) > 0:
            ax.plot(epochs, history["base_char_accs"], '--', color=color, label=f"Base Stage {i+1}")
            
        if "diacritic_accs" in history and len(history["diacritic_accs"]) > 0:
            ax.plot(epochs, history["diacritic_accs"], ':', color=color, label=f"Diac Stage {i+1}")
    
    ax.set_title('Character-Level Accuracies by Curriculum Stage')
    ax.set_xlabel('Epochs within Stage')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure to a temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plt.savefig(tmp.name)
        # Log the temporary file to wandb
        wandb.log({"curriculum/learning_curves": wandb.Image(tmp.name)})
    plt.close()
    
    # Also create a consolidated view showing progression across all epochs
    plt.figure(figsize=(15, 10))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Prepare consolidated data
    all_train_losses = []
    all_val_losses = []
    all_word_accs = []
    all_char_accs = []
    all_base_char_accs = []
    all_diacritic_accs = []
    stage_boundaries = [0]  # Start with epoch 0
    
    for history in stage_histories:
        all_train_losses.extend(history["epoch_losses"])
        all_val_losses.extend(history["val_losses"])
        all_word_accs.extend(history["word_accs"])
        all_char_accs.extend(history["char_accs"])
        
        if "base_char_accs" in history:
            all_base_char_accs.extend(history["base_char_accs"])
            
        if "diacritic_accs" in history:
            all_diacritic_accs.extend(history["diacritic_accs"])
            
        # Add boundary for next stage
        stage_boundaries.append(stage_boundaries[-1] + len(history["epoch_losses"]))
    
    # Create x-axis values (epoch numbers)
    epochs = range(1, len(all_train_losses) + 1)
    
    # Plot training and validation loss
    ax = axes[0, 0]
    ax.plot(epochs, all_train_losses, 'b-', label='Training Loss')
    ax.plot(epochs, all_val_losses, 'r-', label='Validation Loss')
    # Add stage boundary lines
    for i, boundary in enumerate(stage_boundaries[1:-1], 1):
        ax.axvline(x=boundary, color='k', linestyle='--', alpha=0.7)
        ax.text(boundary, max(all_train_losses) * 0.9, f'Stage {i+1}', 
                rotation=90, verticalalignment='top')
    ax.set_title('Loss Across All Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Plot word accuracy
    ax = axes[0, 1]
    ax.plot(epochs, all_word_accs, 'g-', label='Word Accuracy')
    # Add stage boundary lines
    for i, boundary in enumerate(stage_boundaries[1:-1], 1):
        ax.axvline(x=boundary, color='k', linestyle='--', alpha=0.7)
        ax.text(boundary, min(all_word_accs) * 1.1, f'Stage {i+1}', 
                rotation=90, verticalalignment='bottom')
    ax.set_title('Word Accuracy Across All Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    
    # Plot character accuracies
    ax = axes[1, 0]
    ax.plot(epochs, all_char_accs, 'c-', label='Character Accuracy')
    
    if all_base_char_accs:
        ax.plot(epochs, all_base_char_accs, 'm--', label='Base Character Accuracy')
        
    if all_diacritic_accs:
        ax.plot(epochs, all_diacritic_accs, 'y:', label='Diacritic Accuracy')
        
    # Add stage boundary lines
    for i, boundary in enumerate(stage_boundaries[1:-1], 1):
        ax.axvline(x=boundary, color='k', linestyle='--', alpha=0.7)
        ax.text(boundary, min(all_char_accs) * 1.1, f'Stage {i+1}', 
                rotation=90, verticalalignment='bottom')
    ax.set_title('Character-Level Accuracies Across All Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    
    # Plot learning rate if available (placeholder)
    ax = axes[1, 1]
    ax.text(0.5, 0.5, "Curriculum Summary", 
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=14)
    ax.set_title('Curriculum Learning Progress')
    ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plt.savefig(tmp.name)
        # Log the temporary file to wandb
        wandb.log({"curriculum/consolidated_curves": wandb.Image(tmp.name)})
    plt.close()