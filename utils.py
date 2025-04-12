import os
import wandb
import torch
import matplotlib.pyplot as plt
from PIL import Image
import io

class WandbImageLogger:
    """
    Utility class for logging image predictions to Weights & Biases
    """
    def __init__(self, processor, base_char_vocab, diacritic_vocab, log_interval=50):
        self.processor = processor
        self.base_char_vocab = base_char_vocab
        self.diacritic_vocab = diacritic_vocab
        self.log_interval = log_interval
        self.step_counter = 0
        
    def log_predictions(self, batch, outputs, phase="val"):
        """
        Log a batch of image predictions to wandb
        
        Args:
            batch: The input batch dictionary
            outputs: The model outputs dictionary
            phase: Either "train" or "val"
        """
        self.step_counter += 1
        if self.step_counter % self.log_interval != 0:
            return
            
        # Get predictions
        text_preds = outputs['logits'].argmax(-1)
        base_char_preds = outputs['base_char_logits'].argmax(-1)
        diacritic_preds = outputs['diacritic_logits'].argmax(-1)
        
        # Convert tensors to CPU for processing
        pixel_values = batch['pixel_values'].cpu()
        labels = batch['labels'].cpu()
        base_char_labels = batch['base_character_indices'].cpu()
        diacritic_labels = batch['diacritic_indices'].cpu()
        
        # Debug prints to help identify what's in the batch
        print(f"Batch keys: {batch.keys()}")
        
        # Try to get the words using correct key
        if 'words' in batch:
            words = batch['words']
        elif 'word' in batch:
            words = batch['word']
        else:
            # Fallback if neither key exists
            print("Warning: Neither 'words' nor 'word' found in batch. Using placeholders.")
            words = ["unknown"] * len(pixel_values)
        
        # Similarly for full_characters
        if 'full_characters' in batch:
            full_characters = batch['full_characters']
        else:
            print("Warning: 'full_characters' not found in batch. Using placeholders.")
            full_characters = [["unknown"]] * len(pixel_values)
        
        # Number of samples to log (max 4)
        num_samples = min(4, len(pixel_values))
        
        # Log image predictions
        for i in range(num_samples):
            try:
                # Process image
                image = self._tensor_to_pil(pixel_values[i])
                
                # Get ground truth text
                true_text = words[i] if i < len(words) else "unknown"
                
                # Get predicted text
                pred_text = self._decode_prediction(text_preds[i], labels[i])
                
                # Handle all characters in the sequence, not just the first one
                if i < len(base_char_labels) and i < len(diacritic_labels):
                    # Get the sequence length for this example
                    seq_length = min(
                        base_char_labels[i].size(0),
                        base_char_preds[i].size(0),
                        diacritic_labels[i].size(0),
                        diacritic_preds[i].size(0)
                    )
                    
                    # Process each character position
                    true_chars_info = []
                    pred_chars_info = []
                    
                    for pos in range(seq_length):
                        # Get true character info
                        true_base_idx = base_char_labels[i][pos].item()
                        true_diac_idx = diacritic_labels[i][pos].item()
                        
                        # Only add non-padding indices (assuming 0 is padding)
                        if true_base_idx > 0 or true_diac_idx > 0:
                            true_base = self.base_char_vocab[true_base_idx] if 0 <= true_base_idx < len(self.base_char_vocab) else "?"
                            true_diac = self.diacritic_vocab[true_diac_idx] if 0 <= true_diac_idx < len(self.diacritic_vocab) else "?"
                            true_chars_info.append(f"{true_base}+{true_diac}")
                        
                        # Get predicted character info
                        pred_base_idx = base_char_preds[i][pos].item()
                        pred_diac_idx = diacritic_preds[i][pos].item()
                        
                        # Only add non-padding indices
                        if pred_base_idx > 0 or pred_diac_idx > 0:
                            pred_base = self.base_char_vocab[pred_base_idx] if 0 <= pred_base_idx < len(self.base_char_vocab) else "?"
                            pred_diac = self.diacritic_vocab[pred_diac_idx] if 0 <= pred_diac_idx < len(self.diacritic_vocab) else "?"
                            pred_chars_info.append(f"{pred_base}+{pred_diac}")
                    
                    # Create detailed character breakdowns
                    true_breakdown = ", ".join(true_chars_info) if true_chars_info else "empty"
                    pred_breakdown = ", ".join(pred_chars_info) if pred_chars_info else "empty"
                else:
                    true_breakdown = "unknown"
                    pred_breakdown = "unknown"
                
                # Create caption with detailed character information
                caption = f"True: {true_text} ({true_breakdown})\n"
                caption += f"Pred: {pred_text} ({pred_breakdown})"
                
                # Log to wandb
                wandb.log({
                    f"{phase}/image_examples": wandb.Image(image, caption=caption),
                    "step": self.step_counter
                })
                
            except Exception as e:
                print(f"Error logging image prediction: {e}")
    
    def _tensor_to_pil(self, tensor):
        """Convert a normalized tensor to PIL Image"""
        try:
            # Assuming tensor is in the format [C, H, W] with values normalized
            # Denormalize and convert to uint8
            tensor = tensor.permute(1, 2, 0)  # [H, W, C]
            tensor = tensor * 255
            tensor = tensor.byte().numpy()
            
            return Image.fromarray(tensor)
        except Exception as e:
            print(f"Error converting tensor to PIL image: {e}")
            # Return a small blank image as fallback
            return Image.new('RGB', (10, 10), color='white')
    
    def _decode_prediction(self, pred_ids, label_ids):
        """Decode token IDs to text, handling padding"""
        try:
            # Filter out padding tokens (-100)
            valid_indices = label_ids != -100
            if valid_indices.any():
                valid_pred = pred_ids[valid_indices]
                # Decode to text
                return self.processor.tokenizer.decode(valid_pred, skip_special_tokens=True)
            return ""
        except Exception as e:
            print(f"Error decoding prediction: {e}")
            return "decode_error"

def log_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    """
    Create and log a confusion matrix to wandb
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of label names
        title: Title for the confusion matrix
    """
    from sklearn.metrics import confusion_matrix
    import numpy as np
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save figure to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Log to wandb
    wandb.log({title: wandb.Image(buf)})
    plt.close()


def log_model_architecture(model):
    """
    Log model architecture details to wandb
    
    Args:
        model: PyTorch model
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Log model summary
    wandb.config.update({
        "model/total_parameters": total_params,
        "model/trainable_parameters": trainable_params,
        "model/frozen_parameters": total_params - trainable_params,
    })
    
    # Log model architecture as a text artifact
    model_info = str(model)
    artifact = wandb.Artifact("model_architecture", type="model_info")
    with artifact.new_file("model_architecture.txt") as f:
        f.write(model_info)
    wandb.log_artifact(artifact)


def log_learning_curve(train_losses, val_losses, metrics_dict=None):
    """
    Log learning curves to wandb at the end of training
    
    Args:
        train_losses: List of training losses for each epoch
        val_losses: List of validation losses for each epoch
        metrics_dict: Dictionary of additional metrics to plot {name: [values]}
    """
    # Create figure
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    # Plot additional metrics if provided
    if metrics_dict:
        for name, values in metrics_dict.items():
            if len(values) == len(epochs):
                plt.plot(epochs, values, label=name)
    
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Metric Value')
    plt.legend()
    plt.grid(True)
    
    # Save figure to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Log to wandb
    wandb.log({"learning_curves": wandb.Image(buf)})
    plt.close()

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences in the batch.
    This function correctly pads sequences to the max length in the batch.
    """
    import torch
    
    # Get max lengths for padding
    max_label_len = max([item['labels'].size(0) for item in batch])
    
    # Find max character length across all items in the batch
    max_char_len = max([item['base_character_indices'].size(0) for item in batch])

    # Initialize lists
    pixel_values_list = []
    labels_list = []
    base_char_indices_list = []
    diacritic_indices_list = []
    words_list = []
    full_characters_list = []

    # Process each item in the batch
    for item in batch:
        # Add pixel values - check dimensions and handle accordingly
        pixel_values = item['pixel_values']
        
        # Ensure pixel_values has correct dimensions (B, C, H, W) or (C, H, W)
        if pixel_values.dim() == 3:  # If it's (C, H, W)
            pixel_values_list.append(pixel_values)
        elif pixel_values.dim() == 2:  # If it's (H, W) - grayscale without channel
            # Add channel dimension
            pixel_values = pixel_values.unsqueeze(0)  # Now it's (1, H, W)
            pixel_values_list.append(pixel_values)
        else:
            # Handle unexpected dimensions by warning and using as is
            print(f"Warning: Unexpected pixel_values dimensions: {pixel_values.dim()}")
            pixel_values_list.append(pixel_values)

        # Pad labels
        labels = item['labels']
        padding = torch.ones(max_label_len - labels.size(0), dtype=labels.dtype) * -100
        padded_labels = torch.cat([labels, padding], dim=0)
        labels_list.append(padded_labels)

        # Pad character indices - handle cases where the sequence might be longer than expected
        base_indices = item['base_character_indices']
        base_indices_len = base_indices.size(0)
        
        # Only pad if the current tensor is smaller than max_char_len
        if base_indices_len < max_char_len:
            base_padding = torch.zeros(max_char_len - base_indices_len, dtype=base_indices.dtype)
            padded_base_indices = torch.cat([base_indices, base_padding], dim=0)
        else:
            # If tensor is already at max length or longer, use as is (truncate if needed)
            padded_base_indices = base_indices[:max_char_len]
            
        base_char_indices_list.append(padded_base_indices)

        # Similar padding for diacritic indices
        diacritic_indices = item['diacritic_indices']
        diacritic_indices_len = diacritic_indices.size(0)
        
        # Only pad if the current tensor is smaller than max_char_len
        if diacritic_indices_len < max_char_len:
            diacritic_padding = torch.zeros(max_char_len - diacritic_indices_len, dtype=diacritic_indices.dtype)
            padded_diacritic_indices = torch.cat([diacritic_indices, diacritic_padding], dim=0)
        else:
            # If tensor is already at max length or longer, use as is (truncate if needed)
            padded_diacritic_indices = diacritic_indices[:max_char_len]
            
        diacritic_indices_list.append(padded_diacritic_indices)

        # Add word and full_characters
        words_list.append(item['word'])
        full_characters_list.append(item['full_characters'])

    # Stack all tensors
    return {
        'pixel_values': torch.stack(pixel_values_list),
        'labels': torch.stack(labels_list),
        'base_character_indices': torch.stack(base_char_indices_list),
        'diacritic_indices': torch.stack(diacritic_indices_list),
        'word': words_list,  # Use 'word' to match the dataset key
        'full_characters': full_characters_list,
    }