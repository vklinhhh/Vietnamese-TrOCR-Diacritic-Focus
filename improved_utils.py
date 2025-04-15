import os
import wandb
import torch
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
import matplotlib.font_manager as fm
from matplotlib import rcParams
import gc

# Set up font for Vietnamese characters - try different options
# Try to find a font that supports Vietnamese characters
def setup_vietnamese_font():
    # List of font families that typically have good Unicode support
    potential_fonts = [
        'Noto Sans', 'Noto Serif', 'DejaVu Sans', 'Arial Unicode MS', 
        'FreeSans', 'Liberation Sans', 'Ubuntu', 'Segoe UI'
    ]
    
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    
    # Find the first available font from our list
    for font in potential_fonts:
        if font in available_fonts:
            rcParams['font.family'] = font
            print(f"Using font: {font}")
            return font
    
    # If none of our preferred fonts are available, use the default sans-serif
    rcParams['font.family'] = 'sans-serif'
    print("No specific Unicode font found. Using default sans-serif.")
    return 'sans-serif'

# Set up font at module import time
VIETNAMESE_FONT = setup_vietnamese_font()

# Increase the figure warning threshold or disable it
plt.rcParams['figure.max_open_warning'] = 50  # Increase threshold from default 20

# Set auto-close figures
plt.rcParams['figure.figsize'] = [8.0, 6.0]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.autolayout'] = True

class ImprovedWandbImageLogger:
    """
    Improved utility class for logging image predictions to Weights & Biases
    with better handling of different text lengths and Vietnamese characters
    """
    def __init__(self, processor, base_char_vocab, diacritic_vocab, log_interval=50):
        self.processor = processor
        self.base_char_vocab = base_char_vocab
        self.diacritic_vocab = diacritic_vocab
        self.log_interval = log_interval
        self.step_counter = 0
        
    def log_predictions(self, batch, outputs, phase="val"):
        """
        Log a batch of image predictions to wandb with improved visualization
        
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
        
        # For collecting log data
        log_data = {}
        
        # Log image predictions
        for i in range(num_samples):
            try:
                # Process image
                image = self._tensor_to_pil(pixel_values[i])
                
                # Get ground truth text
                true_text = words[i] if i < len(words) else "unknown"
                
                # Get predicted text
                pred_text = self._decode_prediction(text_preds[i], labels[i])
                
                # Determine text length category for better visualization
                text_category = self._determine_text_category(true_text)
                
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
                
                # Create visualized image based on text category
                wandb_img = self._create_visualization(image, true_text, pred_text, true_breakdown, pred_breakdown, text_category)
                
                # Add to log_data
                log_key = f"{phase}/image_examples/{text_category}_{i}"
                log_data[log_key] = wandb_img
            
            except Exception as e:
                print(f"Error logging image prediction: {e}")
                import traceback
                traceback.print_exc()
        
        # Add step counter to log data
        log_data["step"] = self.step_counter
        
        # Log all images at once to reduce wandb API calls
        wandb.log(log_data)
        
        # Force garbage collection to free memory
        gc.collect()
    
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
    
    def _determine_text_category(self, text):
        """Determine if the text is a character, word, or sentence"""
        if not text or len(text) == 0:
            return "unknown"
        
        # If there are spaces, it's likely a sentence
        if " " in text:
            if len(text) > 30:
                return "long_sentence"
            else:
                return "sentence"
        
        # If no spaces but multiple characters, it's a word
        if len(text) > 1:
            return "word"
        
        # Single character
        return "character"
    
    def _create_visualization(self, image, true_text, pred_text, true_breakdown, pred_breakdown, text_category):
        """Create a visualization with proper scaling based on text category"""
        # Adjust figure size based on text category
        if text_category == "long_sentence":
            fig, axes = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={'height_ratios': [1, 1]})
        elif text_category == "sentence":
            fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [1, 1]})
        elif text_category == "word":
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 1]})
        else:  # character
            fig, axes = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [1, 1]})
        
        # Set the font for Vietnamese characters
        plt.rcParams['font.family'] = VIETNAMESE_FONT
        
        # Plot the image with aspect ratio preserved
        axes[0].imshow(image)
        axes[0].set_title(f"Input Image ({text_category})")
        axes[0].axis('off')
        
        # Create information box
        info_text = f"True: {true_text}\nPred: {pred_text}\n\n"
        
        # For sentences, we'll format the breakdown differently
        if text_category in ["sentence", "long_sentence"]:
            # Split the breakdown into multiple lines for better readability
            true_parts = true_breakdown.split(", ")
            pred_parts = pred_breakdown.split(", ")
            
            # Group into chunks of 5 or 10 based on length
            chunk_size = 5 if text_category == "sentence" else 10
            true_chunks = [", ".join(true_parts[i:i+chunk_size]) for i in range(0, len(true_parts), chunk_size)]
            pred_chunks = [", ".join(pred_parts[i:i+chunk_size]) for i in range(0, len(pred_parts), chunk_size)]
            
            info_text += "True Breakdown:\n" + "\n".join(true_chunks) + "\n\n"
            info_text += "Pred Breakdown:\n" + "\n".join(pred_chunks)
        else:
            info_text += f"True Breakdown: {true_breakdown}\nPred Breakdown: {pred_breakdown}"
        
        # Create box with text information
        axes[1].text(0.5, 0.5, info_text, 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=axes[1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10 if text_category in ["sentence", "long_sentence"] else 12)
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Convert plot to wandb Image
        wandb_img = wandb.Image(fig)
        
        # CRITICAL: Close the figure to prevent memory leak
        plt.close(fig)
        
        return wandb_img

# Function to integrate the improved image logger into the existing code
def replace_image_logger(utils_file_path):
    """
    Helper function to replace the existing WandbImageLogger with the improved version
    """
    with open(utils_file_path, 'r') as file:
        content = file.read()
    
    # Find the WandbImageLogger class and replace it
    import_section = """import os
import wandb
import torch
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
import matplotlib.font_manager as fm
from matplotlib import rcParams
import gc

# Set up font for Vietnamese characters - try different options
def setup_vietnamese_font():
    # List of font families that typically have good Unicode support
    potential_fonts = [
        'Noto Sans', 'Noto Serif', 'DejaVu Sans', 'Arial Unicode MS', 
        'FreeSans', 'Liberation Sans', 'Ubuntu', 'Segoe UI'
    ]
    
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    
    # Find the first available font from our list
    for font in potential_fonts:
        if font in available_fonts:
            rcParams['font.family'] = font
            print(f"Using font: {font}")
            return font
    
    # If none of our preferred fonts are available, use the default sans-serif
    rcParams['font.family'] = 'sans-serif'
    print("No specific Unicode font found. Using default sans-serif.")
    return 'sans-serif'

# Set up font at module import time
VIETNAMESE_FONT = setup_vietnamese_font()

# Increase the figure warning threshold and enable auto-closing
plt.rcParams['figure.max_open_warning'] = 50
plt.rcParams['figure.figsize'] = [8.0, 6.0]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.autolayout'] = True
"""
    
    # Extract the beginning of the file up to the WandbImageLogger class
    start_idx = content.find("class WandbImageLogger")
    if start_idx == -1:
        print("WandbImageLogger class not found in the file.")
        return False
    
    # Find the end of the class (next class definition or end of file)
    end_idx = content.find("class ", start_idx + 1)
    if end_idx == -1:
        end_idx = len(content)
    
    # Create the new content by replacing the WandbImageLogger class
    new_content = (
        content[:start_idx] + 
        "class WandbImageLogger(ImprovedWandbImageLogger):\n    \"\"\"\n    Legacy wrapper for backward compatibility\n    \"\"\"\n    pass\n\n" + 
        content[end_idx:]
    )
    
    # Write the new content back to the file
    with open(utils_file_path, 'w') as file:
        file.write(import_section + "\n" + ImprovedWandbImageLogger.__doc__ + "\n" + new_content)
    
    print(f"Successfully updated {utils_file_path} with improved image logger.")
    return True