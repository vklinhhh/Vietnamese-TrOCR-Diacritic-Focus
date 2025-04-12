import torch
from PIL import Image
import numpy as np
from transformers import TrOCRProcessor
from models import VietnameseOCRModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from dataset import VietnameseCharacterDataset
from utils import custom_collate_fn
from tqdm import tqdm
import argparse
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

def predict_single_image(image_path, model, processor, device):
    """
    Predict text and character details for a single image
    """
    # Load and process the image
    image = Image.open(image_path).convert('RGB')
    
    # Process image
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    
    # Initialize the decoder inputs for generation
    tokenizer = processor.tokenizer
    model.trocr_model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.trocr_model.config.pad_token_id = tokenizer.pad_token_id
    model.trocr_model.config.eos_token_id = tokenizer.eos_token_id
    
    # Generate output using the model's generate method
    generated_ids = model.trocr_model.generate(
        pixel_values,
        max_length=32,
        num_beams=4,
        early_stopping=True
    )
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Now run again with output_hidden_states=True to get the features for diacritic prediction
    with torch.no_grad():
        outputs = model.trocr_model(
            pixel_values=pixel_values,
            decoder_input_ids=generated_ids,
            output_hidden_states=True
        )
    
    # Get decoder hidden states for final position
    decoder_hidden_states = outputs.decoder_hidden_states[-1]
    
    # Apply classifiers to all positions
    base_char_logits = model.base_char_head(decoder_hidden_states)
    diacritic_logits = model.diacritic_head(decoder_hidden_states)
    
    # Get base character and diacritic predictions
    base_char_preds = base_char_logits.argmax(-1)
    diacritic_preds = diacritic_logits.argmax(-1)
    
    # Map indices to characters and diacritics
    base_char_seq = []
    diacritic_seq = []
    
    # Get the length of the actual predicted text to limit processing of relevant positions
    actual_text_length = len(generated_text)
    seq_length = min(base_char_preds.size(1), diacritic_preds.size(1), actual_text_length + 2)
    
    valid_chars = 0
    for pos in range(seq_length):
        base_idx = base_char_preds[0, pos].item()
        diacritic_idx = diacritic_preds[0, pos].item()
        
        if base_idx < len(model.base_char_vocab):
            base_char = model.base_char_vocab[base_idx]
            
            # Skip common special tokens, punctuation, and padding
            if base_char in ['!', '.', ',', '?', '<pad>', '<unk>', '<s>', '</s>']:
                continue
                
            base_char_seq.append(base_char)
            
            if diacritic_idx < len(model.diacritic_vocab):
                diacritic = model.diacritic_vocab[diacritic_idx]
                diacritic_seq.append(diacritic)
            else:
                diacritic_seq.append('no_diacritic')  # Default if out of range
                
            valid_chars += 1
            
            # If we've found enough characters to match the generated text, stop
            if valid_chars >= actual_text_length:
                break
    
    # Ensure sequences are same length by truncating longer one
    min_len = min(len(base_char_seq), len(diacritic_seq))
    base_char_seq = base_char_seq[:min_len]
    diacritic_seq = diacritic_seq[:min_len]
    
    return {
        "text": generated_text,
        "base_characters": base_char_seq,
        "diacritics": diacritic_seq
    }

def evaluate_dataset(dataset_name, split, model_path, batch_size=8, num_samples=None):
    """
    Evaluate the model on a HuggingFace dataset
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        split: Dataset split to use ('train', 'test', 'validation')
        model_path: Path to the saved model
        batch_size: Batch size for evaluation
        num_samples: Number of samples to evaluate (None for all)
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and processor
    model = VietnameseOCRModel.from_pretrained(model_path)
    processor = TrOCRProcessor.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load dataset
    print(f"Loading dataset {dataset_name} ({split} split)...")
    hf_dataset = load_dataset(dataset_name, split=split)
    
    # Limit samples if specified
    if num_samples is not None and num_samples < len(hf_dataset):
        hf_dataset = hf_dataset.select(range(num_samples))
    
    print(f"Evaluating on {len(hf_dataset)} samples")
    
    # Extract vocabularies from the model
    base_char_vocab = model.base_char_vocab
    diacritic_vocab = model.diacritic_vocab
    
    # Create dataset wrapper
    test_dataset = VietnameseCharacterDataset(
        hf_dataset,
        processor=processor,
        base_char_vocab=base_char_vocab,
        diacritic_vocab=diacritic_vocab,
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        collate_fn=custom_collate_fn
    )
    
    # Evaluation metrics
    all_pred_texts = []
    all_true_texts = []
    all_base_char_preds = []
    all_base_char_labels = []
    all_diacritic_preds = []
    all_diacritic_labels = []
    
    # Tokenizer configuration
    tokenizer = processor.tokenizer
    model.trocr_model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.trocr_model.config.pad_token_id = tokenizer.pad_token_id
    model.trocr_model.config.eos_token_id = tokenizer.eos_token_id
    
    # Evaluation loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move batch to device
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            base_char_labels = batch['base_character_indices'].to(device)
            diacritic_labels = batch['diacritic_indices'].to(device)
            
            # Get true words
            true_words = batch['words']
            
            # Generate predictions
            generated_ids = model.trocr_model.generate(
                pixel_values,
                max_length=32,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode predicted texts
            predicted_texts = [
                tokenizer.decode(ids, skip_special_tokens=True) 
                for ids in generated_ids
            ]
            
            # Get character-level predictions using the generated_ids
            outputs = model.trocr_model(
                pixel_values=pixel_values,
                decoder_input_ids=generated_ids,  # Use the generated ids as input
                output_hidden_states=True
            )
            
            # Get decoder hidden states
            decoder_hidden_states = outputs.decoder_hidden_states[-1]
            
            # Apply classifiers
            base_char_logits = model.base_char_head(decoder_hidden_states)
            diacritic_logits = model.diacritic_head(decoder_hidden_states)
            
            # Get predictions
            base_char_preds = base_char_logits.argmax(-1)
            diacritic_preds = diacritic_logits.argmax(-1)
            
            # Process predictions for valid positions
            for i, (pred_text, true_word) in enumerate(zip(predicted_texts, true_words)):
                all_pred_texts.append(pred_text)
                all_true_texts.append(true_word)
                
                # Process character-level predictions
                valid_indices = base_char_labels[i] != 0  # Assuming 0 is padding
                
                # Extract valid positions only
                valid_base_preds = base_char_preds[i, :valid_indices.sum()]
                valid_base_labels = base_char_labels[i, valid_indices]
                
                valid_diacritic_preds = diacritic_preds[i, :valid_indices.sum()]
                valid_diacritic_labels = diacritic_labels[i, valid_indices]
                
                # Add to lists
                all_base_char_preds.extend(valid_base_preds.cpu().numpy())
                all_base_char_labels.extend(valid_base_labels.cpu().numpy())
                all_diacritic_preds.extend(valid_diacritic_preds.cpu().numpy())
                all_diacritic_labels.extend(valid_diacritic_labels.cpu().numpy())
    
    # Calculate metrics
    # 1. Text-level metrics (exact match and character error rate)
    exact_match = sum([p == t for p, t in zip(all_pred_texts, all_true_texts)]) / len(all_true_texts)
    
    # Calculate character error rate
    total_chars = sum(len(t) for t in all_true_texts)
    edit_distances = sum(levenshtein_distance(p, t) for p, t in zip(all_pred_texts, all_true_texts))
    character_error_rate = edit_distances / total_chars if total_chars > 0 else 0
    
    # 2. Character-level metrics
    base_char_accuracy = accuracy_score(all_base_char_labels, all_base_char_preds)
    diacritic_accuracy = accuracy_score(all_diacritic_labels, all_diacritic_preds)
    
    # Print detailed metrics
    print("\nEvaluation Results:")
    print(f"Number of samples: {len(all_true_texts)}")
    print(f"Exact match: {exact_match:.4f}")
    print(f"Character error rate: {character_error_rate:.4f}")
    print(f"Base character accuracy: {base_char_accuracy:.4f}")
    print(f"Diacritic accuracy: {diacritic_accuracy:.4f}")
    
    # Generate classification reports if there are enough samples
    if len(all_base_char_labels) > 0:
        # Convert numeric indices to actual character labels for readability
        base_char_label_names = [base_char_vocab[i] for i in sorted(set(all_base_char_labels))]
        diacritic_label_names = [diacritic_vocab[i] for i in sorted(set(all_diacritic_labels))]
        
        print("\nBase Character Classification Report:")
        base_report = classification_report(
            all_base_char_labels, 
            all_base_char_preds,
            target_names=base_char_label_names,
            zero_division=0
        )
        print(base_report)
        
        print("\nDiacritic Classification Report:")
        diac_report = classification_report(
            all_diacritic_labels, 
            all_diacritic_preds,
            target_names=diacritic_label_names,
            zero_division=0
        )
        print(diac_report)
    
    # Return metrics dictionary for further analysis
    return {
        "exact_match": exact_match,
        "character_error_rate": character_error_rate,
        "base_char_accuracy": base_char_accuracy,
        "diacritic_accuracy": diacritic_accuracy,
        "predictions": list(zip(all_true_texts, all_pred_texts))
    }

def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    Used for character error rate calculation.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Calculate cost of operations
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def visualize_results(image_path, prediction_result):
    """
    Visualize prediction results for a single image
    """
    image = Image.open(image_path).convert('RGB')
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')
    
    plt.subplot(2, 1, 2)
    plt.axis('off')
    plt.text(0.5, 0.8, f"Predicted Text: {prediction_result['text']}", 
            fontsize=15, ha='center', transform=plt.gca().transAxes)
    
    # Display character details
    char_details = []
    for base, diac in zip(prediction_result['base_characters'], prediction_result['diacritics']):
        char_details.append(f"{base}+{diac}")
    
    plt.text(0.5, 0.5, "Character Analysis:\n" + " | ".join(char_details),
            fontsize=12, ha='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vietnamese OCR Evaluation")
    
    # Add mode selection
    parser.add_argument("--mode", type=str, required=True, choices=['single', 'batch'],
                        help="Evaluation mode: 'single' for single image or 'batch' for dataset")
    
    # Single image mode arguments
    parser.add_argument("--image_path", type=str, help="Path to input image (for single mode)")
    parser.add_argument("--visualize", action="store_true", help="Visualize results (for single mode)")
    
    # Batch mode arguments
    parser.add_argument("--dataset_name", type=str, help="HuggingFace dataset name (for batch mode)")
    parser.add_argument("--split", type=str, default="test", 
                        help="Dataset split to use: 'train', 'test', 'validation' (for batch mode)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate, None for all (for batch mode)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation (for batch mode)")
    
    # Common arguments
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to saved model directory")
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'single':
            if not args.image_path:
                raise ValueError("--image_path is required for single mode")
                
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load model and processor
            model = VietnameseOCRModel.from_pretrained(args.model_path)
            processor = TrOCRProcessor.from_pretrained(args.model_path)
            model.to(device)
            model.eval()
            
            # Predict
            result = predict_single_image(args.image_path, model, processor, device)
            
            # Print results
            print(f"Predicted text: {result['text']}")
            print(f"Base characters: {result['base_characters']}")
            print(f"Diacritics: {result['diacritics']}")
            
            # Combined results
            if len(result['base_characters']) == len(result['diacritics']):
                combined = []
                for base, diac in zip(result['base_characters'], result['diacritics']):
                    combined.append(f"{base}+{diac}")
                print(f"Combined results: {combined}")
            
            # Visualize if requested
            if args.visualize:
                visualize_results(args.image_path, result)
                
        elif args.mode == 'batch':
            if not args.dataset_name:
                raise ValueError("--dataset_name is required for batch mode")
                
            # Run batch evaluation
            metrics = evaluate_dataset(
                args.dataset_name,
                args.split,
                args.model_path,
                batch_size=args.batch_size,
                num_samples=args.num_samples
            )
            
            # Print summary
            print("\nEvaluation Summary:")
            print(f"Exact Match Accuracy: {metrics['exact_match']:.4f}")
            print(f"Character Error Rate: {metrics['character_error_rate']:.4f}")
            print(f"Base Character Accuracy: {metrics['base_char_accuracy']:.4f}")
            print(f"Diacritic Accuracy: {metrics['diacritic_accuracy']:.4f}")
            
            # Save detailed results to file if needed
            with open("evaluation_results.txt", "w") as f:
                f.write("True Text,Predicted Text\n")
                for true, pred in metrics['predictions']:
                    f.write(f"\"{true}\",\"{pred}\"\n")
            
            print("Detailed results saved to evaluation_results.txt")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)