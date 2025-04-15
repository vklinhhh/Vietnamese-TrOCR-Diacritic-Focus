#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from transformers import TrOCRProcessor
from improved_models import ImprovedVietnameseOCRModel
import time
from typing import List, Dict, Tuple, Optional, Union
import glob
from tqdm import tqdm

def load_model(model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Tuple[ImprovedVietnameseOCRModel, TrOCRProcessor]:
    """
    Load the Vietnamese OCR model and processor from the given path.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on (cuda or cpu)
        
    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading model from {model_path}...")
    print(f"Using device: {device}")
    
    try:
        # Load processor
        processor = TrOCRProcessor.from_pretrained(model_path)
        
        # Load model
        model = ImprovedVietnameseOCRModel.from_pretrained(model_path)
        model.to(device)
        model.eval()
        
        return model, processor
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def preprocess_image(image_path: str, processor: TrOCRProcessor) -> torch.Tensor:
    """
    Preprocess the image for the model.
    
    Args:
        image_path: Path to image file
        processor: TrOCR processor
        
    Returns:
        Preprocessed image tensor
    """
    # Load and convert image to RGB
    image = Image.open(image_path).convert("RGB")
    
    # Process image
    pixel_values = processor(image, return_tensors="pt").pixel_values
    return pixel_values

def predict(model: ImprovedVietnameseOCRModel, processor: TrOCRProcessor, 
                      pixel_values: torch.Tensor, device: str) -> Dict:
    """
    Simplified prediction that uses just the text generation without 
    trying to get character-level predictions.
    """
    # Move inputs to device
    pixel_values = pixel_values.to(device)
    
    # Inference
    with torch.no_grad():
        # Generate text
        generated_ids = model.trocr_model.generate(
            pixel_values,
            max_length=100,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode the generated text
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Create decoder input IDs for confidence calculation
        decoder_input_ids = torch.ones(
            (pixel_values.shape[0], 1), 
            dtype=torch.long, 
            device=device
        ) * model.trocr_model.config.decoder_start_token_id
        
        # Get the model outputs for confidence calculation
        outputs = model.trocr_model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids
        )
        
        # Create character breakdown from the generated text
        char_details = []
        
        for char in generated_text:
            if char.isspace():
                type_name = "space"
            elif any(ord(c) > 127 for c in char):  # Non-ASCII character
                # Vietnamese diacritic mapping
                if char in "áàảãạ":
                    base, type_name = "a", "acute" if char == "á" else "various"
                elif char in "ăắằẳẵặ":
                    base, type_name = "ă", "acute" if char == "ắ" else "various"
                elif char in "âấầẩẫậ":
                    base, type_name = "â", "acute" if char == "ấ" else "various"
                elif char in "éèẻẽẹ":
                    base, type_name = "e", "acute" if char == "é" else "various"
                elif char in "êếềểễệ":
                    base, type_name = "ê", "acute" if char == "ế" else "various"
                elif char in "íìỉĩị":
                    base, type_name = "i", "acute" if char == "í" else "various"
                elif char in "óòỏõọ":
                    base, type_name = "o", "acute" if char == "ó" else "various"
                elif char in "ôốồổỗộ":
                    base, type_name = "ô", "acute" if char == "ố" else "various"
                elif char in "ơớờởỡợ":
                    base, type_name = "ơ", "hook_above" if char == "ở" else "various"
                elif char in "úùủũụ":
                    base, type_name = "u", "acute" if char == "ú" else "various"
                elif char in "ưứừửữự":
                    base, type_name = "ư", "hook_above" if char == "ư" else "various"
                elif char in "ýỳỷỹỵ":
                    base, type_name = "y", "acute" if char == "ý" else "various"
                else:
                    base, type_name = char, "unknown"
                char_details.append(f"{base}+{type_name}")
            else:
                char_details.append(f"{char}+no_diacritic")
        
        # Calculate confidence
        confidence = torch.softmax(outputs.logits[:, 0], dim=-1).max(dim=-1)[0].item()
        
        return {
            "text": generated_text,
            "char_details": char_details,
            "confidence": confidence
        }
# def predict(model: ImprovedVietnameseOCRModel, processor: TrOCRProcessor, 
#             pixel_values: torch.Tensor, device: str) -> Dict:
#     """
#     Run prediction on the preprocessed image.
    
#     Args:
#         model: OCR model
#         processor: TrOCR processor
#         pixel_values: Preprocessed image tensor
#         device: Device to run inference on
        
#     Returns:
#         Dictionary containing prediction results
#     """
#     # Move inputs to device
#     pixel_values = pixel_values.to(device)
    
#     # Inference
#     with torch.no_grad():
#         # Generate text first
#         generated_ids = model.trocr_model.generate(
#             pixel_values,
#             max_length=100,
#             num_beams=4,
#             early_stopping=True
#         )
        
#         # Decode the generated text
#         generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
#         # Use the generated IDs as decoder_input_ids for a complete forward pass
#         # We need to shift them right and add the start token
#         decoder_input_ids = torch.cat([
#             torch.tensor([[model.trocr_model.config.decoder_start_token_id]], device=device),
#             generated_ids[:, :-1]  # Remove the last token (usually end token)
#         ], dim=1)
        
#         # Forward pass to get character predictions for the full sequence
#         outputs = model(
#             pixel_values=pixel_values,
#             decoder_input_ids=decoder_input_ids
#         )
        
#         # Get character predictions
#         base_char_logits = outputs['base_char_logits'][0]  # First batch item
#         diacritic_logits = outputs['diacritic_logits'][0]  # First batch item
        
#         # Get predictions
#         base_char_preds = torch.argmax(base_char_logits, dim=-1).cpu().numpy()
#         diacritic_preds = torch.argmax(diacritic_logits, dim=-1).cpu().numpy()
        
#         # Map indices to characters and diacritics
#         base_char_vocab = model.base_char_vocab
#         diacritic_vocab = model.diacritic_vocab
        
#         # Combine character and diacritic predictions
#         char_details = []
#         for i in range(min(len(base_char_preds), len(diacritic_preds))):
#             base_idx = base_char_preds[i]
#             diacritic_idx = diacritic_preds[i]
            
#             # Only include non-padding characters
#             if base_idx > 0 or diacritic_idx > 0:
#                 base_char = base_char_vocab[base_idx] if 0 <= base_idx < len(base_char_vocab) else "?"
#                 diacritic = diacritic_vocab[diacritic_idx] if 0 <= diacritic_idx < len(diacritic_vocab) else "?"
#                 char_details.append(f"{base_char}+{diacritic}")
        
#         # Calculate confidence scores
#         confidence = torch.softmax(outputs['logits'], dim=-1).max(dim=-1)[0].mean().item()
        
#         return {
#             "text": generated_text,
#             "char_details": char_details,
#             "base_char_preds": base_char_preds,
#             "diacritic_preds": diacritic_preds,
#             "confidence": confidence
#         }



def visualize_results(image_path: str, result: Dict, output_path: Optional[str] = None):
    """
    Visualize the prediction results.
    
    Args:
        image_path: Path to the original image
        result: Prediction results dictionary
        output_path: Path to save the visualization (if None, just display)
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Display image
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    # Create information box
    info_text = f"Predicted Text: {result['text']}\n"
    info_text += f"Confidence: {result['confidence']:.4f}\n\n"
    
    # Add character breakdown
    char_breakdown = ", ".join(result['char_details'])
    info_text += f"Character Breakdown: {char_breakdown}"
    
    # Display info box
    axes[1].text(0.5, 0.5, info_text,
                horizontalalignment='center',
                verticalalignment='center',
                transform=axes[1].transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def process_single_image(model: ImprovedVietnameseOCRModel, processor: TrOCRProcessor, 
                        image_path: str, device: str, visualize: bool = False, 
                        output_dir: Optional[str] = None) -> Dict:
    """
    Process a single image.
    
    Args:
        model: OCR model
        processor: TrOCR processor
        image_path: Path to image file
        device: Device to run inference on
        visualize: Whether to visualize results
        output_dir: Directory to save visualization (if None, just display)
    
    Returns:
        Prediction results
    """
    print(f"Processing image: {image_path}")
    
    # Preprocess image
    pixel_values = preprocess_image(image_path, processor)
    
    # Predict
    start_time = time.time()
    result = predict(model, processor, pixel_values, device)
    inference_time = time.time() - start_time
    
    # Print results
    print(f"Predicted text: {result['text']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Inference time: {inference_time:.4f} seconds")
    
    # Visualize if requested
    if visualize:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            base_filename = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_filename}_result.png")
        else:
            output_path = None
            
        visualize_results(image_path, result, output_path)
    
    return result

def process_batch(model: ImprovedVietnameseOCRModel, processor: TrOCRProcessor, 
                 image_paths: List[str], device: str, visualize: bool = False,
                 output_dir: Optional[str] = None) -> List[Dict]:
    """
    Process a batch of images.
    
    Args:
        model: OCR model
        processor: TrOCR processor
        image_paths: List of paths to image files
        device: Device to run inference on
        visualize: Whether to visualize results
        output_dir: Directory to save visualizations
    
    Returns:
        List of prediction results
    """
    results = []
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            result = process_single_image(model, processor, image_path, device, visualize, output_dir)
            results.append({"path": image_path, "result": result})
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({"path": image_path, "error": str(e)})
    
    # Print summary
    success_count = sum(1 for r in results if "error" not in r)
    print(f"\nProcessed {len(results)} images with {success_count} successes")
    
    return results

def save_results_to_file(results: List[Dict], output_path: str):
    """
    Save results to a text file.
    
    Args:
        results: List of prediction results
        output_path: Path to save results
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            if "error" in item:
                f.write(f"{item['path']}\tERROR: {item['error']}\n")
            else:
                f.write(f"{item['path']}\t{item['result']['text']}\t{item['result']['confidence']:.4f}\n")
    
    print(f"Results saved to {output_path}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Vietnamese OCR Inference")
    parser.add_argument("--mode", type=str, choices=["single", "batch"], default="single",
                      help="Inference mode: single image or batch processing")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the saved model")
    parser.add_argument("--image_path", type=str,
                      help="Path to the input image (for single mode)")
    parser.add_argument("--input_dir", type=str,
                      help="Directory containing images to process (for batch mode)")
    parser.add_argument("--output_dir", type=str, default="results",
                      help="Directory to save results")
    parser.add_argument("--visualize", action="store_true",
                      help="Visualize results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to run inference on (cuda or cpu)")
    parser.add_argument("--file_pattern", type=str, default="*.jpg,*.jpeg,*.png",
                      help="File pattern to match in batch mode (comma-separated)")
    
    args = parser.parse_args()
    
    # Load model
    model, processor = load_model(args.model_path, args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process according to mode
    if args.mode == "single":
        if not args.image_path:
            parser.error("--image_path is required for single mode")
        
        # Process single image
        result = process_single_image(
            model=model,
            processor=processor,
            image_path=args.image_path,
            device=args.device,
            visualize=args.visualize,
            output_dir=args.output_dir if args.visualize else None
        )
        
        # Save result to file
        results_path = os.path.join(args.output_dir, "single_result.txt")
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write(f"Image: {args.image_path}\n")
            f.write(f"Text: {result['text']}\n")
            f.write(f"Confidence: {result['confidence']:.4f}\n")
            f.write(f"Character Breakdown: {', '.join(result['char_details'])}\n")
        
        print(f"Result saved to {results_path}")
        
    else:  # batch mode
        if not args.input_dir:
            parser.error("--input_dir is required for batch mode")
        
        # Get all image files in the directory
        image_paths = []
        for pattern in args.file_pattern.split(','):
            image_paths.extend(glob.glob(os.path.join(args.input_dir, pattern.strip())))
        
        if not image_paths:
            print(f"No images found in {args.input_dir} matching pattern {args.file_pattern}")
            return
        
        print(f"Found {len(image_paths)} images to process")
        
        # Process batch
        results = process_batch(
            model=model,
            processor=processor,
            image_paths=image_paths,
            device=args.device,
            visualize=args.visualize,
            output_dir=args.output_dir if args.visualize else None
        )
        
        # Save results to file
        results_path = os.path.join(args.output_dir, "batch_results.txt")
        save_results_to_file(results, results_path)

if __name__ == "__main__":
    main()