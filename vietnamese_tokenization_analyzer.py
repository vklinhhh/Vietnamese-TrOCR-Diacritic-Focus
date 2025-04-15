#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from transformers import TrOCRProcessor
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

def analyze_tokenization(text, processor):
    """
    Analyze how the tokenizer processes the text.
    
    Args:
        text: String to analyze
        processor: TrOCR processor with tokenizer
    
    Returns:
        Dictionary with tokenization analysis
    """
    # Tokenize the text
    encoding = processor.tokenizer(text, return_tensors="pt")
    token_ids = encoding.input_ids[0]
    
    # Get individual tokens
    tokens = []
    for token_id in token_ids:
        # Get string representation of each token
        token = processor.tokenizer.decode([token_id.item()])
        tokens.append({
            "id": token_id.item(),
            "token": token,
            "bytes": [ord(c) for c in token]
        })
    
    # Get character-by-character representation
    chars = []
    for char in text:
        # Get the individual character encoding
        char_encoding = processor.tokenizer(char, return_tensors="pt")
        char_token_ids = char_encoding.input_ids[0]
        
        chars.append({
            "char": char,
            "token_ids": char_token_ids.tolist(),
            "tokens": processor.tokenizer.decode(char_token_ids, skip_special_tokens=True),
            "bytes": [ord(c) for c in char]
        })
    
    return {
        "original_text": text,
        "token_count": len(tokens),
        "tokens": tokens,
        "character_count": len(chars),
        "characters": chars
    }

def print_tokenization_analysis(text, processor):
    """
    Print a detailed analysis of tokenization.
    """
    analysis = analyze_tokenization(text, processor)
    
    print(f"Text: '{analysis['original_text']}'")
    print(f"Total tokens: {analysis['token_count']}")
    print(f"Total characters: {analysis['character_count']}")
    
    print("\nToken breakdown:")
    for i, token in enumerate(analysis['tokens']):
        print(f"  {i+1}. ID: {token['id']}, Token: '{token['token']}'")
    
    print("\nCharacter breakdown:")
    for i, char in enumerate(analysis['characters']):
        token_str = ', '.join([str(tid) for tid in char['token_ids']])
        print(f"  {i+1}. '{char['char']}' → Token IDs: [{token_str}], Decoded: '{char['tokens']}'")
    
    print("\nAlignment issues:")
    # Look for characters that map to multiple tokens or tokens that contain multiple chars
    for char in analysis['characters']:
        if len(char['token_ids']) > 3:  # More than start + char + end tokens
            print(f"  Character '{char['char']}' maps to {len(char['token_ids'])-2} tokens")
    
    # Check for Vietnamese-specific issues
    vietnamese_chars = [c for c in analysis['characters'] if any(ord(ch) > 127 for ch in c['char'])]
    if vietnamese_chars:
        print("\nVietnamese character analysis:")
        for char in vietnamese_chars:
            print(f"  '{char['char']}' → Token IDs: {char['token_ids']}, Decoded: '{char['tokens']}'")
    
    return analysis

def visualize_tokenization(text, processor, output_path=None):
    """
    Create a visual representation of how text is tokenized
    """
    analysis = analyze_tokenization(text, processor)
    
    # Setup the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create a visual representation of the tokenization
    chars = list(text)
    token_boundaries = []
    
    # Find token boundaries
    current_pos = 0
    decoded_tokens = []
    
    # Decode each token and track its length in the original text
    for token in analysis['tokens']:
        # Skip special tokens like <s> and </s>
        if token['id'] < 3:  # Usually 0, 1, 2 are special tokens
            continue
            
        decoded = token['token']
        decoded_tokens.append(decoded)
        current_pos += len(decoded)
        token_boundaries.append(current_pos)
    
    # Remove the last boundary if it's at the end
    if token_boundaries and token_boundaries[-1] >= len(chars):
        token_boundaries = token_boundaries[:-1]
    
    # Draw the characters with token boundaries
    y_pos = 0
    for i, char in enumerate(chars):
        ax.text(i+0.5, y_pos+0.5, char, ha='center', va='center', 
                fontsize=16, bbox=dict(facecolor='white', edgecolor='black', pad=5))
        
        # Draw a vertical line for token boundaries
        if i in token_boundaries:
            ax.axvline(x=i+1, color='red', linestyle='-', linewidth=2)
    
    # Add token labels below
    y_pos = -0.5
    x_start = 0
    for i, token in enumerate(decoded_tokens):
        token_length = len(token)
        x_center = x_start + token_length / 2
        
        ax.text(x_center, y_pos, f"Token {i+1}", ha='center', va='center',
                fontsize=10, color='blue')
        
        x_start += token_length
    
    # Set axis limits and remove ticks
    ax.set_xlim(-0.5, len(chars)+0.5)
    ax.set_ylim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set title
    ax.set_title(f"Tokenization of: '{text}'")
    
    # Save or show the figure
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def compare_tokenizations(text1, text2, processor, output_path=None):
    """
    Compare two tokenizations side by side.
    
    Args:
        text1: First text (e.g., predicted text)
        text2: Second text (e.g., character breakdown text)
        processor: TrOCR processor
        output_path: Path to save visualization
    """
    analysis1 = analyze_tokenization(text1, processor)
    analysis2 = analyze_tokenization(text2, processor)
    
    # Setup the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Helper function to draw tokenization
    def draw_tokenization(ax, text, analysis, title):
        chars = list(text)
        token_boundaries = []
        
        # Find token boundaries
        current_pos = 0
        decoded_tokens = []
        
        # Decode each token and track its length in the original text
        for token in analysis['tokens']:
            # Skip special tokens like <s> and </s>
            if token['id'] < 3:  # Usually 0, 1, 2 are special tokens
                continue
                
            decoded = token['token']
            decoded_tokens.append(decoded)
            current_pos += len(decoded)
            token_boundaries.append(current_pos)
        
        # Remove the last boundary if it's at the end
        if token_boundaries and token_boundaries[-1] >= len(chars):
            token_boundaries = token_boundaries[:-1]
        
        # Draw the characters with token boundaries
        for i, char in enumerate(chars):
            ax.text(i+0.5, 0.5, char, ha='center', va='center', 
                    fontsize=16, bbox=dict(facecolor='white', edgecolor='black', pad=5))
            
            # Draw a vertical line for token boundaries
            if i in token_boundaries:
                ax.axvline(x=i+1, color='red', linestyle='-', linewidth=2)
        
        # Add token labels below
        y_pos = -0.5
        x_start = 0
        for i, token in enumerate(decoded_tokens):
            token_length = len(token)
            x_center = x_start + token_length / 2
            
            ax.text(x_center, y_pos, f"Token {i+1}", ha='center', va='center',
                    fontsize=10, color='blue')
            
            x_start += token_length
        
        # Set axis limits and remove ticks
        ax.set_xlim(-0.5, max(len(chars), 1)+0.5)
        ax.set_ylim(-1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
    
    # Draw each tokenization
    draw_tokenization(ax1, text1, analysis1, f"Text 1: '{text1}'")
    draw_tokenization(ax2, text2, analysis2, f"Text 2: '{text2}'")
    
    # Add overall title
    fig.suptitle("Tokenization Comparison", fontsize=16)
    
    # Save or show
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make space for suptitle
    if output_path:
        plt.savefig(output_path)
        print(f"Comparison visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze TrOCR tokenization for Vietnamese text")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--text2", type=str, help="Second text to compare (optional)")
    parser.add_argument("--model_path", type=str, default="microsoft/trocr-base-handwritten", 
                        help="Path to TrOCR model or processor")
    parser.add_argument("--visualize", action="store_true", help="Create visualization")
    parser.add_argument("--output_dir", type=str, default="tokenization_analysis",
                        help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.visualize:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load processor
    try:
        processor = TrOCRProcessor.from_pretrained(args.model_path)
        print(f"Loaded processor from {args.model_path}")
    except Exception as e:
        print(f"Error loading processor: {e}")
        print("Using default TrOCR processor instead")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    
    # Analyze text
    if args.text:
        print(f"\n=== Analyzing Text: '{args.text}' ===")
        analysis = print_tokenization_analysis(args.text, processor)
        
        if args.visualize:
            output_path = os.path.join(args.output_dir, "tokenization_visualization.png")
            visualize_tokenization(args.text, processor, output_path)
    
    # Compare texts if second text is provided
    if args.text2:
        print(f"\n=== Comparing with Text 2: '{args.text2}' ===")
        analysis2 = print_tokenization_analysis(args.text2, processor)
        
        # Highlight differences
        print("\n=== Differences ===")
        print(f"Text 1: '{args.text}'")
        print(f"Text 2: '{args.text2}'")
        
        # Compare token counts
        print(f"Token count: {analysis['token_count']} vs {analysis2['token_count']}")
        
        if args.visualize:
            output_path = os.path.join(args.output_dir, "tokenization_comparison.png")
            compare_tokenizations(args.text, args.text2, processor, output_path)
    
    # If no text provided, use examples
    if not args.text:
        examples = [
            "Vở Khánh Linh",
            "cách giản dị đến bất ngờ",
            "Tiếng Việt",
            "Vởhánhhinh"  # Combined form of the first example
        ]
        
        for i, example in enumerate(examples):
            print(f"\n=== Example {i+1}: '{example}' ===")
            analysis = print_tokenization_analysis(example, processor)
            
            if args.visualize:
                output_path = os.path.join(args.output_dir, f"example_{i+1}_tokenization.png")
                visualize_tokenization(example, processor, output_path)

if __name__ == "__main__":
    main()