import argparse
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import re
import ast
from collections import Counter
import os

def process_item_safely(item):
    """Parse string representations of lists into actual lists"""
    if isinstance(item, list):
        return item
    elif isinstance(item, str):
        # Check if it's a string representation of a list
        if item.startswith('[') and item.endswith(']'):
            try:
                return ast.literal_eval(item)
            except (SyntaxError, ValueError):
                return [item]
        else:
            return [item]
    else:
        return [str(item)]

def analyze_dataset_complexity(dataset_name, split='train', output_dir='dataset_analysis'):
    """
    Analyze a dataset to understand its complexity distribution for curriculum learning.
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to analyze (train, validation, test)
        output_dir: Directory to save analysis plots
    """
    print(f"Loading dataset {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name)
        if split not in dataset:
            # If requested split doesn't exist, use 'train'
            print(f"Split '{split}' not found. Using 'train' instead.")
            split = 'train'
        
        dataset = dataset[split]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print(f"Analyzing {len(dataset)} examples...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize counters and data for analysis
    char_lengths = []  # Single character lengths
    word_lengths = []  # Word lengths (in characters)
    sentence_lengths = []  # Sentence lengths (in words)
    diacritic_counts = []  # Number of diacritics per example
    diacritic_types = Counter()  # Count of each diacritic type
    base_char_types = Counter()  # Count of each base character type
    
    # Extract key information for each example
    for example in dataset:
        # Get the text representation (could be character, word, or sentence)
        if 'full_character' in example:
            full_chars = process_item_safely(example['full_character'])
            if isinstance(full_chars, list):
                text = ''.join(full_chars)
            else:
                text = full_chars
        elif 'word' in example:
            text = example['word']
        elif 'text' in example:
            text = example['text']
        else:
            # Try to find any field that might contain text
            text_candidates = [v for k, v in example.items() 
                              if isinstance(v, str) and len(v) > 0 and not k.startswith('image')]
            text = text_candidates[0] if text_candidates else ""
        
        # Process diacritic types
        if 'diacritic_type' in example:
            diacritics = process_item_safely(example['diacritic_type'])
            # Count non-empty diacritics
            diacritic_count = sum(1 for d in diacritics if d and d != 'none')
            diacritic_counts.append(diacritic_count)
            
            # Track diacritic types
            for d in diacritics:
                if d and d != 'none':
                    diacritic_types[d] += 1
        
        # Process base characters
        if 'base_character' in example:
            base_chars = process_item_safely(example['base_character'])
            for c in base_chars:
                if c:
                    base_char_types[c] += 1
        
        # Analyze text length characteristics
        # Remove whitespace for consistent counting
        clean_text = text.strip()
        
        # Character-level analysis
        if len(clean_text) == 1:
            char_lengths.append(1)
        
        # Word-level analysis (if has spaces, it's likely a word or sentence)
        elif ' ' not in clean_text:
            word_lengths.append(len(clean_text))
        
        # Sentence-level analysis
        else:
            words = clean_text.split()
            sentence_lengths.append(len(words))
            # Also track the individual words
            for word in words:
                word_lengths.append(len(word))
    
    # Print summary statistics
    print("\n=== Dataset Complexity Analysis ===")
    print(f"Total examples: {len(dataset)}")
    
    def print_stats(data, name):
        if data:
            print(f"\n{name} Statistics:")
            print(f"  Count: {len(data)}")
            print(f"  Min: {min(data)}")
            print(f"  Max: {max(data)}")
            print(f"  Mean: {np.mean(data):.2f}")
            print(f"  Median: {np.median(data):.2f}")
            print(f"  Std Dev: {np.std(data):.2f}")
    
    print_stats(char_lengths, "Single Character")
    print_stats(word_lengths, "Word")
    print_stats(sentence_lengths, "Sentence")
    print_stats(diacritic_counts, "Diacritics per Example")
    
    print("\nMost Common Base Characters:")
    for char, count in base_char_types.most_common(10):
        print(f"  '{char}': {count}")
    
    print("\nMost Common Diacritic Types:")
    for diac, count in diacritic_types.most_common(10):
        print(f"  '{diac}': {count}")
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Word length distribution
    if word_lengths:
        ax = axes[0, 0]
        ax.hist(word_lengths, bins=min(20, max(word_lengths)), alpha=0.7)
        ax.set_title('Word Length Distribution')
        ax.set_xlabel('Characters per Word')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Sentence length distribution
    if sentence_lengths:
        ax = axes[0, 1]
        ax.hist(sentence_lengths, bins=min(20, max(sentence_lengths)), alpha=0.7)
        ax.set_title('Sentence Length Distribution')
        ax.set_xlabel('Words per Sentence')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Diacritic count distribution
    if diacritic_counts:
        ax = axes[1, 0]
        ax.hist(diacritic_counts, bins=min(10, max(diacritic_counts) + 1), alpha=0.7)
        ax.set_title('Diacritics per Example Distribution')
        ax.set_xlabel('Number of Diacritics')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Most common base characters
    ax = axes[1, 1]
    most_common = base_char_types.most_common(10)
    if most_common:
        chars, counts = zip(*most_common)
        y_pos = np.arange(len(chars))
        ax.barh(y_pos, counts, align='center', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(chars)
        ax.set_title('Most Common Base Characters')
        ax.set_xlabel('Count')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'complexity_distributions.png'))
    
    # Also plot diacritic type distribution
    if diacritic_types:
        plt.figure(figsize=(10, 6))
        most_common = diacritic_types.most_common(10)
        diac, counts = zip(*most_common)
        y_pos = np.arange(len(diac))
        plt.barh(y_pos, counts, align='center', alpha=0.7)
        plt.yticks(y_pos, diac)
        plt.title('Most Common Diacritic Types')
        plt.xlabel('Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'diacritic_distribution.png'))
    
    # Create a complexity score distribution that combines multiple factors
    complexity_scores = []
    
    # Process each example to calculate complexity score
    for i, example in enumerate(dataset):
        score = 0
        
        # Get text representation
        if 'full_character' in example:
            full_chars = process_item_safely(example['full_character'])
            if isinstance(full_chars, list):
                text = ''.join(full_chars)
            else:
                text = full_chars
        elif 'word' in example:
            text = example['word']
        elif 'text' in example:
            text = example['text']
        else:
            text_candidates = [v for k, v in example.items() 
                              if isinstance(v, str) and len(v) > 0 and not k.startswith('image')]
            text = text_candidates[0] if text_candidates else ""
        
        # Calculate text length component
        text_len = len(text.strip())
        score += text_len
        
        # Add diacritic component
        if 'diacritic_type' in example:
            diacritics = process_item_safely(example['diacritic_type'])
            diacritic_count = sum(1 for d in diacritics if d and d != 'none')
            # Weight diacritics more heavily (they add complexity)
            score += diacritic_count * 2
        
        complexity_scores.append(score)
    
    # Plot complexity score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(complexity_scores, bins=30, alpha=0.7)
    plt.title('Complexity Score Distribution')
    plt.xlabel('Complexity Score (length + diacritics*2)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Calculate curriculum stages based on score distribution
    percentiles = [0, 33, 66, 100]
    thresholds = [np.percentile(complexity_scores, p) for p in percentiles]
    
    # Mark thresholds on the histogram
    for i, thresh in enumerate(thresholds[1:-1]):
        plt.axvline(x=thresh, color='r', linestyle='--')
        plt.text(thresh, plt.ylim()[1]*0.9, f'Stage {i+1}/{i+2}', 
                rotation=90, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'complexity_score_distribution.png'))
    
    # Save suggested curriculum stages to a text file
    with open(os.path.join(output_dir, 'suggested_curriculum.txt'), 'w') as f:
        f.write("=== Suggested Curriculum Stages ===\n\n")
        f.write(f"Based on the analysis of {len(dataset)} examples, here are recommended curriculum stages:\n\n")
        
        for i in range(len(percentiles)-1):
            lower = thresholds[i]
            upper = thresholds[i+1]
            count = sum(1 for score in complexity_scores if lower <= score <= upper)
            percent = (count / len(complexity_scores)) * 100
            
            f.write(f"Stage {i+1}:\n")
            f.write(f"  Complexity Range: {lower:.1f} - {upper:.1f}\n")
            f.write(f"  Examples: {count} ({percent:.1f}%)\n")
            
            # Suggest types of examples in this stage
            if i == 0:
                f.write("  Suggested Content: Single characters, simple short words with few diacritics\n")
            elif i == 1:
                f.write("  Suggested Content: Longer words, characters with more diacritics\n")
            elif i == 2:
                f.write("  Suggested Content: Multiple words, sentences, complex diacritic combinations\n")
            f.write("\n")
        
        f.write("These stages can be used with the curriculum learning trainer by setting:\n")
        f.write("--curriculum_stages=3 --curriculum_strategy=combined\n\n")
        f.write("For fixed stage epochs, consider:\n")
        f.write("--stage_epochs=5,5,5 (adjust based on your total training epochs)\n")
    
    print(f"\nAnalysis complete. Results saved to {output_dir}/")
    return {
        'complexity_scores': complexity_scores,
        'thresholds': thresholds,
        'char_lengths': char_lengths,
        'word_lengths': word_lengths,
        'sentence_lengths': sentence_lengths,
        'diacritic_counts': diacritic_counts,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze dataset complexity for curriculum learning')
    parser.add_argument('--dataset_name', type=str, default='vklinhhh/vietnamese_character_diacritic',
                        help='HuggingFace dataset name')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to analyze')
    parser.add_argument('--output_dir', type=str, default='dataset_analysis',
                        help='Directory to save analysis results')
    
    args = parser.parse_args()
    analyze_dataset_complexity(args.dataset_name, args.split, args.output_dir)