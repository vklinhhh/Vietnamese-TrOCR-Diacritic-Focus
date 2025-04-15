import torch
from transformers import TrOCRProcessor
from torch.utils.data import DataLoader
from improved_models import ImprovedVietnameseOCRModel
from dataset import VietnameseCharacterDataset
from improved_curriculum_trainer import train_improved_vietnamese_ocr_with_curriculum
from datasets import load_dataset
import os
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Vietnamese OCR model with improved architecture and curriculum learning')
    
    # Dataset and model parameters
    parser.add_argument('--dataset_name', type=str, default='vklinhhh/vietnamese_character_diacritic_cwl',
                        help='HuggingFace dataset name')
    parser.add_argument('--output_dir', type=str, default='vietnamese-trocr-diacritics-improved',
                        help='Directory to save the model')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=15,
                        help='Total number of training epochs across all stages')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-6,
                        help='Learning rate')
    parser.add_argument('--train_test_split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    
    # Curriculum learning parameters
    parser.add_argument('--curriculum_strategy', type=str, default='combined',
                        choices=['length', 'complexity', 'combined'],
                        help='Strategy for curriculum difficulty progression')
    parser.add_argument('--curriculum_stages', type=int, default=3,
                        help='Number of curriculum stages')
    parser.add_argument('--stage_epochs', type=str, default=None,
                        help='Comma-separated list of epochs per stage (e.g., "5,5,5"). If not provided, '
                             'automatic stage progression based on plateaus will be used.')
    parser.add_argument('--stage_patience', type=int, default=3,
                        help='Epochs to wait before advancing curriculum stage (when using automatic progression)')
    
    # Training management
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                        help='Epochs to wait before early stopping')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval in steps')
    
    # Focal loss parameter
    parser.add_argument('--focal_loss_gamma', type=float, default=2.0,
                        help='Gamma parameter for focal loss (higher = more focus on hard examples)')
    
    # Weights & Biases logging
    parser.add_argument('--wandb_project', type=str, default='vietnamese-trocr-improved-curriculum',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Weights & Biases run name')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset from HuggingFace
    print(f"Loading dataset {args.dataset_name}...")
    hf_dataset = load_dataset(args.dataset_name)
    
    # Split the dataset if it doesn't already have train/val splits
    if 'validation' not in hf_dataset:
        hf_dataset = hf_dataset['train'].train_test_split(
            test_size=args.train_test_split, 
            seed=42,
        )
        train_dataset = hf_dataset['train']
        val_dataset = hf_dataset['test']
    else:
        train_dataset = hf_dataset['train']
        val_dataset = hf_dataset['validation']
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Extract unique values for vocabularies
    # Helper function to process strings or lists properly
    def process_item_safely(item):
        if isinstance(item, list):
            return item
        elif isinstance(item, str):
            # Check if it's a string representation of a list
            if item.startswith('[') and item.endswith(']'):
                try:
                    import ast
                    return ast.literal_eval(item)
                except:
                    return [item]
            else:
                return [item]
        else:
            return [str(item)]
    
    # Extract all unique characters
    all_base_chars = set()
    all_diacritics = set()
    
    for example in train_dataset:
        base_chars = process_item_safely(example['base_character'])
        diacritics = process_item_safely(example['diacritic_type'])
        
        # Add each character to the set
        for char in base_chars:
            all_base_chars.add(char)
        
        for dia in diacritics:
            all_diacritics.add(dia)
    
    # Create vocabularies for base characters and diacritics
    base_char_vocab = sorted(list(all_base_chars))
    diacritic_vocab = sorted(list(all_diacritics))
    
    print(f"Base character vocabulary size: {len(base_char_vocab)}")
    print(f"Base character vocab: {base_char_vocab}")
    print(f"Diacritic vocabulary size: {len(diacritic_vocab)}")
    print(f"Diacritic vocab: {diacritic_vocab}")

    # Initialize processor and improved model
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = ImprovedVietnameseOCRModel(
        base_model_name='microsoft/trocr-base-handwritten',
        base_char_vocab=base_char_vocab,
        diacritic_vocab=diacritic_vocab
    )
    model.to(device)

    # Create datasets
    train_dataset_wrapper = VietnameseCharacterDataset(
        train_dataset,
        processor=processor,
        base_char_vocab=base_char_vocab,
        diacritic_vocab=diacritic_vocab,
    )

    val_dataset_wrapper = VietnameseCharacterDataset(
        val_dataset,
        processor=processor,
        base_char_vocab=base_char_vocab,
        diacritic_vocab=diacritic_vocab,
    )

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse stage_epochs if provided
    stage_epochs = None
    if args.stage_epochs:
        try:
            stage_epochs = [int(e) for e in args.stage_epochs.split(',')]
            assert len(stage_epochs) == args.curriculum_stages, "Number of stages must match curriculum_stages"
        except Exception as e:
            print(f"Error parsing stage_epochs: {e}")
            print("Falling back to automatic stage progression")

    # Train the model with improved trainer
    print("Using improved curriculum learning for training")
    trained_model = train_improved_vietnamese_ocr_with_curriculum(
        model=model,
        train_dataset=train_dataset_wrapper,
        val_dataset=val_dataset_wrapper,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        device=device,
        processor=processor,
        project_name=args.wandb_project,
        run_name=args.wandb_run_name,
        log_interval=args.log_interval,
        curriculum_strategy=args.curriculum_strategy,
        curriculum_stages=args.curriculum_stages,
        stage_epochs=stage_epochs,
        patience=args.stage_patience,
        early_stopping_patience=args.early_stopping_patience,
        focal_loss_gamma=args.focal_loss_gamma
    )

    # Save the trained model
    trained_model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == '__main__':
    main()