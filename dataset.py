import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import ast
import numpy as np  # Add import for numpy

class VietnameseCharacterDataset(Dataset):
    def __init__(self, hf_dataset, processor, base_char_vocab, diacritic_vocab):
        self.dataset = hf_dataset
        self.processor = processor
        self.base_char_vocab = base_char_vocab
        self.diacritic_vocab = diacritic_vocab

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get example from dataset
        example = self.dataset[idx]
        
        # Extract image
        image = example['image']

        # Handle PIL images
        if isinstance(image, Image.Image):
            # Convert PIL image to RGB to ensure 3 channels
            image = image.convert('RGB')
        elif isinstance(image, np.ndarray):
            # Handle numpy arrays
            if image.ndim == 2:
                # Add channel dimension
                image = np.expand_dims(image, axis=2)
                # Repeat to make RGB
                image = np.repeat(image, 3, axis=2)
            # Convert to PIL
            image = Image.fromarray(image.astype(np.uint8))
        
        # Parse string representations of lists if needed
        full_character = self._parse_if_string_list(example['full_character'])
        base_character = self._parse_if_string_list(example['base_character'])
        diacritic_type = self._parse_if_string_list(example['diacritic_type'])
        
        # Process image
        try:
            encoding = self.processor(image, return_tensors='pt')
            pixel_values = encoding.pixel_values.squeeze()
        except Exception as e:
            print(f"Error processing image at index {idx}: {e}")
            print(f"Image info: {type(image)}, shape: {getattr(image, 'shape', None) or getattr(image, 'size', None)}")
            # Create a small blank image as fallback
            dummy_image = Image.new('RGB', (32, 32), color='white')
            encoding = self.processor(dummy_image, return_tensors='pt')
            pixel_values = encoding.pixel_values.squeeze()

        # For the TrOCR model, we need the full word as a string
        if isinstance(full_character, list):
            word = ''.join(full_character)
        else:
            word = full_character
            
        labels = self.processor.tokenizer(word, return_tensors='pt').input_ids.squeeze()

        # Ensure we're working with lists for base characters and diacritics
        if not isinstance(base_character, list):
            base_characters = [base_character]
            diacritic_types = [diacritic_type]
        else:
            base_characters = base_character
            diacritic_types = diacritic_type

        # Convert base characters and diacritics to class indices
        # Add error handling to help debug missing characters
        base_char_indices = []
        for char in base_characters:
            try:
                index = self.base_char_vocab.index(char)
                base_char_indices.append(index)
            except ValueError:
                print(f"Warning: Character '{char}' not found in base_char_vocab. Defaulting to 0.")
                print(f"Available vocab: {self.base_char_vocab}")
                base_char_indices.append(0)  # Default to first character as fallback
        
        diacritic_indices = []
        for dia in diacritic_types:
            try:
                index = self.diacritic_vocab.index(dia)
                diacritic_indices.append(index)
            except ValueError:
                print(f"Warning: Diacritic '{dia}' not found in diacritic_vocab. Defaulting to 0.")
                print(f"Available vocab: {self.diacritic_vocab}")
                diacritic_indices.append(0)  # Default to first diacritic as fallback
        base_char_indices = [
            min(max(0, idx), len(self.base_char_vocab) - 1) for idx in base_char_indices
        ]
        diacritic_indices = [
            min(max(0, idx), len(self.diacritic_vocab) - 1) for idx in diacritic_indices
        ]
        return {
            'pixel_values': pixel_values,
            'labels': labels,
            'base_character_indices': torch.tensor(base_char_indices),
            'diacritic_indices': torch.tensor(diacritic_indices),
            'full_characters': full_character if isinstance(full_character, list) else [full_character],
            'word': word
        }
    
    def _parse_if_string_list(self, value):
        """Parse string representations of lists into actual lists."""
        if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            try:
                # Safely parse the string into a Python object
                return ast.literal_eval(value)
            except (SyntaxError, ValueError):
                # If parsing fails, return the original string
                return value
        return value