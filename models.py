from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from torch import nn
import os
import json

class VietnameseOCRModel(nn.Module):
    def __init__(self, base_model_name='microsoft/trocr-base-handwritten', base_char_vocab=None, diacritic_vocab=None):
        super().__init__()
        # Load the base TrOCR model
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained(base_model_name)
        self.processor = TrOCRProcessor.from_pretrained(base_model_name)
        
        # Store the config from the base model
        self.config = self.trocr_model.config
        
        # Store vocabularies
        self.base_char_vocab = base_char_vocab
        self.diacritic_vocab = diacritic_vocab
        
        # Define sequence classifiers - rename to match used in save_pretrained
        hidden_size = self.trocr_model.decoder.config.hidden_size
        self.base_char_head = nn.Linear(hidden_size, len(self.base_char_vocab))
        self.diacritic_head = nn.Linear(hidden_size, len(self.diacritic_vocab))
        
    def forward(self, pixel_values, labels=None, decoder_input_ids=None):
        # Forward through TrOCR
        outputs = self.trocr_model(
            pixel_values=pixel_values,
            labels=labels,
            decoder_input_ids=decoder_input_ids if labels is None else None,
            output_hidden_states=True
        )
        
        # Get the decoder hidden states for all positions (for character-level tasks)
        decoder_hidden_states = outputs.decoder_hidden_states[-1]
        
        # Apply classifiers to all positions - use the renamed attributes
        base_char_logits = self.base_char_head(decoder_hidden_states)
        diacritic_logits = self.diacritic_head(decoder_hidden_states)
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
            'base_char_logits': base_char_logits,
            'diacritic_logits': diacritic_logits,
            'decoder_hidden_states': outputs.decoder_hidden_states
        }
        
    def save_pretrained(self, save_directory):
        """Custom method to save the model components"""
        os.makedirs(save_directory, exist_ok=True)

        # Save the base TrOCR model
        self.trocr_model.save_pretrained(os.path.join(save_directory, 'trocr_base'))

        # Save the processor
        self.processor.save_pretrained(save_directory)

        # Save the custom heads
        torch.save(
            self.base_char_head.state_dict(), os.path.join(save_directory, 'base_char_head.pt')
        )
        torch.save(
            self.diacritic_head.state_dict(), os.path.join(save_directory, 'diacritic_head.pt')
        )

        # Save vocabularies
        with open(os.path.join(save_directory, 'base_char_vocab.json'), 'w') as f:
            json.dump(self.base_char_vocab, f)

        with open(os.path.join(save_directory, 'diacritic_vocab.json'), 'w') as f:
            json.dump(self.diacritic_vocab, f)

        # Save model config
        self.config.to_json_file(os.path.join(save_directory, 'config.json'))

        return save_directory

    @classmethod
    def from_pretrained(cls, model_path):
        """Load a saved model"""
        # Ensure model_path is a valid path
        if not os.path.exists(model_path):
            raise ValueError(f"Model path '{model_path}' does not exist")
            
        # Check if this is a trained model directory with our expected files
        if not os.path.exists(os.path.join(model_path, 'base_char_vocab.json')):
            raise ValueError(f"Model path '{model_path}' is not a valid Vietnamese OCR model directory")
            
        # Load TrOCR model from the saved subdirectory
        trocr_model_path = os.path.join(model_path, 'trocr_base')
        if not os.path.exists(trocr_model_path):
            raise ValueError(f"TrOCR base model not found at '{trocr_model_path}'")
            
        # Load TrOCR model explicitly without relying on HuggingFace's auto-download
        trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_model_path)

        # Load processor
        processor = TrOCRProcessor.from_pretrained(model_path)

        # Load vocabularies
        with open(os.path.join(model_path, 'base_char_vocab.json'), 'r') as f:
            base_char_vocab = json.load(f)

        with open(os.path.join(model_path, 'diacritic_vocab.json'), 'r') as f:
            diacritic_vocab = json.load(f)

        # Create a new instance with empty initialization - we'll set components manually
        model = cls.__new__(cls)  # Create uninitialized instance
        nn.Module.__init__(model)  # Initialize the nn.Module parent
        
        # Set model components directly
        model.trocr_model = trocr_model
        model.processor = processor
        model.config = trocr_model.config
        model.base_char_vocab = base_char_vocab
        model.diacritic_vocab = diacritic_vocab

        # Initialize heads
        hidden_size = trocr_model.decoder.config.hidden_size
        model.base_char_head = nn.Linear(hidden_size, len(base_char_vocab))
        model.diacritic_head = nn.Linear(hidden_size, len(diacritic_vocab))

        # Load head weights
        model.base_char_head.load_state_dict(
            torch.load(os.path.join(model_path, 'base_char_head.pt'))
        )
        model.diacritic_head.load_state_dict(
            torch.load(os.path.join(model_path, 'diacritic_head.pt'))
        )

        return model