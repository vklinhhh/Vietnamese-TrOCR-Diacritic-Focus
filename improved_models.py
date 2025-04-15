from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from torch import nn
import os
import json

class AttentionModule(nn.Module):
    """Self-attention module to focus on relevant features in the hidden states"""
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = torch.sqrt(torch.tensor(hidden_size, dtype=torch.float32))
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, _ = x.shape
        
        # Calculate query, key, value projections
        q = self.query(x)  # [batch_size, seq_len, hidden_size]
        k = self.key(x)    # [batch_size, seq_len, hidden_size]
        v = self.value(x)  # [batch_size, seq_len, hidden_size]
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch_size, seq_len, seq_len]
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)  # [batch_size, seq_len, hidden_size]
        
        # Residual connection
        output = x + context
        
        return output


class ImprovedCharacterHead(nn.Module):
    """Improved head for base character prediction with attention and deeper layers"""
    def __init__(self, hidden_size, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.attention = AttentionModule(hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        
        # Deeper network with residual connections
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_norm3 = nn.LayerNorm(hidden_size // 2)
        
        # Final prediction layer
        self.classifier = nn.Linear(hidden_size // 2, vocab_size)
        
    def forward(self, x):
        # Apply attention
        attended = self.attention(x)
        attended = self.layer_norm1(attended)
        
        # First dense layer with residual connection
        residual = attended
        x = self.fc1(attended)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.layer_norm2(x + residual)  # Add residual and normalize
        
        # Second dense layer
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.layer_norm3(x)
        
        # Final classification
        logits = self.classifier(x)
        
        return logits


class DiacriticAwareHead(nn.Module):
    """
    Diacritic prediction head that leverages base character information
    """
    def __init__(self, hidden_size, base_char_size, diacritic_vocab_size, dropout_rate=0.1):
        super().__init__()
        # Attention module
        self.attention = AttentionModule(hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        
        # Process base character logits
        self.base_char_projection = nn.Linear(base_char_size, hidden_size // 2)
        
        # Combine hidden states with base character information
        self.combined_layer = nn.Linear(hidden_size + hidden_size // 2, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Second dense layer
        self.fc = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_norm3 = nn.LayerNorm(hidden_size // 2)
        
        # Final prediction layer
        self.classifier = nn.Linear(hidden_size // 2, diacritic_vocab_size)
        
    def forward(self, hidden_states, base_char_logits):
        # Apply attention to hidden states
        attended = self.attention(hidden_states)
        attended = self.layer_norm1(attended)
        
        # Process base character logits
        base_char_features = self.base_char_projection(base_char_logits)
        
        # Combine hidden states with base character information
        combined = torch.cat([attended, base_char_features], dim=-1)
        x = self.combined_layer(combined)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.layer_norm2(x)
        
        # Second dense layer
        x = self.fc(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.layer_norm3(x)
        
        # Final classification
        logits = self.classifier(x)
        
        return logits


class ImprovedVietnameseOCRModel(nn.Module):
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
        
        # Define improved character head
        hidden_size = self.trocr_model.decoder.config.hidden_size
        self.base_char_head = ImprovedCharacterHead(
            hidden_size=hidden_size,
            vocab_size=len(self.base_char_vocab),
            dropout_rate=0.1
        )
        
        # Define diacritic-aware head that uses base character predictions
        self.diacritic_head = DiacriticAwareHead(
            hidden_size=hidden_size,
            base_char_size=len(self.base_char_vocab),
            diacritic_vocab_size=len(self.diacritic_vocab),
            dropout_rate=0.1
        )
        
        # Additional feature: Context aggregation for sequence-level understanding
        self.context_aggregation = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.context_projection = nn.Linear(hidden_size * 2, hidden_size)
        
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
        
        # Apply context aggregation for better sequence understanding
        context_output, _ = self.context_aggregation(decoder_hidden_states)
        context_output = self.context_projection(context_output)
        
        # Combine original hidden states with contextual information
        enhanced_hidden_states = decoder_hidden_states + context_output
        
        # Apply base character head
        base_char_logits = self.base_char_head(enhanced_hidden_states)
        
        # Apply diacritic head with awareness of base character predictions
        diacritic_logits = self.diacritic_head(enhanced_hidden_states, base_char_logits)
        
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

        # Save custom modules directly - we'll use torch.save for the whole module
        torch.save(
            self.base_char_head.state_dict(), os.path.join(save_directory, 'base_char_head.pt')
        )
        torch.save(
            self.diacritic_head.state_dict(), os.path.join(save_directory, 'diacritic_head.pt')
        )
        torch.save(
            self.context_aggregation.state_dict(), os.path.join(save_directory, 'context_aggregation.pt')
        )
        torch.save(
            self.context_projection.state_dict(), os.path.join(save_directory, 'context_projection.pt')
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

        # Create a new instance
        # model = cls(
        #     base_model_name=None,  # We'll set the trocr_model directly
        #     base_char_vocab=base_char_vocab,
        #     diacritic_vocab=diacritic_vocab
        # )
        model = cls.__new__(cls)
        nn.Module.__init__(model)  # Initialize the nn.Module parent

        # Set attributes directly
        model.base_char_vocab = base_char_vocab
        model.diacritic_vocab = diacritic_vocab
        model.trocr_model = trocr_model  # Use the already loaded trocr_model
        model.processor = processor
        model.config = trocr_model.config
        
        # Setup the heads with proper sizes
        hidden_size = trocr_model.decoder.config.hidden_size
        
        # Initialize heads
        model.base_char_head = ImprovedCharacterHead(
            hidden_size=hidden_size,
            vocab_size=len(base_char_vocab),
            dropout_rate=0.1
        )
        
        model.diacritic_head = DiacriticAwareHead(
            hidden_size=hidden_size,
            base_char_size=len(base_char_vocab),
            diacritic_vocab_size=len(diacritic_vocab),
            dropout_rate=0.1
        )
        
        model.context_aggregation = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        model.context_projection = nn.Linear(hidden_size * 2, hidden_size)

        # Load saved weights
        model.base_char_head.load_state_dict(
            torch.load(os.path.join(model_path, 'base_char_head.pt'))
        )
        model.diacritic_head.load_state_dict(
            torch.load(os.path.join(model_path, 'diacritic_head.pt'))
        )
        model.context_aggregation.load_state_dict(
            torch.load(os.path.join(model_path, 'context_aggregation.pt'))
        )
        model.context_projection.load_state_dict(
            torch.load(os.path.join(model_path, 'context_projection.pt'))
        )

        return model