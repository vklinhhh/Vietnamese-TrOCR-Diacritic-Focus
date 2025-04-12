from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm.auto import tqdm
from utils import custom_collate_fn, WandbImageLogger, log_model_architecture
from dataset import VietnameseCharacterDataset
from models import VietnameseOCRModel
import torch
import torch.nn as nn
from transformers import TrOCRProcessor
import wandb
import gc
import os

def train_vietnamese_ocr(model, train_dataset, val_dataset, epochs=5, batch_size=8, lr=5e-6, device=None, processor=None, 
                         project_name="vietnamese-trocr-diacritics-branch", run_name=None, log_interval=10):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Initialize wandb
    wandb.init(project=project_name, name=run_name, config={
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "model_name": type(model).__name__,
        "base_model": "microsoft/trocr-base-handwritten",
        "device": str(device),
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
    })
    
    # Initialize image logger and log model architecture
    base_char_vocab = train_dataset.base_char_vocab
    diacritic_vocab = train_dataset.diacritic_vocab
    image_logger = WandbImageLogger(processor, base_char_vocab, diacritic_vocab, log_interval=log_interval)
    log_model_architecture(model)
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

    # Initialize optimizer - use PyTorch's AdamW to avoid deprecation warning
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Define loss functions
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='mean')

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_text_loss = 0
        train_base_char_loss = 0
        train_diacritic_loss = 0
        step = 0

        for batch in tqdm(train_loader):
            step += 1
            tokenizer = processor.tokenizer
            model.config.decoder_start_token_id = tokenizer.bos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
            model.config.eos_token_id = tokenizer.eos_token_id
            
            # Move batch to device
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            base_char_labels = batch['base_character_indices'].to(device)  # Updated field name
            diacritic_labels = batch['diacritic_indices'].to(device)       # Updated field name

            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)

            # Calculate losses
            text_loss = outputs['loss']  # TrOCR's built-in loss
            
            # FIX: Calculate the minimum sequence length between labels and logits
            seq_length = min(
                base_char_labels.size(1),
                outputs['base_char_logits'].size(1)
            )
            
            base_char_loss = torch.zeros(1, device=device)
            diacritic_loss = torch.zeros(1, device=device)
            
            for pos in range(seq_length):
                base_char_loss += ce_loss(outputs['base_char_logits'][:, pos, :], base_char_labels[:, pos])
                diacritic_loss += ce_loss(outputs['diacritic_logits'][:, pos, :], diacritic_labels[:, pos])

            base_char_loss /= seq_length
            diacritic_loss /= seq_length
            
            # Combine losses
            total_loss = text_loss + base_char_loss + diacritic_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate losses
            train_loss += total_loss.item()
            train_text_loss += text_loss.item()
            train_base_char_loss += base_char_loss.item()
            train_diacritic_loss += diacritic_loss.item()
            
            # Occasionally log train image predictions - with lower frequency than validation
            if step % (log_interval * 5) == 0:
                image_logger.log_predictions(batch, outputs, phase="train")
            
            # Log training metrics at intervals
            if step % log_interval == 0:
                wandb.log({
                    "train/step": epoch * len(train_loader) + step,
                    "train/batch_loss": total_loss.item(),
                    "train/batch_text_loss": text_loss.item(),
                    "train/batch_base_char_loss": base_char_loss.item(),
                    "train/batch_diacritic_loss": diacritic_loss.item(),
                })

        # Average training losses for the epoch
        avg_train_loss = train_loss / len(train_loader)
        avg_train_text_loss = train_text_loss / len(train_loader)
        avg_train_base_char_loss = train_base_char_loss / len(train_loader)
        avg_train_diacritic_loss = train_diacritic_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_text_loss = 0
        val_base_char_loss = 0
        val_diacritic_loss = 0
        text_acc, base_char_acc, diacritic_acc, word_acc, char_acc = 0, 0, 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                tokenizer = processor.tokenizer
                model.config.decoder_start_token_id = tokenizer.bos_token_id
                model.config.pad_token_id = tokenizer.pad_token_id
                model.config.eos_token_id = tokenizer.eos_token_id
                
                # Move batch to device
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                base_char_labels = batch['base_character_indices'].to(device)  # Updated field name
                diacritic_labels = batch['diacritic_indices'].to(device)       # Updated field name

                # Forward pass
                outputs = model(pixel_values=pixel_values, labels=labels)

                # Calculate losses
                text_loss = outputs['loss']  # TrOCR's built-in loss
                
                # FIX: Calculate the minimum sequence length between labels and logits
                seq_length = min(
                    base_char_labels.size(1),
                    outputs['base_char_logits'].size(1)
                )
                
                base_char_loss = torch.zeros(1, device=device)
                diacritic_loss = torch.zeros(1, device=device)
                
                for pos in range(seq_length):
                    base_char_loss += ce_loss(outputs['base_char_logits'][:, pos, :], base_char_labels[:, pos])
                    diacritic_loss += ce_loss(outputs['diacritic_logits'][:, pos, :], diacritic_labels[:, pos])

                base_char_loss /= seq_length
                diacritic_loss /= seq_length
                
                # Combine losses
                total_loss = text_loss + base_char_loss + diacritic_loss
                
                # Accumulate validation losses
                val_loss += total_loss.item()
                val_text_loss += text_loss.item()
                val_base_char_loss += base_char_loss.item()
                val_diacritic_loss += diacritic_loss.item()

                # Calculate accuracies
                # Text accuracy
                predicted_ids = outputs['logits'].argmax(-1)
                word_correct = 0
                char_correct = 0
                char_total = 0
                
                for i, (pred, label) in enumerate(zip(predicted_ids, labels)):
                    valid_indices = label != -100
                    if valid_indices.any():
                        valid_pred = pred[valid_indices]
                        valid_label = label[valid_indices]
                        word_correct += torch.all(valid_pred == valid_label).item()
                        char_correct += (valid_pred == valid_label).sum().item()
                        char_total += valid_indices.sum().item()
                
                word_acc = word_correct / len(labels)
                char_acc = char_correct / char_total if char_total > 0 else 0
                
                # Base character accuracy - use the same seq_length as loss calculation
                base_char_preds = outputs['base_char_logits'].argmax(-1)
                base_char_matches = 0
                base_char_total = 0
                
                for pos in range(seq_length):
                    matches = (base_char_preds[:, pos] == base_char_labels[:, pos]).sum().item()
                    base_char_matches += matches
                    base_char_total += base_char_labels.size(0)  # batch size
                
                base_char_acc += base_char_matches / base_char_total if base_char_total > 0 else 0

                # Diacritic accuracy - use the same seq_length as loss calculation
                diacritic_preds = outputs['diacritic_logits'].argmax(-1)
                diacritic_matches = 0
                diacritic_total = 0
                
                for pos in range(seq_length):
                    matches = (diacritic_preds[:, pos] == diacritic_labels[:, pos]).sum().item()
                    diacritic_matches += matches
                    diacritic_total += diacritic_labels.size(0)  # batch size
                
                diacritic_acc += diacritic_matches / diacritic_total if diacritic_total > 0 else 0
                
                # Log image predictions using the image logger
                image_logger.log_predictions(batch, outputs, phase="val")

            # Average the metrics
            avg_val_loss = val_loss / len(val_loader)
            avg_val_text_loss = val_text_loss / len(val_loader)
            avg_val_base_char_loss = val_base_char_loss / len(val_loader)
            avg_val_diacritic_loss = val_diacritic_loss / len(val_loader)
            avg_base_char_acc = base_char_acc / len(val_loader)
            avg_diacritic_acc = diacritic_acc / len(val_loader)

        # Log epoch metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "train/text_loss": avg_train_text_loss,
            "train/base_char_loss": avg_train_base_char_loss,
            "train/diacritic_loss": avg_train_diacritic_loss,
            "val/loss": avg_val_loss,
            "val/text_loss": avg_val_text_loss,
            "val/base_char_loss": avg_val_base_char_loss,
            "val/diacritic_loss": avg_val_diacritic_loss,
            "val/word_accuracy": word_acc,
            "val/char_accuracy": char_acc,
            "val/base_char_accuracy": avg_base_char_acc,
            "val/diacritic_accuracy": avg_diacritic_acc,
        })

        # Print epoch results
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(
            f'Word Acc: {word_acc:.4f}, Char Acc: {char_acc:.4f}, Base Char Acc: {avg_base_char_acc:.4f}, Diacritic Acc: {avg_diacritic_acc:.4f}'
        )
        
        # Optional: Save model checkpoint
        if (epoch+1) % 5 == 0 or (epoch == epochs - 1):
            import os
            checkpoints_dir = "checkpoints"
            os.makedirs(checkpoints_dir, exist_ok=True)

            model_checkpoint_path = f"{checkpoints_dir}/checkpoint-epoch-{epoch+1}"
            processor.save_pretrained(checkpoints_dir)
            model.save_pretrained(model_checkpoint_path)
            wandb.save(f"{model_checkpoint_path}/*")
        
        gc.collect()
        torch.cuda.empty_cache()

    # Finish wandb run
    # Make sure all files are saved to wandb
    wandb.save("*.py")  # Track all Python scripts
    wandb.save("*.json")  # Track all JSON files
    wandb.save("*.log")  # Track log files
    wandb.finish()
    
    return model