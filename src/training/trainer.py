import os
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from src.evaluation.eval import compute_phenome_error

import pdb
class Trainer:
    def __init__(self, model, config, logger=None):
        self.model = model
        self.config = config
        self.logger = logger
        self.logger.info("Initializing Trainer")
        self._device_setup()
        self.optimizer, self.scheduler = model.configure_optimizers(config)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)  # Using 0 as the blank token
        self.num_epochs = config.get('training', {}).get('num_epochs', 100)
        self.eval_interval = config.get('training', {}).get('eval_interval', 10)
        self.best_per = float('inf')
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.wisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(self.device)

    def _device_setup(self):
        """Setup device and move model to device"""
        self.logger.info("Setting up device")
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            if torch.cuda.device_count() > 1:
                self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
                self.model = nn.DataParallel(self.model)
        else:
            self.device = torch.device('cpu')
            self.logger.info(f"Using device: {self.device}")
        self.model.to(self.device)

        
        
        
    def train(self, train_loader, val_loader):
        """Main training loop"""
        self.logger.info("Starting training process")
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = self._train_epoch(train_loader)
            
            # Validation phase
            if (epoch + 1) % self.eval_interval == 0:
                self.model.eval()
                self.logger.info("Validating model...")
                 # Compute validation loss
                val_loss, per = self._validate(val_loader)
                 # Update learning rate scheduler
                self.scheduler.step(val_loss)
                
                # Save best model
                if per < self.best_per:
                    self.best_per = per 
                    self._save_checkpoint(epoch, val_loss, per)
                
                self.logger.info(f'Epoch {epoch+1}/{self.num_epochs}:')
                self.logger.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, PER: {per:.4f}')
    
    def _train_epoch(self, train_loader):
        """Train for one epoch"""
        epoch_loss = 0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            self.optimizer.zero_grad()
           
            # Get batch data
            inputs, seq_lengths, sentence_labels, sentence_len = self._prepare_batch(batch)
            pdb.set_trace()
            # Forward pass
            logits = self.model(inputs)
            # Compute loss
            loss = self.criterion(
                log_probs = torch.permute(logits.log_softmax(2), [1, 0, 2]),
                targets = sentence_labels,
                input_lengths = seq_lengths,
                target_lengths = sentence_len
            )           
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update weights
            self.optimizer.step()
            
            current_loss = loss.item()
            epoch_loss += current_loss
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})
            
        return epoch_loss / num_batches
    
    def _validate(self, val_loader):
        """Validate the model"""
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            total_per = 0
            per = 0
            pbar = tqdm(val_loader, desc='Training')
            for batch in val_loader:
                # Get batch data
                inputs, seq_lengths, sentence_labels, sentence_len = self._prepare_batch(batch)
                pdb.set_trace()
                # Forward pass
                logits = self.model(inputs)
                
                # Compute loss
                loss = self.criterion(
                    log_probs = torch.permute(logits.log_softmax(2), [1, 0, 2]),
                    targets = sentence_labels,
                    input_lengths = seq_lengths,
                    target_lengths = sentence_len
                )
                
                
                total_loss += loss.item()
                #per = compute_phenome_error(logits, seq_class_ids, seq_lengths, phenome_seq_lengths)
                total_per += per
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'PER': f'{per:.4f}'})
        return total_loss / num_batches, total_per / num_batches
    
    def _prepare_batch(self, batch):
        """Prepare batch data for training/validation"""

        inputs = batch['neural_features']
        seq_lengths = batch['seq_lengths']
        sentence_labels = batch['sentence_label']
        sentence_len = batch['sentence_len']

        inputs = inputs.to(self.device)
        seq_lengths = seq_lengths.to(self.device)
        
        sentence_labels = torch.tensor(sentence_labels).to(self.device)
        return inputs, seq_lengths, sentence_labels, sentence_len

    def _save_checkpoint(self, epoch, val_loss, per):
        """Save model checkpoint"""
        self.logger.info(f"Saving checkpoint for epoch {epoch+1} with val_loss {val_loss:.4f}; PER: {per:.4f}")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        checkpoint_path = self.config.get('training', {}).get('checkpoints_dir')
        os.makedirs(checkpoint_path, exist_ok=True)
        
        filename = f'checkpoint_epoch-{epoch+1:03d}_loss-{val_loss:.4f}_per-{per}.pth'
        checkpoint_path = os.path.join(checkpoint_path, filename)
        
        torch.save(checkpoint, checkpoint_path)
        
        best_model_path = os.path.join(os.path.dirname(checkpoint_path), 'best_model.pth')
        torch.save(checkpoint, best_model_path)

    def _get_wisper_sentence_labels(self, sentence_labels):
        """Convert sentence labels to Whisper-compatible input IDs"""
        sentence_ids = []
        with torch.no_grad():
            for sentence in sentence_labels:
                input_ids = self.processor.tokenizer(sentence, return_tensors="pt", padding=True).input_ids
                sentence_ids.append(input_ids)
        return sentence_ids

        