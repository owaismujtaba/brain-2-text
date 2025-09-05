import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import pdb
class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer, self.scheduler = model.configure_optimizers(config)
        
        # Loss function
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)  # Using 0 as the blank token
        # Training parameters
        self.num_epochs = config.get('training', {}).get('num_epochs', 100)
        self.eval_interval = config.get('training', {}).get('eval_interval', 1)
        
        # Best model tracking
        self.best_val_loss = float('inf')
        
    def train(self, train_loader, val_loader):
        """Main training loop"""
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = self._train_epoch(train_loader)
            
            # Validation phase
            if (epoch + 1) % self.eval_interval == 0:
                self.model.eval()
                val_loss = self._validate(val_loader)
                
                # Update learning rate scheduler
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, val_loss)
                
                print(f'Epoch {epoch+1}/{self.num_epochs}:')
                print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    def _train_epoch(self, train_loader):
        """Train for one epoch"""
        epoch_loss = 0
        num_batches = len(train_loader)
        
        for batch in tqdm(train_loader, desc='Training'):
            self.optimizer.zero_grad()
            
            # Get batch data
            inputs, seq_class_ids, seq_lengths, phenome_seq_lengths = self._prepare_batch(batch)
            
            # Forward pass
            logits = self.model(inputs, seq_lengths)
            # Compute loss
            loss = self.criterion(
                log_probs = torch.permute(logits.log_softmax(2), [1, 0, 2]),
                targets = seq_class_ids,
                input_lengths = seq_lengths,
                target_lengths = phenome_seq_lengths
            )           
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update weights
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss / num_batches
    
    def _validate(self, val_loader):
        """Validate the model"""
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                # Get batch data
                inputs, seq_class_ids, seq_lengths, phenome_seq_lengths = self._prepare_batch(batch)
                
                # Forward pass
                logits = self.model(inputs, seq_lengths)
                import pdb;
                pdb.set_trace()
                # Compute loss
                loss = self.criterion(
                    log_probs = torch.permute(logits.log_softmax(2), [1, 0, 2]),
                    targets = seq_class_ids,
                    input_lengths = seq_lengths,
                    target_lengths = phenome_seq_lengths
                )
                
                
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def _prepare_batch(self, batch):
        """Prepare batch data for training/validation"""

        inputs = batch['neural_features']
        seq_class_ids = batch['seq_class_ids']
        seq_lengths = batch['seq_lengths']
        phenome_seq_lengths = torch.tensor(batch['seq_len'])
        inputs = inputs.to(self.device)
        seq_class_ids = seq_class_ids.to(self.device)
        seq_lengths = seq_lengths.to(self.device)
        phenome_seq_lengths = phenome_seq_lengths.to(self.device)

        return inputs, seq_class_ids, seq_lengths, phenome_seq_lengths
    
    def _save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        checkpoint_path = self.config.get('training', {}).get(
            'checkpoint_path', 'checkpoint.pt'
        )
        torch.save(checkpoint, checkpoint_path)