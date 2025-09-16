import os
import torch
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path

from src.evaluation.eval import compute_phenome_error
from src.utils import log_info
from src.utils import log_info

import pdb
class Trainer:
    def __init__(self, model, config, logger=None):
        self.model = model
        self.config = config
        self.logger = logger
        log_info(logger, "Initializing Trainer")
        self._setup_config()
        self._device_setup()
        self.configure_optimizers()

        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True) 
        self.best_loss = float('inf')
        self.best_per = float('inf')

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

    def _setup_config(self):
        self.num_epochs = self.config.get('training', {}).get('num_epochs', 100)
        self.lr = self.config.get('training', {}).get('learning_rate', 1e-3)
        self.weight_decay = self.config.get('training', {}).get('weight_decay', 1e-5)
        self.checkpoint_dir = self.config.get('training', {}).get('checkpoints_dir')
        self.output_dir = self.config.get('training', {}).get('output_dir')
        self.load_from_checkpoint = self.config.get('training', {}).get('load_checkpoint')
        self.checkpoints_save_interval = self.config.get('training', {}).get('checkpoints_save_interval')
        self.checkpoints_save_interval = self.config.get('training', {}).get('checkpoints_save_interval')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
        
        
    def train(self, train_loader, val_loader):
        """Main training loop"""
        if self.load_from_checkpoint:
            self.load_model_checkpoint()
        self.logger.info("Starting training process")
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = self._train_epoch(train_loader, epoch)         
            
            self.logger.info(f"Validating model after Epoch:{epoch}")
            self.model.eval()
            val_loss, per = self._validate(val_loader)
            self.scheduler.step(val_loss)
            
            # Save best model
            if per < self.best_per:
                self.best_per = per 
                self._save_checkpoint(epoch, val_loss, per, best=True)
            if epoch+1%self.checkpoints_save_interval == 0:
                self._save_checkpoint(epoch, val_loss, per, best=False)
                self._save_checkpoint(epoch, val_loss, per, best=True)
            if epoch+1%self.checkpoints_save_interval == 0:
                self._save_checkpoint(epoch, val_loss, per, best=False)
                
            self.logger.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, PER: {per:.4f}')
    
    def _train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        epoch_loss = 0.0
        num_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} - Training", unit="batch")
        for batch in pbar:
            self.optimizer.zero_grad()

            # Get batch data
            inputs, seq_class_ids, seq_lengths, phenome_seq_lengths = self._prepare_batch(batch)
            
            
            # Forward pass
            logits = self.model(inputs)

            # Compute loss
            loss = self.criterion(
                log_probs=torch.permute(logits.log_softmax(2), [1, 0, 2]),
                targets=seq_class_ids,
                input_lengths=seq_lengths,
                target_lengths=phenome_seq_lengths
            )

            # Backward pass
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Update weights
            self.optimizer.step()

            current_loss = loss.item()
            epoch_loss += current_loss

            # Update progress bar with current + average loss
            avg_loss = epoch_loss / (pbar.n + 1)
            pbar.set_postfix({"batch_loss": f"{current_loss:.4f}", "avg_loss": f"{avg_loss:.4f}"})

        return epoch_loss / num_batches

    
    def _validate(self, val_loader):
        """Validate the model"""
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            total_per = 0
            per = 0
            pbar = tqdm(val_loader, desc='Validation')
            for batch in pbar:
                # Get batch data
                inputs, seq_class_ids, seq_lengths, phenome_seq_lengths = self._prepare_batch(batch)
                
                # Forward pass
                logits = self.model(inputs)
                
                # Compute loss
                loss = self.criterion(
                    log_probs = torch.permute(logits.log_softmax(2), [1, 0, 2]),
                    targets = seq_class_ids,
                    input_lengths = seq_lengths,
                    target_lengths = phenome_seq_lengths
                )
                
                total_loss += loss.item()
                per = compute_phenome_error(logits, seq_class_ids, seq_lengths, phenome_seq_lengths)
                total_per += per
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'PER': f'{per:.4f}'})
        
        
        return total_loss / num_batches, total_per / num_batches
    
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
    
    def _save_checkpoint(self, epoch, val_loss, per, best=False):
        """Save model checkpoint"""
        self.logger.info(f"Saving checkpoint for epoch {epoch+1}. ::: val_loss {val_loss:.4f} PER: {per:.4f}")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        if best==True:
            checkpoint_path = Path(self.checkpoint_dir, f'best_model.pt')  
        else:
            filename = f"checkpoint_epoch-{epoch}_loss-{val_loss}_per-{per}.pt"
            checkpoint_path = Path(self.checkpoint_dir, filename)      
        if best==True:
            checkpoint_path = Path(self.checkpoint_dir, f'best_model.pt')  
        else:
            filename = f"checkpoint_epoch-{epoch}_loss-{val_loss}_per-{per}.pt"
            checkpoint_path = Path(self.checkpoint_dir, filename)      
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Saved model to {checkpoint_path}")

    def load_model_checkpoint(self):
        """Load model checkpoint"""
        checkpoint_path = self.config.get('training')['checkpoint_file']
        self.logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path} or set load_checkpoint False"
            )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model, optimizer, and scheduler states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Optional: load epoch and val_loss if needed
        epoch = checkpoint.get('epoch', None)
        val_loss = checkpoint.get('val_loss', None)
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch}, val_loss {val_loss})")
        
        
