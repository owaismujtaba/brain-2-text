import os
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from src.evaluation.eval import compute_phenome_error

from src.dataset.utils import pad_to_mel_length

import pdb
class Trainer:
    def __init__(self, model, config, logger=None):
        self.model = model
        self.config = config
        self.best_per = float('inf')
        self.best_loss = float('inf')
        self.logger = logger
        self.logger.info("Initializing Trainer")
        
        
        self._setup_config_params()

        # Models
        self.processor = WhisperProcessor.from_pretrained('openai/whisper-small', language='en', task="transcribe")
        self.whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", cache_dir="./hf_cache")
        self._device_setup()
        self.whisper.eval()
        for param in self.whisper.parameters():
            param.requires_grad = False

        # Optimizer & Scheduler
        self.optimizer, self.scheduler = model.configure_optimizers(config)

        # Loss & AMP
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.processor.tokenizer.pad_token_id)
        self.scaler = torch.cuda.amp.GradScaler()
    
    def _setup_config_params(self):
        training_cfg = self.config.get('training', {})
        self.batch_size = training_cfg.get('batch_size', 32)
        self.num_epochs = training_cfg.get('epochs', 10)
        self.checkpoint_dir = training_cfg.get('checkpoints_dir', './checkpoints')
        self.mel_frames = training_cfg.get('mel_frames', 3000)
    
    def _device_setup(self):
        self.logger.info("Setting up device")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.whisper.to(self.device)
        self.logger.info(f"Using device: {self.device}")

    def train(self, train_loader, val_loader=None):
        self.logger.info("Starting training process")
        
        for epoch in range(self.num_epochs):
            self.model.train()
            avg_train_loss = self._train_epoch(train_loader)
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Avg Train Loss: {avg_train_loss:.4f}")

            if val_loader is not None:
                avg_val_loss = self._validate(val_loader)
                self.logger.info(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")
                # Save checkpoint if improved
                if avg_val_loss < self.best_loss:
                    self.best_loss = avg_val_loss
                    self._save_checkpoint(epoch, avg_val_loss, per=0)

    def _train_epoch(self, dataloader):
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc="Training", leave=False)
        for batch in progress:
            inputs, _, tok_labels, _ = self._prepare_batch(batch)
            loss = self._train_step(inputs, tok_labels)
            epoch_loss += loss
            progress.set_postfix({"batch_loss": loss})
        return epoch_loss / len(dataloader)

    def _train_step(self, inputs, tok_labels):
        self.model.train()
        self.whisper.eval()
        
        with torch.cuda.amp.autocast(enabled=(self.device.type=="cuda")):
            input_features = self.model(inputs)
            input_features = pad_to_mel_length(input_features, self.mel_frames)
            outputs = self.whisper(input_features=input_features, labels=tok_labels)
            loss = outputs.loss
        
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()

    def _prepare_batch(self, batch):
        inputs = batch['neural_features'].to(self.device)
        seq_lengths = batch['seq_lengths'].to(self.device)
        tok_labels = batch['tok_labels'].to(self.device)
        sentence_labels = batch['sentence_label']  # not used for training
        return inputs, seq_lengths, tok_labels, sentence_labels

    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            progress = tqdm(val_loader, desc="Validating", leave=False)
            for batch in progress:
                inputs, seq_lengths, tok_labels, _ = self._prepare_batch(batch)
                with torch.cuda.amp.autocast(enabled=(self.device.type=="cuda")):
                    input_features = self.model(inputs)
                    input_features = pad_to_mel_length(input_features, self.mel_frames)
                    outputs = self.whisper(input_features=input_features, labels=tok_labels)
                    loss = outputs.loss.item()
                    val_loss += loss
                    progress.set_postfix({"batch_loss": loss})
        return val_loss / len(val_loader)

    def _save_checkpoint(self, epoch, val_loss, per):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        filename = f'checkpoint_epoch-{epoch+1:03d}_loss-{val_loss:.4f}_per-{per}.pth'
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        torch.save(checkpoint, path)
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best_model.pth'))
        self.logger.info(f"Saved checkpoint: {path}")
