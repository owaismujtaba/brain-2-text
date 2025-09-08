
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


class BrainToTextPretrained(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Config
        self.input_dim = config.get('model', {}).get('input_dim', 512)
        self.num_classes = config.get('model', {}).get('num_classes', 41)  # include blank for CTC
        self.dropout = config.get('model', {}).get('dropout', 0.1)

        # Load pretrained Wav2Vec2 encoder
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

        hidden_dim = self.encoder.config.hidden_size  # usually 768 for base

        # Adapter: EEG → Wav2Vec2 hidden size
        self.brain_adapter = nn.Linear(self.input_dim, hidden_dim)

        # BiLSTM stack for extra modeling power
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            dropout=self.dropout,
            bidirectional=True,
            batch_first=True
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, self.num_classes)
        )

        # CTC loss
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, x):
        """
        x: (batch, time, input_dim) brain features
        y: target sequences (optional, for training)
        y_lengths: target sequence lengths (optional)
        """
        batch_size, seq_len, _ = x.shape

        # Map brain features → Wav2Vec2 space
        x = self.brain_adapter(x)  # (batch, time, hidden_dim)

        # Pass through pretrained speech encoder
        encoder_out = self.encoder(inputs_embeds=x).last_hidden_state  # (batch, time, hidden_dim)

        # Extra LSTM modeling
        x, _ = self.lstm(encoder_out)

        # Classifier
        logits = self.classifier(x)    # (batch, time, num_classes)
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs  # (batch, time, num_classes)

    def configure_optimizers(self, config):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=config.get('training', {}).get('learning_rate', 1e-4),
            weight_decay=config.get('training', {}).get('weight_decay', 1e-4)
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.get('training', {}).get('learning_rate', 1e-4),
            steps_per_epoch=config.get('training', {}).get('steps_per_epoch', 1000),
            epochs=config.get('training', {}).get('epochs', 50),
            anneal_strategy='cos'
        )
        return optimizer, scheduler
