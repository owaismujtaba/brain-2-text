import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor

class BrainToTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._setup_config()

        # Load Whisper (frozen)
        self.whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        
        for param in self.whisper.parameters():
            param.requires_grad = False  # freeze Whisper

        # EEG feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True
        )

        self.whisper_conv1_out_channels = self.whisper.model.encoder.conv1.out_channels

        # Project LSTM output to Whisper encoder dimension
        self.project_to_whisper = nn.Linear(self.hidden_dim * 2,
                                            self.whisper_conv1_out_channels)

        # Optional classifier (EEG → fixed classes)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

    def _setup_config(self):
        self.input_dim = self.config.get('model', {}).get('input_dim', 512)
        self.hidden_dim = self.config.get('model', {}).get('hidden_dim', 512)
        self.num_layers = self.config.get('model', {}).get('num_layers', 2)
        self.dropout = self.config.get('model', {}).get('dropout', 0.1)
        self.num_classes = self.config.get('model', {}).get('num_classes', 10)

    def forward(self, x):
        """
        x: EEG input -> (batch_size, seq_len, input_dim)
        returns: classifier logits
        """
        # Encode features
        x = self.feature_encoder(x)

        # LSTM
        x, _ = self.lstm(x)

        # Project to Whisper encoder dimension (frozen)
        whisper_features = self.project_to_whisper(x)

        # Optional: classifier output
        logits = self.classifier(x)

        return logits, whisper_features

    def configure_optimizers(self, config):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.get('training', {}).get('learning_rate', 1e-3),
            weight_decay=config.get('training', {}).get('weight_decay', 1e-5)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
        return optimizer, scheduler
