import torch

import torch.nn as nn

class BrainToTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Get model parameters from config
        self.input_dim = config.get('model', {}).get('input_dim', 512)
        self.hidden_dim = config.get('model', {}).get('hidden_dim', 512)
        self.num_layers = config.get('model', {}).get('num_layers', 2)
        self.num_classes = config.get('model', {}).get('num_classes', 41)  # Vocabulary size
        self.dropout = config.get('model', {}).get('dropout', 0.1)

        # Layers
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True
        )

        # Output layer (bidirectional LSTM output -> num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

    def forward(self, x, lengths=None):
        # x shape: (batch_size, sequence_length, input_dim)
        
        # Encode features
        x = self.feature_encoder(x)

        # Pack sequence for LSTM if lengths are provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # Process through LSTM
        x, _ = self.lstm(x)

        # Unpack sequence if it was packed
        if lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Apply classifier
        logits = self.classifier(x)
        
        return logits

    def configure_optimizers(self, config):
        """Configure optimizer and learning rate scheduler"""
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