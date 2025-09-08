import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardModule(nn.Module):
    def __init__(self, dim, ff_multiplier=4, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * ff_multiplier)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * ff_multiplier, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return self.layer_norm(x + residual)

class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ff_multiplier=4, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(dim, ff_multiplier, dropout)
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2, groups=dim),
            nn.GLU(dim=1),
            nn.BatchNorm1d(dim // 2),
            nn.Conv1d(dim // 2, dim, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.ff2 = FeedForwardModule(dim, ff_multiplier, dropout)
        self.layer_norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        # Feed-forward
        x = self.ff1(x)
        
        # Multi-head self-attention
        residual = x
        attn_output, _ = self.mha(x, x, x, key_padding_mask=mask)
        x = self.layer_norm1(attn_output + residual)
        
        # Convolution module
        residual = x
        x_conv = x.transpose(1, 2)  # (B, C, T)
        x_conv = self.conv(x_conv)
        x = x_conv.transpose(1, 2) + residual
        
        # Feed-forward
        x = self.ff2(x)
        x = self.layer_norm2(x)
        return x

class BrainToTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.get('model', {}).get('input_dim', 512)
        self.hidden_dim = config.get('model', {}).get('hidden_dim', 512)
        self.num_layers = config.get('model', {}).get('num_layers', 2)
        self.num_classes = config.get('model', {}).get('num_classes', 41)
        self.dropout = config.get('model', {}).get('dropout', 0.1)

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        # Conformer layers
        self.conformers = nn.ModuleList([
            ConformerBlock(self.hidden_dim, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

    def forward(self, x, lengths=None):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.feature_encoder(x)

        # Optional mask for variable lengths
        mask = None
        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)

        # Conformer blocks
        for block in self.conformers:
            x = block(x, mask=mask)

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
