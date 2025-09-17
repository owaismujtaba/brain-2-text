import torch
import torch.nn as nn
import torch.nn.functional as F
torch.cuda.empty_cache()
import pdb


import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual feedforward block: Linear -> ReLU -> BN -> Dropout -> Linear + skip"""
    def __init__(self, dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm(x)
        return F.relu(x + residual)


class GRUDecoderAttention(nn.Module):

    

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._setup_config_parms()
        

        # Dropout
        self.input_dropout = nn.Dropout(self.input_dropout)

        # Input dimension adjustment for patching
        self.input_size = self.neural_dim * (self.patch_size if self.patch_size > 0 else 1)

        # GRU backbone
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.n_units,
            num_layers=self.n_layers,
            dropout=self.gru_dropout if self.n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        out_dim = self.n_units * (2 if self.gru.bidirectional else 1)

        # Multi-head self-attention on GRU outputs
        self.attn = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=self.attn_heads,
            dropout=self.attn_dropout,
            batch_first=True,
        )

        # Residual FC head (stacked)
        self.fc_blocks = nn.Sequential(*[
            ResidualBlock(out_dim, self.hidden_fc, dropout=0.3)
            for _ in range(self.n_resblocks)
        ])

        # Final classification
        self.out = nn.Linear(out_dim, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        # Learnable initial hidden states
        self.h0 = nn.Parameter(
            torch.empty(self.n_layers * (2 if self.gru.bidirectional else 1), 
                1, 
                self.n_units
            )
        )
        nn.init.xavier_uniform_(self.h0)

        self.norm = nn.LayerNorm(self.input_size)

    def _setup_config_parms(self):
        self.neural_dim = self.config.get('model', {}).get('input_dim', 512)
        self.n_units = self.config.get('model', {}).get('n_units', 512)
        self.n_classes = self.config.get('model', {}).get('num_classes', 512)
        self.n_layers = self.config.get('model', {}).get('num_layers', 4)
        self.attn_heads = self.config.get('model', {}).get('attn_heads', 4)
        self.attn_dropout = self.config.get('model', {}).get('attn_dropout', 0.1)
        self.input_dropout = self.config.get('model', {}).get('input_dropout', 0.1)
        self.gru_dropout = self.config.get('model', {}).get('gru_dropout', 0.1)
        self.n_resblocks = self.config.get('model', {}).get('n_resblocks', 4)
        self.patch_size = self.config.get('model', {}).get('patch_size', 512)
        self.patch_stride = self.config.get('model', {}).get('patch_stride', 512)
        self.hidden_fc = self.config.get('model', {}).get('hidden_fc', 512)
    

    def forward(self, x, states=None, return_state=False):
        """
        x: [B, T, D]
        """
        B, T, D = x.shape
        x = self.input_dropout(x)
        
        # Optional patching
        if self.patch_size > 0:
            x = x.permute(0, 2, 1)  # [B, D, T]
            patches = x.unfold(2, self.patch_size, self.patch_stride)  # [B, D, num_patches, patch_size]
            x = patches.permute(0, 2, 3, 1).reshape(B, -1, self.input_size)  # [B, num_patches, input_size]

        x = self.norm(x)

        # Init hidden state
        if states is None:
            states = self.h0.expand(-1, B, -1).contiguous()

        # GRU
        
        output, hidden_states = self.gru(x, states)  # [B, T, H]

        # Self-attention (contextual refinement)
        attn_out, _ = self.attn(output, output, output)  # [B, T, H]
        output = output + attn_out  # residual connection

        # Sequence or pooled output
        
        out_fc = self.fc_blocks(output.reshape(-1, output.size(-1)))  # [B*T, H]
        logits = self.out(out_fc).view(B, -1, self.n_classes)  # [B, T, C]
        
        if return_state:
            return logits, hidden_states
        
        return logits
















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

class BrainToTextModelConformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Feature encoder
        super().__init__()        
        self._setup_config_parm(config)
        
        # Layers
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

        self.lstm = nn.LSTM(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True
        )
        
        self.lstm_output_dim = self.hidden_dim * (2 if self.lstm.bidirectional else 1)
        self.proj = nn.Linear(self.lstm_output_dim, self.hidden_dim)
        self.lstm_output_dim = self.hidden_dim * 2 

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
    def _setup_config_parm(self, config):
        self.input_dim = config.get('model', {}).get('input_dim', 512)
        self.hidden_dim = config.get('model', {}).get('hidden_dim', 512)
        self.num_layers = config.get('model', {}).get('num_layers', 2)
        self.num_classes = config.get('model', {}).get('num_classes', 41)  # Vocabulary size
        self.dropout = config.get('model', {}).get('dropout', 0.1)
    
    def forward(self, x, lengths=None):
        # x shape: (batch_size, seq_len, input_dim)
        
        x = self.feature_encoder(x)
        
        # Conformer blocks
        for block in self.conformers:
            x = block(x)
    
        x, _ = self.lstm(x)
        x = self.proj(x)
       
        
        logits = self.classifier(x)
        
        return logits

    
    
