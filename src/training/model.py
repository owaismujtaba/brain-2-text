import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.modeling_outputs import BaseModelOutput

import pdb





class BrainToTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._setup_config()

        bidirectional = True
        mel_bins = 80

        # EEG feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.rnn = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        out_dim = self.hidden_dim * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_dim, mel_bins)
        self.norm = nn.LayerNorm(mel_bins)

        
    def _setup_config(self):
        self.input_dim = self.config.get('model', {}).get('input_dim', 512)
        self.hidden_dim = self.config.get('model', {}).get('hidden_dim', 512)
        self.num_layers = self.config.get('model', {}).get('num_layers', 2)
        self.dropout = self.config.get('model', {}).get('dropout', 0.1)

    def forward(self, x, labels=None):
        """
        x: EEG input -> (batch_size, seq_len, input_dim)
        labels: tokenized text (for supervised training)
        returns: Whisper outputs (logits + loss if labels are provided)
        """
        # EEG → encoder
        x = self.feature_encoder(x)
        rnn_out, _ = self.rnn(x)                     # (B, mel_frames, H*)
        mel = self.proj(rnn_out)                     # (B, mel_frames, 80)
        mel = self.norm(mel)                         # (B, mel_frames, 80)
        mel = mel.transpose(1, 2).contiguous()       # (B, 80, mel_frames)
        
        return mel
