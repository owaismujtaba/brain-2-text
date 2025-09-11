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

        # Load Whisper (frozen)
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small", cache_dir="./hf_cache")
        self.whisper =WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", cache_dir="./hf_cache")


        for param in self.whisper.parameters():
            param.requires_grad = False  # freeze Whisper

        # EEG feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True
        )

        # Project EEG → Whisper encoder dimension
        self.whisper_hidden_size = self.whisper.model.encoder.conv1.out_channels
        self.project_to_whisper = nn.Linear(
            self.hidden_dim * 2,
            self.whisper_hidden_size
        )

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
        x, _ = self.lstm(x)

        # EEG → Whisper latent space
        encoder_hidden_states = self.project_to_whisper(x)

        # Wrap in correct HF output format
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        # Frozen Whisper decoder
        outputs = self.whisper(
            encoder_outputs=encoder_outputs,
            labels=labels  # required during training
        )
        return outputs

    def generate(self, x, **gen_kwargs):
        """
        EEG → text (generation with frozen Whisper)
        """
        x = self.feature_encoder(x)
        x, _ = self.lstm(x)
        pdb.set_trace()
        encoder_hidden_states = self.project_to_whisper(x)

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        generated_ids = self.whisper.generate(
            encoder_outputs=encoder_outputs,
            **gen_kwargs
        )
        return generated_ids

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
