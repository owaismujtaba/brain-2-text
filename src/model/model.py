"""ConvBiGRU: maps neural activity to Whisper encoder embeddings."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ConvBiGRU(nn.Module):
    """Neural ``(B, T, neural_dim)`` -> Whisper embeddings ``(B, n_out, emb_dim)``.

    The forward pass has four stages:
        1. Conv front-end – two stride-2 convolutions smooth the neural signal
           and downsample it ~4x in time.
        2. Bi-GRU         – models temporal dynamics in both directions. Padding
           is handled with packed sequences so it never affects the outputs.
        3. Cross-attention – a fixed bank of learnable "query" frames pulls the
           relevant information out of the GRU output, producing exactly the
           number of target frames we ask for.
        4. MLP head       – regresses the ``emb_dim`` embedding for each frame.
    """

    def __init__(self, neural_dim=512, conv_channels=256, hidden=256, gru_layers=2,
                 emb_dim=384, n_ctx=1500, dropout=0.3, attn_heads=4):
        super().__init__()
        self.n_ctx = n_ctx
        gru_out = 2 * hidden          # bidirectional GRU concatenates both directions

        self.conv = nn.Sequential(
            nn.Conv1d(neural_dim, conv_channels, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            conv_channels, hidden, num_layers=gru_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.query = nn.Parameter(torch.randn(1, n_ctx, gru_out) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            gru_out, num_heads=attn_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(gru_out)
        self.head = nn.Sequential(
            nn.Linear(gru_out, gru_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_out, emb_dim),
        )

    @staticmethod
    def _downsampled_length(length):
        """Sequence length after the two stride-2 conv layers."""
        for _ in range(2):
            length = (length - 1) // 2 + 1
        return length

    def forward(self, neural, lengths=None, n_out=None):
        batch_size = neural.size(0)
        h = self.conv(neural.transpose(1, 2)).transpose(1, 2)     # (B, T', C)

        # Pack so the GRU ignores padding; unpack back to a dense tensor afterwards.
        if lengths is not None:
            conv_lengths = self._downsampled_length(lengths).clamp(min=1).cpu()
            h = pack_padded_sequence(h, conv_lengths, batch_first=True, enforce_sorted=False)

        h, _ = self.gru(h)

        if lengths is not None:
            h, _ = pad_packed_sequence(h, batch_first=True)

        out_len = n_out if n_out is not None else self.n_ctx
        queries = self.query[:, :out_len].expand(batch_size, -1, -1)

        # Mask so cross-attention never attends to GRU padding frames.
        key_padding_mask = None
        if lengths is not None:
            max_len = h.size(1)
            conv_lengths = self._downsampled_length(lengths).clamp(min=1).to(h.device)
            key_padding_mask = (
                torch.arange(max_len, device=h.device).unsqueeze(0)
                >= conv_lengths.unsqueeze(1)
            )

        attended, _ = self.cross_attn(queries, h, h, key_padding_mask=key_padding_mask)
        return self.head(self.attn_norm(attended))
