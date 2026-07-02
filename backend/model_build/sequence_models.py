"""Shared sequence/chunk models and utility helpers for interaction tasks."""

import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _safe(v):
    """Convert numeric values to rounded Python floats, using None for NaN/Inf."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _train_test_size(hyperparams: dict) -> float:
    """Return the validation fraction derived from the UI train_split value."""
    try:
        train_split = float(hyperparams.get("train_split", 0.8))
    except (TypeError, ValueError):
        train_split = 0.8
    if train_split > 1:
        train_split /= 100
    train_split = min(0.95, max(0.05, train_split))
    return 1 - train_split


def trainable_parameter_count(model: nn.Module) -> int:
    """Return the actual number of trainable PyTorch parameters."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _get_act(name: str) -> nn.Module:
    """Return a fresh activation module for the given name."""
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    if name == "silu":
        return nn.SiLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    return nn.ReLU()


class _SeqLinearBlock(nn.Module):
    def __init__(self, in_dim: int, cfg: dict):
        super().__init__()
        hidden = int(cfg.get("hidden_dim", 256))
        act = cfg.get("activation", "relu")
        drop = float(cfg.get("dropout", 0.3))
        bn = bool(cfg.get("batchnorm", False))
        self.fc = nn.Linear(in_dim, hidden)
        self.bn = nn.BatchNorm1d(hidden) if bn else None
        self.act = _get_act(act)
        self.drop = nn.Dropout(drop)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.fc(x)
        if self.bn is not None:
            b, l, d = x.shape
            x = self.bn(x.reshape(b * l, d)).reshape(b, l, d)
        return self.drop(self.act(x))


class _SeqCNN1DBlock(nn.Module):
    def __init__(self, in_dim: int, cfg: dict):
        super().__init__()
        out_ch = int(cfg.get("out_channels", 64))
        k = int(cfg.get("kernel_size", 3))
        act = cfg.get("activation", "relu")
        drop = float(cfg.get("dropout", 0.3))
        self.conv = nn.Conv1d(in_dim, out_ch, kernel_size=k, padding=k // 2)
        self.act = _get_act(act)
        self.drop = nn.Dropout(drop)
        self.out_dim = out_ch

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.drop(self.act(x))
        return x.transpose(1, 2)


class _SeqBiLSTMBlock(nn.Module):
    def __init__(self, in_dim: int, cfg: dict):
        super().__init__()
        hidden = int(cfg.get("hidden_size", 128))
        num_layers = int(cfg.get("num_layers", 1))
        drop = float(cfg.get("dropout", 0.3))
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=drop if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(drop)
        self.out_dim = 2 * hidden

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            lengths = mask.sum(dim=1).clamp(min=1).to(dtype=torch.long).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            x, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        else:
            x, _ = self.lstm(x)
        return self.drop(x)


class _SeqGRUBlock(nn.Module):
    def __init__(self, in_dim: int, cfg: dict):
        super().__init__()
        hidden = int(cfg.get("hidden_size", 128))
        num_layers = int(cfg.get("num_layers", 1))
        bidirectional = bool(cfg.get("bidirectional", True))
        drop = float(cfg.get("dropout", 0.3))
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=drop if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(drop)
        self.out_dim = (2 if bidirectional else 1) * hidden

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            lengths = mask.sum(dim=1).clamp(min=1).to(dtype=torch.long).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.gru(packed)
            x, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        else:
            x, _ = self.gru(x)
        return self.drop(x)


class _SeqTransformerBlock(nn.Module):
    def __init__(self, in_dim: int, cfg: dict):
        super().__init__()
        d_model = int(cfg.get("d_model", 256))
        nhead = int(cfg.get("nhead", 4))
        num_layers = int(cfg.get("num_layers", 2))
        dim_ff = int(cfg.get("dim_feedforward", d_model * 2))
        drop = float(cfg.get("dropout", 0.1))
        while nhead > 1 and d_model % nhead != 0:
            nhead -= 1
        self.proj = nn.Linear(in_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 4096, d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=drop,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.drop = nn.Dropout(drop)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.proj(x)
        if x.size(1) <= self.pos_embed.size(1):
            x = x + self.pos_embed[:, : x.size(1), :]
        padding_mask = None
        if mask is not None:
            padding_mask = ~mask.bool()
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.drop(x)


class _SeqResidualBlock(nn.Module):
    def __init__(self, in_dim: int, cfg: dict):
        super().__init__()
        hidden = int(cfg.get("hidden_dim", 256))
        act = cfg.get("activation", "relu")
        drop = float(cfg.get("dropout", 0.3))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            _get_act(act),
            nn.Dropout(drop),
            nn.Linear(hidden, in_dim),
        )
        self.norm = nn.LayerNorm(in_dim)
        self.out_dim = in_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.norm(x + self.net(x))


def _build_sequence_layer(layer_type: str, in_dim: int, cfg: dict) -> nn.Module:
    lt = layer_type.lower()
    if lt == "linear":
        return _SeqLinearBlock(in_dim, cfg)
    if lt == "cnn1d":
        return _SeqCNN1DBlock(in_dim, cfg)
    if lt == "bilstm":
        return _SeqBiLSTMBlock(in_dim, cfg)
    if lt == "gru":
        return _SeqGRUBlock(in_dim, cfg)
    if lt == "transformer":
        return _SeqTransformerBlock(in_dim, cfg)
    if lt == "residual":
        return _SeqResidualBlock(in_dim, cfg)
    raise ValueError(f"Unknown layer type: '{layer_type}'")


class FlexiblePPISequenceModel(nn.Module):
    """Flexible PPI model for chunk-pooled embeddings."""

    def __init__(self, input_dim: int, layer_configs: list):
        super().__init__()
        if not layer_configs:
            layer_configs = [
                {"type": "bilstm", "hidden_size": 128, "num_layers": 1, "dropout": 0.3},
                {"type": "linear", "hidden_dim": 64, "activation": "relu", "dropout": 0.2},
            ]

        blocks = []
        cur_dim = input_dim
        for cfg in layer_configs:
            block = _build_sequence_layer(cfg["type"], cur_dim, cfg)
            blocks.append(block)
            cur_dim = block.out_dim

        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Linear(2 * cur_dim, 1)

    @staticmethod
    def _apply_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return x
        return x * mask.to(dtype=x.dtype).unsqueeze(-1)

    @staticmethod
    def _masked_pool(x: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            return x.mean(dim=1), x.max(dim=1).values

        mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        mean_pool = (x * mask_f).sum(dim=1) / denom

        masked_x = x.masked_fill(~mask.bool().unsqueeze(-1), torch.finfo(x.dtype).min)
        max_pool = masked_x.max(dim=1).values
        has_real_chunk = mask.bool().any(dim=1, keepdim=True)
        max_pool = torch.where(has_real_chunk, max_pool, torch.zeros_like(max_pool))
        return mean_pool, max_pool

    @staticmethod
    def _compact_masked_sequence(
        x: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if mask is None:
            return x, mask
        mask = mask.bool()
        lengths = mask.sum(dim=1)
        max_len = int(lengths.max().item()) if lengths.numel() else 0
        max_len = max(max_len, 1)
        compact = x.new_zeros((x.size(0), max_len, x.size(-1)))
        compact_mask = torch.zeros((x.size(0), max_len), dtype=torch.bool, device=x.device)
        for row_idx in range(x.size(0)):
            real = x[row_idx, mask[row_idx]]
            n = real.size(0)
            if n:
                compact[row_idx, :n] = real
                compact_mask[row_idx, :n] = True
        return compact, compact_mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            mask = mask.bool()
            x, mask = self._compact_masked_sequence(x, mask)
            x = self._apply_mask(x, mask)
        for block in self.blocks:
            x = block(x, mask)
            x = self._apply_mask(x, mask)
        mean_pool, max_pool = self._masked_pool(x, mask)
        x = torch.cat([mean_pool, max_pool], dim=-1)
        return self.output(x).squeeze(-1)


class FlexiblePairSequenceModel(nn.Module):
    """Chunk-aware interaction model for two embedded modalities."""

    def __init__(
        self,
        left_dim: int,
        right_dim: int,
        model_dim: int,
        layer_configs: list,
    ):
        super().__init__()
        if not layer_configs:
            layer_configs = [
                {"type": "bilstm", "hidden_size": 128, "num_layers": 1, "dropout": 0.3},
                {"type": "linear", "hidden_dim": 64, "activation": "relu", "dropout": 0.2},
            ]

        model_dim = max(1, int(model_dim))
        self.left_proj = nn.Linear(int(left_dim), model_dim)
        self.right_proj = nn.Linear(int(right_dim), model_dim)
        self.type_embed = nn.Parameter(torch.zeros(2, model_dim))
        nn.init.normal_(self.type_embed, mean=0.0, std=0.02)

        blocks = []
        cur_dim = model_dim
        for cfg in layer_configs:
            block = _build_sequence_layer(cfg["type"], cur_dim, cfg)
            blocks.append(block)
            cur_dim = block.out_dim

        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Linear(2 * cur_dim, 1)

    @staticmethod
    def _apply_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return FlexiblePPISequenceModel._apply_mask(x, mask)

    @staticmethod
    def _masked_pool(x: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        return FlexiblePPISequenceModel._masked_pool(x, mask)

    def forward(
        self,
        left_chunks: torch.Tensor,
        right_chunks: torch.Tensor,
        left_mask: torch.Tensor | None = None,
        right_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        left = self.left_proj(left_chunks) + self.type_embed[0]
        right = self.right_proj(right_chunks) + self.type_embed[1]
        x = torch.cat([left, right], dim=1)

        mask = None
        if left_mask is not None and right_mask is not None:
            mask = torch.cat([left_mask.bool(), right_mask.bool()], dim=1)
        elif left_mask is not None:
            right_real = torch.ones(
                right.size(0), right.size(1), dtype=torch.bool, device=right.device
            )
            mask = torch.cat([left_mask.bool(), right_real], dim=1)
        elif right_mask is not None:
            left_real = torch.ones(
                left.size(0), left.size(1), dtype=torch.bool, device=left.device
            )
            mask = torch.cat([left_real, right_mask.bool()], dim=1)

        if mask is not None:
            x, mask = FlexiblePPISequenceModel._compact_masked_sequence(x, mask)
            x = self._apply_mask(x, mask)
        for block in self.blocks:
            x = block(x, mask)
            x = self._apply_mask(x, mask)
        mean_pool, max_pool = self._masked_pool(x, mask)
        x = torch.cat([mean_pool, max_pool], dim=-1)
        return self.output(x).squeeze(-1)
