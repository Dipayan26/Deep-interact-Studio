"""Shared trainable-parameter estimates for model-builder UIs."""

TRANSFORMER_MAX_POSITIONS = 4096


def safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def layer_output_dim(in_dim: int, cfg: dict) -> int:
    lt = str(cfg.get("type", "linear")).lower()
    if lt == "linear":
        return safe_int(cfg.get("hidden_dim"), 256)
    if lt == "cnn1d":
        return safe_int(cfg.get("out_channels"), 64)
    if lt == "bilstm":
        return 2 * safe_int(cfg.get("hidden_size"), 128)
    if lt == "gru":
        hidden = safe_int(cfg.get("hidden_size"), 128)
        return 2 * hidden if cfg.get("bidirectional", True) else hidden
    if lt == "transformer":
        return safe_int(cfg.get("d_model"), 256)
    return in_dim


def layer_param_count(in_dim: int, cfg: dict, *, sequence_mode: bool = False) -> tuple[int, int]:
    lt = str(cfg.get("type", "linear")).lower()
    out_dim = layer_output_dim(in_dim, cfg)
    total = 0

    if lt == "linear":
        hidden = safe_int(cfg.get("hidden_dim"), 256)
        total += in_dim * hidden + hidden
        if cfg.get("batchnorm"):
            total += 2 * hidden
    elif lt == "cnn1d":
        out_ch = safe_int(cfg.get("out_channels"), 64)
        kernel = safe_int(cfg.get("kernel_size"), 3)
        in_ch = in_dim if sequence_mode else 1
        total += in_ch * out_ch * kernel + out_ch
    elif lt == "bilstm":
        hidden = safe_int(cfg.get("hidden_size"), 128)
        num_layers = safe_int(cfg.get("num_layers"), 1)
        gate = 4
        dirs = 2
        total += dirs * gate * (in_dim * hidden + hidden * hidden + 2 * hidden)
        for _ in range(num_layers - 1):
            total += dirs * gate * (dirs * hidden * hidden + hidden * hidden + 2 * hidden)
    elif lt == "gru":
        hidden = safe_int(cfg.get("hidden_size"), 128)
        num_layers = safe_int(cfg.get("num_layers"), 1)
        dirs = 2 if cfg.get("bidirectional", True) else 1
        gate = 3
        total += dirs * gate * (in_dim * hidden + hidden * hidden + 2 * hidden)
        for _ in range(num_layers - 1):
            total += dirs * gate * (dirs * hidden * hidden + hidden * hidden + 2 * hidden)
    elif lt == "transformer":
        d_model = safe_int(cfg.get("d_model"), 256)
        ff_dim = safe_int(cfg.get("dim_feedforward"), d_model * 2)
        num_layers = safe_int(cfg.get("num_layers"), 2)
        total += in_dim * d_model + d_model
        if sequence_mode:
            total += TRANSFORMER_MAX_POSITIONS * d_model
        total += num_layers * (
            4 * d_model * d_model
            + 4 * d_model
            + d_model * ff_dim
            + ff_dim
            + ff_dim * d_model
            + d_model
            + 4 * d_model
        )
    elif lt == "residual":
        hidden = safe_int(cfg.get("hidden_dim"), 256)
        total += in_dim * hidden + hidden + hidden * in_dim + in_dim
        if cfg.get("batchnorm"):
            total += 2 * hidden
        total += 2 * in_dim

    return total, out_dim


def total_param_count(
    input_dim: int,
    layer_configs: list,
    *,
    sequence_mode: bool = False,
    projection_dims: tuple[int, int] | None = None,
) -> int:
    total = 0
    cur = input_dim

    if sequence_mode and projection_dims is not None:
        left_dim, right_dim = projection_dims
        total += left_dim * input_dim + input_dim
        total += right_dim * input_dim + input_dim
        total += 2 * input_dim

    for cfg in layer_configs or []:
        layer_params, cur = layer_param_count(cur, cfg, sequence_mode=sequence_mode)
        total += layer_params

    head_in = 2 * cur if sequence_mode else cur
    total += head_in + 1
    return total
