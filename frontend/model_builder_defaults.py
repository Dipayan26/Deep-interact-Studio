import streamlit as st


DEFAULT_LAYER_CONFIGS = [
    {"id": 0, "type": "linear", "hidden_dim": 256, "activation": "relu", "dropout": 0.3, "batchnorm": False},
    {"id": 1, "type": "linear", "hidden_dim": 64, "activation": "relu", "dropout": 0.2, "batchnorm": False},
]

_LAYER_WIDGET_PREFIXES = (
    "up_",
    "dn_",
    "rm_",
    "hd_",
    "act_",
    "drop_",
    "bn_",
    "och_",
    "ks_",
    "hs_",
    "nl_",
    "bidir_",
    "dm_",
    "nh_",
    "ff_",
)


def default_layers() -> list[dict]:
    return [layer.copy() for layer in DEFAULT_LAYER_CONFIGS]


def reset_model_builder_state(
    layer_key: str,
    lid_key: str,
    *,
    widget_prefix: str = "",
    new_layer_type_key: str,
    model_defaults: dict[str, str] | None = None,
) -> None:
    key_prefixes = tuple(f"{widget_prefix}{prefix}" for prefix in _LAYER_WIDGET_PREFIXES)
    keys_to_clear = [
        key
        for key in st.session_state.keys()
        if key == new_layer_type_key or key.startswith(key_prefixes)
    ]
    for key in keys_to_clear:
        del st.session_state[key]

    st.session_state[layer_key] = default_layers()
    st.session_state[lid_key] = len(DEFAULT_LAYER_CONFIGS)

    for key, value in (model_defaults or {}).items():
        st.session_state[key] = value
