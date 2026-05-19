"""Shared model-detail summaries for Streamlit result and comparison pages."""

from __future__ import annotations

from typing import Any

from model_params import total_param_count


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def short_model_name(model: Any) -> str:
    text = str(model or "—")
    return text.split("/")[-1] if "/" in text else text


def task_type_from_hp(hp: dict, fallback: str = "ppi") -> str:
    return str((hp or {}).get("task_type") or fallback or "ppi").lower()


def input_dim_from_hp(hp: dict, task_type: str | None = None) -> int:
    hp = hp or {}
    task = (task_type or task_type_from_hp(hp)).lower()
    if hp.get("input_dim") is not None:
        return _safe_int(hp.get("input_dim"), 480)

    esm_dim = _safe_int(hp.get("esm_dim"), 480)
    representation_mode = str(
        hp.get("embedding_representation", hp.get("representation_mode", "pooled"))
    ).lower()
    is_chunked = representation_mode == "chunked"
    if representation_mode == "chunked":
        if task == "dtpi":
            return _safe_int(hp.get("chunk_model_dim"), max(_safe_int(hp.get("chem_dim"), 768), esm_dim))
        if task == "rpi":
            return _safe_int(hp.get("chunk_model_dim"), max(_safe_int(hp.get("rna_dim"), 640), esm_dim))
        if task == "pdi":
            return _safe_int(hp.get("chunk_model_dim"), max(_safe_int(hp.get("dna_dim"), 768), esm_dim))
        return esm_dim

    if task == "dtpi":
        return _safe_int(hp.get("chem_dim"), 768) + esm_dim
    if task == "rpi":
        return _safe_int(hp.get("rna_dim"), 640) + esm_dim
    if task == "pdi":
        return _safe_int(hp.get("dna_dim"), 768) + esm_dim
    return 2 * esm_dim


def embedding_summary(hp: dict, task_type: str | None = None) -> tuple[str, str]:
    hp = hp or {}
    task = (task_type or task_type_from_hp(hp)).lower()
    esm_model = str(hp.get("esm_model", "esm2_t12_35M_UR50D"))
    esm_label = esm_model.replace("esm2_", "ESM2 ").split("_UR")[0]
    esm_dim = _safe_int(hp.get("esm_dim"), 480)
    input_dim = input_dim_from_hp(hp, task)
    representation_mode = str(
        hp.get("embedding_representation", hp.get("representation_mode", "pooled"))
    ).lower()
    is_chunked = representation_mode == "chunked"

    if task == "dtpi":
        chem_dim = _safe_int(hp.get("chem_dim"), 768)
        chem_label = short_model_name(hp.get("chem_model", "seyonec/ChemBERTa-zinc-base-v1"))
        return (
            f"ChemBERTa `{chem_label}` {chem_dim}-dim + {esm_label} {esm_dim}-dim",
            f"{input_dim:,} shared chunk dim" if is_chunked else f"{input_dim:,} ({chem_dim} chem + {esm_dim} prot)",
        )
    if task == "rpi":
        rna_dim = _safe_int(hp.get("rna_dim"), 640)
        rna_label = short_model_name(hp.get("rna_model", "multimolecule/rnafm"))
        return (
            f"RNA-FM `{rna_label}` {rna_dim}-dim + {esm_label} {esm_dim}-dim",
            f"{input_dim:,} shared chunk dim" if is_chunked else f"{input_dim:,} ({rna_dim} rna + {esm_dim} prot)",
        )
    if task == "pdi":
        dna_dim = _safe_int(hp.get("dna_dim"), 768)
        dna_label = short_model_name(hp.get("dna_model", "armheb/DNA_bert_6"))
        return (
            f"DNABERT `{dna_label}` {dna_dim}-dim + {esm_label} {esm_dim}-dim",
            f"{input_dim:,} shared chunk dim" if is_chunked else f"{input_dim:,} ({dna_dim} dna + {esm_dim} prot)",
        )
    return esm_label, f"{input_dim:,} chunk dim" if is_chunked else f"{input_dim:,} (2 x {esm_dim})"


def approx_params(input_dim: int, layer_configs: list) -> int:
    return total_param_count(input_dim, layer_configs)


def approx_params_from_hp(hp: dict, task_type: str | None = None) -> int | None:
    hp = hp or {}
    layer_configs = hp.get("layer_configs", [])
    if not layer_configs:
        return None

    task = (task_type or task_type_from_hp(hp)).lower()
    input_dim = input_dim_from_hp(hp, task)
    representation_mode = str(
        hp.get("embedding_representation", hp.get("representation_mode", "pooled"))
    ).lower()
    sequence_mode = representation_mode == "chunked"
    projection_dims = None
    if sequence_mode and task in {"dtpi", "rpi", "pdi"}:
        left_key = {"dtpi": "chem_dim", "rpi": "rna_dim", "pdi": "dna_dim"}[task]
        left_default = 640 if task == "rpi" else 768
        projection_dims = (_safe_int(hp.get(left_key), left_default), _safe_int(hp.get("esm_dim"), 480))

    return total_param_count(
        input_dim,
        layer_configs,
        sequence_mode=sequence_mode,
        projection_dims=projection_dims,
    )


def layer_config_text(cfg: dict | None) -> str:
    if not cfg:
        return "—"
    lt = str(cfg.get("type", "linear")).lower()
    details = {
        "linear": lambda c: (
            f"LINEAR hidden={c.get('hidden_dim', 256)}, act={c.get('activation', 'relu')}, "
            f"drop={c.get('dropout', 0.3)}, bn={c.get('batchnorm', False)}"
        ),
        "cnn1d": lambda c: (
            f"CNN1D out_ch={c.get('out_channels', 64)}, kernel={c.get('kernel_size', 3)}, "
            f"act={c.get('activation', 'relu')}, drop={c.get('dropout', 0.3)}"
        ),
        "bilstm": lambda c: (
            f"BILSTM hidden={c.get('hidden_size', 128)}, layers={c.get('num_layers', 1)}, "
            f"drop={c.get('dropout', 0.3)}"
        ),
        "gru": lambda c: (
            f"GRU hidden={c.get('hidden_size', 128)}, layers={c.get('num_layers', 1)}, "
            f"bidir={c.get('bidirectional', True)}, drop={c.get('dropout', 0.3)}"
        ),
        "transformer": lambda c: (
            f"TRANSFORMER d_model={c.get('d_model', 256)}, nhead={c.get('nhead', 4)}, "
            f"layers={c.get('num_layers', 2)}, ff={c.get('dim_feedforward', 512)}, "
            f"drop={c.get('dropout', 0.1)}"
        ),
        "residual": lambda c: (
            f"RESIDUAL hidden={c.get('hidden_dim', 256)}, act={c.get('activation', 'relu')}, "
            f"drop={c.get('dropout', 0.3)}, bn={c.get('batchnorm', False)}"
        ),
    }
    return details.get(lt, lambda c: str(c))(cfg)


def layer_rows(layer_configs: list) -> list[dict[str, str]]:
    rows = [
        {
            "Layer": str(i + 1),
            "Type": str(cfg.get("type", "linear")).upper(),
            "Config": layer_config_text(cfg).split(" ", 1)[1] if " " in layer_config_text(cfg) else layer_config_text(cfg),
        }
        for i, cfg in enumerate(layer_configs or [])
    ]
    if layer_configs:
        rows.append({"Layer": "Out", "Type": "LINEAR", "Config": "out=1, sigmoid"})
    return rows


def layer_difference_rows(models: list[dict]) -> list[dict[str, str]]:
    max_layers = max((len(m.get("layer_configs") or []) for m in models), default=0)
    rows: list[dict[str, str]] = []
    for idx in range(max_layers):
        row = {"Layer": str(idx + 1)}
        values = []
        for model in models:
            label = model["label"]
            cfg = (model.get("layer_configs") or [])[idx] if idx < len(model.get("layer_configs") or []) else None
            value = layer_config_text(cfg)
            row[label] = value
            values.append(value)
        row["Difference"] = "Same" if len(set(values)) <= 1 else "Different"
        rows.append(row)
    if models:
        row = {"Layer": "Out"}
        for model in models:
            row[model["label"]] = "LINEAR out=1, sigmoid"
        row["Difference"] = "Same"
        rows.append(row)
    return rows


def render_model_details(
    st,
    pd,
    hp: dict,
    task_type: str | None = None,
    *,
    expanded: bool = True,
    actual_params: int | None = None,
) -> None:
    hp = hp or {}
    task = (task_type or task_type_from_hp(hp)).lower()
    layer_configs = hp.get("layer_configs", [])
    input_dim = input_dim_from_hp(hp, task)
    emb_str, dim_str = embedding_summary(hp, task)
    n_params = approx_params_from_hp(hp, task)
    if actual_params is None:
        actual_params = hp.get("trainable_params")
    subtxt = "#c7ccd3" if st.session_state.get("theme_mode", "Light") == "Dark" else "#6b7280"

    with st.expander("Model details", expanded=expanded):
        mc1, mc2, mc3, mc4 = st.columns(4)

        def card(label: str, val: str) -> str:
            return f"""
                <div style="padding:4px 0">
                    <div style="font-size:0.78rem;color:{subtxt};margin-bottom:2px">{label}</div>
                    <div style="font-size:0.9rem;font-weight:600">{val}</div>
                </div>"""

        mc1.markdown(card("Embedding model", emb_str), unsafe_allow_html=True)
        mc2.markdown(card("Input dim", dim_str), unsafe_allow_html=True)
        mc3.markdown(card("Approx. parameters", f"{n_params:,}" if n_params else "—"), unsafe_allow_html=True)
        mc4.markdown(card("Actual parameters", f"{int(actual_params):,}" if actual_params else "—"), unsafe_allow_html=True)

        if layer_configs:
            st.dataframe(pd.DataFrame(layer_rows(layer_configs)), use_container_width=True, hide_index=True)
        else:
            st.caption("Layer configuration not available for this run.")


def render_layer_difference_table(st, pd, models: list[dict], *, title: str = "Model Layer Differences") -> None:
    st.subheader(title)
    rows = layer_difference_rows(models)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Layer configuration not available for the loaded runs.")
