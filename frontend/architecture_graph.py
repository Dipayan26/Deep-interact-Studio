import html
import json

import streamlit as st
import streamlit.components.v1 as components


_TYPE_STYLES = {
    "linear": ("#EAF2FF", "#2563EB"),
    "cnn1d": ("#ECFDF5", "#059669"),
    "bilstm": ("#FFF7ED", "#EA580C"),
    "gru": ("#F5F3FF", "#7C3AED"),
    "transformer": ("#FEF2F2", "#DC2626"),
    "residual": ("#F8FAFC", "#475569"),
    "input": ("#EFF6FF", "#1D4ED8"),
    "output": ("#F0FDF4", "#16A34A"),
}


def _compute_out_dim(layer_type: str, in_dim: int, cfg: dict) -> int:
    lt = layer_type.lower()
    if lt == "linear":
        return int(cfg.get("hidden_dim", 256))
    if lt == "cnn1d":
        return int(cfg.get("out_channels", 64))
    if lt == "bilstm":
        return 2 * int(cfg.get("hidden_size", 128))
    if lt == "gru":
        hidden = int(cfg.get("hidden_size", 128))
        bidir = bool(cfg.get("bidirectional", True))
        return 2 * hidden if bidir else hidden
    if lt == "transformer":
        return int(cfg.get("d_model", 256))
    if lt == "residual":
        return in_dim
    return in_dim


def _layer_param_count(in_dim: int, cfg: dict) -> tuple[int, int]:
    lt = cfg.get("type", "linear").lower()
    out_dim = _compute_out_dim(lt, in_dim, cfg)
    total = 0

    if lt == "linear":
        h = int(cfg.get("hidden_dim", 256))
        total += in_dim * h + h
        if cfg.get("batchnorm"):
            total += 2 * h
    elif lt == "cnn1d":
        out_ch = int(cfg.get("out_channels", 64))
        k = int(cfg.get("kernel_size", 3))
        total += out_ch * k + out_ch
    elif lt == "bilstm":
        h = int(cfg.get("hidden_size", 128))
        nl = int(cfg.get("num_layers", 1))
        gate = 4
        total += gate * (in_dim * h + h * h + h)
        total += gate * (in_dim * h + h * h + h)
        for _ in range(nl - 1):
            total += 2 * gate * (2 * h * h + h)
    elif lt == "gru":
        h = int(cfg.get("hidden_size", 128))
        nl = int(cfg.get("num_layers", 1))
        bidir = bool(cfg.get("bidirectional", True))
        gate = 3
        dirs = 2 if bidir else 1
        total += dirs * gate * (in_dim * h + h * h + 2 * h)
        for _ in range(nl - 1):
            total += dirs * gate * (dirs * h * h + h * h + 2 * h)
    elif lt == "transformer":
        d = int(cfg.get("d_model", 256))
        ff = int(cfg.get("dim_feedforward", d * 2))
        nl = int(cfg.get("num_layers", 2))
        total += in_dim * d + d
        total += nl * (4 * d * d + 4 * d + d * ff + ff + ff * d + d + 4 * d)
    elif lt == "residual":
        h = int(cfg.get("hidden_dim", 256))
        total += in_dim * h + h + h * in_dim + in_dim
        if cfg.get("batchnorm"):
            total += 2 * h
        total += 2 * in_dim

    return total, out_dim


def _layer_details(cfg: dict) -> str:
    lt = cfg.get("type", "linear").lower()
    if lt == "linear":
        parts = [
            f"hidden={cfg.get('hidden_dim', 256)}",
            f"act={cfg.get('activation', 'relu')}",
            f"dropout={cfg.get('dropout', 0)}",
        ]
        if cfg.get("batchnorm"):
            parts.append("batchnorm")
        return " | ".join(parts)
    if lt == "cnn1d":
        return (
            f"channels={cfg.get('out_channels', 64)} | "
            f"kernel={cfg.get('kernel_size', 3)} | "
            f"dropout={cfg.get('dropout', 0)}"
        )
    if lt == "bilstm":
        return (
            f"hidden={cfg.get('hidden_size', 128)} | "
            f"layers={cfg.get('num_layers', 1)} | "
            f"dropout={cfg.get('dropout', 0)}"
        )
    if lt == "gru":
        direction = "bidirectional" if cfg.get("bidirectional", True) else "single"
        return (
            f"hidden={cfg.get('hidden_size', 128)} | "
            f"layers={cfg.get('num_layers', 1)} | {direction}"
        )
    if lt == "transformer":
        return (
            f"d_model={cfg.get('d_model', 256)} | "
            f"heads={cfg.get('nhead', 4)} | "
            f"layers={cfg.get('num_layers', 2)} | "
            f"ff={cfg.get('dim_feedforward', 512)}"
        )
    if lt == "residual":
        return (
            f"hidden={cfg.get('hidden_dim', 256)} | "
            f"act={cfg.get('activation', 'relu')} | "
            f"dropout={cfg.get('dropout', 0)}"
        )
    return ""


def _pretty_type(layer_type: str) -> str:
    names = {
        "cnn1d": "CNN1D",
        "bilstm": "BiLSTM",
        "gru": "GRU",
        "linear": "Linear",
        "transformer": "Transformer",
        "residual": "Residual",
    }
    return names.get(layer_type.lower(), layer_type.title())


def _node_title(title: str, lines: list[str]) -> str:
    parts = [title, *[line for line in lines if line]]
    return "\n".join(str(part) for part in parts)


def _build_nodes(layer_configs: list, input_dim: int, input_label: str, input_subtitle: str) -> list[dict]:
    nodes = [{
        "id": "input",
        "label": f"{input_label}\n{input_dim:,} dim",
        "title": _node_title(input_label, [input_subtitle, f"{input_dim:,} features"]),
        "kind": "input",
    }]

    cur = input_dim
    for idx, cfg in enumerate(layer_configs, start=1):
        lt = cfg.get("type", "linear").lower()
        layer_params, out_dim = _layer_param_count(cur, cfg)
        layer_name = f"Layer {idx}: {_pretty_type(lt)}"
        nodes.append({
            "id": f"layer_{idx}",
            "label": f"{_pretty_type(lt)}\n{cur:,} -> {out_dim:,}",
            "title": _node_title(
                layer_name,
                [
                    _layer_details(cfg),
                    f"input: {cur:,}",
                    f"output: {out_dim:,}",
                    f"estimated params: {layer_params:,}",
                ],
            ),
            "kind": lt,
        })
        cur = out_dim

    nodes.append({
        "id": "output",
        "label": "Output\n1 logit",
        "title": _node_title("Output", [f"{cur:,} -> 1", "sigmoid(logit)"]),
        "kind": "output",
    })
    return nodes


def _render_pyvis(nodes: list[dict], height: int, key: str) -> bool:
    try:
        from pyvis.network import Network
    except Exception:
        return False

    net = Network(
        height=f"{height}px",
        width="100%",
        directed=True,
        bgcolor="#FFFFFF",
        font_color="#0F172A",
        cdn_resources="in_line",
    )

    for node in nodes:
        bg, border = _TYPE_STYLES.get(node["kind"], ("#F8FAFC", "#64748B"))
        net.add_node(
            node["id"],
            label=node["label"],
            title=node["title"],
            shape="box",
            color={"background": bg, "border": border, "highlight": {"background": bg, "border": "#111827"}},
            font={"face": "Arial", "size": 16, "color": "#0F172A", "multi": "md"},
            margin=12,
            borderWidth=2,
            shadow={"enabled": True, "color": "rgba(15, 23, 42, 0.14)", "size": 8, "x": 0, "y": 3},
            widthConstraint={"minimum": 150, "maximum": 230},
            heightConstraint={"minimum": 72},
        )

    for left, right in zip(nodes, nodes[1:]):
        net.add_edge(left["id"], right["id"])

    options = {
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "LR",
                "sortMethod": "directed",
                "levelSeparation": 215,
                "nodeSpacing": 135,
            }
        },
        "interaction": {
            "hover": True,
            "navigationButtons": True,
            "keyboard": True,
            "dragNodes": False,
            "dragView": True,
            "zoomView": True,
        },
        "physics": {"enabled": False},
        "edges": {
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.8}},
            "color": {"color": "#94A3B8", "highlight": "#2563EB"},
            "smooth": {"enabled": True, "type": "cubicBezier", "forceDirection": "horizontal", "roundness": 0.35},
            "width": 2,
        },
    }
    net.set_options(json.dumps(options))
    tooltip_css = """
<style>
  html,
  body {
    height: 100%;
    margin: 0;
    overflow: hidden;
    width: 100%;
  }
  .vis-tooltip {
    white-space: pre-line;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.16);
    color: #0F172A;
    font-family: Arial, sans-serif;
    font-size: 13px;
    line-height: 1.45;
    padding: 10px 12px;
  }
  #mynetwork {
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px;
    box-sizing: border-box;
    height: 100vh !important;
    width: 100vw !important;
  }
  .vis-network {
    height: 100% !important;
    width: 100% !important;
  }
</style>
"""
    viewport_script = """
<script>
  setTimeout(function () {
    if (typeof network !== "undefined" && typeof nodes !== "undefined") {
      const nodeIds = nodes.getIds();
      const visibleIds = nodeIds.slice(0, Math.min(nodeIds.length, 5));
      network.fit({ nodes: visibleIds, animation: false });
    }
  }, 150);
</script>
"""
    graph_html = net.generate_html(notebook=False).replace("</head>", f"{tooltip_css}</head>", 1).replace(
        "<body>",
        '<body>',
        1,
    ).replace("</body>", f"{viewport_script}</body>", 1)
    components.html(graph_html, height=height, scrolling=False)
    return True


def _dot_label(title: str, subtitle: str, fill: str, border: str) -> str:
    return f"""<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8" COLOR="{border}">
  <TR><TD BGCOLOR="{fill}"><B>{html.escape(title)}</B></TD></TR>
  <TR><TD>{html.escape(subtitle)}</TD></TR>
</TABLE>
>"""


def _render_graphviz(nodes: list[dict]) -> None:
    lines = [
        "digraph Architecture {",
        '  graph [rankdir="LR", bgcolor="transparent", pad="0.25", nodesep="0.55", ranksep="0.75"];',
        '  node [shape=plain, fontname="Arial"];',
        '  edge [color="#94A3B8", arrowsize=0.7, penwidth=1.8];',
    ]
    for node in nodes:
        bg, border = _TYPE_STYLES.get(node["kind"], ("#F8FAFC", "#64748B"))
        label_lines = node["label"].splitlines()
        title = label_lines[0]
        subtitle = " | ".join(label_lines[1:])
        lines.append(f'  {node["id"]} [label={_dot_label(title, subtitle, bg, border)}];')
    for left, right in zip(nodes, nodes[1:]):
        lines.append(f'  {left["id"]} -> {right["id"]};')
    lines.append("}")
    st.graphviz_chart("\n".join(lines), use_container_width=True)


def render_architecture_graph(
    layer_configs: list,
    input_dim: int,
    input_label: str,
    input_subtitle: str,
    key: str,
    height: int = 520,
) -> None:
    nodes = _build_nodes(layer_configs, input_dim, input_label, input_subtitle)
    if not _render_pyvis(nodes, height=height, key=key):
        _render_graphviz(nodes)
