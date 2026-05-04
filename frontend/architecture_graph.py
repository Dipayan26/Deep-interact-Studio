import base64
import html as _html
import json
import math

import streamlit as st
import streamlit.components.v1 as components


_TYPE_COLORS = {
    "linear":      ("#2563EB", "#EAF2FF"),
    "cnn1d":       ("#059669", "#ECFDF5"),
    "bilstm":      ("#EA580C", "#FFF7ED"),
    "gru":         ("#7C3AED", "#F5F3FF"),
    "transformer": ("#DC2626", "#FEF2F2"),
    "residual":    ("#475569", "#F8FAFC"),
    "input":       ("#1D4ED8", "#EFF6FF"),
    "output":      ("#16A34A", "#F0FDF4"),
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


def _pretty_type(layer_type: str) -> str:
    return {
        "cnn1d": "CNN1D", "bilstm": "BiLSTM", "gru": "GRU",
        "linear": "Linear", "transformer": "Transformer",
        "residual": "Residual", "input": "Input", "output": "Output",
    }.get(layer_type.lower(), layer_type.title())


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


# ---------------------------------------------------------------------------
# SVG node generation
# ---------------------------------------------------------------------------

_NODE_W = 175
_HEADER_H = 36
_ROW_H = 19
_PX = 10
_FOOTER_H = 22


def _param_lines(lt: str, cfg: dict) -> list[tuple[str, str]]:
    if lt == "linear":
        rows = [("hidden_dim", str(cfg.get("hidden_dim", 256))),
                ("activation", str(cfg.get("activation", "relu"))),
                ("dropout",    f"{float(cfg.get('dropout', 0.3)):.2f}")]
        if cfg.get("batchnorm"):
            rows.append(("batchnorm", "True"))
        return rows
    if lt == "residual":
        rows = [("hidden_dim", str(cfg.get("hidden_dim", 256))),
                ("activation", str(cfg.get("activation", "relu"))),
                ("dropout",    f"{float(cfg.get('dropout', 0.3)):.2f}")]
        if cfg.get("batchnorm"):
            rows.append(("batchnorm", "True"))
        return rows
    if lt == "cnn1d":
        return [("out_channels", str(cfg.get("out_channels", 64))),
                ("kernel_size",  str(cfg.get("kernel_size", 3))),
                ("activation",   str(cfg.get("activation", "relu"))),
                ("dropout",      f"{float(cfg.get('dropout', 0.3)):.2f}")]
    if lt == "bilstm":
        return [("hidden_size", str(cfg.get("hidden_size", 128))),
                ("num_layers",  str(cfg.get("num_layers", 1))),
                ("dropout",     f"{float(cfg.get('dropout', 0.3)):.2f}")]
    if lt == "gru":
        return [("hidden_size",   str(cfg.get("hidden_size", 128))),
                ("num_layers",    str(cfg.get("num_layers", 1))),
                ("bidirectional", str(cfg.get("bidirectional", True))),
                ("dropout",       f"{float(cfg.get('dropout', 0.3)):.2f}")]
    if lt == "transformer":
        return [("d_model",    str(cfg.get("d_model", 256))),
                ("nhead",      str(cfg.get("nhead", 4))),
                ("num_layers", str(cfg.get("num_layers", 2))),
                ("ff_dim",     str(cfg.get("dim_feedforward", 512))),
                ("dropout",    f"{float(cfg.get('dropout', 0.1)):.2f}")]
    return []


def _hidden_dim_for_scale(lt: str, cfg: dict) -> int:
    if lt in ("linear", "residual"):
        return int(cfg.get("hidden_dim", 256))
    if lt in ("bilstm", "gru"):
        return int(cfg.get("hidden_size", 128))
    if lt == "cnn1d":
        return int(cfg.get("out_channels", 64))
    if lt == "transformer":
        return int(cfg.get("d_model", 256))
    return 128


def _make_svg(lt: str, cfg: dict, in_dim: int, out_dim: int, layer_idx=None) -> tuple[str, int, int]:
    """Return (base64 data URL, width, height)."""
    header_color, bg_color = _TYPE_COLORS.get(lt, ("#475569", "#F8FAFC"))
    W = _NODE_W

    # --- input node: scale with in_dim ---
    if lt == "input":
        scale = max(0.7, min(2.5, 0.5 + math.log2(max(in_dim, 32) / 128 + 1) * 0.75))
        extra = int(scale * 50)
        H = _HEADER_H + 14 + extra + _FOOTER_H + 8
        mid_y = _HEADER_H + 14 + extra // 2 + 4
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">'
            f'<rect x="1" y="1" width="{W-2}" height="{H-2}" rx="8" fill="{bg_color}" stroke="{header_color}" stroke-width="2"/>'
            f'<rect x="1" y="1" width="{W-2}" height="{_HEADER_H}" rx="8" fill="{header_color}"/>'
            f'<rect x="1" y="{_HEADER_H-8}" width="{W-2}" height="8" fill="{header_color}"/>'
            f'<text x="{W//2}" y="23" text-anchor="middle" '
            f'font-family="Arial,sans-serif" font-size="13" font-weight="bold" fill="white">Input</text>'
            f'<text x="{W//2}" y="{mid_y}" text-anchor="middle" '
            f'font-family="Arial,sans-serif" font-size="20" font-weight="bold" fill="{header_color}">{in_dim:,}</text>'
            f'<text x="{W//2}" y="{mid_y + 18}" text-anchor="middle" '
            f'font-family="Arial,sans-serif" font-size="11" fill="#64748B">dimensions</text>'
            f'<line x1="{_PX}" y1="{H-_FOOTER_H}" x2="{W-_PX}" y2="{H-_FOOTER_H}" stroke="#CBD5E1" stroke-width="1"/>'
            f'<text x="{W//2}" y="{H-7}" text-anchor="middle" '
            f'font-family="Arial,sans-serif" font-size="10" font-style="italic" fill="#94A3B8">{_html.escape(cfg.get("subtitle", ""))}</text>'
            f'</svg>'
        )
        return f"data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}", W, H

    # --- output node: fixed small ---
    if lt == "output":
        H = 72
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">'
            f'<rect x="1" y="1" width="{W-2}" height="{H-2}" rx="8" fill="{bg_color}" stroke="{header_color}" stroke-width="2"/>'
            f'<rect x="1" y="1" width="{W-2}" height="{_HEADER_H}" rx="8" fill="{header_color}"/>'
            f'<rect x="1" y="{_HEADER_H-8}" width="{W-2}" height="8" fill="{header_color}"/>'
            f'<text x="{W//2}" y="23" text-anchor="middle" '
            f'font-family="Arial,sans-serif" font-size="13" font-weight="bold" fill="white">Output</text>'
            f'<text x="{W//2}" y="{_HEADER_H + 22}" text-anchor="middle" '
            f'font-family="Arial,sans-serif" font-size="12" fill="#334155">1 logit</text>'
            f'</svg>'
        )
        return f"data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}", W, H

    # --- scale height with hidden dim ---
    hd = _hidden_dim_for_scale(lt, cfg)
    # log2 scale: 32→min, 128→base, 2048→max
    scale = max(0.5, min(2.2, 0.5 + math.log2(max(hd, 32) / 128 + 1) * 0.65))

    rows = _param_lines(lt, cfg)
    extra = int(scale * 30)
    H = _HEADER_H + 10 + extra + len(rows) * _ROW_H + _FOOTER_H + 8

    if lt == "residual":
        h_inner = cfg.get("hidden_dim", 256)
        dim_text = f"{in_dim} → [{h_inner}] → {out_dim}"
    else:
        dim_text = f"{in_dim:,} → {out_dim:,}"

    label = _pretty_type(lt)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
        # body background
        f'<rect x="1" y="1" width="{W-2}" height="{H-2}" rx="8" fill="{bg_color}" stroke="{header_color}" stroke-width="2"/>',
        # header rectangle (rounded top only — square bottom seam covered below)
        f'<rect x="1" y="1" width="{W-2}" height="{_HEADER_H}" rx="8" fill="{header_color}"/>',
        f'<rect x="1" y="{_HEADER_H-8}" width="{W-2}" height="8" fill="{header_color}"/>',
        # layer type name
        f'<text x="{W//2}" y="16" text-anchor="middle" '
        f'font-family="Arial,sans-serif" font-size="13" font-weight="bold" fill="white">{_html.escape(label)}</text>',
    ]

    if layer_idx is not None:
        parts.append(
            f'<text x="{W//2}" y="30" text-anchor="middle" '
            f'font-family="Arial,sans-serif" font-size="10" fill="rgba(255,255,255,0.75)">Layer {layer_idx}</text>'
        )

    # separator below header
    sep_y = _HEADER_H + 8
    parts.append(f'<line x1="{_PX}" y1="{sep_y}" x2="{W-_PX}" y2="{sep_y}" stroke="#CBD5E1" stroke-width="1"/>')

    # param rows
    y = sep_y + _ROW_H
    for k, v in rows:
        parts.append(
            f'<text x="{_PX}" y="{y}" font-family="Arial,sans-serif" font-size="11" fill="#1E293B">'
            f'<tspan font-weight="600">{_html.escape(k)}</tspan>'
            f'<tspan fill="#64748B"> = {_html.escape(str(v))}</tspan>'
            f'</text>'
        )
        y += _ROW_H

    # footer separator + dim line
    foot_y = H - _FOOTER_H
    parts.extend([
        f'<line x1="{_PX}" y1="{foot_y}" x2="{W-_PX}" y2="{foot_y}" stroke="#CBD5E1" stroke-width="1"/>',
        f'<text x="{W//2}" y="{H-7}" text-anchor="middle" '
        f'font-family="Arial,sans-serif" font-size="10" font-style="italic" fill="#94A3B8">{_html.escape(dim_text)}</text>',
    ])

    parts.append('</svg>')
    b64 = base64.b64encode(''.join(parts).encode()).decode()
    return f"data:image/svg+xml;base64,{b64}", W, H


# ---------------------------------------------------------------------------
# Node list builder
# ---------------------------------------------------------------------------

def _tooltip(title: str, lines: list[str]) -> str:
    return "\n".join([title] + [l for l in lines if l])


def _build_nodes(layer_configs: list, input_dim: int, input_label: str, input_subtitle: str) -> list[dict]:
    nodes = []

    svg_url, w, h = _make_svg("input", {"subtitle": input_subtitle}, input_dim, input_dim)
    nodes.append({
        "id": "input", "svg_url": svg_url, "w": w, "h": h,
        "title": _tooltip(input_label, [input_subtitle, f"{input_dim:,} features"]),
    })

    cur = input_dim
    for idx, cfg in enumerate(layer_configs, start=1):
        lt = cfg.get("type", "linear").lower()
        layer_params, out_dim = _layer_param_count(cur, cfg)
        svg_url, w, h = _make_svg(lt, cfg, cur, out_dim, idx)
        nodes.append({
            "id": f"layer_{idx}", "svg_url": svg_url, "w": w, "h": h,
            "title": _tooltip(
                f"Layer {idx}: {_pretty_type(lt)}",
                [f"input: {cur:,}", f"output: {out_dim:,}", f"~{layer_params:,} params"],
            ),
        })
        cur = out_dim

    svg_url, w, h = _make_svg("output", {}, cur, 1)
    nodes.append({
        "id": "output", "svg_url": svg_url, "w": w, "h": h,
        "title": _tooltip("Output", [f"{cur:,} → 1", "sigmoid(logit)"]),
    })

    return nodes


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

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
        net.add_node(
            node["id"],
            shape="image",
            image=node["svg_url"],
            label="",
            title=node["title"],
            shapeProperties={"useImageSize": True, "interpolation": False},
        )

    for left, right in zip(nodes, nodes[1:]):
        net.add_edge(left["id"], right["id"])

    max_h = max(n["h"] for n in nodes)
    options = {
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "LR",
                "sortMethod": "directed",
                "levelSeparation": _NODE_W + 60,
                "nodeSpacing": max_h + 40,
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
            "smooth": {"enabled": True, "type": "cubicBezier",
                       "forceDirection": "horizontal", "roundness": 0.4},
            "width": 2,
        },
    }
    net.set_options(json.dumps(options))

    tooltip_css = """
<style>
  html, body { height: 100%; margin: 0; overflow: hidden; width: 100%; }
  .vis-tooltip {
    white-space: pre-line;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
    box-shadow: 0 12px 28px rgba(15,23,42,0.16);
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
  .vis-network { height: 100% !important; width: 100% !important; }
</style>
"""
    viewport_script = """
<script>
  setTimeout(function () {
    if (typeof network !== "undefined" && typeof nodes !== "undefined") {
      network.fit({ animation: false });
    }
  }, 200);
</script>
"""
    graph_html = (
        net.generate_html(notebook=False)
        .replace("</head>", f"{tooltip_css}</head>", 1)
        .replace("</body>", f"{viewport_script}</body>", 1)
    )
    components.html(graph_html, height=height, scrolling=False)
    return True


def _render_graphviz(nodes: list[dict]) -> None:
    import html as h
    lines = [
        "digraph Architecture {",
        '  graph [rankdir="LR", bgcolor="transparent", pad="0.25", nodesep="0.55", ranksep="0.75"];',
        '  node [shape=plain, fontname="Arial"];',
        '  edge [color="#94A3B8", arrowsize=0.7, penwidth=1.8];',
    ]
    for node in nodes:
        lt = node["id"].split("_")[0] if "_" in node["id"] else node["id"]
        header_color, bg_color = _TYPE_COLORS.get(lt, ("#475569", "#F8FAFC"))
        title_line = node["title"].splitlines()[0]
        sub_lines = " | ".join(node["title"].splitlines()[1:])
        label = (
            f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8" COLOR="{header_color}">'
            f'<TR><TD BGCOLOR="{header_color}"><B><FONT COLOR="white">{h.escape(title_line)}</FONT></B></TD></TR>'
            f'<TR><TD>{h.escape(sub_lines)}</TD></TR>'
            f'</TABLE>>'
        )
        lines.append(f'  {node["id"]} [label={label}];')
    for left, right in zip(nodes, nodes[1:]):
        lines.append(f'  {left["id"]} -> {right["id"]};')
    lines.append("}")
    st.graphviz_chart("\n".join(lines), use_container_width=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

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
