"""Helpers for interaction hub network rendering."""

from __future__ import annotations


MAX_CYTOSCAPE_NODES = 350
MAX_CYTOSCAPE_EDGES = 800


def graph_size_warning(node_count: int, edge_count: int) -> str | None:
    if node_count > MAX_CYTOSCAPE_NODES or edge_count > MAX_CYTOSCAPE_EDGES:
        return (
            f"The filtered hub graph has {node_count:,} nodes and {edge_count:,} edges. "
            f"Interactive rendering is capped at {MAX_CYTOSCAPE_NODES:,} nodes and "
            f"{MAX_CYTOSCAPE_EDGES:,} edges. Increase the network threshold or minimum "
            "hub degree, or reduce top hubs / partners per hub."
        )
    return None


def cytoscape_elements_from_payload(payload: dict, node_type_fn, short_value_fn) -> list[dict]:
    selected_hubs = set(payload["selected_hubs"])
    displayed_nodes = payload["displayed_nodes"]
    displayed_edges = payload["displayed_edges"]
    node_ids = payload["node_ids"]
    degrees = payload["degrees"]
    weighted = payload["weighted"]
    cfg = payload["cfg"]

    elements = []
    for node in sorted(displayed_nodes, key=lambda item: (-int(item in selected_hubs), node_ids[item])):
        node_id = node_ids[node]
        role = "Hub" if node in selected_hubs else "Partner"
        node_type = node_type_fn(node, cfg)
        value = node[1]
        elements.append({
            "data": {
                "id": node_id,
                "label": node_id,
                "role": role,
                "type": node_type,
                "degree": int(degrees.get(node, 0)),
                "weighted": round(float(weighted.get(node, 0.0)), 4),
                "length": len(value),
                "preview": short_value_fn(value, 72),
                "size": (38 if role == "Hub" else 22) + min(int(degrees.get(node, 0)), 28),
            },
            "classes": "hub" if role == "Hub" else "partner",
        })

    for idx, edge in enumerate(displayed_edges, start=1):
        source = node_ids[edge["source"]]
        target = node_ids[edge["target"]]
        probability = float(edge["probability"])
        elements.append({
            "data": {
                "id": f"e{idx}",
                "source": source,
                "target": target,
                "probability": round(probability, 4),
                "width": 1.5 + 5.0 * max(0.0, min(1.0, probability)),
            },
            "classes": "interaction",
        })

    return elements
