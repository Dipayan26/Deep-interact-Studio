import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
if str(FRONTEND) not in sys.path:
    sys.path.insert(0, str(FRONTEND))

from hub_network import (
    MAX_CYTOSCAPE_EDGES,
    MAX_CYTOSCAPE_NODES,
    cytoscape_elements_from_payload,
    graph_size_warning,
)


def _node_type(node, _cfg):
    return "Protein" if node[0] == "protein" else node[0]


def _short(value, width):
    return value[:width]


def test_cytoscape_elements_assign_hub_and_partner_roles():
    hub = ("protein", "AAAA")
    partner = ("protein", "BBBB")
    payload = {
        "selected_hubs": [hub],
        "displayed_nodes": {hub, partner},
        "displayed_edges": [{"source": hub, "target": partner, "probability": 0.87}],
        "node_ids": {hub: "P1", partner: "P2"},
        "degrees": {hub: 12, partner: 1},
        "weighted": {hub: 9.1, partner: 0.87},
        "cfg": {"merge_sides": True},
    }

    elements = cytoscape_elements_from_payload(payload, _node_type, _short)
    nodes = [element for element in elements if "source" not in element["data"]]
    edges = [element for element in elements if "source" in element["data"]]

    assert nodes[0]["classes"] == "hub"
    assert nodes[0]["data"]["role"] == "Hub"
    assert nodes[1]["classes"] == "partner"
    assert nodes[1]["data"]["role"] == "Partner"
    assert edges == [{
        "data": {
            "id": "e1",
            "source": "P1",
            "target": "P2",
            "probability": 0.87,
            "width": 5.85,
        },
        "classes": "interaction",
    }]


def test_graph_size_warning_blocks_oversized_graphs():
    assert graph_size_warning(MAX_CYTOSCAPE_NODES, MAX_CYTOSCAPE_EDGES) is None
    assert "capped" in graph_size_warning(MAX_CYTOSCAPE_NODES + 1, 1)
    assert "capped" in graph_size_warning(1, MAX_CYTOSCAPE_EDGES + 1)
