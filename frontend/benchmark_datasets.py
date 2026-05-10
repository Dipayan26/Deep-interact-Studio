from pathlib import Path
import csv

import streamlit as st


def _resolve_data_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parents[1] / "gold_std_data",
        Path("/gold_std_data"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


DATA_DIR = _resolve_data_dir()


# Add new benchmark files by appending another dictionary to the relevant list.
DATASET_CATEGORIES = {
    "PPI": {
        "title": "PPI",
        "subtitle": "Protein-Protein Interaction",
        "columns": "ProteinA, ProteinB, lable",
        "datasets": [
            {
                "name": "Human PPI Benchmark",
                "file": "PPI-processed.csv",
                "paper": "Large-Scale Prediction of Human Protein-Protein Interactions from Amino Acid Sequence Based on Latent Topic Features",
                "citation": "Pan, Zhang, and Shen, Journal of Proteome Research, 2010.",
                "description": "Human protein pairs labeled as interacting or non-interacting.",
            }
        ],
    },
    "DTPI": {
        "title": "DTPI",
        "subtitle": "Drug-Target Protein Interaction",
        "columns": "SMILES, Protein, lable",
        "datasets": [
            {
                "name": "NASNet-DTI Benchmark",
                "file": "DTPI-processed.csv",
                "paper": "NASNet-DTI: accurate drug-target interaction prediction using heterogeneous graphs and node adaptation",
                "citation": "Zhong and Du, Briefings in Bioinformatics, 2025.",
                "description": "Drug SMILES and protein sequences labeled for interaction prediction.",
            }
        ],
    },
    "RPI": {
        "title": "RPI",
        "subtitle": "RNA-Protein Interaction",
        "columns": "RNA, Protein, lable",
        "datasets": [
            {
                "name": "RNA-Protein Benchmark",
                "file": "RPI-processed.csv",
                "paper": "RNA-protein interaction prediction using network-guided deep learning",
                "citation": "Liu, Jian, Zeng, and Zhao, Communications Biology, 2025.",
                "description": "RNA and protein sequence pairs labeled as interacting or non-interacting.",
            }
        ],
    },
    "PDI": {
        "title": "PDI",
        "subtitle": "Protein-DNA Interaction",
        "columns": "DNA, Protein, lable",
        "datasets": [],
        "empty_message": "Will be added later.",
    },
}


@st.cache_data(show_spinner=False)
def _dataset_summary(filename: str) -> dict:
    path = DATA_DIR / filename
    if not path.exists():
        return {"rows": "Not found", "positive": "-", "negative": "-", "size": "-"}

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    label_counts = {"0": 0, "1": 0}
    for row in rows:
        label = str(row.get("lable", row.get("label", ""))).strip()
        if label in label_counts:
            label_counts[label] += 1

    return {
        "rows": f"{len(rows):,}",
        "positive": f"{label_counts['1']:,}",
        "negative": f"{label_counts['0']:,}",
        "size": f"{path.stat().st_size / (1024 * 1024):.1f} MB",
    }


def _download_button(filename: str, label: str) -> None:
    path = DATA_DIR / filename
    key = f"benchmark-download-{filename}"
    if not path.exists():
        st.button(label, disabled=True, key=key, use_container_width=True)
        return

    st.download_button(
        label,
        data=path.read_bytes(),
        file_name=filename,
        mime="text/csv",
        key=key,
        use_container_width=True,
    )


def _dataset_card(dataset: dict, expected_columns: str) -> None:
    summary = _dataset_summary(dataset["file"])

    with st.container(border=True):
        title_col, action_col = st.columns([4, 1])
        with title_col:
            st.subheader(dataset["name"])
            st.caption(dataset["description"])
        with action_col:
            _download_button(dataset["file"], "Download CSV")

        metric_cols = st.columns(4)
        metric_cols[0].metric("Pairs", summary["rows"])
        metric_cols[1].metric("Positive", summary["positive"])
        metric_cols[2].metric("Negative", summary["negative"])
        metric_cols[3].metric("File size", summary["size"])

        st.markdown(f"**File:** `{dataset['file']}`")
        st.markdown(f"**Columns:** `{expected_columns}`")
        st.markdown(f"**Paper:** {dataset['paper']}")
        st.caption(dataset["citation"])


st.title("Benchmark Datasets")
st.caption("Small curated benchmark files bundled with the app.")

st.markdown(
    """
Use these CSV files directly in the matching model-building page. The label column is named
`lable` in the bundled files, where `1` means interaction and `0` means no interaction.
"""
)

for category in DATASET_CATEGORIES.values():
    st.divider()
    st.header(category["title"])
    st.caption(category["subtitle"])
    st.markdown(f"**Expected columns:** `{category['columns']}`")

    if not category["datasets"]:
        st.info(category.get("empty_message", "No dataset available yet."))
        continue

    for dataset in category["datasets"]:
        _dataset_card(dataset, category["columns"])
