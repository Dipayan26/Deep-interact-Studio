import streamlit as st
import plotly.io as pio

st.set_page_config(
    page_title="Deep-Interact Studio",
    page_icon=":material/biotech:",
    layout="wide",
)


st.session_state["theme_mode"] = "Light"

def _qp_scalar(v):
    if isinstance(v, list):
        return str(v[0]).strip() if v else ""
    return str(v).strip() if v is not None else ""

qp_goto = _qp_scalar(st.query_params.get("goto"))
qp_run_id = _qp_scalar(st.query_params.get("run_id"))

if qp_goto == "check_results":
    if qp_run_id:
        st.session_state["last_run_id"] = qp_run_id
        st.session_state["active_rid"] = qp_run_id
    st.query_params.clear()
    st.switch_page("check_results.py")

st.session_state["plotly_template"] = "plotly_white"
pio.templates.default = st.session_state["plotly_template"]
st.markdown("""
    <style>
        /* Hide header and toolbar completely */
        [data-testid="stHeader"] {
            display: none;
        }

        [data-testid="stToolbar"] {
            display: none;
        }

        /* Optional: remove top padding caused by header */
        .block-container {
            padding-top: 0rem;
        }

        /* Your existing styles */
        [data-testid="stAppViewContainer"] {
            background: #f8fafc;
            color: #0f172a;
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            color: #0f172a;
        }
    </style>
""", unsafe_allow_html=True)

pg = st.navigation({
    "": [
        st.Page("home.py",       title="Home",       icon=":material/home:"),
        st.Page("manual.py",     title="Manual",     icon=":material/menu_book:"),
        st.Page("contact.py",    title="Contact Us", icon=":material/contact_mail:"),
    ],
    "Data": [
        st.Page("benchmark_datasets.py", title="Benchmark Datasets", icon=":material/dataset:"),
    ],
    "Model Building": [
        st.Page("ppi.py",              title="PPI Prediction(PPI)",           icon=":material/hub:"),
        st.Page("dtpi.py",             title="Drug-Target Protein Interaction(DTPI)", icon=":material/medication:"),
        st.Page("rna_prot.py",         title="RNA–Protein Interaction(RPI)",   icon=":material/genetics:"),
        st.Page("prot_dna.py",         title="Protein–DNA Interaction(PDI)",   icon=":material/biotech:"),
    ],
    "Tools": [
        st.Page("job_status.py",            title="Job Status",               icon=":material/list_alt:"),
        st.Page("check_results.py",         title="Check Model Results",      icon=":material/monitor_heart:"),
        st.Page("inference.py",             title="Run Inference",            icon=":material/play_arrow:"),
        st.Page("inference_results.py",     title="Inference Results",        icon=":material/analytics:"),
        st.Page("comparison.py",            title="Multi-Model Comparison",   icon=":material/compare:"),
        st.Page("inference_comparison.py",  title="Multi-Model Inference",    icon=":material/difference:"),
    ],
})
pg.run()
