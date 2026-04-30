import os
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

_IST = timezone(timedelta(hours=5, minutes=30))

def _fmt_ist(ts_str) -> str:
    """Convert a UTC ISO timestamp string to IST and format as YYYY-MM-DD HH:MM:SS."""
    if not ts_str or ts_str == "—":
        return "—"
    try:
        dt = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_IST).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts_str)[:19].replace("T", " ")

BACKEND = os.getenv("BACKEND_URL", "http://backend:8005")
is_dark = st.session_state.get("theme_mode", "Light") == "Dark"

st.title("Job Status")
st.caption("All submitted jobs — training and inference.")

st.divider()

st.session_state.setdefault("js_page", 0)
st.session_state.setdefault("js_page_size", 50)
st.session_state.setdefault("js_status_filter", [])
st.session_state.setdefault("js_job_type_filter", "all")
st.session_state.setdefault("js_task_type_filter", "all")
st.session_state.setdefault("js_run_contains", "")
st.session_state.setdefault("js_filter_sig", None)
st.session_state.setdefault("js_cmp_selected", [])
st.session_state.setdefault("js_cmp_task_map", {})

def _reset_job_status_filters():
    st.session_state["js_status_filter"] = []
    st.session_state["js_job_type_filter"] = "all"
    st.session_state["js_task_type_filter"] = "all"
    st.session_state["js_run_contains"] = ""
    st.session_state["js_page_size"] = 50
    st.session_state["js_page"] = 0
    st.session_state["js_filter_sig"] = None

status_opts = ["queued", "running", "completed", "failed", "cancelled"]
page_size_opts = [25, 50, 100, 200]

f1, f2, f3, f4 = st.columns([2, 1.2, 1.2, 2])
with f1:
    status_filter = st.multiselect(
        "Status",
        options=status_opts,
        default=st.session_state["js_status_filter"],
        key="js_status_filter",
    )
with f2:
    job_type_filter = st.selectbox(
        "Job Type",
        options=["all", "train", "inference"],
        index=["all", "train", "inference"].index(st.session_state["js_job_type_filter"]),
        key="js_job_type_filter",
    )
with f3:
    task_type_filter = st.selectbox(
        "Task Type",
        options=["all", "ppi", "dtpi", "rpi", "pdi"],
        index=["all", "ppi", "dtpi", "rpi", "pdi"].index(st.session_state["js_task_type_filter"]),
        key="js_task_type_filter",
    )
with f4:
    run_contains = st.text_input(
        "Run ID contains",
        value=st.session_state["js_run_contains"],
        placeholder="e.g. 3f2a",
        key="js_run_contains",
    )

c1, c2, _ = st.columns([1.2, 1.2, 4.6])
with c1:
    page_size = st.selectbox(
        "Rows per page",
        options=page_size_opts,
        index=page_size_opts.index(st.session_state["js_page_size"]),
        key="js_page_size",
    )
with c2:
    st.button("Reset Filters", on_click=_reset_job_status_filters)

current_sig = (
    tuple(sorted(status_filter)),
    job_type_filter,
    task_type_filter,
    run_contains.strip(),
    int(page_size),
)
if st.session_state["js_filter_sig"] != current_sig:
    st.session_state["js_filter_sig"] = current_sig
    st.session_state["js_page"] = 0

try:
    page = int(st.session_state["js_page"])
    limit = int(page_size)
    offset = page * limit

    query_params = {
        "limit": limit,
        "offset": offset,
    }
    if status_filter:
        query_params["status"] = ",".join(status_filter)
    if job_type_filter != "all":
        query_params["job_type"] = job_type_filter
    if task_type_filter != "all":
        query_params["task_type"] = task_type_filter
    if run_contains.strip():
        query_params["run_id_contains"] = run_contains.strip()

    r = requests.get(f"{BACKEND}/jobs", params=query_params, timeout=5)
    r.raise_for_status()
    jobs = r.json()

    if not jobs:
        if page > 0:
            st.warning("No rows on this page. Go to previous page.")
        else:
            st.info("No jobs found for the selected filters.")
        st.stop()

    df = pd.DataFrame(jobs)
    if "created_at" in df.columns:
        df = df.sort_values("created_at", ascending=False)

    # ── Job table with copy buttons on run IDs ───────────────────────────────
    STATUS_COLOUR = {
        "completed": "#34d399" if is_dark else "#1a9e6a",
        "running":   "#60a5fa" if is_dark else "#1a6bbf",
        "queued":    "#fbbf24" if is_dark else "#b07d12",
        "failed":    "#f87171" if is_dark else "#c0392b",
        "cancelled": "#9ca3af" if is_dark else "#888888",
    }

    if is_dark:
        TASK_LABEL = {
            "ppi": ("PPI",             "#bfdbfe", "#1e3a5f"),
            "dtpi": ("DTPI", "#99f6e4", "#134e4a"),
            "rpi": ("RPI", "#ddd6fe", "#4c1d95"),
            "pdi": ("PDI", "#fed7aa", "#7c2d12"),
        }
        row_bg = "#3a414b"
        row_sep = "rgba(203,213,225,0.20)"
        hdr_col = "#cfd6de"
        txt_col = "#e6e8eb"
        subtle_col = "#c7ced8"
        run_bg = "#4b5563"
        run_border = "#6b7280"
    else:
        TASK_LABEL = {
            "ppi": ("PPI",             "#1a5fa5", "#dceeff"),
            "dtpi": ("DTPI", "#2A7D6F", "#d6f5f0"),
            "rpi": ("RPI", "#7A3E9D", "#efe3fb"),
            "pdi": ("PDI", "#A05000", "#ffe8d3"),
        }
        row_bg = "#ffffff"
        row_sep = "rgba(15,23,42,0.10)"
        hdr_col = "#5b6470"
        txt_col = "#1f2937"
        subtle_col = "#4b5563"
        run_bg = "#f3f4f6"
        run_border = "#d1d5db"

    st.markdown("##### Jobs")
    st.markdown(
        f"""
        <style>
        .js-hdr {{
          color: {hdr_col};
          font-size: 0.78rem;
          font-weight: 600;
        }}
        .js-cell {{
          color: {txt_col};
          font-size: 0.88rem;
          line-height: 1.35;
        }}
        .js-run {{
          display: inline-block;
          font-family: monospace;
          font-size: 0.80rem;
          color: {txt_col};
          background: {run_bg};
          border: 1px solid {run_border};
          border-radius: 6px;
          padding: 2px 8px;
        }}
        .js-subtle {{
          color: {subtle_col};
          font-size: 0.82rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    # cache run_id -> task_type for selection validation across pages
    task_map = st.session_state.get("js_cmp_task_map", {})
    for _, r in df.iterrows():
        rid = str(r.get("run_id", "")).strip()
        tt = str(r.get("task_type", "")).strip().lower()
        if rid and tt:
            task_map[rid] = tt
    st.session_state["js_cmp_task_map"] = task_map

    def _selected_run_ids() -> list[str]:
        ids = []
        for k, v in st.session_state.items():
            if k.startswith("js_cmp_sel_") and bool(v):
                ids.append(k.replace("js_cmp_sel_", "", 1))
        # stable order
        return sorted(set(ids))

    selected_ids = _selected_run_ids()
    if len(selected_ids) > 5:
        keep = set(selected_ids[:5])
        for rid in selected_ids[5:]:
            st.session_state[f"js_cmp_sel_{rid}"] = False
        selected_ids = sorted(keep)

    st.session_state["js_cmp_selected"] = selected_ids
    selected_task_types = sorted({
        st.session_state["js_cmp_task_map"].get(rid, "")
        for rid in selected_ids
        if st.session_state["js_cmp_task_map"].get(rid, "")
    })
    mixed_task_selection = len(selected_task_types) > 1

    cc1, cc2, _ = st.columns([2.2, 1.2, 4.6])
    with cc1:
        st.caption("Select up to 5 completed training runs for model comparison (same task type only).")
    with cc2:
        if st.button(
            f"Compare Selected ({len(selected_ids)}/5)",
            use_container_width=True,
            type="primary",
            disabled=(len(selected_ids) < 2 or mixed_task_selection),
        ):
            st.session_state["cmp_run_ids"] = selected_ids[:5]
            st.session_state.pop("cmp_data", None)
            st.switch_page("comparison.py")

    if mixed_task_selection:
        st.warning(
            "Selected runs contain mixed task types "
            f"({', '.join(t.upper() for t in selected_task_types)}). "
            "Please select runs from only one task type (e.g., only PPI or only DTPI)."
        )

    h_sel, h_task, h_view, h_run, h_type, h_status, h_created, h_acc, h_auroc = st.columns(
        [0.7, 1.4, 0.8, 1.6, 1.0, 1.0, 1.8, 0.8, 0.8]
    )
    h_sel.markdown('<div class="js-hdr">Select</div>', unsafe_allow_html=True)
    h_task.markdown('<div class="js-hdr">Task</div>', unsafe_allow_html=True)
    h_view.markdown('<div class="js-hdr">View</div>', unsafe_allow_html=True)
    h_run.markdown('<div class="js-hdr">Run ID</div>', unsafe_allow_html=True)
    h_type.markdown('<div class="js-hdr">Job Type</div>', unsafe_allow_html=True)
    h_status.markdown('<div class="js-hdr">Status</div>', unsafe_allow_html=True)
    h_created.markdown('<div class="js-hdr">Submitted (IST)</div>', unsafe_allow_html=True)
    h_acc.markdown('<div class="js-hdr">Val Acc</div>', unsafe_allow_html=True)
    h_auroc.markdown('<div class="js-hdr">AUROC</div>', unsafe_allow_html=True)

    with st.container(height=560, border=True):
        for _, row in df.iterrows():
            rid     = str(row.get("run_id",    "—"))
            jtype   = str(row.get("job_type",  "—"))
            ttype   = str(row.get("task_type", "ppi")).lower()
            status  = str(row.get("status",    "—"))
            created = _fmt_ist(row.get("created_at", "—"))
            acc     = row.get("val_acc")
            auroc   = row.get("auroc")
            acc_s   = f"{acc:.4f}"   if acc   is not None else "—"
            auroc_s = f"{auroc:.4f}" if auroc is not None else "—"
            s_col   = STATUS_COLOUR.get(status, "#888")
            t_label, t_fg, t_bg = TASK_LABEL.get(ttype, (ttype.upper(), "#555", "#eee"))
            sel_key = f"js_cmp_sel_{rid}"
            eligible = (jtype == "train" and status == "completed")
            currently_selected = bool(st.session_state.get(sel_key, False))
            disable_select = (not currently_selected and len(selected_ids) >= 5)

            c_sel, c_task, c_view, c_run, c_type, c_status, c_created, c_acc, c_auroc = st.columns(
                [0.7, 1.4, 0.8, 1.6, 1.0, 1.0, 1.8, 0.8, 0.8]
            )
            if eligible:
                c_sel.checkbox(
                    "Select",
                    key=sel_key,
                    disabled=disable_select,
                    label_visibility="collapsed",
                )
            else:
                st.session_state[sel_key] = False
                c_sel.markdown('<span class="js-subtle">—</span>', unsafe_allow_html=True)

            c_task.markdown(
                f'<span style="display:inline-block;padding:2px 9px;border-radius:4px;'
                f'font-size:11px;font-weight:700;color:{t_fg};background:{t_bg};">{t_label}</span>',
                unsafe_allow_html=True,
            )
            if c_view.button("View", key=f"js_view_{rid}", use_container_width=True):
                st.session_state["last_run_id"] = rid
                st.session_state["active_rid"] = rid
                st.switch_page("check_results.py")
            c_run.markdown(f'<span class="js-run">{rid}</span>', unsafe_allow_html=True)
            c_type.markdown(f'<span class="js-cell">{jtype}</span>', unsafe_allow_html=True)
            c_status.markdown(
                f'<span style="display:inline-block;padding:2px 8px;border-radius:20px;'
                f'border:1px solid {s_col};color:{s_col};font-size:11px;font-weight:600;">{status}</span>',
                unsafe_allow_html=True,
            )
            c_created.markdown(f'<span class="js-subtle">{created}</span>', unsafe_allow_html=True)
            c_acc.markdown(f'<span class="js-cell">{acc_s}</span>', unsafe_allow_html=True)
            c_auroc.markdown(f'<span class="js-cell">{auroc_s}</span>', unsafe_allow_html=True)
            st.markdown(
                f"<hr style='margin:0.35rem 0; border-color:{row_sep};'>",
                unsafe_allow_html=True,
            )

    has_more = len(df) >= limit
    st.caption(
        f"Page {page + 1} · Showing {offset + 1}–{offset + len(df)} rows"
        + (" · More pages available" if has_more else "")
    )
    p1, p2, p3 = st.columns([1, 1, 5])
    with p1:
        if st.button("Previous", disabled=(page == 0), use_container_width=True):
            st.session_state["js_page"] = max(0, page - 1)
            st.rerun()
    with p2:
        if st.button("Next", disabled=(not has_more), use_container_width=True):
            st.session_state["js_page"] = page + 1
            st.rerun()

    # ── Detail panel ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Job Details")

    selected = st.selectbox("Select run ID", df["run_id"].tolist(), label_visibility="collapsed")

    if st.button("View details"):
        row = df[df["run_id"] == selected].iloc[0]
        rid = row['run_id']
        rc1, rc2 = st.columns([6, 1])
        with rc1:
            st.write(f"**Run ID:** `{rid}`")
        with rc2:
            components.html(
                f"""<script>
                function cp(){{navigator.clipboard.writeText('{rid}');
                  var b=document.getElementById('cb');
                  b.textContent='✓'; b.style.color='#1a9e6a';
                  setTimeout(()=>{{b.textContent='⧉';b.style.color='';}},1200);}}
                </script>
                <button id="cb" onclick="cp()" title="Copy Run ID"
                  style="background:none;border:1px solid #ccc;border-radius:4px;
                         padding:2px 7px;cursor:pointer;font-size:14px;color:#666;">⧉</button>
                """,
                height=36,
            )
        st.write(f"**Type:** {row.get('job_type', '—')}")
        st.write(f"**Status:** {row['status']}")
        st.write(f"**Submitted:** {_fmt_ist(row.get('created_at', '—'))} IST")
        if row.get("source_run_id"):
            st.write(f"**Source training run:** `{row['source_run_id']}`")

        # metrics chart for training jobs
        if row.get("job_type") == "train" and row["status"] in ("running", "completed"):
            try:
                mr = requests.get(f"{BACKEND}/metrics/{selected}", timeout=5)
                metrics = mr.json()
                history = metrics.get("history", {})
                if history.get("epoch"):
                    chart_df = pd.DataFrame({
                        "train_loss": history.get("train_loss", []),
                        "val_loss":   history.get("val_loss",   []),
                    }, index=history["epoch"])
                    st.line_chart(chart_df, x_label="Epoch", y_label="Loss")

                    final = metrics.get("final", {})
                    if final:
                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.metric("Val Acc",   f"{final.get('val_acc', 0):.4f}"   if final.get('val_acc')   else "—")
                        c2.metric("AUROC",     f"{final.get('auroc', 0):.4f}"     if final.get('auroc')     else "—")
                        c3.metric("Precision", f"{final.get('precision', 0):.4f}" if final.get('precision') else "—")
                        c4.metric("Recall",    f"{final.get('recall', 0):.4f}"    if final.get('recall')    else "—")
                        c5.metric("F1",        f"{final.get('f1', 0):.4f}"        if final.get('f1')        else "—")
            except Exception:
                pass

        # downloads
        if row["status"] == "cancelled":
            st.warning("This job was cancelled by the user.")

        elif row["status"] == "completed":
            st.divider()
            col_a, col_b, col_c = st.columns(3)

            if row.get("job_type") == "train":
                with col_a:
                    resp = requests.get(f"{BACKEND}/download_embedding/{selected}", stream=True)
                    if resp.status_code == 200:
                        st.download_button("Embeddings (.pkl)", data=resp.content,
                                           file_name=f"embedding_{selected}.pkl",
                                           mime="application/octet-stream")
                with col_b:
                    resp = requests.get(f"{BACKEND}/download_model/{selected}", stream=True)
                    if resp.status_code == 200:
                        st.download_button("Model weights (.pt)", data=resp.content,
                                           file_name=f"model_{selected}.pt",
                                           mime="application/octet-stream")
                with col_c:
                    resp = requests.get(f"{BACKEND}/download_bundle/{selected}", stream=True)
                    if resp.status_code == 200:
                        st.download_button("Artifact bundle (.zip)", data=resp.content,
                                           file_name=f"artifacts_{selected}.zip",
                                           mime="application/zip")

            elif row.get("job_type") == "inference":
                with col_a:
                    resp = requests.get(f"{BACKEND}/download_results/{selected}", stream=True)
                    if resp.status_code == 200:
                        st.download_button("Results (.csv)", data=resp.content,
                                           file_name=f"results_{selected}.csv",
                                           mime="text/csv")
                with col_b:
                    resp = requests.get(f"{BACKEND}/download_bundle/{selected}", stream=True)
                    if resp.status_code == 200:
                        st.download_button("Artifact bundle (.zip)", data=resp.content,
                                           file_name=f"artifacts_{selected}.zip",
                                           mime="application/zip")

except Exception as e:
    st.error(f"Could not reach backend: {e}")
