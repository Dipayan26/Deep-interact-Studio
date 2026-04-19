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
        options=["all", "ppi", "dti", "rpi", "pdi"],
        index=["all", "ppi", "dti", "rpi", "pdi"].index(st.session_state["js_task_type_filter"]),
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
    if st.button("Reset Filters"):
        st.session_state["js_status_filter"] = []
        st.session_state["js_job_type_filter"] = "all"
        st.session_state["js_task_type_filter"] = "all"
        st.session_state["js_run_contains"] = ""
        st.session_state["js_page_size"] = 50
        st.session_state["js_page"] = 0
        st.session_state["js_filter_sig"] = None
        st.rerun()

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
        "completed": "#1a9e6a",
        "running":   "#1a6bbf",
        "queued":    "#b07d12",
        "failed":    "#c0392b",
        "cancelled": "#888888",
    }

    TASK_LABEL = {
        "ppi": ("PPI",             "#1a5fa5", "#dceeff"),
        "dti": ("Drug Target DTI", "#2A7D6F", "#d6f5f0"),
        "rpi": ("RNA-Protein RPI", "#7A3E9D", "#efe3fb"),
        "pdi": ("Protein-DNA PDI", "#A05000", "#ffe8d3"),
    }

    rows_html = ""
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

        rows_html += f"""
        <tr>
          <td>
            <span class="task-badge" style="color:{t_fg};background:{t_bg};">{t_label}</span>
          </td>
          <td>
            <span class="rid">{rid}</span>
            <button class="copy-btn" onclick="copyId(this, '{rid}')" title="Copy Run ID">
              <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24"
                   fill="none" stroke="currentColor" stroke-width="2"
                   stroke-linecap="round" stroke-linejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
              </svg>
            </button>
          </td>
          <td>{jtype}</td>
          <td><span class="badge" style="color:{s_col};border-color:{s_col};">{status}</span></td>
          <td>{created}</td>
          <td>{acc_s}</td>
          <td>{auroc_s}</td>
        </tr>"""

    table_html = f"""
    <style>
      body {{ margin:0; font-family: sans-serif; font-size:13px; }}
      .wrapper {{
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
      }}
      table {{ width:100%; border-collapse:collapse; }}
      thead th {{
        position: sticky; top: 0; z-index: 1;
        background: #f9f9f9;
        text-align:left; padding:7px 10px;
        border-bottom:2px solid #e0e0e0;
        color:#555; font-weight:600; font-size:12px; white-space:nowrap;
      }}
      td {{ padding:6px 10px; border-bottom:1px solid #f0f0f0; vertical-align:middle; }}
      tr:last-child td {{ border-bottom: none; }}
      tr:hover td {{ background:#f7f7f7; }}
      .task-badge {{
        display:inline-block; padding:2px 9px; border-radius:4px;
        font-size:11px; font-weight:700; white-space:nowrap;
      }}
      .rid {{ font-family:monospace; font-size:12px; }}
      .copy-btn {{
        background:none; border:none; cursor:pointer;
        color:#bbb; padding:0 0 0 5px;
        vertical-align:middle; transition:color .15s;
      }}
      .copy-btn:hover {{ color:#1a6bbf; }}
      .copy-btn.ok {{ color:#1a9e6a !important; }}
      .badge {{
        display:inline-block; padding:2px 8px; border-radius:20px;
        border:1px solid; font-size:11px; font-weight:600;
      }}
    </style>
    <div class="wrapper">
      <table>
        <thead>
          <tr>
            <th>Task</th><th>Run ID</th><th>Job Type</th>
            <th>Status</th><th>Submitted (IST)</th><th>Val Acc</th><th>AUROC</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    <script>
      function copyId(btn, id) {{
        navigator.clipboard.writeText(id).then(function() {{
          btn.classList.add('ok');
          setTimeout(function() {{ btn.classList.remove('ok'); }}, 1200);
        }});
      }}
    </script>
    """
    components.html(table_html, height=420, scrolling=False)

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

            elif row.get("job_type") == "inference":
                with col_a:
                    resp = requests.get(f"{BACKEND}/download_results/{selected}", stream=True)
                    if resp.status_code == 200:
                        st.download_button("Results (.csv)", data=resp.content,
                                           file_name=f"results_{selected}.csv",
                                           mime="text/csv")

except Exception as e:
    st.error(f"Could not reach backend: {e}")
