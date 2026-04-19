import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER)

TASK_LABELS = {
    "ppi": "Protein–Protein Interaction (PPI)",
    "dti": "Drug–Target Interaction (DTI)",
    "rpi": "RNA–Protein Interaction (RPI)",
    "pdi": "Protein–DNA Interaction (PDI)",
}


def send_job_notification(
    to_email: str,
    run_id: str,
    status: str,
    task_type: str = "ppi",
    metrics: dict | None = None,
    error_msg: str = "",
):
    if not to_email or not SMTP_USER or not SMTP_PASS:
        return

    is_ok = status == "completed"
    subject = (
        f"[Deep-Prot Studio] Job {run_id} — {'Completed ✓' if is_ok else 'Failed ✗'}"
    )

    task_label = TASK_LABELS.get(task_type, task_type.upper())

    if is_ok and metrics:
        auroc = metrics.get("auroc")
        f1    = metrics.get("f1")
        acc   = metrics.get("val_acc")
        metrics_html = f"""
        <table style="border-collapse:collapse;margin-top:12px">
          <tr><th style="text-align:left;padding:4px 12px 4px 0;color:#555">Val Accuracy</th>
              <td style="padding:4px 0"><b>{f'{acc:.4f}' if acc is not None else '—'}</b></td></tr>
          <tr><th style="text-align:left;padding:4px 12px 4px 0;color:#555">AUROC</th>
              <td style="padding:4px 0"><b>{f'{auroc:.4f}' if auroc is not None else '—'}</b></td></tr>
          <tr><th style="text-align:left;padding:4px 12px 4px 0;color:#555">F1 Score</th>
              <td style="padding:4px 0"><b>{f'{f1:.4f}' if f1 is not None else '—'}</b></td></tr>
        </table>"""
    else:
        metrics_html = ""

    status_color = "#2ecc71" if is_ok else "#e74c3c"
    status_text  = "completed successfully" if is_ok else "failed"
    body_detail  = (
        f"{metrics_html}"
        if is_ok
        else f'<p style="color:#e74c3c;margin-top:8px"><b>Reason:</b> {error_msg or "Unknown error"}</p>'
    )

    html = f"""
    <html><body style="font-family:Arial,sans-serif;color:#222;max-width:520px;margin:auto">
      <div style="background:#355E8E;padding:18px 24px;border-radius:6px 6px 0 0">
        <h2 style="color:white;margin:0">Deep-Prot Studio</h2>
      </div>
      <div style="border:1px solid #ddd;border-top:none;padding:24px;border-radius:0 0 6px 6px">
        <p>Your <b>{task_label}</b> training job has
           <span style="color:{status_color}"><b>{status_text}</b></span>.</p>
        <table style="border-collapse:collapse;margin-bottom:12px">
          <tr><th style="text-align:left;padding:4px 12px 4px 0;color:#555">Run ID</th>
              <td><code style="background:#f4f4f4;padding:2px 6px;border-radius:3px">{run_id}</code></td></tr>
          <tr><th style="text-align:left;padding:4px 12px 4px 0;color:#555">Task</th>
              <td>{task_label}</td></tr>
          <tr><th style="text-align:left;padding:4px 12px 4px 0;color:#555">Status</th>
              <td><span style="color:{status_color};font-weight:bold">{status.capitalize()}</span></td></tr>
        </table>
        {body_detail}
        <p style="margin-top:20px">View full results in
           <b>Tools → Check Results</b> using run ID <code>{run_id}</code>.</p>
        <hr style="border:none;border-top:1px solid #eee;margin:20px 0">
        <p style="font-size:12px;color:#888">Deep-Prot Studio.<br> Computational Systems Biology Laboratory.<br> University of North Bengal<br>
           This is an automated notification.</p>
      </div>
    </body></html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SMTP_FROM
    msg["To"]      = to_email
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASS)
            smtp.sendmail(SMTP_FROM, [to_email], msg.as_string())
        print(f"[email] Notification sent to {to_email} for run {run_id} ({status})", flush=True)
    except Exception as exc:
        print(f"[email] Failed to send notification: {exc}", flush=True)
