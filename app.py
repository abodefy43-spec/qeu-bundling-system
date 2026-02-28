"""Flask presentation app for QEU bundle outputs."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for

from src.presentation.bundle_view import load_bundle_view
from src.presentation.run_service import get_status, start_pipeline_run

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)


@app.get("/")
def dashboard():
    status = get_status()
    data = load_bundle_view(BASE_DIR)
    return render_template(
        "dashboard.html",
        page_title="Executive Dashboard",
        kpis=data.kpis,
        top10_rows=data.top10_rows,
        data_warning=data.data_warning,
        run_status=status,
    )


@app.get("/bundles")
def all_bundles():
    query = request.args.get("q", "")
    status = get_status()
    data = load_bundle_view(BASE_DIR, query=query)
    return render_template(
        "all_bundles.html",
        page_title="All Bundles",
        rows=data.all_rows,
        query=query,
        data_warning=data.data_warning,
        run_status=status,
    )


@app.post("/refresh")
def refresh():
    ok, message = start_pipeline_run(BASE_DIR)
    flash(message, "success" if ok else "warning")
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

