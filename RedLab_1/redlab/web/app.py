"""RedLab web prototype: FastAPI app demonstrating the closed Identify ->
Generate -> Defend -> Loop system.

Serves pre-computed artifacts (see scripts/) plus live scoring of a persisted
detector against a held-out transaction sample, so the demo is fast and
reproducible without retraining on every request.

    uvicorn redlab.web.app:app --reload --port 8000
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import charts, data
from ..taxonomy.schema import Family, Maturity

APP_DIR = Path(__file__).resolve().parent
app = FastAPI(title="RedLab — AI Defense Lab for Payment Security")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    tax = data.taxonomy()
    fl = data.fidelity_legit()
    fa = data.fidelity_attacks()
    de = data.detector_eval()
    loop = data.adversarial_loop()

    summary = tax.summary()
    disc_auc = fl["discriminator_auc"]
    attack_gap = fa["generated"]["pr_auc"] - fa["reference"]["pr_auc"]

    ctx = {
        "request": request, "active": "home",
        "n_vectors": summary["total_vectors"],
        "n_families": len(summary["families"]),
        "mean_novelty": summary["mean_novelty"],
        "disc_auc": disc_auc,
        "attack_gap": attack_gap,
        "pr_in": de["in_distribution"]["pr_auc"],
        "pr_mech": de["mechanism_holdout"]["pr_auc"],
        "recall_mech": de["mechanism_holdout"]["recall_at_fpr"]["0.5%"],
        "evasion_start": loop[0]["evasion_rate"],
        "evasion_end": loop[-1]["evasion_rate"],
        "gaps": summary["gaps"],
    }
    return templates.TemplateResponse("home.html", ctx)


# --------------------------------------------------------------------------
# Red vs Blue Arena (D2)
# --------------------------------------------------------------------------

@app.get("/arena", response_class=HTMLResponse)
def arena(request: Request):
    loop = data.adversarial_loop()
    rounds = [f"R{r['round']}" for r in loop]
    evasion = [r["evasion_rate"] * 100 for r in loop]
    value = [r["value_retention"] * 100 for r in loop]
    recall = [r["detector_recall"] * 100 for r in loop]

    evasion_svg = charts.line_chart(
        [("evasion %", evasion, "#ff6b6b"), ("detector recall %", recall, "#59d38a")],
        rounds, y_fmt=lambda v: f"{v:.0f}%", y_domain=(0, 100))
    value_svg = charts.line_chart(
        [("attacker value retained %", value, "#4f9dff")],
        rounds, y_fmt=lambda v: f"{v:.0f}%")

    genome_keys = list(loop[-1]["genome"].keys())
    genome_rows = [(r["round"], {k: round(r["genome"][k], 3) for k in genome_keys})
                   for r in loop]

    ctx = {"request": request, "active": "arena", "evasion_svg": evasion_svg,
           "value_svg": value_svg, "rounds": loop, "genome_keys": genome_keys,
           "genome_rows": genome_rows,
           "evasion_start": loop[0]["evasion_rate"], "evasion_end": loop[-1]["evasion_rate"]}
    return templates.TemplateResponse("arena.html", ctx)


# --------------------------------------------------------------------------
# Live Defense Console
# --------------------------------------------------------------------------

@app.get("/console", response_class=HTMLResponse)
def console(request: Request, decision: Optional[str] = Query(None),
           only_fraud: bool = Query(False), limit: int = Query(150, le=1000),
           window: str = Query("busy")):
    df = data.console_sample()
    if window == "busy":
        # Default view: the busiest fraud day, so the feed demonstrates the
        # system catching something rather than showing an arbitrary quiet
        # slice. "full" shows the literal chronological tail instead.
        start, end = data.demo_window()
        df = df[(df.timestamp >= start) & (df.timestamp < end)]
    if decision and decision != "ALL":
        df = df[df.decision == decision]
    if only_fraud:
        df = df[df.is_fraud == 1]
    view = df.tail(limit).iloc[::-1]  # most recent first

    bundle = data.model_bundle()
    total = data.console_sample()
    caught = int(((total.decision != "ALLOW") & (total.is_fraud == 1)).sum())
    total_fraud = int(total.is_fraud.sum())
    false_blocks = int(((total.decision == "BLOCK") & (total.is_fraud == 0)).sum())
    total_legit = int((total.is_fraud == 0).sum())

    ctx = {"request": request, "active": "console",
           "rows": view.to_dict("records"),
           "decision_filter": decision or "ALL", "only_fraud": only_fraud,
           "window": window, "demo_day": data.demo_window()[0].strftime("%Y-%m-%d"),
           "thresh_review": bundle["thresh_review"], "thresh_block": bundle["thresh_block"],
           "caught": caught, "total_fraud": total_fraud,
           "recall_pct": 100 * caught / max(total_fraud, 1),
           "false_block_rate": 100 * false_blocks / max(total_legit, 1),
           "n_shown": len(view)}
    return templates.TemplateResponse("console.html", ctx)


# --------------------------------------------------------------------------
# Attack Atlas
# --------------------------------------------------------------------------

@app.get("/atlas", response_class=HTMLResponse)
def atlas(request: Request, family: Optional[str] = Query(None),
         maturity: Optional[str] = Query(None)):
    tax = data.taxonomy()
    vectors = tax.vectors
    if family and family != "ALL":
        vectors = [v for v in vectors if v.family.value == family]
    if maturity and maturity != "ALL":
        vectors = [v for v in vectors if v.maturity.value == maturity]
    vectors = sorted(vectors, key=lambda v: -v.priority)

    matrix = tax.family_rail_matrix()
    rails = sorted({r for row in matrix.values() for r in row})

    ctx = {"request": request, "active": "atlas", "vectors": vectors,
           "families": [f.value for f in Family], "maturities": [m.value for m in Maturity],
           "sel_family": family or "ALL", "sel_maturity": maturity or "ALL",
           "matrix": matrix, "rails": rails}
    return templates.TemplateResponse("atlas.html", ctx)


@app.get("/atlas/{vector_id}", response_class=HTMLResponse)
def atlas_detail(request: Request, vector_id: str):
    tax = data.taxonomy()
    try:
        v = tax[vector_id]
    except KeyError:
        return HTMLResponse(f"<p>unknown vector {vector_id}</p>", status_code=404)
    return templates.TemplateResponse("atlas_detail.html",
                                      {"request": request, "active": "atlas", "v": v})


# --------------------------------------------------------------------------
# Fidelity Report
# --------------------------------------------------------------------------

@app.get("/fidelity", response_class=HTMLResponse)
def fidelity(request: Request):
    fl = data.fidelity_legit()
    fa = data.fidelity_attacks()

    fam_metrics = {}
    for m in fl["metrics"]:
        fam_metrics.setdefault(m["family"], []).append(m)

    feat_labels = list(fa["reference"]["per_feature_auc"].keys())
    ref_vals = [fa["reference"]["per_feature_auc"][k] for k in feat_labels]
    gen_vals = [fa["generated"]["per_feature_auc"][k] for k in feat_labels]
    sig_svg = charts.line_chart(
        [("reference fraud", ref_vals, "#ff6b6b"), ("generated fraud", gen_vals, "#4f9dff")],
        feat_labels, y_fmt=lambda v: f"{v:.2f}", y_domain=(0.4, 1.0))

    ctx = {"request": request, "active": "fidelity", "fam_metrics": fam_metrics,
           "disc_auc": fl["discriminator_auc"], "n_gen": fl["n_generated"],
           "n_ref": fl["n_reference"], "ref_name": fl["reference_name"],
           "fa_ref": fa["reference"], "fa_gen": fa["generated"], "sig_svg": sig_svg}
    return templates.TemplateResponse("fidelity.html", ctx)
