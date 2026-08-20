"""
FIFA World Cup Dashboard — Minimal Backend API
=================================================

Framework: Flask (simplest framework already installed in the project's
environment; no framework existed previously in the project code, and
no new framework was introduced beyond this minimal choice).

Scope: exposes exactly the functionality the dashboard needs, as thin
HTTP wrappers around the existing, already-tested analytics functions
in pipeline/analytics.py and the stored ML results in ml/results.json.
No analytical logic is reimplemented here — every route either:
  (a) calls a function from pipeline/analytics.py, or
  (b) reads a pre-generated file produced by ml/train_win_predictor.py.

Explicitly OUT of scope: authentication, database, frontend, new ML
training/inference, new analytical logic.

Endpoints
---------
GET /api/countries
    -> list of every valid, selectable country/team name.

GET /api/years
    -> list of every valid World Cup year.

GET /api/global?year=<optional int>
    -> bundle of all 6 global-view statistics (G1-G6). If `year` is
       given, all charts are scoped to that single tournament, except
       goals_per_tournament (G1), which is inherently a full-history
       trend line and is returned in full regardless of `year`.

GET /api/tournaments/<int:year>
    -> the same bundle as /api/global, hard-scoped to one tournament.
       Distinct route from /api/global for semantic clarity (dashboard
       "tournament view" vs "global view"), but reuses identical logic.

GET /api/countries/<country>?year=<optional int>
    -> bundle of all 5 country-view statistics (C1-C5) for one country.
       Passing `year` additionally satisfies "country + tournament
       filtering" by scoping every chart to that single tournament.

GET /api/ml/result
    -> the approved ML result (win predictor): metrics, baseline
       comparison, and feature importance, as stored in ml/results.json.

Run:
    python api/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # so 'pipeline' resolves regardless of cwd

import pandas as pd
from flask import Flask, jsonify, request

from pipeline import analytics as an

# ---------------------------------------------------------------------------
# Paths & data loading (loaded once at startup, not per-request)
# ---------------------------------------------------------------------------
MATCHES_CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "matches_clean.csv"
MATCHES_LONG_PATH = PROJECT_ROOT / "data" / "processed" / "matches_long.csv"
ML_RESULTS_PATH = PROJECT_ROOT / "ml" / "results.json"

matches_df = pd.read_csv(MATCHES_CLEAN_PATH)
long_df = pd.read_csv(MATCHES_LONG_PATH)

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    """
    Minimal CORS support so the static frontend (served from a different
    origin/port) can call this API. Adds headers only — no route logic,
    validation, or analytical behavior is changed.
    """
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET"
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Converts an analytics DataFrame into clean JSON-serializable records."""
    return json.loads(df.to_json(orient="records"))


def get_year_param() -> int | None:
    """
    Reads the optional `year` query parameter. Returns None if absent
    (meaning "all years"). Raises ValueError if present but not an int,
    which the route handlers translate into a 400 response.
    """
    raw = request.args.get("year")
    if raw is None or raw == "":
        return None
    return int(raw)  # raises ValueError on bad input, handled by caller


# ---------------------------------------------------------------------------
# Error handling — analytics.py's own validation errors become clean JSON
# ---------------------------------------------------------------------------

@app.errorhandler(an.InvalidCountryError)
def handle_invalid_country(err):
    return jsonify({"error": str(err)}), 400


@app.errorhandler(an.InvalidYearError)
def handle_invalid_year(err):
    return jsonify({"error": str(err)}), 400


@app.errorhandler(ValueError)
def handle_value_error(err):
    return jsonify({"error": f"Invalid parameter: {err}"}), 400


@app.errorhandler(404)
def handle_not_found(err):
    """
    BUG FIX (audit findings M1/L2): unmatched routes and int-converter
    failures (e.g. /api/tournaments/abc, /api/nonexistent) previously fell
    through to Flask's default HTML 404 page, violating "return clean
    JSON." Every 404 now returns the same structured JSON error shape as
    every other error in this API.
    """
    return jsonify({"error": "Not found. Check the endpoint path and parameter types."}), 404


# ---------------------------------------------------------------------------
# 2 & 3. Available countries / years
# ---------------------------------------------------------------------------

@app.get("/api/countries")
def get_countries():
    return jsonify({"countries": an.list_valid_countries(long_df)})


@app.get("/api/years")
def get_years():
    return jsonify({"years": an.list_valid_years(matches_df)})


# ---------------------------------------------------------------------------
# Shared bundler for global/tournament statistics (used by both routes
# below) — keeps the two routes from duplicating the same six calls.
# ---------------------------------------------------------------------------

def build_global_bundle(year: int | None) -> dict:
    return {
        "year": year,
        "goals_per_tournament": df_to_records(an.goals_per_tournament(matches_df)),
        "matches_by_stage": df_to_records(an.matches_by_stage(matches_df, year=year)),
        "result_method_breakdown": df_to_records(an.result_method_breakdown(matches_df, year=year)),
        "top_teams_by_wins": df_to_records(an.top_teams_by_wins(long_df, year=year)),
        "host_country_summary": df_to_records(an.host_country_summary(matches_df, year=year)),
        "goal_margin_distribution": df_to_records(an.goal_margin_distribution(matches_df, year=year)),
    }


# ---------------------------------------------------------------------------
# 1. Global statistics
# ---------------------------------------------------------------------------

@app.get("/api/global")
def get_global_stats():
    year = get_year_param()
    if year is not None:
        an.validate_year(matches_df, year)
    return jsonify(build_global_bundle(year))


# ---------------------------------------------------------------------------
# 5. Tournament statistics (global bundle hard-scoped to one year)
# ---------------------------------------------------------------------------

@app.get("/api/tournaments/<int:year>")
def get_tournament_stats(year: int):
    an.validate_year(matches_df, year)
    return jsonify(build_global_bundle(year))


# ---------------------------------------------------------------------------
# 4 & 6. Country statistics, with optional tournament-year filtering
# ---------------------------------------------------------------------------

@app.get("/api/countries/<country>")
def get_country_stats(country: str):
    an.validate_country(long_df, country)
    year = get_year_param()
    if year is not None:
        an.validate_year(long_df, year)

    return jsonify({
        "country": country,
        "year": year,
        "record_by_year": df_to_records(an.country_record_by_year(long_df, country, year=year)),
        "goals_for_against": df_to_records(an.country_goals_for_against(long_df, country, year=year)),
        "deepest_stage_by_year": df_to_records(an.country_deepest_stage_by_year(long_df, country, year=year)),
        "result_method_breakdown": df_to_records(an.country_result_method_breakdown(long_df, country, year=year)),
        "top_opponents": df_to_records(an.country_top_opponents(long_df, country, year=year)),
    })


# ---------------------------------------------------------------------------
# 7. Approved ML result
# ---------------------------------------------------------------------------

@app.get("/api/ml/result")
def get_ml_result():
    if not ML_RESULTS_PATH.exists():
        return jsonify({"error": "ML results not found. Run ml/train_win_predictor.py first."}), 404
    return jsonify(json.loads(ML_RESULTS_PATH.read_text()))


if __name__ == "__main__":
    app.run(debug=True, port=5000)