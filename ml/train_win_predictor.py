"""
FIFA World Cup Dashboard — ML Component
=========================================

APPROVED PROBLEM (adjusted per stakeholder direction to center on
win/loss framing, since the dashboard's primary goal is producing
"win probability" style graphs):

    Predict whether a team WINS a given World Cup match, using only
    information available BEFORE that match is played.

    Target      : team_won (binary: 1 = Win, 0 = Draw or Loss)
    Type        : Binary classification
    Model       : Logistic Regression (simple, explainable, coefficients
                  are directly interpretable as feature importance)
    Baseline    : DummyClassifier (majority class + stratified) for
                  honest comparison against "no model" performance

Explicitly OUT of scope: deep learning, NLP, external data, fabricated
labels, additional models beyond the required baseline comparison,
frontend, API.

--------------------------------------------------------------------
LEAKAGE PREVENTION (read before touching the feature list)
--------------------------------------------------------------------
Every feature is a HISTORICAL AGGREGATE computed strictly from a
team's matches BEFORE the match being predicted (via a per-team,
date-ordered expanding window shifted by one match). No column that
describes the outcome of the match itself (scores, goal difference,
result_method, is_draw, went_to_extra_time/penalties, winner) is used
as a feature. A team's very first World Cup appearance has no prior
history and is therefore dropped rather than imputed with a fabricated
value (e.g. a 50% prior win rate would not be real data).

Match-level (not row-level) splitting is used: both team-perspective
rows of a single match_id always land in the same split (train or
test), so no match's outcome is visible from one side in training and
the other side in testing.

--------------------------------------------------------------------
SPLIT STRATEGY
--------------------------------------------------------------------
Time-based, not random: the 3 most recent tournaments (2018, 2022,
2026) are held out entirely as the test set; everything before that
is training data. This mimics realistic deployment (predicting future
tournaments from past ones) and is more defensible for this domain
than a random split.

Usage:
    python ml/train_win_predictor.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42  # fixed for reproducibility throughout

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LONG_PATH = PROJECT_ROOT / "data" / "processed" / "matches_long.csv"

ML_DIR = PROJECT_ROOT / "ml"
MODEL_PATH = ML_DIR / "win_predictor_model.joblib"
EVAL_PATH = ML_DIR / "evaluation_results.txt"
IMPORTANCE_PATH = ML_DIR / "feature_importance.csv"
SUMMARY_PATH = ML_DIR / "model_summary.md"
RESULTS_JSON_PATH = ML_DIR / "results.json"  # structured metrics for API consumption

TEST_YEARS = [2018, 2022, 2026]  # held-out tournaments, never seen in training

FEATURE_COLUMNS = [
    "team_prior_matches_played",
    "team_prior_win_rate",
    "team_prior_avg_goals_for",
    "team_prior_avg_goals_against",
    "opp_prior_matches_played",
    "opp_prior_win_rate",
    "opp_prior_avg_goals_for",
    "opp_prior_avg_goals_against",
    "is_host",
    "tournament_stage_order",
]

MIN_PRIOR_MATCHES = 1  # a team needs at least 1 prior WC match to have any history


# ---------------------------------------------------------------------------
# Feature engineering (all strictly pre-match / historical)
# ---------------------------------------------------------------------------

def build_prior_team_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every (team, match_id) row, compute that team's aggregate stats
    from ONLY the matches strictly before this one (by date, then match_id
    as a tiebreaker for matches on the same date). Uses expanding().mean()
    on a shifted series so the current match is always excluded.
    """
    df = long_df.sort_values(["team", "date", "match_id"]).copy()
    df["is_win"] = (df["result"] == "Win").astype(int)

    grouped = df.groupby("team", group_keys=False)

    # shift(1) removes the current match from its own history window
    df["team_prior_matches_played"] = grouped.cumcount()
    df["team_prior_win_rate"] = grouped["is_win"].apply(lambda s: s.shift(1).expanding().mean())
    df["team_prior_avg_goals_for"] = grouped["goals_for"].apply(lambda s: s.shift(1).expanding().mean())
    df["team_prior_avg_goals_against"] = grouped["goals_against"].apply(lambda s: s.shift(1).expanding().mean())

    return df[
        [
            "match_id", "team", "date",
            "team_prior_matches_played", "team_prior_win_rate",
            "team_prior_avg_goals_for", "team_prior_avg_goals_against",
        ]
    ]


def build_feature_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assembles the full modeling table: one row per team-per-match, with
    that team's prior stats AND their opponent's prior stats attached,
    plus host status and stage. Target = team_won.
    """
    prior_stats = build_prior_team_stats(long_df)

    # Attach the acting team's own prior stats
    df = long_df.merge(prior_stats, on=["match_id", "team", "date"], how="left")

    # Attach the opponent's prior stats by re-keying the same prior_stats
    # table on (match_id, opponent) -> (match_id, team)
    opp_stats = prior_stats.rename(
        columns={
            "team": "opponent",
            "team_prior_matches_played": "opp_prior_matches_played",
            "team_prior_win_rate": "opp_prior_win_rate",
            "team_prior_avg_goals_for": "opp_prior_avg_goals_for",
            "team_prior_avg_goals_against": "opp_prior_avg_goals_against",
        }
    )
    df = df.merge(opp_stats, on=["match_id", "opponent"], how="left", suffixes=("", "_dup"))
    df = df.drop(columns=[c for c in df.columns if c.endswith("_dup")])

    # Host status: is this team playing in its own hosted tournament
    # (string containment handles multi-host tournaments like "Canada, Mexico, United States")
    df["is_host"] = df.apply(lambda r: str(r["team"]) in str(r["host_country"]), axis=1).astype(int)

    # Target
    df["team_won"] = (df["result"] == "Win").astype(int)

    return df


def filter_rows_with_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where either the team or opponent has no prior World Cup
    history (their first-ever appearance). These would otherwise require
    imputing a fabricated "average" prior record, which is not permitted.
    """
    has_history = (
        (df["team_prior_matches_played"] >= MIN_PRIOR_MATCHES)
        & (df["opp_prior_matches_played"] >= MIN_PRIOR_MATCHES)
    )
    return df[has_history].copy()


# ---------------------------------------------------------------------------
# Train / test split (match-level, time-based)
# ---------------------------------------------------------------------------

def time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits by world_cup_year so that:
      - both team-perspective rows of any given match_id stay together
      - the test set is entirely tournaments the model never trained on
    """
    test_df = df[df["world_cup_year"].isin(TEST_YEARS)].copy()
    train_df = df[~df["world_cup_year"].isin(TEST_YEARS)].copy()
    return train_df, test_df


# ---------------------------------------------------------------------------
# Model + baseline
# ---------------------------------------------------------------------------

def build_model_pipeline() -> Pipeline:
    """Simple, explainable model: standardized features -> logistic regression."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run() -> dict:
    long_df = pd.read_csv(LONG_PATH)

    features_df_all = build_feature_table(long_df)
    features_df = filter_rows_with_history(features_df_all)
    dropped_rows_no_history = len(features_df_all) - len(features_df)

    train_df, test_df = time_based_split(features_df)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["team_won"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["team_won"]

    # --- Model ---
    model = build_model_pipeline()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    model_metrics = evaluate(y_test, y_pred, y_proba)

    # --- Baselines ---
    baseline_majority = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    baseline_majority.fit(X_train, y_train)
    y_pred_majority = baseline_majority.predict(X_test)
    y_proba_majority = baseline_majority.predict_proba(X_test)[:, 1]
    majority_metrics = evaluate(y_test, y_pred_majority, y_proba_majority)

    baseline_stratified = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    baseline_stratified.fit(X_train, y_train)
    y_pred_stratified = baseline_stratified.predict(X_test)
    y_proba_stratified = baseline_stratified.predict_proba(X_test)[:, 1]
    stratified_metrics = evaluate(y_test, y_pred_stratified, y_proba_stratified)

    # --- Feature importance (standardized logistic regression coefficients) ---
    coefs = model.named_steps["clf"].coef_[0]
    importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "coefficient": coefs})
    importance["abs_coefficient"] = importance["coefficient"].abs()
    importance = importance.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    importance = importance.drop(columns="abs_coefficient")

    ML_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    # --- Save model (kept minimal: one file, only the fitted pipeline) ---
    joblib.dump(model, MODEL_PATH)

    return {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_win_rate": round(y_train.mean(), 3),
        "test_win_rate": round(y_test.mean(), 3),
        "model_metrics": model_metrics,
        "majority_baseline_metrics": majority_metrics,
        "stratified_baseline_metrics": stratified_metrics,
        "feature_importance": importance,
        "dropped_rows_no_history": dropped_rows_no_history,
    }


def write_results_json(results: dict) -> None:
    """
    Writes a compact, structured JSON file with everything an API layer
    would need to serve the 'approved ML result' without re-parsing the
    human-readable .txt/.md reports. This is the single source of truth
    the API reads from — no metrics are recomputed or reformatted there.
    """
    payload = {
        "problem": "Predict whether a team wins a given World Cup match",
        "model_type": "Logistic Regression (binary classification)",
        "test_years": TEST_YEARS,
        "n_train": results["n_train"],
        "n_test": results["n_test"],
        "train_win_rate": results["train_win_rate"],
        "test_win_rate": results["test_win_rate"],
        "model_metrics": {
            k: v for k, v in results["model_metrics"].items()
        },
        "majority_baseline_metrics": {
            k: v for k, v in results["majority_baseline_metrics"].items()
        },
        "stratified_baseline_metrics": {
            k: v for k, v in results["stratified_baseline_metrics"].items()
        },
        "feature_importance": results["feature_importance"].to_dict(orient="records"),
    }
    RESULTS_JSON_PATH.write_text(json.dumps(payload, indent=2))


def write_evaluation_report(results: dict) -> None:
    def fmt_metrics(m: dict, label: str) -> str:
        lines = [f"{label}:"]
        lines.append(f"  Accuracy : {m['accuracy']:.3f}")
        lines.append(f"  Precision: {m['precision']:.3f}")
        lines.append(f"  Recall   : {m['recall']:.3f}")
        lines.append(f"  F1 score : {m['f1']:.3f}")
        lines.append(f"  ROC-AUC  : {m['roc_auc']:.3f}")
        lines.append(f"  Confusion matrix [[TN,FP],[FN,TP]]: {m['confusion_matrix']}")
        return "\n".join(lines)

    lines = []
    lines.append("FIFA World Cup Win Predictor — Evaluation Results")
    lines.append("=" * 52)
    lines.append(f"Train rows (team-per-match, with prior history): {results['n_train']}")
    lines.append(f"Test rows  (2018, 2022, 2026 tournaments only)  : {results['n_test']}")
    lines.append(f"Train win rate (class balance): {results['train_win_rate']}")
    lines.append(f"Test win rate (class balance) : {results['test_win_rate']}")
    lines.append("")
    lines.append(fmt_metrics(results["model_metrics"], "Logistic Regression (our model)"))
    lines.append("")
    lines.append(fmt_metrics(results["majority_baseline_metrics"], "Baseline: Majority Class"))
    lines.append("")
    lines.append(fmt_metrics(results["stratified_baseline_metrics"], "Baseline: Stratified Random"))
    lines.append("")
    lines.append("Feature importance (standardized logistic regression coefficients):")
    lines.append(results["feature_importance"].to_string(index=False))

    EVAL_PATH.write_text("\n".join(lines))


def write_model_summary(results: dict) -> None:
    fi = results["feature_importance"]
    top_features = fi.head(3)["feature"].tolist()

    text = f"""# Model Summary — World Cup Match Win Predictor

## What it does
Given a team about to play a World Cup match, estimates the probability that
team wins (as opposed to drawing or losing), using only information known
before kickoff.

## Problem type
Binary classification (Win = 1, Draw or Loss = 0).

## Model
Logistic Regression on 10 standardized numeric features. Chosen for
simplicity and explainability — coefficients map directly to each feature's
direction and relative strength of influence, with no black-box behavior.

## Features used (all pre-match, historical aggregates only)
- Team's prior matches played, prior win rate, prior avg goals for/against
- Opponent's prior matches played, prior win rate, prior avg goals for/against
- Whether the team is playing in its own hosted tournament
- The stage of the current match (group stage vs later knockout rounds)

## Data split
Time-based: trained on all tournaments through 2014, tested on the three
most recent tournaments (2018, 2022, 2026) — {results['n_train']} training
rows, {results['n_test']} test rows. Neither team's prior stats nor the
split itself ever use information from the future relative to the match
being predicted.

## Performance vs. baseline

| Metric | Model | Majority-class baseline | Stratified baseline |
|---|---|---|---|
| Accuracy | {results['model_metrics']['accuracy']:.3f} | {results['majority_baseline_metrics']['accuracy']:.3f} | {results['stratified_baseline_metrics']['accuracy']:.3f} |
| F1 score | {results['model_metrics']['f1']:.3f} | {results['majority_baseline_metrics']['f1']:.3f} | {results['stratified_baseline_metrics']['f1']:.3f} |
| ROC-AUC | {results['model_metrics']['roc_auc']:.3f} | {results['majority_baseline_metrics']['roc_auc']:.3f} | {results['stratified_baseline_metrics']['roc_auc']:.3f} |

## Most influential features
{', '.join(top_features)} — see `feature_importance.csv` for the full ranked
list with signed coefficients (positive = pushes prediction toward "Win").

## Limitations
- **Thin history for many teams.** World Cups occur once every 4 years, so
  even a team's 5th tournament appearance has a small number of prior
  matches to average over — historical rates are noisy, especially early
  in a team's tournament history.
- **First-appearance teams are excluded entirely**, not predicted. A team
  with zero prior World Cup matches has no historical features to compute,
  and was dropped from both training and evaluation rather than assigned a
  fabricated average.
- **No non-World-Cup information.** No current squad strength, injuries,
  qualifying form, or FIFA ranking is available in this dataset — the model
  only knows what happened in prior World Cups, which is a limited proxy
  for a team's actual current strength.
- **Small test set.** The test set covers 3 tournaments only; metrics
  should be read as indicative, not as a precise measure of real-world
  predictive power.
- **Not intended as a betting or forecasting tool.** This is an
  explanatory/exploratory dashboard component, not a validated predictive
  system.
- **Correlated features can distort individual coefficient signs.** Several
  features are naturally correlated (e.g. `team_prior_win_rate` and
  `team_prior_avg_goals_for` both track team strength). Logistic regression
  coefficients should be read as *relative influence within this feature
  set*, not as isolated, causal effects of a single stat.
"""
    SUMMARY_PATH.write_text(text)


if __name__ == "__main__":
    results = run()
    write_evaluation_report(results)
    write_model_summary(results)
    write_results_json(results)
    print("Training complete.")
    print(f"Train rows: {results['n_train']}  Test rows: {results['n_test']}")
    print(f"Model accuracy: {results['model_metrics']['accuracy']:.3f}  "
          f"ROC-AUC: {results['model_metrics']['roc_auc']:.3f}")
    print(f"Majority baseline accuracy: {results['majority_baseline_metrics']['accuracy']:.3f}")
    print(f"Saved: {MODEL_PATH}, {EVAL_PATH}, {IMPORTANCE_PATH}, {SUMMARY_PATH}, {RESULTS_JSON_PATH}")