"""
FIFA World Cup Dataset — Preprocessing & Feature Engineering Pipeline
======================================================================

Scope (per project spec):
    - Cleaning
    - Standardizing team/country names
    - Handling missing values
    - Creating derived statistical features
    - Creating analysis-ready datasets

Explicitly OUT of scope: frontend, API, ML training, external data,
modification of the original raw file.

Reproducibility:
    - Pure function pipeline, no randomness, no external calls.
    - Input:  data/raw/fifa_world_cup_all_matches_1930_2026.csv  (untouched)
    - Output: data/processed/  (created fresh each run)
    - Running this script twice on the same input produces identical output.

Usage:
    python pipeline/preprocess.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to project root, resolved from this file's location)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "fifa_world_cup_all_matches_1930_2026.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUT_MATCHES_CLEAN = PROCESSED_DIR / "matches_clean.csv"
OUT_MATCHES_LONG = PROCESSED_DIR / "matches_long.csv"
OUT_NAME_MAP = PROCESSED_DIR / "team_name_map.csv"
OUT_REPORT = PROCESSED_DIR / "pipeline_report.txt"

# ---------------------------------------------------------------------------
# Stage 2.1 — explicit team/country name standardization map
# Built only from inconsistencies confirmed in the audit. Historical entities
# (West Germany, Soviet Union, Czechoslovakia, Yugoslavia, Serbia and
# Montenegro, East Germany) are intentionally NOT merged — see 2.2.
# ---------------------------------------------------------------------------
TEAM_NAME_MAP = {
    "USA": "United States",
    "Ivory Coast": "Côte d'Ivoire",
    "Bosnia-Herzegovina": "Bosnia & Herzegovina",
}

# Recognized historical / defunct entities (Stage 2.2 flag, not merged)
HISTORICAL_ENTITIES = {
    "West Germany",
    "East Germany",
    "Soviet Union",
    "Czechoslovakia",
    "Yugoslavia",
    "Serbia and Montenegro",
}

# Stage 4.8 — ordinal progression for tournament stages (football knowledge,
# not external data). Unmapped / unseen stage labels get NaN, not guessed.
STAGE_ORDER = {
    "Preliminary round": 0,
    "First round": 1,
    "First round, Replays": 1,
    "Group Stage": 1,
    "Final Round": 2,
    "Round of 32": 2,
    "Round of 16": 3,
    "Quarter-final": 4,
    "Semi-final": 5,
    "Third Place Play-off": 6,
    "Third-place play-off": 6,
    "Third-place match": 6,
    "Final": 7,
}


def load_raw(path: Path) -> pd.DataFrame:
    """Load the raw CSV as-is. No mutation of the source file ever occurs."""
    df = pd.read_csv(path)
    return df


def clean_types(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 1.3 — type normalization."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")

    score_cols = [
        "halftime_score_team1", "halftime_score_team2",
        "fulltime_score_team1", "fulltime_score_team2",
        "extra_time_score_team1", "extra_time_score_team2",
        "penalty_score_team1", "penalty_score_team2",
    ]
    for col in score_cols:
        df[col] = df[col].astype("Int64")  # nullable integer, preserves NaN

    return df


def drop_unreliable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 1.1 — total_goals_team1/2 mismatch fulltime score in 58.1% of
    rows and are not used as a source of truth. Dropped; goals are
    recomputed from fulltime scores in Stage 4."""
    df = df.copy()
    df = df.drop(columns=["total_goals_team1", "total_goals_team2"])
    return df


def standardize_names(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 2 — apply the standardization map identically to every column
    that holds a team/country name, plus record which rows involve a
    historical/defunct entity."""
    df = df.copy()

    name_cols = ["team1", "team2", "winner"]
    for col in name_cols:
        df[col] = df[col].replace(TEAM_NAME_MAP)

    # winner can also be "Draw" - not a team name, left untouched by replace()

    df["host_country"] = df["host_country"].replace(TEAM_NAME_MAP)

    # Stage 2.2 - historical entity flag, at MATCH level (either side involved).
    # This is intentionally match-scoped, not team-scoped - it answers "did this
    # match involve a defunct nation," not "is this team defunct." A team-scoped
    # version is computed separately in build_matches_long() for per-team filtering.
    df["match_involves_historical_entity"] = (
        df["team1"].isin(HISTORICAL_ENTITIES) | df["team2"].isin(HISTORICAL_ENTITIES)
    )

    # Stage 2.3 - normalize host_country multi-nation delimiter to comma-separated
    df["host_country"] = df["host_country"].str.replace(" / ", ", ", regex=False)

    return df


def add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 3 — convert meaningful missingness into explicit booleans.
    No score/city value is ever imputed."""
    df = df.copy()
    df["has_halftime_data"] = df["halftime_score_team1"].notna()
    df["went_to_extra_time"] = df["extra_time_score_team1"].notna()
    df["went_to_penalties"] = df["penalty_score_team1"].notna()
    df["is_walkover"] = df["result_method"] == "Walkover"
    return df


def add_match_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 4.1-4.5, 4.8 — derived statistical features at match grain."""
    df = df.copy()

    # 4.1 total_goals_match — canonical replacement for the dropped raw column
    df["total_goals_match"] = df["fulltime_score_team1"] + df["fulltime_score_team2"]

    # 4.2 goal_difference
    df["goal_difference"] = (df["fulltime_score_team1"] - df["fulltime_score_team2"]).abs()

    # 4.3 is_draw — from the dataset's own winner label, not re-derived from scores
    df["is_draw"] = df["winner"] == "Draw"

    # 4.6 team1_result / team2_result
    conditions_t1 = [df["winner"] == df["team1"], df["is_draw"]]
    choices_t1 = ["Win", "Draw"]
    df["team1_result"] = np.select(conditions_t1, choices_t1, default="Loss")

    conditions_t2 = [df["winner"] == df["team2"], df["is_draw"]]
    choices_t2 = ["Win", "Draw"]
    df["team2_result"] = np.select(conditions_t2, choices_t2, default="Loss")

    # Walkover rows have no winner-vs-team logic issue (winner is still set),
    # but goal-based stats (4.1, 4.2) will be NaN there since fulltime scores
    # are null — this is intentional (see is_walkover flag).

    # 4.8 tournament_stage_order
    df["tournament_stage_order"] = df["stage"].map(STAGE_ORDER)

    return df


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Duplicate handling. Confirmed 0 duplicates in the audit; this check
    is kept in the pipeline so it remains correct if the raw file changes."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    return df, removed


def build_matches_long(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 4.7 — reshape to one row per team per match.

    BUG FIX (audit finding H1): is_historical_entity must reflect whether
    THIS ROW'S team is a defunct/historical entity, not whether EITHER side
    of the match was. Previously this column copied the match-level flag
    (team1 OR team2 historical) onto both team-perspective rows, which meant
    a modern team's row was incorrectly flagged historical whenever its
    opponent was a historical entity — corrupting top_teams_by_wins() when
    called with include_historical=False. It is now computed fresh, per row,
    from that row's own `team` column.
    """
    base_cols = [
        "match_id", "world_cup_year", "host_country", "date", "stage",
        "tournament_stage_order", "group", "result_method",
        "is_walkover", "went_to_extra_time", "went_to_penalties",
        "has_halftime_data",
    ]

    team1_view = df[base_cols + ["team1", "team2", "fulltime_score_team1", "fulltime_score_team2", "team1_result"]].copy()
    team1_view = team1_view.rename(columns={
        "team1": "team",
        "team2": "opponent",
        "fulltime_score_team1": "goals_for",
        "fulltime_score_team2": "goals_against",
        "team1_result": "result",
    })

    team2_view = df[base_cols + ["team2", "team1", "fulltime_score_team2", "fulltime_score_team1", "team2_result"]].copy()
    team2_view = team2_view.rename(columns={
        "team2": "team",
        "team1": "opponent",
        "fulltime_score_team2": "goals_for",
        "fulltime_score_team1": "goals_against",
        "team2_result": "result",
    })

    matches_long = pd.concat([team1_view, team2_view], ignore_index=True)

    # Correctly team-scoped flag: is THIS row's team itself a historical entity.
    matches_long["is_historical_entity"] = matches_long["team"].isin(HISTORICAL_ENTITIES)

    matches_long = matches_long.sort_values(["match_id", "team"]).reset_index(drop=True)
    return matches_long


def build_name_map_table() -> pd.DataFrame:
    """Stage 5.3 — the standardization map as a transparent, standalone artifact."""
    rows = [{"raw_name": raw, "standardized_name": std} for raw, std in TEAM_NAME_MAP.items()]
    return pd.DataFrame(rows)


def run_pipeline() -> dict:
    """Executes the full pipeline and returns a summary dict for verification."""
    summary = {}

    # --- Load ---
    raw_df = load_raw(RAW_PATH)
    summary["rows_before"] = len(raw_df)
    summary["cols_before"] = raw_df.shape[1]

    # --- Stage 1: Cleaning ---
    df = clean_types(raw_df)
    df = drop_unreliable_columns(df)

    # --- Stage 2: Standardizing names ---
    df = standardize_names(df)

    # --- Stage 3: Missing value handling (flags only, no imputation) ---
    df = add_missingness_flags(df)

    # --- Stage 4: Derived features ---
    df = add_match_level_features(df)

    # --- Duplicate handling ---
    df, duplicates_removed = deduplicate(df)
    summary["duplicates_removed"] = duplicates_removed

    matches_clean = df
    summary["rows_after"] = len(matches_clean)
    summary["cols_after"] = matches_clean.shape[1]

    # --- Stage 5: Analysis-ready datasets ---
    matches_long = build_matches_long(matches_clean)
    name_map = build_name_map_table()

    # --- Write outputs ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matches_clean.to_csv(OUT_MATCHES_CLEAN, index=False)
    matches_long.to_csv(OUT_MATCHES_LONG, index=False)
    name_map.to_csv(OUT_NAME_MAP, index=False)

    # --- Verification metrics ---
    summary["missing_values_after"] = matches_clean.isna().sum().to_dict()
    summary["matches_long_rows"] = len(matches_long)
    summary["derived_columns"] = [
        "is_historical_entity", "has_halftime_data", "went_to_extra_time",
        "went_to_penalties", "is_walkover", "total_goals_match",
        "goal_difference", "is_draw", "team1_result", "team2_result",
        "tournament_stage_order",
    ]
    summary["output_files"] = [str(OUT_MATCHES_CLEAN), str(OUT_MATCHES_LONG), str(OUT_NAME_MAP)]

    return summary


def write_report(summary: dict) -> None:
    lines = []
    lines.append("FIFA World Cup Pipeline — Run Report")
    lines.append("=" * 40)
    lines.append(f"Rows before: {summary['rows_before']}")
    lines.append(f"Rows after:  {summary['rows_after']}")
    lines.append(f"Columns before: {summary['cols_before']}")
    lines.append(f"Columns after (matches_clean): {summary['cols_after']}")
    lines.append(f"Duplicates removed: {summary['duplicates_removed']}")
    lines.append(f"matches_long rows: {summary['matches_long_rows']}")
    lines.append("")
    lines.append("Missing values after processing (matches_clean):")
    for col, n in summary["missing_values_after"].items():
        if n > 0:
            lines.append(f"  {col}: {n}")
    lines.append("")
    lines.append("Derived columns added:")
    for col in summary["derived_columns"]:
        lines.append(f"  - {col}")
    lines.append("")
    lines.append("Output files:")
    for f in summary["output_files"]:
        lines.append(f"  - {f}")

    OUT_REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    summary = run_pipeline()
    write_report(summary)
    print(f"Pipeline complete. Rows: {summary['rows_before']} -> {summary['rows_after']}")
    print(f"Report written to: {OUT_REPORT}")