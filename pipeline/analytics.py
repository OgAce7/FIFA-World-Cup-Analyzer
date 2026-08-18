"""
FIFA World Cup Dashboard — Analytical Calculations
====================================================

Scope (per project spec): aggregations, statistics, filtering,
country-specific calculations, tournament-specific calculations,
global calculations — one function per approved visualization
(G1-G6, C1-C5), plus shared validation helpers.

Explicitly OUT of scope: frontend components, API, ML models,
modification of raw/processed data (all functions are read-only
transforms over the already-processed DataFrames).

Design principles:
    - Every function accepts data as a parameter (a DataFrame) —
      nothing is hardcoded to a specific country/year/file path.
    - Country and year selection are validated up front and raise
      clear, typed exceptions on invalid input.
    - Functions return tidy DataFrames ready to hand to a charting
      layer — no chart objects are created here.

Usage:
    import pandas as pd
    from analytics import *

    matches = pd.read_csv("data/processed/matches_clean.csv")
    long_df = pd.read_csv("data/processed/matches_long.csv")

    goals_per_tournament(matches)
    country_record_by_year(long_df, "Brazil")
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class InvalidCountryError(ValueError):
    """Raised when a selected country does not appear in the dataset."""


class InvalidYearError(ValueError):
    """Raised when a selected World Cup year does not appear in the dataset."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_country(long_df: pd.DataFrame, country: str) -> str:
    """
    Confirm `country` appears in the dataset's `team` column.

    Parameters
    ----------
    long_df : matches_long-shaped DataFrame (must contain a `team` column)
    country : the country name to validate, e.g. "Brazil"

    Returns
    -------
    The validated country string (unchanged), for convenient chaining.

    Raises
    ------
    InvalidCountryError if the country never appears as a team in the data.
    """
    known_teams = set(long_df["team"].unique())
    if country not in known_teams:
        raise InvalidCountryError(
            f"'{country}' is not a recognized country/team in the dataset. "
            f"Example valid values: {sorted(known_teams)[:5]}..."
        )
    return country


def validate_year(matches_df: pd.DataFrame, year: int) -> int:
    """
    Confirm `year` appears in the dataset's `world_cup_year` column.

    Parameters
    ----------
    matches_df : any DataFrame containing a `world_cup_year` column
    year : the tournament year to validate, e.g. 2018

    Returns
    -------
    The validated year (unchanged), for convenient chaining.

    Raises
    ------
    InvalidYearError if the year is not a tournament present in the data.
    """
    known_years = set(int(y) for y in matches_df["world_cup_year"].unique())
    if year not in known_years:
        raise InvalidYearError(
            f"{year} is not a recognized World Cup year in the dataset. "
            f"Valid years: {sorted(known_years)}"
        )
    return year


def apply_year_filter(df: pd.DataFrame, year: int | None) -> pd.DataFrame:
    """
    Shared hard-filter helper: returns rows for a single tournament year,
    or the full DataFrame unchanged if year is None ("All years").
    Does not validate — call validate_year() first if the year comes
    from user input.
    """
    if year is None:
        return df
    return df[df["world_cup_year"] == year]


def apply_historical_entity_filter(long_df: pd.DataFrame, include_historical: bool = True) -> pd.DataFrame:
    """
    Optional toggle used by G4: include/exclude rows involving defunct
    historical entities (West Germany, Soviet Union, etc.).
    """
    if include_historical:
        return long_df
    return long_df[~long_df["is_historical_entity"]]


# ---------------------------------------------------------------------------
# GLOBAL VIEW CALCULATIONS
# ---------------------------------------------------------------------------

def goals_per_tournament(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    G1 (default global scatter). Average goals per match, per tournament.

    Data required: matches_clean
    Returns: DataFrame [world_cup_year, avg_goals_per_match, matches_played]
    """
    valid = matches_df[matches_df["total_goals_match"].notna()]
    result = (
        valid.groupby("world_cup_year")
        .agg(
            avg_goals_per_match=("total_goals_match", "mean"),
            matches_played=("match_id", "count"),
        )
        .reset_index()
        .sort_values("world_cup_year")
    )
    result["avg_goals_per_match"] = result["avg_goals_per_match"].round(2)
    return result


def matches_by_stage(matches_df: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    """
    G2. Count of matches per tournament stage, optionally scoped to one year.

    Data required: matches_clean
    Filters: world_cup_year (optional; None = all tournaments aggregated)
    Returns: DataFrame [stage, tournament_stage_order, match_count], ordered by stage progression
    """
    if year is not None:
        validate_year(matches_df, year)
    scoped = apply_year_filter(matches_df, year)

    result = (
        scoped.groupby(["stage", "tournament_stage_order"])
        .size()
        .reset_index(name="match_count")
        .sort_values("tournament_stage_order")
        .reset_index(drop=True)
    )
    return result


def result_method_breakdown(matches_df: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    """
    G3. Share of matches by result_method (Normal Time, Draw, Penalties, etc).

    Data required: matches_clean
    Filters: world_cup_year (optional)
    Returns: DataFrame [result_method, match_count, pct_of_matches]
    """
    if year is not None:
        validate_year(matches_df, year)
    scoped = apply_year_filter(matches_df, year)

    counts = scoped["result_method"].value_counts().reset_index()
    counts.columns = ["result_method", "match_count"]
    counts["pct_of_matches"] = (counts["match_count"] / counts["match_count"].sum() * 100).round(1)
    return counts


def top_teams_by_wins(
    long_df: pd.DataFrame,
    year: int | None = None,
    top_n: int = 10,
    include_historical: bool = True,
) -> pd.DataFrame:
    """
    G4. Top N teams ranked by total win count.

    Data required: matches_long
    Filters: world_cup_year (optional), include_historical (optional toggle)
    Returns: DataFrame [team, wins], sorted descending, top_n rows
    """
    if year is not None:
        validate_year(long_df, year)
    scoped = apply_year_filter(long_df, year)
    scoped = apply_historical_entity_filter(scoped, include_historical)

    wins = (
        scoped[scoped["result"] == "Win"]
        .groupby("team")
        .size()
        .reset_index(name="wins")
        .sort_values("wins", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return wins


def host_country_summary(matches_df: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    """
    G5. Number of tournaments hosted per country (one row per tournament,
    deduplicated, since host_country repeats across every match row).

    Data required: matches_clean
    Filters: world_cup_year (optional; scopes to a single tournament's host)
    Returns: DataFrame [host_country, tournaments_hosted, years_hosted]
    """
    if year is not None:
        validate_year(matches_df, year)
    scoped = apply_year_filter(matches_df, year)

    tournaments = scoped[["world_cup_year", "host_country"]].drop_duplicates()
    result = (
        tournaments.groupby("host_country")["world_cup_year"]
        .apply(lambda s: sorted(s.unique().tolist()))
        .reset_index(name="years_hosted")
    )
    result["tournaments_hosted"] = result["years_hosted"].apply(len)
    return result[["host_country", "tournaments_hosted", "years_hosted"]].sort_values(
        "tournaments_hosted", ascending=False
    ).reset_index(drop=True)


def goal_margin_distribution(matches_df: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    """
    G6. Distribution of matches by goal_difference, binned into 0,1,2,3,4+.

    Data required: matches_clean
    Filters: world_cup_year (optional)
    Returns: DataFrame [margin_bin, match_count]
    """
    if year is not None:
        validate_year(matches_df, year)
    scoped = apply_year_filter(matches_df, year)

    valid = scoped[scoped["goal_difference"].notna()].copy()

    def bin_margin(diff: float) -> str:
        diff = int(diff)
        if diff >= 4:
            return "4+"
        return str(diff)

    valid["margin_bin"] = valid["goal_difference"].apply(bin_margin)
    order = ["0", "1", "2", "3", "4+"]
    result = (
        valid.groupby("margin_bin")
        .size()
        .reindex(order, fill_value=0)
        .reset_index(name="match_count")
        .rename(columns={"index": "margin_bin"})
    )
    return result


# ---------------------------------------------------------------------------
# COUNTRY VIEW CALCULATIONS
# ---------------------------------------------------------------------------

def country_record_by_year(
    long_df: pd.DataFrame, country: str, year: int | None = None
) -> pd.DataFrame:
    """
    C1. Win/Draw/Loss counts per tournament year for one country.

    Data required: matches_long
    Filters: country (required), world_cup_year (optional)
    Returns: DataFrame [world_cup_year, Win, Draw, Loss]
    """
    validate_country(long_df, country)
    if year is not None:
        validate_year(long_df, year)

    scoped = long_df[long_df["team"] == country]
    scoped = apply_year_filter(scoped, year)

    result = (
        scoped.groupby(["world_cup_year", "result"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["Win", "Draw", "Loss"]:
        if col not in result.columns:
            result[col] = 0
    return result[["world_cup_year", "Win", "Draw", "Loss"]].sort_values("world_cup_year").reset_index(drop=True)


def country_goals_for_against(
    long_df: pd.DataFrame, country: str, year: int | None = None
) -> pd.DataFrame:
    """
    C2. Total goals scored vs conceded per tournament year for one country.

    Data required: matches_long
    Filters: country (required), world_cup_year (optional)
    Returns: DataFrame [world_cup_year, goals_for, goals_against, goal_diff]
    """
    validate_country(long_df, country)
    if year is not None:
        validate_year(long_df, year)

    scoped = long_df[long_df["team"] == country]
    scoped = apply_year_filter(scoped, year)

    result = (
        scoped.groupby("world_cup_year")
        .agg(goals_for=("goals_for", "sum"), goals_against=("goals_against", "sum"))
        .reset_index()
        .sort_values("world_cup_year")
    )
    result["goal_diff"] = result["goals_for"] - result["goals_against"]
    return result.reset_index(drop=True)


def country_deepest_stage_by_year(
    long_df: pd.DataFrame, country: str, year: int | None = None
) -> pd.DataFrame:
    """
    C3. Deepest tournament stage reached per year for one country
    (max tournament_stage_order across that country's matches that year).

    Data required: matches_long
    Filters: country (required), world_cup_year (optional)
    Returns: DataFrame [world_cup_year, deepest_stage_order, deepest_stage]
    """
    validate_country(long_df, country)
    if year is not None:
        validate_year(long_df, year)

    scoped = long_df[long_df["team"] == country]
    scoped = apply_year_filter(scoped, year)

    idx = scoped.groupby("world_cup_year")["tournament_stage_order"].idxmax()
    deepest = scoped.loc[idx, ["world_cup_year", "tournament_stage_order", "stage"]]
    deepest = deepest.rename(
        columns={"tournament_stage_order": "deepest_stage_order", "stage": "deepest_stage"}
    ).sort_values("world_cup_year").reset_index(drop=True)
    return deepest


def country_result_method_breakdown(
    long_df: pd.DataFrame, country: str, year: int | None = None
) -> pd.DataFrame:
    """
    C4. Share of this country's matches by result_method.

    Data required: matches_long (already carries result_method per team-row)
    Filters: country (required), world_cup_year (optional)
    Returns: DataFrame [result_method, match_count, pct_of_matches]
    """
    validate_country(long_df, country)
    if year is not None:
        validate_year(long_df, year)

    scoped = long_df[long_df["team"] == country]
    scoped = apply_year_filter(scoped, year)

    counts = scoped["result_method"].value_counts().reset_index()
    counts.columns = ["result_method", "match_count"]
    counts["pct_of_matches"] = (counts["match_count"] / counts["match_count"].sum() * 100).round(1)
    return counts


def country_top_opponents(
    long_df: pd.DataFrame, country: str, year: int | None = None, top_n: int = 8
) -> pd.DataFrame:
    """
    C5. Most frequently faced opponents for one country.

    Data required: matches_long
    Filters: country (required), world_cup_year (optional)
    Returns: DataFrame [opponent, matches_played], top_n rows, sorted descending
    """
    validate_country(long_df, country)
    if year is not None:
        validate_year(long_df, year)

    scoped = long_df[long_df["team"] == country]
    scoped = apply_year_filter(scoped, year)

    result = (
        scoped.groupby("opponent")
        .size()
        .reset_index(name="matches_played")
        .sort_values("matches_played", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return result


# ---------------------------------------------------------------------------
# SHARED / CROSS-CUTTING CALCULATIONS
# (not tied to a single chart, but reusable building blocks referenced
#  by multiple charts above and useful for future summary stats)
# ---------------------------------------------------------------------------

def list_valid_countries(long_df: pd.DataFrame) -> list[str]:
    """Returns every valid, selectable country/team name in the dataset."""
    return sorted(long_df["team"].unique().tolist())


def list_valid_years(matches_df: pd.DataFrame) -> list[int]:
    """Returns every valid, selectable World Cup year in the dataset."""
    return sorted(int(y) for y in matches_df["world_cup_year"].unique())
