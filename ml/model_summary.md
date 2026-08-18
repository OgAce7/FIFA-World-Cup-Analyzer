# Model Summary — World Cup Match Win Predictor

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
most recent tournaments (2018, 2022, 2026) — 1528 training
rows, 448 test rows. Neither team's prior stats nor the
split itself ever use information from the future relative to the match
being predicted.

## Performance vs. baseline

| Metric | Model | Majority-class baseline | Stratified baseline |
|---|---|---|---|
| Accuracy | 0.705 | 0.580 | 0.551 |
| F1 score | 0.598 | 0.000 | 0.464 |
| ROC-AUC | 0.770 | 0.500 | 0.539 |

## Most influential features
team_prior_avg_goals_for, opp_prior_avg_goals_for, opp_prior_matches_played — see `feature_importance.csv` for the full ranked
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
