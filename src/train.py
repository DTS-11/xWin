import os
import warnings
from collections import defaultdict, deque

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

DATA_DIR = "data"
MODEL_DIR = "model"
ROLLING_WINDOW = 12
MIN_HISTORY = 3

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading data...")
results = pd.read_csv(f"{DATA_DIR}/results.csv")
former = pd.read_csv(f"{DATA_DIR}/former_names.csv")
ranking = pd.read_csv(f"{DATA_DIR}/fifa_ranking_2022-10-06.csv")

name_map = {}
for _, r in former.iterrows():
    name_map[r["former"].lower()] = r["current"]


def norm(name):
    n = name.strip()
    return name_map.get(n.lower(), n)


results["home_team"] = results["home_team"].apply(norm)
results["away_team"] = results["away_team"].apply(norm)
results["date"] = pd.to_datetime(results["date"])

ranking["team"] = ranking["team"].apply(norm)
fifa_pts = dict(zip(ranking["team"], ranking["points"]))
fifa_rank = dict(zip(ranking["team"], ranking["rank"]))

results = results[results["date"] >= "1990-01-01"].copy()
# Drop future/unplayed matches (NaN scores)
results = results.dropna(subset=["home_score", "away_score"])
results["home_score"] = results["home_score"].astype(int)
results["away_score"] = results["away_score"].astype(int)
results = results.sort_values("date").reset_index(drop=True)
print(f"Matches since 1990 (played): {len(results)}")

avg_home = results["home_score"].mean()
avg_away = results["away_score"].mean()

tournament_map = {
    "FIFA World Cup": "world_cup",
    "UEFA Euro": "major",
    "Copa América": "major",
    "African Cup of Nations": "major",
    "AFC Asian Cup": "major",
    "Gold Cup": "major",
    "OFC Nations Cup": "major",
    "FIFA World Cup qualification": "qualifier",
    "UEFA Euro qualification": "qualifier",
    "African Cup of Nations qualification": "qualifier",
    "AFC Asian Cup qualification": "qualifier",
    "FIFA World Cup qualification play-off": "qualifier",
    "Friendly": "friendly",
}


def get_tourn_cat(t):
    return tournament_map.get(t, "other")


print("Engineering features...")

team_history = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))


def points_from_match(gf, ga):
    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0


def rolling_features(team, is_home):
    hist = list(team_history[team])
    n = len(hist)
    if n < MIN_HISTORY:
        return (
            avg_home if is_home else avg_away,
            avg_away if is_home else avg_home,
            1.0,
            n,
        )
    gf = np.mean([m[0] for m in hist])
    ga = np.mean([m[1] for m in hist])
    pts = np.mean([m[2] for m in hist])
    return (gf, ga, pts, n)


rows_home = []
rows_away = []
y_home = []
y_away = []

for date, group in results.groupby("date", sort=True):
    batch = []
    for idx, row in group.iterrows():
        h_team = row["home_team"]
        a_team = row["away_team"]

        h_gf, h_ga, h_pts, h_n = rolling_features(h_team, True)
        a_gf, a_ga, a_pts, a_n = rolling_features(a_team, False)

        h_fp = fifa_pts.get(h_team, 1300)
        a_fp = fifa_pts.get(a_team, 1300)
        h_r = fifa_rank.get(h_team, 100)
        a_r = fifa_rank.get(a_team, 100)

        is_neutral = int(row["neutral"])
        cat = get_tourn_cat(row["tournament"])
        is_wc = int(cat == "world_cup")
        is_major = int(cat == "major")
        is_friendly = int(cat == "friendly")
        is_qual = int(cat == "qualifier")

        # HOME goals: home attack + away defense, AWAY goals: swapped
        home_feats = [
            h_gf,
            h_ga,
            h_pts,
            h_n,
            a_gf,
            a_ga,
            a_pts,
            a_n,
            h_fp,
            a_fp,
            h_r,
            a_r,
            is_neutral,
            is_wc,
            is_major,
            is_qual,
            is_friendly,
        ]
        away_feats = [
            a_gf,
            a_ga,
            a_pts,
            a_n,
            h_gf,
            h_ga,
            h_pts,
            h_n,
            a_fp,
            h_fp,
            a_r,
            h_r,
            is_neutral,
            is_wc,
            is_major,
            is_qual,
            is_friendly,
        ]

        batch.append((home_feats, away_feats, row["home_score"], row["away_score"]))

    for home_feats, away_feats, hs, aws in batch:
        rows_home.append(home_feats)
        rows_away.append(away_feats)
        y_home.append(hs)
        y_away.append(aws)

    for idx, row in group.iterrows():
        h_team = row["home_team"]
        a_team = row["away_team"]
        hs, aws = row["home_score"], row["away_score"]
        h_pts = points_from_match(hs, aws)
        a_pts = points_from_match(aws, hs)
        team_history[h_team].append((hs, aws, h_pts))
        team_history[a_team].append((aws, hs, a_pts))

X_home = np.array(rows_home)
X_away = np.array(rows_away)
y_home = np.array(y_home)
y_away = np.array(y_away)

valid_h = ~(np.isnan(X_home).any(axis=1) | np.isnan(y_home))
valid_a = ~(np.isnan(X_away).any(axis=1) | np.isnan(y_away))
X_home = X_home[valid_h]
y_home = y_home[valid_h]
X_away = X_away[valid_a]
y_away = y_away[valid_a]

print(f"Training samples - home: {len(X_home)}, away: {len(X_away)}")

print("Training Poisson regression models...")

home_model = PoissonRegressor(alpha=0.3, max_iter=500, tol=1e-4)
away_model = PoissonRegressor(alpha=0.3, max_iter=500, tol=1e-4)

home_model.fit(X_home, y_home)
away_model.fit(X_away, y_away)

home_pred = home_model.predict(X_home)
away_pred = away_model.predict(X_away)
home_mae = mean_absolute_error(y_home, home_pred)
away_mae = mean_absolute_error(y_away, away_pred)
print(f"Home goals MAE: {home_mae:.3f}")
print(f"Away goals MAE: {away_mae:.3f}")

print("Saving model...")
joblib.dump(home_model, f"{MODEL_DIR}/home_model.pkl")
joblib.dump(away_model, f"{MODEL_DIR}/away_model.pkl")

import json

metadata = {
    "avg_home_goals": float(avg_home),
    "avg_away_goals": float(avg_away),
    "home_mae": float(home_mae),
    "away_mae": float(away_mae),
    "rolling_window": ROLLING_WINDOW,
    "min_history": MIN_HISTORY,
    "trained_on": results["date"].max().strftime("%Y-%m-%d"),
}
json.dump(metadata, open(f"{MODEL_DIR}/metadata.json", "w"))
json.dump({k: v for k, v in fifa_pts.items()}, open(f"{MODEL_DIR}/fifa_pts.json", "w"))
json.dump(
    {k: v for k, v in fifa_rank.items()}, open(f"{MODEL_DIR}/fifa_rank.json", "w")
)
json.dump(name_map, open(f"{MODEL_DIR}/name_map.json", "w"))

team_form = {}
for team, hist in team_history.items():
    h = list(hist)
    if len(h) >= MIN_HISTORY:
        team_form[team] = {
            "gf_avg": float(np.mean([m[0] for m in h])),
            "ga_avg": float(np.mean([m[1] for m in h])),
            "pts_avg": float(np.mean([m[2] for m in h])),
            "n_matches": len(h),
        }
json.dump(team_form, open(f"{MODEL_DIR}/team_form.json", "w"))
print(f"Team form saved for {len(team_form)} teams")

print(f"Models saved to {MODEL_DIR}/")
print("Done!")
