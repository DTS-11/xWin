import json
import os
import warnings
from collections import defaultdict, deque

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

DATA_DIR = "data"
MODEL_DIR = "model"
ROLLING_WINDOW = 20
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

results = results[results["date"] >= "1995-01-01"].copy()
results = results.dropna(subset=["home_score", "away_score"])
results["home_score"] = results["home_score"].astype(int)
results["away_score"] = results["away_score"].astype(int)
results = results.sort_values("date").reset_index(drop=True)
print(f"Matches since 1995 (played): {len(results)}")

avg_home = results["home_score"].mean()
avg_away = results["away_score"].mean()
avg_home_neutral = results[results["neutral"] == True]["home_score"].mean() if results["neutral"].sum() > 0 else avg_home
avg_away_neutral = results[results["neutral"] == True]["away_score"].mean() if results["neutral"].sum() > 0 else avg_away

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


print("Engineering features (recency-weighted)...")

team_history = defaultdict(list)


def points_from_match(gf, ga):
    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0


def weighted_avg(values, weights):
    return np.average(values, weights=weights)


def rolling_features(team, is_home, default_gf=None, default_ga=None):
    hist = team_history.get(team, [])
    n = len(hist)
    if default_gf is None:
        default_gf = avg_home if is_home else avg_away
        default_ga = avg_away if is_home else avg_home

    if n == 0:
        return (default_gf, default_ga, 1.0, 0, 0.0, 1.0)

    if n < MIN_HISTORY:
        gf = np.mean([m[0] for m in hist])
        ga = np.mean([m[1] for m in hist])
        pts = np.mean([m[2] for m in hist])
        gd = np.mean([m[3] for m in hist])
        tr = 1.0
        return (gf, ga, pts, n, gd, tr)

    # Exponential recency weighting
    weights = np.exp(np.linspace(0, 1, n))
    gf_list = [m[0] for m in hist]
    ga_list = [m[1] for m in hist]
    pts_list = [m[2] for m in hist]
    gd_list = [m[3] for m in hist]

    gf = float(weighted_avg(gf_list, weights))
    ga = float(weighted_avg(ga_list, weights))
    pts = float(weighted_avg(pts_list, weights))
    gd = float(weighted_avg(gd_list, weights))

    # Form trend: recent 3 vs previous 3
    if n >= 6:
        recent = [m[4] for m in hist[-3:]]
        prev = [m[4] for m in hist[-6:-3]]
        trend = float(np.mean(recent) - np.mean(prev))
    else:
        trend = 0.0

    return (gf, ga, pts, n, gd, trend)


rows_home = []
rows_away = []
y_home = []
y_away = []

for date, group in results.groupby("date", sort=True):
    batch = []
    for idx, row in group.iterrows():
        h_team = row["home_team"]
        a_team = row["away_team"]

        is_neutral = bool(row["neutral"])
        default_h_gf = avg_home_neutral if is_neutral else avg_home
        default_h_ga = avg_away_neutral if is_neutral else avg_away
        default_a_gf = avg_away_neutral if is_neutral else avg_away
        default_a_ga = avg_home_neutral if is_neutral else avg_home

        h_gf, h_ga, h_pts, h_n, h_gd, h_tr = rolling_features(h_team, True, default_h_gf, default_h_ga)
        a_gf, a_ga, a_pts, a_n, a_gd, a_tr = rolling_features(a_team, False, default_a_gf, default_a_ga)

        h_fp = fifa_pts.get(h_team, 1300)
        a_fp = fifa_pts.get(a_team, 1300)
        h_r = fifa_rank.get(h_team, 100)
        a_r = fifa_rank.get(a_team, 100)

        # FIFA points ratio and rank diff as additional features
        fp_ratio = h_fp / max(a_fp, 1)
        rank_diff = a_r - h_r

        is_neutral_int = int(is_neutral)
        cat = get_tourn_cat(row["tournament"])
        is_wc = int(cat == "world_cup")
        is_major = int(cat == "major")
        is_friendly = int(cat == "friendly")
        is_qual = int(cat == "qualifier")

        # HOME feature vector
        home_feats = [
            h_gf,    # home attack strength
            h_ga,    # home defense weakness
            h_pts,   # home points per game
            h_gd,    # home goal difference
            h_n,     # home matches in window
            h_tr,    # home form trend
            a_gf,    # away attack strength
            a_ga,    # away defense weakness
            a_pts,   # away points per game
            a_gd,    # away goal difference
            a_n,     # away matches in window
            h_fp,    # home FIFA points
            a_fp,    # away FIFA points
            fp_ratio, # FIFA points ratio
            h_r,     # home FIFA rank
            a_r,     # away FIFA rank
            rank_diff, # rank difference (positive = home better ranked)
            is_neutral_int,
            is_wc,
            is_major,
            is_qual,
            is_friendly,
        ]
        # AWAY feature vector (swap home/away perspectives)
        away_feats = [
            a_gf,
            a_ga,
            a_pts,
            a_gd,
            a_n,
            a_tr,
            h_gf,
            h_ga,
            h_pts,
            h_gd,
            h_n,
            a_fp,
            h_fp,
            1.0 / max(fp_ratio, 0.01),
            a_r,
            h_r,
            -rank_diff,
            is_neutral_int,
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
        h_gd = hs - aws
        a_gd = aws - hs
        team_history[h_team].append((hs, aws, h_pts, h_gd, h_pts))
        team_history[a_team].append((aws, hs, a_pts, a_gd, a_pts))

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

home_model = PoissonRegressor(alpha=0.2, max_iter=1000, tol=1e-5)
away_model = PoissonRegressor(alpha=0.2, max_iter=1000, tol=1e-5)

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

metadata = {
    "avg_home_goals": float(avg_home),
    "avg_away_goals": float(avg_away),
    "avg_home_goals_neutral": float(avg_home_neutral),
    "avg_away_goals_neutral": float(avg_away_neutral),
    "home_mae": float(home_mae),
    "away_mae": float(away_mae),
    "rolling_window": ROLLING_WINDOW,
    "min_history": MIN_HISTORY,
    "trained_on": results["date"].max().strftime("%Y-%m-%d"),
    "feature_version": 2,
}
json.dump(metadata, open(f"{MODEL_DIR}/metadata.json", "w"))
json.dump({k: v for k, v in fifa_pts.items()}, open(f"{MODEL_DIR}/fifa_pts.json", "w"))
json.dump(
    {k: v for k, v in fifa_rank.items()}, open(f"{MODEL_DIR}/fifa_rank.json", "w")
)
json.dump(name_map, open(f"{MODEL_DIR}/name_map.json", "w"))

team_form = {}
for team, hist in team_history.items():
    h = hist
    if len(h) >= MIN_HISTORY:
        team_form[team] = {
            "gf_avg": float(np.mean([m[0] for m in h])),
            "ga_avg": float(np.mean([m[1] for m in h])),
            "pts_avg": float(np.mean([m[2] for m in h])),
            "gd_avg": float(np.mean([m[3] for m in h])),
            "n_matches": len(h),
        }
json.dump(team_form, open(f"{MODEL_DIR}/team_form.json", "w"))
print(f"Team form saved for {len(team_form)} teams")

print(f"Models saved to {MODEL_DIR}/")
print("Done!")
