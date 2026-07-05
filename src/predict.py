import json
import os
import re
import sys

import joblib
import numpy as np
from scipy.stats import poisson

MODEL_DIR = "model"


class WorldCupPredictor:
    def __init__(self):
        self.home_model = joblib.load(f"{MODEL_DIR}/home_model.pkl")
        self.away_model = joblib.load(f"{MODEL_DIR}/away_model.pkl")
        self.metadata = json.load(open(f"{MODEL_DIR}/metadata.json"))
        self.fifa_pts = json.load(open(f"{MODEL_DIR}/fifa_pts.json"))
        self.fifa_rank = json.load(open(f"{MODEL_DIR}/fifa_rank.json"))
        self.team_form = json.load(open(f"{MODEL_DIR}/team_form.json"))
        self.name_map = json.load(open(f"{MODEL_DIR}/name_map.json"))

        all_teams = set(self.fifa_pts.keys()) | set(self.team_form.keys())
        self.all_teams = sorted(all_teams)
        self._team_lookup = {t.lower(): t for t in self.all_teams}

    def _norm(self, name):
        n = name.strip()
        return self.name_map.get(n.lower(), n)

    def _fuzzy_match(self, name):
        key = name.lower().strip()
        if key in self._team_lookup:
            return self._team_lookup[key]
        matches = [t for t in self.all_teams if key in t.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"  Did you mean one of: {', '.join(matches[:5])}?")
            return matches[0]
        return name

    def _team_features(self, team, is_home=True, is_neutral=False):
        form = self.team_form.get(team)
        if form:
            gd = form.get("gd_avg", 0.0)
            return (
                form["gf_avg"],
                form["ga_avg"],
                form["pts_avg"],
                form["n_matches"],
                gd,
                0.0,
            )
        if is_neutral:
            avg_h = self.metadata.get(
                "avg_home_goals_neutral", self.metadata["avg_home_goals"]
            )
            avg_a = self.metadata.get(
                "avg_away_goals_neutral", self.metadata["avg_away_goals"]
            )
        else:
            avg_h = self.metadata["avg_home_goals"]
            avg_a = self.metadata["avg_away_goals"]
        gf = avg_h if is_home else avg_a
        ga = avg_a if is_home else avg_h
        return (gf, ga, 1.0, 0, 0.0, 1.0)

    def predict(
        self,
        home_team,
        away_team,
        neutral=False,
        tournament="FIFA World Cup",
        is_knockout=False,
    ):
        h_team = self._fuzzy_match(home_team)
        a_team = self._fuzzy_match(away_team)

        h_fp = float(self.fifa_pts.get(h_team, 1300))
        a_fp = float(self.fifa_pts.get(a_team, 1300))
        h_r = float(self.fifa_rank.get(h_team, 100))
        a_r = float(self.fifa_rank.get(a_team, 100))

        fp_ratio = h_fp / max(a_fp, 1)
        rank_diff = a_r - h_r

        h_gf, h_ga, h_pts, h_n, h_gd, h_tr = self._team_features(h_team, True, neutral)
        a_gf, a_ga, a_pts, a_n, a_gd, a_tr = self._team_features(a_team, False, neutral)

        major_tourns = {
            "FIFA World Cup",
            "UEFA Euro",
            "Copa América",
            "African Cup of Nations",
            "AFC Asian Cup",
            "Gold Cup",
            "OFC Nations Cup",
        }
        qual_tourns = {
            "FIFA World Cup qualification",
            "UEFA Euro qualification",
            "African Cup of Nations qualification",
            "AFC Asian Cup qualification",
            "FIFA World Cup qualification play-off",
        }

        is_wc = int(tournament == "FIFA World Cup")
        is_major = int(tournament in major_tourns and tournament != "FIFA World Cup")
        is_qual = int(tournament in qual_tourns)
        is_friendly = int(tournament == "Friendly")

        home_feats = np.array(
            [
                [
                    h_gf,
                    h_ga,
                    h_pts,
                    h_gd,
                    h_n,
                    h_tr,
                    a_gf,
                    a_ga,
                    a_pts,
                    a_gd,
                    a_n,
                    h_fp,
                    a_fp,
                    fp_ratio,
                    h_r,
                    a_r,
                    rank_diff,
                    int(neutral),
                    is_wc,
                    is_major,
                    is_qual,
                    is_friendly,
                ]
            ]
        )
        away_feats = np.array(
            [
                [
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
                    int(neutral),
                    is_wc,
                    is_major,
                    is_qual,
                    is_friendly,
                ]
            ]
        )

        exp_home = self.home_model.predict(home_feats)[0]
        exp_away = self.away_model.predict(away_feats)[0]

        return self._score_probs(
            exp_home, exp_away, h_team, a_team, is_knockout=is_knockout
        )

    def _score_probs(self, lam_h, lam_a, h_team, a_team, is_knockout=False):
        max_goals = 8
        probs = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                probs[i][j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
        probs /= probs.sum()

        win_h = np.sum(
            probs * (np.arange(max_goals + 1)[:, None] > np.arange(max_goals + 1))
        )
        win_a = np.sum(
            probs * (np.arange(max_goals + 1)[:, None] < np.arange(max_goals + 1))
        )
        draw = 1 - win_h - win_a

        if is_knockout:
            total = win_h + win_a
            if total > 0:
                win_h_adj = win_h / total
                win_a_adj = win_a / total
            else:
                win_h_adj = win_a_adj = 0.5
            win_probabilities = {
                h_team: round(win_h_adj * 100, 1),
                a_team: round(win_a_adj * 100, 1),
            }

            ko_probs = probs.copy()
            np.fill_diagonal(ko_probs, 0)
            ko_probs /= ko_probs.sum()

            best_idx = np.unravel_index(ko_probs.argmax(), ko_probs.shape)
            best_score = (int(best_idx[0]), int(best_idx[1]))
            best_prob = float(ko_probs[best_idx])

            top5_idx = np.argsort(ko_probs.ravel())[-5:][::-1]
            top5 = []
            for idx in top5_idx:
                i, j = np.unravel_index(idx, ko_probs.shape)
                top5.append(
                    {
                        "score": f"{h_team} {i} : {j} {a_team}",
                        "probability": float(ko_probs[i][j]) * 100,
                    }
                )

            result = {
                "home_team": h_team,
                "away_team": a_team,
                "expected_goals": {
                    h_team: round(lam_h, 2),
                    a_team: round(lam_a, 2),
                },
                "most_likely_score": {
                    "score": f"{h_team} {best_score[0]} : {best_score[1]} {a_team}",
                    "probability": round(best_prob * 100, 1),
                },
                "win_probabilities": win_probabilities,
                "top_5_scores": top5,
                "is_knockout": True,
                "note": "Draw eliminated for knockout match. Win probabilities normalized.",
            }
            return result

        best_idx = np.unravel_index(probs.argmax(), probs.shape)
        best_score = (int(best_idx[0]), int(best_idx[1]))
        best_prob = float(probs[best_idx])

        top5_idx = np.argsort(probs.ravel())[-5:][::-1]
        top5 = []
        for idx in top5_idx:
            i, j = np.unravel_index(idx, probs.shape)
            top5.append(
                {
                    "score": f"{h_team} {i} : {j} {a_team}",
                    "probability": float(probs[i][j]) * 100,
                }
            )

        return {
            "home_team": h_team,
            "away_team": a_team,
            "expected_goals": {
                h_team: round(lam_h, 2),
                a_team: round(lam_a, 2),
            },
            "most_likely_score": {
                "score": f"{h_team} {best_score[0]} : {best_score[1]} {a_team}",
                "probability": round(best_prob * 100, 1),
            },
            "win_probabilities": {
                h_team: round(win_h * 100, 1),
                "Draw": round(draw * 100, 1),
                a_team: round(win_a * 100, 1),
            },
            "top_5_scores": top5,
        }

    def list_teams(self, query=""):
        q = query.lower()
        return [t for t in self.all_teams if q in t.lower()]


def pretty_print(result):
    print()
    label = "KNOCKOUT PREDICTION" if result.get("is_knockout") else "PREDICTION"
    print(f"{'=' * 55}")
    w = result["most_likely_score"]["score"]
    p = result["most_likely_score"]["probability"]
    print(f"  {w}")
    print(f"  (Probability: {p}%)")
    print(f"  [{label}]")
    print(f"{'=' * 55}")
    print()
    h = result["home_team"]
    a = result["away_team"]
    print(
        f"  Expected goals: {h} {result['expected_goals'][h]} - {result['expected_goals'][a]} {a}"
    )
    print()
    print(f"  Win probabilities:")
    for team, prob in result["win_probabilities"].items():
        print(f"    {team:25s} {prob:5.1f}%")
    print()
    print(f"  Top 5 most likely scores:")
    for s in result["top_5_scores"][:5]:
        print(f"    {s['score']:35s} {s['probability']:.1f}%")
    print()


def interactive(predictor):
    print()
    print('World Cup Predictor (type "exit" to quit, "teams" to list teams)')
    print()
    while True:
        try:
            inp = input("  Predict (home vs away): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not inp:
            continue
        if inp.lower() in ("exit", "quit", "q"):
            break
        if inp.lower() in ("teams", "list"):
            teams = predictor.list_teams()
            for t in teams:
                print(f"    {t}")
            continue
        is_ko = inp.startswith("ko:") or " --knockout" in inp
        if inp.startswith("ko:"):
            inp = inp[3:].strip()
        inp = inp.replace(" --knockout", "").strip()
        parts = re.split(r"\s+vs\s+|\s+v\.\s+|\s{2,}", inp)
        if len(parts) >= 2:
            home = parts[0].strip()
            away = parts[1].strip()
        else:
            parts = inp.split()
            if len(parts) >= 2:
                home, away = parts[0], parts[1]
            else:
                print("  Usage: <home_team> vs <away_team>")
                continue
        result = predictor.predict(home, away, is_knockout=is_ko)
        pretty_print(result)


def main():
    predictor = WorldCupPredictor()

    if len(sys.argv) > 1 and sys.argv[1] in ("--interactive", "-i"):
        interactive(predictor)
        return

    if len(sys.argv) > 1 and sys.argv[1] in ("--teams", "--list"):
        teams = predictor.list_teams(" ".join(sys.argv[2:]))
        for t in teams:
            try:
                print(t)
            except UnicodeEncodeError:
                print(t.encode("ascii", "replace").decode())
        return

    if len(sys.argv) < 3:
        print("Usage:")
        print(
            "  python predict.py <home_team> <away_team> [--neutral] [--tournament NAME] [--knockout]"
        )
        print("  python predict.py --interactive")
        print("  python predict.py --teams [query]")
        print()
        print("Examples:")
        print("  python src/predict.py Portugal Argentina")
        print("  python src/predict.py Brazil France --neutral")
        print(
            '  python src/predict.py "South Korea" Germany --tournament "FIFA World Cup"'
        )
        print("  python src/predict.py England France --knockout")
        print()
        return

    neutral = "--neutral" in sys.argv
    is_knockout = "--knockout" in sys.argv

    tournament = "FIFA World Cup"
    if "--tournament" in sys.argv:
        idx = sys.argv.index("--tournament")
        if idx + 1 < len(sys.argv):
            tournament = sys.argv[idx + 1]

    args = [a for a in sys.argv[1:] if not a.startswith("--") and a != tournament]

    if len(args) < 2:
        print("Please provide two team names.")
        return

    home_team, away_team = args[0], args[1]
    result = predictor.predict(
        home_team,
        away_team,
        neutral=neutral,
        tournament=tournament,
        is_knockout=is_knockout,
    )
    pretty_print(result)


if __name__ == "__main__":
    main()
