import sys

from predict import WorldCupPredictor, pretty_print
from wc2026_api import (
    KO_STAGES,
    STAGE_MAP,
    Match,
    get_finished_matches,
    get_groups,
    get_knockout_matches,
    get_matches,
    get_teams,
    get_upcoming_matches,
)


def predict_match(predictor, match: Match):
    return predictor.predict(
        match.home_team,
        match.away_team,
        neutral=True,
        tournament="FIFA World Cup",
        is_knockout=match.is_knockout,
    )


def show_upcoming(predictor):
    matches = get_upcoming_matches()
    if not matches:
        print("  No upcoming matches found.")
        return
    matches.sort(key=lambda m: m.local_date or "")
    print(f"\n  {'=' * 55}")
    print(f"  UPCOMING MATCHES ({len(matches)})")
    print(f"  {'=' * 55}\n")
    for m in matches:
        stage_str = f"[{m.stage_name}]" if m.stage != "group" else ""
        print(f"  {m.home_team:25s} vs {m.away_team:25s}  {stage_str}")
        if m.local_date:
            print(f"  {' ':25s} {m.local_date}")
        print()


def show_results(predictor):
    matches = get_finished_matches()
    if not matches:
        print("  No finished matches found.")
        return
    matches.sort(key=lambda m: m.local_date or "", reverse=True)
    print(f"\n  {'=' * 55}")
    print(f"  FINISHED MATCHES ({len(matches)})")
    print(f"  {'=' * 55}\n")
    for m in matches[:20]:
        stage_str = f"[{m.stage_name}]" if m.stage != "group" else ""
        score = f"{m.home_score} - {m.away_score}" if m.home_score is not None else "?"
        print(f"  {m.home_team:25s} {score:7s} {m.away_team:25s}  {stage_str}")
    if len(matches) > 20:
        print(f"  ... and {len(matches) - 20} more")


def show_standings():
    groups = get_groups()
    teams = {t.team_id: t.name for t in get_teams()}
    if not groups:
        print("  No standings data available.")
        return
    print(f"\n  {'=' * 55}")
    print(f"  GROUP STANDINGS")
    print(f"  {'=' * 55}\n")
    for group_name in sorted(groups.keys()):
        print(f"  Group {group_name}")
        print(
            f"  {'Team':25s} {'P':3s} {'W':3s} {'D':3s} {'L':3s} {'GF':3s} {'GA':3s} {'GD':4s} {'Pts':3s}"
        )
        print(f"  {'-' * 48}")
        standings = sorted(groups[group_name], key=lambda x: x["pts"], reverse=True)
        for t in standings:
            name = teams.get(t["team_id"], t["team_id"])
            print(
                f"  {name:25s} {t['mp']:3d} {t['w']:3d} {t['d']:3d} {t['l']:3d} "
                f"{t['gf']:3d} {t['ga']:3d} {t['gd']:4d} {t['pts']:3d}"
            )
        print()


def _is_tbd(team_name: str) -> bool:
    return not team_name or team_name.lower().startswith("winner match")


def show_knockout_bracket(predictor):
    ko_matches = get_knockout_matches()
    if not ko_matches:
        print("  No knockout matches yet.")
        return
    stage_order = ["r32", "r16", "qf", "sf", "third", "final"]
    ko_matches.sort(
        key=lambda m: stage_order.index(m.stage) if m.stage in stage_order else 99
    )

    print(f"\n  {'=' * 55}")
    print(f"  KNOCKOUT BRACKET")
    print(f"  {'=' * 55}\n")

    for stage in stage_order:
        stage_matches = [m for m in ko_matches if m.stage == stage]
        if not stage_matches:
            continue
        print(f"  --- {STAGE_MAP.get(stage, stage)} ---\n")
        for m in stage_matches:
            if _is_tbd(m.home_team) or _is_tbd(m.away_team):
                print(f"  TBD vs TBD")
                print()
                continue
            if m.finished and m.home_score is not None:
                print(
                    f"  {m.home_team:25s} {m.home_score}-{m.away_score}  {m.away_team:25s}"
                )
            else:
                pred = predict_match(predictor, m)
                h = pred["home_team"]
                a = pred["away_team"]
                hw = pred["win_probabilities"][h]
                aw = pred["win_probabilities"][a]
                ms = pred["most_likely_score"]["score"]
                print(f"  {h:25s} vs {a:25s}")
                print(f"  {' ':25s} Win: {hw:5.1f}% | Draw -> normalized")
                print(f"  {' ':25s} Most likely: {ms}")
            print()


def interactive_api(predictor):
    print()
    print("  xWin - FIFA World Cup 2026 Predictor")
    print("  Commands: upcoming, results, standings, bracket, groups,")
    print("            predict <home> vs <away> [--knockout], teams, help, exit")
    print()
    while True:
        try:
            inp = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not inp:
            continue
        if inp.lower() in ("exit", "quit", "q"):
            break
        if inp.lower() in ("help", "h"):
            print("  Commands:")
            print("    upcoming          - Show upcoming matches")
            print("    results           - Show recent results")
            print("    standings/groups  - Show group standings")
            print("    bracket           - Show knockout bracket with predictions")
            print("    teams             - List all teams")
            print("    predict <h> vs <a> -- Predict a match")
            print("    ko:<h> vs <a>     - Predict a knockout match")
            print("    exit              - Exit")
            continue
        if inp.lower() in ("teams",):
            for t in get_teams():
                print(f"    {t.name:30s} ({t.fifa_code})  Group {t.group or '?'}")
            continue
        if inp.lower() in ("standings", "groups"):
            show_standings()
            continue
        if inp.lower() in ("upcoming",):
            show_upcoming(predictor)
            continue
        if inp.lower() in ("results",):
            show_results(predictor)
            continue
        if inp.lower() in ("bracket", "knockout"):
            show_knockout_bracket(predictor)
            continue
        if inp.lower().startswith("predict ") or inp.lower().startswith("ko:"):
            is_ko = inp.lower().startswith("ko:")
            if is_ko:
                inp = inp[3:].strip()
            inp = inp.replace("predict ", "", 1).strip()
            parts = inp.split(" vs ")
            if len(parts) == 2:
                home = parts[0].strip()
                away = parts[1].strip()
                result = predictor.predict(
                    home,
                    away,
                    neutral=True,
                    tournament="FIFA World Cup",
                    is_knockout=is_ko,
                )
                pretty_print(result)
            else:
                print("  Usage: predict <home> vs <away>")
            continue
        print(f"  Unknown command: {inp}")


def main():
    predictor = WorldCupPredictor()

    if len(sys.argv) < 2:
        interactive_api(predictor)
        return

    cmd = sys.argv[1].lower()

    if cmd in ("upcoming",):
        show_upcoming(predictor)
        return
    if cmd in ("results",):
        show_results(predictor)
        return
    if cmd in ("standings", "groups"):
        show_standings()
        return
    if cmd in ("bracket", "knockout"):
        show_knockout_bracket(predictor)
        return
    if cmd in ("predict",):
        if len(sys.argv) < 4:
            print("Usage: python src/wc_predict.py predict <home> <away> [--knockout]")
            return
        is_ko = "--knockout" in sys.argv
        home = sys.argv[2]
        away = sys.argv[3]
        result = predictor.predict(
            home, away, neutral=True, tournament="FIFA World Cup", is_knockout=is_ko
        )
        pretty_print(result)
        return
    if cmd in ("--interactive", "-i"):
        interactive_api(predictor)
        return

    print("Commands: upcoming, results, standings, bracket, predict, --interactive")
    print("Example: python src/wc_predict.py predict Argentina France --knockout")


if __name__ == "__main__":
    main()
