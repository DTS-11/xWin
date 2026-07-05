import json
import ssl
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

API_BASE = "https://worldcup26.ir"

# World Cup 26 API uses a Let's Encrypt cert that may expire;
# we create a context that works even with expired certs for dev use.
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE

KO_STAGES = {"r32", "r16", "qf", "sf", "third", "final"}

STAGE_MAP = {
    "group": "Group Stage",
    "r32": "Round of 32",
    "r16": "Round of 16",
    "qf": "Quarter-finals",
    "sf": "Semi-finals",
    "third": "Third Place",
    "final": "Final",
}


@dataclass
class Match:
    match_id: str
    home_team: str
    away_team: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    stage: str = "group"
    group: Optional[str] = None
    finished: bool = False
    local_date: Optional[str] = None
    home_team_id: Optional[str] = None
    away_team_id: Optional[str] = None

    @property
    def is_knockout(self):
        return self.stage in KO_STAGES

    @property
    def stage_name(self):
        return STAGE_MAP.get(self.stage, self.stage)


@dataclass
class Team:
    team_id: str
    name: str
    fifa_code: str
    group: Optional[str] = None


def _fetch_json(endpoint: str) -> dict:
    url = f"{API_BASE}{endpoint}"
    req = urllib.request.Request(url, headers={"User-Agent": "xWin/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15, context=_ssl_ctx) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            raw = raw.replace("\u201c", '"').replace("\u201d", '"')
            raw = raw.replace("\u2018", "'").replace("\u2019", "'")
            # API returns unescaped quotes inside scorer string values.
            # We don't need scorers, so strip those fields entirely.
            raw = _remove_bad_fields(raw)
            return json.loads(raw)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        raise RuntimeError(f"API request failed: {e}") from e


def _remove_bad_fields(raw: str) -> str:
    """Remove home_scorers and away_scorers fields that contain invalid JSON."""
    bad_fields = ["home_scorers", "away_scorers"]
    known_next = ["group", "matchday", "away_scorers", "home_scorers"]
    for field in bad_fields:
        field_pattern = f'"{field}":"'
        while True:
            start = raw.find(field_pattern)
            if start == -1:
                break
            search_start = start + len(field_pattern)
            best_pos = len(raw)
            for nf in known_next:
                boundary = ',"' + nf + '":'
                pos = raw.find(boundary, search_start)
                if pos != -1 and pos < best_pos:
                    best_pos = pos
            if best_pos == len(raw):
                break
            replacement = f'"{field}":null'
            raw = raw[:start] + replacement + raw[best_pos:]
    return raw


def get_matches() -> list[Match]:
    data = _fetch_json("/get/games")
    matches = []
    for g in data.get("games", []):
        hs = g.get("home_score")
        aws = g.get("away_score")
        finished = g.get("finished", "FALSE").upper() == "TRUE"
        try:
            home_score = int(hs) if hs and str(hs).strip() and finished else None
            away_score = int(aws) if aws and str(aws).strip() and finished else None
        except (ValueError, TypeError):
            home_score = None
            away_score = None
        matches.append(
            Match(
                match_id=g.get("id", ""),
                home_team=g.get("home_team_name_en", ""),
                away_team=g.get("away_team_name_en", ""),
                home_score=home_score,
                away_score=away_score,
                stage=g.get("type", "group"),
                group=g.get("group"),
                finished=finished,
                local_date=g.get("local_date"),
                home_team_id=g.get("home_team_id"),
                away_team_id=g.get("away_team_id"),
            )
        )
    return matches


def get_teams() -> list[Team]:
    data = _fetch_json("/get/teams")
    teams = []
    for t in data.get("teams", []):
        teams.append(
            Team(
                team_id=t.get("id", ""),
                name=t.get("name_en", ""),
                fifa_code=t.get("fifa_code", ""),
                group=t.get("groups"),
            )
        )
    return teams


def get_groups() -> dict:
    data = _fetch_json("/get/groups")
    standings = {}
    for g in data.get("groups", []):
        name = g.get("name", "")
        teams_data = []
        for t in g.get("teams", []):
            teams_data.append(
                {
                    "team_id": t.get("team_id"),
                    "mp": int(t.get("mp", 0)),
                    "w": int(t.get("w", 0)),
                    "d": int(t.get("d", 0)),
                    "l": int(t.get("l", 0)),
                    "pts": int(t.get("pts", 0)),
                    "gf": int(t.get("gf", 0)),
                    "ga": int(t.get("ga", 0)),
                    "gd": int(t.get("gd", 0)),
                }
            )
        standings[name] = teams_data
    return standings


def get_upcoming_matches() -> list[Match]:
    return [m for m in get_matches() if not m.finished]


def get_finished_matches() -> list[Match]:
    return [m for m in get_matches() if m.finished]


def get_matches_by_stage(stage: str) -> list[Match]:
    return [m for m in get_matches() if m.stage == stage]


def get_knockout_matches() -> list[Match]:
    return [m for m in get_matches() if m.is_knockout]


def build_team_map() -> dict[str, str]:
    return {t.name.lower(): t.name for t in get_teams()}
