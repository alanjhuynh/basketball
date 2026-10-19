# Pick 8 draft kit

Draft tools for a 12-team Yahoo league: 9 counting categories (FGM, FTM, 3PM, OREB, DREB, AST, STL, BLK, TO),
14-man rosters, snake draft from pick 8, and a $20 weekly prize for the team that wins the most categories
across all 12 teams (no head-to-head).

Lineup: PG, SG, G, SF, PF, F, C, C, UTIL, UTIL, UTIL plus 3 bench, daily changes.

## Board

`board/pick8-war-room.html` is the live draft board, also published at
https://claude.ai/artifact/WKqs61j1c9kP6QXorvikmy. Tap each player as anyone drafts him; the board tracks
the snake order and suggests picks.

- Picks 8 and 17: best player available.
- Right before pick 32 it decides Bigs or Guards. Guards if 2+ other teams took a center 10+ picks ahead
  of his Yahoo rank; otherwise Bigs (caps: 7 pure C, 10 C-eligible).
- While another team is on the clock, Auto shows the general best-available list.
- Players who share an NBA team with your picks get a tag and a 0.1 tiebreaker penalty.

## Simulation

Everything in `sim/` runs from that directory with Python 3, numpy and pandas.

| File | What it does |
| --- | --- |
| `sim.py` | Data loading, z-scores, snake order, first strategy comparison |
| `sim2.py` | Positional lineup slots and center caps |
| `sim3.py` | Per-category and weekly-prize odds, starting-slot check, rival bigs drafters |
| `sim4.py` | Mid-draft switch fallbacks (General vs Guards) |
| `sim5.py` | Sharper Hashtag-rank opponents plus 2 AI managers, 4 scenarios |
| `sim6.py` | Mid-draft switch rules against those opponents |
| `sim7.py` | Yahoo-ranking opponents (7 Yahoo, 2 Hashtag, 2 AI) |
| `sim8.py` | The pick-32 bigs-or-guards decision rule |
| `sim9.py` | Shared NBA team schedules with daily lineups, teammate penalty |
| `diag.py` | Category win rates and typical picks per round |

Data:

- `proj_raw.tsv`: Hashtag Basketball projections as provided.
- `proj.tsv`: the same minus long-term injuries as of Oct 19, 2026 (J. Butler, B. Ingram, K. Porzingis, M. Moody).
- `yrank.csv`: estimated Yahoo ranking. Real Yahoo ADP was found for 13 players; the rest use a
  standard-category estimate (PTS, REB, AST, STL, BLK, 3PM, TO).
- `players.json`: the scores embedded in the board.

## Main results

Weekly prize money per season (fair share is about $32):

| Strategy | Avg over 4 rival scenarios |
| --- | --- |
| 2 general picks, then decide bigs or guards at #32 | $99 |
| Bigs after pick 8 | $96 |
| 2 general picks, then always bigs | $89 |
| 2 general picks, then always guards | $86 |

Not modeled: the real NBA schedule, multi-week injuries, waiver pickups and Yahoo games limits.
