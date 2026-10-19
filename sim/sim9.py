import numpy as np, sim, sim2, sim5, sim7, sim8, sys
from sim import X, GP, N, TEAMS, MYSLOT, ORDER, Zg, df
from sim2 import POS, SLOTS, pureC, hasC
from sim5 import STRATS, GEN, BIG, GRD
from sim7 import YAHOO, HASH, SCEN
NBA = df.TEAM.to_numpy(); TEAMIDS = {t: i for i, t in enumerate(sorted(set(NBA)))}
TID = np.array([TEAMIDS[t] for t in NBA]); NT = len(TEAMIDS)
gen = Zg @ GEN

def week_sched(rng, correlated):
    """bool [unit, 7 days]: unit = NBA team (correlated) or player (independent)"""
    n = NT if correlated else N
    g = rng.choice([2, 3, 3, 4, 4, 4, 5], size=n)
    s = np.zeros((n, 7), bool)
    for u in range(n): s[u, rng.choice(7, g[u], replace=False)] = True
    return s

def season(rng, rosters, correlated=True, weeks=19):
    R = np.array(rosters); lam = X[R]; res = []
    order = [sorted(r, key=lambda p: -gen[p]) for r in rosters]
    for _ in range(weeks):
        s = week_sched(rng, correlated)
        plays = s[TID] if correlated else s            # player x day
        healthy = rng.random(N) < GP / 82
        counted = np.zeros(R.shape)
        for k, ros in enumerate(order):
            col = {p: j for j, p in enumerate(rosters[k])}
            for d in range(7):
                free = [p for p in ros if healthy[p] and plays[p, d]]
                for sl in SLOTS:
                    for p in free:
                        if sl is None or POS[p] & sl:
                            counted[k, col[p]] += 1; free.remove(p); break
        tot = rng.poisson(lam * counted[..., None]).sum(1).astype(float); tot[:, 8] *= -1
        win = tot == tot.max(0); c = win.sum(1); top = np.flatnonzero(c == c.max())
        res.append((win[MYSLOT], c[MYSLOT], MYSLOT in top, len(top), 20/len(top) if MYSLOT in top else 0,
                    counted[MYSLOT].sum()))
    return res

def pick_pen(strat, mine, avail, rnd, pen):
    g, w, (cp, cc) = strat
    s = Zg @ (GEN if rnd < g else w)
    if rnd >= g:
        if sum(pureC[mine]) >= cp: s[pureC] -= 1e6
        if sum(hasC[mine]) >= cc: s[hasC] -= 1e6
    if pen and mine:
        mates = np.bincount(TID[mine], minlength=NT)
        s = s - pen * mates[TID]                      # penalty per teammate already rostered
    s = s + np.random.default_rng().normal(0, .15, N)
    s[~avail] = -1e9
    return int(np.argmax(s))

def draft(rng, rival_strats, pen):
    """Yahoo room + 2 hashtag + 2 AI rivals; me = 2 general picks then bigs-or-guards call at #32."""
    others = [t for t in range(TEAMS) if t != MYSLOT]; rng.shuffle(others)
    rivals = dict(zip(others[:2], rival_strats)); hashers = set(others[2:4])
    adp = {"y": YAHOO + rng.normal(0,1,N)*(0.08*YAHOO+1.5), "h": HASH + rng.normal(0,1,N)*(0.08*HASH+1.5)}
    avail = np.ones(N, bool); ros = [[] for _ in range(TEAMS)]; log = []
    mine = (2, BIG, (7, 10))
    for i, t in enumerate(ORDER):
        if i == 31 and sim8.reach_teams(ros, log, 10) >= 2: mine = (2, GRD, (99, 99))
        rnd = i // TEAMS
        if t == MYSLOT: p = pick_pen(mine, ros[t], avail, rnd, pen)
        elif t in rivals: p = sim5.pick(rivals[t], ros[t], avail, rnd)
        else:
            a = np.where(avail, adp["h" if t in hashers else "y"], 1e9)
            if sum(pureC[ros[t]]) >= 4: a[pureC] += 1e6
            p = int(np.argmin(a))
        avail[p] = False; ros[t].append(p); log.append((t, p))
    return ros

def run(label, n, pen, correlated):
    tot = []; per = {}
    for sc, rv in SCEN.items():
        rng = np.random.default_rng(55); wk = []; mates = []
        for _ in range(n):
            ros = draft(rng, [STRATS[x] for x in rv], pen)
            m = ros[MYSLOT]; mates.append(len(m) - len(set(TID[m])))
            wk += season(rng, ros, correlated)
        per[sc] = (np.mean([w[4] for w in wk])*19, np.mean(mates), np.mean([w[5] for w in wk]),
                   np.mean([w[2] and w[3]==1 for w in wk]), np.mean([w[0] for w in wk], 0))
    avg = np.mean([v[0] for v in per.values()])
    print(f"{label:44s} avg ${avg:4.0f} | " + " | ".join(f"{k.split()[1]+k.split()[2]}: ${v[0]:.0f}" for k, v in per.items())
          + f" | extra teammates {np.mean([v[1] for v in per.values()]):.1f}, counted games/wk {np.mean([v[2] for v in per.values()]):.1f}", flush=True)
    return per

if __name__ == "__main__":
    n = int(sys.argv[1])
    run("Independent schedules (old model)", n, 0, False)
    run("Team schedules, no teammate penalty", n, 0, True)
    for pen in (0.3, 0.6, 1.0):
        run(f"Team schedules, penalty {pen} per teammate", n, pen, True)
