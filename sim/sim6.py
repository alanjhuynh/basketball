import numpy as np, sim, sim3, sim5
from sim5 import STRATS, SCEN, pick, GRD, GEN
from sim import TEAMS, MYSLOT, ORDER, N, X, GP
from sim2 import pureC
def rivals_on_bigs(ros, me):
    """number of other teams whose per-player OREB+BLK beats mine"""
    v = lambda r: (X[r][:, [3, 7]].sum(1) * GP[r] / 82).mean() if r else 0
    return sum(v(r) >= v(ros[me]) for k, r in enumerate(ros) if k != me)
def draft(rng, cut_pick, thresh, rival_strats):
    others = [t for t in range(TEAMS) if t != MYSLOT]
    rivals = dict(zip(rng.choice(others, len(rival_strats), replace=False), rival_strats))
    avail = np.ones(N, bool); r = np.arange(N)
    adp = r + rng.normal(0, 1, N) * (0.08*r + 1.5)
    ros = [[] for _ in range(TEAMS)]; mine = STRATS["Bigs after R1"]; sw = False
    for i, t in enumerate(ORDER):
        if i == cut_pick - 1 and rivals_on_bigs(ros, MYSLOT) >= thresh:
            mine = (1, GRD, (99, 99)); sw = True
        rnd = i // TEAMS
        if t == MYSLOT: p = pick(mine, ros[t], avail, rnd)
        elif t in rivals: p = pick(rivals[t], ros[t], avail, rnd)
        else:
            a = np.where(avail, adp, 1e9)
            if sum(pureC[ros[t]]) >= 4: a[pureC] += 1e6
            p = int(np.argmin(a))
        avail[p] = False; ros[t].append(p)
    return ros, sw
n = 120
for cut, th in [(41, 1), (41, 2), (56, 1), (56, 2)]:
    tot = []
    for sc in ["2 AIs draft z-score BPA", "2 AIs go bigs", "1 AI bigs, 1 AI guards", "2 AIs go guards"]:
        rng = np.random.default_rng(21); wk = []; s = 0
        for _ in range(n):
            ros, w = draft(rng, cut, th, [STRATS[x] for x in SCEN[sc]]); s += w; wk += sim3.season(rng, ros)
        m = np.mean([w[4] for w in wk])*19; tot.append(m)
        print(f"check before #{cut}, switch if >={th} teams match my bigs | {sc:24s} ${m:5.0f} (switched {s*100/n:.0f}%)", flush=True)
    print(f"   -> average ${np.mean(tot):.0f}\n")
