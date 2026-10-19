import numpy as np, sim, sim3, sim5, sim7
from sim import TEAMS, MYSLOT, ORDER, N, Zg
from sim2 import pureC, hasC
from sim5 import STRATS, pick, GEN, BIG, GRD
from sim7 import YAHOO, HASH, SCEN
G_, B_ = Zg @ GEN, Zg @ BIG
def reach_teams(ros, picks_so_far, margin):
    """teams (not me) that took a specialist big well ahead of Yahoo rank"""
    teams = set()
    for i, (t, p) in enumerate(picks_so_far):
        if t != MYSLOT and hasC[p] and B_[p] > G_[p] and YAHOO[p] > i + margin: teams.add(t)
    return len(teams)
def draft(rng, decide_at, margin, need, rival_strats):
    others = [t for t in range(TEAMS) if t != MYSLOT]; rng.shuffle(others)
    rivals = dict(zip(others[:2], rival_strats)); hashers = set(others[2:4])
    adp = {"y": YAHOO + rng.normal(0,1,N)*(0.08*YAHOO+1.5), "h": HASH + rng.normal(0,1,N)*(0.08*HASH+1.5)}
    avail = np.ones(N, bool); ros = [[] for _ in range(TEAMS)]; log = []
    gen_rounds = (decide_at - 1) // TEAMS
    mine = (gen_rounds, BIG, (7, 10)); sw = False
    for i, t in enumerate(ORDER):
        if i == decide_at - 1 and reach_teams(ros, log, margin) >= need:
            mine = (gen_rounds, GRD, (99, 99)); sw = True
        rnd = i // TEAMS
        if t == MYSLOT: p = pick(mine, ros[t], avail, rnd)
        elif t in rivals: p = pick(rivals[t], ros[t], avail, rnd)
        else:
            a = np.where(avail, adp["h" if t in hashers else "y"], 1e9)
            if sum(pureC[ros[t]]) >= 4: a[pureC] += 1e6
            p = int(np.argmin(a))
        avail[p] = False; ros[t].append(p); log.append((t, p))
    return ros, sw
if __name__ == "__main__":
    n = 100
    for dec, mg, need in [(32, 10, 1), (32, 10, 2), (32, 6, 2), (41, 10, 2)]:
        tot = []
        for sc, rv in SCEN.items():
            rng = np.random.default_rng(31); wk = []; s = 0
            for _ in range(n):
                ros, w = draft(rng, dec, mg, need, [STRATS[x] for x in rv]); s += w; wk += sim3.season(rng, ros)
            m = np.mean([w[4] for w in wk])*19; tot.append(m)
            print(f"decide #{dec}, reach>{mg}, need {need} teams | {sc:20s} ${m:4.0f}  switched {s*100/n:3.0f}%", flush=True)
        print(f"   average ${np.mean(tot):.0f}\n", flush=True)
