import numpy as np, pandas as pd, sim, sim3, sim5
from sim import TEAMS, MYSLOT, ORDER, N
from sim2 import pureC
from sim5 import STRATS, pick
yr = pd.read_csv("yrank.csv").yrank.to_numpy(float)
ANCH = {"N.Jokic":1.9,"C.Cunningham":6.2,"K.Leonard":29.3,"W.Kessler":38,"C.Boozer":56,"R.Gobert":72.5,
        "Z.Edey":74.9,"J.Tatum":5,"C.Flagg":8,"J.Johnson":12,"D.Clingan":44,"I.Zubac":66,"K.Ware":78}
for n, a in ANCH.items(): yr[np.flatnonzero(sim.df.PLAYER.values == n)[0]] = a
YAHOO = np.argsort(np.argsort(yr)).astype(float)       # 0-based Yahoo-style rank
HASH = np.arange(N, dtype=float)

def draft(rng, mine, n_hash, rival_strats):
    others = [t for t in range(TEAMS) if t != MYSLOT]; rng.shuffle(others)
    rivals = dict(zip(others[:len(rival_strats)], rival_strats))
    hashers = set(others[len(rival_strats):len(rival_strats) + n_hash])
    noise = rng.normal(0, 1, N)
    adp = {"y": YAHOO + noise*(0.08*YAHOO + 1.5), "h": HASH + rng.normal(0, 1, N)*(0.08*HASH + 1.5)}
    avail = np.ones(N, bool); ros = [[] for _ in range(TEAMS)]
    for i, t in enumerate(ORDER):
        rnd = i // TEAMS
        if t == MYSLOT: p = pick(mine, ros[t], avail, rnd)
        elif t in rivals: p = pick(rivals[t], ros[t], avail, rnd)
        else:
            a = np.where(avail, adp["h" if t in hashers else "y"], 1e9)
            if sum(pureC[ros[t]]) >= 4: a[pureC] += 1e6
            p = int(np.argmin(a))
        avail[p] = False; ros[t].append(p)
    return ros

SCEN = {
  "2 AIs balanced": ["BPA (z-score)"]*2,
  "2 AIs go bigs": ["Bigs after R2"]*2,
  "1 AI bigs, 1 guards": ["Bigs after R2", "Guards after R2"],
  "2 AIs go guards": ["Guards after R2"]*2,
}
MINE = ["Bigs after R1", "Bigs after R2", "Bigs after R3", "Guards after R1", "Guards after R2"]
if __name__ == "__main__":
    import sys; n = int(sys.argv[1]); res = {}
    for sc, rv in SCEN.items():
        for ms in MINE:
            rng = np.random.default_rng(31); wk = []
            for _ in range(n):
                wk += sim3.season(rng, draft(rng, STRATS[ms], 2, [STRATS[x] for x in rv]))
            cat = np.mean([w[0] for w in wk], 0); m = np.mean([w[4] for w in wk])*19
            solo = np.mean([w[2] and w[3]==1 for w in wk]); top = np.mean([w[2] for w in wk])
            res[(sc, ms)] = m
            print(f"{sc:20s} | {ms:16s} ${m:4.0f} outright {solo*100:3.0f}% win/tie {top*100:3.0f}%  " +
                  " ".join(f"{c}:{v*100:.0f}" for c, v in zip(sim.CATS, cat) if v > .04), flush=True)
    print("\naverage:", {ms: round(np.mean([res[(s, ms)] for s in SCEN])) for ms in MINE})
