import numpy as np, sim, sim2, sim3, sys, itertools
from sim import X, GP, Zg, N, TEAMS, MYSLOT, ORDER, CATS
from sim2 import pureC, hasC
GEN = np.array([1,1,1,1,1,1,1,1,.25])
BIG = sim2.H
GRD = np.array([.8,1.2,1.6,.2,.4,1.6,1.2,.2,.25])
# strategy = (general rounds, weights after, caps)
STRATS = {
  "BPA (z-score)": (99, GEN, (99,99)),
  "Bigs after R1": (1, BIG, (7,10)),
  "Bigs after R2": (2, BIG, (7,10)),
  "Bigs after R3": (3, BIG, (7,10)),
  "Guards after R1": (1, GRD, (99,99)),
  "Guards after R2": (2, GRD, (99,99)),
}
def pick(strat, mine, avail, rnd):
    g, w, (cp, cc) = strat
    s = Zg @ (GEN if rnd < g else w)
    if rnd >= g:
        if sum(pureC[mine]) >= cp: s[pureC] -= 1e6
        if sum(hasC[mine]) >= cc: s[hasC] -= 1e6
    elif sum(pureC[mine]) >= 4: s[pureC] -= 1e6
    s = s + avail_noise(len(s))
    s[~avail] = -1e9
    return int(np.argmax(s))
_rng = np.random.default_rng(0)
def avail_noise(n): return _rng.normal(0, .15, n)   # AI drafters aren't perfectly identical

def draft(rng, mine, rival_strats):
    others = [t for t in range(TEAMS) if t != MYSLOT]
    rivals = dict(zip(rng.choice(others, len(rival_strats), replace=False), rival_strats))
    avail = np.ones(N, bool)
    r = np.arange(N)
    adp = r + rng.normal(0, 1, N) * (0.08*r + 1.5)      # hashtag-rank drafters, tighter than before
    ros = [[] for _ in range(TEAMS)]
    for i, t in enumerate(ORDER):
        rnd = i // TEAMS
        if t == MYSLOT: p = pick(mine, ros[t], avail, rnd)
        elif t in rivals: p = pick(rivals[t], ros[t], avail, rnd)
        else:
            a = np.where(avail, adp, 1e9)
            if sum(pureC[ros[t]]) >= 4: a[pureC] += 1e6
            p = int(np.argmin(a))
        avail[p] = False; ros[t].append(p)
    return ros

SCEN = {
  "2 AIs draft z-score BPA": ["BPA (z-score)"]*2,
  "2 AIs go bigs": ["Bigs after R2"]*2,
  "1 AI bigs, 1 AI guards": ["Bigs after R2", "Guards after R2"],
  "2 AIs go guards": ["Guards after R2"]*2,
}
if __name__ == "__main__":
    n = int(sys.argv[1]); res = {}
    for sc, rv in SCEN.items():
        for ms in STRATS:
            rng = np.random.default_rng(21); wk = []
            for _ in range(n):
                ros = draft(rng, STRATS[ms], [STRATS[x] for x in rv]); wk += sim3.season(rng, ros)
            cat = np.mean([w[0] for w in wk], 0); money = np.mean([w[4] for w in wk])*19
            solo = np.mean([w[2] and w[3]==1 for w in wk]); top = np.mean([w[2] for w in wk])
            res[(sc, ms)] = (money, solo, top, cat)
            print(f"{sc:26s} | {ms:16s} ${money:5.0f}  outright {solo*100:3.0f}%  win/tie {top*100:3.0f}%  " +
                  " ".join(f"{c}:{v*100:.0f}" for c, v in zip(CATS, cat) if v > .04), flush=True)
    print("\nAverage over the 4 scenarios:")
    for ms in STRATS:
        print(f"  {ms:16s} ${np.mean([res[(sc, ms)][0] for sc in SCEN]):5.0f}  outright {np.mean([res[(sc, ms)][1] for sc in SCEN])*100:.0f}%")
