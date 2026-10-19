import numpy as np, sim
from sim import X, GP, Zg, N, TEAMS, MYSLOT, ORDER, df
POS = [set(p.split(",")) for p in df.POS]
SLOTS = [{"C"},{"C"},{"PG"},{"SG"},{"SF"},{"PF"},{"PG","SG"},{"SF","PF"},None,None,None]
pureC = np.array([p=={"C"} for p in POS])
hasC = np.array(["C" in p for p in POS])
gen = Zg @ np.array([1,1,1,1,1,1,1,1,.25])

def counted_share(roster, rng, days=60):
    """fraction of each player's games that land in an active slot (daily lineups)."""
    order = sorted(roster, key=lambda p: -gen[p])
    play = rng.random((days, len(order))) < 0.5 * GP[order] / 82
    used = np.zeros(len(order))
    for d in range(days):
        free = [i for i in range(len(order)) if play[d, i]]
        for s in SLOTS:
            for i in free:
                if s is None or POS[order[i]] & s:
                    used[i] += 1; free.remove(i); break
    tot = play.sum(0)
    share = np.where(tot > 0, used / np.maximum(tot, 1), 1.0)
    return dict(zip(order, share))

def draft(rng, w_fn, cap_pure, cap_c):
    avail = np.ones(N, bool)
    adp = np.arange(N) + rng.normal(0, 1, N) * (0.15 * np.arange(N) + 2.5)
    rosters = [[] for _ in range(TEAMS)]
    for i, t in enumerate(ORDER):
        rnd = i // TEAMS
        if t == MYSLOT:
            s = Zg @ w_fn(rnd)
            mine = rosters[t]
            if sum(pureC[mine]) >= cap_pure: s[pureC] -= 1e6
            if sum(hasC[mine]) >= cap_c: s[hasC] -= 1e6
            s[~avail] = -1e9
            p = int(np.argmax(s))
        else:
            a = np.where(avail, adp, 1e9)
            # opponents avoid absurd C stacks
            if sum(pureC[rosters[t]]) >= 4: a[pureC] += 1e6
            p = int(np.argmin(a))
        avail[p] = False; rosters[t].append(p)
    return rosters

def season(rng, rosters):
    R = np.array(rosters)
    share = np.array([[counted_share(r, rng)[p] for p in r] for r in rosters])
    lam = X[R]; money = 0; cw = np.zeros(9)
    for _ in range(19):
        games = rng.choice([2,3,3,4,4,4,5], size=R.shape)
        healthy = rng.random(R.shape) < GP[R] / 82
        g = (games * healthy * share)[..., None]
        tot = rng.poisson(lam * g).sum(1).astype(float); tot[:, 8] *= -1
        tot += rng.random(tot.shape) * 1e-3
        w = tot.argmax(0); c = np.bincount(w, minlength=TEAMS)
        top = np.flatnonzero(c == c.max())
        if MYSLOT in top: money += 20 / len(top)
        cw += (w == MYSLOT)
    return money, cw / 19

H = np.array([1.2,.5,0,1.8,1.5,0,.6,1.6,.2])
B = np.array([1.2,.6,.3,1.6,1.4,.3,.8,1.5,.25])
G = np.array([.8,1.2,1.6,.2,.4,1.6,1.2,.2,.25])
W = sim.weights
STRATS = {
 "BPA":                    (W(99,None), 99, 99),
 "Hard bigs R2, pureC<=4, C<=7": (W(2,H), 4, 7),
 "Hard bigs R2, pureC<=5, C<=8": (W(2,H), 5, 8),
 "Hard bigs R2, pureC<=6, C<=9": (W(2,H), 6, 9),
 "Hard bigs R2, no caps":        (W(2,H), 99, 99),
 "Hard bigs R1, pureC<=5, C<=8": (W(1,H), 5, 8),
 "Hard bigs R3, pureC<=5, C<=8": (W(3,H), 5, 8),
 "Soft bigs R2, pureC<=5, C<=8": (W(2,B), 5, 8),
 "Guards R2":                    (W(2,G), 3, 4),
}
if __name__ == "__main__":
    import sys; n = int(sys.argv[1])
    for name, (f, a, b) in STRATS.items():
        rng = np.random.default_rng(3); m = []; c = []
        for _ in range(n):
            ros = draft(rng, f, a, b); mm, cc = season(rng, ros); m.append(mm); c.append(cc)
        c = np.mean(c, 0)
        print(f"{name:32s} ${np.mean(m):6.1f}  {c.sum():.2f} cats/wk  " +
              " ".join(f"{k}:{v:.2f}" for k, v in zip(sim.CATS, c) if v > .05), flush=True)
