import numpy as np, pandas as pd, sys

df = pd.read_csv("proj.tsv", sep="\t")
CATS = ["FGM", "FTM", "3PM", "OREB", "DREB", "AST", "STL", "BLK", "TO"]
X = df[CATS].to_numpy(float)
GP = df["GP"].to_numpy(float)
N = len(df)
TEAMS, ROUNDS, MYSLOT, WEEKS = 12, 14, 7, 19  # MYSLOT is 0-based (pick 8)

pool = X[:168]
Z = (X - pool.mean(0)) / pool.std(0)
Z[:, 8] *= -1  # fewer TO is better
# scale by games played so injury-prone guys are discounted
Zg = Z * (GP / 82)[:, None] ** 0.5

def pick_order():
    order = []
    for r in range(ROUNDS):
        order += list(range(TEAMS)) if r % 2 == 0 else list(range(TEAMS))[::-1]
    return order

ORDER = pick_order()
MY_PICKS = [i + 1 for i, t in enumerate(ORDER) if t == MYSLOT]

def weights(general_rounds, w_after):
    def f(rnd):
        if rnd < general_rounds:
            return np.array([1, 1, 1, 1, 1, 1, 1, 1, 0.25])
        return w_after
    return f

STRATS = {
    "BPA (TO 0.25)": weights(99, None),
    "Bigs after R2": weights(2, np.array([1.2, .6, .3, 1.6, 1.4, .3, .8, 1.5, .25])),
    "Bigs after R3": weights(3, np.array([1.2, .6, .3, 1.6, 1.4, .3, .8, 1.5, .25])),
    "Bigs after R4": weights(4, np.array([1.2, .6, .3, 1.6, 1.4, .3, .8, 1.5, .25])),
    "Hard bigs after R2 (punt AST/3PM)": weights(2, np.array([1.2, .5, 0, 1.8, 1.5, 0, .6, 1.6, .2])),
    "Hard bigs after R3 (punt AST/3PM)": weights(3, np.array([1.2, .5, 0, 1.8, 1.5, 0, .6, 1.6, .2])),
    "Guards after R3": weights(3, np.array([.8, 1.2, 1.6, .2, .4, 1.6, 1.2, .2, .25])),
}

def draft(rng, strat):
    avail = np.ones(N, bool)
    adp = np.arange(N) + rng.normal(0, 1, N) * (0.15 * np.arange(N) + 2.5)
    rosters = [[] for _ in range(TEAMS)]
    for i, t in enumerate(ORDER):
        rnd = i // TEAMS
        if t == MYSLOT:
            w = strat(rnd)
            s = Zg @ w
            s[~avail] = -1e9
            p = int(np.argmax(s))
        else:
            a = np.where(avail, adp, 1e9)
            p = int(np.argmin(a))
        avail[p] = False
        rosters[t].append(p)
    return rosters

def season(rng, rosters):
    R = np.array(rosters)                       # teams x 14
    lam = X[R]                                  # teams x 14 x 9
    wins = np.zeros(TEAMS)
    money = np.zeros(TEAMS)
    for _ in range(WEEKS):
        games = rng.choice([2, 3, 3, 4, 4, 4, 5], size=R.shape)
        healthy = rng.random(R.shape) < GP[R] / 82
        g = (games * healthy)[..., None]
        tot = rng.poisson(lam * g).sum(1).astype(float)  # teams x 9
        tot[:, 8] *= -1
        tot += rng.random(tot.shape) * 1e-3              # random tiebreak
        catwin = np.zeros(TEAMS)
        np.add.at(catwin, tot.argmax(0), 1)
        top = np.flatnonzero(catwin == catwin.max())
        money[top] += 20 / len(top)
        wins += catwin
    return money, wins

def run(n=400, seed=1):
    out = {}
    for name, strat in STRATS.items():
        rng = np.random.default_rng(seed)
        m = []; c = []
        for _ in range(n):
            ros = draft(rng, strat)
            money, wins = season(rng, ros)
            m.append(money[MYSLOT]); c.append(wins[MYSLOT] / WEEKS)
        out[name] = (np.mean(m), np.mean(c))
        print(f"{name:38s} ${np.mean(m):6.1f}/season  {np.mean(c):.2f} cats/wk", flush=True)
    return out

if __name__ == "__main__":
    print("my picks:", MY_PICKS)
    print("fair share = $%.1f" % (20 * WEEKS / TEAMS))
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
