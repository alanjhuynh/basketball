"""ADP lookahead: take the player that maximizes value now + expected best value still there at my next pick.
Survival is estimated by sampling noisy Yahoo ADP for the picks between my turns."""
import numpy as np, sim, sim3, sim5, sim8, sim9, sys
from sim import N, TEAMS, MYSLOT, ORDER, Zg, MY_PICKS
from sim2 import pureC, hasC
from sim5 import STRATS, GEN, BIG, GRD
from sim7 import YAHOO, HASH, SCEN
VAL = {"gen": Zg @ GEN, "big": Zg @ BIG, "grd": Zg @ GRD}
CAPS = {"gen": (99, 99), "big": (7, 10), "grd": (99, 99)}

def eligible(mode, mine):
    ok = np.ones(N, bool); cp, cc = CAPS[mode]
    if sum(pureC[mine]) >= cp: ok &= ~pureC
    if sum(hasC[mine]) >= cc: ok &= ~hasC
    return ok

def lookahead_pick(rng, mine, avail, i, mode_now, mode_next, S=150, widen=1.25):
    vn = np.where(avail & eligible(mode_now, mine), VAL[mode_now], -1e9)
    nxt = [n - 1 for n in MY_PICKS if n - 1 > i]
    if not nxt: return int(np.argmax(vn))
    k = nxt[0] - i - 1                                  # other teams' picks before my next turn
    cand = np.argsort(-vn)[:15]
    av = np.flatnonzero(avail)
    sd = (0.08 * YAHOO[av] + 1.5) * widen
    noisy = YAHOO[av][None, :] + rng.normal(0, 1, (S, len(av))) * sd
    taken = np.zeros((S, len(av)), bool)
    taken[np.arange(S)[:, None], np.argsort(noisy, 1)[:, :k]] = True
    best = []
    for a in cand:
        el = eligible(mode_next, mine + [a])[av] & (av != a)
        v = np.where(~taken & el[None, :], VAL[mode_next][av][None, :], -1e9)
        best.append(v.max(1).mean())
    tot = vn[cand] + np.array(best)
    return int(cand[np.argmax(tot)])

def draft(rng, rival_strats, look, widen=1.25):
    others = [t for t in range(TEAMS) if t != MYSLOT]; rng.shuffle(others)
    rivals = dict(zip(others[:2], rival_strats)); hashers = set(others[2:4])
    adp = {"y": YAHOO + rng.normal(0,1,N)*(0.08*YAHOO+1.5), "h": HASH + rng.normal(0,1,N)*(0.08*HASH+1.5)}
    avail = np.ones(N, bool); ros = [[] for _ in range(TEAMS)]; log = []; late = "big"
    for i, t in enumerate(ORDER):
        if i == 31 and sim8.reach_teams(ros, log, 10) >= 2: late = "grd"
        rnd = i // TEAMS
        if t == MYSLOT:
            mode = "gen" if rnd < 2 else late
            nmode = "gen" if rnd + 1 < 2 else late
            if look: p = lookahead_pick(rng, ros[t], avail, i, mode, nmode, widen=widen)
            else:
                v = np.where(avail & eligible(mode, ros[t]), VAL[mode] + rng.normal(0, .15, N), -1e9); p = int(np.argmax(v))
        elif t in rivals: p = sim5.pick(rivals[t], ros[t], avail, rnd)
        else:
            a = np.where(avail, adp["h" if t in hashers else "y"], 1e9)
            if sum(pureC[ros[t]]) >= 4: a[pureC] += 1e6
            p = int(np.argmin(a))
        avail[p] = False; ros[t].append(p); log.append((t, p))
    return ros

def run(label, n, **kw):
    out = []
    for sc, rv in SCEN.items():
        rng = np.random.default_rng(91); wk = []
        for _ in range(n): wk += sim9.season(rng, draft(rng, [STRATS[x] for x in rv], **kw), True)
        out.append(np.mean([w[4] for w in wk]) * 19)
    print(f"{label:34s} avg ${np.mean(out):4.0f} | " + " ".join(f"${x:.0f}" for x in out), flush=True)

if __name__ == "__main__":
    n = int(sys.argv[1])
    run("Current board (value only)", n, look=False)
    run("ADP lookahead, noise x1.25", n, look=True, widen=1.25)
    run("ADP lookahead, noise x2", n, look=True, widen=2.0)
