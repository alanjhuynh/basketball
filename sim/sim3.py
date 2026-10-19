import numpy as np, sim, sim2, sys
from sim import X, GP, Zg, N, TEAMS, MYSLOT, ORDER, CATS
from sim2 import POS, pureC, hasC, counted_share
REQ = [{"PG"},{"SG"},{"PG","SG"},{"SF"},{"PF"},{"SF","PF"},{"C"},{"C"}]  # non-UTIL starting slots
H = sim2.H; GEN = np.array([1,1,1,1,1,1,1,1,.25])
MY_PICKS = sim.MY_PICKS

def matched(roster):
    """max number of REQ slots the roster can fill at once (bipartite matching)."""
    match = [-1]*len(REQ)
    def aug(p, seen):
        for s, el in enumerate(REQ):
            if s not in seen and POS[p] & el:
                seen.add(s)
                if match[s] < 0 or aug(match[s], seen): match[s] = p; return True
        return False
    return sum(aug(p, set()) for p in roster)

def my_pick(rosters, avail, rnd, slot_aware, switched, me=MYSLOT, caps=(7,10)):
    mine = rosters[me]
    w = GEN if rnd < 2 or switched else H
    s = Zg @ w
    if not switched and rnd >= 2:
        if sum(pureC[mine]) >= caps[0]: s[pureC] -= 1e6
        if sum(hasC[mine]) >= caps[1]: s[hasC] -= 1e6
    if slot_aware:
        left = 14 - len(mine)                 # picks left including this one
        gap = len(REQ) - matched(mine)
        if gap >= left:                       # must fill a starting slot now
            base = matched(mine)
            cand = np.flatnonzero(avail)
            ok = np.zeros(N, bool)
            for p in cand[np.argsort(-s[cand])][:60]:
                if matched(mine + [p]) > base: ok[p] = True
            s[~ok] -= 1e7
    s[~avail] = -1e9
    return int(np.argmax(s))

def checkpoint(rosters, me):
    avg = lambda r, i: X[r, i].dot(GP[r] / 82) / len(r)
    rank = lambda i: 1 + sum(avg(r, i) >= avg(rosters[me], i) for k, r in enumerate(rosters) if k != me)
    return rank(3) > 1 and rank(7) > 1

def draft(rng, slot_aware=True, use_cp=True, rivals=()):
    avail = np.ones(N, bool)
    adp = np.arange(N) + rng.normal(0, 1, N) * (0.15*np.arange(N) + 2.5)
    rosters = [[] for _ in range(TEAMS)]
    switched = False; cut = MY_PICKS[5] - 1
    for i, t in enumerate(ORDER):
        if use_cp and i == cut: switched = checkpoint(rosters, MYSLOT)
        rnd = i // TEAMS
        if t == MYSLOT:
            p = my_pick(rosters, avail, rnd, slot_aware, switched)
        elif t in rivals:                      # another manager running a bigs build
            p = my_pick(rosters, avail, rnd, False, False, me=t)
        else:
            a = np.where(avail, adp, 1e9)
            if sum(pureC[rosters[t]]) >= 4: a[pureC] += 1e6
            p = int(np.argmin(a))
        avail[p] = False; rosters[t].append(p)
    return rosters, switched

def season(rng, rosters):
    R = np.array(rosters)
    share = np.array([[counted_share(r, rng)[p] for p in r] for r in rosters])
    lam = X[R]; res = []
    for _ in range(19):
        games = rng.choice([2,3,3,4,4,4,5], size=R.shape)
        healthy = rng.random(R.shape) < GP[R]/82
        tot = rng.poisson(lam * (games*healthy*share)[..., None]).sum(1).astype(float)
        tot[:, 8] *= -1
        best = tot.max(0)
        win = tot == best                      # ties: every tied team gets the cat
        c = win.sum(1)
        top = np.flatnonzero(c == c.max())
        res.append((win[MYSLOT], c[MYSLOT], MYSLOT in top, len(top), 20/len(top) if MYSLOT in top else 0))
    return res

def report(name, n, **kw):
    rng = np.random.default_rng(11); weeks = []; sw = 0; slotfill = []
    for _ in range(n):
        ros, s = draft(rng, **kw); sw += s
        slotfill.append(matched(ros[MYSLOT]))
        weeks += season(rng, ros)
    cat = np.mean([w[0] for w in weeks], 0); nc = np.array([w[1] for w in weeks])
    top = np.mean([w[2] for w in weeks]); solo = np.mean([w[2] and w[3] == 1 for w in weeks])
    money = np.mean([w[4] for w in weeks]) * 19
    print(f"\n== {name}  (switch fired {sw*100/n:.0f}% of drafts, starting slots filled {np.mean(slotfill):.1f}/8)")
    print("  win each cat/week: " + "  ".join(f"{c} {v*100:.0f}%" for c, v in zip(CATS, cat)))
    print(f"  cats won/week avg {nc.mean():.2f} | dist " + " ".join(f"{k}:{np.mean(nc==k)*100:.0f}%" for k in range(0, 7)))
    print(f"  weekly prize: win outright {solo*100:.0f}%, win or tie for most {top*100:.0f}%, "
          f"expected ${money:.0f}/season (prize split on ties), ~{money/20:.1f} of 19 weeks", flush=True)

if __name__ == "__main__":
    n = int(sys.argv[1])
    report("Board as-is (caps only, no slot check)", n, slot_aware=False)
    report("Board + starting-slot check", n, slot_aware=True)
    report("Slot check, 1 rival also drafting bigs", n, slot_aware=True, rivals=(2,))
    report("Slot check, 2 rivals also drafting bigs", n, slot_aware=True, rivals=(2, 10))
    # baseline: everyone ADP, me BPA
    rng = np.random.default_rng(11)
