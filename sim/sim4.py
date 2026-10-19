import numpy as np, sim3, sim2
G = np.array([.8,1.2,1.6,.2,.4,1.6,1.2,.2,.25])
orig = sim3.my_pick
def patched(fallback):
    def f(rosters, avail, rnd, slot_aware, switched, me=sim3.MYSLOT, caps=(7,10)):
        if switched and me == sim3.MYSLOT:
            s = sim3.Zg @ fallback; s[~avail] = -1e9
            # keep the same slot check when falling back
            if slot_aware:
                mine = rosters[me]; left = 14-len(mine); base = sim3.matched(mine)
                if len(sim3.REQ)-base >= left:
                    ok = np.array([avail[p] and sim3.matched(mine+[p]) > base for p in range(sim3.N)]); s[~ok] -= 1e7
            return int(np.argmax(s))
        return orig(rosters, avail, rnd, slot_aware, switched, me, caps)
    return f
n = 150
for riv, lab in [((), "no rival"), ((2,), "1 rival bigs")]:
    sim3.my_pick = orig
    sim3.report(f"{lab}: stay bigs (no switch)", n, slot_aware=False, use_cp=False, rivals=riv)
    sim3.report(f"{lab}: switch -> General", n, slot_aware=False, use_cp=True, rivals=riv)
    sim3.my_pick = patched(G)
    sim3.report(f"{lab}: switch -> Guards", n, slot_aware=False, use_cp=True, rivals=riv)
