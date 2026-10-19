import numpy as np, sim
from collections import Counter
H = np.array([1.2,.5,0,1.8,1.5,0,.6,1.6,.2])
H2 = np.array([1.5,.4,0,1.8,1.5,0,.4,1.6,.1])
sim.STRATS = {
 "Hard bigs after R1": sim.weights(1,H),
 "Hard bigs after R2": sim.weights(2,H),
 "Harder bigs (FGM up) after R2": sim.weights(2,H2),
}
sim.run(300)
# diagnostics for chosen strategy
rng=np.random.default_rng(5); strat=sim.weights(2,H)
cw=np.zeros(9); cnt=Counter(); n=300
for _ in range(n):
    ros=sim.draft(rng,strat)
    for k,p in enumerate(ros[sim.MYSLOT]): cnt[(k,sim.df.PLAYER[p])]+=1
    R=np.array(ros); lam=sim.X[R]
    for _ in range(19):
        g=(rng.choice([2,3,3,4,4,4,5],size=R.shape)*(rng.random(R.shape)<sim.GP[R]/82))[...,None]
        t=rng.poisson(lam*g).sum(1).astype(float); t[:,8]*=-1; t+=rng.random(t.shape)*1e-3
        cw+= (t.argmax(0)==sim.MYSLOT)
print("cat win rate:",dict(zip(sim.CATS,np.round(cw/(n*19),2))))
for k in range(14):
    top=sorted([(c,nm) for (kk,nm),c in cnt.items() if kk==k],reverse=True)[:5]
    print(f"R{k+1} pick {sim.MY_PICKS[k]}:", ", ".join(f"{nm} {c*100//n}%" for c,nm in top))
