# 拡散MDの初期構造(replicate後)と最終ckptのPS4骨格・距離・Li分布を比較
import sys, numpy as np
from ase.io import read
from scipy.spatial import cKDTree
from collections import Counter
REG={"GB1":(13.86,20.86),"BULK":(30,40),"GB2":(48.58,55.58)}
def analyze(a,label):
    a.wrap(); sym=np.array(a.get_chemical_symbols()); pos=a.get_positions(); L=np.diag(a.get_cell())
    t=cKDTree(pos%L,boxsize=L)
    P=np.where(sym=="P")[0]; S=np.where(sym=="S")[0]; X=np.where(np.isin(sym,["Cl","Br"]))[0]; Li=np.where(sym=="Li")[0]
    isS=np.isin(np.arange(len(a)),S); isX=np.isin(np.arange(len(a)),X); isP=np.isin(np.arange(len(a)),P)
    cn=Counter(); vtxX=0; pp=0
    Sbond=np.zeros(len(a),int)
    for p in P:
        nb=[j for j in t.query_ball_point(pos[p]%L,2.4) if j!=p]
        ns=sum(isS[j] for j in nb); cn[ns]+=1
        vtxX+=sum(isX[j] for j in nb); pp+=sum(isP[j] for j in nb)
        for j in nb:
            if isS[j]: Sbond[j]+=1
    bridging=int((Sbond[S]>=2).sum()); freeS=S[Sbond[S]==0]
    fx=pos[freeS,0]%L[0]
    fS_gb=sum(((fx>lo)&(fx<hi)).sum() for k,(lo,hi) in REG.items() if k!="BULK")
    d,_=t.query(pos%L,k=2); dmin=d[:,1]
    # 種別ごと最小距離
    pairs=t.query_pairs(1.6,output_type="ndarray")
    short=Counter(tuple(sorted((sym[i],sym[j]))) for i,j in pairs)
    lix=pos[Li,0]%L[0]
    occ={k:int(((lix>lo)&(lix<hi)).sum()) for k,(lo,hi) in REG.items()}
    f=a.arrays.get("forces"); fmax=np.linalg.norm(f,axis=1).max() if f is not None else float("nan")
    c=Counter(sym)
    print(f"[{label}] nat={len(a)} {dict(c)}")
    print(f"  P配位S数分布={dict(sorted(cn.items()))}  頂点X={vtxX}  P-P(<2.4)={pp//2}  架橋S={bridging}  遊離S={len(freeS)}(GB内{fS_gb})")
    print(f"  最小原子間距離={dmin.min():.3f}A  <1.6A対={dict(short)}  fmax={fmax:.2f} eV/A")
    print(f"  Li数/領域={occ}")
for fn in sys.argv[1:]:
    a=read(fn)
    if len(a)<2000: a=a.repeat((1,1,3))
    analyze(a,fn)
