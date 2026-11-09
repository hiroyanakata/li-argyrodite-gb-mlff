#!/usr/bin/env python3
"""Refit GB-bulk Arrhenius from per-seed region D (reg/reg_V2<st>_<model>_<T>K_s<seed>.json).
D(T)=mean over seeds; ln D vs 1/T unweighted, same as the 10/20 analysis.
Uncertainty: resample seeds within each temperature (only T with >1 seed contribute).

  python refit_arrhenius.py --reg_dir reg --out refit_summary.csv
reg_dir holds the msd_regions.py outputs named reg_V2<struct>_<model>_<T>K_s<seed>.json
(struct uni|seg|amo, model v5D|pretrained)."""
import json,glob,re,os,csv
import numpy as np
kB=8.617333262e-2 # meV/K
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--reg_dir',default='reg');ap.add_argument('--out',default='refit_summary.csv');args=ap.parse_args()
D={}
for f in glob.glob(os.path.join(args.reg_dir,'reg_V2*_s*.json')):
    m=re.search(r'reg_V2(\w+?)_(v5D|pretrained)_(\d+)K_s(\d)\.json',f)
    st,mo,T,s=m.group(1),m.group(2),int(m.group(3)),int(m.group(4))
    r=json.load(open(f))['regions']
    D.setdefault((st,mo),{}).setdefault(T,[]).append((s,r['GB1']['D_cm2_s'],r['GB2']['D_cm2_s'],r['BULK']['D_cm2_s'],r['ALL']['D_cm2_s']))
def fit(T,y):
    a,b=np.polyfit(1/np.array(T,float),np.log(y),1);return -a*kB
rng=np.random.default_rng(0)
rows=[]
for k in [(s,m) for s in('uni','seg','amo') for m in('pretrained','v5D')]:
    d=D[k];Ts=sorted(d)
    A={T:np.array(d[T])[:,1:] for T in Ts}
    def de(sel):
        M=np.array([sel[T].mean(0) for T in Ts])
        Ea=[fit(Ts,M[:,i]) for i in range(4)]
        return Ea[0]-Ea[2],Ea[1]-Ea[2],Ea[2],Ea[3]
    e=de(A)
    bs=np.array([de({T:A[T][rng.integers(0,len(A[T]),len(A[T]))] for T in Ts})[:2] for _ in range(2000)])
    print(f'\n{k[0]} {k[1]}: dEa GB1 {e[0]:+.1f} (boot sd {bs[:,0].std():.1f}) GB2 {e[1]:+.1f} (sd {bs[:,1].std():.1f}) Ea_BULK {e[2]:.1f} Ea_ALL {e[3]:.1f}')
    for T in Ts:
        a=A[T];r=a[:,0]/a[:,2];r2=a[:,1]/a[:,2]
        se=r.std(ddof=1)/len(r)**.5 if len(r)>1 else float('nan')
        print(f'  {T}K n={len(r)} GB1/B seeds {np.round(r,2)} mean {r.mean():.2f} se {se:.2f} RoM {a[:,0].mean()/a[:,2].mean():.2f} GB2/B {r2.mean():.2f} D_B {a[:,2].mean()*1e6:.2f}e-6')
        rows.append([k[0],k[1],T,len(r),round(r.mean(),3),round(se,3),round(a[:,0].mean()/a[:,2].mean(),3),round(r2.mean(),3),a[:,2].mean(),round(e[0],1),round(e[1],1),round(e[2],1),round(e[3],1)])
csv.writer(open(args.out,'w',newline='')).writerows([['struct','model','T','nseed','GB1_B_mean','GB1_B_se','GB1_B_ratio_of_means','GB2_B_mean','D_bulk','dEa_GB1','dEa_GB2','Ea_BULK','Ea_ALL']]+rows)
