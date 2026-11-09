import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = "."   # directory holding charge_profile_avg.npz
d = np.load(f"{SP}/charge_profile_avg.npz")
GB = d["_GB"]; ks = ["initial", "phaseB", "generic"]
col = {"initial": "#7f8c8d", "phaseB": "#c0392b", "generic": "#2471a3"}
lbl = {"initial": "initial (as-built)", "phaseB": "phaseB", "generic": "generic"}
def sd(k, key): return d[f"{k}__{key}_sd"] if f"{k}__{key}_sd" in d.files else np.zeros_like(d[f"{k}__{key}"])

fig, ax = plt.subplots(2, 2, figsize=(11, 7))
# (a) Li density
for k in ks:
    x = d[k+"__xc"]; y = d[k+"__Li"]
    ax[0,0].plot(x, y, color=col[k], lw=1.8, label=lbl[k])
    ax[0,0].fill_between(x, y-sd(k,"Li"), y+sd(k,"Li"), color=col[k], alpha=0.18)
ax[0,0].set_title("(a) Li density $n_{Li}(x)$  (mean $\\pm$ sd, 3 seeds)"); ax[0,0].set_ylabel(r"$n$ [$\AA^{-3}$]"); ax[0,0].legend(fontsize=8)
# (b) net charge density
for k in ks:
    x = d[k+"__xc"]; y = d[k+"__rho"]
    ax[0,1].plot(x, y, color=col[k], lw=1.4, label=lbl[k])
    ax[0,1].fill_between(x, y-sd(k,"rho"), y+sd(k,"rho"), color=col[k], alpha=0.15)
ax[0,1].axhline(0, color="k", lw=0.6); ax[0,1].set_title(r"(b) Net formal charge density $\rho_q(x)$"); ax[0,1].set_ylabel(r"$\rho_q$ [$e\,\AA^{-3}$]")
# (c) potential
for k in ks:
    x = d[k+"__xc"]; y = d[k+"__phi15"]
    ax[1,0].plot(x, y, color=col[k], lw=1.8, label=lbl[k])
    ax[1,0].fill_between(x, y-sd(k,"phi15"), y+sd(k,"phi15"), color=col[k], alpha=0.18)
ax[1,0].axhline(0, color="k", lw=0.6); ax[1,0].set_title(r"(c) Macroscopic potential $\phi(x)$ ($\epsilon_r{=}15$)"); ax[1,0].set_ylabel(r"$\phi$ [V]")
# (d) summary bars
axd = ax[1,1]
labels = ["initial", "phaseB", "generic"]
dphi = [8.85, d["phaseB__scal"][2], d["generic__scal"][2]]
dphi_e = [0, d["phaseB__scal"][3], d["generic__scal"][3]]
xb = np.arange(3)
axd.bar(xb, dphi, yerr=dphi_e, color=[col[k] for k in labels], capsize=5, alpha=0.85)
for i,(v,e) in enumerate(zip(dphi,dphi_e)): axd.text(i, v+0.15, f"{v:.2f}"+("" if i==0 else f"$\\pm${e:.2f}"), ha="center", fontsize=9)
axd.set_xticks(xb); axd.set_xticklabels(labels); axd.set_ylabel(r"space-charge potential $\Delta\phi$ [V]")
axd.set_title(r"(d) GB space-charge potential (lower = more relaxed)")
for a in ax.ravel()[:3]:
    for g in GB: a.axvline(g, color="k", ls="--", lw=0.8, alpha=0.4)
    a.set_xlabel("x (GB normal) [$\\AA$]"); a.set_xlim(0, d["initial__xc"][-1])
fig.suptitle("GB charge balance (3-seed avg, 700 K, formal charges → upper-bound $\\phi$). Globally neutral; phaseB most relaxed.", y=1.0, fontsize=10.5)
fig.tight_layout(); fig.savefig(f"{SP}/fig_charge_balance_avg.png", dpi=160, bbox_inches="tight")
print("wrote fig_charge_balance_avg.png")
