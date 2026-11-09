#!/usr/bin/env python
# =============================================================================
# tier3_fwV2.py -- framework (Tier-3) summary of the all-atom gbV2 runs
# (framework_md.py --out fwV2_<model>_<cell>_<T>K -> *_fwtraj.npz).
#
# Thin driver around tier3_analysis.analyze() (keep tier3_analysis.py in the same
# directory): PS4 reorientation C1(tau), halide / free-S / Li MSD in the GB and
# bulk windows at lag 10 ps. Values used for the framework table/figure of the
# paper (C1 at 700 K, halide MSD at 700 K, Li GB/bulk MSD ratio at 500 K).
#
#   python tier3_fwV2.py fwV2/fwV2_*_fwtraj.npz --out fwV2/tier3_fwV2_summary.json
# =============================================================================
import argparse, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_analysis as t3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--lag", type=float, default=10.0)
    ap.add_argument("--stride", type=float, default=1.0, help="time-origin stride [ps]")
    ap.add_argument("--out", default="tier3_fwV2_summary.json")
    a = ap.parse_args()
    out = {}
    for f in a.files:
        key = re.sub(r"^fwV2_|_fwtraj\.npz$", "", os.path.basename(f))
        r = t3.analyze(f, lag_ps=a.lag, stride_ps=a.stride)
        out[key] = t3.jsonable(r)
        s = (r["ps4"]["GB"]["c1"][-1], r["halide"]["GB"]["msd"][-1],
             r["Li"]["GB"]["msd"][-1] / r["Li"]["BULK"]["msd"][-1])
        print("%-22s C1_GB(lag)=%.3f  halideMSD_GB=%.2f A2  Li MSD GB/bulk=%.2f" % ((key,) + s), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
