#!/usr/bin/env python3
"""
E4 — Annotator agreement scoring.

Tests whether our automatic step labeller stays reliable when the text has been
deliberately damaged. This is Hypothesis H4.

USAGE
    python score_annotations.py annot_A.csv annot_B.csv annotation_key_DO_NOT_SHARE.csv

No GPU required. Runs in seconds.
"""
import sys, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score

VALID = {"Compute", "Verify", "Final"}

def norm(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().capitalize()
    return s if s in VALID else np.nan

def kappa(a, b):
    m = pd.notna(a) & pd.notna(b)
    if m.sum() < 5: return np.nan
    return cohen_kappa_score(np.asarray(a)[m], np.asarray(b)[m])

def main(pa, pb, pkey):
    A = pd.read_csv(pa); B = pd.read_csv(pb); K = pd.read_csv(pkey)
    for d in (A, B): d["label"] = d["label"].map(norm)

    M = (K.merge(A[["sheet_chain","step_no","label"]], on=["sheet_chain","step_no"], how="left")
           .rename(columns={"label_y":"A","label_x":"key_blank"})
           .merge(B[["sheet_chain","step_no","label"]], on=["sheet_chain","step_no"], how="left")
           .rename(columns={"label":"B"}))
    M["model"] = M["_hidden_model_label"].map(norm)
    M["is_damaged"] = ~M["_hidden_record_id"].str.endswith("C0")

    done = M[pd.notna(M.A) & pd.notna(M.B)].copy()
    print("="*66)
    print("E4 — ANNOTATOR AGREEMENT (H4)")
    print("="*66)
    print(f"steps in sheet        : {len(M)}")
    print(f"steps labelled by both: {len(done)}")
    if len(done) < 30:
        print("\n!! Too few labelled rows. Both annotators must finish before scoring.")
        return

    # human majority: agree -> that label; disagree -> annotator A (declared tie-break)
    done["human"] = np.where(done.A == done.B, done.A, done.A)
    n_dis = int((done.A != done.B).sum())

    print(f"\nhuman-vs-human disagreements: {n_dis}/{len(done)} ({100*n_dis/len(done):.1f}%)")
    print(f"\n1. Human vs human            kappa = {kappa(done.A, done.B):.3f}   (target >= 0.70)")
    print(f"2. Model vs human majority   kappa = {kappa(done.model, done.human):.3f}   (target >= 0.60)")

    cl = done[~done.is_damaged]; dm = done[done.is_damaged]
    kc = kappa(cl.model, cl.human); kd = kappa(dm.model, dm.human)
    print(f"\n3. THE HEADLINE — does the labeller degrade on damaged text?")
    print(f"     model vs human, CLEAN   kappa = {kc:.3f}   (n={len(cl)} steps)")
    print(f"     model vs human, DAMAGED kappa = {kd:.3f}   (n={len(dm)} steps)")
    print(f"     DELTA (clean - damaged)       = {kc-kd:+.3f}")

    bs = []
    for _ in range(4000):
        c = cl.sample(len(cl), replace=True); g = dm.sample(len(dm), replace=True)
        v = kappa(c.model, c.human) - kappa(g.model, g.human)
        if not np.isnan(v): bs.append(v)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"     bootstrap 95% CI of delta     = [{lo:+.3f}, {hi:+.3f}]")

    print("\n" + "-"*66)
    if lo <= 0 <= hi:
        print("VERDICT: CI includes zero — no detectable degradation on damaged text.")
        print("  -> Automated labels are trustworthy here. Report the delta and proceed.")
    else:
        print("VERDICT: CI excludes zero — the labeller measurably degrades.")
        print("  -> Report prominently and qualify every structural claim.")
    print("-"*66)

    gate = "PASS" if kd >= 0.60 else "FAIL"
    print(f"\nGATE (kappa on damaged text >= 0.60): {gate}  [{kd:.3f}]")
    if gate == "FAIL":
        print("  -> Per the proposal, automated structural findings are reported as")
        print("     provisional; the parsing limitation becomes a finding in its own right.")

    print("\nPer-label agreement (model vs human majority, damaged subset):")
    for lab in sorted(VALID):
        s = dm[dm.human == lab]
        if len(s) == 0: continue
        print(f"   {lab:8s} n={len(s):3d}  model agreed {100*(s.model==lab).mean():5.1f}%")

    out = done[["sheet_chain","step_no","step_text","A","B","model","is_damaged"]]
    out.to_csv("e4_scored.csv", index=False)
    print("\nper-step detail written to e4_scored.csv")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__); sys.exit(1)
    main(*sys.argv[1:])
