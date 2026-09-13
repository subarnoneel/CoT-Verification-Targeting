import json, re, numpy as np, pandas as pd
from collections import Counter
import networkx as nx
from scipy import stats

R = "/home/claude/an/res"
S = [json.loads(l) for l in open(f"{R}/parsed_steps.jsonl")]
L = {}
for l in open(f"{R}/node_labels.jsonl"):
    r = json.loads(l); L[r["record_id"]] = r["label"]

CUES = re.compile(r"(wait|hold on|actually|let me (double.?)?check|recheck|recompute|"
                  r"re-?examine|verif(y|ies|ying)|as (computed|shown|found|noted) (above|earlier)|"
                  r"revisit|going back|earlier (i|we) (said|got|computed)|"
                  r"that (contradicts|doesn't match|isn't right|is wrong)|but earlier|"
                  r"hmm|on second thought|correction|instead|double.?check|confirm)", re.I)
NUM = re.compile(r"-?\d+(?:\.\d+)?")

def nums(t):
    return set(float(x) for x in NUM.findall(t.replace(",", "")))

def resolve_target(step_text, earlier_steps):
    """Rule-based back-reference resolution: the earlier step whose numeric content
       best matches the numbers in the checking step. Returns 1-indexed pos or None."""
    tn = nums(step_text)
    if not tn: return None
    best, best_score = None, 0.0
    for i, s in enumerate(earlier_steps, start=1):
        sn = nums(s)
        if not sn: continue
        inter = tn & sn
        if not inter: continue
        # Jaccard, with a tie-break preferring the NEAREST earlier step
        score = len(inter) / len(tn | sn)
        if score > best_score + 1e-9 or (abs(score-best_score) < 1e-9 and best is not None and i > best):
            best, best_score = i, score
    return best if best_score >= 0.34 else None   # need meaningful overlap

rows = []
for ch in S:
    rid = ch["record_id"]; steps = ch["steps"]; n = len(steps)
    labels = [L.get(f"{rid}#n{i}", "Compute") for i in range(1, n+1)]

    G = nx.DiGraph()
    for i, s in enumerate(steps, 1):
        G.add_node(f"p{i}", text=s, pos=i, label=labels[i-1])
    for i in range(1, n): G.add_edge(f"p{i}", f"p{i+1}", kind="seq")

    refs = []
    for i, s in enumerate(steps, 1):
        if i == 1 or labels[i-1] != "Verify": continue
        if not CUES.search(s): continue
        tgt = resolve_target(s, steps[:i-1])
        if tgt:
            G.add_edge(f"p{i}", f"p{tgt}", kind="ref"); refs.append((i, tgt))

    ip = ch["injected_pos"]
    cs = set()
    if ip:
        cs = {ip} | {j for j in range(ip+1, n+1) if str(ch["wrong_val"]) in steps[j-1]}

    verify = [i for i in range(1, n+1) if labels[i-1] == "Verify"]
    # exclude a Verify label that landed ON the injected node itself (it is the damage,
    # not a check of the damage)
    verify_clean = [i for i in verify if i != ip]
    init = int(len(verify_clean) > 0)

    hit = None; dg = do = None
    if cs and refs:
        hit = int(any(t in cs for _, t in refs))
        if hit == 0:
            U = G.to_undirected()
            cand_o = [abs(t - ip) for _, t in refs]
            cand_g = []
            for _, t in refs:
                try: cand_g.append(nx.shortest_path_length(U, f"p{t}", f"p{ip}"))
                except nx.NetworkXNoPath: pass
            do = min(cand_o) if cand_o else None
            dg = min(cand_g) if cand_g else None
    elif cs and not refs:
        hit = 0   # a check may exist but nothing resolvable points at the damage

    orphan = None
    if ch["condition"] == "C3" and ch.get("deleted_val"):
        dv = str(ch["deleted_val"])
        uses = [i for i in range(1, n+1) if dv in steps[i-1]]
        orphan = 0 if any(re.search(r"[+\-*x×/]", steps[i-1]) and dv in steps[i-1] for i in uses) else len(uses)

    rows.append(dict(record_id=rid, problem_id=ch["problem_id"], condition=ch["condition"],
                     n_steps=n, continuation_mode=ch["continuation_mode"],
                     check_initiation=init, n_verify=len(verify_clean),
                     n_ref_edges=len(refs), audit_hit=hit,
                     disp_graph=dg, disp_ordinal=do, orphan_count=orphan,
                     final_correct=ch["final_correct"],
                     cue_present=int(any(CUES.search(s) for s in steps)),
                     injected_pos=ip,
                     wrong_val=ch.get("wrong_val"), true_val=ch.get("true_val")))

df = pd.DataFrame(rows)
df.to_csv("/home/claude/an/analysis_table_RESCUED.csv", index=False)

print("="*70)
print("RESCUED ANALYSIS  (rule-based back-reference resolution)")
print("="*70)
print(f"\nchains with >=1 resolved ref edge: {(df.n_ref_edges>0).sum()} / {len(df)}"
      f"   (was 16 / 480)")

d = df[df.continuation_mode == "continued"]
NAMES = {"C0":"Control","C1":"Arithmetic","C2":"Authority","C3":"Missing step","C4":"Contradiction"}
g = d.groupby("condition").agg(
    n=("record_id","size"), Init=("check_initiation","mean"),
    RefEdges=("n_ref_edges","mean"), Hit=("audit_hit","mean"),
    Cue=("cue_present","mean"), Steps=("n_steps","mean"),
    Correct=("final_correct","mean"))
g.index = [f"{i} {NAMES[i]}" for i in g.index]
print("\n" + g.round(3).to_string())

sub = d[d.condition.isin(["C1","C2","C4"])].dropna(subset=["audit_hit"])
print(f"\nrows usable for Audit Hit: {len(sub)}")
print("\nAudit Hit by condition:")
print(sub.groupby("condition").audit_hit.agg(["size","sum","mean"]).round(3).to_string())

if sub.audit_hit.nunique() > 1:
    tab = pd.crosstab(sub.condition, sub.audit_hit)
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    V = np.sqrt(chi2/(tab.values.sum()*(min(tab.shape)-1)))
    print(f"\nchi2({dof}) = {chi2:.2f}, p = {p:.4f}, Cramer's V = {V:.3f}")

print("\n--- displacement (how far the check landed from the damage) ---")
print(d.groupby("condition")[["disp_ordinal","disp_graph"]].median().to_string())
print("\ndisp_ordinal distribution (C1/C2/C4):")
print(sub.disp_ordinal.value_counts().sort_index().to_string())

