# Reproducing this study

Two paths. **Path A** re-derives every number and figure from the committed data in
about a minute on a laptop. **Path B** regenerates the reasoning chains from scratch
and needs a GPU.

---

## Path A — statistics and figures only (no GPU)

```bash
git clone https://github.com/subarnoneel/CoT-Verification-Targeting.git
cd CoT-Verification-Targeting
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 1. Regenerate the figures

```bash
python src/make_figures.py
```

Reads `data/results/analysis_table_FINAL.csv` and `data/results/e4_scored.csv`.
Writes three PNGs into `figures/`. They should be byte-similar to the committed ones.

### 2. Re-score the annotation study (E4 / H4)

```bash
python src/score_annotations.py annotation/annot_A.csv \
                                annotation/annot_B.csv \
                                annotation/annotation_key.csv
```

Expected output:

```
Human vs human            kappa = 0.988
Model vs human majority   kappa = 0.793
  model vs human, CLEAN   kappa = 0.856
  model vs human, DAMAGED kappa = 0.740
  DELTA                         = +0.116
  bootstrap 95% CI of delta     = [-0.063, +0.297]
GATE (kappa on damaged text >= 0.60): PASS
```

### 3. Rebuild the analysis table from the graph data

```bash
python src/rescue_edge_resolution.py
```

Reads `data/processed/parsed_steps.jsonl` and `data/processed/node_labels.jsonl`,
re-resolves back-references with the numeric-overlap rule, rebuilds every graph and
recomputes the metrics. Overwrites `data/results/analysis_table_FINAL.csv`.

Sanity check: 388 of 471 usable chains should carry at least one resolved
back-reference edge.

---

## Path B — regenerate the chains (needs a GPU)

1. Upload `notebooks/DP_Pilot_Study.ipynb` to Google Colab.
2. **Runtime → Change runtime type → T4 GPU.**
3. Run the cells top to bottom.

Everything writes to Google Drive under `MyDrive/dp_pilot/<RUN_NAME>/` and every
generation stage is resumable — it skips records already present in the output file.
After a disconnect, re-run cells 1–4 and jump back to the interrupted cell.

**Do not skip the calibration cell.** It tests 24 problems under two prompts and
decides which one the main run uses. Without it the model checks its work about 4%
of the time and every targeting metric collapses to zero for want of anything to
measure.

Approximate cost and time on a free Colab T4:

| Stage | Time |
|---|---|
| Setup and model download | 45 min |
| Calibration probe | 10 min |
| Clean generation (150 problems) | 12 min |
| Continuation (96 × 5 conditions) | 35 min |
| Labelling (~3,500 steps) | 8 min |
| Back-reference detection | 4 min |
| Graphs and metrics | 3 min |
| **Total** | **≈ 2 h** |

Total inference spend was under **US$5**.

---

## Frozen configuration

`data/config/config.json` holds every setting used for the reported run. The values
that matter for reproducibility:

| Setting | Value |
|---|---|
| Dataset | `openai/gsm8k`, config `main`, split `test` |
| Model | `Qwen/Qwen2.5-3B-Instruct` |
| Decoding | greedy, temperature 0 |
| Seed | 42 |
| Problems sampled | 150 |
| Eligible after filtering | 96 |
| Injection position | earliest arithmetic-bearing step in positions 2–4 |

Use the config `main`, **not** `socratic` — the socratic variant pre-splits solutions
into question/answer sub-steps, which would contaminate our segmentation.

---

## Expected headline numbers

If a rerun reproduces the study, these should come out within sampling noise:

| Quantity | Value |
|---|---|
| Eligible problems | 96 of 150 (64.0%) |
| Continuations that continued rather than restarted | 471 of 480 (98.1%) |
| Audit Hit — C1 / C2 / C4 | 0.447 / 0.221 / 0.351 |
| Chains with verification language that missed the damage | 136 of 232 (58.6%) |
| E3 matched pairs, discordant | 28 vs 7, McNemar *p* = 0.0007 |
| E4 kappa on damaged text | 0.740 |

Note that Path B will not reproduce these *exactly*. Greedy decoding is deterministic
for a fixed model build, but library and kernel versions change the sampled 150
problems and therefore the eligible set. The direction and rough size of every effect
should hold.
