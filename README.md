# Locating Verification Failure in Chain-of-Thought Reasoning

**Design Project (SWE 4606 / CSE 4610) — Systems and Software Lab, Department of Computer Science and Engineering, Islamic University of Technology.**

When a language model writes out its reasoning, it often pauses to check its own work. We break one step of that reasoning on purpose and ask a question the literature has not asked: **does the check actually look at the broken step, or somewhere else?**

That question is about a *link between two steps*, not a property of one step, so a list of sentences cannot answer it. We build a small graph over each chain of reasoning and answer it there.

---

## Headline results

From a pilot on 96 GSM8K problems × 5 conditions (480 runs) with Qwen2.5-3B-Instruct:

| Finding | Number |
|---|---|
| Chains using verification language that examined a step **other than** the damaged one | **58.6%** (136 / 232) |
| Drop in correct targeting when the *identical* wrong value is attributed to a professor | **−22.6 pp**, McNemar *p* = 0.0007 |
| Correct targeting differs across damage types | χ²(2) = 10.83, *p* = 0.004 |
| Labelling agreement on damaged text (Cohen's κ) | 0.740 — passes the pre-set 0.60 threshold |

The most interesting number is the last row of the results table: **targeting and final accuracy rank the damage types in opposite orders.** An accuracy-only study would call the authority attack the mildest of the three; our measure shows it is where checking is most thoroughly misdirected.

---

## What the pipeline does

```
1. Generate  →  2. Filter  →  3. Damage  →  4. Continue  →  5. Measure
```

1. **Generate** — the model solves a GSM8K problem in numbered steps.
2. **Filter** — keep only problems it already answers correctly, so any later failure is caused by our damage rather than by the model being bad at the problem.
3. **Damage** — break exactly one step, in one of four controlled ways.
4. **Continue** — hand the damaged prefix back and let the model finish. Each condition is a separate request with no shared history.
5. **Measure** — split the finished text into steps, label each one, resolve back-references, build the graph, compute the metrics.

### The four damage types

| Condition | What changes | Real-world parallel |
|---|---|---|
| **C0** Control | nothing | baseline behaviour |
| **C1** Arithmetic error | step *k*: "the total is 60 × 5 = **360**" (true: 300) | an upstream agent or tool passes on a wrong value |
| **C2** Authority claim | step *k*: "*a mathematics professor states that* 60 × 5 = **360**" | a retrieved document or user asserts something false but credible |
| **C3** Missing step | step *k* deleted, the rest renumbered | an earlier trace omits a needed justification |
| **C4** Contradiction | new step after *k*: "however, that result is actually **360**, not 300" | two sources or two agents disagree |

**C1 and C2 use the identical wrong number at the identical position in a chain of the identical length.** The only difference is who says it. That is what makes the framing effect (H3) testable.

### The metrics

| Metric | Definition |
|---|---|
| **Check Initiation** | does the chain contain any step labelled *Verify*? |
| **Cue present** | does the chain contain any checking phrase? *(the keyword baseline — what existing methods measure)* |
| **Audit Hit** | does a back-reference from a *Verify* step land on the damaged step, or on a later step carrying the wrong value? |
| **Audit Displacement** | if the check missed, how far from the damage did it land? |
| **Final correct** | is the final answer right? *(what existing studies report)* |

**Audit Hit is undefined — not zero — for C0 and C3.** C0 has no damage, and in C3 the damage *is* a deletion, so there is no step left to point at. Writing 0 there would claim the model failed to check something that was never present. Those two conditions are reported on accuracy instead.

---

## Repository layout

```
.
├── notebooks/
│   └── DP_Pilot_Study.ipynb          full Colab pipeline, resumable
│
├── src/
│   ├── rescue_edge_resolution.py     rule-based back-reference resolver + metrics
│   ├── score_annotations.py          E4 annotator agreement (H4)
│   └── make_figures.py               regenerates every figure from the tables
│
├── data/
│   ├── config/
│   │   ├── config.json               frozen run configuration
│   │   ├── eligible_ids.json         the 96 problems that passed filtering
│   │   └── probe_results.json        prompt calibration (4% vs 54%)
│   ├── raw/
│   │   ├── clean_generations.jsonl   undamaged chains
│   │   ├── perturbation_records.jsonl  every damaged prefix, with metadata
│   │   └── continuation_generations.jsonl  what the model wrote afterwards
│   ├── processed/
│   │   ├── parsed_steps.jsonl        segmented chains
│   │   ├── node_labels.jsonl         Compute / Verify / Final per step
│   │   └── edges_model_resolver.jsonl  model-based resolver output (ablation)
│   └── results/
│       ├── analysis_table_FINAL.csv  one row per problem × condition
│       ├── results_summary_FINAL.csv the main results table
│       ├── sensitivity_analysis.csv  primary vs strict subset
│       ├── consort_table.csv         filtering flow, 150 → 96
│       └── e4_scored.csv             per-step annotation agreement
│
├── annotation/
│   ├── ANNOTATOR_INSTRUCTIONS.md     given to annotators verbatim
│   ├── annotation_sheet.csv          the blind sheet, 209 steps / 30 chains
│   ├── annotation_key.csv            condition + model label (see warning below)
│   ├── annot_A.csv                   annotator A's labels
│   └── annot_B.csv                   annotator B's labels
│
├── figures/                          PNGs used in the report and slides
├── diagrams/                         editable draw.io sources
├── docs/                             proposal, pipeline document, presentation
├── REPRODUCE.md                      step-by-step rerun instructions
├── requirements.txt
└── LICENSE
```

> **Warning on `annotation/annotation_key.csv`.** It reveals which chains are damaged and what the automatic labeller predicted. It must not be shown to an annotator before they have finished labelling. It is committed here only because annotation is complete and the file is needed to reproduce the E4 scores.

---

## Reproducing the results

Full instructions are in [`REPRODUCE.md`](REPRODUCE.md). The short version:

```bash
pip install -r requirements.txt

# statistics and figures, from the committed tables — no GPU, a few seconds
python src/make_figures.py
python src/score_annotations.py annotation/annot_A.csv \
                                annotation/annot_B.csv \
                                annotation/annotation_key.csv
```

Regenerating the chains themselves needs a GPU. Open `notebooks/DP_Pilot_Study.ipynb` in Google Colab, set the runtime to **T4 GPU**, and run top to bottom. Everything writes to Google Drive and is resumable, so a disconnect costs at most one batch. Expect roughly 2.5 hours and under US$5.

---

## A note on back-reference resolution

We evaluated two ways of deciding which earlier step a check refers to.

| Method | References resolved | Chains with ≥1 edge |
|---|---|---|
| Ask the model which step | 16 / 495 (3.2%) | 16 / 480 |
| Numeric-overlap rule | 405 / 495 (81.8%) | 388 / 471 |

The 3B model returned "no specific step" on 97% of flagged steps, even where the referenced step was numerically unambiguous. **All reported results use the rule.** The model-based output is kept in `data/processed/edges_model_resolver.jsonl` as the record of that comparison.

This is direct evidence for the project's cost-bounded design: the cheap deterministic component outperformed the model call.

**Limitation:** the rule has not been validated against human annotation. E4 validated the step *labels*, not the *edges*. That validation is planned thesis work.

---

## Known limitations

- **Verification is prompt-elicited.** Unprompted, this model checks its work 4% of the time. We add a neutral instruction, applied identically across all five conditions, which never says an error exists or where. It constrains *whether* the model checks, not *what* it checks — but every number should be read with this in mind.
- **"Referring to" is not "auditing".** Audit Hit records that a check mentioned the damaged step, not that it scrutinised it critically.
- **Underpowered.** 96 problems, one model, one dataset. Only large effects are detectable.
- **Short chains.** GSM8K chains average 7 steps. Graph distance and plain step distance agreed in 98.5% of cases, so the graph gave no advantage on that particular metric here.
- **Injection quality.** 15 of 288 injections produced a degenerate value (zero or negative), about 5% of cases.
- **One metric did not work.** An orphan-count measure for the missing-step condition returned zero in every case. We trialled it and are not reporting it.

---

## Related work

| Work | Venue | What it does | What it does not do |
|---|---|---|---|
| Aravindan & Kejriwal, *Fragile Thoughts* | arXiv:2603.03332 | five perturbation types across 13 models, same dataset | records only whether the final answer changed |
| von Recum, Girrbach & Akata | arXiv:2602.07470 | counts doubt-expressing sentences after an injection | counts how much doubt, never which step it concerns |
| Tsui, *Self-Correction Bench* | arXiv:2507.02778 | same error framed as the model's own vs external | measures whether the fix happened, not where checking aimed |
| Tyen et al. | Findings of ACL 2024 | models cannot *find* errors but can *fix* them when told where | the model is asked to look; ours is never asked |
| Lee et al., *ReasoningFlow* | arXiv:2506.02532 | parses reasoning into a graph, human-validated | every trace they validated was undamaged |

We do not claim to be first on the damage taxonomy, on measuring checking behaviour, or on comparing framings. What remains open is measuring **which earlier step a check points at**, scoring it against a **known fault location**, and **re-validating the labeller on damaged text**.

---

## Team

| Member | Student ID | Contribution |
|---|---|---|
| Ferdous Reza Habib | 220041138 | Literature review, statistical analysis |
| Shadab Bin Habib | 220041201 | Graph construction, generation runs |
| Subarno Neel | 220041206 | Pipeline implementation, perturbation design |

Supervised by the Systems and Software Lab (SSL), Department of CSE, IUT.

---

## Licence

MIT — see [`LICENSE`](LICENSE). GSM8K is distributed by OpenAI under its own terms.
