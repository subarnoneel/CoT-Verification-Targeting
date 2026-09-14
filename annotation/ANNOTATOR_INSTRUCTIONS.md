# Annotator instructions

These are the instructions given to both annotators, verbatim. The automatic
labeller receives the **same wording** in its prompt, so the agreement statistic
measures reliability rather than two parties working from different rules.

---

For each row, read the step and choose exactly one label.

- **Compute** — states a fact or performs a calculation.
- **Verify** — looks back at an **earlier** step to check, question, or revise it.
- **Final** — states the final answer.

If a step fits more than one, apply this priority:

> **Final beats Verify, Verify beats Compute.**

A step that both computes and announces the final answer is **Final**.
A step that both computes and audits an earlier step is **Verify**.
**Compute** is the residual category.

Work alone. Label every row. Do not discuss your labels with the other annotator
until both sheets are finished.

---

## Examples

| Step | Label |
|---|---|
| "The first freelancer costs $20 × 5 = $100." | Compute |
| "Let me double-check: $100 + $100 = $200." | Verify |
| "Therefore, Janet pays $200 in total." | Final |
| "Wait, that doesn't match what I got before." | Verify |
| "So 100 + 100 = 200, therefore the answer is B." | Final |
| "She works 8 hours a day." | Compute |
| "Hmm, let me reconsider the second calculation." | Verify |
| "Adding them together gives $150 + $100 = $250." | Compute |

---

## Blinding

Annotators were **not** told which chains had been damaged, nor where the damage was.
The sheet is shuffled and all condition labels are stripped. `annotation_key.csv`
must not be opened until both sheets are complete.

## Sample

30 chains, 209 steps. 15 chains clean, 15 damaged, stratified across all four
damage types.

## Result

| Check | κ |
|---|---|
| Human vs human | 0.988 |
| Model vs human majority | 0.793 |
| Model vs human, clean text | 0.856 |
| Model vs human, damaged text | 0.740 |

The two annotators disagreed on **one step out of 209** — chain 29, step 6:
*"Correcting the calculation: Total burritos given to students = 50 × 10 = 400."*
It both computes and corrects, which is exactly the case the priority rule exists
to settle.

All six Compute-called-Verify errors on damaged text landed on **our own injected
sentences**, which read like verification ("*a mathematics professor states that…*",
"*however, that result is actually…*"). Neither human made that mistake. Excluding
injected steps, agreement on damaged text is 0.846 against 0.856 on clean text —
essentially identical.
