# V92 Placebo Integrity Incident

Date: 2026-07-14

Status: corrected rerun completed; V90 replication remains valid; original V92 placebo conclusion is void; corrected placebo does not confirm event-specific alpha.

## Summary

V92 exactly reproduced the frozen V90 experiment, but its within-month event-feature placebo did not alter any feature-to-stock assignment in the JoinQuant runtime. The real event and placebo models therefore received identical matrices and produced identical predictions, Top8 selections, feature importances, and performance.

The zero real-minus-placebo result is not a financial finding. It is an invalid negative control.

## Evidence

- All 33 exact-replication gates passed, including fold samples, feature semantics, monthly metrics, and Top8 sets.
- All 129,216 aligned real/placebo prediction scores were exactly equal.
- All 681 aligned real/placebo feature-importance rows were exactly equal.
- Every direct, rerank, and cap2 real-minus-placebo metric and bootstrap bound was exactly zero.

## Root Cause

The shuffle loop converted dates with `Series.drop_duplicates().tolist()` and compared the returned values with a `datetime64[ns]` NumPy array. In the old pandas runtime used by JoinQuant, the list values can be integer nanosecond timestamps. Comparing those integers with the datetime array matched no rows, so every monthly position array was empty and the shuffle silently became a no-op.

The local smoke test used a newer pandas version whose `Series.tolist()` returned `Timestamp` objects, so the same code happened to work locally. The test asserted only that placebo outputs existed, not that the placebo input matrix changed.

## Impact

Valid:

- V92's exact V90 replication.
- The conclusion that the frozen V90 event model had about `+0.635%` mean monthly Top8-edge increment under its original specification, with a positive six-month block-bootstrap lower bound.
- The cap2 policy's performance relative to V46 within the real-event branch.

Invalid until rerun:

- Every V92 real-versus-placebo comparison.
- Signal or promotion decisions requiring placebo advantage.
- Any claim that event information did or did not beat a matched random control.

## Correction

The shuffle now uses `pd.factorize()` integer month codes. Before any placebo model is trained, a hard integrity gate checks:

1. every row belongs to a month group;
2. at least 5% of row-level event vectors changed assignment;
3. each month's sorted values for every shuffled feature are exactly preserved.

The notebook writes `v92_placebo_shuffle_audit.csv` and stops immediately if any split/fold/seed fails.

## Permanent Rules

- Negative controls require an input-level intervention audit, not just output files.
- A no-op placebo is an experiment failure, never a zero-effect conclusion.
- Cross-version datetime logic must use integer codes or group indices.
- Smoke tests must assert that treatment matrices and resulting synthetic predictions differ while preserved-distribution invariants hold.

## Corrected Rerun Outcome

The corrected JoinQuant rerun passed all 33 replication gates and all 30 placebo-integrity rows. Depending on fold and split, `42.1%` to `60.6%` of row-level event vectors changed assignment, while every within-month feature distribution was preserved exactly. Real and placebo prediction panels were no longer identical.

The frozen real-event model still improved mean monthly Top8 edge versus V46 by `+0.635%`, with a six-month block-bootstrap interval of approximately `[+0.171%, +1.094%]`. The cap2 policy improved it by `+0.618%`, with interval `[+0.175%, +0.989%]`, positive results in all five folds, all three model seeds, and a `24.38%` replacement rate.

However, the matched-placebo comparison did not pass:

- direct real-minus-placebo: `+0.100%/month`, interval `[-0.783%, +1.005%]`;
- cap2 real-minus-placebo: `+0.150%/month`, interval `[-0.390%, +0.484%]`;
- Top30 rerank real-minus-placebo: `+0.404%/month`, interval `[-0.118%, +0.920%]`.

The real mapping beat placebo mainly in 2022-2024. Its direct advantage reversed in 2025-2026. A single matched permutation per model seed is also too noisy to estimate the null distribution reliably.

Final disposition: keep V90 event features as a research candidate, but do not promote them to an execution or live-trading candidate. The next admissible test is a preregistered multi-permutation placebo distribution using the frozen V90 treatment and no additional feature or model changes.
