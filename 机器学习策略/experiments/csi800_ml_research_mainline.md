# CSI800 ML Research Mainline

Last updated: 2026-07-15

## Purpose

This document is the standing research contract for CSI800 machine-learning stock selection experiments.
Before writing or modifying any CSI800 ML notebook, read this file first and make the experiment explicitly consistent with it.

The goal is not to produce the highest single backtest curve. The goal is to build a relatively robust, explainable, reproducible, and JoinQuant-deployable model system for monthly CSI800 stock selection.

## Primary Objective

Build a model system that can select CSI800 stocks with durable out-of-sample edge under realistic monthly rebalancing.

The system should satisfy four properties:

1. Robustness: performance should not depend on one year, one market style, one rebalance month, or one lucky top holding.
2. Reproducibility: offline research, exported model bundles, and JoinQuant backtest code should use aligned features, labels, training windows, and portfolio rules.
3. Explainability: we must understand the major return drivers, factor exposures, failure modes, and why a change should help.
4. Deployability: the strategy should be simple enough to load as pkl models in JoinQuant, handle missing data and tradability filters, and degrade safely when signals weaken.

## Core GitHub Anchors

The public `zycoldness/quant-research` package is now the primary external anchor for this research line. Current experiments have mostly not beaten that package, so every new experiment must compare against these three roles before being considered meaningful:

1. Practical baseline: `model_candidate_v46_lgb_direct_hybrid_l2_ff10_2019_2025q1_legacy_unsealed.pkl`
   - Role: current main production-style baseline.
   - Structure: direct LightGBM monthly alpha model, top30 candidate pool, industry-neutral top10 portfolio.
   - Strength: best observed live/backtest parity among existing lines.
   - Caveat: `legacy_unsealed` label boundary is less strict and may be somewhat optimistic, so it is a practical anchor rather than a clean research proof.

2. Clean single-model anchor: `model_candidate_v410_fixed_iter20_rolling5y_l2_ff10_2025q4.pkl`
   - Role: newer fixed-iteration single-model observation line.
   - Structure: direct model with stricter label-safe policy and later train cutoff.
   - Strength: cleaner research interpretation than v46.
   - Caveat: must be judged against v46 because cleaner does not automatically mean stronger.

3. Attack/cascade anchor: `model_candidate_v54_portfolio_oof_rerank_v410_q1.pkl`
   - Role: recall-rerank observation line.
   - Structure: recall40 -> OOF rerank -> top5 portfolio.
   - Strength: concrete cascade implementation that can guide future rerank work.
   - Caveat: more aggressive topN and different portfolio profile; compare path, drawdown, and concentration, not just return.

These anchors are different from the later `fixed_jq_baseline_lgb` research baseline. The GitHub package uses the V4 hybrid-light feature stack and a tested JoinQuant backtest shell; many later JQ-only experiments were cleaner but not directly comparable in strength.

## Current Anchor

The current anchor is:

- Universe: CSI800 / `000906.XSHG`.
- Frequency: monthly rebalance.
- Label: `alpha_1m` / next-month excess return style label has been the strongest and most stable among tested labels.
- Model: LightGBM regression.
- Feature direction: fixed JQ factor pool is currently more reliable than dynamic replacement of factor pools.
- Strong practical baseline: GitHub v46 legacy_unsealed direct model.
- Strong clean baseline: GitHub v410 fixed20 label-safe model.
- Research-minimal baseline: `fixed_jq_baseline_lgb` from V54.
- Preferred current deployment candidate: V97 Money2, using the frozen V46 training protocol plus validated money-flow quality and temporal information.
- Required deployment shadow: V46 full or current Money, because Money2 improves the Top8 tail but has not fully cleared the preregistered adjacent-cutoff stability thresholds.

The anchor is not final. It is the benchmark set that every new experiment must beat or clarify. A new experiment that only beats the research-minimal baseline but loses to v46/v410 should be treated as a diagnostic, not a replacement candidate.

## 2026-07-15 Mainline Update

V90-V98 tested genuinely new information after V83-V89 showed that the existing feature set was the main bottleneck.

- V90 found positive Top8 directions in earnings events and money behavior, but neither was immediately promotable.
- V91 was not a semantic reproduction of V90. Corrected V92 exactly reproduced V90, then showed that a matched event placebo performed similarly. Event-specific Alpha is therefore not confirmed.
- V94 confirmed that the first Money feature set improved Top8 metrics versus full V46, while fixed score blends did not add stable value.
- V96 Money2 produced the strongest current top-tail evidence: versus full V46, Precision Lift improved from about `1.81x` to `2.31x`, MAP@8 from about `0.0171` to `0.0262`, and monthly Top8 edge from about `1.320%` to `2.095%`. RankIC was slightly lower, so this is a top-tail improvement rather than a universal ranking improvement.
- Money2 improved Top8 edge versus current Money by about `+0.599 percentage points/month`, won across all three seeds, and had about `98.6%` block-bootstrap probability of a positive edge increment. Its adjacent-cutoff score correlation (`0.692`) and Top8 overlap (`28.6%`) were slightly below the preregistered update thresholds, so V46/current Money remains a mandatory shadow.
- V97 exported the Money2 model. Real JoinQuant backtests were directionally consistent with the offline result: Money2 was stronger than first-generation Money and remained operationally acceptable.
- Final V98 reproduced the V96 Money2 baseline and found no Top8 improvement from shareholder/unlock, actual-fundamental-reaction, holder-structure, or all-available external feature sets. P2 raised RankIC but lowered Top8 edge, directly confirming that broad cross-sectional ranking and Top8 portfolio quality are not equivalent objectives.
- V98 also found a year-coverage break in the sparse unlock-ratio fields. Reject the current P2 bundle, but do not generalize that result to a future pure shareholder-change treatment with a clean data contract.
- True row-level analyst expectation revision remains untested because no source with reliable publication date, forecast horizon, institution/analyst identity, and forecast value has been connected.

Current production-research default:

- CSI800, monthly Top8, board caps, raw `alpha_1m`, LightGBM L2, fixed120, and `feature_fraction=1.0`.
- Money2 as the preferred deployment candidate.
- V46 full or current Money as a fixed-cutoff shadow and rollback baseline.
- No event overlay, broad external-information fusion, hard risk veto, or fixed model-score blend.
- RankIC is a secondary broad-ranking health metric. Promotion decisions prioritize Precision Lift, MAP/NDCG, Top8 edge, update stability, and real JoinQuant execution.

The consolidated evidence is recorded in `机器学习策略/experiments/weekly_reviews/2026-07-15_csi800_v61_v98_stage_review.md`.

## 2026-07-13 Mainline Update

V83-V89 materially narrowed the research space. The current production-research default is now explicit:

- Keep CSI800, V46 full 43 features, raw `alpha_1m` regression, fixed120, and `feature_fraction=1.0` as the mainline.
- Robust labels, Huber/L1, rank/LambdaRank, recency weighting, market-state features, and `feature_fraction<1` did not produce stable cross-fold and cross-seed improvement.
- V86 found a feature-information bottleneck: historical nearest-neighbor labels remain noisy and do not reproduce V46 RankIC. Further tuning on the same information set has low expected value.
- Liquid All-A 1500 materially improved broad RankIC but did not improve Precision Lift or recent Top8 stability. Treat it as a separate small/mid-cap challenger, not a CSI800 replacement.
- Broad price, drawdown, liquidity, financial-distress, and model-consensus veto layers failed V89. Do not add them to production. Recent repeated limit-down events may be logged as a `REVIEW` reason, but are not an automatic veto.
- The next source of model improvement should be genuinely new information, especially expectation revisions, event/announcement data, capital-flow behavior, crowding, and higher-frequency price-volume structure.

Detailed evidence and metric definitions are recorded in `机器学习策略/experiments/weekly_reviews/2026-07-13_csi800_v83_v89_review.md`.

## Current Evidence

Observed so far:

- `alpha_1m` regression is materially better than classification labels and several alternative labels.
- Early stopping has been unstable and should not be used in future notebooks unless we explicitly reopen that decision.
- JQ-only factor models are cleaner and more reproducible than models that depend heavily on self-built price/liquidity/temporal features.
- Adding a small number of useful factors can help, but broad factor expansion easily becomes noise.
- Dynamic factor-pool replacement did not beat the fixed baseline in V54.
- V54 suggests rolling retraining plus a fixed broad factor pool is stronger than aggressive dynamic factor deletion.
- The GitHub v46/v410 anchors preserve the V4 hybrid-light features, especially liquidity/price/temporal fields, while many later JQ-only experiments removed them. This is a likely reason later models were cleaner but weaker.
- The existing v54 recall-rerank line is the only current rerank anchor with a reasonably aligned online/offline candidate-generation process. New rerank work should start by matching or simplifying that line, not by inventing a disconnected rerank distribution.
- Rerank has not yet shown stable improvement over a strong direct model, but remains a valid direction if recall gap diagnostics prove there is room to improve top30-to-top10 ordering.
- External evidence supports ML plus factor/characteristic return prediction as a viable research direction, especially tree models and neural networks that capture nonlinear interactions. The same evidence also warns that factor returns are cyclical, noisy, and prone to long underperformance periods. A robust system must expect weak years and define monitoring/fallback rules instead of treating every weak year as proof that the model is invalid.
- V58 showed that a notebook based mainly on label/topN diagnostics can rank models differently from JoinQuant backtests. Future notebooks must include an offline portfolio simulator that approximates the actual deployment path: score top30, portfolio constraints, equal-weight topN, turnover, approximate costs, benchmark excess, drawdown, target list, and a clear report of which JoinQuant tradability filters cannot be reproduced from the CSV.
- V96/V98 showed that a model can improve broad RankIC while weakening the Top8 portfolio, or improve Top8 while slightly weakening RankIC. RankIC must not be used as the sole promotion metric for a concentrated long-only strategy.
- Sparse event features must be audited by calendar year, not only by aggregate row coverage. A mostly-zero field can receive high tree gain during the years where it exists and then disappear in later OOS periods, creating hidden source drift.

## Train-Inference And Evaluation Parity

This is a P0 research rule. If train, offline evaluation, and JoinQuant inference are not aligned, the experiment cannot be used to judge algorithm quality.

Every deployable ML experiment must explicitly state and verify:

1. Feature parity
   - Training features must be exactly reproducible by the JoinQuant backtest or live inference path.
   - Missing-value fill rules, factor names, price adjustment mode, feature dates, universe filters, and column order must be recorded in the exported bundle or manifest.
   - A model cannot be promoted if the notebook uses features that the backtest cannot rebuild at `context.previous_date`.

2. Universe and filter parity
   - The notebook must use the same base universe, listing-age rule, liquidity rule, board filter, paused/ST/limit filter assumptions, and min-lot constraints as the JoinQuant strategy, or clearly mark which filters cannot be reproduced offline.
   - If a filter exists only in JoinQuant, the offline report must label the result as a diagnostic proxy, not a backtest-equivalent result.

3. Label and execution timing parity
   - Labels and portfolio simulation must match the intended execution path as closely as available data allows.
   - A close-to-close proxy such as `feature_date close -> future close` is only a signal diagnostic. It is not acceptable as the final model-selection metric when the live path trades next session at open or at a scheduled intraday time.
   - The preferred offline simulator should use: signal generated from data available at `feature_date`, entry on the next tradable rebalance time, exit on the next rebalance time, realistic costs, turnover, cash, and failed-order handling.

4. Portfolio construction parity
   - Offline evaluation must simulate the same topN, equal-weight or target-weight rule, holding buffer, industry/board constraints, min-share lot, cash sufficiency, and skip-next-candidate behavior as the JoinQuant backtest.
   - Model score quality and portfolio return are separate diagnostics. IC/topK hit/random percentile can explain the signal, but JQ-like portfolio simulation is required before exporting or trusting a candidate.

5. Acceptance threshold
   - A new notebook result is considered actionable only if its offline JQ-like simulator and real JoinQuant backtest rank candidate models in the same broad direction.
   - If the two disagree, fix the simulator or execution assumptions first. Do not continue optimizing the model on a broken proxy.

## Non-Goals

Avoid these unless explicitly reopened:

- Do not chase a single spectacular backtest result.
- Do not replace the whole framework because one short period underperformed.
- Do not add complex multi-model machinery before proving a simple version has value.
- Do not rely on early stopping or small validation slices.
- Do not use walk-forward exported models when the intended backtest is a fixed train-end model.
- Do not let research notebooks depend on hidden in-memory DataFrames unless the notebook clearly rebuilds or loads the data.
- Do not use `globals()` or `locals()` to check JoinQuant API availability.
- Do not introduce features that cannot be reproduced in JoinQuant backtest code.
- Do not evaluate new CSI800 ML experiments only against weak or convenient baselines. Always report where the result stands versus v46 practical, v410 clean, and v54 cascade anchors when comparable.
- Do not treat close-to-close proxy returns, label IC, topK hit rate, or random percentile alone as enough evidence for deployment. They are diagnostics; the model-selection layer must use a JoinQuant-like portfolio simulator or real JoinQuant backtest.

## System Design Target

The target system should have three layers:

1. Baseline model bank
   - Fixed factor set.
   - Fixed label definition.
   - Fixed LightGBM training recipe.
   - Explicit train start and train end dates.
   - Exported pkl bundle names must include train date range.

2. Challenger experiments
   - Rerank, factor subset, factor family weighting, parameter changes, or portfolio construction changes.
   - A challenger should preserve the baseline when disabled.
   - A challenger should be tested against the same train/test windows and portfolio rules.

3. Monitoring and fallback
   - Monitor recent RankIC, TopK recall, monthly excess, drawdown, turnover, and board/industry/style exposure drift.
   - If a challenger weakens, fall back to the baseline model rather than forcing a regime switch.
   - If the baseline weakens, reduce confidence, increase diversification, and require new out-of-sample evidence before replacement.

## Evaluation Standard

Every serious experiment should output at least:

- Summary by strategy/profile/window.
- Monthly return and excess return.
- Drawdown path or monthly drawdown contribution.
- RankIC and ICIR by year/window.
- Top10 and Top20 portfolio profiles.
- Constrained profile if relevant, such as industry/board-cap top10.
- Turnover.
- Board exposure: main board, ChiNext, STAR.
- Drop-top-month or top-contribution stress.
- Latest targets for review when models are export candidates.
- Offline portfolio simulation outputs for exported-model candidates: monthly gross return, approximate net return, benchmark excess, turnover, cost estimate, cumulative return, drawdown, and target list. IC/label diagnostics are not enough for model selection if the intended use is a JoinQuant portfolio backtest.
- Train-inference parity report: feature parity, universe/filter parity, execution timing, portfolio construction, min-lot/cash constraints, and any known differences from the JoinQuant backtest file.

Preferred comparison windows:

- 2016-2022 train -> 2023 test, if data is available.
- 2017-2023 train -> 2024 test.
- 2018-2024 train -> 2025 test.
- 2019-2025 train -> 2026 test.

When using different data availability, state the exact train and test dates.

## Acceptance Rules

A new model or strategy is worth keeping only if most of these are true:

- It beats or matches the anchor in multiple OOS windows, not just one year.
- Top10 and Top20 point in the same direction.
- It improves either return, drawdown, turnover, or exposure stability without a hidden unacceptable cost.
- Its monthly path is not dominated by one or two lucky months.
- Its factor/board/industry exposure is explainable.
- It can be exported and loaded in a JoinQuant backtest without changing the research logic.

Treat it as a candidate, not a strategy, if:

- Only one profile wins.
- Only one year wins.
- It improves return but worsens drawdown and concentration.
- It depends on features that are hard to reproduce online.

Reject or pause it if:

- It loses to the anchor across most windows.
- It wins only through obvious style concentration.
- It requires fragile timing thresholds.
- A shuffled, stale, or otherwise negative-control variant performs similarly.

## Experiment Design Rules

One experiment should change one concept at a time when possible.

Good experiment examples:

- Same data, same label, same model, compare fixed factor set vs one factor-family adjustment.
- Same model, compare `alpha_1m` vs one new label.
- Same baseline score, compare direct top10 vs top30 local rerank.
- Same exported model, compare portfolio construction rules.

Bad experiment examples:

- Change data, label, model, factor set, and portfolio rule together.
- Train a new model and also change the backtest executor.
- Use a notebook that cannot rebuild or explicitly load its data.
- Add a complex ensemble without proving the base learner is strong.

## Exact Replication Gate

This gate is mandatory whenever a notebook uses words such as `reproduction`, `parity`, `confirmation`, or claims that a prior result has been overturned.

1. Freeze the reference artifact before changing anything:
   - exact input file and required columns;
   - row keys, date range, row/month counts, and target;
   - candidate and selected feature lists, order, dtypes, missing-value rules, and coverage;
   - train/test boundaries, seeds, model parameters, fixed iteration count, and portfolio rule;
   - reference monthly outputs and acceptance tolerances.
2. Run the reference implementation first. The notebook must stop before challengers if feature identity, selected features, sample counts, or monthly metrics fail their tolerance.
3. A feature rebuilt from the same source is a new treatment unless row-level equality against the frozen feature is proved. Similar names, economic meaning, or aggregate coverage do not establish equivalence.
4. Do not combine replication with source refetching, parser fixes, feature expansion, or portfolio changes. After exact replication passes, change one layer at a time and label each treatment accurately.
5. Smoke tests must cover semantics as well as execution. In addition to running every cell, assert frozen feature columns, selected features, fold/seed counts, reference metric tolerances, placebo integrity, and portfolio replacement caps.
6. A failed replication cannot invalidate the prior experiment. It invalidates only the attempted replication until the discrepancy is explained.

## Negative-Control Integrity Gate

Placebo, shuffle, delay, and permutation controls must prove that the intended intervention actually happened before their performance can be interpreted.

1. Record an input-level audit before model fitting: changed row count/rate, grouping coverage, treatment columns, split, fold, and seed.
2. For within-group permutation controls, prove both sides of the contract: row-to-stock assignment changed, while every group's feature distribution was preserved.
3. Fail closed when the changed rate is below its preregistered minimum, any row is left outside a group, or the preserved-distribution check fails.
4. Identical treatment and placebo predictions are an implementation alarm until the input-level audit passes. They are not evidence that the treatment has no value.
5. Compatibility tests must exercise the production pandas/Python semantics. Date grouping should use integer factor codes or explicit group indices, not comparisons between values returned by version-sensitive datetime conversion APIs.

The detailed V90/V91 incident record is in `机器学习策略/experiments/postmortems/2026-07-14_v90_v91_semantic_replication.md`.
The V92 placebo incident record is in `机器学习策略/experiments/postmortems/2026-07-14_v92_placebo_integrity.md`.

## Factor Research Principles

Factor selection should be governed, not opportunistic.

Use factor evidence in this order:

1. Economic intuition and monthly horizon relevance.
2. Coverage and data stability in CSI800.
3. Long-window IC/RankIC and monotonicity.
4. Recent-window behavior, but shrunk toward long-term evidence.
5. Interaction with existing factor families.
6. Contribution after portfolio constraints and transaction costs.

Do not assume strong single factors always combine well.
Do not assume weak single factors are useless; interaction tests are valid, but they must be OOS and controlled.

## Rerank Direction

Rerank is only valuable if the direct model has a measurable recall gap.

Before rerank, check:

- Whether direct top30 contains enough realized top10/top5 names.
- Whether the top30 candidate pool has meaningful dispersion in realized returns.
- Whether rerank labels match the online candidate-generation process.
- Whether rerank improves top10 without hurting top20 and constrained profiles.

Rerank training must be offline/online aligned:

- Recall model is trained on the full eligible universe.
- Rerank model should be trained on historical candidates generated by the recall model, not on the full universe as if it saw the same distribution.

## Portfolio Construction Principles

Portfolio construction is part of the strategy, not an afterthought.

At minimum:

- Equal-weight topN is the baseline.
- Industry and board exposure should be reported.
- Min-lot and STAR board 200-share constraints must be handled in backtest code.
- Turnover should be measured and not ignored.
- A lower drawdown portfolio with materially lower return is not automatically better; evaluate risk-adjusted and path stability.

## Export and Backtest Rules

Exported model bundles must state:

- Objective.
- Research version.
- Train start and train end.
- Feature list.
- Fill values.
- Label definition.
- Model parameters and fixed iteration count.
- Stock count and portfolio profile assumptions if relevant.

Backtest code should load one model bundle and infer required features from the bundle.
It must not assume every model has price features, temporal features, or the same columns.

## Pre-Experiment Checklist

Before writing a new notebook:

1. Read this file.
2. State the single hypothesis being tested.
3. State the anchor comparison.
4. State the exact data source or rebuild method.
5. State train/test split dates.
6. State whether the experiment changes data, label, model, factor set, portfolio construction, or export format.
7. Confirm the result can be judged by monthly OOS outputs, not only aggregate return.
8. Confirm whether the experiment is trying to beat the GitHub practical anchor, the clean anchor, or only explain a research mechanism.
9. Confirm that offline model selection uses a JQ-like portfolio simulator, or explicitly mark the notebook as signal-only research.
10. If this is a replication or confirmation, define machine-checkable identity fields and stop-on-failure tolerances before adding challengers.

## Post-Experiment Checklist

After reading outputs:

1. Compare against the anchor first.
2. Check yearly and monthly stability.
3. Check Top10 and Top20 consistency.
4. Check drawdown, turnover, and exposure.
5. Identify whether improvement is stock selection, portfolio construction, timing, or hidden exposure.
6. Decide: keep, candidate, reject, or needs backtest.
7. Add the lesson to this document if it changes the mainline.

## Lessons Log

- 2026-06-15: V54 dynamic factor-pool replacement underperformed the fixed JQ baseline. The next robust-system direction should not be aggressive dynamic factor deletion. Prefer fixed baseline model bank plus controlled challengers and explicit monitoring/fallback.
- 2026-06-15: `fixed_jq_baseline_lgb` uses one fixed factor set across yearly train windows. Its strength appears to come from stable broad factors plus rolling retraining, not from changing the factor list.
- 2026-06-15: Annual underperformance should be treated as a normal property of factor/ML return prediction, not as an automatic failure. The research target is not "win every year"; it is stable positive expectancy with controlled drawdown, diagnosed exposure, and a predefined response when recent RankIC/monthly excess deteriorates.
- 2026-06-16: CSI800 vs CSI500 factor-stability test showed no decisive universe winner at the aggregate factor level. CSI500 had slightly higher mean factor ICIR and lower yearly IC volatility, while CSI800 segment-neutral had better top/bottom spread IR and hit rate. Value and quality factors were at least as stable in CSI800/segment-neutral CSI800, while growth/balance and some technical-volume factors leaned CSI500. Keep CSI800 as the main research universe and treat CSI500 as an attack sleeve or diagnostic sub-universe rather than replacing the mainline.
- 2026-06-24: The GitHub `zycoldness/quant-research` package should be treated as the current external benchmark set. Its core lines are v46 practical baseline, v410 clean single-model anchor, and v54 recall-rerank attack anchor. Later experiments that did not preserve the V4 hybrid-light feature stack or tested only against JQ-only baselines should not be interpreted as having invalidated the GitHub anchors.
- 2026-06-30: ETF V6 showed a concrete train/evaluation mismatch: the offline proxy target list mostly matched the JoinQuant `manual_trend_r2 top3` backtest, but the proxy used `feature_date close -> future close` while the backtest traded at the next scheduled 09:35 rebalance. The signal was mostly reproduced, but the performance estimate was not. This lesson now applies to all ML experiments: close-to-close proxy can screen signal direction, but model promotion requires JQ-like execution simulation or real JoinQuant backtest parity.
- 2026-07-14: V91 mislabeled a reconstructed event treatment as a V90 reproduction. It discarded the frozen V90 `evt_forecast_*` columns, refetched `STK_FIN_FORCAST`, changed parsing/coverage/selected features, and then used the changed monthly path to discuss V90 validity. The code ran correctly, but the experiment answered a different question. A replication now requires the Exact Replication Gate above; failed semantic parity cannot overturn the reference result.
- 2026-07-14: Corrected V92 exactly reproduced V90 and verified that the event model improved Top8 edge versus V46 by about `+0.635%/month`, but the matched placebo also improved and the real-minus-placebo intervals crossed zero. Event-specific alpha is therefore not confirmed. Keep V90 as a research candidate only; the next test must estimate a preregistered multi-permutation placebo distribution without changing features, model parameters, or portfolio rules.
- 2026-07-15: V96 Money2 is the strongest current Top8 candidate. It improved monthly Top8 edge by about `+0.599 percentage points` versus current Money and by about `+0.775 percentage points` versus full V46, while increasing Precision Lift and MAP@8. Because adjacent-cutoff stability remained slightly below the preregistered threshold, deploy it only with a fixed-cutoff V46/current Money shadow and explicit update monitoring.
- 2026-07-15: Final V98 rejected P2/P3/P4 and all-available external-information fusion for Money2 Top8. P2 improved RankIC but lowered Top8 edge; its sparse unlock fields also had a severe year-coverage break and disproportionate feature gain. The current P2 implementation is rejected, while true analyst revision and a clean pure-shareholder-change treatment remain untested.
