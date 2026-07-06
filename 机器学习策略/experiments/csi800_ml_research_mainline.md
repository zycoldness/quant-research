# CSI800 ML Research Mainline

Last updated: 2026-06-30

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

The anchor is not final. It is the benchmark set that every new experiment must beat or clarify. A new experiment that only beats the research-minimal baseline but loses to v46/v410 should be treated as a diagnostic, not a replacement candidate.

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
