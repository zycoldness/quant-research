# CSI800 ML Mainline Family V3

Date: 2026-05-18

This folder archives the V3 CSI800 machine-learning strategy family experiment and the exported JoinQuant-loadable model bundles.

The experiment compared the current raw LGB baseline against broader framework families:

- raw v22a LGB baseline,
- hybrid-light LGB,
- factor-group recall,
- rank-industry LGB,
- recall + rerank LGB,
- fixed sleeve portfolios built from the better components.

The goal was not to finalize a live model. The goal was to decide which model families are still worth carrying forward.

## Main Findings

The V3 run did not find a standalone model that clearly replaces the baseline. It did find better portfolio constructions around the baseline.

Best rows by cumulative CSI800 excess:

| Strategy | Profile | Cum Excess CSI800 | Win Rate | Max Drawdown | Drop Top3 Excess | Note |
|---|---:|---:|---:|---:|---:|---|
| `raw_v22a_lgb_alpha_vs_alla` | top10 | 1.0065 | 0.7500 | -0.1345 | 0.2395 | Highest return anchor |
| `sleeve_60_base_top10_40_v3blend_top20` | sleeve | 0.9866 | 0.7500 | -0.0928 | 0.2626 | Best fixed sleeve |
| `raw_v22a_lgb_alpha_vs_alla` | industry_top10 | 0.9515 | 0.6250 | -0.1112 | 0.2225 | Strong constrained baseline |
| `v3_lgb_blend` | top20 | 0.9450 | 0.7500 | -0.0390 | 0.2413 | Best standalone risk-controlled challenger |
| `sleeve_50_base_top20_50_v3blend_top20` | sleeve | 0.9448 | 0.8125 | -0.0568 | 0.2629 | Balanced sleeve candidate |
| `raw_v22a_lgb_alpha_vs_alla` | top20 | 0.9397 | 0.7500 | -0.0851 | 0.2827 | More diversified baseline |
| `sleeve_60_base_top10_40_group_industry_top10` | sleeve | 0.9096 | 0.7500 | -0.0948 | 0.2757 | Diversified group sleeve |

## Decisions

Keep:

- `raw_v22a_lgb_alpha_vs_alla top10` as the return anchor.
- `raw_v22a_lgb_alpha_vs_alla top20` as the broader baseline.
- `v3_lgb_blend top20` as the risk-controlled challenger.
- `sleeve_60_base_top10_40_v3blend_top20` as the main sleeve candidate.
- `sleeve_50_base_top20_50_v3blend_top20` as the conservative sleeve candidate.
- `sleeve_60_base_top10_40_group_industry_top10` as a diversified sleeve candidate.

Do not promote yet:

- standalone `rank_industry_lgb`,
- standalone `recall_rerank_lgb`,
- standalone `manual_v3`,
- standalone `hybrid_light_lgb`.

## Why The Sleeve Matters

The best sleeve keeps nearly all baseline return while reducing path risk:

- baseline top10 excess: `1.0065`
- sleeve 60/40 excess: `0.9866`
- baseline max drawdown: `-13.45%`
- sleeve max drawdown: `-9.28%`
- baseline drop-top3 excess: `0.2395`
- sleeve drop-top3 excess: `0.2626`

This is a portfolio-construction improvement, not a new alpha model breakthrough.

## Files

See [outputs/README.md](outputs/README.md) for the exported CSV and PKL file descriptions.

The six PKL files are JoinQuant backtest model bundles. They are designed for the corresponding model-loading executor, not for direct manual inspection.

## Research Caveat

All reported metrics are research/backtest outputs. Promotion to paper trading still requires:

- JoinQuant online backtest loading the exported PKL bundle,
- execution checks for lot size, STAR-board minimum shares, cash insufficiency, limit-up/down, and T+1,
- cost and slippage sensitivity,
- additional out-of-sample and regime review.
