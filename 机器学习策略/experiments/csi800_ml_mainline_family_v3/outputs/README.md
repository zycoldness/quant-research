# V3 Outputs

This folder contains the archived outputs from `csi800_ml_mainline_family_v3_full_families`.

## CSV Results

| File | Description |
|---|---|
| `mainline_family_v3_summary.csv` | Main strategy comparison table across strategy families and portfolio profiles. Start here. |
| `mainline_family_v3_monthly.csv` | Monthly portfolio returns, targets, exposures, and cumulative metrics. Use for attribution and backtest replay checks. |
| `mainline_family_v3_rank_ic.csv` | Monthly RankIC by strategy score. Use to separate signal quality from portfolio construction. |
| `mainline_family_v3_importance.csv` | LightGBM gain importance by trained model family. Use for feature diagnostics. |
| `mainline_family_v3_model_meta.csv` | Feature sets, target columns, and removed feature counts for each model family. |
| `mainline_family_v3_scores.csv` | Per-stock monthly scores for strategy analysis. Large file. |
| `model_csi800_lgb_v3_walkforward_manifest.csv` | Manifest for exported walk-forward model bundles. |
| `model_csi800_lgb_v3_walkforward_model_meta.csv` | Model-date-level metadata for the exported PKL model bank. |

## Exported Model Bundles

| File | Candidate | Intended Use |
|---|---|---|
| `model_csi800_lgb_v3_baseline_top10_walkforward.pkl` | `v3_baseline_top10` | Raw v22a LGB baseline, top 10. |
| `model_csi800_lgb_v3_baseline_top20_walkforward.pkl` | `v3_baseline_top20` | Raw v22a LGB baseline, top 20. |
| `model_csi800_lgb_v3_v3blend_top20_walkforward.pkl` | `v3_blend_top20` | Blend score, top 20. |
| `model_csi800_lgb_v3_sleeve_60_base10_40_v3blend20_walkforward.pkl` | `v3_sleeve_60_base10_40_v3blend20` | Main 60/40 sleeve candidate. |
| `model_csi800_lgb_v3_sleeve_50_base20_50_v3blend20_walkforward.pkl` | `v3_sleeve_50_base20_50_v3blend20` | Conservative 50/50 sleeve candidate. |
| `model_csi800_lgb_v3_sleeve_60_base10_40_groupind10_walkforward.pkl` | `v3_sleeve_60_base10_40_groupind10` | Diversified group-industry sleeve candidate. |

## Bundle Notes

The sleeve bundles use `score_col = sleeve_weighted_components` in the manifest. That value is a portfolio-composition marker, not an actual score column. The JoinQuant executor must read `sleeve_components` and generate each component score separately.

For example, the main sleeve bundle combines:

- 60% `baseline_lgb_score`, top 10,
- 40% `v3_lgb_blend_score`, top 20.

## Reproduction Notes

The model bundles are walk-forward exports with 16 model dates for the 2025-01 to 2026-04 evaluation window.

These files were archived from:

`/Users/youzou/Downloads/csi800_ml_mainline_family_v3_outputs.zip`
