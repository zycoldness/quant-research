# V66 聚宽模型横向回测计划

## 回测脚本

使用：

`机器学习策略/backtests/jq_backtest_v66_top8_boardcap_model_sweep.py`

这个脚本基于原 sample 回测文件改造，核心口径固定为：

- universe: `000906.XSHG`
- rebalance: 每月第 1 个交易日，9:40 卖出，9:50 买入
- cost/slippage: `PriceRelatedSlippage(0.00246)` + 股票佣金税费
- portfolio rule: `top8_board_cap`
- stock_num: `8`
- board caps: 创业板最多 3 只，科创板最多 2 只
- no intraday limit-up sell
- no stop-loss / take-profit

## 先跑的模型

在聚宽上传下面几个 pkl 到 `test/` 目录，然后每次只改脚本顶部的 `MODEL_INDEX`。

| MODEL_INDEX | 模型 | 用途 |
|---:|---|---|
| 0 | `model_candidate_v61_full_2019_2023_start20190101_cutoff20231231_fixed120.pkl` | 2024 起 OOS，验证早期 cutoff |
| 1 | `model_candidate_v61_full_2019_2024_start20190101_cutoff20241231_fixed120.pkl` | 2025 起 OOS，V61 固定 anchor 主力候选 |
| 2 | `model_candidate_v61_full_2019_2025_start20190101_cutoff20251231_fixed120.pkl` | 2026 起 OOS，最新 expanding 固定模型 |
| 3 | `model_candidate_v46_lgb_direct_hybrid_l2_ff10_2019_2025q1_legacy_unsealed.pkl` | legacy anchor sanity check |

本地已解压前三个 full 模型到：

`机器学习策略/models/`

后续新模型统一从 `中证800_V67_纯模型训练导出器_可配置训练窗口.ipynb` 导出。V67 默认会保存到 `机器学习策略/models/`，同时生成 `v67_joinquant_upload_list.csv`；把 `joinquant_file` 那一列复制进回测脚本顶部的 `MODEL_CANDIDATES` 即可。

## 建议比较指标

每个回测结果至少记录：

- start/end
- total return
- benchmark return
- relative excess return
- max drawdown
- annual return
- Sharpe
- monthly win rate
- worst month
- trade count / turnover feel
- 最近 3 次调仓目标股日志

## 初始判断规则

- 如果 `MODEL_INDEX=1` 显著优于 `0/2`，说明当前固定 anchor 仍比最新 full retrain 稳。
- 如果 `MODEL_INDEX=2` 在 2026 后显著改善，但样本短，只能作为最新生产候选，不能替代完整 OOS 证据。
- 如果 `MODEL_INDEX=3` 接近或超过 V61 full，说明 V61 的 full-only 改造没有提供足够增益，需要回看特征删除/填充差异。
- 如果三者都在聚宽真实撮合下明显弱于离线 JQ-like，优先排查成交价、停牌/涨跌停过滤、文件特征适配，而不是先怀疑模型。
