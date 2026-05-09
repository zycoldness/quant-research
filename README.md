# 机器学习策略

## 结论先行

这个文件夹是当前中证800月调仓机器学习策略的核心观察包。我们不再把 V6.2 对齐重训线作为候选，因为特征审计已经确认旧 `train_csi800_factor_v40_data_enhancement.csv` 与 JoinQuant 回测/实盘现场计算的核心特征基本一致。

当前核心观察模型固定为三个：

- `model_candidate_v46_lgb_direct_hybrid_l2_ff10_2019_2025q1_legacy_unsealed.pkl`
- `model_candidate_v54_portfolio_oof_rerank_v410_q1.pkl`
- `model_candidate_v410_fixed_iter20_rolling5y_l2_ff10_2025q4.pkl`

其中 `v46 legacy_unsealed` 是当前主 baseline；`v54 recall-rerank` 是进攻型 cascade 观察线；`v410 fixed20 2025q4` 是较新训练截止、固定迭代的单模型观察线。

## 文件结构

```text
机器学习策略/
  README.md
  models/
    README.md
  manifests/
    core_models.csv
  notebooks/
    中证800_V4数据重建模块.ipynb
    中证800月调仓候选模型导出.ipynb
    中证800月调仓候选模型导出_2019_20251231新版训练.ipynb
    中证800召回精排V5.4组合构建实验.ipynb
  research_scripts/
    candidate_model_export.py
    rerank_portfolio_lab.py
  backtests/
    jq_backtest_v46_legacy_unsealed.py
    jq_backtest_v54_recall_rerank_top5.py
    jq_backtest_v410_fixed20_2025q4.py
```

说明：

- `notebooks/` 中的训练 notebook 是从此前稳定实验目录复制过来的原始 notebook，没有重写核心训练逻辑。
- `中证800_V4数据重建模块.ipynb` 是新增的数据重建入口，用于从 JoinQuant API 生成带日期区间的新数据文件，不覆盖旧 `train_csi800_factor_v40_data_enhancement.csv`。
- `中证800月调仓候选模型导出_2019_20251231新版训练.ipynb` 是新增的新版训练截止 notebook，用于把 V4 全部候选模型批量保存为 `2025-12-31` 训练截止版本。
- `research_scripts/` 是 notebook 对应导出的 `.py` 脚本，方便查代码和复现。
- `backtests/` 是 JoinQuant 回测脚本。除模型文件名和 V5.4 topN 配置外，保留既有回测逻辑。
- `models/` 只放模型文件说明。当前本地没有搜到三个 pkl，所以没有复制 pkl；请在 JoinQuant 回测时把对应 pkl 上传到策略文件同目录。

## 数据构成

核心观察模型的历史训练数据来自：

```text
train_csi800_factor_v40_data_enhancement.csv
```

该数据由 `candidate_model_export.py` / `中证800月调仓候选模型导出.ipynb` 的 V4 数据构造逻辑读取或重建。

新版数据重建默认输出：

```text
train_csi800_factor_v40_data_enhancement_20190101_20260531.csv
```

若需要从零复现数据，先在 JoinQuant 研究环境运行：

```text
notebooks/中证800_V4数据重建模块.ipynb
```

该 notebook 默认：

- `V4_DATA_START = 2019-01-01`
- `V4_DATA_END_FOR_LABEL = 2026-05-31`
- `FORCE_REBUILD_V4_DATA = True`
- 输出 `train_csi800_factor_v40_data_enhancement_20190101_20260531.csv`
- 输出 `train_csi800_factor_v40_data_enhancement_20190101_20260531_manifest.csv`

`V4_DATA_END_FOR_LABEL` 是数据层面的 label 覆盖截止日；如果之后要把标签延伸到更靠后的 2026 日期，只需要改这个参数后重建数据。文件名会自动用 `V4_DATA_START` 和 `V4_DATA_END_FOR_LABEL` 生成日期后缀。

基础设定：

- 股票池：中证800，`000906.XSHG`
- 调仓频率：月度
- `rebalance_date`：每月第一个交易日
- `feature_date`：`rebalance_date` 前一个交易日
- `next_date`：下一个月度调仓日
- Benchmark：中证800，`000906.XSHG`
- 上市天数过滤：至少 180 天
- ST/停牌过滤：数据构建时按 `feature_date` 过滤；回测时按当日 `get_current_data()` 再过滤

### 特征组成

基础因子来自 `jqfactor.get_factor_values`，核心字段包括：

```text
cash_flow_to_price_ratio
book_to_price_ratio
earnings_yield
sales_to_price_ratio
cash_earnings_to_price_ratio
earnings_to_price_ratio
roe_ttm
roa_ttm
gross_profit_ttm
operating_profit_to_total_profit
net_operate_cash_flow_to_total_liability
net_operating_cash_flow_coverage
adjusted_profit_to_total_profit
ACCA
growth
net_working_capital
operating_profit_per_share
net_operate_cash_flow_per_share
total_operating_revenue_per_share
super_quick_ratio
MLEV
debt_to_equity_ratio
debt_to_tangible_equity_ratio
momentum
Rank1M
sharpe_ratio_60
Variance20
liquidity
beta
ATR6
MFI14
DAVOL10
VOL10
VMACD
VOSC
Skewness20
Kurtosis20
```

V4 hybrid-light 额外特征主要是：

```text
liq_money_ratio_20_60
liq_paused_count_20
px_close_to_ma60
px_drawdown_60
ts_cash_flow_to_price_ratio_rank_mean_3m
ts_Rank1M_rank_chg_1m
```

特征审计结论：

- `train_csi800_factor_v40_data_enhancement.csv` 与策略1现场计算的核心特征高度一致。
- 基础因子和主要量价/流动性特征 rank correlation 基本为 `1.0`。
- 两个 temporal 特征平均 rank correlation 约 `0.995+`，最低月份也约 `0.98+`。
- 因此旧 CSV 与 JoinQuant 回测/实盘特征口径可以继续视为基本对齐。

## Label 构造

月度 label 为未来一个调仓周期的超额收益：

```text
alpha_1m = stock_return(rebalance_date -> next_date) - benchmark_return(rebalance_date -> next_date)
```

在旧 V4 数据构建中，收益计算使用日线前复权价格：

```python
get_price(
    stock_list,
    start_date=rebalance_date,
    end_date=next_date,
    frequency="daily",
    fields=["close"],
    skip_paused=True,
    fq="pre",
    panel=False,
)
```

股票收益使用区间内 close 矩阵近似：

```text
stock_ret = close[-1] / close[1] - 1
```

Benchmark 同口径计算，得到 `alpha_1m`。

注意：研究 label 是 close-to-close 的月度超额收益；JoinQuant 回测实际是 9:40 卖、9:50 买，并包含手续费、滑点、涨跌停过滤、最小手数限制。因此研究收益和回测收益不是完全同一东西，但旧模型已经在策略1回测中能复现较好结果。

## 模型 Roadmap

### 1. V46 Legacy Unsealed Baseline

模型文件：

```text
model_candidate_v46_lgb_direct_hybrid_l2_ff10_2019_2025q1_legacy_unsealed.pkl
```

回测脚本：

```text
backtests/jq_backtest_v46_legacy_unsealed.py
```

训练来源：

```text
notebooks/中证800月调仓候选模型导出.ipynb
research_scripts/candidate_model_export.py
```

训练设置：

- 数据文件：`train_csi800_factor_v40_data_enhancement.csv`
- 训练窗口：`2019-01-01 -> 2025-03-31`
- `label_end = 2025-03-31`
- `require_label_end_within_train = False`
- 也就是 legacy unsealed：允许最后训练月份的 label 跨过训练截止日
- 模型类型：LightGBM regression
- 组合规则：`base_score_z` 排序，先取 top30，再做行业分散 top10
- `overlay_mode = direct`
- `stock_num = 10`
- `industry_cap_ratio = 0.20`

观察定位：

- 当前主 baseline。
- 优点是历史和回测复现效果最好。
- 风险是 label boundary 不够干净，严格研究口径下可能偏乐观。
- 观察目标不是证明它最干净，而是作为实际表现锚点持续跟踪。

### 2. V54 Recall + Rerank Cascade

模型文件：

```text
model_candidate_v54_portfolio_oof_rerank_v410_q1.pkl
```

回测脚本：

```text
backtests/jq_backtest_v54_recall_rerank_top5.py
```

训练来源：

```text
notebooks/中证800召回精排V5.4组合构建实验.ipynb
research_scripts/rerank_portfolio_lab.py
```

训练设置：

- 数据文件：`train_csi800_factor_v40_data_enhancement.csv`
- 训练窗口：`2020-04-01 -> 2025-03-31`
- `label_end = 2025-03-31`
- 训练口径：label-safe
- 第一阶段：recall model，LightGBM regression，目标 `alpha_1m`
- 第二阶段：rerank model，OOF recall pool 上训练 binary classifier
- OOF fold 大致覆盖 2023、2024、2025Q1
- Pool relative features：在 recall pool 内做 rank、median diff、行业内 rank 等相对特征

当前回测观察配置：

```python
g.recall_top_k = 40
g.final_top_n = 5
g.selection_mode = "prob_topN"
g.use_baseline_recall_only = False
```

观察定位：

- 进攻型候选，top5 集中度更高。
- 优点是如果精排有效，收益弹性高。
- 风险是样本短、OOF pool 小、top5 集中度高，回撤和稳定性需要持续观察。
- 不作为当前主 baseline，但值得并行跟踪。

### 3. V410 Fixed Iter20 Rolling 5Y

模型文件：

```text
model_candidate_v410_fixed_iter20_rolling5y_l2_ff10_2025q4.pkl
```

回测脚本：

```text
backtests/jq_backtest_v410_fixed20_2025q4.py
```

训练来源：

```text
notebooks/中证800月调仓候选模型导出.ipynb
research_scripts/candidate_model_export.py
```

训练设置：

- 数据文件：`train_csi800_factor_v40_data_enhancement.csv`
- 训练窗口：`2021-01-01 -> 2025-12-31`
- `label_end = 2025-12-31`
- `require_label_end_within_train = True`
- 训练口径：label-safe
- 模型类型：LightGBM regression
- 迭代策略：`fixed_iter_20`
- 组合规则：`base_score_z` 排序，先取 top30，再做行业分散 top10
- `overlay_mode = direct`
- `stock_num = 10`
- `industry_cap_ratio = 0.20`

观察定位：

- 更接近“干净定版”的单模型候选。
- 优点是训练截止更近、固定迭代更克制、label-safe。
- 风险是近期表现可能不如 legacy unsealed，进攻性较弱。
- 用于判断是否能逐步从 legacy baseline 迁移到更稳健的干净口径。

## 回测策略统一约束

三个回测脚本统一遵守：

- Benchmark：`000906.XSHG`
- 月调仓：每月第一个交易日
- 9:05 记录持仓和昨日涨停
- 9:40 卖出不在目标池且非昨日涨停的持仓
- 9:50 买入目标股
- 默认不开启盘中炸板卖出：`g.enable_limit_up_sell = False`
- 不启用止损/止盈：`g.stop_loss_pct = None`，`g.take_profit_pct = None`
- 滑点：`PriceRelatedSlippage(0.00246)`
- 交易成本：

```python
open_tax=0
close_tax=0.001
open_commission=0.0003
close_commission=0.0003
min_commission=5
```

交易过滤：

- ST 过滤
- 停牌过滤
- 上市不足 180 天过滤
- 涨停不可买过滤
- 跌停不可卖/不可买过滤
- 最小手数/资金过滤
- 科创板最小交易数量 200 股，其它 100 股

## 如何复现

### 复现训练

可选但推荐先复现数据：

1. 打开 `notebooks/中证800_V4数据重建模块.ipynb`
2. 确认在 JoinQuant 研究环境运行
3. 保持默认 `FORCE_REBUILD_V4_DATA = True`
4. 运行后得到 `train_csi800_factor_v40_data_enhancement_20190101_20260531.csv`

单模型路线：

1. 打开 `notebooks/中证800月调仓候选模型导出.ipynb`
2. 确认同目录存在 `train_csi800_factor_v40_data_enhancement.csv`
3. 运行 notebook
4. 关注导出的候选：

```text
model_candidate_v46_lgb_direct_hybrid_l2_ff10_2019_2025q1_legacy_unsealed.pkl
model_candidate_v410_fixed_iter20_rolling5y_l2_ff10_2025q4.pkl
```

新版训练截止批量导出路线：

1. 打开 `notebooks/中证800月调仓候选模型导出_2019_20251231新版训练.ipynb`
2. 默认读取或自动重建 `train_csi800_factor_v40_data_enhancement_20190101_20260531.csv`
3. 默认训练截止为 `NEW_TRAIN_END = 2025-12-31`
4. 默认 `MODEL_LABEL_END = 2026-05-31`，可按需要改成其它 `2026-xx-xx`
5. `NEW_MODEL_SUFFIX` 会根据 `NEW_TRAIN_START`、`NEW_TRAIN_END`、`MODEL_LABEL_END` 自动生成
6. 默认只导出两条核心单模型候选：`v46` 和 `v410 fixed_iter20 rolling5y`，不导出 legacy V2.10，也不导出 V4 实验支线
7. 输出 pkl 会自动追加类似 `train20190101_20251231_labelend20260531` 的后缀，避免覆盖旧模型

说明：rolling5y 版本会保留 rolling5y 口径，训练起点为 `2021-01-01`；其它 expanding 版本训练起点为 `2019-01-01`。

Cascade 路线：

1. 打开 `notebooks/中证800召回精排V5.4组合构建实验.ipynb`
2. 确认同目录存在 `train_csi800_factor_v40_data_enhancement.csv`
3. 运行 notebook
4. 关注导出：

```text
model_candidate_v54_portfolio_oof_rerank_v410_q1.pkl
```

### 复现回测

在 JoinQuant 回测环境中，分别使用：

```text
backtests/jq_backtest_v46_legacy_unsealed.py
backtests/jq_backtest_v54_recall_rerank_top5.py
backtests/jq_backtest_v410_fixed20_2025q4.py
```

每个脚本需要对应 pkl 文件和脚本在同一策略文件目录，或者确保 `read_file(g.model_file)` 能读取到该 pkl。

推荐统一测试区间：

```text
2025-04-01 -> 当前最新可测日期
```

如果测试 `v410_fixed20_2025q4`，注意它训练到了 `2025-12-31`，所以它从 `2026-01-05` 以后才是严格 OOS。`2025-04-01 -> 2025-12-31` 对它不是严格 OOS，只适合观察，不适合和 Q1 模型公平比较。

## 当前决策规则

- 主 baseline：`v46 legacy_unsealed`
- 进攻观察：`v54 recall-rerank top5`
- 干净单模型观察：`v410 fixed20 rolling5y 2025q4`

短期不再推进：

- V6.2 aligned retrain
- 新 canonical builder 重训线
- 周调仓模型

原因：

- 旧 CSV 与现场特征已经通过 V6.3 审计，核心特征对齐良好。
- 旧 v46 模型用策略1可以复现历史好结果。
- 当前最重要的是持续观察三条确定路线，不再频繁引入新口径。

## 注意事项

- 不要覆盖旧 pkl，尤其不要用同名文件覆盖 `v46 legacy_unsealed`。
- 新模型必须在文件名里写清楚训练截止日期和 label policy。
- 回测比较时只换 `g.model_file` 或 V5.4 的 topN 配置，不要同时改交易规则。
- 若要比较严格 OOS，必须确认模型训练截止日期早于回测开始日期。
- 若出现研究结果和回测差异，优先检查模型文件、回测脚本配置、测试区间，再检查特征。
