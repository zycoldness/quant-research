# V99 Money2 稳定性蒸馏与相对资金流实验协议

## 研究问题

V99 合并验证两个预注册问题：

1. 删除 Money2 的单个信息块，或者对同族三个 seed 的分数做月内分位平均，能否在保留 Top8 Alpha 的同时改善换手和相邻训练截止日稳定性。
2. 行业同伴残差、行业资金强度和行业资金扩散度，能否提供 Money2 尚未包含的 Top8 增量。

V99 不导出模型，也不改变 V46/V96 的标签、训练窗口、LightGBM 参数、持仓数量或板块上限。

## 冻结基线

- 输入面板：V96 `v96_money2_enriched_panel.csv`。
- 精确参照：V96 `v96_core_monthly.csv`；缺失时允许完成计算，但所有自动晋级结论强制为 `KEEP_MONEY2`。
- 模型：LightGBM regression，L2，fixed120，`feature_fraction=1.0`。
- 训练：2019 起 expanding，`next_date <= cutoff`。
- OOS：2021-2025 五个 cutoff 后各 12 个月。
- Seeds：`42 / 2024 / 2026`。
- 组合：Top8，创业板最多3只，科创板最多2只。

## 实验一：稳定性蒸馏

以完整 Money2 为基线，只做四个 leave-one-block-out：

- 删除 current Money。
- 删除 flow quality。
- 删除 flow temporal。
- 删除 margin quality。

候选先满足：

- Top8 edge 至少保留基线的90%。
- Precision lift 相对基线下降不超过0.10。
- 任一年度 Top8 edge 相对基线下降不超过0.5个百分点/月。

通过核心门禁的候选按换手最低、Top8 edge最高的固定顺序选择一个进入相邻 cutoff 测试。只有 score correlation 或 Top8 overlap 至少一项明显改善，且另一项没有实质恶化，才允许晋级。

## 实验二：相对资金流

只增加8个紧凑特征：

- 个股5日、20日、加速度和60日资金排名相对行业中位数的残差。
- 行业20日资金强度、20日资金扩散度、资金加速度强度和加速度扩散度。

行业分类按每月 `feature_date` 查询，未知行业不参与行业统计。每个行业至少5只样本；行业覆盖率必须达到90%，每个新特征覆盖率必须达到80%。

相对资金候选必须满足：

- 月均 Top8 edge 相对 Money2 至少提高0.2个百分点。
- RankIC 下降不超过0.002。
- 至少3个年度 fold、2个 seed 为正。
- 三个月 block bootstrap 正增量概率至少65%。
- 相邻 cutoff 的 score correlation 和 Top8 overlap 不得显著弱于 Money2。
- Top8 行业 HHI 相对 Money2 增加不超过0.05，平均最大行业占比增加不超过12.5个百分点。

## 同族 Bagging

三个 seed 的原始分数先在每个月内转换为横截面分位，再取平均。这样避免不同 seed 的分数尺度影响融合。

Bagging 与三个单 seed 的同月均值配对比较，同时与 seed42 比较相邻 cutoff 稳定性。Bagging 只有在 Top8 edge 下降不超过0.1个百分点/月，并改善至少一个更新稳定性指标时才建议使用。

## 内存与恢复

- 每次只训练一个 variant/cutoff 的一个模型，模型预测完成后立即释放。
- 输入特征转为 `float32`。
- 核心和相邻 cutoff 任务分别写入 `checkpoints/core` 与 `checkpoints/update`。
- OOM 或中断后直接重跑 notebook，默认跳过签名一致的完成任务。
- 修改实验语义后必须更新 `EXPERIMENT_SIGNATURE` 或打开 `FORCE_RERUN_*`。

## 关键输出

- `v99_final_decision_table.csv`：三个方向的最终自动决策。
- `v99_core_summary_overall.csv`：单 seed 与 Bagging 的总体指标。
- `v99_core_summary_year.csv`：年度稳定性。
- `v99_distillation_core_decision.csv`：四个蒸馏候选。
- `v99_context_core_decision.csv`：三种相对资金组合。
- `v99_bagging_comparison.csv`：Bagging 对单 seed 均值。
- `v99_seed_top8_overlap_summary.csv`：三个单 seed 之间的持仓重合度。
- `v99_selection_industry_concentration_summary.csv`：Top8 行业 HHI 与最大行业占比。
- `v99_context_block_bootstrap.csv`：相对资金候选的 block bootstrap。
- `v99_adjacent_cutoff_stability_summary.csv`：相邻 cutoff 分数相关与 Top8 重合。
- `qa/v99_money2_baseline_reconciliation.csv`：V96 精确复现门禁。

任何自动 `PROMOTE` 都只代表进入模型导出和 JoinQuant 真实回测，不代表直接替换实盘模型。
