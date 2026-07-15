import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = ROOT / "机器学习策略" / "notebooks"
V96_PATH = NOTEBOOK_DIR / "csi800_v96_money_quality_information_experiment_jq.ipynb"
OUTPUT_PATHS = [
    NOTEBOOK_DIR / "csi800_v99_money2_stability_context_jq.ipynb",
    NOTEBOOK_DIR / "中证800_V99_Money2稳定性蒸馏与相对资金流实验.ipynb",
]


def source_lines(source):
    source = textwrap.dedent(source).strip("\n") + "\n"
    return source.splitlines(True)


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(source)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(source),
    }


with V96_PATH.open("r", encoding="utf-8") as handle:
    v96 = json.load(handle)
utility_source = "".join(v96["cells"][12]["source"])
required_utility_symbols = [
    "def select_v46_features", "def select_incremental_features",
    "def train_predict_many_seeds", "def evaluate_score_panel",
]
for symbol in required_utility_symbols:
    if symbol not in utility_source:
        raise RuntimeError("V96 utility cell missing %s" % symbol)


cells = [
    markdown(r'''
    # 中证800 V99 Money2 稳定性蒸馏与相对资金流实验

    本 notebook 合并回答两个问题：

    1. Money2 能否通过预注册的特征块 leave-one-out 与三 seed 同族 Bagging，在保留 Top8 Alpha 的同时改善换手和相邻训练截止日稳定性。
    2. 在不增加外部数据源的前提下，行业同伴残差、行业资金强度和行业资金扩散度能否为 Money2 提供新的 Top8 增量。

    冻结协议：V96 Money2 面板、expanding、label-safe、raw `alpha_1m`、LightGBM L2、fixed120、`feature_fraction=1.0`、Top8 与既有板块上限。

    本实验不导出生产模型。所有候选必须先逐月复现 V96 Money2 基线，再经过年度 fold、三 seed、block bootstrap 和相邻 cutoff 门禁。
    '''),
    markdown("## 0. 导入、进度条与兼容层"),
    code(r'''
    from jqdata import *
    import builtins as _bi
    import datetime as _dt
    import gc
    import json
    from pathlib import Path

    import lightgbm as lgb
    import numpy as np
    import pandas as pd

    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None


    def progress_iter(iterable, total=None, desc="progress", leave=True):
        if tqdm is not None:
            return tqdm(iterable, total=total, desc=desc, leave=leave)
        def _gen():
            every = _bi.max(1, int((total or 100) / 20))
            for i, item in enumerate(iterable, 1):
                if i == 1 or i % every == 0 or (total is not None and i == total):
                    print("%s %s%s" % (desc, i, "/%s" % total if total else ""))
                yield item
        return _gen()


    def display_df(df, n=20):
        try:
            display(df.head(n))
        except Exception:
            print(df.head(n).to_string(index=False))
    '''),
    markdown("## 1. 冻结配置、运行开关与验收门槛"),
    code(r'''
    PROJECT_DIR = Path.cwd()
    PRODUCT_VERSION = "v99"
    RUN_NAME = "money2_stability_distillation_relative_flow"
    RUN_TIMESTAMP = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    EXPERIMENT_SIGNATURE = "v99_money2_stability_context_contract_1"

    OUT_DIR = PROJECT_DIR / "csi800_ml_v99_money2_stability_context_outputs"
    FIG_DIR = OUT_DIR / "figures"
    QA_DIR = OUT_DIR / "qa"
    CHECKPOINT_DIR = OUT_DIR / "checkpoints"
    CORE_CHECKPOINT_DIR = CHECKPOINT_DIR / "core"
    UPDATE_CHECKPOINT_DIR = CHECKPOINT_DIR / "update"
    for path in [OUT_DIR, FIG_DIR, QA_DIR, CHECKPOINT_DIR, CORE_CHECKPOINT_DIR, UPDATE_CHECKPOINT_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    PANEL_PATH_OVERRIDE = None
    PANEL_CANDIDATES = [
        Path("csi800_ml_v96_money_quality_outputs/v96_money2_enriched_panel.csv"),
        Path("../csi800_ml_v96_money_quality_outputs/v96_money2_enriched_panel.csv"),
        Path("v96_money2_enriched_panel.csv"),
    ]

    STOCK_COL = "stock"
    DATE_COL = "rebalance_date"
    NEXT_DATE_COL = "next_date"
    FEATURE_DATE_COL = "feature_date"
    TARGET_COL = "alpha_1m"
    INDUSTRY_COL = "v99_industry_code"

    PRED_TOP_K = 8
    TRUE_TOP_N = 20
    BOARD_CAPS = {"chinext": 3, "star": 2}
    FIXED_ITER = 120
    FEATURE_FRACTION = 1.0
    NUM_THREADS = 4
    CORE_CUTOFFS = ["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"]
    CORE_SEEDS = [42, 2024, 2026]
    UPDATE_SEED = 42
    MIN_TRAIN_MONTHS = 30
    CORR_THRESHOLD_V46 = 0.70
    CORR_THRESHOLD_INCREMENTAL = 0.95
    MIN_INCREMENTAL_TRAIN_COVERAGE = 0.01
    MIN_CONTEXT_COVERAGE = 0.80
    MIN_INDUSTRY_COVERAGE = 0.90
    MIN_INDUSTRY_MEMBERS = 5
    BASELINE_RANK_IC_ATOL = 1e-10
    BASELINE_TOP8_EDGE_ATOL = 1e-7

    BASE_PARAMS = {
        "objective": "regression", "metric": "l2", "boosting_type": "gbdt",
        "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200,
        "feature_fraction": 1.0, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l1": 0.1, "lambda_l2": 0.3, "verbose": -1,
    }

    UPDATE_PAIRS = [
        {"pair_id": "2021_to_2022_eval2023", "old_cutoff": "2021-12-31", "new_cutoff": "2022-12-31", "eval_start": "2023-01-01", "eval_end": "2023-12-31"},
        {"pair_id": "2022_to_2023_eval2024", "old_cutoff": "2022-12-31", "new_cutoff": "2023-12-31", "eval_start": "2024-01-01", "eval_end": "2024-12-31"},
        {"pair_id": "2023_to_2024_eval2025", "old_cutoff": "2023-12-31", "new_cutoff": "2024-12-31", "eval_start": "2025-01-01", "eval_end": "2025-12-31"},
        {"pair_id": "2024_to_2025_eval2026", "old_cutoff": "2024-12-31", "new_cutoff": "2025-12-31", "eval_start": "2026-01-01", "eval_end": "2026-12-31"},
    ]

    BOOTSTRAP_SAMPLES = 2000
    BOOTSTRAP_BLOCK_MONTHS = 3
    BOOTSTRAP_SEED = 9901

    # Stability distillation gates.
    ACCEPT_MIN_EDGE_RETENTION = 0.90
    ACCEPT_MIN_PRECISION_LIFT_DELTA = -0.10
    ACCEPT_MAX_WORST_YEAR_EDGE_LOSS = -0.005
    ACCEPT_MIN_UPDATE_CORR_IMPROVEMENT = 0.01
    ACCEPT_MIN_UPDATE_OVERLAP_IMPROVEMENT = 0.025

    # Relative-flow incremental-alpha gates.
    ACCEPT_MIN_CONTEXT_DELTA_EDGE = 0.002
    ACCEPT_MIN_CONTEXT_DELTA_RANK_IC = -0.002
    ACCEPT_MIN_POSITIVE_YEARS = 3
    ACCEPT_MIN_POSITIVE_SEEDS = 2
    ACCEPT_MIN_BOOTSTRAP_PROBABILITY = 0.65
    ACCEPT_MAX_UPDATE_CORR_LOSS = -0.02
    ACCEPT_MAX_UPDATE_OVERLAP_LOSS = -0.025
    ACCEPT_MAX_INDUSTRY_HHI_INCREASE = 0.05
    ACCEPT_MAX_INDUSTRY_SHARE_INCREASE = 0.125

    # Same-family bagging must preserve alpha and improve at least one update-stability metric.
    ACCEPT_MAX_BAGGING_EDGE_LOSS = -0.001

    FORCE_REFETCH_INDUSTRY = False
    FORCE_RERUN_CORE = False
    FORCE_RERUN_UPDATE = False
    RUN_UPDATE_STABILITY = True
    SMOKE_MODE = False
    SMOKE_MAX_MONTHS = 2
    if SMOKE_MODE:
        CORE_CUTOFFS = CORE_CUTOFFS[-1:]
        CORE_SEEDS = CORE_SEEDS[:1]
        UPDATE_PAIRS = UPDATE_PAIRS[-1:]
        BOOTSTRAP_SAMPLES = 100

    print("OUT_DIR:", OUT_DIR)
    print("Core protocol: expanding + label-safe + fixed120 + feature_fraction=1.0")
    '''),
    markdown("## 2. Money2 特征块与预注册候选"),
    code(r'''
    def unique_keep_order(cols):
        seen = set()
        out = []
        for col in cols:
            if col not in seen:
                out.append(col)
                seen.add(col)
        return out


    BASE_FACTOR_COLS = [
        "cash_flow_to_price_ratio", "book_to_price_ratio", "earnings_yield", "sales_to_price_ratio",
        "cash_earnings_to_price_ratio", "earnings_to_price_ratio", "roe_ttm", "roa_ttm",
        "gross_profit_ttm", "operating_profit_to_total_profit", "net_operate_cash_flow_to_total_liability",
        "net_operating_cash_flow_coverage", "adjusted_profit_to_total_profit", "ACCA", "growth",
        "net_working_capital", "operating_profit_per_share", "net_operate_cash_flow_per_share",
        "total_operating_revenue_per_share", "super_quick_ratio", "MLEV", "debt_to_equity_ratio",
        "debt_to_tangible_equity_ratio", "momentum", "Rank1M", "sharpe_ratio_60", "Variance20",
        "liquidity", "beta", "ATR6", "MFI14", "DAVOL10", "VOL10", "VMACD", "VOSC",
        "Skewness20", "Kurtosis20",
    ]
    HYBRID_LIGHT_EXTRA_COLS = [
        "liq_money_ratio_20_60", "liq_paused_count_20", "px_close_to_ma60", "px_drawdown_60",
        "ts_cash_flow_to_price_ratio_rank_mean_3m", "ts_Rank1M_rank_chg_1m",
    ]
    FULL_V46_COLS = unique_keep_order(BASE_FACTOR_COLS + HYBRID_LIGHT_EXTRA_COLS)

    CURRENT_MONEY_COLS = [
        "mf_main_mean5_rank", "mf_main_mean20_rank", "mf_main_accel_rank",
        "mf_main_positive_ratio20", "mf_xl_mean20_rank", "mf_big_small_spread20_rank",
        "mf_main_return_divergence", "mf_observation_days20",
        "mt_fin_value_chg5_rank", "mt_fin_value_chg20_rank", "mt_fin_buy_to_balance20_rank",
        "mt_observation_days20", "mt_is_marginable",
    ]
    FLOW_QUALITY_COLS = [
        "mq_main_std20_rank", "mq_main_ir20_rank", "mq_main_positive_ratio5_rank",
        "mq_main_sign_flip20_rank", "mq_main_slope20_rank", "mq_flow_return_corr20_rank",
        "mq_main_abs_concentration5_20_rank", "mq_big_flow_share20_rank",
        "mq_signed_price_efficiency20_rank",
    ]
    FLOW_TEMPORAL_COLS = [
        "mq_main_mean60_rank", "mq_main_accel5_60_rank", "mq_main20_rank_chg1m",
        "mq_main20_rank_mean3m", "mq_xl20_rank_chg1m",
    ]
    MARGIN_QUALITY_COLS = [
        "mq_fin_balance_vs_mean60_rank", "mq_fin_balance_z60_rank",
        "mq_fin_buy_accel5_20_rank", "mq_fin_balance_up_ratio20_rank",
        "mq_fin20_rank_chg1m",
    ]
    MONEY2_COLS = unique_keep_order(FLOW_QUALITY_COLS + FLOW_TEMPORAL_COLS + MARGIN_QUALITY_COLS)
    MONEY2_BASELINE_COLS = unique_keep_order(CURRENT_MONEY_COLS + MONEY2_COLS)

    PEER_RESIDUAL_COLS = [
        "ctx_main5_industry_residual", "ctx_main20_industry_residual",
        "ctx_accel_industry_residual", "ctx_main60_industry_residual",
    ]
    PEER_CONTEXT_COLS = [
        "ctx_industry_main20_strength", "ctx_industry_main20_breadth",
        "ctx_industry_accel_strength", "ctx_industry_accel_breadth",
    ]
    CONTEXT_COLS = unique_keep_order(PEER_RESIDUAL_COLS + PEER_CONTEXT_COLS)

    BLOCKS = [
        ("current_money", CURRENT_MONEY_COLS),
        ("flow_quality", FLOW_QUALITY_COLS),
        ("flow_temporal", FLOW_TEMPORAL_COLS),
        ("margin_quality", MARGIN_QUALITY_COLS),
    ]

    VARIANTS = [{
        "variant": "money2_baseline", "experiment": "baseline",
        "baseline_cols": MONEY2_BASELINE_COLS, "context_cols": [],
    }]
    for block_name, block_cols in BLOCKS:
        VARIANTS.append({
            "variant": "money2_drop_" + block_name, "experiment": "distillation",
            "baseline_cols": [c for c in MONEY2_BASELINE_COLS if c not in block_cols],
            "context_cols": [],
        })
    VARIANTS.extend([
        {"variant": "money2_plus_peer_residual", "experiment": "relative_flow",
         "baseline_cols": MONEY2_BASELINE_COLS, "context_cols": PEER_RESIDUAL_COLS},
        {"variant": "money2_plus_peer_context", "experiment": "relative_flow",
         "baseline_cols": MONEY2_BASELINE_COLS, "context_cols": PEER_CONTEXT_COLS},
        {"variant": "money2_plus_relative_all", "experiment": "relative_flow",
         "baseline_cols": MONEY2_BASELINE_COLS, "context_cols": CONTEXT_COLS},
    ])

    feature_contract_rows = []
    for group_name, cols in [("v46", FULL_V46_COLS)] + BLOCKS + [
        ("peer_residual", PEER_RESIDUAL_COLS), ("peer_context", PEER_CONTEXT_COLS),
    ]:
        feature_contract_rows.append({"group": group_name, "count": len(cols), "features": ",".join(cols)})
    feature_contract_df = pd.DataFrame(feature_contract_rows)
    variant_manifest_df = pd.DataFrame([{
        "variant": spec["variant"], "experiment": spec["experiment"],
        "baseline_feature_count": len(spec["baseline_cols"]),
        "context_feature_count": len(spec["context_cols"]),
        "baseline_features": ",".join(spec["baseline_cols"]),
        "context_features": ",".join(spec["context_cols"]),
    } for spec in VARIANTS])
    feature_contract_df.to_csv(OUT_DIR / "v99_feature_contract.csv", index=False)
    variant_manifest_df.to_csv(OUT_DIR / "v99_variant_manifest.csv", index=False)
    display_df(variant_manifest_df, 20)
    print("core fits:", len(CORE_CUTOFFS) * len(VARIANTS) * len(CORE_SEEDS))
    '''),
    markdown("## 3. 加载并审计冻结 V96 Money2 面板"),
    code(r'''
    def resolve_panel_path():
        candidates = []
        if PANEL_PATH_OVERRIDE:
            candidates.append(Path(str(PANEL_PATH_OVERRIDE)))
        candidates.extend(PANEL_CANDIDATES)
        for root in [PROJECT_DIR, PROJECT_DIR.parent]:
            try:
                candidates.extend(list(root.glob("**/v96_money2_enriched_panel.csv")))
            except Exception:
                pass
        seen = set()
        valid = []
        required = unique_keep_order(
            [STOCK_COL, DATE_COL, NEXT_DATE_COL, FEATURE_DATE_COL, TARGET_COL] +
            FULL_V46_COLS + MONEY2_BASELINE_COLS
        )
        for candidate in candidates:
            path = Path(str(candidate))
            if str(path) in seen or not path.exists() or not path.is_file():
                continue
            seen.add(str(path))
            try:
                header = pd.read_csv(str(path), nrows=0)
                if _bi.all(col in header.columns for col in required):
                    is_override = bool(PANEL_PATH_OVERRIDE) and str(path) == str(Path(str(PANEL_PATH_OVERRIDE)))
                    valid.append((path, path.stat().st_mtime, int(is_override)))
            except Exception:
                pass
        if len(valid) == 0:
            raise IOError("V96 Money2 panel not found; upload or generate v96_money2_enriched_panel.csv first")
        valid = _bi.sorted(valid, key=lambda item: (item[2], item[1]), reverse=True)
        return valid[0][0]


    PANEL_PATH = resolve_panel_path()
    required_cols = unique_keep_order(
        [STOCK_COL, DATE_COL, NEXT_DATE_COL, FEATURE_DATE_COL, TARGET_COL] +
        FULL_V46_COLS + MONEY2_BASELINE_COLS
    )
    df_all = pd.read_csv(str(PANEL_PATH), usecols=required_cols, low_memory=False)
    for col in [DATE_COL, NEXT_DATE_COL, FEATURE_DATE_COL]:
        df_all[col] = pd.to_datetime(df_all[col], errors="coerce").dt.normalize()
    df_all[STOCK_COL] = df_all[STOCK_COL].astype(str)
    numeric_cols = [TARGET_COL] + FULL_V46_COLS + MONEY2_BASELINE_COLS
    for col in progress_iter(numeric_cols, total=len(numeric_cols), desc="cast V96 panel"):
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)
    df_all = df_all.dropna(subset=[STOCK_COL, DATE_COL, NEXT_DATE_COL, FEATURE_DATE_COL, TARGET_COL])
    df_all = df_all.sort_values([DATE_COL, STOCK_COL]).drop_duplicates([DATE_COL, STOCK_COL], keep="last").reset_index(drop=True)

    audit_df = pd.DataFrame([
        {"check": "rows", "value": len(df_all), "status": "ok" if len(df_all) > 10000 else "fail"},
        {"check": "months", "value": df_all[DATE_COL].nunique(), "status": "ok" if df_all[DATE_COL].nunique() >= 60 else "fail"},
        {"check": "duplicate_keys", "value": int(df_all.duplicated([DATE_COL, STOCK_COL]).sum()),
         "status": "ok" if int(df_all.duplicated([DATE_COL, STOCK_COL]).sum()) == 0 else "fail"},
        {"check": "feature_date_after_rebalance", "value": int((df_all[FEATURE_DATE_COL] >= df_all[DATE_COL]).sum()),
         "status": "ok" if int((df_all[FEATURE_DATE_COL] >= df_all[DATE_COL]).sum()) == 0 else "fail"},
        {"check": "next_date_not_after_rebalance", "value": int((df_all[NEXT_DATE_COL] <= df_all[DATE_COL]).sum()),
         "status": "ok" if int((df_all[NEXT_DATE_COL] <= df_all[DATE_COL]).sum()) == 0 else "fail"},
    ])
    audit_df.to_csv(QA_DIR / "v99_base_data_audit.csv", index=False)
    display_df(audit_df, 20)
    if len(audit_df[audit_df["status"] == "fail"]):
        raise ValueError("V99 frozen V96 panel audit failed")
    print("PANEL_PATH:", PANEL_PATH, "shape:", df_all.shape)
    '''),
    markdown("## 4. 构造 PIT 行业快照与相对资金流特征"),
    code(r'''
    INDUSTRY_CACHE_PATH = QA_DIR / "v99_industry_snapshot.csv"


    def industry_code_from_payload(payload):
        if not isinstance(payload, dict):
            return "UNKNOWN"
        for key in ["sw_l1", "jq_l1", "zjw"]:
            sub = payload.get(key)
            if isinstance(sub, dict):
                value = sub.get("industry_code") or sub.get("industry_name")
                if value:
                    return str(value)
        return "UNKNOWN"


    cached_industry = pd.DataFrame(columns=[STOCK_COL, DATE_COL, FEATURE_DATE_COL, INDUSTRY_COL])
    if INDUSTRY_CACHE_PATH.exists() and not FORCE_REFETCH_INDUSTRY:
        try:
            cached_industry = pd.read_csv(str(INDUSTRY_CACHE_PATH), low_memory=False)
            for col in [DATE_COL, FEATURE_DATE_COL]:
                cached_industry[col] = pd.to_datetime(cached_industry[col], errors="coerce").dt.normalize()
            cached_industry[STOCK_COL] = cached_industry[STOCK_COL].astype(str)
        except Exception:
            cached_industry = pd.DataFrame(columns=[STOCK_COL, DATE_COL, FEATURE_DATE_COL, INDUSTRY_COL])

    snapshot_rows = []
    cached_keys = set()
    if len(cached_industry):
        for row in cached_industry[[STOCK_COL, DATE_COL]].itertuples(index=False, name=None):
            cached_keys.add((str(row[0]), pd.Timestamp(row[1])))
        snapshot_rows.extend(cached_industry.to_dict("records"))

    month_meta = df_all[[DATE_COL, FEATURE_DATE_COL]].drop_duplicates().sort_values(DATE_COL)
    for _, meta in progress_iter(month_meta.iterrows(), total=len(month_meta), desc="build PIT industry snapshots"):
        rebalance_date = pd.Timestamp(meta[DATE_COL]).normalize()
        feature_date = pd.Timestamp(meta[FEATURE_DATE_COL]).normalize()
        month_stocks = df_all.loc[df_all[DATE_COL] == rebalance_date, STOCK_COL].astype(str).tolist()
        missing = [stock for stock in month_stocks if (stock, rebalance_date) not in cached_keys]
        if len(missing) == 0:
            continue
        try:
            payload = get_industry(missing, date=feature_date.date())
        except Exception as exc:
            print("industry fetch failed", rebalance_date.date(), str(exc))
            payload = {}
        for stock in missing:
            snapshot_rows.append({
                STOCK_COL: stock, DATE_COL: rebalance_date, FEATURE_DATE_COL: feature_date,
                INDUSTRY_COL: industry_code_from_payload(payload.get(stock, {})),
            })

    industry_df = pd.DataFrame(snapshot_rows)
    industry_df = industry_df.drop_duplicates([STOCK_COL, DATE_COL], keep="last")
    industry_df.to_csv(INDUSTRY_CACHE_PATH, index=False)
    df_all = df_all.merge(industry_df[[STOCK_COL, DATE_COL, INDUSTRY_COL]], on=[STOCK_COL, DATE_COL], how="left")
    df_all[INDUSTRY_COL] = df_all[INDUSTRY_COL].fillna("UNKNOWN").astype(str)

    industry_coverage_rows = []
    for date_value, part in df_all.groupby(DATE_COL):
        coverage = float((part[INDUSTRY_COL] != "UNKNOWN").mean())
        industry_coverage_rows.append({"rebalance_date": pd.Timestamp(date_value), "industry_coverage": coverage})
    industry_coverage_df = pd.DataFrame(industry_coverage_rows).sort_values("rebalance_date")
    industry_coverage_df.to_csv(QA_DIR / "v99_industry_coverage.csv", index=False)
    if len(industry_coverage_df[industry_coverage_df["industry_coverage"] < MIN_INDUSTRY_COVERAGE]):
        display_df(industry_coverage_df[industry_coverage_df["industry_coverage"] < MIN_INDUSTRY_COVERAGE], 100)
        raise ValueError("V99 industry coverage gate failed")


    def numeric_series_local(frame, col):
        if col not in frame.columns:
            return pd.Series(np.nan, index=frame.index)
        return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


    context_frames = []
    grouped_months = list(df_all.groupby(DATE_COL))
    for date_value, part in progress_iter(grouped_months, total=len(grouped_months), desc="build relative money context"):
        out = part[[STOCK_COL, DATE_COL, FEATURE_DATE_COL, INDUSTRY_COL]].copy()
        known = part[INDUSTRY_COL] != "UNKNOWN"
        known_part = part.loc[known].copy()
        group_sizes = known_part.groupby(INDUSTRY_COL)[STOCK_COL].count()
        valid_industries = set(group_sizes[group_sizes >= MIN_INDUSTRY_MEMBERS].index.tolist())
        valid_mask = known & part[INDUSTRY_COL].isin(valid_industries)

        residual_specs = [
            ("mf_main_mean5_rank", "ctx_main5_industry_residual"),
            ("mf_main_mean20_rank", "ctx_main20_industry_residual"),
            ("mf_main_accel_rank", "ctx_accel_industry_residual"),
            ("mq_main_mean60_rank", "ctx_main60_industry_residual"),
        ]
        for raw_col, output_col in residual_specs:
            values = numeric_series_local(part, raw_col)
            work = pd.DataFrame({INDUSTRY_COL: part[INDUSTRY_COL], "value": values})
            medians = work.loc[valid_mask].groupby(INDUSTRY_COL)["value"].median()
            out[output_col] = values - part[INDUSTRY_COL].map(medians)

        context_specs = [
            ("mf_main_mean20_rank", "ctx_industry_main20_strength", "ctx_industry_main20_breadth"),
            ("mf_main_accel_rank", "ctx_industry_accel_strength", "ctx_industry_accel_breadth"),
        ]
        for raw_col, strength_col, breadth_col in context_specs:
            values = numeric_series_local(part, raw_col)
            market_mean = float(values.mean()) if len(values.dropna()) else np.nan
            market_median = float(values.median()) if len(values.dropna()) else np.nan
            work = pd.DataFrame({INDUSTRY_COL: part[INDUSTRY_COL], "value": values})
            industry_mean = work.loc[valid_mask].groupby(INDUSTRY_COL)["value"].mean()
            flags = pd.Series(np.nan, index=part.index, dtype=float)
            flags.loc[values.notnull()] = (values.loc[values.notnull()] >= market_median).astype(float)
            flag_work = pd.DataFrame({INDUSTRY_COL: part[INDUSTRY_COL], "flag": flags})
            industry_breadth = flag_work.loc[valid_mask].groupby(INDUSTRY_COL)["flag"].mean()
            out[strength_col] = part[INDUSTRY_COL].map(industry_mean) - market_mean
            out[breadth_col] = part[INDUSTRY_COL].map(industry_breadth)

        context_frames.append(out)

    context_df = pd.concat(context_frames, ignore_index=True)
    context_df = context_df.drop_duplicates([STOCK_COL, DATE_COL], keep="last")
    for col in CONTEXT_COLS:
        context_df[col] = pd.to_numeric(context_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)
    df_all = df_all.merge(
        context_df[[STOCK_COL, DATE_COL] + CONTEXT_COLS], on=[STOCK_COL, DATE_COL], how="left",
    )

    context_coverage_rows = []
    for col in CONTEXT_COLS:
        context_coverage_rows.append({
            "feature": col, "coverage": float(df_all[col].notnull().mean()),
            "unique_values": int(df_all[col].nunique(dropna=True)),
        })
    context_coverage_df = pd.DataFrame(context_coverage_rows)
    context_coverage_df["status"] = np.where(
        (context_coverage_df["coverage"] >= MIN_CONTEXT_COVERAGE) &
        (context_coverage_df["unique_values"] > 1), "ok", "fail",
    )
    context_coverage_df.to_csv(QA_DIR / "v99_context_feature_coverage.csv", index=False)
    context_df.to_csv(OUT_DIR / "v99_relative_money_context_panel.csv", index=False)
    display_df(context_coverage_df, 20)
    if len(context_coverage_df[context_coverage_df["status"] == "fail"]):
        raise ValueError("V99 context feature coverage gate failed")
    del context_frames, grouped_months, context_df, cached_industry, snapshot_rows
    gc.collect()
    '''),
    markdown("## 5. 冻结 V96 训练、排序与 Top8 指标函数"),
    code(utility_source),
    markdown("## 6. V99 增量特征选择、Bagging 与断点续跑"),
    code(r'''
    def select_additive_context_features(train_df, candidate_cols, frozen_incremental):
        selected, removed, coverage_map = select_incremental_features(train_df, candidate_cols)
        if len(selected) == 0 or len(frozen_incremental) == 0:
            return selected, removed, coverage_map
        corr_cols = unique_keep_order(frozen_incremental + selected)
        corr = train_df[corr_cols].corr()
        keep = []
        for col in selected:
            too_close = False
            for base_col in frozen_incremental:
                value = corr.loc[col, base_col]
                if not pd.isnull(value) and abs(value) > CORR_THRESHOLD_INCREMENTAL:
                    too_close = True
                    break
            if too_close:
                removed.append(col)
            else:
                keep.append(col)
        return unique_keep_order(keep), unique_keep_order(removed), coverage_map


    def variant_features_v99(train_df, spec):
        base_features, removed_base = select_v46_features(train_df)
        baseline_inc, removed_baseline, baseline_coverage = select_incremental_features(
            train_df, spec["baseline_cols"],
        )
        context_inc, removed_context, context_coverage = select_additive_context_features(
            train_df, spec["context_cols"], baseline_inc,
        )
        features = unique_keep_order(base_features + baseline_inc + context_inc)
        return {
            "features": features, "removed_base": removed_base,
            "baseline_incremental": baseline_inc, "removed_baseline": removed_baseline,
            "context_incremental": context_inc, "removed_context": removed_context,
            "baseline_coverage": baseline_coverage, "context_coverage": context_coverage,
        }


    def add_bagged_rank_score(panel, score_cols, output_col="score_bagged"):
        result = pd.Series(np.nan, index=panel.index, dtype=float)
        groups = list(panel.groupby(DATE_COL).groups.items())
        for _, indices in groups:
            rank_arrays = []
            for score_col in score_cols:
                values = pd.to_numeric(panel.loc[indices, score_col], errors="coerce")
                valid_count = int(values.notnull().sum())
                if valid_count == 0:
                    ranks = pd.Series(0.5, index=indices)
                else:
                    ranks = values.rank(method="average") / float(valid_count)
                    ranks = ranks.fillna(0.5)
                rank_arrays.append(ranks.values.astype(float))
            result.loc[indices] = np.mean(np.vstack(rank_arrays), axis=0)
        panel[output_col] = result.astype(np.float32)
        return panel


    def checkpoint_paths(folder, task_id):
        return {
            "metrics": folder / (task_id + "_metrics.csv"),
            "selected": folder / (task_id + "_selected.csv"),
            "meta": folder / (task_id + "_meta.csv"),
            "importance": folder / (task_id + "_importance.csv"),
        }


    def checkpoint_is_valid(paths):
        for path in paths.values():
            if not path.exists() or not path.is_file():
                return False
        try:
            sample = pd.read_csv(str(paths["metrics"]), nrows=1)
            return len(sample) == 1 and sample["experiment_signature"].iloc[0] == EXPERIMENT_SIGNATURE
        except Exception:
            return False


    def load_checkpoint(paths):
        return tuple(pd.read_csv(str(paths[key]), low_memory=False) for key in ["metrics", "selected", "meta", "importance"])


    def save_checkpoint(paths, metrics_df, selected_df, meta_df, importance_df):
        for frame in [metrics_df, selected_df, meta_df, importance_df]:
            frame["experiment_signature"] = EXPERIMENT_SIGNATURE
        metrics_df.to_csv(paths["metrics"], index=False)
        selected_df.to_csv(paths["selected"], index=False)
        meta_df.to_csv(paths["meta"], index=False)
        importance_df.to_csv(paths["importance"], index=False)
    '''),
    markdown("## 7. 五个年度 fold、三 seed 与同族 Bagging"),
    code(r'''
    core_monthly_parts = []
    selected_parts = []
    model_meta_parts = []
    importance_parts = []
    core_tasks = [(cutoff, spec) for cutoff in CORE_CUTOFFS for spec in VARIANTS]

    for cutoff, spec in progress_iter(core_tasks, total=len(core_tasks), desc="V99 core OOS"):
        task_id = "%s_%s" % (str(cutoff).replace("-", ""), spec["variant"])
        paths = checkpoint_paths(CORE_CHECKPOINT_DIR, task_id)
        if checkpoint_is_valid(paths) and not FORCE_RERUN_CORE:
            metrics_df, selected_df, meta_df, importance_df = load_checkpoint(paths)
            core_monthly_parts.append(metrics_df)
            selected_parts.append(selected_df)
            model_meta_parts.append(meta_df)
            importance_parts.append(importance_df)
            continue

        train_df, test_df, train_start = build_train_test(cutoff, "expanding")
        if train_df[DATE_COL].nunique() < MIN_TRAIN_MONTHS or len(test_df) == 0:
            continue
        selection = variant_features_v99(train_df, spec)
        feature_cols = selection["features"]
        meta = {
            "family": spec["variant"], "variant": spec["variant"],
            "experiment": spec["experiment"], "cutoff": str(pd.Timestamp(cutoff).date()),
            "policy": "expanding", "train_start": str(pd.Timestamp(train_start).date()),
            "removed_base": ",".join(selection["removed_base"]),
            "removed_baseline": ",".join(selection["removed_baseline"]),
            "selected_context": ",".join(selection["context_incremental"]),
            "removed_context": ",".join(selection["removed_context"]),
        }
        score_map, fit_meta, fit_importance, _ = train_predict_many_seeds(
            train_df, test_df, feature_cols, CORE_SEEDS, FEATURE_FRACTION, meta,
        )
        panel = test_df[[STOCK_COL, DATE_COL, TARGET_COL]].copy()
        metric_rows = []
        selected_rows = []
        score_cols = []
        for seed in CORE_SEEDS:
            score_col = "score_%s" % seed
            panel[score_col] = score_map[int(seed)]
            score_cols.append(score_col)
            metrics, single_selected = evaluate_score_panel(
                panel, {spec["variant"]: score_col},
                {"cutoff": str(pd.Timestamp(cutoff).date()), "policy": "expanding",
                 "seed": int(seed), "model_mode": "single", "experiment": spec["experiment"],
                 "feature_fraction": FEATURE_FRACTION},
                store_selected=True,
            )
            metric_rows.extend(metrics)
            selected_rows.extend(single_selected)
        panel = add_bagged_rank_score(panel, score_cols)
        metrics, selected = evaluate_score_panel(
            panel, {spec["variant"]: "score_bagged"},
            {"cutoff": str(pd.Timestamp(cutoff).date()), "policy": "expanding",
             "seed": "bagged", "model_mode": "bagged", "experiment": spec["experiment"],
             "feature_fraction": FEATURE_FRACTION},
            store_selected=True,
        )
        metric_rows.extend(metrics)
        selected_rows.extend(selected)

        for row in fit_importance:
            row["variant"] = spec["variant"]
            row["feature_origin"] = (
                "relative_flow" if row["feature"] in CONTEXT_COLS else
                ("money2" if row["feature"] in MONEY2_COLS else
                 ("current_money" if row["feature"] in CURRENT_MONEY_COLS else "v46"))
            )

        metrics_df = pd.DataFrame(metric_rows)
        selected_df = pd.DataFrame(selected_rows)
        meta_df = pd.DataFrame(fit_meta)
        importance_df = pd.DataFrame(fit_importance)
        save_checkpoint(paths, metrics_df, selected_df, meta_df, importance_df)
        core_monthly_parts.append(metrics_df)
        selected_parts.append(selected_df)
        model_meta_parts.append(meta_df)
        importance_parts.append(importance_df)
        del train_df, test_df, panel, score_map, metrics_df, selected_df, meta_df, importance_df
        gc.collect()

    core_monthly_df = pd.concat(core_monthly_parts, ignore_index=True)
    selected_df = pd.concat(selected_parts, ignore_index=True)
    model_meta_df = pd.concat(model_meta_parts, ignore_index=True)
    feature_importance_df = pd.concat(importance_parts, ignore_index=True)
    core_monthly_df[DATE_COL] = pd.to_datetime(core_monthly_df[DATE_COL], errors="coerce")
    core_monthly_df.to_csv(OUT_DIR / "v99_core_monthly.csv", index=False)
    selected_df.to_csv(OUT_DIR / "v99_selected_top8.csv", index=False)
    model_meta_df.to_csv(OUT_DIR / "v99_model_meta.csv", index=False)
    feature_importance_df.to_csv(OUT_DIR / "v99_feature_importance.csv", index=False)
    '''),
    markdown("## 8. V96 精确基线复现门禁"),
    code(r'''
    reference_candidates = [
        PANEL_PATH.parent / "v96_core_monthly.csv",
        PROJECT_DIR / "csi800_ml_v96_money_quality_outputs" / "v96_core_monthly.csv",
        PROJECT_DIR / "v96_core_monthly.csv",
    ]
    for root in [PROJECT_DIR, PROJECT_DIR.parent]:
        try:
            reference_candidates.extend(list(root.glob("**/v96_core_monthly.csv")))
        except Exception:
            pass
    reference_path = None
    for candidate in reference_candidates:
        if candidate.exists() and candidate.is_file():
            reference_path = candidate
            break

    baseline_reconciliation_df = pd.DataFrame([{
        "status": "reference_not_found", "reference_path": "", "expected_rows": 0, "matched_rows": 0,
        "rank_ic_max_abs_delta": np.nan, "top8_edge_max_abs_delta": np.nan,
    }])
    if reference_path is not None:
        reference = pd.read_csv(str(reference_path), low_memory=False)
        reference = reference[reference["variant"] == "money2_all"].copy()
        reference[DATE_COL] = pd.to_datetime(reference[DATE_COL], errors="coerce")
        reference["cutoff"] = reference["cutoff"].astype(str)
        reference["seed"] = pd.to_numeric(reference["seed"], errors="coerce")
        baseline = core_monthly_df[
            (core_monthly_df["variant"] == "money2_baseline") &
            (core_monthly_df["model_mode"] == "single")
        ].copy()
        baseline["cutoff"] = baseline["cutoff"].astype(str)
        baseline["seed"] = pd.to_numeric(baseline["seed"], errors="coerce")
        keys = ["cutoff", "seed", DATE_COL]
        matched = baseline.merge(
            reference[keys + ["rank_ic", "top8_edge"]], on=keys, how="inner",
            suffixes=("_v99", "_v96"),
        )
        rank_delta = float((matched["rank_ic_v99"] - matched["rank_ic_v96"]).abs().max()) if len(matched) else np.nan
        edge_delta = float((matched["top8_edge_v99"] - matched["top8_edge_v96"]).abs().max()) if len(matched) else np.nan
        status = "ok" if (
            len(matched) == len(baseline) and
            rank_delta <= BASELINE_RANK_IC_ATOL and edge_delta <= BASELINE_TOP8_EDGE_ATOL
        ) else "fail"
        baseline_reconciliation_df = pd.DataFrame([{
            "status": status, "reference_path": str(reference_path),
            "expected_rows": len(baseline), "matched_rows": len(matched),
            "rank_ic_max_abs_delta": rank_delta, "top8_edge_max_abs_delta": edge_delta,
            "rank_ic_atol": BASELINE_RANK_IC_ATOL, "top8_edge_atol": BASELINE_TOP8_EDGE_ATOL,
        }])
        if status != "ok":
            raise RuntimeError("V99 Money2 baseline does not reproduce V96: %s" % baseline_reconciliation_df.to_dict("records"))
    baseline_reconciliation_df.to_csv(QA_DIR / "v99_money2_baseline_reconciliation.csv", index=False)
    display_df(baseline_reconciliation_df, 10)
    '''),
    markdown("## 9. 核心汇总、蒸馏候选与相对资金候选选择"),
    code(r'''
    METRIC_COLS = [
        "rank_ic", "auc_true_top20", "precision_at8_true20", "precision_lift",
        "recall_at8_true20", "map_at8_true20", "ndcg_at8_true20",
        "top8_alpha", "top8_edge", "turnover",
    ]


    def summarize_metrics(df, group_cols):
        rows = []
        groups = list(df.groupby(group_cols))
        for keys, part in progress_iter(groups, total=len(groups), desc="summarize", leave=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict((group_cols[i], keys[i]) for i in range(len(group_cols)))
            row["months"] = len(part)
            for col in METRIC_COLS:
                values = pd.to_numeric(part[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                row[col + "_mean"] = float(values.mean()) if len(values) else np.nan
                row[col + "_std"] = float(values.std()) if len(values) > 1 else np.nan
            row["rank_ic_positive_rate"] = float((pd.to_numeric(part["rank_ic"], errors="coerce") > 0).mean())
            row["top8_edge_positive_rate"] = float((pd.to_numeric(part["top8_edge"], errors="coerce") > 0).mean())
            rows.append(row)
        return pd.DataFrame(rows)


    def moving_block_bootstrap(values, samples, block_size, seed):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return np.asarray([])
        rng = np.random.RandomState(seed)
        n = len(values)
        block_size = _bi.max(1, _bi.min(int(block_size), n))
        starts = np.arange(0, n - block_size + 1)
        blocks = int(np.ceil(n / float(block_size)))
        draws = []
        for _ in range(int(samples)):
            sample = []
            for _block in range(blocks):
                start = int(rng.choice(starts))
                sample.extend(values[start:start + block_size])
            draws.append(float(np.mean(sample[:n])))
        return np.asarray(draws)


    overall_summary_df = summarize_metrics(core_monthly_df, ["variant", "experiment", "model_mode"])
    yearly_summary_df = summarize_metrics(core_monthly_df, ["variant", "model_mode", "year"])
    seed_summary_df = summarize_metrics(
        core_monthly_df[core_monthly_df["model_mode"] == "single"], ["variant", "seed"],
    )
    overall_summary_df.to_csv(OUT_DIR / "v99_core_summary_overall.csv", index=False)
    yearly_summary_df.to_csv(OUT_DIR / "v99_core_summary_year.csv", index=False)
    seed_summary_df.to_csv(OUT_DIR / "v99_core_summary_seed.csv", index=False)

    selected_df[DATE_COL] = pd.to_datetime(selected_df[DATE_COL], errors="coerce")
    selected_with_industry = selected_df.merge(
        df_all[[STOCK_COL, DATE_COL, INDUSTRY_COL]].drop_duplicates([STOCK_COL, DATE_COL]),
        on=[STOCK_COL, DATE_COL], how="left",
    )
    industry_concentration_rows = []
    bagged_selected = selected_with_industry[selected_with_industry["model_mode"] == "bagged"]
    for keys, part in bagged_selected.groupby(["variant", "cutoff", DATE_COL]):
        counts = part[INDUSTRY_COL].fillna("UNKNOWN").value_counts()
        shares = counts / float(counts.sum())
        industry_concentration_rows.append({
            "variant": keys[0], "cutoff": keys[1], DATE_COL: pd.Timestamp(keys[2]),
            "industry_count": int(len(counts)), "industry_hhi": float((shares * shares).sum()),
            "max_industry_share": float(shares.max()),
            "unknown_industry_share": float(shares.get("UNKNOWN", 0.0)),
        })
    industry_concentration_df = pd.DataFrame(industry_concentration_rows)
    industry_concentration_summary_rows = []
    for variant, part in industry_concentration_df.groupby("variant"):
        industry_concentration_summary_rows.append({
            "variant": variant, "months": len(part),
            "industry_count_mean": float(part["industry_count"].mean()),
            "industry_hhi_mean": float(part["industry_hhi"].mean()),
            "max_industry_share_mean": float(part["max_industry_share"].mean()),
            "unknown_industry_share_mean": float(part["unknown_industry_share"].mean()),
        })
    industry_concentration_summary_df = pd.DataFrame(industry_concentration_summary_rows)
    industry_concentration_df.to_csv(OUT_DIR / "v99_selection_industry_concentration_monthly.csv", index=False)
    industry_concentration_summary_df.to_csv(OUT_DIR / "v99_selection_industry_concentration_summary.csv", index=False)

    seed_overlap_rows = []
    single_selected = selected_with_industry[selected_with_industry["model_mode"] == "single"]
    for keys, part in single_selected.groupby(["variant", "cutoff", DATE_COL]):
        seed_sets = []
        for seed, seed_part in part.groupby("seed"):
            seed_sets.append((seed, set(seed_part[STOCK_COL].tolist())))
        for i in range(len(seed_sets)):
            for j in range(i + 1, len(seed_sets)):
                left_seed, left_set = seed_sets[i]
                right_seed, right_set = seed_sets[j]
                seed_overlap_rows.append({
                    "variant": keys[0], "cutoff": keys[1], DATE_COL: pd.Timestamp(keys[2]),
                    "seed_a": left_seed, "seed_b": right_seed,
                    "top8_overlap": len(left_set.intersection(right_set)) / float(PRED_TOP_K),
                })
    seed_overlap_df = pd.DataFrame(seed_overlap_rows)
    seed_overlap_summary_rows = []
    for variant, part in seed_overlap_df.groupby("variant"):
        seed_overlap_summary_rows.append({
            "variant": variant, "month_pairs": len(part),
            "seed_top8_overlap_mean": float(part["top8_overlap"].mean()),
            "seed_top8_overlap_median": float(part["top8_overlap"].median()),
        })
    seed_overlap_summary_df = pd.DataFrame(seed_overlap_summary_rows)
    seed_overlap_df.to_csv(OUT_DIR / "v99_seed_top8_overlap_monthly.csv", index=False)
    seed_overlap_summary_df.to_csv(OUT_DIR / "v99_seed_top8_overlap_summary.csv", index=False)

    bagged_summary = overall_summary_df[overall_summary_df["model_mode"] == "bagged"].copy()
    baseline_row = bagged_summary[bagged_summary["variant"] == "money2_baseline"].iloc[0]
    baseline_edge = float(baseline_row["top8_edge_mean"])
    baseline_precision_lift = float(baseline_row["precision_lift_mean"])
    baseline_turnover = float(baseline_row["turnover_mean"])

    distillation_rows = []
    for _, row in bagged_summary[bagged_summary["experiment"] == "distillation"].iterrows():
        variant = row["variant"]
        year_delta = yearly_summary_df[
            (yearly_summary_df["variant"] == variant) & (yearly_summary_df["model_mode"] == "bagged")
        ][["year", "top8_edge_mean"]].merge(
            yearly_summary_df[
                (yearly_summary_df["variant"] == "money2_baseline") &
                (yearly_summary_df["model_mode"] == "bagged")
            ][["year", "top8_edge_mean"]], on="year", suffixes=("_candidate", "_baseline"),
        )
        year_delta["delta"] = year_delta["top8_edge_mean_candidate"] - year_delta["top8_edge_mean_baseline"]
        edge_retention = float(row["top8_edge_mean"]) / baseline_edge if baseline_edge > 0 else np.nan
        precision_delta = float(row["precision_lift_mean"]) - baseline_precision_lift
        turnover_delta = float(row["turnover_mean"]) - baseline_turnover
        worst_year_delta = float(year_delta["delta"].min()) if len(year_delta) else np.nan
        eligible = (
            not pd.isnull(edge_retention) and edge_retention >= ACCEPT_MIN_EDGE_RETENTION and
            precision_delta >= ACCEPT_MIN_PRECISION_LIFT_DELTA and
            not pd.isnull(worst_year_delta) and worst_year_delta >= ACCEPT_MAX_WORST_YEAR_EDGE_LOSS
        )
        distillation_rows.append({
            "variant": variant, "edge_retention": edge_retention,
            "delta_top8_edge": float(row["top8_edge_mean"]) - baseline_edge,
            "delta_precision_lift": precision_delta, "delta_turnover": turnover_delta,
            "worst_year_delta_top8_edge": worst_year_delta, "core_eligible": bool(eligible),
        })
    distillation_table_df = pd.DataFrame(distillation_rows)
    eligible_distillation = distillation_table_df[distillation_table_df["core_eligible"]].copy()
    if len(eligible_distillation):
        eligible_distillation = eligible_distillation.sort_values(
            ["delta_turnover", "delta_top8_edge"], ascending=[True, False],
        )
        selected_distillation_variant = str(eligible_distillation.iloc[0]["variant"])
    else:
        selected_distillation_variant = str(
            distillation_table_df.sort_values("edge_retention", ascending=False).iloc[0]["variant"]
        )
    distillation_table_df["selected_for_update_test"] = (
        distillation_table_df["variant"] == selected_distillation_variant
    )
    distillation_table_df.to_csv(OUT_DIR / "v99_distillation_core_decision.csv", index=False)

    context_table_df = bagged_summary[bagged_summary["experiment"] == "relative_flow"].copy()
    context_table_df["delta_top8_edge"] = context_table_df["top8_edge_mean"] - baseline_edge
    context_table_df["delta_rank_ic"] = context_table_df["rank_ic_mean"] - float(baseline_row["rank_ic_mean"])
    context_table_df["delta_precision_lift"] = context_table_df["precision_lift_mean"] - baseline_precision_lift
    context_table_df = context_table_df.sort_values("top8_edge_mean", ascending=False)
    selected_context_variant = str(context_table_df.iloc[0]["variant"])
    context_table_df["selected_for_update_test"] = context_table_df["variant"] == selected_context_variant
    context_table_df.to_csv(OUT_DIR / "v99_context_core_decision.csv", index=False)

    display_df(distillation_table_df.sort_values("edge_retention", ascending=False), 20)
    display_df(context_table_df, 20)
    print("selected distillation:", selected_distillation_variant)
    print("selected relative-flow:", selected_context_variant)
    '''),
    markdown("## 10. 相邻训练截止日稳定性：baseline、蒸馏候选与相对资金候选"),
    code(r'''
    spec_by_variant = dict((spec["variant"], spec) for spec in VARIANTS)
    update_variant_names = unique_keep_order([
        "money2_baseline", selected_distillation_variant, selected_context_variant,
    ])


    def train_score_window(cutoff, spec, eval_start, eval_end):
        train_df, test_df, train_start = build_train_test(
            cutoff, "expanding", eval_start=eval_start, eval_end=eval_end,
        )
        if len(train_df) == 0 or len(test_df) == 0:
            return pd.DataFrame()
        selection = variant_features_v99(train_df, spec)
        score_map, _, _, _ = train_predict_many_seeds(
            train_df, test_df, selection["features"], CORE_SEEDS, FEATURE_FRACTION,
            {"family": spec["variant"], "variant": spec["variant"],
             "cutoff": str(pd.Timestamp(cutoff).date()), "policy": "expanding",
             "train_start": str(pd.Timestamp(train_start).date())},
        )
        panel = test_df[[STOCK_COL, DATE_COL, TARGET_COL]].copy()
        score_cols = []
        for seed in CORE_SEEDS:
            score_col = "score_%s" % seed
            panel[score_col] = score_map[int(seed)]
            score_cols.append(score_col)
        panel = add_bagged_rank_score(panel, score_cols)
        keep_cols = [STOCK_COL, DATE_COL, TARGET_COL, "score_%s" % UPDATE_SEED, "score_bagged"]
        result = panel[keep_cols].copy()
        del train_df, test_df, panel, score_map
        gc.collect()
        return result


    update_parts = []
    if RUN_UPDATE_STABILITY:
        update_tasks = [(pair, variant) for pair in UPDATE_PAIRS for variant in update_variant_names]
        for pair, variant in progress_iter(update_tasks, total=len(update_tasks), desc="V99 adjacent cutoff"):
            task_id = "%s_%s" % (pair["pair_id"], variant)
            path = UPDATE_CHECKPOINT_DIR / (task_id + "_metrics.csv")
            if path.exists() and not FORCE_RERUN_UPDATE:
                try:
                    cached = pd.read_csv(str(path), low_memory=False)
                    if len(cached) and cached["experiment_signature"].iloc[0] == EXPERIMENT_SIGNATURE:
                        update_parts.append(cached)
                        continue
                except Exception:
                    pass
            spec = spec_by_variant[variant]
            old_panel = train_score_window(pair["old_cutoff"], spec, pair["eval_start"], pair["eval_end"])
            new_panel = train_score_window(pair["new_cutoff"], spec, pair["eval_start"], pair["eval_end"])
            rows = []
            if len(old_panel) and len(new_panel):
                merged = old_panel.merge(
                    new_panel, on=[STOCK_COL, DATE_COL], suffixes=("_old", "_new"),
                )
                mode_cols = [
                    ("seed42", "score_%s_old" % UPDATE_SEED, "score_%s_new" % UPDATE_SEED),
                    ("bagged", "score_bagged_old", "score_bagged_new"),
                ]
                for date_value, month in merged.groupby(DATE_COL):
                    universe_alpha = float(month[TARGET_COL + "_old"].mean())
                    for model_mode, old_col, new_col in mode_cols:
                        old_indices = select_top8_board_capped(month, old_col)
                        new_indices = select_top8_board_capped(month, new_col)
                        old_selected = month.loc[old_indices]
                        new_selected = month.loc[new_indices]
                        old_set = set(old_selected[STOCK_COL])
                        new_set = set(new_selected[STOCK_COL])
                        rows.append({
                            "pair_id": pair["pair_id"], "variant": variant,
                            "model_mode": model_mode, DATE_COL: pd.Timestamp(date_value),
                            "score_rank_corr": safe_rank_ic(month[old_col], month[new_col]),
                            "top8_overlap": len(old_set.intersection(new_set)) / float(PRED_TOP_K),
                            "old_top8_edge": float(old_selected[TARGET_COL + "_old"].mean()) - universe_alpha,
                            "new_top8_edge": float(new_selected[TARGET_COL + "_new"].mean()) - universe_alpha,
                        })
                del merged
            task_df = pd.DataFrame(rows)
            task_df["experiment_signature"] = EXPERIMENT_SIGNATURE
            task_df.to_csv(path, index=False)
            update_parts.append(task_df)
            del old_panel, new_panel, task_df
            gc.collect()

    update_stability_df = pd.concat(update_parts, ignore_index=True) if len(update_parts) else pd.DataFrame()
    if len(update_stability_df):
        update_stability_df[DATE_COL] = pd.to_datetime(update_stability_df[DATE_COL], errors="coerce")
    update_stability_df.to_csv(OUT_DIR / "v99_adjacent_cutoff_stability_monthly.csv", index=False)

    update_summary_rows = []
    if len(update_stability_df):
        for keys, part in update_stability_df.groupby(["variant", "model_mode"]):
            update_summary_rows.append({
                "variant": keys[0], "model_mode": keys[1], "months": len(part),
                "score_rank_corr_mean": float(part["score_rank_corr"].mean()),
                "top8_overlap_mean": float(part["top8_overlap"].mean()),
                "old_top8_edge_mean": float(part["old_top8_edge"].mean()),
                "new_top8_edge_mean": float(part["new_top8_edge"].mean()),
            })
    update_summary_df = pd.DataFrame(update_summary_rows, columns=[
        "variant", "model_mode", "months", "score_rank_corr_mean", "top8_overlap_mean",
        "old_top8_edge_mean", "new_top8_edge_mean",
    ])
    update_summary_df.to_csv(OUT_DIR / "v99_adjacent_cutoff_stability_summary.csv", index=False)
    display_df(update_summary_df, 20)
    '''),
    markdown("## 11. 配对 Bootstrap、Bagging 比较与最终决策"),
    code(r'''
    pair_keys = ["cutoff", DATE_COL]
    metric_pair_cols = ["rank_ic", "precision_lift", "map_at8_true20", "top8_edge", "turnover"]

    bagged_core = core_monthly_df[core_monthly_df["model_mode"] == "bagged"].copy()
    single_core = core_monthly_df[core_monthly_df["model_mode"] == "single"].copy()
    single_month_avg = single_core.groupby(["variant"] + pair_keys)[metric_pair_cols].mean().reset_index()
    bagging_paired_df = bagged_core[["variant"] + pair_keys + metric_pair_cols].merge(
        single_month_avg, on=["variant"] + pair_keys, suffixes=("_bagged", "_single_mean"),
    )
    for col in metric_pair_cols:
        bagging_paired_df["delta_" + col] = bagging_paired_df[col + "_bagged"] - bagging_paired_df[col + "_single_mean"]
    bagging_comparison_rows = []
    for variant, part in bagging_paired_df.groupby("variant"):
        row = {"variant": variant, "months": len(part)}
        for col in metric_pair_cols:
            row["delta_" + col + "_mean"] = float(part["delta_" + col].mean())
        bagging_comparison_rows.append(row)
    bagging_comparison_df = pd.DataFrame(bagging_comparison_rows)
    bagging_paired_df.to_csv(OUT_DIR / "v99_bagging_paired_monthly.csv", index=False)
    bagging_comparison_df.to_csv(OUT_DIR / "v99_bagging_comparison.csv", index=False)

    baseline_bagged = bagged_core[bagged_core["variant"] == "money2_baseline"][
        pair_keys + ["rank_ic", "top8_edge"]
    ].rename(columns={"rank_ic": "rank_ic_baseline", "top8_edge": "top8_edge_baseline"})
    selected_context_monthly = bagged_core[bagged_core["variant"] == selected_context_variant].merge(
        baseline_bagged, on=pair_keys, how="inner",
    )
    selected_context_monthly["delta_rank_ic"] = selected_context_monthly["rank_ic"] - selected_context_monthly["rank_ic_baseline"]
    selected_context_monthly["delta_top8_edge"] = selected_context_monthly["top8_edge"] - selected_context_monthly["top8_edge_baseline"]
    selected_context_monthly.to_csv(OUT_DIR / "v99_paired_context_vs_money2.csv", index=False)
    context_draws = moving_block_bootstrap(
        selected_context_monthly.sort_values(DATE_COL)["delta_top8_edge"].values,
        BOOTSTRAP_SAMPLES, BOOTSTRAP_BLOCK_MONTHS, BOOTSTRAP_SEED,
    )
    context_bootstrap_df = pd.DataFrame([{
        "variant": selected_context_variant, "months": len(selected_context_monthly),
        "mean_delta_top8_edge": float(selected_context_monthly["delta_top8_edge"].mean()),
        "ci_low": float(np.percentile(context_draws, 2.5)),
        "ci_high": float(np.percentile(context_draws, 97.5)),
        "probability_delta_positive": float((context_draws > 0).mean()),
    }])
    context_bootstrap_df.to_csv(OUT_DIR / "v99_context_block_bootstrap.csv", index=False)

    def update_value(variant, mode, col):
        rows = update_summary_df[
            (update_summary_df["variant"] == variant) & (update_summary_df["model_mode"] == mode)
        ]
        return float(rows[col].iloc[0]) if len(rows) else np.nan


    baseline_update_corr = update_value("money2_baseline", "bagged", "score_rank_corr_mean")
    baseline_update_overlap = update_value("money2_baseline", "bagged", "top8_overlap_mean")
    baseline_gate_ok = bool(
        len(baseline_reconciliation_df) and
        baseline_reconciliation_df["status"].iloc[0] == "ok"
    )

    selected_distill_row = distillation_table_df[
        distillation_table_df["variant"] == selected_distillation_variant
    ].iloc[0]
    distill_update_corr = update_value(selected_distillation_variant, "bagged", "score_rank_corr_mean")
    distill_update_overlap = update_value(selected_distillation_variant, "bagged", "top8_overlap_mean")
    distill_corr_delta = distill_update_corr - baseline_update_corr
    distill_overlap_delta = distill_update_overlap - baseline_update_overlap
    distill_stability_improved = (
        (distill_corr_delta >= ACCEPT_MIN_UPDATE_CORR_IMPROVEMENT) or
        (distill_overlap_delta >= ACCEPT_MIN_UPDATE_OVERLAP_IMPROVEMENT)
    )
    distill_no_material_stability_harm = (
        distill_corr_delta >= ACCEPT_MAX_UPDATE_CORR_LOSS and
        distill_overlap_delta >= ACCEPT_MAX_UPDATE_OVERLAP_LOSS
    )
    distill_promote = bool(
        baseline_gate_ok and selected_distill_row["core_eligible"] and distill_stability_improved and
        distill_no_material_stability_harm
    )

    context_year_delta = yearly_summary_df[
        (yearly_summary_df["variant"] == selected_context_variant) &
        (yearly_summary_df["model_mode"] == "bagged")
    ][["year", "top8_edge_mean"]].merge(
        yearly_summary_df[
            (yearly_summary_df["variant"] == "money2_baseline") &
            (yearly_summary_df["model_mode"] == "bagged")
        ][["year", "top8_edge_mean"]], on="year", suffixes=("_context", "_baseline"),
    )
    context_year_delta["delta"] = context_year_delta["top8_edge_mean_context"] - context_year_delta["top8_edge_mean_baseline"]
    context_seed_delta = seed_summary_df[seed_summary_df["variant"] == selected_context_variant][
        ["seed", "top8_edge_mean"]
    ].merge(
        seed_summary_df[seed_summary_df["variant"] == "money2_baseline"][["seed", "top8_edge_mean"]],
        on="seed", suffixes=("_context", "_baseline"),
    )
    context_seed_delta["delta"] = context_seed_delta["top8_edge_mean_context"] - context_seed_delta["top8_edge_mean_baseline"]
    selected_context_summary = context_table_df[context_table_df["variant"] == selected_context_variant].iloc[0]
    context_update_corr = update_value(selected_context_variant, "bagged", "score_rank_corr_mean")
    context_update_overlap = update_value(selected_context_variant, "bagged", "top8_overlap_mean")
    baseline_industry_row = industry_concentration_summary_df[
        industry_concentration_summary_df["variant"] == "money2_baseline"
    ].iloc[0]
    context_industry_row = industry_concentration_summary_df[
        industry_concentration_summary_df["variant"] == selected_context_variant
    ].iloc[0]
    context_industry_hhi_delta = (
        float(context_industry_row["industry_hhi_mean"]) - float(baseline_industry_row["industry_hhi_mean"])
    )
    context_industry_share_delta = (
        float(context_industry_row["max_industry_share_mean"]) -
        float(baseline_industry_row["max_industry_share_mean"])
    )
    context_checks = {
        "delta_top8_edge": float(selected_context_summary["delta_top8_edge"]) >= ACCEPT_MIN_CONTEXT_DELTA_EDGE,
        "delta_rank_ic": float(selected_context_summary["delta_rank_ic"]) >= ACCEPT_MIN_CONTEXT_DELTA_RANK_IC,
        "positive_years": int((context_year_delta["delta"] > 0).sum()) >= ACCEPT_MIN_POSITIVE_YEARS,
        "positive_seeds": int((context_seed_delta["delta"] > 0).sum()) >= ACCEPT_MIN_POSITIVE_SEEDS,
        "bootstrap_probability": float(context_bootstrap_df["probability_delta_positive"].iloc[0]) >= ACCEPT_MIN_BOOTSTRAP_PROBABILITY,
        "update_corr": context_update_corr - baseline_update_corr >= ACCEPT_MAX_UPDATE_CORR_LOSS,
        "update_overlap": context_update_overlap - baseline_update_overlap >= ACCEPT_MAX_UPDATE_OVERLAP_LOSS,
        "industry_hhi": context_industry_hhi_delta <= ACCEPT_MAX_INDUSTRY_HHI_INCREASE,
        "max_industry_share": context_industry_share_delta <= ACCEPT_MAX_INDUSTRY_SHARE_INCREASE,
    }
    context_promote = baseline_gate_ok and _bi.all(context_checks.values())

    baseline_bagging_row = bagging_comparison_df[
        bagging_comparison_df["variant"] == "money2_baseline"
    ].iloc[0]
    seed42_update_corr = update_value("money2_baseline", "seed42", "score_rank_corr_mean")
    seed42_update_overlap = update_value("money2_baseline", "seed42", "top8_overlap_mean")
    bagging_corr_delta = baseline_update_corr - seed42_update_corr
    bagging_overlap_delta = baseline_update_overlap - seed42_update_overlap
    bagging_promote = bool(
        baseline_gate_ok and
        float(baseline_bagging_row["delta_top8_edge_mean"]) >= ACCEPT_MAX_BAGGING_EDGE_LOSS and
        (bagging_corr_delta > 0 or bagging_overlap_delta > 0)
    )

    final_decision_df = pd.DataFrame([
        {
            "experiment": "money2_stability_distillation", "candidate": selected_distillation_variant,
            "decision": "PROMOTE" if distill_promote else "KEEP_MONEY2",
            "delta_top8_edge": float(selected_distill_row["delta_top8_edge"]),
            "edge_retention": float(selected_distill_row["edge_retention"]),
            "delta_turnover": float(selected_distill_row["delta_turnover"]),
            "delta_update_score_corr": distill_corr_delta,
            "delta_update_top8_overlap": distill_overlap_delta,
            "failed_checks": "" if distill_promote else "core_or_update_stability_gate",
        },
        {
            "experiment": "relative_money_flow_context", "candidate": selected_context_variant,
            "decision": "PROMOTE" if context_promote else "KEEP_MONEY2",
            "delta_top8_edge": float(selected_context_summary["delta_top8_edge"]),
            "edge_retention": np.nan,
            "delta_turnover": float(selected_context_summary["turnover_mean"]) - baseline_turnover,
            "delta_update_score_corr": context_update_corr - baseline_update_corr,
            "delta_update_top8_overlap": context_update_overlap - baseline_update_overlap,
            "delta_industry_hhi": context_industry_hhi_delta,
            "delta_max_industry_share": context_industry_share_delta,
            "failed_checks": ",".join([name for name, passed in context_checks.items() if not passed]),
        },
        {
            "experiment": "same_family_three_seed_bagging", "candidate": "money2_baseline_bagged",
            "decision": "USE_BAGGING" if bagging_promote else "KEEP_SEED42",
            "delta_top8_edge": float(baseline_bagging_row["delta_top8_edge_mean"]),
            "edge_retention": np.nan,
            "delta_turnover": float(baseline_bagging_row["delta_turnover_mean"]),
            "delta_update_score_corr": bagging_corr_delta,
            "delta_update_top8_overlap": bagging_overlap_delta,
            "delta_industry_hhi": np.nan,
            "delta_max_industry_share": np.nan,
            "failed_checks": "" if bagging_promote else "alpha_preservation_or_update_stability_gate",
        },
    ])
    final_decision_df.to_csv(OUT_DIR / "v99_final_decision_table.csv", index=False)
    display_df(bagging_comparison_df, 20)
    display_df(context_bootstrap_df, 10)
    display_df(final_decision_df, 10)
    '''),
    markdown("## 12. 可视化"),
    code(r'''
    if plt is not None and len(overall_summary_df):
        plot_df = overall_summary_df[overall_summary_df["model_mode"] == "bagged"].copy()
        plot_df = plot_df.sort_values("top8_edge_mean", ascending=False)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].bar(np.arange(len(plot_df)), plot_df["top8_edge_mean"] * 100.0, color="#2F6B5F")
        axes[0].set_xticks(np.arange(len(plot_df)))
        axes[0].set_xticklabels(plot_df["variant"], rotation=55, ha="right")
        axes[0].set_ylabel("Mean Top8 edge (%)")
        axes[0].set_title("Bagged OOS Top8 edge")
        axes[1].bar(np.arange(len(plot_df)), plot_df["precision_lift_mean"], color="#C06C3E")
        axes[1].set_xticks(np.arange(len(plot_df)))
        axes[1].set_xticklabels(plot_df["variant"], rotation=55, ha="right")
        axes[1].set_ylabel("Precision lift")
        axes[1].set_title("Bagged OOS Precision lift")
        fig.tight_layout()
        fig.savefig(str(FIG_DIR / "v99_core_top8_precision.png"), dpi=140, bbox_inches="tight")
        plt.show()

        yearly_plot = yearly_summary_df[
            (yearly_summary_df["model_mode"] == "bagged") &
            (yearly_summary_df["variant"].isin(unique_keep_order([
                "money2_baseline", selected_distillation_variant, selected_context_variant,
            ])))
        ]
        fig, ax = plt.subplots(figsize=(11, 5))
        for variant, part in yearly_plot.groupby("variant"):
            part = part.sort_values("year")
            ax.plot(part["year"], part["top8_edge_mean"] * 100.0, marker="o", label=variant)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_ylabel("Top8 edge (%)")
        ax.set_title("Yearly OOS Top8 edge")
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(FIG_DIR / "v99_yearly_top8_edge.png"), dpi=140, bbox_inches="tight")
        plt.show()

    if plt is not None and len(update_summary_df):
        bagged_update = update_summary_df[update_summary_df["model_mode"] == "bagged"].copy()
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        axes[0].bar(np.arange(len(bagged_update)), bagged_update["score_rank_corr_mean"], color="#4E79A7")
        axes[0].set_xticks(np.arange(len(bagged_update)))
        axes[0].set_xticklabels(bagged_update["variant"], rotation=45, ha="right")
        axes[0].set_title("Adjacent-cutoff score correlation")
        axes[1].bar(np.arange(len(bagged_update)), bagged_update["top8_overlap_mean"], color="#E59F3A")
        axes[1].set_xticks(np.arange(len(bagged_update)))
        axes[1].set_xticklabels(bagged_update["variant"], rotation=45, ha="right")
        axes[1].set_title("Adjacent-cutoff Top8 overlap")
        fig.tight_layout()
        fig.savefig(str(FIG_DIR / "v99_update_stability.png"), dpi=140, bbox_inches="tight")
        plt.show()
    '''),
    markdown("## 13. 输出完整性门禁"),
    code(r'''
    required_outputs = [
        OUT_DIR / "v99_feature_contract.csv",
        OUT_DIR / "v99_variant_manifest.csv",
        OUT_DIR / "v99_relative_money_context_panel.csv",
        OUT_DIR / "v99_core_monthly.csv",
        OUT_DIR / "v99_core_summary_overall.csv",
        OUT_DIR / "v99_core_summary_year.csv",
        OUT_DIR / "v99_core_summary_seed.csv",
        OUT_DIR / "v99_selection_industry_concentration_summary.csv",
        OUT_DIR / "v99_seed_top8_overlap_summary.csv",
        OUT_DIR / "v99_distillation_core_decision.csv",
        OUT_DIR / "v99_context_core_decision.csv",
        OUT_DIR / "v99_bagging_comparison.csv",
        OUT_DIR / "v99_context_block_bootstrap.csv",
        OUT_DIR / "v99_adjacent_cutoff_stability_monthly.csv",
        OUT_DIR / "v99_adjacent_cutoff_stability_summary.csv",
        OUT_DIR / "v99_final_decision_table.csv",
        QA_DIR / "v99_base_data_audit.csv",
        QA_DIR / "v99_industry_coverage.csv",
        QA_DIR / "v99_context_feature_coverage.csv",
        QA_DIR / "v99_money2_baseline_reconciliation.csv",
    ]
    output_manifest_df = pd.DataFrame([{
        "path": str(path), "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    } for path in required_outputs])
    output_manifest_df.to_csv(OUT_DIR / "v99_output_manifest.csv", index=False)
    if not bool(output_manifest_df["exists"].all()):
        display_df(output_manifest_df[~output_manifest_df["exists"]], 50)
        raise ValueError("V99 output completeness gate failed")

    run_summary = {
        "product_version": PRODUCT_VERSION, "run_name": RUN_NAME,
        "run_timestamp": RUN_TIMESTAMP, "experiment_signature": EXPERIMENT_SIGNATURE,
        "panel_path": str(PANEL_PATH), "core_cutoffs": list(CORE_CUTOFFS),
        "core_seeds": list(CORE_SEEDS), "fixed_iter": FIXED_ITER,
        "feature_fraction": FEATURE_FRACTION,
        "selected_distillation_variant": selected_distillation_variant,
        "selected_context_variant": selected_context_variant,
        "decisions": final_decision_df.to_dict("records"),
    }
    with open(str(OUT_DIR / "v99_run_summary.json"), "w") as handle:
        json.dump(run_summary, handle, ensure_ascii=False, indent=2, default=str)

    print("V99 complete ->", OUT_DIR)
    display_df(output_manifest_df, 30)
    '''),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "name": "python", "version": "3.6.8", "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3", "nbconvert_exporter": "python", "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 2,
}

for output_path in OUTPUT_PATHS:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(notebook, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print(output_path)
