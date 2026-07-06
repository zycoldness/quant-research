from jqdata import *
import datetime
import io
import pickle
import numpy as np
import pandas as pd


# V3 backtest harness:
# 1) No ETF industry/theme hard cap by default.
# 2) Supports both exported ML pkl and manual trend/R2 formula.
# 3) Optional ETF-pool breadth risk layer can switch to cash/defensive ETF.
# Default points to the V6 ret3d model exported by
# ETF_V6_anchor_ml_filter_dedup_governance实验.ipynb.
# Change only MODEL_FILE / STOCK_NUM_OVERRIDE when comparing model variants.
MODEL_FILE = "etf_ml_v6_anchor_ml_filter_dedup_governance_outputs/models/model_etf_ml_v6_lgb_ret3d_train20190101_20241231.pkl"
SCORE_MODE = "model"  # "model" or "manual_trend_r2"
STOCK_NUM_OVERRIDE = None
TOP_CANDIDATE_LOG_N = 10
ORDER_DEBUG_LOG = True
CAPITAL_AWARE_TARGETS = True
MAX_ETF_GROUP_HOLDINGS = 0  # 0 disables group cap. Keep only for explicit stress tests.
USE_ORTHOGONAL_POOL = True
ORTHOGONAL_POOL_FILE = "etf_orthogonal_pool_v1.csv"
RISK_LAYER_MODE = "none"  # "none" or "breadth_cash"
RISK_BREADTH_THRESHOLD = 0.35
DEFENSIVE_ETF = None  # e.g. "511880.XSHG"; None means risk-off fully holds cash.
ETF_GROUP_RULES = []

DEFAULT_EXCLUDE_NAME_KEYWORDS = [
    "债", "国债", "地债", "政金债", "公司债", "城投", "可转债",
    "货币", "现金", "快线", "快钱", "同业存单",
    "短融", "中票", "AAA", "信用", "0-3", "1-3", "政策性金融债",
]

DEFAULT_TREND_WINDOWS = [10, 20, 25, 60]
DEFAULT_RAW_FEATURE_COLS = [
    "ret_1", "ret_5", "ret_10", "ret_20", "ret_60",
    "vol_5", "vol_20", "vol_60",
    "close_to_ma20", "close_to_ma60", "ma5_to_ma20", "ma20_to_ma60",
    "drawdown_20", "drawdown_60",
    "amp_20", "amp_60",
    "money_mean_20", "money_ratio_5_20", "money_ratio_20_60",
    "volume_ratio_5_20", "volume_ratio_20_60",
    "max_ret_20", "min_ret_20",
]
DEFAULT_TREND_FEATURE_COLS = []
for _w in DEFAULT_TREND_WINDOWS:
    DEFAULT_TREND_FEATURE_COLS.extend([
        "trend_ann_%s" % _w,
        "trend_r2_%s" % _w,
        "trend_score_%s" % _w,
        "trend_vol_%s" % _w,
        "trend_score_vol_adj_%s" % _w,
        "trend_simple_ann_%s" % _w,
    ])
DEFAULT_RAW_FEATURE_COLS = DEFAULT_RAW_FEATURE_COLS + DEFAULT_TREND_FEATURE_COLS


def initialize(context):
    g.score_mode = SCORE_MODE
    g.model_file = MODEL_FILE
    bundle = {}
    if g.score_mode == "model":
        bundle = pickle.loads(read_file(g.model_file))
        validate_bundle(bundle)
        g.model = bundle["model"]
        g.feature_cols = list(bundle["feature_cols"])
        g.raw_feature_cols = list(bundle.get("raw_feature_cols", []))
        g.rank_feature_cols = list(bundle.get("rank_feature_cols", []))
        g.context_feature_cols = list(bundle.get("context_feature_cols", []))
        g.fill_values = dict(bundle.get("fill_values", {}))
    elif g.score_mode == "manual_trend_r2":
        g.model = None
        g.raw_feature_cols = list(DEFAULT_RAW_FEATURE_COLS)
        g.rank_feature_cols = ["rank_" + col for col in g.raw_feature_cols]
        g.context_feature_cols = ["pool_breadth_25", "pool_median_vol_25"]
        g.feature_cols = list(g.raw_feature_cols + g.rank_feature_cols + g.context_feature_cols)
        g.fill_values = {}
    else:
        raise ValueError("unsupported SCORE_MODE: %s" % g.score_mode)

    g.lookback_days = int(bundle.get("lookback_days", 60))
    g.trend_windows = list(bundle.get("trend_windows", DEFAULT_TREND_WINDOWS))
    g.trend_weight_end = float(bundle.get("trend_weight_end", 2.0))
    g.price_history_count = int(bundle.get(
        "price_history_count",
        max(g.lookback_days + 1, max(g.trend_windows) + 1 if len(g.trend_windows) else g.lookback_days + 1)
    ))
    g.pool_context_window = int(bundle.get("pool_context_window", 25))
    g.min_listing_days = int(bundle.get("min_listing_days", 180))
    g.min_avg_money_20 = float(bundle.get("min_avg_money_20", 20000000.0))
    g.stock_num = int(bundle.get("stock_num", 3))
    if STOCK_NUM_OVERRIDE is not None:
        g.stock_num = int(STOCK_NUM_OVERRIDE)
    g.benchmark = bundle.get("benchmark", "000985.XSHG")
    g.exclude_name_keywords = list(bundle.get("exclude_name_keywords", DEFAULT_EXCLUDE_NAME_KEYWORDS))
    g.max_etf_group_holdings = int(MAX_ETF_GROUP_HOLDINGS)
    if g.max_etf_group_holdings <= 0:
        g.max_etf_group_holdings = None
    g.use_orthogonal_pool = bool(USE_ORTHOGONAL_POOL)
    g.orthogonal_pool_file = ORTHOGONAL_POOL_FILE
    g.orthogonal_pool_codes = load_orthogonal_pool_codes(g.orthogonal_pool_file) if g.use_orthogonal_pool else set()
    g.risk_layer_mode = RISK_LAYER_MODE
    g.risk_breadth_threshold = float(RISK_BREADTH_THRESHOLD)
    g.defensive_etf = DEFENSIVE_ETF

    g.hold_list = []
    g.target_list = []
    g.target_list_date = None
    g.etf_name_map = {}

    set_benchmark(g.benchmark)
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_slippage(FixedSlippage(0.001))
    try:
        set_order_cost(OrderCost(
            open_tax=0,
            close_tax=0,
            open_commission=0.0001,
            close_commission=0.0001,
            close_today_commission=0,
            min_commission=0
        ), type="fund")
    except Exception:
        set_order_cost(OrderCost(
            open_tax=0,
            close_tax=0,
            open_commission=0.0001,
            close_commission=0.0001,
            close_today_commission=0,
            min_commission=0
        ), type="stock")
    log.set_level("order", "error")

    log.info(
        "loaded ETF selector mode=%s model_file=%s objective=%s target=%s features=%s raw=%s rank=%s "
        "lookback=%s history_count=%s stock_num=%s min_money20=%.0f max_group=%s orthogonal=%s pool_size=%s "
        "risk=%s threshold=%.2f defensive=%s benchmark=%s" % (
            g.score_mode,
            g.model_file,
            bundle.get("objective", ""),
            bundle.get("target_col", ""),
            len(g.feature_cols),
            len(g.raw_feature_cols),
            len(g.rank_feature_cols),
            g.lookback_days,
            g.price_history_count,
            g.stock_num,
            g.min_avg_money_20,
            str(g.max_etf_group_holdings),
            g.use_orthogonal_pool,
            len(g.orthogonal_pool_codes),
            g.risk_layer_mode,
            g.risk_breadth_threshold,
            str(g.defensive_etf),
            g.benchmark,
        )
    )
    log.info("feature cols: %s" % ",".join(g.feature_cols))

    run_daily(prepare_hold_list, "9:05")
    run_weekly(weekly_rebalance, 1, "9:35")


def load_orthogonal_pool_codes(path):
    try:
        raw = read_file(path)
    except Exception as err:
        raise ValueError("failed to read orthogonal pool file %s: %s" % (path, err))
    if isinstance(raw, bytes):
        text = raw.decode("utf-8")
    else:
        text = str(raw)
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as err:
        raise ValueError("failed to parse orthogonal pool csv %s: %s" % (path, err))
    if "code" not in df.columns:
        raise ValueError("orthogonal pool csv missing code column: %s" % path)
    codes = set(df["code"].dropna().astype(str).tolist())
    if len(codes) == 0:
        raise ValueError("orthogonal pool is empty: %s" % path)
    return codes


def validate_bundle(bundle):
    if not isinstance(bundle, dict):
        raise ValueError("model bundle should be dict")
    for key in ["model", "feature_cols", "fill_values"]:
        if key not in bundle:
            raise ValueError("model bundle missing key: %s" % key)
    if len(bundle.get("feature_cols", [])) == 0:
        raise ValueError("model bundle feature_cols is empty")


def prepare_hold_list(context):
    g.hold_list = [
        position.security
        for position in context.portfolio.positions.values()
        if position.total_amount > 0
    ]


def weekly_rebalance(context):
    target_list = get_target_list(context)
    g.target_list = list(target_list)
    g.target_list_date = context.current_dt.date()
    log.info("ETF_ML_TARGETS date=%s targets=%s" % (
        str(context.previous_date),
        ",".join(target_list)
    ))
    log.info("ETF_ML_HOLD_BEFORE date=%s total_value=%.2f cash=%.2f holds=%s" % (
        str(context.previous_date),
        float(context.portfolio.total_value),
        float(context.portfolio.cash),
        format_current_positions(context),
    ))

    if len(target_list) == 0:
        for stock in list(g.hold_list):
            if stock in context.portfolio.positions:
                close_position(context.portfolio.positions[stock])
        return

    for stock in list(g.hold_list):
        if stock not in target_list and stock in context.portfolio.positions:
            close_position(context.portfolio.positions[stock])

    # Match the notebook proxy: every rebalance holds the selected ETF basket
    # at equal weights. Do not rely on same-callback cash/position refresh after
    # sell orders, otherwise the strategy can sit in stale holdings or cash.
    target_value = context.portfolio.total_value / float(len(target_list))
    log.info("ETF_ML_REBALANCE target_count=%s target_value=%.2f" % (
        len(target_list),
        target_value,
    ))
    for stock in target_list:
        adjust_position_to_value(stock, target_value, context)

    log.info("ETF_ML_HOLD_AFTER_SUBMIT date=%s cash=%.2f holds=%s" % (
        str(context.previous_date),
        float(context.portfolio.cash),
        format_current_positions(context),
    ))


def get_target_list(context):
    feature_date = context.previous_date
    etfs = get_etf_universe(feature_date)
    log.info("ETF_ML_POOL date=%s universe_count=%s orthogonal=%s pool_file=%s" % (
        str(feature_date),
        len(etfs),
        g.use_orthogonal_pool,
        g.orthogonal_pool_file,
    ))
    if len(etfs) == 0:
        return []

    feature_df = build_feature_panel(etfs, feature_date)
    if feature_df is None or feature_df.empty:
        return []
    log.info("ETF_ML_FEATURES date=%s feature_count=%s breadth=%s" % (
        str(feature_date),
        len(feature_df),
        format_pool_breadth(feature_df),
    ))

    score_df = score_feature_panel(feature_df)
    if score_df.empty:
        return []

    top_log = score_df.head(TOP_CANDIDATE_LOG_N)
    log.info("ETF_ML_TOP score snapshot: %s" % format_top_scores(top_log))
    if is_risk_off(feature_df):
        log.info("ETF_ML_RISK_OFF date=%s breadth=%s threshold=%.2f mode=%s" % (
            str(feature_date),
            format_pool_breadth(feature_df),
            g.risk_breadth_threshold,
            g.risk_layer_mode,
        ))
        target_list = get_risk_off_targets(feature_date)
    else:
        target_value = context.portfolio.total_value / float(max(1, g.stock_num))
        target_list = select_targets(score_df, g.stock_num, target_value)
    log.info("ETF_ML_GROUPS %s" % format_target_groups(target_list))
    return target_list


def is_risk_off(feature_df):
    if g.risk_layer_mode == "none":
        return False
    if g.risk_layer_mode != "breadth_cash":
        return False
    breadth_col = "pool_breadth_%s" % g.pool_context_window
    if breadth_col not in feature_df.columns:
        return False
    s = pd.to_numeric(feature_df[breadth_col], errors="coerce").dropna()
    if len(s) == 0:
        return False
    return float(s.iloc[0]) < g.risk_breadth_threshold


def format_pool_breadth(feature_df):
    breadth_col = "pool_breadth_%s" % g.pool_context_window
    if breadth_col not in feature_df.columns:
        return "nan"
    s = pd.to_numeric(feature_df[breadth_col], errors="coerce").dropna()
    if len(s) == 0:
        return "nan"
    return "%.4f" % float(s.iloc[0])


def get_risk_off_targets(date):
    if g.defensive_etf is None or str(g.defensive_etf) == "":
        return []
    defensive = str(g.defensive_etf)
    try:
        current_data = get_current_data()
        if current_data[defensive].paused:
            return []
    except Exception:
        pass
    return [defensive]


def get_etf_universe(date):
    try:
        sec_df = get_all_securities(["etf"], date=date)
    except Exception as err:
        log.warn("get_all_securities etf failed: %s" % err)
        return []
    if sec_df is None or sec_df.empty:
        return []

    out = []
    name_map = {}
    current_data = get_current_data()
    for code, row in sec_df.iterrows():
        try:
            start_date = row.get("start_date", None)
            if pd.isnull(start_date):
                start_date = get_security_info(code).start_date
            start_date = pd.Timestamp(start_date).date()
            feature_date = pd.Timestamp(date).date()
            if feature_date - start_date < datetime.timedelta(days=g.min_listing_days):
                continue
            name = str(row.get("display_name", "")) or str(get_security_info(code).display_name)
            if should_exclude_name(name):
                continue
            if g.use_orthogonal_pool and code not in g.orthogonal_pool_codes:
                continue
            try:
                if current_data[code].paused:
                    continue
            except Exception:
                pass
            out.append(code)
            name_map[code] = name
        except Exception:
            continue
    g.etf_name_map = name_map
    return out


def should_exclude_name(name):
    for kw in g.exclude_name_keywords:
        if kw and kw in name:
            return True
    return False


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def build_feature_panel(etfs, date):
    rows = []
    fields = ["open", "high", "low", "close", "volume", "money"]
    for etf_chunk in chunks(etfs, 120):
        try:
            price_df = get_price(
                etf_chunk,
                end_date=date,
                frequency="daily",
                fields=fields,
                count=g.price_history_count,
                panel=False,
                fq="pre",
                skip_paused=False
            )
        except Exception as err:
            log.warn("get_price feature chunk failed: %s" % err)
            continue
        if price_df is None or price_df.empty:
            continue
        for code, one in price_df.groupby("code"):
            rec = calc_one_etf_features(code, one)
            if rec is not None:
                rows.append(rec)
    if len(rows) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("code")
    df = df.replace([np.inf, -np.inf], np.nan)
    if "money_mean_20" in df.columns:
        before_liq = len(df)
        df = df[df["money_mean_20"] >= g.min_avg_money_20]
        log.info("ETF_ML_LIQ_FILTER before=%s after=%s min_money20=%.0f" % (
            before_liq,
            len(df),
            g.min_avg_money_20,
        ))
    df = add_pool_context_features(df)
    df = add_rank_features(df)
    return df


def calc_one_etf_features(code, price_df):
    df = price_df.sort_values("time").copy()
    if len(df) < max(40, int(g.lookback_days * 0.8)):
        return None
    for col in ["open", "high", "low", "close", "volume", "money"]:
        if col not in df.columns:
            return None
    close = pd.Series(df["close"].astype(float).values)
    high = pd.Series(df["high"].astype(float).values)
    low = pd.Series(df["low"].astype(float).values)
    volume = pd.Series(df["volume"].astype(float).values)
    money = pd.Series(df["money"].astype(float).values)
    if close.isnull().any() or close.iloc[-1] <= 0:
        return None

    ret = close.pct_change()
    rec = {"code": code}
    rec["ret_1"] = safe_ret(close, 1)
    rec["ret_5"] = safe_ret(close, 5)
    rec["ret_10"] = safe_ret(close, 10)
    rec["ret_20"] = safe_ret(close, 20)
    rec["ret_60"] = safe_ret(close, 60)
    rec["vol_5"] = ret.tail(5).std()
    rec["vol_20"] = ret.tail(20).std()
    rec["vol_60"] = ret.tail(60).std()
    rec["close_to_ma20"] = safe_ratio(close.iloc[-1], close.tail(20).mean()) - 1
    rec["close_to_ma60"] = safe_ratio(close.iloc[-1], close.tail(60).mean()) - 1
    rec["ma5_to_ma20"] = safe_ratio(close.tail(5).mean(), close.tail(20).mean()) - 1
    rec["ma20_to_ma60"] = safe_ratio(close.tail(20).mean(), close.tail(60).mean()) - 1
    rec["drawdown_20"] = safe_ratio(close.iloc[-1], close.tail(20).max()) - 1
    rec["drawdown_60"] = safe_ratio(close.iloc[-1], close.tail(60).max()) - 1
    rec["amp_20"] = (high.tail(20) / low.tail(20) - 1).replace([np.inf, -np.inf], np.nan).mean()
    rec["amp_60"] = (high.tail(60) / low.tail(60) - 1).replace([np.inf, -np.inf], np.nan).mean()
    rec["money_mean_20"] = money.tail(20).mean()
    rec["money_ratio_5_20"] = safe_ratio(money.tail(5).mean(), money.tail(20).mean()) - 1
    rec["money_ratio_20_60"] = safe_ratio(money.tail(20).mean(), money.tail(60).mean()) - 1
    rec["volume_ratio_5_20"] = safe_ratio(volume.tail(5).mean(), volume.tail(20).mean()) - 1
    rec["volume_ratio_20_60"] = safe_ratio(volume.tail(20).mean(), volume.tail(60).mean()) - 1
    rec["max_ret_20"] = ret.tail(20).max()
    rec["min_ret_20"] = ret.tail(20).min()
    for w in g.trend_windows:
        tm = calc_trend_metrics(close, int(w))
        rec["trend_ann_%s" % w] = tm["ann"]
        rec["trend_r2_%s" % w] = tm["r2"]
        rec["trend_score_%s" % w] = tm["score"]
        rec["trend_vol_%s" % w] = tm["vol"]
        rec["trend_score_vol_adj_%s" % w] = tm["score_vol_adj"]
        rec["trend_simple_ann_%s" % w] = tm["simple_ann"]
    return rec


def safe_ret(close, days):
    if len(close) <= days:
        return np.nan
    base = close.iloc[-days - 1]
    if pd.isnull(base) or base <= 0:
        return np.nan
    return close.iloc[-1] / base - 1


def safe_ratio(a, b):
    if pd.isnull(a) or pd.isnull(b) or b == 0:
        return np.nan
    return float(a) / float(b)


def calc_trend_metrics(close, days):
    if len(close) <= days:
        return {
            "ann": np.nan,
            "r2": np.nan,
            "score": np.nan,
            "vol": np.nan,
            "score_vol_adj": np.nan,
            "simple_ann": np.nan,
        }
    recent = pd.Series(close.iloc[-(days + 1):].astype(float).values)
    if recent.isnull().any() or (recent <= 0).any():
        return {
            "ann": np.nan,
            "r2": np.nan,
            "score": np.nan,
            "vol": np.nan,
            "score_vol_adj": np.nan,
            "simple_ann": np.nan,
        }
    y = np.log(recent.values)
    x = np.arange(len(y))
    weights = np.linspace(1.0, g.trend_weight_end, len(y))
    try:
        slope, intercept = np.polyfit(x, y, 1, w=weights)
    except Exception:
        return {
            "ann": np.nan,
            "r2": np.nan,
            "score": np.nan,
            "vol": np.nan,
            "score_vol_adj": np.nan,
            "simple_ann": np.nan,
        }
    ann = np.exp(slope * 250.0) - 1.0
    fit = slope * x + intercept
    ss_res = np.sum(weights * (y - fit) ** 2)
    ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    r2 = max(0.0, min(1.0, float(r2)))
    daily_returns = recent.pct_change().dropna()
    vol = float(daily_returns.std() * np.sqrt(250.0)) if len(daily_returns) > 1 else np.nan
    period_ret = recent.iloc[-1] / recent.iloc[0] - 1.0
    simple_ann = (1.0 + period_ret) ** (250.0 / float(days)) - 1.0 if 1.0 + period_ret > 0 else np.nan
    score = ann * r2
    score_vol_adj = score * 0.20 / max(vol, 0.05) if not pd.isnull(vol) else np.nan
    return {
        "ann": ann,
        "r2": r2,
        "score": score,
        "vol": vol,
        "score_vol_adj": score_vol_adj,
        "simple_ann": simple_ann,
    }


def add_pool_context_features(df):
    out = df.copy()
    if len(out) == 0:
        return out
    ann_col = "trend_ann_%s" % g.pool_context_window
    vol_col = "trend_vol_%s" % g.pool_context_window
    breadth_col = "pool_breadth_%s" % g.pool_context_window
    median_vol_col = "pool_median_vol_%s" % g.pool_context_window
    if ann_col in out.columns:
        out[breadth_col] = float((out[ann_col] > 0).mean())
    if vol_col in out.columns:
        out[median_vol_col] = float(out[vol_col].median())
    return out


def add_rank_features(df):
    out = df.copy()
    for col in g.raw_feature_cols:
        if col in out.columns:
            rank_col = "rank_" + col
            out[rank_col] = out[col].rank(pct=True)
    return out


def score_feature_panel(feature_df):
    out = feature_df.copy()
    if g.score_mode == "manual_trend_r2":
        out["score"] = calc_manual_trend_score(out)
    else:
        X = feature_df.reindex(columns=g.feature_cols).replace([np.inf, -np.inf], np.nan)
        X = X.fillna(pd.Series(g.fill_values)).fillna(0)
        scores = np.asarray(g.model.predict(X[g.feature_cols])).reshape(-1)
        out["score"] = scores
    out = out.sort_values("score", ascending=False)
    return out


def calc_manual_trend_score(df):
    out = pd.Series(0.0, index=df.index)
    weights = [
        ("rank_trend_score_vol_adj_25", 0.40),
        ("rank_trend_score_20", 0.20),
        ("rank_trend_r2_25", 0.15),
        ("rank_ret_20", 0.10),
        ("rank_drawdown_20", 0.10),
        ("rank_money_ratio_5_20", 0.05),
        ("rank_vol_20", -0.10),
    ]
    used = 0
    for col, weight in weights:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        fill_value = s.median()
        if pd.isnull(fill_value):
            fill_value = 0.5
        out = out + s.fillna(fill_value).fillna(0.5).astype(float) * weight
        used += 1
    if used == 0:
        return pd.Series(0.0, index=df.index)
    return out


def classify_etf_group(code):
    name = str(g.etf_name_map.get(code, ""))
    for group_name, keywords in ETF_GROUP_RULES:
        for kw in keywords:
            if kw and kw in name:
                return group_name
    return "single_" + str(code)


def select_targets(score_df, target_num, target_value):
    if g.max_etf_group_holdings is None:
        return select_capital_aware_targets(score_df, target_num, target_value)
    return select_group_limited_targets(score_df, target_num, target_value)


def select_group_limited_targets(score_df, target_num, target_value):
    selected = []
    selected_set = set()
    group_count = {}

    for code, _ in score_df.iterrows():
        if not can_hold_target(code, target_value):
            continue
        group_name = classify_etf_group(code)
        cnt = group_count.get(group_name, 0)
        if cnt >= g.max_etf_group_holdings:
            continue
        selected.append(code)
        selected_set.add(code)
        group_count[group_name] = cnt + 1
        if len(selected) >= target_num:
            return selected

    # Fallback: if grouping is too strict or names are incomplete, fill by score.
    for code, _ in score_df.iterrows():
        if code in selected_set:
            continue
        if not can_hold_target(code, target_value):
            continue
        selected.append(code)
        if len(selected) >= target_num:
            break
    return selected


def select_capital_aware_targets(score_df, target_num, target_value):
    selected = []
    for code, _ in score_df.iterrows():
        if not can_hold_target(code, target_value):
            continue
        selected.append(code)
        if len(selected) >= target_num:
            break
    return selected


def can_hold_target(security, target_value):
    if not CAPITAL_AWARE_TARGETS:
        return True
    if security in g.hold_list:
        return True
    price = get_last_price(security)
    if price is None or price <= 0:
        if ORDER_DEBUG_LOG:
            log.info("ETF_ML_TARGET_SKIP security=%s reason=no_price target_value=%.2f" % (
                security,
                target_value,
            ))
        return False
    min_value = price * 100
    if target_value * 0.98 < min_value:
        if ORDER_DEBUG_LOG:
            log.info("ETF_ML_TARGET_SKIP security=%s reason=min_lot price=%.4f target_value=%.2f min_value=%.2f" % (
                security,
                price,
                target_value,
                min_value,
            ))
        return False
    return True


def format_target_groups(target_list):
    parts = []
    for code in target_list:
        name = str(g.etf_name_map.get(code, ""))
        parts.append("%s:%s:%s" % (code, classify_etf_group(code), name))
    return "; ".join(parts)


def format_top_scores(df):
    parts = []
    for code, row in df.iterrows():
        name = str(g.etf_name_map.get(code, ""))
        text = "%s:%s score=%.6f" % (code, name, float(row["score"]))
        if g.score_mode == "manual_trend_r2":
            text += " r25=%.3f s25v=%.3f s20=%.3f ret20=%.3f dd20=%.3f vol20=%.3f" % (
                safe_float(row.get("rank_trend_r2_25", np.nan)),
                safe_float(row.get("rank_trend_score_vol_adj_25", np.nan)),
                safe_float(row.get("rank_trend_score_20", np.nan)),
                safe_float(row.get("rank_ret_20", np.nan)),
                safe_float(row.get("rank_drawdown_20", np.nan)),
                safe_float(row.get("rank_vol_20", np.nan)),
            )
        parts.append(text)
    return "; ".join(parts)


def safe_float(x):
    try:
        if pd.isnull(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def format_current_positions(context):
    parts = []
    for stock, pos in context.portfolio.positions.items():
        try:
            if pos.total_amount <= 0:
                continue
            parts.append("%s:amount=%s,value=%.2f,closeable=%s" % (
                stock,
                int(pos.total_amount),
                float(pos.value),
                int(getattr(pos, "closeable_amount", 0)),
            ))
        except Exception:
            continue
    if len(parts) == 0:
        return "EMPTY"
    return "; ".join(parts)


def adjust_position_to_value(security, value, context=None):
    if value <= 0:
        return False
    price = get_last_price(security)
    if price is None or price <= 0:
        if ORDER_DEBUG_LOG:
            log.warn("ETF_ML_ORDER_SKIP security=%s reason=no_price target_value=%.2f" % (security, value))
        return False
    if value < price * 100:
        log.info("[%s] value too small: value=%.2f min_value=%.2f" % (security, value, price * 100))
        return False
    if context is not None and security in context.portfolio.positions:
        pos = context.portfolio.positions[security]
        current_value = float(getattr(pos, "value", 0.0))
        current_amount = int(getattr(pos, "total_amount", 0))
        diff_value = float(value) - current_value
        if current_amount > 0 and abs(diff_value) < price * 100:
            if ORDER_DEBUG_LOG:
                log.info("ETF_ML_ORDER_SKIP security=%s reason=small_adjust current_value=%.2f target_value=%.2f price=%.4f" % (
                    security,
                    current_value,
                    value,
                    price,
                ))
            return True
    order = order_target_value(security, value)
    if ORDER_DEBUG_LOG:
        if order is None:
            log.info("ETF_ML_ORDER security=%s target_value=%.2f price=%.4f order=None" % (
                security,
                value,
                price,
            ))
        else:
            log.info("ETF_ML_ORDER security=%s target_value=%.2f price=%.4f amount=%s filled=%s status=%s" % (
                security,
                value,
                price,
                str(getattr(order, "amount", "")),
                str(getattr(order, "filled", "")),
                str(getattr(order, "status", "")),
            ))
    if order is not None and order.filled > 0:
        return True
    return False


def open_position(security, value):
    return adjust_position_to_value(security, value)


def close_position(position):
    if hasattr(position, "closeable_amount") and position.closeable_amount <= 0:
        if ORDER_DEBUG_LOG:
            log.info("ETF_ML_CLOSE_SKIP security=%s reason=no_closeable amount=%s" % (
                position.security,
                str(position.total_amount),
            ))
        return False
    if hasattr(position, "total_amount") and position.total_amount < 100:
        if ORDER_DEBUG_LOG:
            log.info("ETF_ML_CLOSE_SKIP security=%s reason=small_lot amount=%s" % (
                position.security,
                str(position.total_amount),
            ))
        return False
    order = order_target_value(position.security, 0)
    if ORDER_DEBUG_LOG:
        if order is None:
            log.info("ETF_ML_CLOSE security=%s order=None" % position.security)
        else:
            log.info("ETF_ML_CLOSE security=%s amount=%s filled=%s status=%s" % (
                position.security,
                str(getattr(order, "amount", "")),
                str(getattr(order, "filled", "")),
                str(getattr(order, "status", "")),
            ))
    if order is not None:
        if order.status == OrderStatus.held and order.filled == order.amount:
            return True
    return False


def get_last_price(security):
    try:
        current_data = get_current_data()
        price = current_data[security].last_price
        if price is not None and (not pd.isnull(price)) and float(price) > 0:
            return float(price)
    except Exception:
        pass
    try:
        last_prices = history(1, unit="1m", field="close", security_list=[security])
        if security in last_prices.columns:
            price = float(last_prices[security][-1])
            if price > 0:
                return price
    except Exception:
        pass
    return None
