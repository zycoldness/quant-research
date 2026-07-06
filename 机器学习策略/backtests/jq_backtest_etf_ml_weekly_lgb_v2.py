from jqdata import *
import datetime
import pickle
import numpy as np
import pandas as pd


MODEL_FILE = "etf_ml_v2_outputs/model_etf_ml_v2_lgb_alpha5d_train20160101_20241231.pkl"
STOCK_NUM_OVERRIDE = None
TOP_CANDIDATE_LOG_N = 10
MAX_ETF_GROUP_HOLDINGS = 1
MAX_ETF_GROUP_HOLDINGS_OVERRIDE = None  # None uses model bundle; 0 disables theme cap.
REBALANCE_MODE_OVERRIDE = None  # None uses model bundle; "weekly" or "monthly".
REBALANCE_INTERVAL_WEEKS_OVERRIDE = None  # None uses model bundle; e.g. 2 for 10d labels.

# Risk controls are deliberately small and easy to switch off for A/B tests.
# Breadth risk-off uses the same pool_breadth feature family as the model.
ENABLE_BREADTH_RISK_OFF = False
BREADTH_RISK_OFF_THRESHOLD = 0.35
BREADTH_RISK_OFF_COL = None  # None means "pool_breadth_%s" % pool_context_window.

# Position stop checks only existing holdings. It does not refill before the next
# weekly rebalance, so the model remains the only entry signal.
ENABLE_POSITION_STOP = False
STOP_LOSS_PCT = -0.08
STOP_CHECK_TIME = "14:50"


# Portfolio-layer de-duplication. The model ranks ETFs by strength; this layer
# prevents top3 from becoming a single industry bet such as three bank ETFs.
ETF_GROUP_RULES = [
    ("bank", ["银行"]),
    ("innovative_drug", ["创新药", "新药", "医药", "医疗", "生物", "药"]),
    ("oil_gas", ["油气", "石油", "能源"]),
    ("coal", ["煤炭"]),
    ("semiconductor", ["半导", "芯片", "集成电路"]),
    ("software_ai", ["软件", "人工智能", "AI", "云计算", "大数据", "计算机", "信创"]),
    ("broker", ["证券", "券商"]),
    ("military", ["军工", "国防", "航空航天", "通用航空"]),
    ("gold", ["黄金"]),
    ("nonferrous", ["有色", "稀有金属", "稀土"]),
    ("consumer", ["消费", "食品", "酒"]),
    ("real_estate", ["地产", "房地产"]),
    ("new_energy", ["新能源", "光伏", "电池", "锂电", "储能"]),
]


def initialize(context):
    g.model_file = MODEL_FILE
    bundle = pickle.loads(read_file(g.model_file))
    validate_bundle(bundle)

    g.model = bundle["model"]
    g.feature_cols = list(bundle["feature_cols"])
    g.raw_feature_cols = list(bundle.get("raw_feature_cols", []))
    g.rank_feature_cols = list(bundle.get("rank_feature_cols", []))
    g.context_feature_cols = list(bundle.get("context_feature_cols", []))
    g.fill_values = dict(bundle.get("fill_values", {}))
    g.lookback_days = int(bundle.get("lookback_days", 60))
    g.trend_windows = list(bundle.get("trend_windows", [10, 20, 25, 60]))
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
    g.exclude_name_keywords = list(bundle.get("exclude_name_keywords", []))
    if MAX_ETF_GROUP_HOLDINGS_OVERRIDE is None:
        g.max_etf_group_holdings = int(bundle.get("max_etf_group_holdings", MAX_ETF_GROUP_HOLDINGS))
    else:
        g.max_etf_group_holdings = int(MAX_ETF_GROUP_HOLDINGS_OVERRIDE)
    if g.max_etf_group_holdings <= 0:
        g.max_etf_group_holdings = 999
    g.rebalance_mode = str(bundle.get("rebalance_mode", "weekly"))
    if REBALANCE_MODE_OVERRIDE is not None:
        g.rebalance_mode = str(REBALANCE_MODE_OVERRIDE)
    if g.rebalance_mode not in ["weekly", "monthly"]:
        g.rebalance_mode = "weekly"
    g.rebalance_interval_weeks = int(bundle.get("rebalance_interval_weeks", 1))
    if REBALANCE_INTERVAL_WEEKS_OVERRIDE is not None:
        g.rebalance_interval_weeks = int(REBALANCE_INTERVAL_WEEKS_OVERRIDE)
    if g.rebalance_interval_weeks < 1:
        g.rebalance_interval_weeks = 1
    g.rebalance_week_counter = 0
    g.enable_breadth_risk_off = bool(ENABLE_BREADTH_RISK_OFF)
    g.breadth_risk_off_threshold = float(BREADTH_RISK_OFF_THRESHOLD)
    if BREADTH_RISK_OFF_COL is None:
        g.breadth_risk_off_col = "pool_breadth_%s" % g.pool_context_window
    else:
        g.breadth_risk_off_col = str(BREADTH_RISK_OFF_COL)
    g.enable_position_stop = bool(ENABLE_POSITION_STOP)
    g.stop_loss_pct = float(STOP_LOSS_PCT)

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
        "loaded ETF ML model file=%s objective=%s target=%s features=%s raw=%s rank=%s "
        "lookback=%s history_count=%s stock_num=%s min_money20=%.0f max_group=%s "
        "rebalance_mode=%s interval_weeks=%s benchmark=%s" % (
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
            g.max_etf_group_holdings,
            g.rebalance_mode,
            g.rebalance_interval_weeks,
            g.benchmark,
        )
    )
    log.info("feature cols: %s" % ",".join(g.feature_cols))
    log.info(
        "risk controls: breadth_off=%s breadth_col=%s breadth_threshold=%.4f "
        "position_stop=%s stop_loss=%.4f" % (
            g.enable_breadth_risk_off,
            g.breadth_risk_off_col,
            g.breadth_risk_off_threshold,
            g.enable_position_stop,
            g.stop_loss_pct,
        )
    )

    run_daily(prepare_hold_list, "9:05")
    run_daily(check_position_stop, STOP_CHECK_TIME)
    if g.rebalance_mode == "monthly":
        run_monthly(weekly_rebalance, 1, "9:35")
    else:
        run_weekly(weekly_rebalance, 1, "9:35")


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
    if should_skip_weekly_rebalance(context):
        return
    target_list = get_target_list(context)
    g.target_list = list(target_list)
    g.target_list_date = context.current_dt.date()
    log.info("ETF_ML_TARGETS date=%s targets=%s" % (
        str(context.previous_date),
        ",".join(target_list)
    ))

    for stock in list(g.hold_list):
        if stock not in target_list and stock in context.portfolio.positions:
            close_position(context.portfolio.positions[stock])

    current_holds = [
        p.security
        for p in context.portfolio.positions.values()
        if p.total_amount > 0
    ]
    buy_list = [s for s in target_list if s not in current_holds]
    slots = len(target_list) - len(current_holds)
    if slots <= 0 or context.portfolio.cash <= 0:
        return

    value = context.portfolio.cash / float(slots)
    for stock in buy_list:
        if open_position(stock, value):
            current_holds.append(stock)
            if len(current_holds) >= len(target_list):
                break


def should_skip_weekly_rebalance(context):
    if g.rebalance_mode == "monthly":
        return False
    if g.rebalance_interval_weeks <= 1:
        return False
    week_counter = int(getattr(g, "rebalance_week_counter", 0))
    if week_counter % g.rebalance_interval_weeks != 0:
        log.info("ETF_ML_SKIP_REBALANCE date=%s interval_weeks=%s week_counter=%s" % (
            str(context.previous_date),
            g.rebalance_interval_weeks,
            week_counter,
        ))
        g.rebalance_week_counter = week_counter + 1
        return True
    g.rebalance_week_counter = week_counter + 1
    return False


def check_position_stop(context):
    if not g.enable_position_stop:
        return
    positions = context.portfolio.positions
    if positions is None or len(positions) == 0:
        return
    for stock, position in list(positions.items()):
        if position.total_amount <= 0:
            continue
        price = get_last_price(stock)
        if price is None or price <= 0:
            continue
        avg_cost = float(position.avg_cost) if position.avg_cost is not None else 0.0
        if avg_cost <= 0:
            continue
        pnl = price / avg_cost - 1.0
        if pnl <= g.stop_loss_pct:
            log.warn("ETF_STOP_LOSS security=%s pnl=%.4f price=%.4f avg_cost=%.4f threshold=%.4f" % (
                stock,
                pnl,
                price,
                avg_cost,
                g.stop_loss_pct,
            ))
            close_position(position)


def get_target_list(context):
    feature_date = context.previous_date
    etfs = get_etf_universe(feature_date)
    if len(etfs) == 0:
        return []

    feature_df = build_feature_panel(etfs, feature_date)
    if feature_df is None or feature_df.empty:
        return []

    score_df = score_feature_panel(feature_df)
    if score_df.empty:
        return []

    top_log = score_df.head(TOP_CANDIDATE_LOG_N)
    log.info("ETF_ML_TOP score snapshot: %s" % format_top_scores(top_log))
    if should_risk_off(score_df, feature_date):
        return []
    target_list = select_diversified_targets(score_df, g.stock_num)
    log.info("ETF_ML_GROUPS %s" % format_target_groups(target_list))
    return target_list


def should_risk_off(score_df, feature_date):
    if not g.enable_breadth_risk_off:
        return False
    if score_df is None or score_df.empty:
        return False
    col = g.breadth_risk_off_col
    if col not in score_df.columns:
        log.warn("ETF_RISK_CHECK missing breadth col=%s date=%s" % (col, str(feature_date)))
        return False
    breadth_series = score_df[col].dropna()
    if len(breadth_series) == 0:
        log.warn("ETF_RISK_CHECK empty breadth col=%s date=%s" % (col, str(feature_date)))
        return False
    breadth = float(breadth_series.iloc[0])
    log.info("ETF_RISK_CHECK date=%s %s=%.4f threshold=%.4f" % (
        str(feature_date),
        col,
        breadth,
        g.breadth_risk_off_threshold,
    ))
    if breadth < g.breadth_risk_off_threshold:
        log.warn("ETF_RISK_OFF date=%s reason=breadth %s=%.4f threshold=%.4f" % (
            str(feature_date),
            col,
            breadth,
            g.breadth_risk_off_threshold,
        ))
        return True
    return False


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
        df = df[df["money_mean_20"] >= g.min_avg_money_20]
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
    X = feature_df.reindex(columns=g.feature_cols).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(pd.Series(g.fill_values)).fillna(0)
    scores = np.asarray(g.model.predict(X[g.feature_cols])).reshape(-1)
    out = feature_df.copy()
    out["score"] = scores
    out = out.sort_values("score", ascending=False)
    return out


def classify_etf_group(code):
    name = str(g.etf_name_map.get(code, ""))
    for group_name, keywords in ETF_GROUP_RULES:
        for kw in keywords:
            if kw and kw in name:
                return group_name
    return "single_" + str(code)


def select_diversified_targets(score_df, target_num):
    selected = []
    selected_set = set()
    group_count = {}

    for code, _ in score_df.iterrows():
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
        selected.append(code)
        if len(selected) >= target_num:
            break
    return selected


def format_target_groups(target_list):
    parts = []
    for code in target_list:
        name = str(g.etf_name_map.get(code, ""))
        parts.append("%s:%s:%s" % (code, classify_etf_group(code), name))
    return "; ".join(parts)


def format_top_scores(df):
    parts = []
    for code, row in df.iterrows():
        parts.append("%s=%.6f" % (code, float(row["score"])))
    return "; ".join(parts)


def open_position(security, value):
    if value <= 0:
        return False
    price = get_last_price(security)
    if price is None or price <= 0:
        return False
    if value < price * 100:
        log.info("[%s] value too small: value=%.2f min_value=%.2f" % (security, value, price * 100))
        return False
    order = order_target_value(security, value)
    if order is not None and order.filled > 0:
        return True
    return False


def close_position(position):
    if hasattr(position, "closeable_amount") and position.closeable_amount <= 0:
        return False
    order = order_target_value(position.security, 0)
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
