from jqdata import *
import datetime
import pickle
import numpy as np
import pandas as pd


MODEL_FILE = "etf_ml_v1_outputs/model_etf_ml_lgb_alpha5d_train20160101_20241231.pkl"
STOCK_NUM_OVERRIDE = None
TOP_CANDIDATE_LOG_N = 10


def initialize(context):
    g.model_file = MODEL_FILE
    bundle = pickle.loads(read_file(g.model_file))
    validate_bundle(bundle)

    g.model = bundle["model"]
    g.feature_cols = list(bundle["feature_cols"])
    g.raw_feature_cols = list(bundle.get("raw_feature_cols", []))
    g.rank_feature_cols = list(bundle.get("rank_feature_cols", []))
    g.fill_values = dict(bundle.get("fill_values", {}))
    g.lookback_days = int(bundle.get("lookback_days", 60))
    g.min_listing_days = int(bundle.get("min_listing_days", 180))
    g.min_avg_money_20 = float(bundle.get("min_avg_money_20", 20000000.0))
    g.stock_num = int(bundle.get("stock_num", 3))
    if STOCK_NUM_OVERRIDE is not None:
        g.stock_num = int(STOCK_NUM_OVERRIDE)
    g.benchmark = bundle.get("benchmark", "000985.XSHG")
    g.exclude_name_keywords = list(bundle.get("exclude_name_keywords", []))

    g.hold_list = []
    g.target_list = []
    g.target_list_date = None

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
        "lookback=%s stock_num=%s min_money20=%.0f benchmark=%s" % (
            g.model_file,
            bundle.get("objective", ""),
            bundle.get("target_col", ""),
            len(g.feature_cols),
            len(g.raw_feature_cols),
            len(g.rank_feature_cols),
            g.lookback_days,
            g.stock_num,
            g.min_avg_money_20,
            g.benchmark,
        )
    )
    log.info("feature cols: %s" % ",".join(g.feature_cols))

    run_daily(prepare_hold_list, "9:05")
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
    return score_df.index.tolist()[:min(g.stock_num, len(score_df))]


def get_etf_universe(date):
    try:
        sec_df = get_all_securities(["etf"], date=date)
    except Exception as err:
        log.warn("get_all_securities etf failed: %s" % err)
        return []
    if sec_df is None or sec_df.empty:
        return []

    out = []
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
        except Exception:
            continue
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
                count=g.lookback_days,
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
