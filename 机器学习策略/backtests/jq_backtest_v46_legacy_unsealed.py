from jqdata import *
from jqfactor import get_factor_values
import datetime
import gc
import numpy as np
import pandas as pd
import pickle


# Clean V46 executor.
# The model bundle's base_feature_cols is the only source of truth.
# JQ-native factors are fetched by get_factor_values; only the columns below
# are computed by this script.
MODEL_FILE = "model_candidate_v46_lgb_direct_hybrid_l2_ff10_2019_2025q1_legacy_unsealed.pkl"

SUPPORTED_PRICE_FEATURE_COLS = [
    "px_close_to_ma60",
    "px_drawdown_60",
    "liq_money_ratio_20_60",
    "liq_paused_count_20",
]

SUPPORTED_TEMPORAL_FEATURE_COLS = [
    "ts_cash_flow_to_price_ratio_rank_mean_3m",
    "ts_Rank1M_rank_chg_1m",
]

SELF_BUILT_FEATURE_COLS = SUPPORTED_PRICE_FEATURE_COLS + SUPPORTED_TEMPORAL_FEATURE_COLS

UNIVERSE_INDEX = "000906.XSHG"
STOCK_NUM_OVERRIDE = None
CANDIDATE_POOL_SIZE_OVERRIDE = None
MIN_LISTING_DAYS = 180
INDUSTRY_CAP_RATIO_DEFAULT = 0.20
KCB_PROTECT_PCT = 0.02


def initialize(context):
    g.model_file = MODEL_FILE
    bundle = pickle.loads(read_file(g.model_file))
    validate_bundle(bundle)

    g.model = bundle["base_model"]
    g.feature_cols = unique_keep_order(list(bundle["base_feature_cols"]))
    g.fill_values = dict(bundle.get("base_fill_values", {}))

    g.price_feature_cols = [c for c in g.feature_cols if c in SUPPORTED_PRICE_FEATURE_COLS]
    g.temporal_feature_cols = [c for c in g.feature_cols if c in SUPPORTED_TEMPORAL_FEATURE_COLS]
    g.jq_factor_cols = [
        c for c in g.feature_cols
        if c not in SUPPORTED_PRICE_FEATURE_COLS
        and c not in SUPPORTED_TEMPORAL_FEATURE_COLS
    ]

    check_unsupported_self_built_features(g.feature_cols)

    g.universe_index = UNIVERSE_INDEX
    g.benchmark = bundle.get("benchmark", UNIVERSE_INDEX)
    g.stock_num = int(bundle.get("stock_num", 10))
    if STOCK_NUM_OVERRIDE is not None:
        g.stock_num = int(STOCK_NUM_OVERRIDE)

    g.candidate_pool_size = int(bundle.get("top_n_candidates", 30))
    if CANDIDATE_POOL_SIZE_OVERRIDE is not None:
        g.candidate_pool_size = int(CANDIDATE_POOL_SIZE_OVERRIDE)

    g.min_listing_days = MIN_LISTING_DAYS
    g.industry_cap_ratio = float(bundle.get("industry_cap_ratio", INDUSTRY_CAP_RATIO_DEFAULT))
    g.kcb_protect_pct = KCB_PROTECT_PCT

    g.hold_list = []
    g.yesterday_HL_list = []
    g.target_list = []
    g.target_list_date = None

    set_benchmark(g.benchmark)
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_slippage(PriceRelatedSlippage(0.00246))
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=0.001,
        open_commission=0.0003,
        close_commission=0.0003,
        close_today_commission=0,
        min_commission=5
    ), type="stock")
    log.set_level("order", "error")

    log.info(
        "loaded clean V46 model file=%s features=%s jq=%s price=%s temporal=%s "
        "stock_num=%s candidates=%s" % (
            g.model_file,
            len(g.feature_cols),
            len(g.jq_factor_cols),
            len(g.price_feature_cols),
            len(g.temporal_feature_cols),
            g.stock_num,
            g.candidate_pool_size,
        )
    )
    log.info("feature cols: %s" % ",".join(g.feature_cols))

    run_daily(prepare_stock_list, "9:05")
    run_monthly(monthly_sell, 1, "9:40")
    run_monthly(monthly_buy, 1, "9:50")


def validate_bundle(bundle):
    if not isinstance(bundle, dict):
        raise ValueError("model bundle should be dict")
    for key in ["base_model", "base_feature_cols", "base_fill_values"]:
        if key not in bundle:
            raise ValueError("model bundle missing key: %s" % key)
    if len(bundle.get("base_feature_cols", [])) == 0:
        raise ValueError("model bundle base_feature_cols is empty")
    overlay_mode = bundle.get("overlay_mode", "direct")
    if overlay_mode not in ["direct", None]:
        raise ValueError("clean V46 executor only supports direct base_model bundles, got: %s" % overlay_mode)


def check_unsupported_self_built_features(feature_cols):
    unsupported = []
    for col in feature_cols:
        if col.startswith("px_") or col.startswith("liq_") or col.startswith("ts_"):
            if col not in SELF_BUILT_FEATURE_COLS:
                unsupported.append(col)
    if len(unsupported) > 0:
        raise ValueError(
            "unsupported self-built features: %s. Add a small explicit adapter before testing this model." %
            ",".join(unsupported)
        )


def unique_keep_order(cols):
    seen = set()
    out = []
    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def prepare_stock_list(context):
    g.hold_list = [
        position.security
        for position in context.portfolio.positions.values()
        if position.total_amount > 0
    ]

    if len(g.hold_list) == 0:
        g.yesterday_HL_list = []
        return

    df = get_price(
        g.hold_list,
        end_date=context.previous_date,
        frequency="daily",
        fields=["close", "high_limit"],
        count=1,
        panel=False,
        fill_paused=False
    )
    if df is None or df.empty:
        g.yesterday_HL_list = []
        return

    df = df[df["close"] == df["high_limit"]]
    g.yesterday_HL_list = list(df["code"])


def get_stock_list(context):
    yesterday = context.previous_date
    stock_list = get_index_stocks(g.universe_index, yesterday)
    if len(stock_list) == 0:
        return []

    stock_list = filter_st_stock(stock_list)
    stock_list = filter_paused_stock(stock_list)
    stock_list = filter_new_stock(context, stock_list)
    stock_list = filter_limitup_stock(context, stock_list)
    stock_list = filter_limitdown_stock(context, stock_list)
    stock_list = filter_min_lot_stock(context, stock_list)
    if len(stock_list) == 0:
        return []

    factor_df = pd.DataFrame(index=stock_list)
    if len(g.jq_factor_cols) > 0:
        factor_df = factor_df.join(fetch_jq_factor_data(stock_list, g.jq_factor_cols, yesterday), how="left")

    if len(g.price_feature_cols) > 0:
        price_df = get_price_feature_data(stock_list, yesterday, g.price_feature_cols)
        factor_df = factor_df.join(price_df[g.price_feature_cols], how="left")

    if len(g.temporal_feature_cols) > 0:
        temporal_df = get_temporal_feature_data(stock_list, yesterday, g.temporal_feature_cols)
        factor_df = factor_df.join(temporal_df[g.temporal_feature_cols], how="left")

    if factor_df.empty:
        return []

    score_df = add_model_scores(factor_df)
    score_df = score_df.sort_values("score", ascending=False)
    if g.candidate_pool_size > 0:
        score_df = score_df.head(min(g.candidate_pool_size, len(score_df)))

    sorted_stocks = score_df.index.tolist()
    industry_map = get_industry_bucket_map(sorted_stocks, yesterday)
    target_list = build_industry_neutral_targets(
        sorted_stocks=sorted_stocks,
        industry_map=industry_map,
        target_num=g.stock_num,
        max_per_industry=max(1, int(np.floor(g.stock_num * g.industry_cap_ratio)))
    )

    log.info("target list: %s" % ",".join(target_list))
    log_industry_distribution(target_list, industry_map)
    return target_list


def fetch_jq_factor_data(stock_list, factor_cols, date):
    out = pd.DataFrame(index=stock_list)
    if len(stock_list) == 0 or len(factor_cols) == 0:
        return out

    for factor_chunk in chunks(factor_cols, 20):
        try:
            factor_data = get_factor_values(stock_list, factor_chunk, end_date=date, count=1)
        except Exception as err:
            log.warn("factor chunk fetch failed: %s, err=%s" % (",".join(factor_chunk), err))
            factor_data = None

        for factor in factor_chunk:
            try:
                if factor_data is not None and factor in factor_data:
                    out[factor] = factor_data[factor].iloc[0, :].reindex(stock_list)
                else:
                    one = get_factor_values(stock_list, [factor], end_date=date, count=1)
                    if one is None or factor not in one:
                        out[factor] = np.nan
                    else:
                        out[factor] = one[factor].iloc[0, :].reindex(stock_list)
            except Exception as err:
                log.warn("factor fetch failed: %s, err=%s" % (factor, err))
                out[factor] = np.nan

    return out.reindex(index=stock_list, columns=factor_cols)


def get_price_feature_data(stock_list, date, feature_cols, chunk_size=180):
    out = pd.DataFrame(index=stock_list, columns=feature_cols, dtype=float)
    if len(stock_list) == 0 or len(feature_cols) == 0:
        return out

    for stock_chunk in chunks(stock_list, chunk_size):
        try:
            price_df = get_price(
                stock_chunk,
                end_date=date,
                frequency="daily",
                fields=["close", "money", "paused"],
                count=61,
                skip_paused=False,
                fq="pre",
                panel=False,
                fill_paused=True,
            )
        except Exception as err:
            log.warn("price feature fetch failed, err=%s" % err)
            price_df = None
        if price_df is None or price_df.empty:
            continue

        for col in ["close", "money", "paused"]:
            if col not in price_df.columns:
                price_df[col] = np.nan

        price_df["time"] = pd.to_datetime(price_df["time"]).dt.normalize()
        close_mat = price_df.pivot_table(index="time", columns="code", values="close").sort_index()
        money_mat = price_df.pivot_table(index="time", columns="code", values="money").sort_index()
        paused_mat = price_df.pivot_table(index="time", columns="code", values="paused").sort_index()
        if close_mat.empty:
            continue

        chunk_out = pd.DataFrame(index=stock_chunk, columns=feature_cols, dtype=float)
        last_close = close_mat.iloc[-1]

        if "px_close_to_ma60" in feature_cols:
            ma60 = close_mat.tail(60).mean()
            chunk_out["px_close_to_ma60"] = last_close / ma60 - 1
        if "px_drawdown_60" in feature_cols:
            chunk_out["px_drawdown_60"] = last_close / close_mat.tail(60).max() - 1
        if "liq_money_ratio_20_60" in feature_cols:
            money20 = money_mat.tail(20).mean()
            money60 = money_mat.tail(60).mean()
            chunk_out["liq_money_ratio_20_60"] = money20 / money60 - 1
        if "liq_paused_count_20" in feature_cols:
            chunk_out["liq_paused_count_20"] = paused_mat.tail(20).fillna(0).sum()

        out.loc[chunk_out.index, chunk_out.columns] = chunk_out.replace([np.inf, -np.inf], np.nan)
        del price_df, close_mat, money_mat, paused_mat, chunk_out
        gc.collect()

    return out.reindex(index=stock_list, columns=feature_cols)


def get_month_end_feature_dates(date, months=5):
    trade_days = pd.to_datetime(get_trade_days(end_date=date, count=150))
    if len(trade_days) == 0:
        return []
    month_last = []
    trade_day_series = pd.Series(trade_days)
    for _, gdf in trade_day_series.groupby(trade_day_series.dt.strftime("%Y-%m")):
        month_last.append(gdf.max())
    dates = [d for d in month_last if d <= pd.Timestamp(date)]
    return dates[-months:]


def get_factor_rank_on_date(stock_list, factor, date):
    out = pd.Series(index=stock_list, dtype=float)
    try:
        factor_data = get_factor_values(stock_list, [factor], end_date=date, count=1)
    except Exception as err:
        log.warn("temporal factor fetch failed: %s, err=%s" % (factor, err))
        factor_data = None
    if factor_data is None or factor not in factor_data:
        return out
    try:
        s = factor_data[factor].iloc[0, :].reindex(stock_list)
    except Exception as err:
        log.warn("temporal factor parse failed: %s, err=%s" % (factor, err))
        return out
    return s.rank(pct=True)


def get_temporal_feature_data(stock_list, date, feature_cols):
    out = pd.DataFrame(index=stock_list, columns=feature_cols, dtype=float)
    if len(stock_list) == 0 or len(feature_cols) == 0:
        return out

    feature_dates = get_month_end_feature_dates(date, months=5)
    if len(feature_dates) < 2:
        return out

    need_cf = "ts_cash_flow_to_price_ratio_rank_mean_3m" in feature_cols
    need_rank1m = "ts_Rank1M_rank_chg_1m" in feature_cols

    cf_ranks = []
    rank1m_ranks = []
    for dt in feature_dates:
        dt_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
        if need_cf:
            cf_ranks.append(get_factor_rank_on_date(stock_list, "cash_flow_to_price_ratio", dt_str))
        if need_rank1m:
            rank1m_ranks.append(get_factor_rank_on_date(stock_list, "Rank1M", dt_str))

    if need_cf:
        if len(cf_ranks) >= 4:
            out["ts_cash_flow_to_price_ratio_rank_mean_3m"] = pd.concat(cf_ranks[-4:-1], axis=1).mean(axis=1)
        elif len(cf_ranks) >= 2:
            out["ts_cash_flow_to_price_ratio_rank_mean_3m"] = pd.concat(cf_ranks[:-1], axis=1).mean(axis=1)

    if need_rank1m and len(rank1m_ranks) >= 2:
        out["ts_Rank1M_rank_chg_1m"] = rank1m_ranks[-1] - rank1m_ranks[-2]

    return out.replace([np.inf, -np.inf], np.nan).reindex(index=stock_list, columns=feature_cols)


def add_model_scores(factor_df):
    out = factor_df.copy()
    X = out.reindex(columns=g.feature_cols).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(pd.Series(g.fill_values)).fillna(0)
    out["score"] = np.asarray(g.model.predict(X[g.feature_cols])).reshape(-1)
    return out


def get_industry_bucket_map(stock_list, date):
    if len(stock_list) == 0:
        return {}
    try:
        industry_info = get_industry(stock_list, date=date)
    except Exception:
        return {stock: "UNKNOWN" for stock in stock_list}

    out = {}
    for stock in stock_list:
        info = industry_info.get(stock, {})
        bucket = None
        for key in ["sw_l1", "jq_l1", "zjw"]:
            sub = info.get(key, None)
            if isinstance(sub, dict):
                bucket = sub.get("industry_code") or sub.get("industry_name")
                if bucket:
                    break
        out[stock] = bucket if bucket else "UNKNOWN"
    return out


def build_industry_neutral_targets(sorted_stocks, industry_map, target_num, max_per_industry):
    known_industries = set([
        industry_map.get(stock, "UNKNOWN")
        for stock in sorted_stocks
        if industry_map.get(stock, "UNKNOWN") != "UNKNOWN"
    ])
    if len(known_industries) < 3:
        return sorted_stocks[:min(target_num, len(sorted_stocks))]

    selected = []
    industry_count = {}
    for stock in sorted_stocks:
        industry = industry_map.get(stock, "UNKNOWN")
        if industry == "UNKNOWN":
            continue
        if industry_count.get(industry, 0) == 0:
            selected.append(stock)
            industry_count[industry] = 1
            if len(selected) >= target_num:
                return selected

    for stock in sorted_stocks:
        if stock in selected:
            continue
        industry = industry_map.get(stock, "UNKNOWN")
        if industry == "UNKNOWN":
            continue
        cnt = industry_count.get(industry, 0)
        if cnt < max_per_industry:
            selected.append(stock)
            industry_count[industry] = cnt + 1
            if len(selected) >= target_num:
                return selected

    for stock in sorted_stocks:
        if stock not in selected:
            selected.append(stock)
            if len(selected) >= target_num:
                break
    return selected


def log_industry_distribution(target_list, industry_map):
    if len(target_list) == 0:
        return
    dist = {}
    for stock in target_list:
        industry = industry_map.get(stock, "UNKNOWN")
        dist[industry] = dist.get(industry, 0) + 1
    log.info("industry distribution: %s" % dist)


def monthly_sell(context):
    target_list = get_stock_list(context)
    g.target_list = list(target_list)
    g.target_list_date = context.current_dt.date()

    for stock in g.hold_list:
        if (stock not in target_list) and (stock not in g.yesterday_HL_list):
            log.info("sell [%s]" % stock)
            close_position(context.portfolio.positions[stock])
        else:
            log.info("hold [%s]" % stock)


def monthly_buy(context):
    if g.target_list_date != context.current_dt.date():
        log.warn("target list missing for current rebalance date, recomputing before buy")
        g.target_list = get_stock_list(context)
        g.target_list_date = context.current_dt.date()

    target_list = list(g.target_list)
    current_holds = [
        position.security
        for position in context.portfolio.positions.values()
        if position.total_amount > 0
    ]
    target_num = len(target_list)
    if target_num <= len(current_holds):
        return

    buy_list = [stock for stock in target_list if stock not in current_holds]
    buy_slots = target_num - len(current_holds)
    if buy_slots <= 0 or context.portfolio.cash <= 0:
        return

    value = context.portfolio.cash / buy_slots
    for stock in buy_list:
        if stock in context.portfolio.positions and context.portfolio.positions[stock].total_amount > 0:
            continue
        if open_position(stock, value):
            current_count = len([
                p for p in context.portfolio.positions.values()
                if p.total_amount > 0
            ])
            if current_count >= target_num:
                break


def order_target_value_(security, value):
    style = get_order_style(security, value)
    if style is not None:
        return order_target_value(security, value, style=style)
    return order_target_value(security, value)


def open_position(security, value):
    if value <= 0:
        return False
    last_price = get_last_price(security)
    if last_price is None or last_price <= 0:
        return False
    min_amount = get_min_trade_amount(security)
    if value < last_price * min_amount:
        log.info("[%s] value too small: value=%.2f min_value=%.2f" % (security, value, last_price * min_amount))
        return False
    order = order_target_value_(security, value)
    if order is not None and order.filled > 0:
        return True
    return False


def close_position(position):
    if hasattr(position, "closeable_amount") and position.closeable_amount <= 0:
        return False
    order = order_target_value_(position.security, 0)
    if order is not None:
        if order.status == OrderStatus.held and order.filled == order.amount:
            return True
    return False


def get_order_style(security, target_value):
    if not is_kcb_stock(security):
        return None
    try:
        current_data = get_current_data()
        data = current_data[security]
    except Exception:
        return None

    last_price = get_last_price(security)
    if last_price is None or last_price <= 0:
        return None
    if target_value > 0:
        protect_price = min(last_price * (1 + g.kcb_protect_pct), float(data.high_limit))
    else:
        protect_price = max(last_price * (1 - g.kcb_protect_pct), float(data.low_limit))
    return MarketOrderStyle(limit_price=round(float(protect_price), 2))


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


def is_kcb_stock(stock):
    return stock.startswith(("688", "689"))


def get_min_trade_amount(stock):
    return 200 if is_kcb_stock(stock) else 100


def filter_paused_stock(stock_list):
    if len(stock_list) == 0:
        return []
    current_data = get_current_data()
    return [stock for stock in stock_list if not current_data[stock].paused]


def filter_st_stock(stock_list):
    if len(stock_list) == 0:
        return []
    current_data = get_current_data()
    return [
        stock for stock in stock_list
        if not current_data[stock].is_st
        and "ST" not in current_data[stock].name
        and "*" not in current_data[stock].name
        and "退" not in current_data[stock].name
    ]


def filter_limitup_stock(context, stock_list):
    if len(stock_list) == 0:
        return []
    last_prices = history(1, unit="1m", field="close", security_list=stock_list)
    current_data = get_current_data()
    valid_list = []
    for stock in stock_list:
        if stock in context.portfolio.positions:
            valid_list.append(stock)
            continue
        if stock in last_prices.columns and last_prices[stock][-1] < current_data[stock].high_limit:
            valid_list.append(stock)
    return valid_list


def filter_limitdown_stock(context, stock_list):
    if len(stock_list) == 0:
        return []
    last_prices = history(1, unit="1m", field="close", security_list=stock_list)
    current_data = get_current_data()
    valid_list = []
    for stock in stock_list:
        if stock in context.portfolio.positions:
            valid_list.append(stock)
            continue
        if stock in last_prices.columns and last_prices[stock][-1] > current_data[stock].low_limit:
            valid_list.append(stock)
    return valid_list


def filter_min_lot_stock(context, stock_list):
    if len(stock_list) == 0:
        return []
    target_value = context.portfolio.total_value / max(1, g.stock_num)
    last_prices = history(1, unit="1m", field="close", security_list=stock_list)
    valid_list = []
    for stock in stock_list:
        if stock in context.portfolio.positions:
            valid_list.append(stock)
            continue
        if stock not in last_prices.columns:
            continue
        last_price = float(last_prices[stock][-1])
        min_amount = get_min_trade_amount(stock)
        if target_value * 0.98 >= last_price * min_amount:
            valid_list.append(stock)
    return valid_list


def filter_new_stock(context, stock_list):
    yesterday = context.previous_date
    return [
        stock for stock in stock_list
        if not yesterday - get_security_info(stock).start_date < datetime.timedelta(days=g.min_listing_days)
    ]
