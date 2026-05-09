from jqdata import *
from jqfactor import get_factor_values
import datetime
import gc
import numpy as np
import pandas as pd
import pickle


LEGACY_PRICE_FEATURE_COLS = [
    "px_ret_5",
    "px_ret_20",
    "px_ret_60",
    "px_ret_120",
    "px_close_to_ma20",
    "px_close_to_ma60",
    "px_ma20_to_ma60",
    "px_volatility_20",
    "px_volatility_60",
    "px_drawdown_60",
    "px_drawdown_120",
    "px_money_mean_20",
    "px_money_mean_60",
    "px_money_ratio_20_60",
    "px_volume_ratio_20_60",
    "px_amplitude_20",
    "px_amplitude_60",
    "px_skew_20",
    "px_kurt_20",
]

V4_ONLY_PRICE_FEATURE_COLS = [
    "px_drawdown_20",
    "px_up_day_ratio_20",
    "px_new_high_distance_60",
    "px_new_low_distance_60",
    "liq_money_mean_20",
    "liq_money_mean_60",
    "liq_money_ratio_20_60",
    "liq_volume_mean_20",
    "liq_volume_ratio_20_60",
    "liq_amplitude_mean_20",
    "liq_amplitude_mean_60",
    "liq_paused_count_20",
    "liq_paused_count_60",
    "liq_low_money_days_20",
    "liq_limit_up_count_20",
    "liq_limit_down_count_20",
    "liq_one_price_limit_count_20",
]

PRICE_FEATURE_COLS = LEGACY_PRICE_FEATURE_COLS + V4_ONLY_PRICE_FEATURE_COLS

TEMPORAL_FEATURE_COLS = [
    "ts_cash_flow_to_price_ratio_rank_mean_3m",
    "ts_Rank1M_rank_chg_1m",
]

INDUSTRY_RELATIVE_SOURCE_FACTORS = [
    "book_to_price_ratio",
    "earnings_yield",
    "cash_flow_to_price_ratio",
    "roe_ttm",
    "roa_ttm",
    "Rank1M",
    "sharpe_ratio_60",
]

INDUSTRY_RELATIVE_FEATURE_COLS = []
for _fac in INDUSTRY_RELATIVE_SOURCE_FACTORS:
    INDUSTRY_RELATIVE_FEATURE_COLS.append("v45_{}_minus_industry_median".format(_fac))
    INDUSTRY_RELATIVE_FEATURE_COLS.append("v45_{}_rank_in_industry".format(_fac))


def initialize(context):
    # 改这里即可逐个测试候选模型 pkl。
    # 核心观察模型 3：V410 fixed_iter20 rolling5y，训练到 2025Q4。
    # 用作更“新”、更干净的固定迭代单模型观察线。
    g.model_file = "model_candidate_v410_fixed_iter20_rolling5y_l2_ff10_2025q4.pkl"

    bundle = pickle.loads(read_file(g.model_file))
    if not isinstance(bundle, dict) or bundle.get("objective") != "v210_refit_fixed_iter_overlay":
        raise ValueError("model bundle should be v210_refit_fixed_iter_overlay")

    g.overlay_mode = bundle.get("overlay_mode", "top30_rerank")
    direct_mode = g.overlay_mode == "direct"

    required_keys = [
        "base_model",
        "base_feature_cols",
        "base_fill_values",
        "overlay_weight",
    ]
    if not direct_mode:
        required_keys.extend([
            "residual_model",
            "residual_feature_cols",
            "residual_fill_values",
        ])
    for key in required_keys:
        if key not in bundle:
            raise ValueError("model bundle missing key: {}".format(key))

    g.base_model = bundle["base_model"]
    g.base_feature_cols = list(bundle["base_feature_cols"])
    g.base_fill_values = dict(bundle.get("base_fill_values", {}))
    g.residual_model = bundle.get("residual_model", None)
    g.residual_feature_cols = list(bundle["residual_feature_cols"])
    g.residual_fill_values = dict(bundle.get("residual_fill_values", {}))

    g.overlay_weight = float(bundle.get("overlay_weight", 0.15))
    g.top_n_candidates = int(bundle.get("top_n_candidates", 30))

    g.feature_cols = unique_keep_order(g.base_feature_cols + g.residual_feature_cols)
    g.price_feature_cols = [col for col in g.feature_cols if col in PRICE_FEATURE_COLS]
    g.temporal_feature_cols = [col for col in g.feature_cols if col in TEMPORAL_FEATURE_COLS]
    g.industry_relative_feature_cols = [col for col in g.feature_cols if col in INDUSTRY_RELATIVE_FEATURE_COLS]
    g.industry_relative_source_cols = get_industry_relative_source_cols(g.industry_relative_feature_cols)
    g.jq_factor_cols = [
        col for col in g.feature_cols
        if (
            col not in PRICE_FEATURE_COLS
            and col not in TEMPORAL_FEATURE_COLS
            and col not in INDUSTRY_RELATIVE_FEATURE_COLS
        )
    ]
    g.jq_factor_cols = unique_keep_order(g.jq_factor_cols + g.industry_relative_source_cols)

    has_v4_only_cols = any(col in V4_ONLY_PRICE_FEATURE_COLS for col in g.price_feature_cols)
    has_temporal_cols = len(g.temporal_feature_cols) > 0
    has_industry_relative_cols = len(g.industry_relative_feature_cols) > 0
    g.requires_v4_feature_adapter = (
        bool(bundle.get("requires_v4_feature_adapter", False))
        or has_v4_only_cols
        or has_temporal_cols
        or has_industry_relative_cols
    )

    g.universe_index = "000906.XSHG"
    g.benchmark = bundle.get("benchmark", "000906.XSHG")
    g.stock_num = int(bundle.get("stock_num", 10))
    g.min_listing_days = 180
    g.industry_cap_ratio = float(bundle.get("industry_cap_ratio", 0.20))
    g.kcb_protect_pct = 0.02
    g.stop_loss_pct = None
    g.take_profit_pct = None
    # 纯月调基准：默认不做盘中开板卖出，避免把交易规则变化混入模型比较。
    g.enable_limit_up_sell = False

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

    model_version = bundle.get("research_version", "v210_refit_fixed_iter_overlay")
    price_mode = "v4_fill_paused" if g.requires_v4_feature_adapter else "legacy_skip_paused"
    log.info(
        "loaded candidate pure-monthly model %s: base_features=%s residual_features=%s "
        "weight=%.2f mode=%s candidates=%s limit_up_sell=%s feature_adapter=%s price_mode=%s" % (
            model_version,
            len(g.base_feature_cols),
            len(g.residual_feature_cols),
            g.overlay_weight,
            g.overlay_mode,
            g.top_n_candidates,
            g.enable_limit_up_sell,
            g.requires_v4_feature_adapter,
            price_mode,
        )
    )

    run_daily(prepare_stock_list, "9:05")
    run_monthly(monthly_sell, 1, "9:40")
    run_monthly(monthly_buy, 1, "9:50")
    if g.enable_limit_up_sell:
        run_daily(check_limit_up, "14:00")
    run_daily(check_stop_rules, "14:20")


def unique_keep_order(cols):
    seen = set()
    out = []
    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def get_industry_relative_source_cols(feature_cols):
    out = []
    for col in feature_cols:
        for fac in INDUSTRY_RELATIVE_SOURCE_FACTORS:
            if col in [
                "v45_{}_minus_industry_median".format(fac),
                "v45_{}_rank_in_industry".format(fac),
            ]:
                out.append(fac)
    return unique_keep_order(out)


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


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
                    one_factor = factor_data[factor].iloc[0, :]
                    out[factor] = one_factor.reindex(stock_list)
                else:
                    one = get_factor_values(stock_list, [factor], end_date=date, count=1)
                    if one is None or factor not in one:
                        out[factor] = np.nan
                        continue
                    out[factor] = one[factor].iloc[0, :].reindex(stock_list)
            except Exception as err:
                log.warn("factor fetch failed: %s, err=%s" % (factor, err))
                out[factor] = np.nan

    return out.reindex(index=stock_list, columns=factor_cols)


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

    df_factor = pd.DataFrame(index=stock_list)
    if len(g.jq_factor_cols) > 0:
        df_factor = df_factor.join(fetch_jq_factor_data(stock_list, g.jq_factor_cols, yesterday), how="left")

    if len(g.price_feature_cols) > 0:
        price_feature_df = get_price_feature_data(stock_list, yesterday)
        df_factor = df_factor.join(price_feature_df[g.price_feature_cols], how="left")

    if len(g.temporal_feature_cols) > 0:
        temporal_feature_df = get_temporal_feature_data(stock_list, yesterday)
        df_factor = df_factor.join(temporal_feature_df[g.temporal_feature_cols], how="left")

    if len(g.industry_relative_feature_cols) > 0:
        industry_map_full = get_industry_bucket_map(stock_list, yesterday)
        industry_relative_df = get_industry_relative_feature_data(
            df_factor,
            industry_map_full,
            g.industry_relative_feature_cols
        )
        df_factor = df_factor.join(industry_relative_df[g.industry_relative_feature_cols], how="left")

    if df_factor.empty:
        return []

    df_factor = add_model_scores(df_factor)
    if g.overlay_mode in ["top30_rerank", "direct"]:
        candidate_count = min(g.top_n_candidates, len(df_factor))
        candidate_df = df_factor.nlargest(candidate_count, "base_score_z")
        sorted_stocks = candidate_df.sort_values("final_score", ascending=False).index.tolist()
        industry_map = get_industry_bucket_map(sorted_stocks, yesterday)
    else:
        sorted_stocks = df_factor.sort_values("final_score", ascending=False).index.tolist()
        industry_map = get_industry_bucket_map(sorted_stocks, yesterday)

    target_list = build_industry_neutral_targets(
        sorted_stocks=sorted_stocks,
        industry_map=industry_map,
        target_num=g.stock_num,
        max_per_industry=max(1, int(np.floor(g.stock_num * g.industry_cap_ratio)))
    )

    log.info("V2.10 weight=%.2f mode=%s candidates=%s" % (g.overlay_weight, g.overlay_mode, g.top_n_candidates))
    log.info("target list: {}".format(",".join(target_list)))
    log_industry_distribution(target_list, industry_map)
    return target_list


def add_model_scores(df_factor):
    out = df_factor.copy()

    base_X = out.reindex(columns=g.base_feature_cols).replace([np.inf, -np.inf], np.nan)
    base_X = base_X.fillna(pd.Series(g.base_fill_values)).fillna(0)
    out["base_score"] = np.asarray(g.base_model.predict(base_X[g.base_feature_cols])).reshape(-1)
    out["base_score_z"] = zscore_series(out["base_score"])

    residual_X = out.reindex(columns=g.residual_feature_cols).replace([np.inf, -np.inf], np.nan)
    if g.overlay_mode == "direct" or g.residual_model is None or len(g.residual_feature_cols) == 0:
        out["residual_score"] = 0.0
        out["residual_score_z"] = 0.0
        out["final_score"] = out["base_score_z"]
    else:
        residual_X = residual_X.fillna(pd.Series(g.residual_fill_values)).fillna(0)
        out["residual_score"] = np.asarray(g.residual_model.predict(residual_X[g.residual_feature_cols])).reshape(-1)
        out["residual_score_z"] = zscore_series(out["residual_score"])
        out["final_score"] = out["base_score_z"] + g.overlay_weight * out["residual_score_z"]
    return out


def zscore_series(s):
    s = pd.Series(s).astype(float)
    std = s.std()
    if pd.isnull(std) or std <= 0:
        return s * 0.0
    return (s - s.mean()) / std


def calc_ret(close_mat, days):
    if close_mat is None or close_mat.empty or len(close_mat) <= days:
        return pd.Series(dtype=float)
    return close_mat.iloc[-1] / close_mat.iloc[-days - 1] - 1


def calc_up_day_ratio(ret_mat, days):
    if ret_mat is None or ret_mat.empty:
        return pd.Series(dtype=float)
    return (ret_mat.tail(days) > 0).mean()


def calc_new_low_distance(close_mat, days):
    if close_mat is None or close_mat.empty:
        return pd.Series(dtype=float)
    last_close = close_mat.iloc[-1]
    min_close = close_mat.tail(days).min()
    return last_close / min_close - 1


def get_price_feature_data(stock_list, date):
    if g.requires_v4_feature_adapter:
        return get_v4_price_feature_data(stock_list, date)
    return get_legacy_price_feature_data(stock_list, date)


def get_legacy_price_feature_data(stock_list, date):
    out = pd.DataFrame(index=stock_list, columns=PRICE_FEATURE_COLS, dtype=float)
    if len(stock_list) == 0:
        return out

    price_df = get_price(
        stock_list,
        end_date=date,
        frequency="daily",
        fields=["close", "high", "low", "volume", "money"],
        count=121,
        skip_paused=True,
        fq="pre",
        panel=False
    )
    if price_df is None or price_df.empty:
        return out

    price_df["time"] = pd.to_datetime(price_df["time"]).dt.normalize()
    close_mat = price_df.pivot_table(index="time", columns="code", values="close").sort_index()
    high_mat = price_df.pivot_table(index="time", columns="code", values="high").sort_index()
    low_mat = price_df.pivot_table(index="time", columns="code", values="low").sort_index()
    volume_mat = price_df.pivot_table(index="time", columns="code", values="volume").sort_index()
    money_mat = price_df.pivot_table(index="time", columns="code", values="money").sort_index()
    if close_mat.empty:
        return out

    ret_mat = close_mat.pct_change()
    last_close = close_mat.iloc[-1]
    ma20 = close_mat.tail(20).mean()
    ma60 = close_mat.tail(60).mean()

    out["px_ret_5"] = calc_ret(close_mat, 5)
    out["px_ret_20"] = calc_ret(close_mat, 20)
    out["px_ret_60"] = calc_ret(close_mat, 60)
    out["px_ret_120"] = calc_ret(close_mat, 120)
    out["px_close_to_ma20"] = last_close / ma20 - 1
    out["px_close_to_ma60"] = last_close / ma60 - 1
    out["px_ma20_to_ma60"] = ma20 / ma60 - 1
    out["px_volatility_20"] = ret_mat.tail(20).std()
    out["px_volatility_60"] = ret_mat.tail(60).std()
    out["px_drawdown_60"] = last_close / close_mat.tail(60).max() - 1
    out["px_drawdown_120"] = last_close / close_mat.tail(120).max() - 1
    out["px_money_mean_20"] = money_mat.tail(20).mean()
    out["px_money_mean_60"] = money_mat.tail(60).mean()
    out["px_money_ratio_20_60"] = money_mat.tail(20).mean() / money_mat.tail(60).mean() - 1
    out["px_volume_ratio_20_60"] = volume_mat.tail(20).mean() / volume_mat.tail(60).mean() - 1
    out["px_amplitude_20"] = (high_mat.tail(20) / low_mat.tail(20) - 1).mean()
    out["px_amplitude_60"] = (high_mat.tail(60) / low_mat.tail(60) - 1).mean()
    out["px_skew_20"] = ret_mat.tail(20).skew()
    out["px_kurt_20"] = ret_mat.tail(20).kurt()

    out = out.replace([np.inf, -np.inf], np.nan)
    return out.reindex(index=stock_list, columns=PRICE_FEATURE_COLS)


def get_v4_price_feature_data(stock_list, date, chunk_size=160):
    out = pd.DataFrame(index=stock_list, columns=PRICE_FEATURE_COLS, dtype=float)
    if len(stock_list) == 0:
        return out

    for stock_chunk in chunks(stock_list, chunk_size):
        try:
            price_df = get_price(
                stock_chunk,
                end_date=date,
                frequency="daily",
                fields=["close", "high", "low", "volume", "money", "paused", "high_limit", "low_limit"],
                count=121,
                skip_paused=False,
                fq="pre",
                panel=False,
                fill_paused=True,
            )
        except Exception as err:
            log.warn("v4 price feature fetch failed, err=%s" % err)
            price_df = None
        if price_df is None or price_df.empty:
            continue

        for col in ["close", "high", "low", "volume", "money", "paused", "high_limit", "low_limit"]:
            if col not in price_df.columns:
                price_df[col] = np.nan
        price_df["time"] = pd.to_datetime(price_df["time"]).dt.normalize()
        close_mat = price_df.pivot_table(index="time", columns="code", values="close").sort_index()
        high_mat = price_df.pivot_table(index="time", columns="code", values="high").sort_index()
        low_mat = price_df.pivot_table(index="time", columns="code", values="low").sort_index()
        volume_mat = price_df.pivot_table(index="time", columns="code", values="volume").sort_index()
        money_mat = price_df.pivot_table(index="time", columns="code", values="money").sort_index()
        paused_mat = price_df.pivot_table(index="time", columns="code", values="paused").sort_index()
        high_limit_mat = price_df.pivot_table(index="time", columns="code", values="high_limit").sort_index()
        low_limit_mat = price_df.pivot_table(index="time", columns="code", values="low_limit").sort_index()
        if close_mat.empty:
            continue

        ret_mat = close_mat.pct_change()
        last_close = close_mat.iloc[-1]
        ma20 = close_mat.tail(20).mean()
        ma60 = close_mat.tail(60).mean()
        money20 = money_mat.tail(20).mean()
        money60 = money_mat.tail(60).mean()
        volume20 = volume_mat.tail(20).mean()
        volume60 = volume_mat.tail(60).mean()

        chunk_out = pd.DataFrame(index=stock_chunk, columns=PRICE_FEATURE_COLS, dtype=float)
        chunk_out["px_ret_5"] = calc_ret(close_mat, 5)
        chunk_out["px_ret_20"] = calc_ret(close_mat, 20)
        chunk_out["px_ret_60"] = calc_ret(close_mat, 60)
        chunk_out["px_ret_120"] = calc_ret(close_mat, 120)
        chunk_out["px_close_to_ma20"] = last_close / ma20 - 1
        chunk_out["px_close_to_ma60"] = last_close / ma60 - 1
        chunk_out["px_ma20_to_ma60"] = ma20 / ma60 - 1
        chunk_out["px_volatility_20"] = ret_mat.tail(20).std()
        chunk_out["px_volatility_60"] = ret_mat.tail(60).std()
        chunk_out["px_drawdown_20"] = last_close / close_mat.tail(20).max() - 1
        chunk_out["px_drawdown_60"] = last_close / close_mat.tail(60).max() - 1
        chunk_out["px_drawdown_120"] = last_close / close_mat.tail(120).max() - 1
        chunk_out["px_up_day_ratio_20"] = calc_up_day_ratio(ret_mat, 20)
        chunk_out["px_new_high_distance_60"] = last_close / close_mat.tail(60).max() - 1
        chunk_out["px_new_low_distance_60"] = calc_new_low_distance(close_mat, 60)
        chunk_out["px_skew_20"] = ret_mat.tail(20).skew()
        chunk_out["px_kurt_20"] = ret_mat.tail(20).kurt()

        chunk_out["px_money_mean_20"] = money20
        chunk_out["px_money_mean_60"] = money60
        chunk_out["px_money_ratio_20_60"] = money20 / money60 - 1
        chunk_out["px_volume_ratio_20_60"] = volume20 / volume60 - 1
        chunk_out["px_amplitude_20"] = (high_mat.tail(20) / low_mat.tail(20) - 1).mean()
        chunk_out["px_amplitude_60"] = (high_mat.tail(60) / low_mat.tail(60) - 1).mean()

        chunk_out["liq_money_mean_20"] = money20
        chunk_out["liq_money_mean_60"] = money60
        chunk_out["liq_money_ratio_20_60"] = money20 / money60 - 1
        chunk_out["liq_volume_mean_20"] = volume20
        chunk_out["liq_volume_ratio_20_60"] = volume20 / volume60 - 1
        chunk_out["liq_amplitude_mean_20"] = (high_mat.tail(20) / low_mat.tail(20) - 1).mean()
        chunk_out["liq_amplitude_mean_60"] = (high_mat.tail(60) / low_mat.tail(60) - 1).mean()
        chunk_out["liq_paused_count_20"] = paused_mat.tail(20).fillna(0).sum()
        chunk_out["liq_paused_count_60"] = paused_mat.tail(60).fillna(0).sum()

        money_stack = money_mat.stack().dropna()
        money_q20 = money_stack.quantile(0.20) if len(money_stack) else np.nan
        chunk_out["liq_low_money_days_20"] = (money_mat.tail(20) < money_q20).sum() if not pd.isnull(money_q20) else np.nan
        limit_up = close_mat >= (high_limit_mat * 0.999)
        limit_down = close_mat <= (low_limit_mat * 1.001)
        one_price = (high_mat <= low_mat * 1.0001) & (limit_up | limit_down)
        chunk_out["liq_limit_up_count_20"] = limit_up.tail(20).sum()
        chunk_out["liq_limit_down_count_20"] = limit_down.tail(20).sum()
        chunk_out["liq_one_price_limit_count_20"] = one_price.tail(20).sum()

        out.loc[chunk_out.index, chunk_out.columns] = chunk_out.replace([np.inf, -np.inf], np.nan)
        del price_df, close_mat, high_mat, low_mat, volume_mat, money_mat, paused_mat
        del high_limit_mat, low_limit_mat, ret_mat, chunk_out
        gc.collect()

    out = out.replace([np.inf, -np.inf], np.nan)
    return out.reindex(index=stock_list, columns=PRICE_FEATURE_COLS)


def get_month_end_feature_dates(date, months=5):
    trade_days = pd.to_datetime(get_trade_days(end_date=date, count=150))
    if len(trade_days) == 0:
        return []
    month_last = []
    for _, gdf in pd.Series(trade_days).groupby(trade_days.strftime("%Y-%m")):
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


def get_temporal_feature_data(stock_list, date):
    out = pd.DataFrame(index=stock_list, columns=TEMPORAL_FEATURE_COLS, dtype=float)
    if len(stock_list) == 0:
        return out

    feature_dates = get_month_end_feature_dates(date, months=5)
    if len(feature_dates) < 2:
        return out

    cf_ranks = []
    rank1m_ranks = []
    for dt in feature_dates:
        dt_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
        cf_ranks.append(get_factor_rank_on_date(stock_list, "cash_flow_to_price_ratio", dt_str))
        rank1m_ranks.append(get_factor_rank_on_date(stock_list, "Rank1M", dt_str))

    if len(cf_ranks) >= 4:
        out["ts_cash_flow_to_price_ratio_rank_mean_3m"] = pd.concat(cf_ranks[-4:-1], axis=1).mean(axis=1)
    elif len(cf_ranks) >= 2:
        out["ts_cash_flow_to_price_ratio_rank_mean_3m"] = pd.concat(cf_ranks[:-1], axis=1).mean(axis=1)

    if len(rank1m_ranks) >= 2:
        out["ts_Rank1M_rank_chg_1m"] = rank1m_ranks[-1] - rank1m_ranks[-2]

    return out.replace([np.inf, -np.inf], np.nan).reindex(index=stock_list, columns=TEMPORAL_FEATURE_COLS)


def get_industry_relative_feature_data(factor_df, industry_map, feature_cols):
    out = pd.DataFrame(index=factor_df.index, columns=feature_cols, dtype=float)
    if factor_df.empty or len(feature_cols) == 0:
        return out

    tmp = factor_df.copy()
    tmp["industry_bucket"] = pd.Series(industry_map).reindex(tmp.index).fillna("UNKNOWN").astype(str)

    for fac in INDUSTRY_RELATIVE_SOURCE_FACTORS:
        if fac not in tmp.columns:
            continue
        median_col = "v45_{}_minus_industry_median".format(fac)
        rank_col = "v45_{}_rank_in_industry".format(fac)
        if median_col in feature_cols:
            out[median_col] = tmp[fac] - tmp.groupby("industry_bucket")[fac].transform("median")
        if rank_col in feature_cols:
            out[rank_col] = tmp.groupby("industry_bucket")[fac].transform(lambda s: s.rank(pct=True))

    return out.replace([np.inf, -np.inf], np.nan).reindex(index=factor_df.index, columns=feature_cols)


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
    log.info("industry distribution: {}".format(dist))


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


def check_stop_rules(context):
    if g.stop_loss_pct is None and g.take_profit_pct is None:
        return
    if len(g.hold_list) == 0:
        return

    quote_df = get_price(
        g.hold_list,
        end_date=context.current_dt,
        frequency="1m",
        fields=["close", "high_limit", "low_limit"],
        skip_paused=False,
        fq="pre",
        count=1,
        panel=False,
        fill_paused=True
    )
    if quote_df is None or quote_df.empty:
        return

    latest_map = {}
    for _, row in quote_df.iterrows():
        latest_map[row["code"]] = {
            "close": float(row["close"]),
            "high_limit": float(row["high_limit"]),
            "low_limit": float(row["low_limit"])
        }

    for stock in list(g.hold_list):
        if stock not in context.portfolio.positions or stock not in latest_map:
            continue
        position = context.portfolio.positions[stock]
        avg_cost = float(position.avg_cost) if position.avg_cost is not None else 0.0
        if avg_cost <= 0:
            continue
        last_price = latest_map[stock]["close"]
        high_limit = latest_map[stock]["high_limit"]
        low_limit = latest_map[stock]["low_limit"]
        pnl_pct = last_price / avg_cost - 1
        if last_price <= low_limit:
            continue
        if (g.stop_loss_pct is not None) and (pnl_pct <= g.stop_loss_pct):
            close_position(position)
            continue
        if (g.take_profit_pct is not None) and (pnl_pct >= g.take_profit_pct) and last_price < high_limit:
            close_position(position)


def check_limit_up(context):
    if len(g.yesterday_HL_list) == 0:
        return
    for stock in g.yesterday_HL_list:
        if stock not in context.portfolio.positions:
            continue
        current_data = get_price(
            stock,
            end_date=context.current_dt,
            frequency="1m",
            fields=["close", "high_limit"],
            skip_paused=False,
            fq="pre",
            count=1,
            panel=False,
            fill_paused=True
        )
        if current_data is None or current_data.empty:
            continue
        if float(current_data.iloc[0]["close"]) < float(current_data.iloc[0]["high_limit"]):
            close_position(context.portfolio.positions[stock])


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
