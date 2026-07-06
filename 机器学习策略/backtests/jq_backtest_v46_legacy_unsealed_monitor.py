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
    # 旧 V2.10 严格复现可用：model_csi800_lgb_factor_v210_refit_fixed_iter.pkl
    g.model_file = "test/model_candidate_v46_lgb_direct_hybrid_l2_ff10_2019_2025q1_legacy_unsealed_q4.pkl"

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
    g.stock_num = 6
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
    g.enable_model_monitor = True
    g.monitor_top_ns = [g.stock_num, 10, 20, g.top_n_candidates]
    g.monitor_recall_top_ns = [g.stock_num, 10, 20, g.top_n_candidates]
    g.monitor_bottom_top_ns = [g.stock_num, 10, 30]
    g.monitor_last_snapshot = None
    g.monitor_last_eval_key = None

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
    if g.enable_model_monitor:
        log_previous_model_monitor(context, yesterday)

    stock_list = get_index_stocks(g.universe_index, yesterday)
    raw_count = len(stock_list)
    if len(stock_list) == 0:
        return []

    stock_list = filter_st_stock(stock_list)
    after_st_count = len(stock_list)
    stock_list = filter_paused_stock(stock_list)
    after_paused_count = len(stock_list)
    stock_list = filter_new_stock(context, stock_list)
    after_new_count = len(stock_list)
    stock_list = filter_limitup_stock(context, stock_list)
    after_limitup_count = len(stock_list)
    stock_list = filter_limitdown_stock(context, stock_list)
    after_limitdown_count = len(stock_list)
    stock_list = filter_min_lot_stock(context, stock_list)
    after_min_lot_count = len(stock_list)
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
    log.info(
        "monitor filter counts raw=%s st=%s paused=%s new=%s limitup=%s limitdown=%s minlot=%s" % (
            raw_count,
            after_st_count,
            after_paused_count,
            after_new_count,
            after_limitup_count,
            after_limitdown_count,
            after_min_lot_count,
        )
    )
    log.info("target list: {}".format(",".join(target_list)))
    log_industry_distribution(target_list, industry_map)
    if g.enable_model_monitor:
        log_current_model_monitor(
            context=context,
            score_df=df_factor,
            sorted_stocks=sorted_stocks,
            target_list=target_list,
            industry_map=industry_map,
            feature_date=yesterday,
        )
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


def infer_board(stock):
    if stock.startswith(("688", "689")):
        return "STAR"
    if stock.startswith(("300", "301")):
        return "CHINEXT"
    if stock.startswith(("8", "4", "43", "87", "92")):
        return "BSE"
    return "MAIN"


def top_items_text(counter, max_items=5):
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    return ",".join(["{}:{}".format(k, v) for k, v in items[:max_items]])


def calc_code_exposure(stocks, industry_map):
    board_count = {}
    industry_count = {}
    for stock in stocks:
        board = infer_board(stock)
        board_count[board] = board_count.get(board, 0) + 1
        industry = industry_map.get(stock, "UNKNOWN")
        industry_count[industry] = industry_count.get(industry, 0) + 1
    return board_count, industry_count


def log_code_exposure(label, stocks, industry_map):
    if len(stocks) == 0:
        return
    board_count, industry_count = calc_code_exposure(stocks, industry_map)
    log.info(
        "MONITOR exposure %s count=%s board=%s industry_top=%s" % (
            label,
            len(stocks),
            top_items_text(board_count),
            top_items_text(industry_count),
        )
    )


def mean_or_nan(values):
    s = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return np.nan
    return float(s.mean())


def get_score_stats(score_df, stocks):
    cols = ["base_score_z", "final_score", "residual_score_z"]
    out = {}
    if score_df is None or score_df.empty or len(stocks) == 0:
        return out
    sub = score_df.reindex(stocks)
    for col in cols:
        if col in sub.columns:
            s = pd.to_numeric(sub[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) > 0:
                out[col + "_mean"] = float(s.mean())
                out[col + "_min"] = float(s.min())
                out[col + "_max"] = float(s.max())
    return out


def score_stats_text(stats):
    keys = sorted(stats.keys())
    return " ".join(["{}={:.4f}".format(k, stats[k]) for k in keys])


def fmt_pct(x):
    if pd.isnull(x):
        return "nan"
    return "{:.1f}%".format(float(x) * 100.0)


def fmt_num(x, digits=3):
    if pd.isnull(x):
        return "nan"
    return ("{:." + str(digits) + "f}").format(float(x))


def fmt_bps(x):
    if pd.isnull(x):
        return "nan"
    return "{:.1f}bp".format(float(x) * 10000.0)


def calc_missing_feature_stats(score_df, stocks):
    if score_df is None or score_df.empty or len(stocks) == 0:
        return {}
    cols = [c for c in g.feature_cols if c in score_df.columns]
    if len(cols) == 0:
        return {}
    sub = score_df.reindex(stocks)[cols]
    miss_by_row = sub.isnull().sum(axis=1)
    return {
        "feature_cols": len(cols),
        "avg_missing": float(miss_by_row.mean()),
        "max_missing": float(miss_by_row.max()),
    }


def log_missing_feature_stats(label, score_df, stocks):
    stats = calc_missing_feature_stats(score_df, stocks)
    if not stats:
        return
    log.info(
        "MONITOR missing %s features=%s avg_missing=%.2f max_missing=%.0f" % (
            label,
            stats["feature_cols"],
            stats["avg_missing"],
            stats["max_missing"],
        )
    )


def log_current_model_monitor(context, score_df, sorted_stocks, target_list, industry_map, feature_date):
    if score_df is None or score_df.empty or len(sorted_stocks) == 0:
        return

    current_holds = [
        position.security
        for position in context.portfolio.positions.values()
        if position.total_amount > 0
    ]
    overlap_count = len(set(target_list).intersection(set(current_holds)))
    target_turnover = 1.0 - float(overlap_count) / float(max(1, len(target_list)))
    log.info("")
    log.info("========== MODEL MONITOR SELECT %s ==========" % context.current_dt.date())
    log.info(
        "MONITOR SELECT | rebalance=%s | feature=%s | stock_num=%s | hold_overlap=%s/%s | est_turnover=%s" % (
            context.current_dt.date(),
            feature_date,
            g.stock_num,
            overlap_count,
            len(target_list),
            fmt_pct(target_turnover),
        )
    )

    monitor_groups = {
        "target_top{}".format(g.stock_num): list(target_list),
        "raw_top{}".format(g.stock_num): list(sorted_stocks[:g.stock_num]),
    }
    for n in g.monitor_top_ns:
        n = int(n)
        if n <= 0:
            continue
        label = "raw_top{}".format(n)
        if label not in monitor_groups:
            monitor_groups[label] = list(sorted_stocks[:min(n, len(sorted_stocks))])

    for label in sorted(monitor_groups.keys()):
        stocks = monitor_groups[label]
        stats = get_score_stats(score_df, stocks)
        log.info("MONITOR score %s %s" % (label, score_stats_text(stats)))
        log_code_exposure(label, stocks, industry_map)
        log_missing_feature_stats(label, score_df, stocks)

    g.monitor_last_snapshot = {
        "rebalance_date": context.current_dt.date(),
        "feature_date": feature_date,
        "benchmark": g.benchmark,
        "universe": list(score_df.index),
        "groups": monitor_groups,
    }
    g.monitor_last_eval_key = None


def calc_equal_weight_close_monitor(stocks, start_date, end_date):
    result = {
        "ret": np.nan,
        "mdd": np.nan,
        "win_rate": np.nan,
        "valid_count": 0,
    }
    stocks = list(stocks)
    if len(stocks) == 0:
        return result
    try:
        price_df = get_price(
            stocks,
            start_date=start_date,
            end_date=end_date,
            frequency="daily",
            fields=["close"],
            skip_paused=False,
            fq="pre",
            panel=False,
            fill_paused=True,
        )
    except Exception as err:
        log.warn("monitor price fetch failed: %s" % err)
        return result
    if price_df is None or price_df.empty:
        return result
    if "time" not in price_df.columns or "code" not in price_df.columns or "close" not in price_df.columns:
        return result
    price_df["time"] = pd.to_datetime(price_df["time"]).dt.normalize()
    close_mat = price_df.pivot_table(index="time", columns="code", values="close").sort_index()
    if close_mat.empty or len(close_mat) < 2:
        return result
    first = close_mat.iloc[0].replace(0, np.nan)
    last = close_mat.iloc[-1]
    stock_rets = (last / first - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
    if len(stock_rets) == 0:
        return result
    norm = close_mat.reindex(columns=stock_rets.index) / first.reindex(stock_rets.index)
    curve = norm.mean(axis=1).dropna()
    if len(curve) > 0:
        drawdown = curve / curve.cummax() - 1.0
        result["mdd"] = float(drawdown.min())
    result["ret"] = float(stock_rets.mean())
    result["win_rate"] = float((stock_rets > 0).mean())
    result["valid_count"] = int(len(stock_rets))
    return result


def calc_benchmark_close_return(benchmark, start_date, end_date):
    try:
        bench_df = get_price(
            benchmark,
            start_date=start_date,
            end_date=end_date,
            frequency="daily",
            fields=["close"],
            skip_paused=False,
            fq="pre",
            panel=False,
            fill_paused=True,
        )
    except Exception as err:
        log.warn("monitor benchmark fetch failed: %s" % err)
        return np.nan
    if bench_df is None or bench_df.empty or "close" not in bench_df.columns or len(bench_df) < 2:
        return np.nan
    close = pd.to_numeric(bench_df["close"], errors="coerce").dropna()
    if len(close) < 2 or close.iloc[0] == 0:
        return np.nan
    return float(close.iloc[-1] / close.iloc[0] - 1.0)


def calc_stock_close_returns(stocks, start_date, end_date, chunk_size=200):
    stocks = unique_keep_order(list(stocks))
    if len(stocks) == 0:
        return pd.Series(dtype=float)
    parts = []
    for stock_chunk in chunks(stocks, chunk_size):
        try:
            price_df = get_price(
                stock_chunk,
                start_date=start_date,
                end_date=end_date,
                frequency="daily",
                fields=["close"],
                skip_paused=False,
                fq="pre",
                panel=False,
                fill_paused=True,
            )
        except Exception as err:
            log.warn("monitor recall price fetch failed: %s" % err)
            price_df = None
        if price_df is not None and not price_df.empty:
            parts.append(price_df)
    if len(parts) == 0:
        return pd.Series(dtype=float)
    price_df = pd.concat(parts, axis=0)
    if "time" not in price_df.columns or "code" not in price_df.columns or "close" not in price_df.columns:
        return pd.Series(dtype=float)
    price_df["time"] = pd.to_datetime(price_df["time"]).dt.normalize()
    close_mat = price_df.pivot_table(index="time", columns="code", values="close").sort_index()
    if close_mat.empty or len(close_mat) < 2:
        return pd.Series(dtype=float)
    first = close_mat.iloc[0].replace(0, np.nan)
    last = close_mat.iloc[-1]
    rets = (last / first - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
    return rets.sort_values(ascending=False)


def calc_selected_return_quality(stocks, universe_returns):
    result = {
        "valid": 0,
        "avg_ret": np.nan,
        "median_ret": np.nan,
        "worst_ret": np.nan,
        "avg_pct": np.nan,
        "median_pct": np.nan,
        "worst_pct": np.nan,
    }
    if universe_returns is None or len(universe_returns) == 0 or len(stocks) == 0:
        return result
    selected = universe_returns.reindex(stocks).replace([np.inf, -np.inf], np.nan).dropna()
    if len(selected) == 0:
        return result
    pct_rank = universe_returns.rank(pct=True)
    selected_pct = pct_rank.reindex(selected.index).replace([np.inf, -np.inf], np.nan).dropna()
    result["valid"] = int(len(selected))
    result["avg_ret"] = float(selected.mean())
    result["median_ret"] = float(selected.median())
    result["worst_ret"] = float(selected.min())
    if len(selected_pct) > 0:
        result["avg_pct"] = float(selected_pct.mean())
        result["median_pct"] = float(selected_pct.median())
        result["worst_pct"] = float(selected_pct.min())
    return result


def bottom_hit_text(stocks, universe_returns):
    if universe_returns is None or len(universe_returns) == 0 or len(stocks) == 0:
        return ""
    valid_stocks = [stock for stock in stocks if stock in universe_returns.index]
    if len(valid_stocks) == 0:
        return ""
    parts = []
    universe_size = len(universe_returns)
    for k in g.monitor_bottom_top_ns:
        k = int(k)
        if k <= 0:
            continue
        bottom_stocks = set(list(universe_returns.index[-min(k, universe_size):]))
        hit = len([stock for stock in valid_stocks if stock in bottom_stocks])
        parts.append("B%s=%s" % (k, hit))
    return " ".join(parts)


def recall_lift_text(label, stocks, universe_returns):
    if universe_returns is None or len(universe_returns) == 0:
        return ""
    valid_stocks = [stock for stock in stocks if stock in universe_returns.index]
    if len(valid_stocks) == 0:
        return ""
    universe_size = int(len(universe_returns))
    parts = []
    for k in g.monitor_recall_top_ns:
        k = int(k)
        if k <= 0:
            continue
        realized_count = min(k, universe_size)
        top_realized = set(list(universe_returns.index[:realized_count]))
        hit_stocks = [stock for stock in valid_stocks if stock in top_realized]
        hit = len(hit_stocks)
        recall = float(hit) / float(max(1, realized_count))
        precision = float(hit) / float(max(1, len(valid_stocks)))
        expected = float(len(valid_stocks) * realized_count) / float(max(1, universe_size))
        lift = float(hit) / expected if expected > 0 else np.nan
        parts.append(
            "T%s hit=%s R=%s P=%s L=%sx" % (
                k,
                hit,
                fmt_pct(recall),
                fmt_pct(precision),
                fmt_num(lift, 1),
            )
        )
    return " | ".join(parts)


def log_topk_recall_monitor(groups, universe_returns):
    if universe_returns is None or len(universe_returns) == 0:
        log.warn("MONITOR recall skipped: empty universe returns")
        return
    universe_size = int(len(universe_returns))
    for k in g.monitor_recall_top_ns:
        k = int(k)
        if k <= 0:
            continue
        top_realized = list(universe_returns.index[:min(k, universe_size)])
        top_ret = float(universe_returns.iloc[0]) if len(universe_returns) > 0 else np.nan
        kth_ret = float(universe_returns.iloc[min(k, universe_size) - 1]) if universe_size >= 1 else np.nan
        log.info(
            "MONITOR TOP%s | universe=%s | top_ret=%s | kth_ret=%s | codes=%s" % (
                k,
                universe_size,
                fmt_pct(top_ret),
                fmt_pct(kth_ret),
                ",".join(top_realized[:min(10, len(top_realized))]),
            )
        )

    for label in sorted(groups.keys()):
        stocks = list(groups[label])
        if len(stocks) == 0:
            continue
        quality = calc_selected_return_quality(stocks, universe_returns)
        log.info(
            "MONITOR RECALL | %-12s | valid=%s/%s | avg_ret=%s median_ret=%s worst_ret=%s | "
            "avg_rank=%s median_rank=%s worst_rank=%s | %s | %s" % (
                label,
                quality["valid"],
                len(stocks),
                fmt_pct(quality["avg_ret"]),
                fmt_pct(quality["median_ret"]),
                fmt_pct(quality["worst_ret"]),
                fmt_pct(quality["avg_pct"]),
                fmt_pct(quality["median_pct"]),
                fmt_pct(quality["worst_pct"]),
                recall_lift_text(label, stocks, universe_returns),
                bottom_hit_text(stocks, universe_returns),
            )
        )


def log_target_vs_raw_monitor(groups, universe_returns):
    target_label = "target_top{}".format(g.stock_num)
    if target_label not in groups:
        return
    target_quality = calc_selected_return_quality(groups[target_label], universe_returns)
    target_set = set(groups[target_label])
    for label in ["raw_top{}".format(g.stock_num), "raw_top10", "raw_top20", "raw_top30"]:
        if label not in groups or label == target_label:
            continue
        raw_quality = calc_selected_return_quality(groups[label], universe_returns)
        overlap = len(target_set.intersection(set(groups[label])))
        ret_gap = target_quality["avg_ret"] - raw_quality["avg_ret"]
        rank_gap = target_quality["avg_pct"] - raw_quality["avg_pct"]
        log.info(
            "MONITOR TARGET_GAP | %s vs %s | overlap=%s/%s | avg_ret_gap=%s | avg_rank_gap=%s" % (
                target_label,
                label,
                overlap,
                len(groups[target_label]),
                fmt_bps(ret_gap),
                fmt_pct(rank_gap),
            )
        )


def log_previous_model_monitor(context, eval_end_date):
    snapshot = g.monitor_last_snapshot
    if not snapshot:
        return
    start_date = snapshot.get("rebalance_date")
    if start_date is None:
        return
    eval_key = "{}_{}".format(start_date, eval_end_date)
    if g.monitor_last_eval_key == eval_key:
        return
    if pd.Timestamp(eval_end_date) <= pd.Timestamp(start_date):
        return

    benchmark = snapshot.get("benchmark", g.benchmark)
    bench_ret = calc_benchmark_close_return(benchmark, start_date, eval_end_date)
    log.info("")
    log.info("========== MODEL MONITOR REALIZED %s -> %s ==========" % (start_date, eval_end_date))
    log.info(
        "MONITOR PERIOD | start=%s | end=%s | benchmark=%s | bench_ret=%s" % (
            start_date,
            eval_end_date,
            benchmark,
            fmt_pct(bench_ret),
        )
    )
    groups = snapshot.get("groups", {})
    universe = snapshot.get("universe", [])
    universe_returns = calc_stock_close_returns(universe, start_date, eval_end_date)
    log_topk_recall_monitor(groups, universe_returns)
    log_target_vs_raw_monitor(groups, universe_returns)
    for label in sorted(groups.keys()):
        stocks = groups[label]
        stats = calc_equal_weight_close_monitor(stocks, start_date, eval_end_date)
        excess = stats["ret"] - bench_ret if not pd.isnull(bench_ret) and not pd.isnull(stats["ret"]) else np.nan
        log.info(
            "MONITOR REALIZED | %-12s | count=%s valid=%s | ret=%s excess=%s mdd=%s win=%s" % (
                label,
                len(stocks),
                stats["valid_count"],
                fmt_pct(stats["ret"]),
                fmt_pct(excess),
                fmt_pct(stats["mdd"]),
                fmt_pct(stats["win_rate"]),
            )
        )
    g.monitor_last_eval_key = eval_key


def log_execution_monitor(context, target_list, stage):
    if not g.enable_model_monitor:
        return
    target_list = list(target_list)
    current_holds = [
        position.security
        for position in context.portfolio.positions.values()
        if position.total_amount > 0
    ]
    target_set = set(target_list)
    hold_set = set(current_holds)
    missing = [stock for stock in target_list if stock not in hold_set]
    extra = [stock for stock in current_holds if stock not in target_set]
    overlap = len(target_set.intersection(hold_set))
    total_value = float(context.portfolio.total_value) if context.portfolio.total_value else 0.0
    cash = float(context.portfolio.cash) if context.portfolio.cash else 0.0
    cash_ratio = cash / total_value if total_value > 0 else np.nan
    log.info(
        "MONITOR EXEC | %s | target=%s hold=%s overlap=%s/%s cash=%s cash_ratio=%s missing=%s extra=%s" % (
            stage,
            len(target_list),
            len(current_holds),
            overlap,
            len(target_list),
            fmt_num(cash, 0),
            fmt_pct(cash_ratio),
            ",".join(missing),
            ",".join(extra),
        )
    )


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
        log_execution_monitor(context, target_list, "after_buy_skip_full")
        return

    buy_list = [stock for stock in target_list if stock not in current_holds]
    buy_slots = target_num - len(current_holds)
    if buy_slots <= 0 or context.portfolio.cash <= 0:
        log_execution_monitor(context, target_list, "after_buy_skip_cash")
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
    log_execution_monitor(context, target_list, "after_buy")


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
