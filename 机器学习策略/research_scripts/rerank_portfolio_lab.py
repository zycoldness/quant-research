import gc
import datetime
import os
import pickle
import warnings

try:
    from jqdata import *
    from jqfactor import get_factor_values
except Exception:
    # Local syntax checks do not have JoinQuant APIs. The rebuild path is only
    # used inside JoinQuant research/notebook runtime.
    pass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 240)
pd.set_option("display.width", 240)


# %%
# =========================
# Cell 1: candidate model export config
# =========================
# Goal:
# - Keep the existing V2.10 backtest logic unchanged:
#   base_score_z -> top30 candidates -> final_score rerank -> top10 with industry cap.
# - Export each model candidate as a standalone pkl bundle compatible with
#   csi800_lgb_factor_v210_refit_fixed_iter_strategy_*.
# - No walk-forward here; this cell is for generating files you can upload and
#   test one by one in JoinQuant.
FINAL_TRAIN_START = "2019-01-01"
FINAL_TRAIN_END = "2025-03-31"

LEGACY_DATA_FILE = "train_csi800_factor_v23_price_all.csv"
V4_DATA_FILE = "train_csi800_factor_v40_data_enhancement.csv"

# Data pipeline switches:
# - default reads cached CSV to save time;
# - set FORCE_REBUILD_V4_DATA=True to rebuild V4 data from JoinQuant APIs.
EXPORT_LEGACY_IF_AVAILABLE = True
EXPORT_V210_LEGACY = True
EXPORT_V4_CANDIDATES = True
EXPORT_ONLY_2026_OOS_MAINLINE = True
FINAL_EXPORT_RESEARCH_VERSIONS = [
    "candidate_v46_lgb_direct_hybrid_l2_ff10_2019_2025q1_legacy_unsealed",
    "candidate_v46_lgb_direct_hybrid_l2_ff10_2019_2025q1_label_safe",
    "candidate_v48_current_rolling5y_l2_ff10_2025q4",
    "candidate_v48_current_no_bagging_expanding_l2_ff10_2025q4",
    "candidate_v410_fixed_iter20_rolling5y_l2_ff10_2025q4",
    "candidate_v410_fixed_iter50_rolling5y_l2_ff10_2025q4",
    "candidate_v410_fixed_iter20_expanding_l2_ff10_2025q4",
]
AUTO_REBUILD_MISSING_V4_DATA = True
FORCE_REBUILD_V4_DATA = False
V4_DATA_START = "2019-01-01"
V4_DATA_END_FOR_LABEL = "2026-05-31"
UNIVERSE_NAME = "CSI800"
UNIVERSE_INDEX = "000906.XSHG"
BENCHMARK = "000906.XSHG"
MIN_LISTING_DAYS = 180

TOP_N_PORTFOLIO = 10
TOP_N_CANDIDATES = 30
INDUSTRY_CAP_RATIO = 0.20
CORR_THRESHOLD = 0.70

INNER_VALID_FRAC = 0.20
INNER_VALID_MIN_MONTHS = 6
RESID_TOP_K = 5
RESID_WEIGHT = 0.15

BASE_FACTOR_COLS = [
    "cash_flow_to_price_ratio",
    "book_to_price_ratio",
    "earnings_yield",
    "sales_to_price_ratio",
    "cash_earnings_to_price_ratio",
    "earnings_to_price_ratio",
    "roe_ttm",
    "roa_ttm",
    "gross_profit_ttm",
    "operating_profit_to_total_profit",
    "net_operate_cash_flow_to_total_liability",
    "net_operating_cash_flow_coverage",
    "adjusted_profit_to_total_profit",
    "ACCA",
    "growth",
    "net_working_capital",
    "operating_profit_per_share",
    "net_operate_cash_flow_per_share",
    "total_operating_revenue_per_share",
    "super_quick_ratio",
    "MLEV",
    "debt_to_equity_ratio",
    "debt_to_tangible_equity_ratio",
    "momentum",
    "Rank1M",
    "sharpe_ratio_60",
    "Variance20",
    "liquidity",
    "beta",
    "ATR6",
    "MFI14",
    "DAVOL10",
    "VOL10",
    "VMACD",
    "VOSC",
    "Skewness20",
    "Kurtosis20",
]

LEGACY_PX_FEATURE_COLS = [
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

V4_PRICE_PATH_COLS = [
    "px_ret_5",
    "px_ret_20",
    "px_ret_60",
    "px_ret_120",
    "px_close_to_ma20",
    "px_close_to_ma60",
    "px_ma20_to_ma60",
    "px_volatility_20",
    "px_volatility_60",
    "px_drawdown_20",
    "px_drawdown_60",
    "px_drawdown_120",
    "px_up_day_ratio_20",
    "px_new_high_distance_60",
    "px_new_low_distance_60",
    "px_skew_20",
    "px_kurt_20",
]

TRADE_LIQUIDITY_COLS = [
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

CONTEXT_COLS = [
    "ctx_industry_ret_20",
    "ctx_industry_ret_60",
    "ctx_stock_minus_industry_ret_20",
    "ctx_stock_minus_industry_ret_60",
    "ctx_stock_rank_industry_ret_20",
    "ctx_stock_rank_industry_volatility_20",
    "ctx_market_ret_20",
    "ctx_market_ret_60",
    "ctx_market_volatility_20",
]

CORE_TEMPORAL_FACTORS = [
    "book_to_price_ratio",
    "earnings_yield",
    "cash_flow_to_price_ratio",
    "Rank1M",
    "sharpe_ratio_60",
    "VOSC",
    "MFI14",
]

PRICE_PATH_LIGHT_COLS = [
    "px_ret_20",
    "px_close_to_ma60",
    "px_volatility_60",
    "px_drawdown_60",
    "px_up_day_ratio_20",
    "px_new_high_distance_60",
]

HYBRID_LIGHT_EXTRA_COLS = [
    "liq_money_ratio_20_60",
    "liq_paused_count_20",
    "px_close_to_ma60",
    "px_drawdown_60",
    "ts_cash_flow_to_price_ratio_rank_mean_3m",
    "ts_Rank1M_rank_chg_1m",
]

INDUSTRY_RELATIVE_FACTORS = [
    "book_to_price_ratio",
    "earnings_yield",
    "cash_flow_to_price_ratio",
    "roe_ttm",
    "roa_ttm",
    "Rank1M",
    "sharpe_ratio_60",
]

BASE_PARAMS_FF10 = {
    "objective": "regression",
    "metric": "l2",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 1.0,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.3,
    "verbose": -1,
}

BASE_PARAMS_FF10_NO_BAGGING = dict(BASE_PARAMS_FF10)
BASE_PARAMS_FF10_NO_BAGGING.update({
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
})

RESID_PARAMS_FF10 = {
    "objective": "regression",
    "metric": "l2",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 500,
    "feature_fraction": 1.0,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.5,
    "lambda_l2": 1.0,
    "verbose": -1,
}

EXPORT_MANIFEST_CSV = "candidate_model_export_manifest.csv"

print("candidate export config ready")


# %%
# =========================
# Cell 2: helpers
# =========================
def unique_keep_order(cols):
    seen = set()
    out = []
    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def safe_corr(a, b):
    s = pd.DataFrame({
        "a": np.asarray(a, dtype=float),
        "b": np.asarray(b, dtype=float),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 3:
        return np.nan
    if s["a"].nunique() < 2 or s["b"].nunique() < 2:
        return np.nan
    return s["a"].corr(s["b"])


def safe_rank_ic(a, b):
    s = pd.DataFrame({
        "a": np.asarray(a, dtype=float),
        "b": np.asarray(b, dtype=float),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 3:
        return np.nan
    if s["a"].nunique() < 2 or s["b"].nunique() < 2:
        return np.nan
    return s["a"].rank(pct=True).corr(s["b"].rank(pct=True))


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def require_joinquant_api():
    missing = [
        name for name in [
            "get_trade_days",
            "get_index_stocks",
            "get_factor_values",
            "get_price",
            "get_security_info",
        ]
        if name not in globals()
    ]
    if missing:
        raise RuntimeError(
            "V4 data rebuild requires JoinQuant research runtime; missing APIs: {}".format(
                ",".join(missing)
            )
        )


def get_period_date(period, start_date, end_date):
    require_joinquant_api()
    trade_days = pd.to_datetime(get_trade_days(start_date=start_date, end_date=end_date))
    if len(trade_days) == 0:
        return []
    if period != "M":
        raise ValueError("V4 data pipeline only supports monthly period M")

    dates = []
    last_key = None
    for d in trade_days:
        key = d.strftime("%Y-%m")
        if key != last_key:
            dates.append(d.strftime("%Y-%m-%d"))
            last_key = key
    return dates


def get_previous_trade_date(date):
    require_joinquant_api()
    trade_days = pd.to_datetime(get_trade_days(end_date=date, count=2))
    if len(trade_days) < 2:
        return None
    return trade_days[-2].strftime("%Y-%m-%d")


def delect_stop(stocks, begin_date, n=180):
    stock_list = []
    begin_dt = pd.Timestamp(begin_date).to_pydatetime()
    for stock in stocks:
        info = get_security_info(stock)
        if info is None:
            continue
        if info.start_date <= (begin_dt - datetime.timedelta(days=n)).date():
            stock_list.append(stock)
    return stock_list


def filter_paused_stock_by_date(stock_list, date):
    if len(stock_list) == 0:
        return []
    try:
        paused_df = get_price(
            stock_list,
            end_date=date,
            frequency="daily",
            fields=["paused"],
            count=1,
            skip_paused=False,
            panel=False,
            fill_paused=True,
        )
    except Exception:
        return stock_list

    if paused_df is None or paused_df.empty or "paused" not in paused_df.columns:
        return stock_list

    paused_map = paused_df.groupby("code")["paused"].last()
    return [
        stock for stock in stock_list
        if (stock not in paused_map.index) or (not bool(paused_map.loc[stock]))
    ]


def get_stock(stock_pool, feature_date):
    require_joinquant_api()
    if stock_pool == "CSI800":
        stock_list = get_index_stocks(UNIVERSE_INDEX, feature_date)
    elif stock_pool == "HS300":
        stock_list = get_index_stocks("000300.XSHG", feature_date)
    elif stock_pool == "ZZ1000":
        stock_list = get_index_stocks("000852.XSHG", feature_date)
    elif stock_pool == "A":
        stock_list = get_index_stocks("000985.XSHG", feature_date)
    else:
        raise ValueError("unsupported stock_pool: {}".format(stock_pool))

    if len(stock_list) == 0:
        return []

    st_data = get_extras("is_st", stock_list, count=1, end_date=feature_date)
    if st_data is not None and len(st_data) > 0:
        st_row = st_data.iloc[0]
        stock_list = [
            stock for stock in stock_list
            if (stock not in st_row.index) or pd.isnull(st_row[stock]) or (not bool(st_row[stock]))
        ]

    stock_list = filter_paused_stock_by_date(stock_list, feature_date)
    stock_list = delect_stop(stock_list, feature_date, n=MIN_LISTING_DAYS)
    return stock_list


def get_industry_bucket_map_for_data(stock_list, date):
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


def get_factor_data(stock_list, date):
    if len(stock_list) == 0:
        return pd.DataFrame()

    df_factor = pd.DataFrame(index=stock_list)
    for fac_chunk in chunks(BASE_FACTOR_COLS, 20):
        try:
            factor_data = get_factor_values(
                securities=stock_list,
                factors=fac_chunk,
                count=1,
                end_date=date,
            )
        except Exception:
            factor_data = None

        for fac in fac_chunk:
            try:
                if factor_data is not None and fac in factor_data:
                    df_factor[fac] = factor_data[fac].iloc[0, :]
                else:
                    df_factor[fac] = np.nan
            except Exception:
                df_factor[fac] = np.nan
    return df_factor


def calc_ret(close_mat, days):
    if close_mat is None or close_mat.empty or len(close_mat) <= days:
        return pd.Series(index=close_mat.columns if close_mat is not None else [], dtype=float)
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


def get_price_path_and_liquidity_data(stock_list, date, lookback=121, chunk_size=160):
    cols = V4_PRICE_PATH_COLS + TRADE_LIQUIDITY_COLS[:9]
    out_all = []
    for stock_chunk in chunks(stock_list, chunk_size):
        out = pd.DataFrame(index=stock_chunk, columns=cols, dtype=float)
        try:
            price_df = get_price(
                stock_chunk,
                end_date=date,
                frequency="daily",
                fields=["close", "high", "low", "volume", "money", "paused"],
                count=lookback,
                skip_paused=False,
                fq="pre",
                panel=False,
                fill_paused=True,
            )
        except Exception:
            price_df = None

        if price_df is None or price_df.empty:
            out_all.append(out)
            continue

        for col in ["close", "high", "low", "volume", "money", "paused"]:
            if col not in price_df.columns:
                price_df[col] = np.nan
        price_df["time"] = pd.to_datetime(price_df["time"]).dt.normalize()
        close_mat = price_df.pivot_table(index="time", columns="code", values="close").sort_index()
        high_mat = price_df.pivot_table(index="time", columns="code", values="high").sort_index()
        low_mat = price_df.pivot_table(index="time", columns="code", values="low").sort_index()
        volume_mat = price_df.pivot_table(index="time", columns="code", values="volume").sort_index()
        money_mat = price_df.pivot_table(index="time", columns="code", values="money").sort_index()
        paused_mat = price_df.pivot_table(index="time", columns="code", values="paused").sort_index()

        ret_mat = close_mat.pct_change()
        last_close = close_mat.iloc[-1]
        ma20 = close_mat.tail(20).mean()
        ma60 = close_mat.tail(60).mean()
        money20 = money_mat.tail(20).mean()
        money60 = money_mat.tail(60).mean()
        volume20 = volume_mat.tail(20).mean()
        volume60 = volume_mat.tail(60).mean()

        out["px_ret_5"] = calc_ret(close_mat, 5)
        out["px_ret_20"] = calc_ret(close_mat, 20)
        out["px_ret_60"] = calc_ret(close_mat, 60)
        out["px_ret_120"] = calc_ret(close_mat, 120)
        out["px_close_to_ma20"] = last_close / ma20 - 1
        out["px_close_to_ma60"] = last_close / ma60 - 1
        out["px_ma20_to_ma60"] = ma20 / ma60 - 1
        out["px_volatility_20"] = ret_mat.tail(20).std()
        out["px_volatility_60"] = ret_mat.tail(60).std()
        out["px_drawdown_20"] = last_close / close_mat.tail(20).max() - 1
        out["px_drawdown_60"] = last_close / close_mat.tail(60).max() - 1
        out["px_drawdown_120"] = last_close / close_mat.tail(120).max() - 1
        out["px_up_day_ratio_20"] = calc_up_day_ratio(ret_mat, 20)
        out["px_new_high_distance_60"] = last_close / close_mat.tail(60).max() - 1
        out["px_new_low_distance_60"] = calc_new_low_distance(close_mat, 60)
        out["px_skew_20"] = ret_mat.tail(20).skew()
        out["px_kurt_20"] = ret_mat.tail(20).kurt()

        out["liq_money_mean_20"] = money20
        out["liq_money_mean_60"] = money60
        out["liq_money_ratio_20_60"] = money20 / money60 - 1
        out["liq_volume_mean_20"] = volume20
        out["liq_volume_ratio_20_60"] = volume20 / volume60 - 1
        out["liq_amplitude_mean_20"] = (high_mat.tail(20) / low_mat.tail(20) - 1).mean()
        out["liq_amplitude_mean_60"] = (high_mat.tail(60) / low_mat.tail(60) - 1).mean()
        out["liq_paused_count_20"] = paused_mat.tail(20).fillna(0).sum()
        out["liq_paused_count_60"] = paused_mat.tail(60).fillna(0).sum()

        out_all.append(out.replace([np.inf, -np.inf], np.nan))
        del price_df, close_mat, high_mat, low_mat, volume_mat, money_mat, paused_mat, ret_mat
        gc.collect()
    return pd.concat(out_all).reindex(index=stock_list)


def get_limit_state_data(stock_list, date, lookback=20, chunk_size=160):
    cols = [
        "liq_low_money_days_20",
        "liq_limit_up_count_20",
        "liq_limit_down_count_20",
        "liq_one_price_limit_count_20",
    ]
    out_all = []
    for stock_chunk in chunks(stock_list, chunk_size):
        out = pd.DataFrame(index=stock_chunk, columns=cols, dtype=float)
        try:
            price_df = get_price(
                stock_chunk,
                end_date=date,
                frequency="daily",
                fields=["close", "high", "low", "money", "paused", "high_limit", "low_limit"],
                count=lookback,
                skip_paused=False,
                fq=None,
                panel=False,
                fill_paused=True,
            )
        except Exception:
            price_df = None

        if price_df is None or price_df.empty:
            out_all.append(out)
            continue

        for col in ["close", "high", "low", "money", "paused", "high_limit", "low_limit"]:
            if col not in price_df.columns:
                price_df[col] = np.nan
        price_df["time"] = pd.to_datetime(price_df["time"]).dt.normalize()
        money_mat = price_df.pivot_table(index="time", columns="code", values="money").sort_index()
        close_mat = price_df.pivot_table(index="time", columns="code", values="close").sort_index()
        high_mat = price_df.pivot_table(index="time", columns="code", values="high").sort_index()
        low_mat = price_df.pivot_table(index="time", columns="code", values="low").sort_index()
        high_limit_mat = price_df.pivot_table(index="time", columns="code", values="high_limit").sort_index()
        low_limit_mat = price_df.pivot_table(index="time", columns="code", values="low_limit").sort_index()

        money_q20 = money_mat.stack().quantile(0.20) if len(money_mat.stack().dropna()) else np.nan
        out["liq_low_money_days_20"] = (money_mat.tail(20) < money_q20).sum() if not pd.isnull(money_q20) else np.nan

        limit_up = close_mat >= (high_limit_mat * 0.999)
        limit_down = close_mat <= (low_limit_mat * 1.001)
        one_price = (high_mat <= low_mat * 1.0001) & (limit_up | limit_down)
        out["liq_limit_up_count_20"] = limit_up.tail(20).sum()
        out["liq_limit_down_count_20"] = limit_down.tail(20).sum()
        out["liq_one_price_limit_count_20"] = one_price.tail(20).sum()

        out_all.append(out.replace([np.inf, -np.inf], np.nan))
        del price_df, money_mat, close_mat, high_mat, low_mat, high_limit_mat, low_limit_mat
        gc.collect()
    return pd.concat(out_all).reindex(index=stock_list)


def get_market_context(date, benchmark=BENCHMARK, lookback=61):
    out = {
        "ctx_market_ret_20": np.nan,
        "ctx_market_ret_60": np.nan,
        "ctx_market_volatility_20": np.nan,
    }
    try:
        bench_df = get_price(
            benchmark,
            end_date=date,
            frequency="daily",
            fields=["close"],
            count=lookback,
            skip_paused=True,
            fq="pre",
        )
    except Exception:
        bench_df = None

    if bench_df is None or bench_df.empty or "close" not in bench_df.columns:
        return out
    close = bench_df["close"].dropna()
    if len(close) > 20:
        out["ctx_market_ret_20"] = close.iloc[-1] / close.iloc[-21] - 1
        out["ctx_market_volatility_20"] = close.pct_change().tail(20).std()
    if len(close) > 60:
        out["ctx_market_ret_60"] = close.iloc[-1] / close.iloc[-61] - 1
    return out


def attach_industry_context(factor_data, market_context):
    out = factor_data.copy()
    for col, value in market_context.items():
        out[col] = value

    for ret_col, ctx_col in [
        ("px_ret_20", "ctx_industry_ret_20"),
        ("px_ret_60", "ctx_industry_ret_60"),
    ]:
        out[ctx_col] = out.groupby("industry_bucket")[ret_col].transform("mean")

    out["ctx_stock_minus_industry_ret_20"] = out["px_ret_20"] - out["ctx_industry_ret_20"]
    out["ctx_stock_minus_industry_ret_60"] = out["px_ret_60"] - out["ctx_industry_ret_60"]
    out["ctx_stock_rank_industry_ret_20"] = out.groupby("industry_bucket")["px_ret_20"].rank(pct=True)
    out["ctx_stock_rank_industry_volatility_20"] = out.groupby("industry_bucket")["px_volatility_20"].rank(pct=True)
    return out


def get_forward_alpha(stock_list, date, next_date, benchmark):
    if len(stock_list) == 0:
        return pd.Series(dtype=float)

    price_df = get_price(
        stock_list,
        start_date=date,
        end_date=next_date,
        frequency="daily",
        fields=["close"],
        skip_paused=True,
        fq="pre",
        panel=False,
    )
    if price_df is None or price_df.empty:
        return pd.Series(dtype=float)

    price_df["time"] = pd.to_datetime(price_df["time"]).dt.normalize()
    close_mat = price_df.pivot_table(index="time", columns="code", values="close").sort_index()
    if len(close_mat) < 2:
        return pd.Series(dtype=float)
    stock_ret = close_mat.iloc[-1] / close_mat.iloc[1] - 1

    bench_df = get_price(
        benchmark,
        start_date=date,
        end_date=next_date,
        frequency="daily",
        fields=["close"],
        skip_paused=True,
        fq="pre",
    )
    if bench_df is None or bench_df.empty or len(bench_df) < 2:
        return pd.Series(dtype=float)

    bench_ret = bench_df["close"].iloc[-1] / bench_df["close"].iloc[1] - 1
    return stock_ret - bench_ret


def add_core_factor_temporal_features(df):
    out = df.copy()
    out = out.sort_values(["rebalance_date", "stock"]).reset_index(drop=True)
    for factor in CORE_TEMPORAL_FACTORS:
        if factor not in out.columns:
            continue
        rank_col = "tmp_{}_rank".format(factor)
        out[rank_col] = out.groupby("rebalance_date")[factor].rank(pct=True)
        g_stock = out.groupby("stock")[rank_col]
        for lag in [1, 3]:
            col = "ts_{}_rank_chg_{}m".format(factor, lag)
            out[col] = out[rank_col] - g_stock.shift(lag)
        mean_col = "ts_{}_rank_mean_3m".format(factor)
        std_col = "ts_{}_rank_std_3m".format(factor)
        z_col = "ts_{}_rank_z_6m".format(factor)
        out[mean_col] = g_stock.transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
        out[std_col] = g_stock.transform(lambda s: s.shift(1).rolling(3, min_periods=2).std())
        rolling_mean_6 = g_stock.transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
        rolling_std_6 = g_stock.transform(lambda s: s.shift(1).rolling(6, min_periods=3).std())
        out[z_col] = (out[rank_col] - rolling_mean_6) / rolling_std_6
        out = out.drop(columns=[rank_col])
    return out


def build_v4_dataset():
    require_joinquant_api()
    date_list = get_period_date("M", V4_DATA_START, V4_DATA_END_FOR_LABEL)
    print("V4 rebuild rebalance dates =", len(date_list), "|", V4_DATA_START, "->", V4_DATA_END_FOR_LABEL)

    all_rows = []
    for i, rebalance_date in enumerate(date_list[:-1]):
        next_date = date_list[i + 1]
        feature_date = get_previous_trade_date(rebalance_date)
        if feature_date is None:
            continue

        stock_list = get_stock(UNIVERSE_NAME, feature_date)
        if len(stock_list) == 0:
            continue

        jq_factor_data = get_factor_data(stock_list, feature_date)
        if jq_factor_data is None or jq_factor_data.empty:
            continue

        industry_map = get_industry_bucket_map_for_data(stock_list, feature_date)
        price_liq_data = get_price_path_and_liquidity_data(stock_list, feature_date)
        limit_data = get_limit_state_data(stock_list, feature_date)
        alpha = get_forward_alpha(stock_list, rebalance_date, next_date, BENCHMARK)
        if alpha.empty:
            continue

        factor_data = jq_factor_data.join(price_liq_data, how="left").join(limit_data, how="left")
        factor_data["stock"] = factor_data.index
        factor_data["industry_bucket"] = factor_data["stock"].map(industry_map).fillna("UNKNOWN")
        factor_data = attach_industry_context(factor_data, get_market_context(feature_date, BENCHMARK))
        factor_data["alpha_1m"] = alpha
        factor_data["rebalance_date"] = rebalance_date
        factor_data["feature_date"] = feature_date
        factor_data["next_date"] = next_date
        factor_data = factor_data.dropna(subset=["alpha_1m"]).copy()
        if len(factor_data) < 30:
            continue

        factor_data["alpha_rank_pct"] = factor_data["alpha_1m"].rank(pct=True, method="first")
        all_rows.append(factor_data.reset_index(drop=True))
        print(
            "  rebuilt {}/{} rebalance={} feature={} rows={}".format(
                i + 1, max(1, len(date_list) - 1), rebalance_date, feature_date, len(factor_data)
            )
        )

        del jq_factor_data, price_liq_data, limit_data, alpha, factor_data
        gc.collect()

    if len(all_rows) == 0:
        raise ValueError("V4 data rebuild produced no rows")
    df = pd.concat(all_rows, ignore_index=True)
    df = add_core_factor_temporal_features(df)
    df.to_csv(V4_DATA_FILE, index=False)
    print("V4 data rebuilt rows =", len(df), "saved ->", V4_DATA_FILE)
    return df


def split_inner_train_valid(train_df):
    dates = sorted(pd.to_datetime(train_df["rebalance_date"].dropna().unique()))
    if len(dates) < 3:
        return train_df.copy(), train_df.copy()

    n_valid = int(np.ceil(len(dates) * INNER_VALID_FRAC))
    n_valid = max(INNER_VALID_MIN_MONTHS, n_valid)
    n_valid = min(max(1, n_valid), len(dates) - 1)
    valid_dates = set(dates[-n_valid:])

    fit_df = train_df[~train_df["rebalance_date"].isin(valid_dates)].copy()
    inner_valid_df = train_df[train_df["rebalance_date"].isin(valid_dates)].copy()
    if fit_df.empty or inner_valid_df.empty:
        return train_df.copy(), train_df.copy()
    return fit_df, inner_valid_df


def build_corr_components(train_df, feature_cols, threshold):
    from collections import defaultdict

    feature_cols = [col for col in feature_cols if col in train_df.columns]
    if len(feature_cols) == 0:
        return []

    corr_matrix = train_df[feature_cols].corr()
    graph = defaultdict(list)
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            col1 = feature_cols[i]
            col2 = feature_cols[j]
            corr_value = corr_matrix.iloc[i, j]
            if not pd.isnull(corr_value) and abs(corr_value) > threshold:
                graph[col1].append(col2)
                graph[col2].append(col1)

    for col in feature_cols:
        graph[col]

    visited = set()
    components = []

    def dfs(node, comp):
        visited.add(node)
        comp.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, comp)

    for col in feature_cols:
        if col not in visited:
            comp = []
            dfs(col, comp)
            components.append(comp)
    return components


def select_features_train_only(train_df, candidate_cols):
    candidate_cols = unique_keep_order([col for col in candidate_cols if col in train_df.columns])
    if len(candidate_cols) == 0:
        raise ValueError("candidate_cols is empty")

    missing_counts = train_df[candidate_cols].isnull().sum().to_dict()
    components = build_corr_components(train_df, candidate_cols, CORR_THRESHOLD)

    # Preserve the original V2.10 component traversal order so that the legacy
    # export remains comparable if feature sampling is tuned again later.
    to_keep = []
    to_remove = []
    for comp in components:
        if len(comp) == 1:
            to_keep.append(comp[0])
            continue
        comp_sorted = sorted(comp, key=lambda x: (missing_counts[x], x))
        to_keep.append(comp_sorted[0])
        to_remove.extend(comp_sorted[1:])
    return to_keep, to_remove


def prepare_xy(df, feature_cols, target_col, fill_values=None):
    d = df.dropna(subset=[target_col]).copy()
    X = d[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    y = d[target_col].astype(float).copy()
    if fill_values is None:
        fill_values = X.median().replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.fillna(fill_values).fillna(0)
    return X, y, fill_values


def fit_lgb_es(fit_df, inner_valid_df, feature_cols, target_col, params, seed, num_boost_round=500):
    fit_df = fit_df.dropna(subset=[target_col]).copy()
    inner_valid_df = inner_valid_df.dropna(subset=[target_col]).copy()
    if fit_df.empty:
        raise ValueError("fit_df empty for target {}".format(target_col))
    if inner_valid_df.empty:
        inner_valid_df = fit_df.copy()

    params = dict(params)
    params["seed"] = seed

    X_fit, y_fit, fill_values = prepare_xy(fit_df, feature_cols, target_col)
    X_inner, y_inner, _ = prepare_xy(inner_valid_df, feature_cols, target_col, fill_values)

    model = lgb.train(
        params,
        lgb.Dataset(X_fit, label=y_fit),
        num_boost_round=num_boost_round,
        valid_sets=[lgb.Dataset(X_fit, label=y_fit), lgb.Dataset(X_inner, label=y_inner)],
        valid_names=["fit", "inner_valid"],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    best_iteration = int(getattr(model, "best_iteration", 0) or num_boost_round)
    inner_pred = np.asarray(model.predict(X_inner[feature_cols], num_iteration=best_iteration)).reshape(-1)
    corr = safe_corr(y_inner, inner_pred)
    rank_ic = safe_rank_ic(y_inner, inner_pred)
    return {
        "model": model,
        "feature_cols": list(feature_cols),
        "fill_values": fill_values.to_dict(),
        "metrics": {
            "best_iter": best_iteration,
            "fit_rows": len(fit_df),
            "inner_valid_rows": len(inner_valid_df),
            "inner_rmse": float(np.sqrt(mean_squared_error(y_inner, inner_pred))),
            "inner_mae": float(mean_absolute_error(y_inner, inner_pred)),
            "inner_corr": float(corr) if not pd.isnull(corr) else np.nan,
            "inner_rank_ic": float(rank_ic) if not pd.isnull(rank_ic) else np.nan,
        },
    }


def fit_lgb_fixed(train_df, feature_cols, target_col, params, seed, num_boost_round):
    train_df = train_df.dropna(subset=[target_col]).copy()
    if train_df.empty:
        raise ValueError("train_df empty for target {}".format(target_col))

    params = dict(params)
    params["seed"] = seed

    X_train, y_train, fill_values = prepare_xy(train_df, feature_cols, target_col)
    model = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train),
        num_boost_round=max(1, int(num_boost_round)),
        valid_sets=[lgb.Dataset(X_train, label=y_train)],
        valid_names=["train"],
        verbose_eval=False,
    )
    return {
        "model": model,
        "feature_cols": list(feature_cols),
        "fill_values": fill_values.to_dict(),
        "metrics": {
            "fixed_iter": max(1, int(num_boost_round)),
            "train_rows": len(train_df),
        },
    }


def predict_model(model_meta, df):
    feature_cols = list(model_meta["feature_cols"])
    fill_values = pd.Series(model_meta["fill_values"])
    X = df.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(fill_values).fillna(0)
    model = model_meta["model"]
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None and best_iteration > 0:
        return np.asarray(model.predict(X[feature_cols], num_iteration=best_iteration)).reshape(-1)
    return np.asarray(model.predict(X[feature_cols])).reshape(-1)


def select_px_features_by_residual_ic(train_df, px_cols, top_k):
    rows = []
    for col in px_cols:
        if col not in train_df.columns:
            continue
        ic = safe_rank_ic(train_df[col], train_df["residual_label"])
        rows.append({
            "feature": col,
            "residual_rank_ic": ic,
            "abs_residual_rank_ic": abs(ic) if not pd.isnull(ic) else np.nan,
            "missing": int(train_df[col].isnull().sum()),
        })
    ic_df = pd.DataFrame(rows).sort_values(
        ["abs_residual_rank_ic", "missing", "feature"],
        ascending=[False, True, True],
    )
    return ic_df.head(min(top_k, len(ic_df)))["feature"].tolist(), ic_df


def load_dataset(path):
    if not os.path.exists(path):
        print("skip missing data file:", path)
        return None
    df = pd.read_csv(path)
    df["rebalance_date"] = pd.to_datetime(df["rebalance_date"])
    if "feature_date" in df.columns:
        df["feature_date"] = pd.to_datetime(df["feature_date"])
    if "next_date" in df.columns:
        df["next_date"] = pd.to_datetime(df["next_date"])
    df["alpha_1m"] = pd.to_numeric(df["alpha_1m"], errors="coerce")
    df = df.dropna(subset=["alpha_1m"]).copy()
    print("loaded", path, "rows =", len(df), "date =", df["rebalance_date"].min(), "->", df["rebalance_date"].max())
    return df


def add_v45_industry_relative_features(df):
    out = df.copy()
    if "industry_bucket" not in out.columns:
        out["industry_bucket"] = "UNKNOWN"
    out["industry_bucket"] = out["industry_bucket"].fillna("UNKNOWN").astype(str)

    new_cols = []
    for fac in INDUSTRY_RELATIVE_FACTORS:
        if fac not in out.columns:
            continue
        median_col = "v45_{}_minus_industry_median".format(fac)
        rank_col = "v45_{}_rank_in_industry".format(fac)
        out[median_col] = out[fac] - out.groupby(["rebalance_date", "industry_bucket"])[fac].transform("median")
        out[rank_col] = out.groupby(["rebalance_date", "industry_bucket"])[fac].transform(lambda s: s.rank(pct=True))
        new_cols.extend([median_col, rank_col])
    return out, new_cols


def ensure_v4_dataset():
    should_rebuild = FORCE_REBUILD_V4_DATA or (
        AUTO_REBUILD_MISSING_V4_DATA and (not os.path.exists(V4_DATA_FILE))
    )
    if should_rebuild:
        reason = "FORCE_REBUILD_V4_DATA=True" if FORCE_REBUILD_V4_DATA else "cached V4 data missing"
        print("rebuilding V4 dataset because:", reason)
        return build_v4_dataset()

    df = load_dataset(V4_DATA_FILE)
    if df is None:
        raise ValueError(
            "{} not found. Set AUTO_REBUILD_MISSING_V4_DATA=True or run V4 data rebuild first.".format(
                V4_DATA_FILE
            )
        )
    return df


def get_variant_train_window(variant):
    train_start = variant.get("train_start", FINAL_TRAIN_START)
    train_end = variant.get("train_end", FINAL_TRAIN_END)
    label_end = variant.get("label_end", train_end)
    label_safe = bool(variant.get("require_label_end_within_train", False))
    return train_start, train_end, label_end, label_safe


def get_train_boundary_audit(df_all, train_start, train_end, label_end):
    audit_df = df_all[
        (df_all["rebalance_date"] >= pd.Timestamp(train_start)) &
        (df_all["rebalance_date"] <= pd.Timestamp(train_end))
    ].copy()
    out = {
        "pre_label_row_count": int(len(audit_df)),
        "pre_label_month_count": int(audit_df["rebalance_date"].nunique()) if "rebalance_date" in audit_df.columns else 0,
        "label_boundary_crossing_row_count": np.nan,
        "label_boundary_crossing_month_count": np.nan,
        "label_safe_row_count": np.nan,
        "label_safe_month_count": np.nan,
        "max_train_rebalance_date": "",
        "max_train_next_date": "",
    }
    if audit_df.empty:
        return out

    out["max_train_rebalance_date"] = str(pd.to_datetime(audit_df["rebalance_date"]).max().date())
    if "next_date" not in audit_df.columns:
        return out

    next_dates = pd.to_datetime(audit_df["next_date"])
    crossing = audit_df[next_dates > pd.Timestamp(label_end)]
    safe = audit_df[next_dates <= pd.Timestamp(label_end)]
    out["label_boundary_crossing_row_count"] = int(len(crossing))
    out["label_boundary_crossing_month_count"] = int(crossing["rebalance_date"].nunique())
    out["label_safe_row_count"] = int(len(safe))
    out["label_safe_month_count"] = int(safe["rebalance_date"].nunique())
    out["max_train_next_date"] = str(next_dates.max().date()) if len(next_dates.dropna()) else ""
    return out


def get_variant_train_df(df_all, variant):
    train_start, train_end, label_end, label_safe = get_variant_train_window(variant)
    train_df = df_all[
        (df_all["rebalance_date"] >= pd.Timestamp(train_start)) &
        (df_all["rebalance_date"] <= pd.Timestamp(train_end))
    ].copy()

    if label_safe:
        if "next_date" not in train_df.columns:
            raise ValueError(
                "{} requires label-safe training, but next_date is missing".format(
                    variant["research_version"]
                )
            )
        train_df = train_df[pd.to_datetime(train_df["next_date"]) <= pd.Timestamp(label_end)].copy()

    return train_df, train_start, train_end, label_end, label_safe


def export_candidate_model(df_all, variant):
    train_df, train_start, train_end, label_end, label_safe = get_variant_train_df(df_all, variant)
    boundary_audit = get_train_boundary_audit(df_all, train_start, train_end, label_end)
    if train_df.empty:
        raise ValueError("train_df empty for {}".format(variant["research_version"]))

    fit_df, inner_valid_df = split_inner_train_valid(train_df)
    candidate_cols = [col for col in variant["candidate_cols"] if col in train_df.columns]
    residual_cols = [col for col in variant["residual_cols"] if col in train_df.columns]
    base_features, base_removed = select_features_train_only(fit_df, candidate_cols)

    base_es = fit_lgb_es(
        fit_df,
        inner_valid_df,
        base_features,
        "alpha_1m",
        variant["base_params"],
        seed=42,
    )
    base_best_iter = base_es["metrics"]["best_iter"]

    train_for_resid = train_df.copy()
    train_for_resid["base_pred"] = predict_model(base_es, train_for_resid)
    train_for_resid["residual_label"] = (
        train_for_resid["alpha_1m"].astype(float) -
        train_for_resid["base_pred"].astype(float)
    )

    resid_features, resid_ic_df = select_px_features_by_residual_ic(train_for_resid, residual_cols, RESID_TOP_K)
    resid_fit_df = train_for_resid[train_for_resid["rebalance_date"].isin(fit_df["rebalance_date"].unique())].copy()
    resid_inner_df = train_for_resid[train_for_resid["rebalance_date"].isin(inner_valid_df["rebalance_date"].unique())].copy()

    resid_es = fit_lgb_es(
        resid_fit_df,
        resid_inner_df,
        resid_features,
        "residual_label",
        variant["resid_params"],
        seed=42,
        num_boost_round=300,
    )
    resid_best_iter = resid_es["metrics"]["best_iter"]

    base_model = fit_lgb_fixed(
        train_df,
        base_features,
        "alpha_1m",
        variant["base_params"],
        seed=42,
        num_boost_round=base_best_iter,
    )

    train_refit_resid = train_df.copy()
    train_refit_resid["base_pred"] = predict_model(base_model, train_refit_resid)
    train_refit_resid["residual_label"] = (
        train_refit_resid["alpha_1m"].astype(float) -
        train_refit_resid["base_pred"].astype(float)
    )

    residual_model = fit_lgb_fixed(
        train_refit_resid,
        resid_features,
        "residual_label",
        variant["resid_params"],
        seed=42,
        num_boost_round=resid_best_iter,
    )

    bundle = {
        "objective": "v210_refit_fixed_iter_overlay",
        "research_version": variant["research_version"],
        "benchmark": "000906.XSHG",
        "train_start": train_start,
        "train_end": train_end,
        "label_end": label_end,
        "require_label_end_within_train": label_safe,
        "train_boundary_audit": boundary_audit,
        "data_file": variant["data_file"],
        "protocol": "inner_valid_best_iter_then_full_train_refit",
        "base_params": variant["base_params"],
        "residual_params": variant["resid_params"],
        "base_model": base_model["model"],
        "base_feature_cols": base_model["feature_cols"],
        "base_fill_values": base_model["fill_values"],
        "base_best_iter": base_best_iter,
        "base_inner_metrics": base_es["metrics"],
        "base_removed_features": base_removed,
        "residual_model": residual_model["model"],
        "residual_feature_cols": residual_model["feature_cols"],
        "residual_fill_values": residual_model["fill_values"],
        "residual_best_iter": resid_best_iter,
        "residual_inner_metrics": resid_es["metrics"],
        "residual_feature_ic": resid_ic_df.to_dict("records"),
        "overlay_weight": RESID_WEIGHT,
        "overlay_mode": "top30_rerank",
        "top_n_candidates": TOP_N_CANDIDATES,
        "stock_num": TOP_N_PORTFOLIO,
        "industry_cap_ratio": INDUSTRY_CAP_RATIO,
        "uses_time_weight": False,
        "uses_current_valid_for_training": False,
        "final_role": variant.get("final_role", ""),
        "training_policy": variant.get("training_policy", ""),
        "param_set": variant.get("param_set", ""),
        "requires_v4_feature_adapter": bool(variant.get("requires_v4_feature_adapter", False)),
        "requires_industry_relative_adapter": bool(variant.get("requires_industry_relative_adapter", False)),
    }
    with open(variant["model_out"], "wb") as f:
        pickle.dump(bundle, f)

    row = {
        "research_version": variant["research_version"],
        "model_out": variant["model_out"],
        "data_file": variant["data_file"],
        "train_start": train_start,
        "train_end": train_end,
        "label_end": label_end,
        "require_label_end_within_train": label_safe,
        "train_row_count": len(train_df),
        "train_month_count": train_df["rebalance_date"].nunique(),
        "pre_label_row_count": boundary_audit["pre_label_row_count"],
        "pre_label_month_count": boundary_audit["pre_label_month_count"],
        "label_boundary_crossing_row_count": boundary_audit["label_boundary_crossing_row_count"],
        "label_boundary_crossing_month_count": boundary_audit["label_boundary_crossing_month_count"],
        "label_safe_row_count": boundary_audit["label_safe_row_count"],
        "label_safe_month_count": boundary_audit["label_safe_month_count"],
        "max_train_rebalance_date": boundary_audit["max_train_rebalance_date"],
        "max_train_next_date": boundary_audit["max_train_next_date"],
        "base_feature_count": len(base_model["feature_cols"]),
        "base_removed_count": len(base_removed),
        "base_best_iter": base_best_iter,
        "residual_feature_count": len(resid_features),
        "residual_features": ",".join(resid_features),
        "residual_best_iter": resid_best_iter,
        "final_role": variant.get("final_role", ""),
        "training_policy": variant.get("training_policy", ""),
        "param_set": variant.get("param_set", ""),
        "requires_v4_feature_adapter": bool(variant.get("requires_v4_feature_adapter", False)),
        "requires_industry_relative_adapter": bool(variant.get("requires_industry_relative_adapter", False)),
    }
    print("\nexported:", variant["research_version"])
    print("  ->", variant["model_out"])
    print(
        "  train = {} -> {} | label_end = {} | label_safe = {} | rows = {} | months = {}".format(
            train_start, train_end, label_end, label_safe, len(train_df), train_df["rebalance_date"].nunique()
        )
    )
    print("  base_features =", row["base_feature_count"], "base_iter =", base_best_iter)
    print("  resid_features =", resid_features, "resid_iter =", resid_best_iter)

    del base_es, resid_es, base_model, residual_model
    del train_for_resid, train_refit_resid, resid_fit_df, resid_inner_df
    gc.collect()
    return row


def export_direct_candidate_model(df_all, variant):
    train_df, train_start, train_end, label_end, label_safe = get_variant_train_df(df_all, variant)
    boundary_audit = get_train_boundary_audit(df_all, train_start, train_end, label_end)
    if train_df.empty:
        raise ValueError("train_df empty for {}".format(variant["research_version"]))

    fit_df, inner_valid_df = split_inner_train_valid(train_df)
    candidate_cols = [col for col in variant["candidate_cols"] if col in train_df.columns]
    base_features, base_removed = select_features_train_only(fit_df, candidate_cols)

    base_es = fit_lgb_es(
        fit_df,
        inner_valid_df,
        base_features,
        "alpha_1m",
        variant["base_params"],
        seed=42,
    )
    base_best_iter = base_es["metrics"]["best_iter"]
    forced_iter = variant.get("forced_iter", None)
    model_iter = base_best_iter if forced_iter is None else int(forced_iter)
    iteration_policy = "early_stop" if forced_iter is None else "fixed_iter_{}".format(int(forced_iter))

    base_model = fit_lgb_fixed(
        train_df,
        base_features,
        "alpha_1m",
        variant["base_params"],
        seed=42,
        num_boost_round=model_iter,
    )

    bundle = {
        "objective": "v210_refit_fixed_iter_overlay",
        "research_version": variant["research_version"],
        "benchmark": "000906.XSHG",
        "train_start": train_start,
        "train_end": train_end,
        "label_end": label_end,
        "require_label_end_within_train": label_safe,
        "train_boundary_audit": boundary_audit,
        "data_file": variant["data_file"],
        "protocol": "inner_valid_iter_diagnostic_then_full_train_refit_direct",
        "iteration_policy": iteration_policy,
        "forced_iter": forced_iter,
        "es_best_iter": base_best_iter,
        "model_iter": model_iter,
        "base_params": variant["base_params"],
        "base_model": base_model["model"],
        "base_feature_cols": base_model["feature_cols"],
        "base_fill_values": base_model["fill_values"],
        "base_best_iter": model_iter,
        "base_inner_metrics": base_es["metrics"],
        "base_removed_features": base_removed,
        "residual_feature_cols": [],
        "residual_fill_values": {},
        "overlay_weight": 0.0,
        "overlay_mode": "direct",
        "top_n_candidates": TOP_N_CANDIDATES,
        "stock_num": TOP_N_PORTFOLIO,
        "industry_cap_ratio": INDUSTRY_CAP_RATIO,
        "uses_time_weight": False,
        "uses_current_valid_for_training": False,
        "final_role": variant.get("final_role", ""),
        "training_policy": variant.get("training_policy", ""),
        "param_set": variant.get("param_set", ""),
        "requires_v4_feature_adapter": bool(variant.get("requires_v4_feature_adapter", False)),
        "requires_industry_relative_adapter": bool(variant.get("requires_industry_relative_adapter", False)),
    }
    with open(variant["model_out"], "wb") as f:
        pickle.dump(bundle, f)

    row = {
        "research_version": variant["research_version"],
        "model_out": variant["model_out"],
        "data_file": variant["data_file"],
        "train_start": train_start,
        "train_end": train_end,
        "label_end": label_end,
        "require_label_end_within_train": label_safe,
        "train_row_count": len(train_df),
        "train_month_count": train_df["rebalance_date"].nunique(),
        "pre_label_row_count": boundary_audit["pre_label_row_count"],
        "pre_label_month_count": boundary_audit["pre_label_month_count"],
        "label_boundary_crossing_row_count": boundary_audit["label_boundary_crossing_row_count"],
        "label_boundary_crossing_month_count": boundary_audit["label_boundary_crossing_month_count"],
        "label_safe_row_count": boundary_audit["label_safe_row_count"],
        "label_safe_month_count": boundary_audit["label_safe_month_count"],
        "max_train_rebalance_date": boundary_audit["max_train_rebalance_date"],
        "max_train_next_date": boundary_audit["max_train_next_date"],
        "base_feature_count": len(base_model["feature_cols"]),
        "base_removed_count": len(base_removed),
        "base_best_iter": model_iter,
        "es_best_iter": base_best_iter,
        "forced_iter": forced_iter,
        "iteration_policy": iteration_policy,
        "residual_feature_count": 0,
        "residual_features": "",
        "residual_best_iter": np.nan,
        "final_role": variant.get("final_role", ""),
        "training_policy": variant.get("training_policy", ""),
        "param_set": variant.get("param_set", ""),
        "requires_v4_feature_adapter": bool(variant.get("requires_v4_feature_adapter", False)),
        "requires_industry_relative_adapter": bool(variant.get("requires_industry_relative_adapter", False)),
    }
    print("\nexported:", variant["research_version"])
    print("  ->", variant["model_out"])
    print(
        "  train = {} -> {} | label_end = {} | label_safe = {} | rows = {} | months = {}".format(
            train_start, train_end, label_end, label_safe, len(train_df), train_df["rebalance_date"].nunique()
        )
    )
    print("  base_features =", row["base_feature_count"], "base_iter =", model_iter, "es_iter =", base_best_iter)
    print("  direct mode: no residual model")

    del base_es, base_model
    gc.collect()
    return row


print("candidate export helpers ready")


# %%
# =========================
# Cell 3: V5.4 recall-rerank portfolio construction lab
# =========================
# Research purpose:
# - Keep the V5.3 winning rerank idea: pool-relative features + top20 binary classifier.
# - Do not change model family. Only test recall pool size, final holding count,
#   and final portfolio construction score.
# - Keep OOF recall pool training discipline.

V54_LABEL_END = "2025-03-31"
V54_TRAIN_START = "2020-04-01"
V54_TRAIN_END = "2025-03-31"
V54_TEST_START = "2025-04-01"
V54_TEST_END = "2026-04-30"
V54_RECALL_ITER = 20
V54_RERANK_ITER = 20
V54_RECALL_KS = [40, 50, 60]
V54_FINAL_TOP_NS = [6, 8, 10]
V54_SELECTION_MODES = ["prob_topN", "prob_threshold_then_topN", "prob_x_recall_blend"]
V54_PROB_THRESHOLD = 0.50
V54_RECALL_BLEND_WEIGHT = 0.35

V54_OUT_BUNDLE = "model_candidate_v54_portfolio_oof_rerank_v410_q1.pkl"
V54_OUT_SUMMARY = "wf_v54_portfolio_oof_rerank_summary.csv"
V54_OUT_MONTHLY_DETAIL = "wf_v54_portfolio_oof_rerank_monthly_detail.csv"
V54_OUT_POOL_DIAGNOSTIC = "wf_v54_portfolio_recall_pool_diagnostic.csv"
V54_OUT_FOLD_SUMMARY = "wf_v54_portfolio_oof_fold_summary.csv"
V54_OUT_FEATURES = "wf_v54_portfolio_rerank_features.csv"

V54_OOF_FOLDS = [
    {
        "fold": "oof_2023",
        "recall_train_start": "2020-04-01",
        "recall_train_end": "2022-12-31",
        "valid_start": "2023-01-01",
        "valid_end": "2023-12-31",
    },
    {
        "fold": "oof_2024",
        "recall_train_start": "2020-04-01",
        "recall_train_end": "2023-12-31",
        "valid_start": "2024-01-01",
        "valid_end": "2024-12-31",
    },
    {
        "fold": "oof_2025q1",
        "recall_train_start": "2020-04-01",
        "recall_train_end": "2024-12-31",
        "valid_start": "2025-01-01",
        "valid_end": "2025-03-31",
    },
]

V54_POOL_RELATIVE_SOURCE_COLS = [
    "cash_flow_to_price_ratio",
    "book_to_price_ratio",
    "earnings_yield",
    "roe_ttm",
    "roa_ttm",
    "Rank1M",
    "sharpe_ratio_60",
    "liquidity",
    "beta",
    "px_ret_20",
    "px_close_to_ma60",
    "px_drawdown_60",
    "liq_money_ratio_20_60",
    "liq_paused_count_20",
    "ts_cash_flow_to_price_ratio_rank_mean_3m",
    "ts_Rank1M_rank_chg_1m",
]


def cross_section_zscore(df, score_col):
    def _z(s):
        s = pd.Series(s).astype(float)
        std = s.std()
        if pd.isnull(std) or std <= 0:
            return s * 0.0
        return (s - s.mean()) / std
    return df.groupby("rebalance_date")[score_col].transform(_z)


def score_model_to_col(model_meta, df, out_col):
    out = df.copy()
    out[out_col] = predict_model(model_meta, out)
    out[out_col + "_z"] = cross_section_zscore(out, out_col)
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


def select_recall_pool(scored_df, recall_top_k):
    rows = []
    for dt, gdf in scored_df.groupby("rebalance_date"):
        if len(gdf) < recall_top_k:
            continue
        pool = gdf.nlargest(recall_top_k, "recall_score_z").copy()
        pool = pool.sort_values("recall_score_z", ascending=False).copy()
        pool["recall_rank"] = np.arange(1, len(pool) + 1)
        pool["recall_rank_pct"] = 1.0 - (pool["recall_rank"] - 1.0) / max(1.0, len(pool) - 1.0)
        pool["recall_top_k"] = recall_top_k
        rows.append(pool)
    if len(rows) == 0:
        return pd.DataFrame(columns=list(scored_df.columns) + ["recall_rank", "recall_rank_pct", "recall_top_k"])
    return pd.concat(rows, ignore_index=True)


def add_pool_relative_features(pool_df, source_cols):
    out = pool_df.copy()
    out["recall_score_z_pool_rank"] = out.groupby("rebalance_date")["recall_score_z"].rank(pct=True)
    out["recall_score_z_minus_pool_median"] = (
        out["recall_score_z"] - out.groupby("rebalance_date")["recall_score_z"].transform("median")
    )
    new_cols = [
        "recall_score_z",
        "recall_rank_pct",
        "recall_score_z_pool_rank",
        "recall_score_z_minus_pool_median",
    ]
    for col in source_cols:
        if col not in out.columns:
            continue
        med_col = "poolrel_{}_minus_pool_median".format(col)
        rank_col = "poolrel_{}_rank_in_pool".format(col)
        industry_rank_col = "poolrel_{}_rank_in_pool_industry".format(col)
        out[med_col] = out[col] - out.groupby("rebalance_date")[col].transform("median")
        out[rank_col] = out.groupby("rebalance_date")[col].transform(lambda s: s.rank(pct=True))
        out[industry_rank_col] = out.groupby(["rebalance_date", "industry_bucket"])[col].transform(lambda s: s.rank(pct=True))
        new_cols.extend([med_col, rank_col, industry_rank_col])
    return out.replace([np.inf, -np.inf], np.nan), unique_keep_order(new_cols)


def add_top20_binary_target(pool_df):
    out = pool_df.copy()
    rank_pct = out.groupby("rebalance_date")["alpha_1m"].rank(pct=True, method="first")
    out["rerank_target"] = (rank_pct >= 0.80).astype(float)
    return out


def get_binary_lgb_params():
    params = dict(BASE_PARAMS_FF10)
    params["objective"] = "binary"
    params["metric"] = "binary_logloss"
    return params


def calc_final_score(scored_pool, selection_mode):
    out = scored_pool.copy()
    if selection_mode == "prob_topN":
        out["final_score"] = out["rerank_score_z"]
    elif selection_mode == "prob_threshold_then_topN":
        out["final_score"] = out["rerank_score_z"]
        out.loc[out["rerank_score"] < V54_PROB_THRESHOLD, "final_score"] = -999.0 + out.loc[
            out["rerank_score"] < V54_PROB_THRESHOLD, "rerank_score_z"
        ]
    elif selection_mode == "prob_x_recall_blend":
        out["final_score"] = out["rerank_score_z"] + V54_RECALL_BLEND_WEIGHT * out["recall_score_z"]
    else:
        raise ValueError("unsupported selection_mode: {}".format(selection_mode))
    return out


def eval_selected_portfolio(scored_df, final_top_n, version, selection_mode, recall_top_k):
    rows = []
    max_per_industry = max(1, int(np.floor(final_top_n * INDUSTRY_CAP_RATIO)))
    scored_df = calc_final_score(scored_df, selection_mode)
    for dt, gdf in scored_df.groupby("rebalance_date"):
        if len(gdf) < final_top_n:
            continue
        ordered = gdf.sort_values("final_score", ascending=False).copy()
        industry_map = dict(zip(ordered["stock"], ordered["industry_bucket"]))
        selected = build_industry_neutral_targets(
            sorted_stocks=ordered["stock"].tolist(),
            industry_map=industry_map,
            target_num=final_top_n,
            max_per_industry=max_per_industry,
        )
        selected_df = ordered[ordered["stock"].isin(selected)].copy()
        if selected_df.empty:
            continue
        rows.append({
            "version": version,
            "rebalance_date": dt,
            "recall_top_k": recall_top_k,
            "final_top_n": final_top_n,
            "selection_mode": selection_mode,
            "selected_alpha": float(selected_df["alpha_1m"].mean()),
            "selected_count": int(len(selected_df)),
            "industry_count": int(selected_df["industry_bucket"].nunique()),
            "rank_ic_in_pool": safe_rank_ic(ordered["final_score"], ordered["alpha_1m"]),
            "mean_rerank_prob": float(selected_df["rerank_score"].mean()) if "rerank_score" in selected_df.columns else np.nan,
            "selected_stocks": "|".join(selected),
            "candidate_stocks": "|".join(ordered["stock"].tolist()),
        })
    return pd.DataFrame(rows)


def eval_baseline(scored_df):
    rows = []
    baseline_pool = select_recall_pool(scored_df, 30)
    max_per_industry = max(1, int(np.floor(10 * INDUSTRY_CAP_RATIO)))
    for dt, gdf in baseline_pool.groupby("rebalance_date"):
        ordered = gdf.sort_values("recall_score_z", ascending=False).copy()
        industry_map = dict(zip(ordered["stock"], ordered["industry_bucket"]))
        selected = build_industry_neutral_targets(
            sorted_stocks=ordered["stock"].tolist(),
            industry_map=industry_map,
            target_num=10,
            max_per_industry=max_per_industry,
        )
        selected_df = ordered[ordered["stock"].isin(selected)].copy()
        if selected_df.empty:
            continue
        rows.append({
            "version": "baseline_v410_fixed20_recall30_top10",
            "rebalance_date": dt,
            "recall_top_k": 30,
            "final_top_n": 10,
            "selection_mode": "baseline_recall_score",
            "selected_alpha": float(selected_df["alpha_1m"].mean()),
            "selected_count": int(len(selected_df)),
            "industry_count": int(selected_df["industry_bucket"].nunique()),
            "rank_ic_in_pool": safe_rank_ic(ordered["recall_score_z"], ordered["alpha_1m"]),
            "mean_rerank_prob": np.nan,
            "selected_stocks": "|".join(selected),
            "candidate_stocks": "|".join(ordered["stock"].tolist()),
        })
    return pd.DataFrame(rows)


def calc_t_stat(s):
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        return np.nan
    std = s.std(ddof=1)
    if pd.isnull(std) or std <= 0:
        return np.nan
    return float(s.mean() / std * np.sqrt(len(s)))


def calc_max_drawdown_from_alpha(s):
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if s.empty:
        return np.nan
    curve = (1.0 + s).cumprod()
    dd = curve / curve.cummax() - 1.0
    return float(dd.min())


def summarize_detail(detail_df):
    rows = []
    if detail_df.empty:
        return pd.DataFrame()
    for version, gdf in detail_df.groupby("version"):
        gdf = gdf.sort_values("rebalance_date").copy()
        rows.append({
            "version": version,
            "months": int(len(gdf)),
            "mean_alpha": float(gdf["selected_alpha"].mean()),
            "cum_alpha": float((1.0 + gdf["selected_alpha"]).prod() - 1.0),
            "win_rate": float((gdf["selected_alpha"] > 0).mean()),
            "t_stat": calc_t_stat(gdf["selected_alpha"]),
            "max_drawdown": calc_max_drawdown_from_alpha(gdf["selected_alpha"]),
            "worst_month": float(gdf["selected_alpha"].min()),
            "avg_industry_count": float(gdf["industry_count"].mean()),
            "mean_rank_ic_in_pool": float(gdf["rank_ic_in_pool"].mean()),
            "mean_rerank_prob": float(gdf["mean_rerank_prob"].mean()) if "mean_rerank_prob" in gdf.columns else np.nan,
            "recall_top_k": int(gdf["recall_top_k"].iloc[0]),
            "final_top_n": int(gdf["final_top_n"].iloc[0]),
            "selection_mode": str(gdf["selection_mode"].iloc[0]),
        })
    return pd.DataFrame(rows).sort_values(["mean_alpha", "t_stat"], ascending=[False, False])


def calc_recall_pool_diagnostic(scored_df):
    rows = []
    for dt, gdf in scored_df.groupby("rebalance_date"):
        if len(gdf) < max(V54_RECALL_KS):
            continue
        true_top10 = set(gdf.nlargest(10, "alpha_1m")["stock"].tolist())
        true_top20 = set(gdf.nlargest(20, "alpha_1m")["stock"].tolist())
        for k in V54_RECALL_KS:
            pool = gdf.nlargest(k, "recall_score_z").copy()
            pool_set = set(pool["stock"].tolist())
            true_topk = set(gdf.nlargest(k, "alpha_1m")["stock"].tolist())
            union = pool_set.union(true_topk)
            rows.append({
                "rebalance_date": dt,
                "recall_top_k": k,
                "pool_alpha": float(pool["alpha_1m"].mean()),
                "pool_win_rate": float((pool["alpha_1m"] > 0).mean()),
                "true_top10_recall_rate": len(pool_set.intersection(true_top10)) / 10.0,
                "true_top20_recall_rate": len(pool_set.intersection(true_top20)) / 20.0,
                "true_topk_iou": len(pool_set.intersection(true_topk)) / float(len(union)) if len(union) else np.nan,
                "pool_rank_ic": safe_rank_ic(pool["recall_score_z"], pool["alpha_1m"]),
            })
    return pd.DataFrame(rows)


def get_label_safe_slice(df, start_date, end_date, label_end):
    out = df[
        (df["rebalance_date"] >= pd.Timestamp(start_date)) &
        (df["rebalance_date"] <= pd.Timestamp(end_date))
    ].copy()
    out = out[pd.to_datetime(out["next_date"]) <= pd.Timestamp(label_end)].copy()
    return out


def train_recall_model(train_df, candidate_cols, seed):
    fit_df, _inner_valid_df = split_inner_train_valid(train_df)
    recall_features, recall_removed = select_features_train_only(fit_df, candidate_cols)
    recall_model = fit_lgb_fixed(
        train_df=train_df,
        feature_cols=recall_features,
        target_col="alpha_1m",
        params=BASE_PARAMS_FF10,
        seed=seed,
        num_boost_round=V54_RECALL_ITER,
    )
    return recall_model, recall_removed


def train_binary_rerank_model(oof_pool_df, feature_cols):
    train_df = add_top20_binary_target(oof_pool_df)
    fit_df, _inner_valid_df = split_inner_train_valid(train_df)
    rerank_features, rerank_removed = select_features_train_only(fit_df, feature_cols)
    rerank_model = fit_lgb_fixed(
        train_df=train_df,
        feature_cols=rerank_features,
        target_col="rerank_target",
        params=get_binary_lgb_params(),
        seed=20260507,
        num_boost_round=V54_RERANK_ITER,
    )
    return rerank_model, rerank_removed


def build_oof_recall_pools(df_all, candidate_cols, recall_ks):
    pool_frames = []
    fold_rows = []
    for i, fold in enumerate(V54_OOF_FOLDS):
        fold_train = get_label_safe_slice(
            df_all,
            fold["recall_train_start"],
            fold["recall_train_end"],
            fold["recall_train_end"],
        )
        fold_valid = get_label_safe_slice(
            df_all,
            fold["valid_start"],
            fold["valid_end"],
            V54_LABEL_END,
        )
        if fold_train.empty or fold_valid.empty:
            fold_rows.append({
                "fold": fold["fold"],
                "status": "skipped_empty",
                "train_rows": int(len(fold_train)),
                "valid_rows": int(len(fold_valid)),
            })
            continue
        print("\nOOF recall fold:", fold["fold"])
        print("  train rows =", len(fold_train), "months =", fold_train["rebalance_date"].nunique())
        print("  valid rows =", len(fold_valid), "months =", fold_valid["rebalance_date"].nunique())
        recall_model, recall_removed = train_recall_model(fold_train, candidate_cols, seed=20260507 + i)
        valid_scored = score_model_to_col(recall_model, fold_valid, "recall_score")
        fold_pool_rows = 0
        for k in recall_ks:
            pool = select_recall_pool(valid_scored, k)
            if not pool.empty:
                pool["oof_fold"] = fold["fold"]
                pool_frames.append(pool)
                fold_pool_rows += len(pool)
        fold_rows.append({
            "fold": fold["fold"],
            "status": "ok",
            "train_rows": int(len(fold_train)),
            "train_months": int(fold_train["rebalance_date"].nunique()),
            "valid_rows": int(len(fold_valid)),
            "valid_months": int(fold_valid["rebalance_date"].nunique()),
            "feature_count": int(len(recall_model["feature_cols"])),
            "removed_feature_count": int(len(recall_removed)),
            "pool_rows": int(fold_pool_rows),
        })
        del recall_model, valid_scored, fold_train, fold_valid
        gc.collect()
    if len(pool_frames) == 0:
        raise ValueError("OOF recall pools are empty")
    return pd.concat(pool_frames, ignore_index=True), pd.DataFrame(fold_rows)


df_all = ensure_v4_dataset()
df_all, _v45_industry_relative_cols = add_v45_industry_relative_features(df_all)
df_all["industry_bucket"] = df_all.get("industry_bucket", "UNKNOWN")
df_all["industry_bucket"] = df_all["industry_bucket"].fillna("UNKNOWN").astype(str)

v4_base_cols = [col for col in BASE_FACTOR_COLS if col in df_all.columns]
v4_hybrid_light_cols = [col for col in HYBRID_LIGHT_EXTRA_COLS if col in df_all.columns]
v54_candidate_cols = unique_keep_order(v4_base_cols + v4_hybrid_light_cols)
v54_pool_source_cols = [col for col in V54_POOL_RELATIVE_SOURCE_COLS if col in df_all.columns]
if len(v54_candidate_cols) == 0:
    raise ValueError("v54_candidate_cols is empty")
if len(v54_pool_source_cols) == 0:
    raise ValueError("v54_pool_source_cols is empty")

train_df = get_label_safe_slice(df_all, V54_TRAIN_START, V54_TRAIN_END, V54_LABEL_END)
test_df = df_all[
    (df_all["rebalance_date"] >= pd.Timestamp(V54_TEST_START)) &
    (df_all["rebalance_date"] <= pd.Timestamp(V54_TEST_END))
].copy()
boundary_audit = get_train_boundary_audit(df_all, V54_TRAIN_START, V54_TRAIN_END, V54_LABEL_END)
print("V5.4 train rows =", len(train_df), "months =", train_df["rebalance_date"].nunique())
print("V5.4 test rows =", len(test_df), "months =", test_df["rebalance_date"].nunique())
print("base candidate cols =", len(v54_candidate_cols), "pool source cols =", len(v54_pool_source_cols))
print("boundary audit =", boundary_audit)

oof_pool_raw, fold_summary_df = build_oof_recall_pools(df_all, v54_candidate_cols, V54_RECALL_KS)
fold_summary_df.to_csv(V54_OUT_FOLD_SUMMARY, index=False)

final_recall_model, final_recall_removed = train_recall_model(train_df, v54_candidate_cols, seed=20260507)
test_scored = score_model_to_col(final_recall_model, test_df, "recall_score")
pool_diag = calc_recall_pool_diagnostic(test_scored)
pool_diag.to_csv(V54_OUT_POOL_DIAGNOSTIC, index=False)

detail_frames = [eval_baseline(test_scored)]
rerank_bundles = {}
feature_rows = []

for recall_k in V54_RECALL_KS:
    print("\ntraining V5.4 rerank classifier for recall_k =", recall_k)
    oof_pool_k = oof_pool_raw[oof_pool_raw["recall_top_k"] == recall_k].copy()
    oof_pool_k, pool_relative_cols = add_pool_relative_features(oof_pool_k, v54_pool_source_cols)
    rerank_model, rerank_removed = train_binary_rerank_model(oof_pool_k, pool_relative_cols)
    test_pool_raw = select_recall_pool(test_scored, recall_k)
    test_pool, _ = add_pool_relative_features(test_pool_raw, v54_pool_source_cols)
    test_pool_scored = score_model_to_col(rerank_model, test_pool, "rerank_score")

    for final_top_n in V54_FINAL_TOP_NS:
        for selection_mode in V54_SELECTION_MODES:
            version = "v54_recall{}_top{}_{}".format(recall_k, final_top_n, selection_mode)
            detail = eval_selected_portfolio(
                scored_df=test_pool_scored,
                final_top_n=final_top_n,
                version=version,
                selection_mode=selection_mode,
                recall_top_k=recall_k,
            )
            detail_frames.append(detail)

    rerank_bundles["recall{}".format(recall_k)] = {
        "recall_top_k": int(recall_k),
        "rerank_model": rerank_model["model"],
        "rerank_feature_cols": list(rerank_model["feature_cols"]),
        "rerank_fill_values": dict(rerank_model["fill_values"]),
        "rerank_removed_features": list(rerank_removed),
        "oof_train_pool_rows": int(len(oof_pool_k)),
        "oof_train_pool_months": int(oof_pool_k["rebalance_date"].nunique()),
    }
    for col in pool_relative_cols:
        feature_rows.append({"recall_top_k": recall_k, "rerank_feature": col})
    del oof_pool_k, test_pool_raw, test_pool, test_pool_scored, rerank_model
    gc.collect()

detail_df = pd.concat(detail_frames, ignore_index=True) if len(detail_frames) else pd.DataFrame()
summary_df = summarize_detail(detail_df)

detail_df.to_csv(V54_OUT_MONTHLY_DETAIL, index=False)
summary_df.to_csv(V54_OUT_SUMMARY, index=False)
pd.DataFrame(feature_rows).to_csv(V54_OUT_FEATURES, index=False)

bundle = {
    "objective": "v54_portfolio_oof_recall_rerank",
    "research_version": "candidate_v54_portfolio_oof_rerank_v410_fixed20_rolling5y_2025q1",
    "benchmark": BENCHMARK,
    "universe": UNIVERSE_NAME,
    "universe_index": UNIVERSE_INDEX,
    "train_start": V54_TRAIN_START,
    "train_end": V54_TRAIN_END,
    "label_end": V54_LABEL_END,
    "test_start": V54_TEST_START,
    "test_end": V54_TEST_END,
    "require_label_end_within_train": True,
    "train_boundary_audit": boundary_audit,
    "data_file": V4_DATA_FILE,
    "protocol": "v54_top20_binary_pool_relative_oof_rerank_portfolio_search",
    "recall_model_family": "v410_fixed_iter20_rolling5y_2025q1",
    "recall_model": final_recall_model["model"],
    "recall_feature_cols": list(final_recall_model["feature_cols"]),
    "recall_fill_values": dict(final_recall_model["fill_values"]),
    "recall_removed_features": list(final_recall_removed),
    "recall_iter": V54_RECALL_ITER,
    "recall_ks": list(V54_RECALL_KS),
    "final_top_ns": list(V54_FINAL_TOP_NS),
    "selection_modes": list(V54_SELECTION_MODES),
    "prob_threshold": V54_PROB_THRESHOLD,
    "recall_blend_weight": V54_RECALL_BLEND_WEIGHT,
    "rerank_iter": V54_RERANK_ITER,
    "base_params": BASE_PARAMS_FF10,
    "rerank_params": get_binary_lgb_params(),
    "candidate_cols": list(v54_candidate_cols),
    "pool_relative_source_cols": list(v54_pool_source_cols),
    "oof_folds": list(V54_OOF_FOLDS),
    "oof_fold_summary": fold_summary_df.to_dict("records"),
    "rerank_models": rerank_bundles,
    "summary": summary_df.to_dict("records"),
    "top_n_candidates": TOP_N_CANDIDATES,
    "stock_num": TOP_N_PORTFOLIO,
    "industry_cap_ratio": INDUSTRY_CAP_RATIO,
    "requires_v4_feature_adapter": True,
}
with open(V54_OUT_BUNDLE, "wb") as f:
    pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

print("\n===== V5.4 Portfolio OOF Rerank Summary =====")
print(summary_df)
print("saved ->", V54_OUT_SUMMARY)
print("saved ->", V54_OUT_MONTHLY_DETAIL)
print("saved ->", V54_OUT_POOL_DIAGNOSTIC)
print("saved ->", V54_OUT_FOLD_SUMMARY)
print("saved ->", V54_OUT_FEATURES)
print("saved ->", V54_OUT_BUNDLE)
