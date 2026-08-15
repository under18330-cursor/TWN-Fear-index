import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from twn_fear_greed import data_sources as ds


def test_parse_roc_date_converts_roc_year():
    # "1150803" = ROC year 115 (1911 + 115 = 2026), month 08, day 03
    result = ds._parse_roc_date(pd.Series(["1150803", "1150814"]))
    assert list(result) == [pd.Timestamp("2026-08-03"), pd.Timestamp("2026-08-14")]


def test_parse_roc_slash_date_converts_roc_year():
    # classic www.twse.com.tw/rwd endpoints slash-delimit the same ROC date
    result = ds._parse_roc_slash_date(pd.Series(["115/08/03", "115/08/14"]))
    assert list(result) == [pd.Timestamp("2026-08-03"), pd.Timestamp("2026-08-14")]


def test_index_price_from_classic_rows_parses_slash_dates_and_commas():
    rows = [
        ["115/08/03", "42,780.42", "43,784.19", "42,780.42", "43,386.41"],
        ["115/08/14", "46,103.81", "46,402.60", "45,798.31", "45,811.01"],
    ]
    result = ds._index_price_from_classic_rows(rows)
    assert list(result.columns) == ["close"]
    assert result.loc[pd.Timestamp("2026-08-03"), "close"] == 43386.41
    assert result.loc[pd.Timestamp("2026-08-14"), "close"] == 45811.01
    assert result.index.is_monotonic_increasing


def test_index_price_from_classic_rows_empty_month():
    result = ds._index_price_from_classic_rows([])
    assert result.empty
    assert list(result.columns) == ["close"]


def test_foreign_cash_from_bfi82u_picks_the_right_row():
    payload = {
        "date": "20260814",
        "fields": ["單位名稱", "買進金額", "賣出金額", "買賣差額"],
        "data": [
            ["自營商(自行買賣)", "11,160,429,833", "13,896,418,209", "-2,735,988,376"],
            ["外資及陸資(不含外資自營商)", "432,487,254,209", "387,135,923,788", "45,351,330,421"],
            ["合計", "511,271,838,131", "459,609,095,937", "51,662,742,194"],
        ],
    }
    result = ds._foreign_cash_from_bfi82u(payload)
    assert result.loc[pd.Timestamp("2026-08-14"), "net_buy_sell"] == 45351330421


def test_foreign_cash_from_bfi82u_empty_data():
    result = ds._foreign_cash_from_bfi82u({"date": "20260814", "fields": [], "data": []})
    assert result.empty
    assert list(result.columns) == ["net_buy_sell"]


def test_index_price_from_hist_parses_roc_dates_and_close():
    raw = [
        {"Date": "1150803", "OpeningIndex": "42780.42", "HighestIndex": "43784.19",
         "LowestIndex": "42780.42", "ClosingIndex": "43386.41"},
        {"Date": "1150814", "OpeningIndex": "46103.81", "HighestIndex": "46402.60",
         "LowestIndex": "45798.31", "ClosingIndex": "45811.01"},
    ]
    result = ds._index_price_from_hist(raw)
    assert list(result.columns) == ["close"]
    assert result.loc[pd.Timestamp("2026-08-03"), "close"] == 43386.41
    assert result.loc[pd.Timestamp("2026-08-14"), "close"] == 45811.01
    assert result.index.is_monotonic_increasing


def test_index_price_from_hist_empty_input():
    result = ds._index_price_from_hist([])
    assert result.empty
    assert list(result.columns) == ["close"]


def test_put_call_from_ratio_parses_ad_dates():
    raw = [
        {"Date": "20260814", "PutVolume": "326821", "CallVolume": "316754",
         "PutCallVolumeRatio%": "103.18", "PutOI": "65051", "CallOI": "56564"},
        {"Date": "20260715", "PutVolume": "369841", "CallVolume": "324234",
         "PutCallVolumeRatio%": "114.07", "PutOI": "35475", "CallOI": "34248"},
    ]
    result = ds._put_call_from_ratio(raw)
    assert list(result.columns) == ["put_volume", "call_volume"]
    assert result.loc[pd.Timestamp("2026-08-14"), "put_volume"] == 326821
    assert result.loc[pd.Timestamp("2026-08-14"), "call_volume"] == 316754
    assert result.index.is_monotonic_increasing


def test_margin_balance_from_stockwise_sums_across_stocks():
    raw = [
        {"股票代號": "2330", "融資今日餘額": "9427"},
        {"股票代號": "2317", "融資今日餘額": "1631"},
        {"股票代號": "0050", "融資今日餘額": ""},  # blank -> excluded from sum
    ]
    as_of = pd.Timestamp("2026-08-14")
    result = ds._margin_balance_from_stockwise(raw, as_of)
    assert result.loc[as_of, "margin_balance"] == 9427 + 1631


def test_margin_balance_from_stockwise_empty_input():
    result = ds._margin_balance_from_stockwise([], pd.Timestamp("2026-08-14"))
    assert result.empty
    assert list(result.columns) == ["margin_balance"]


def test_foreign_futures_oi_filters_contract_and_item():
    raw = [
        {"Date": "20260814", "ContractCode": "臺股期貨", "Item": "自營商", "OpenInterest(Net)": "1464"},
        {"Date": "20260814", "ContractCode": "臺股期貨", "Item": "外資及陸資", "OpenInterest(Net)": "-85179"},
        {"Date": "20260814", "ContractCode": "電子期貨", "Item": "外資及陸資", "OpenInterest(Net)": "999999"},
    ]
    fallback = pd.Timestamp("2020-01-01")  # should be ignored: a real match carries its own date
    result = ds._foreign_futures_oi_from_institutional(raw, fallback)
    row_date = pd.Timestamp("2026-08-14")
    assert row_date in result.index
    assert result.loc[row_date, "futures_net_oi"] == -85179
    assert result.loc[row_date, "net_buy_sell"] == 0.0


def test_foreign_futures_oi_no_match_yields_nan():
    raw = [{"Date": "20260814", "ContractCode": "電子期貨", "Item": "外資及陸資", "OpenInterest(Net)": "100"}]
    as_of = pd.Timestamp("2026-08-14")
    result = ds._foreign_futures_oi_from_institutional(raw, as_of)
    assert pd.isna(result.loc[as_of, "futures_net_oi"])


def test_vix_close_from_taifex_file_reads_last_1_min_avg():
    # real file shape: tab-separated, CP950-encoded, one row per 15s,
    # trailing summary row labeled "Last 1 min AVG"
    body = (
        "20260814\t9000000\t\t\t30.41\n"
        "20260814\t9001500\t\t\t30.41\n"
        "20260814\t13450000\t\t\t30.22\n"
        "20260814\tLast 1 min AVG\t\t\t30.23\n"
    ).encode("cp950")
    date = pd.Timestamp("2026-08-14")
    result = ds._vix_close_from_taifex_file(body, date)
    assert result.loc[date, "taiex_vix"] == 30.23


def test_vix_close_from_taifex_file_no_summary_row_is_empty():
    body = "20260814\t9000000\t\t\t30.41\n".encode("cp950")
    result = ds._vix_close_from_taifex_file(body, pd.Timestamp("2026-08-14"))
    assert result.empty
    assert list(result.columns) == ["taiex_vix"]


def test_vix_from_monthly_file_parses_one_row_per_trading_day():
    body = (
        "名稱\t時間\t\t\t前1分鐘\n"
        "--------\t-------------------\t--------------------\t-------------------\n"
        "20260803\t13450000\t\t\t39.46\t\t39.47\n"
        "20260814\t13450000\t\t\t30.22\t\t30.23\n"
    ).encode("cp950")
    result = ds._vix_from_monthly_file(body)
    assert list(result.columns) == ["taiex_vix"]
    assert result.loc[pd.Timestamp("2026-08-03"), "taiex_vix"] == 39.47
    assert result.loc[pd.Timestamp("2026-08-14"), "taiex_vix"] == 30.23
    assert result.index.is_monotonic_increasing


def test_vix_from_monthly_file_detects_the_200_status_404_page():
    body = (
        "<HTML xmlns='http://www.w3.org/1999/xhtml' lang='zh-TW'>"
        "<title>404</title><body>content not found</body></html>"
    ).encode("cp950")
    result = ds._vix_from_monthly_file(body)
    assert result.empty
    assert list(result.columns) == ["taiex_vix"]


def test_last_probable_trading_day_rolls_weekend_back_to_friday():
    saturday = pd.Timestamp("2026-08-15")
    sunday = pd.Timestamp("2026-08-16")
    friday = pd.Timestamp("2026-08-14")
    assert ds._last_probable_trading_day(saturday) == friday
    assert ds._last_probable_trading_day(sunday) == friday


def test_last_probable_trading_day_leaves_a_weekday_alone():
    monday = pd.Timestamp("2026-08-17")
    assert ds._last_probable_trading_day(monday) == monday


MI_INDEX_FIELDS = [
    "證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額",
    "開盤價", "最高價", "最低價", "收盤價", "漲跌(+/-)", "漲跌價差",
    "最後揭示買價", "最後揭示買量", "最後揭示賣價", "最後揭示賣量", "本益比",
]


def _mi_index_row(code, close, volume, direction):
    tag = {"up": "<p style= color:red>+</p>", "down": "<p style= color:green>-</p>", "flat": "<p> </p>"}[direction]
    return [code, "名稱", volume, "10", "0", close, close, close, close, tag, "0", "", "", "", "", ""]


def _mi_index_payload(rows):
    return {"stat": "OK", "date": "20260803", "tables": [{"fields": MI_INDEX_FIELDS, "data": rows}]}


def test_market_snapshot_from_mi_index_keeps_only_common_stock_codes():
    rows = [
        _mi_index_row("2330", "500.0", "1000", "up"),
        _mi_index_row("0050", "150.0", "2000", "down"),      # ETF -- excluded
        _mi_index_row("00679B", "16.0", "3000", "up"),        # bond ETF -- excluded
        _mi_index_row("020000", "10.0", "500", "up"),         # warrant -- excluded
    ]
    result = ds._market_snapshot_from_mi_index(_mi_index_payload(rows))
    assert list(result.index) == ["2330"]
    assert result.loc["2330", "close"] == 500.0
    assert result.loc["2330", "volume"] == 1000
    assert result.loc["2330", "is_up"] and not result.loc["2330", "is_down"]


def test_market_snapshot_from_mi_index_flat_day_is_neither_up_nor_down():
    rows = [_mi_index_row("2330", "500.0", "1000", "flat")]
    result = ds._market_snapshot_from_mi_index(_mi_index_payload(rows))
    assert not result.loc["2330", "is_up"]
    assert not result.loc["2330", "is_down"]


def test_market_snapshot_from_mi_index_no_matching_table_is_empty():
    result = ds._market_snapshot_from_mi_index({"stat": "OK", "tables": [{"fields": ["x"], "data": [[1]]}]})
    assert result.empty
    assert list(result.columns) == ["close", "volume", "is_up", "is_down"]


def test_aggregate_breadth_sums_volume_by_direction():
    snapshots = {
        pd.Timestamp("2026-08-03"): pd.DataFrame(
            {"close": [100.0, 50.0], "volume": [10.0, 20.0], "is_up": [True, False], "is_down": [False, True]},
            index=["2330", "2317"],
        ),
    }
    breadth, _ = ds._aggregate_breadth_and_strength(snapshots, new_high_low_window=252)
    assert breadth.loc[pd.Timestamp("2026-08-03"), "advancing_volume"] == 10.0
    assert breadth.loc[pd.Timestamp("2026-08-03"), "declining_volume"] == 20.0


def test_aggregate_strength_flags_new_high_after_full_window():
    dates = pd.bdate_range("2026-01-01", periods=5)
    # stock climbs steadily -> every day after the (short, window=3) lookback is a new high
    closes = [10.0, 11.0, 12.0, 13.0, 14.0]
    snapshots = {
        d: pd.DataFrame({"close": [c], "volume": [1.0], "is_up": [True], "is_down": [False]}, index=["2330"])
        for d, c in zip(dates, closes)
    }
    _, new_high_low = ds._aggregate_breadth_and_strength(snapshots, new_high_low_window=3)
    assert pd.isna(new_high_low.loc[dates[0], "new_highs"])  # not enough lookback yet
    assert pd.isna(new_high_low.loc[dates[1], "new_highs"])
    assert new_high_low.loc[dates[2], "new_highs"] == 1  # first day with a full 3-day window
    assert new_high_low.loc[dates[4], "new_highs"] == 1
    assert new_high_low.loc[dates[4], "new_lows"] == 0


def test_aggregate_breadth_and_strength_empty_input():
    breadth, new_high_low = ds._aggregate_breadth_and_strength({})
    assert breadth.empty and list(breadth.columns) == ["advancing_volume", "declining_volume"]
    assert new_high_low.empty and list(new_high_low.columns) == ["new_highs", "new_lows"]


def test_realized_volatility_is_annualized_and_positive():
    dates = pd.bdate_range("2026-07-01", periods=15)
    # alternating +1%/-1% returns -> nonzero realized vol once window fills
    prices = [100.0]
    for i in range(1, 15):
        prices.append(prices[-1] * (1.01 if i % 2 else 0.99))
    index_price = pd.DataFrame({"close": prices}, index=dates)
    result = ds.realized_volatility(index_price, window=10)
    assert list(result.columns) == ["taiex_vix"]
    assert result["taiex_vix"].dropna().gt(0).all()


def test_fetch_foreign_cash_history_chunks_and_cools_down(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(ds.time, "sleep", lambda s: sleep_calls.append(s))

    def fake_get_json(url, params=None):
        return {
            "date": params["dayDate"],
            "fields": ["單位名稱", "買進金額", "賣出金額", "買賣差額"],
            "data": [["外資及陸資(不含外資自營商)", "1", "1", "100"]],
        }

    monkeypatch.setattr(ds, "_get_json", fake_get_json)

    result = ds.fetch_foreign_cash_history("2024-01-01", "2024-01-05", chunk_size=2, cooldown_seconds=99)

    assert len(result) == 5  # 2024-01-01..05 are all weekdays
    # chunk_size=2 over 5 requests: cooldown fires before request index 2 and 4
    assert sleep_calls.count(99) == 2


def test_fetch_foreign_cash_history_retries_once_then_skips_a_persistently_failing_day(monkeypatch):
    monkeypatch.setattr(ds.time, "sleep", lambda s: None)

    def flaky_get_json(url, params=None):
        if params["dayDate"] == "20240102":
            raise ValueError("simulated WAF block")
        return {
            "date": params["dayDate"],
            "fields": ["單位名稱", "買進金額", "賣出金額", "買賣差額"],
            "data": [["外資及陸資(不含外資自營商)", "1", "1", "50"]],
        }

    monkeypatch.setattr(ds, "_get_json", flaky_get_json)
    result = ds.fetch_foreign_cash_history("2024-01-01", "2024-01-03")

    assert pd.Timestamp("2024-01-02") not in result.index
    assert pd.Timestamp("2024-01-01") in result.index
    assert pd.Timestamp("2024-01-03") in result.index


def test_fetch_foreign_with_start_backfills_cash_and_defaults_futures_to_zero(monkeypatch):
    cash_history = pd.DataFrame(
        {"net_buy_sell": [100.0, 200.0, 300.0]},
        index=pd.DatetimeIndex(["2026-08-10", "2026-08-11", "2026-08-12"], name="date"),
    )
    monkeypatch.setattr(ds, "fetch_foreign_cash_history", lambda start, end=None: cash_history)
    monkeypatch.setattr(ds, "_last_probable_trading_day", lambda now=None: pd.Timestamp("2026-08-12"))
    monkeypatch.setattr(ds, "_get_json", lambda url, params=None: [])
    monkeypatch.setattr(
        ds,
        "_foreign_futures_oi_from_institutional",
        lambda raw, as_of: pd.DataFrame({"futures_net_oi": [-999.0]}, index=pd.DatetimeIndex([as_of], name="date")),
    )

    result = ds.fetch_foreign(start="2026-08-10")

    assert result.loc[pd.Timestamp("2026-08-10"), "futures_net_oi"] == 0.0
    assert result.loc[pd.Timestamp("2026-08-11"), "futures_net_oi"] == 0.0
    assert result.loc[pd.Timestamp("2026-08-12"), "futures_net_oi"] == -999.0
    assert result.loc[pd.Timestamp("2026-08-12"), "net_buy_sell"] == 300.0


def test_fetch_all_raw_data_omits_unimplemented_sources(monkeypatch):
    idx = pd.DataFrame({"close": [100.0, 101.0]}, index=pd.bdate_range("2026-08-01", periods=2))
    monkeypatch.setattr(ds, "fetch_index_price", lambda: idx)
    monkeypatch.setattr(ds, "fetch_options", lambda: pd.DataFrame(
        {"put_volume": [1.0, 2.0], "call_volume": [1.0, 2.0]}, index=idx.index))
    monkeypatch.setattr(ds, "fetch_margin", lambda: pd.DataFrame(
        {"margin_balance": [5.0]}, index=[idx.index[-1]]))
    monkeypatch.setattr(ds, "fetch_foreign", lambda: pd.DataFrame(
        {"net_buy_sell": [0.0], "futures_net_oi": [10.0]}, index=[idx.index[-1]]))

    raw = ds.fetch_all_raw_data()

    assert set(raw) == {"index_price", "options", "vix", "margin", "foreign"}
    assert "new_high_low" not in raw
    assert "breadth" not in raw
    assert "bond_index" not in raw
