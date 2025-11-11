#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Nov 11 00:26:29 2025
Last updated: Nov 11 14:48:25 2025

@author: kieranzhane

UK Electricity Mega Script — 2024 Freeze (paper-ready)

This version hard-locks the analysis window to calendar year 2024:
- Carbon Intensity (API or local CSVs)
- MID market prices (Elexon BMRS)
- Ofgem price cap series (for context lines)
- NESO grid frequency (1-minute, cached by month) — sliced to 2024
- Assumed UK Domestic LV usage profile (half-hourly) — SSEN-independent
- DR potential using 2024 windows
- Full plotting suite + console outputs + CSV/Parquet exports named with 2024:
  * Dual-axis half-hourly & daily series (you can overlay cap if desired)
  * Minutes ≤ £0 per month (full 2024 months only)
  * 2024 hourly profile (free minutes + CI overlay)
  * Completed 2024 month heatmaps; last-12-month heatmap (to 2024-12-31)
  * Exact free-interval broken-bar (per-month, if called)
  * Seasonal/diurnal shares + weekend vs weekday + neg. price depth (tables/plots)
  * Optional logistic regression (statsmodels) saved to text
  * Frequency vs price: hourly medians, hexbin density, monthly correlations

Notes:
- Saves useful CSV/Parquet side files for reproducibility.
- All dates are UTC. Ensure CI/MID alignment on 30-min settlement periods.
"""

import datetime as dt
from typing import Dict, Any, List, Optional, Iterable, Tuple
import glob
import numpy as np
import os
import sys
import json
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pathlib
import re
from datetime import datetime, timezone, timedelta
from matplotlib.colors import PowerNorm
from matplotlib.ticker import PercentFormatter, MaxNLocator

# LOESS (LOWESS) for smoothing
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    _HAS_LOWESS = True
except Exception:
    _HAS_LOWESS = False
    
print(f"[diag] LOWESS available? {_HAS_LOWESS}")

# =========== PUBLISHING THEME ===========
from matplotlib.ticker import MaxNLocator, PercentFormatter, FuncFormatter
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.constrained_layout.use": True,
    "font.size": 7,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "grid.color": "#d0d0d0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.grid": True,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Okabe–Ito palette (colour-blind safe)
PALETTE = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "brown":  "#8B4513",
    "grey":   "#666666",
}


def _figsize_wide():      # ~140mm x 80mm
    return (5.5, 3.1)

def _figsize_single():    # ~90mm x 70mm
    return (3.5, 2.8)

def _savefig(base: str):
    for ext in (".pdf", ".svg", ".png"):
        plt.savefig(f"fig_{base}{ext}", bbox_inches="tight")

def _thousands(x, pos):   # 12,345 style
    try:
        return f"{int(x):,}"
    except Exception:
        return ""
THOUSANDS = FuncFormatter(_thousands)

def _circular_moving_average(hours: np.ndarray, y: np.ndarray, k: int = 3, n_out: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple circular moving average on a 24-hour domain.
    k = half-window in hours (3 -> 7-hour window).
    Returns a dense curve (x_fit, y_fit) for smooth plotting.
    """
    # Wrap data 0..23 by padding 24 hours on each side
    x_pad = np.r_[hours - 24, hours, hours + 24]
    y_pad = np.r_[y, y, y]

    # Uniform grid to evaluate on (dense, for a smooth line)
    x_fit = np.linspace(0, 23, n_out)

    # For each x_fit, take indices within +/- k hours and average
    y_fit = np.empty_like(x_fit, dtype=float)
    for i, xf in enumerate(x_fit):
        mask = (x_pad >= xf - k) & (x_pad <= xf + k)
        y_fit[i] = float(np.mean(y_pad[mask])) if mask.any() else np.nan

    # Guarantee non-negative percentages
    y_fit = np.maximum(0.0, y_fit)
    return x_fit, y_fit


def _circular_loess(hours: np.ndarray, y: np.ndarray, frac: float = 0.3,
                n_out: int = 240, it: int = 1, weights: Optional[np.ndarray] = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """
    Cyclic LOWESS over 24h using statsmodels.lowess with optional weights.
    Pads data by ±24h so smoothing respects wrap at 0/24.
    """
    if not _HAS_LOWESS:
        raise RuntimeError("LOWESS unavailable")

    x = hours.astype(float)
    y = y.astype(float)

    # pad domain so smoothing respects wrap at 0/24
    x_pad = np.r_[x - 24.0, x, x + 24.0]
    y_pad = np.r_[y, y, y]
    if weights is not None:
        w_pad = np.r_[weights, weights, weights].astype(float)
    else:
        w_pad = None

        # Call statsmodels.lowess; only pass weights if the build supports it
    kws = dict(frac=frac, it=it, return_sorted=True)
    if w_pad is not None:
        kws["weights"] = w_pad  # pass only when present/supported
    try:
        lo = lowess(y_pad, x_pad, **kws)
    except TypeError:
        # Older statsmodels versions don't accept 'weights' → retry without
        lo = lowess(y_pad, x_pad, frac=frac, it=it, return_sorted=True)
    x_s, y_s = lo[:, 0], lo[:, 1]

    # interpolate onto dense grid and slice back to 0..23
    x_fit_dense = np.linspace(-24.0, 48.0 - 1e-3, 3 * n_out)
    y_fit_dense = np.interp(x_fit_dense, x_s, y_s)

    mask = (x_fit_dense >= 0.0) & (x_fit_dense < 24.0)
    x_fit = x_fit_dense[mask]
    y_fit = np.maximum(0.0, y_fit_dense[mask])
    return x_fit, y_fit

# Optional logistic regression
try:
    import statsmodels.api as sm
    _HAS_SM = True
except Exception:
    _HAS_SM = False

# ---------------- Config ----------------
START = dt.date(2024, 1, 1)
END   = dt.date(2024, 12, 31)   # Hard freeze to 2024
TODAY = END

# Carbon Intensity CSVs you may have locally (optional)
CSV_PATTERN = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-*.csv"
CI_CACHE_CSV = "CI_2024_halfhourly.csv"

# Frequency cache dir (per-month parquet files)
FREQ_CACHE_DIR = pathlib.Path("frequency_cache")
FREQ_CACHE_DIR.mkdir(exist_ok=True)

# === SSEN Smart Meter LV Feeder API (GraphQL) ===============================
SSEN_GRAPHQL_CANDIDATES = [
    "https://gql.internal.datopian.com/graphql",
    "https://ssen.opendata.arcus.energy/graphql",
]
FEEDER_ID = None
SSEN_START = START
SSEN_END   = END

# DR scenario defaults
DR_SHIFTABLE_SHARE = 0.3
DR_MIN_RUN_BLOCK_HH = 2
DR_MAX_SHIFT_HOURS = 12

# ---- Pretty printing helpers ----
def _hdr(text: str):
    print("\n" + "=" * 8 + " " + text + " " + "=" * 8)

def _pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:.2f}%"

def _fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "—"

def _print_table(df: pd.DataFrame, max_rows: int = 30):
    if df is None or df.empty:
        print("(no rows)")
        return
    if len(df) > max_rows:
        print(df.head(max_rows).to_string())
        print(f"... ({len(df)-max_rows} more rows)")
    else:
        print(df.to_string())

def save_df_parquet_or_csv(df: pd.DataFrame, path) -> pathlib.Path:
    path = pathlib.Path(path)
    try:
        for c in df.columns:
            if isinstance(df[c].dtype, pd.PeriodDtype):
                df[c] = df[c].astype(str)
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        for c in obj_cols:
            if df[c].apply(lambda x: isinstance(x, pd.Period) if pd.notna(x) else False).any():
                df[c] = df[c].astype(str)
        df.to_parquet(path)
        print(f"Saved Parquet -> {path}")
        return path
    except Exception as e:
        alt = path.with_suffix(".csv")
        df.to_csv(alt)
        print(f"Parquet failed ({type(e).__name__}: {e}). Saved CSV -> {alt}")
        return alt

def load_df_parquet_or_csv(path) -> pd.DataFrame:
    path = pathlib.Path(path)
    try:
        return pd.read_parquet(path)
    except Exception:
        alt = path.with_suffix(".csv")
        if alt.exists():
            return pd.read_csv(alt, parse_dates=[0], index_col=0)
        raise

# ---- Ofgem price cap helper ----
CAP_PERIODS = [
    (dt.date(2024, 1, 1),  dt.date(2024, 3, 31), 29.00),
    (dt.date(2024, 4, 1),  dt.date(2024, 6, 30), 22.00),
    (dt.date(2024, 7, 1),  dt.date(2024, 9, 30), 22.36),
    (dt.date(2024,10, 1),  dt.date(2024,12,31), 24.50),
    # entries beyond 2024 are harmless; time-slicing keeps range to END
]

def cap_timeseries(start: dt.date, end: dt.date, freq: str) -> pd.DataFrame:
    frames = []
    for a, b, p_per_kwh in CAP_PERIODS:
        s = max(a, start); e = min(b, end)
        if s > e: continue
        ts_start = pd.Timestamp(s, tz='UTC')
        ts_end = pd.Timestamp(e, tz='UTC') + pd.Timedelta(days=1)
        rng = pd.date_range(ts_start, ts_end, freq=freq, inclusive='left')
        if rng.size == 0: continue
        frames.append(pd.DataFrame({"ts_utc": rng, "cap_gbp_per_mwh": p_per_kwh * 10.0}))
    if not frames:
        return pd.DataFrame(columns=["ts_utc","cap_gbp_per_mwh"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("ts_utc").sort_values("ts_utc")

# ---------------- Carbon Intensity: load OR fetch ----------------
def load_ci_from_csv(csv_pattern: str) -> Optional[pd.DataFrame]:
    files = sorted(glob.glob(csv_pattern))
    if not files: return None
    li = []
    for f in files:
        df = pd.read_csv(f)
        if "Datetime (UTC)" not in df.columns or "Actual Carbon Intensity (gCO2/kWh)" not in df.columns:
            raise KeyError(f"{f} missing required columns.")
        li.append(df[["Datetime (UTC)", "Actual Carbon Intensity (gCO2/kWh)"]])
    ci = pd.concat(li, ignore_index=True)
    ci["ts_utc"] = pd.to_datetime(ci["Datetime (UTC)"], utc=True)
    ci = (ci.dropna(subset=["ts_utc", "Actual Carbon Intensity (gCO2/kWh)"])
            .rename(columns={"Actual Carbon Intensity (gCO2/kWh)":"ci_g_per_kwh"}))
    ci["ts_utc"] = ci["ts_utc"].dt.floor("30min")
    return ci.groupby("ts_utc", as_index=False)["ci_g_per_kwh"].mean().sort_values("ts_utc")

def _chunks(start: dt.datetime, end: dt.datetime, days: int = 30) -> Iterable[Tuple[dt.datetime, dt.datetime]]:
    cursor = start
    delta = dt.timedelta(days=days)
    while cursor < end:
        nxt = min(cursor + delta, end)
        yield cursor, nxt
        cursor = nxt

def fetch_ci_from_api(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    sess = requests.Session()
    rows: List[Dict[str, Any]] = []
    start_dt = dt.datetime.combine(start_date, dt.time(0,0))
    end_dt   = dt.datetime.combine(end_date + dt.timedelta(days=1), dt.time(0,0))
    for a, b in _chunks(start_dt, end_dt, days=30):
        frm = a.strftime("%Y-%m-%dT%H:%MZ")
        to  = (b - dt.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%MZ")
        url = f"https://api.carbonintensity.org.uk/intensity/{frm}/{to}"
        r = sess.get(url, headers={"Accept":"application/json","User-Agent":"ci-merge-script"})
        if r.status_code != 200:
            raise requests.HTTPError(f"CI API {r.status_code} for {url}\nBody: {r.text[:400]}")
        items = r.json().get("data") or []
        for it in items:
            ts_from = pd.to_datetime(it.get("from"), utc=True, errors="coerce")
            inten = it.get("intensity") or {}
            actual = inten.get("actual"); forecast = inten.get("forecast")
            val = actual if actual is not None else forecast
            if ts_from is not None and val is not None:
                rows.append({"ts_utc": ts_from.floor("30min"), "ci_g_per_kwh": float(val)})
    if not rows:
        raise RuntimeError("Carbon Intensity API returned no rows for the requested range.")
    ci = pd.DataFrame(rows).groupby("ts_utc", as_index=False)["ci_g_per_kwh"].mean().sort_values("ts_utc")
    ci.to_csv(CI_CACHE_CSV, index=False)
    return ci

def load_or_fetch_ci(start: dt.date, end: dt.date) -> pd.DataFrame:
    ci = load_ci_from_csv(CSV_PATTERN)
    if ci is not None:
        ci = ci[(ci["ts_utc"].dt.date >= start) & (ci["ts_utc"].dt.date <= end)]
        if not ci.empty: return ci
    return fetch_ci_from_api(start, end)

# ---------------- MID price (Elexon Insights) ------------------
BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
MID_DATASET = f"{BASE_URL}/datasets/MID"

def _session_with_retries(total: int = 3, backoff: float = 0.5) -> requests.Session:
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter
    s = requests.Session()
    retry = Retry(total=total, backoff_factor=backoff,
                  status_forcelist=[429,500,502,503,504],
                  allowed_methods=frozenset(["GET"]),
                  raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def fetch_mid_window(*, t_from_iso: str, t_to_iso: str,
                     data_provider: Optional[str] = None,
                     limit: int = 2000, timeout: int = 30) -> pd.DataFrame:
    sess = _session_with_retries()
    params: Dict[str, Any] = {"format":"json","limit":limit,"from":t_from_iso,"to":t_to_iso}
    if data_provider: params["dataProvider"] = data_provider
    headers = {"Accept":"application/json","User-Agent":"python-requests MID fetch"}
    url = MID_DATASET
    rows: List[Dict[str, Any]] = []
    while True:
        r = sess.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code != 200:
            raise requests.HTTPError(f"{r.status_code} for {r.url}\nBody: {r.text[:400]}")
        if "application/json" not in (r.headers.get("Content-Type","").lower()):
            raise requests.HTTPError(f"Non-JSON from {r.url}\nBody: {r.text[:400]}")
        data = r.json()
        items = data.get("data") or data.get("items") or []
        rows.extend(items)
        next_url = (data.get("links") or {}).get("next")
        if not next_url: break
        url, params = next_url, {}
    if not rows: return pd.DataFrame()
    df = pd.json_normalize(rows)
    for fld in ["price","settlementPeriod","settlementDate"]:
        col = f"attributes.{fld}"
        if col in df.columns and fld not in df.columns:
            df[fld] = df[col]
    if {"settlementDate","settlementPeriod"}.issubset(df.columns):
        df["settlementDate"] = pd.to_datetime(df["settlementDate"], errors="coerce")
        df["settlementPeriod"] = pd.to_numeric(df["settlementPeriod"], errors="coerce")
        df = df.dropna(subset=["settlementDate","settlementPeriod"])
        df["ts_utc"] = (df["settlementDate"].dt.floor("D")
                        + pd.to_timedelta((df["settlementPeriod"].astype(int)-1)*30, unit="m"))
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    else:
        if "from" not in df.columns:
            raise RuntimeError("MID payload missing settlement fields and 'from'.")
        df["ts_utc"] = pd.to_datetime(df["from"], utc=True, errors="coerce").dt.floor("30min")
    price_col = "price" if "price" in df.columns else "attributes.price"
    if price_col not in df.columns:
        price_col = next((c for c in df.columns if "price" in c.lower()), None)
    df["mid_price_gbp_per_mwh"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["ts_utc","mid_price_gbp_per_mwh"])
    return (df[["ts_utc","mid_price_gbp_per_mwh"]]
            .groupby("ts_utc", as_index=False).mean()
            .sort_values("ts_utc"))

def _iso_day(d: dt.date) -> tuple[str, str]:
    a = dt.datetime(d.year, d.month, d.day, 0, 0)
    b = a + dt.timedelta(days=1)
    return a.strftime("%Y-%m-%dT%H:%MZ"), b.strftime("%Y-%m-%dT%H:%MZ")

def fetch_mid_ytd(start: dt.date, end: dt.date) -> pd.DataFrame:
    frames = []
    cursor = start
    while cursor <= end:
        t_from, t_to = _iso_day(cursor)
        day = fetch_mid_window(t_from_iso=t_from, t_to_iso=t_to)
        if day.empty:
            for prov in ("N2EXMIDP","APXMIDP"):
                day = fetch_mid_window(t_from_iso=t_from, t_to_iso=t_to, data_provider=prov)
                if not day.empty: break
        if not day.empty:
            frames.append(day)
        cursor += dt.timedelta(days=1)
    if not frames:
        raise RuntimeError("No MID rows fetched in requested range.")
    return pd.concat(frames, ignore_index=True)

# ---------------- NESO grid frequency (minute aggregation + cache) ------
CKAN_BASE = "https://api.neso.energy/api/3/action"

def _http_session(total: int = 5, backoff: float = 0.5) -> requests.Session:
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter
    s = requests.Session()
    retry = Retry(total=total, backoff_factor=backoff,
                  status_forcelist=[429,500,502,503,504],
                  allowed_methods=frozenset(["GET"]),
                  raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def _month_name_to_int(month_name: str) -> int:
    return pd.to_datetime(month_name, format="%B").month

def _extract_year_month_from_name(name: str) -> Optional[Tuple[int,int]]:
    m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})", name, re.I)
    if m:
        return int(m.group(2)), _month_name_to_int(m.group(1).title())
    m = re.search(r"(\d{4})-(\d{2})", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

def get_frequency_resource_map(session: Optional[requests.Session] = None) -> Dict[Tuple[int,int], str]:
    sess = session or _http_session()
    package_ids_try = ["system-frequency","system-frequency-data","system-frequency-dataset"]
    mapping: Dict[Tuple[int,int], str] = {}
    for pkg_id in package_ids_try:
        try:
            r = sess.get(f"{CKAN_BASE}/package_show", params={"id": pkg_id}, timeout=30)
            if r.status_code != 200:
                continue
            for res in r.json().get("result", {}).get("resources", []):
                name = res.get("name") or ""
                rid = res.get("id") or res.get("resource_id")
                ym = _extract_year_month_from_name(name)
                if rid and ym:
                    mapping[ym] = rid
        except Exception:
            pass
    if not mapping:
        raise RuntimeError("Could not discover NESO System Frequency resources via CKAN package_show.")
    return mapping

def _iso(dt_: datetime) -> str:
    return dt_.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def fetch_frequency_minute_agg_for_window(session: requests.Session, resource_id: str,
                                          start_dt: datetime, end_dt: datetime,
                                          timeout: int = 60) -> pd.DataFrame:
    sql = f"""
    SELECT date_trunc('minute', "dtm") AS ts, AVG("f") AS frequency
    FROM "{resource_id}"
    WHERE "dtm" >= '{_iso(start_dt)}' AND "dtm" < '{_iso(end_dt)}'
    GROUP BY ts
    ORDER BY ts
    """
    r = session.get(f"{CKAN_BASE}/datastore_search_sql", params={"sql": sql}, timeout=timeout)
    r.raise_for_status()
    recs = r.json().get("result", {}).get("records", [])
    if not recs:
        return pd.DataFrame(columns=["frequency"]).set_index(pd.DatetimeIndex([], tz="UTC"))
    df = pd.DataFrame.from_records(recs)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce")
    return df.set_index("ts").sort_index()

def _month_bounds(year: int, month: int) -> Tuple[datetime, datetime]:
    start = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(year+1, 1, 1, 0, 0, 0, tzinfo=timezone.utc) if month == 12 else datetime(year, month+1, 1, 0, 0, 0, tzinfo=timezone.utc)
    return start, end

def fetch_frequency_month_minute_agg(session: requests.Session, resource_id: str, year: int, month: int,
                                     chunk_days: int = 15) -> pd.DataFrame:
    start, end = _month_bounds(year, month)
    frames: List[pd.DataFrame] = []
    cursor = start
    while cursor < end:
        nxt = min(cursor + timedelta(days=chunk_days), end)
        df = fetch_frequency_minute_agg_for_window(session, resource_id, cursor, nxt)
        if not df.empty:
            frames.append(df)
        cursor = nxt
    if not frames:
        return pd.DataFrame(columns=["frequency"]).set_index(pd.DatetimeIndex([], tz="UTC"))
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="first")]

def fetch_frequency_month_with_cache(session: requests.Session, resource_id: str, year: int, month: int,
                                     chunk_days: int = 15) -> pd.DataFrame:
    fname = FREQ_CACHE_DIR / f"freq_{year}-{month:02d}.parquet"
    try:
        return load_df_parquet_or_csv(fname)
    except Exception:
        pass
    df = fetch_frequency_month_minute_agg(session, resource_id, year, month, chunk_days=chunk_days)
    if not df.empty:
        save_df_parquet_or_csv(df, fname)
    return df

def fetch_frequency_jan2024_to_dec2024_minute(chunk_days: int = 15) -> pd.DataFrame:
    sess = _http_session()
    resource_map = get_frequency_resource_map(sess)
    months_sorted = sorted([ym for ym in resource_map.keys() if (ym >= (2024, 1) and ym <= (2024, 12))])
    if not months_sorted:
        raise RuntimeError("No System Frequency months available for 2024.")
    frames: List[pd.DataFrame] = []
    for (y, m) in months_sorted:
        rid = resource_map[(y, m)]
        print(f"[{y}-{m:02d}] NESO minute aggregation (resource={rid}) …")
        monthly = fetch_frequency_month_with_cache(sess, rid, y, m, chunk_days=chunk_days)
        if not monthly.empty:
            frames.append(monthly)
    if not frames:
        raise RuntimeError("No frequency data fetched for 2024.")
    freq_1min = pd.concat(frames).sort_index()
    freq_1min = freq_1min[~freq_1min.index.duplicated(keep="first")]
    # hard slice to 2024 just in case
    return freq_1min.loc["2024-01-01":"2024-12-31 23:59:59"]

# --------------- Visual helpers for "free" times ----------------
def plot_minutes_free_per_month(half_hour_price: pd.Series, END_DATE: dt.date):
    # Boolean mask & monthly total minutes
    free_mask = (half_hour_price <= 0)
    monthly_minutes = free_mask.resample("MS").sum() * 30
    if monthly_minutes.empty:
        _hdr("Minutes ≤ £0 — no data"); return

    # Keep only 2024 months
    mm_2024 = monthly_minutes.loc["2024-01-01":"2024-12-01"]
    if mm_2024.empty:
        _hdr("Minutes ≤ £0 — no full months to plot"); return

    # Convert to naive timestamps (month starts) and compute month ends/centres
    starts = mm_2024.index.tz_localize(None).to_period("M").to_timestamp()
    ends   = (starts + pd.offsets.MonthEnd(1))
    mids   = starts + (ends - starts)/2
    widths_days = (ends - starts).days  # use day widths

    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    # Bars centred at month midpoints
    ax.bar(mids, mm_2024.values, width=widths_days, align="center", color=PALETTE["blue"])

    ax.set_title("Minutes with Electricity Price ≤ £0 by Month — 2024")
    ax.set_xlabel("Month")
    ax.set_ylabel("Minutes ≤ £0")
    ax.yaxis.set_major_formatter(THOUSANDS)

    # Ticks centred on bars
    ax.set_xticks(mids)
    ax.set_xticklabels([d.strftime("%b\n%Y") for d in starts])

    # Clamp x-limits strictly to 2024 to avoid a Jan 2025 tick
    ax.set_xlim(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31 23:59:59"))

    # No legend (not needed)
    # ax.legend().remove()  # not created, so nothing to remove

    _savefig("minutes_le0_per_month_2024_centered")
    plt.show()

    # CSV export (unchanged, but 2024-only)
    out = pd.DataFrame({"minutes_le_0": mm_2024.astype(int)})
    out.index = out.index.to_period("M").strftime("%Y-%b")
    out.to_csv("minutes_le0_2024.csv")
    _hdr("Minutes ≤ £0 — 2024 by month")
    _print_table(out)

def plot_year_hourly_profile(half_hour_price: pd.Series, half_hour_ci: pd.Series, year: int):
    start = pd.Timestamp(f"{year}-01-01", tz=half_hour_price.index.tz)
    end   = pd.Timestamp(f"{year}-12-31 23:59", tz=half_hour_price.index.tz)
    price_y = half_hour_price.loc[start:end]
    ci_y    = half_hour_ci.loc[start:end]

    free_mask  = (price_y <= 0).astype(int)
    free_minutes = free_mask.groupby(free_mask.index.hour).mean() * 60
    avg_ci_hour = ci_y.groupby(ci_y.index.hour).mean()

    fig, ax1 = plt.subplots(figsize=(7.2, 3.9))
    hours = np.arange(24)

    ax1.bar(hours, free_minutes.values, width=0.8, alpha=0.85,
            color=PALETTE["blue"], label="Avg minutes ≤ £0\nper hour")
    ax1.set_xlabel("Hour of day (UTC)")
    ax1.set_ylabel("Minutes ≤ £0 per hour")
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

    ax2 = ax1.twinx()
    ax2.plot(hours, avg_ci_hour.values, marker="o", linewidth=1.3,
             color=PALETTE["green"], label="Avg carbon intensity")
    ax2.set_ylabel("Carbon intensity (gCO₂/kWh)")

    ax1.set_xticks(range(0,24,3))
    ax1.set_xlim(-0.5, 23.5)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, frameon=False, loc="upper left", bbox_to_anchor=(1.10, 1))

    ax1.set_title(f"{year} Hour-of-Day Profile: Free Minutes & Carbon Intensity")
    _savefig(f"{year}_hourly_profile_free_and_ci")
    plt.show()

def _plot_single_month_heatmap(half_hour_price: pd.Series, year: int, month: int):
    tz = getattr(half_hour_price.index, "tz", None)
    start = pd.Timestamp(year=year, month=month, day=1, tz=tz)
    end_excl = (start + pd.offsets.MonthEnd(1)) + pd.Timedelta(days=1)

    mask = (half_hour_price.index >= start) & (half_hour_price.index < end_excl)
    m = (half_hour_price <= 0).loc[mask].astype(int)
    if m.empty:
        print(f"No data for {start:%Y-%m}.")
        return

    hour_frac = m.resample("1h").mean()
    df = pd.DataFrame({
        "day":  hour_frac.index.day,
        "hour": hour_frac.index.hour,
        "free_frac": hour_frac.values
    })
    pivot = df.pivot_table(index="day", columns="hour", values="free_frac", aggfunc="mean")

    fig, ax = plt.subplots(figsize=_figsize_wide())
    im = ax.imshow(pivot, aspect="auto", origin="upper", vmin=0, vmax=1, cmap="viridis")
    ax.set_title(f"Fraction of Each Hour with Price ≤ £0 — {start:%B %Y}")
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Day of month")

    y_positions = np.arange(len(pivot.index))
    ax.set_yticks(y_positions)
    day_vals = pivot.index.to_numpy()
    y_labels = [str(int(d)) if ((i % 3) == 0) else "" for i, d in enumerate(day_vals)]
    ax.set_yticklabels(y_labels)

    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 3)])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Fraction of hour free")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    _savefig(f"heatmap_month_{year}-{month:02d}")
    plt.show()

def plot_full_month_heatmaps(half_hour_price: pd.Series, start_date: dt.date, end_date: dt.date):
    tz = getattr(half_hour_price.index, "tz", None)
    first = pd.Timestamp(start_date.replace(day=1), tz=tz)
    last_full = pd.Timestamp(end_date.replace(day=1), tz=tz)

    if last_full < first:
        print("No fully completed months in the requested range.")
        return

    cur = first
    while cur <= last_full:
        _plot_single_month_heatmap(half_hour_price, cur.year, cur.month)
        cur = cur + pd.offsets.MonthBegin(1)

def plot_heatmap_last12m_monthly(half_hour_price: pd.Series, end_date: dt.date):
    tz = getattr(half_hour_price.index, "tz", None)
    end = pd.Timestamp(end_date, tz=tz) + pd.Timedelta(days=1)
    start = pd.Timestamp("2024-01-01", tz=tz)

    m = (half_hour_price.loc[start:end] <= 0).astype(int)
    hour_frac = m.resample("1h").mean()

    df = pd.DataFrame({
        "month": hour_frac.index.tz_localize(None).to_period("M"),
        "hour":  hour_frac.index.hour,
        "free_frac": hour_frac.values
    })
    pivot = (df.pivot_table(index="month", columns="hour",
                            values="free_frac", aggfunc="mean")
               .sort_index())

    data_vals = pivot.values.astype(float)
    if np.all(np.isnan(data_vals)):
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanmin(data_vals))
        vmax = float(np.nanmax(data_vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=_figsize_wide())
    im = ax.imshow(pivot, aspect="auto", origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)

    ax.set_title(f"Price ≤ £0: Hour-of-Day Fraction — last 12 months to {end_date:%Y-%m-%d}")
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Month")

    xticks = list(range(0, 24, 3))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{h:02d}" for h in xticks])

    ytick_pos = np.arange(len(pivot.index))
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels([p.strftime("%b %Y") for p in pivot.index])

    cbar = plt.colorbar(im, ax=ax)
    ticks = [vmin] if vmin == vmax else np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t*100:.0f}%" for t in ticks])
    cbar.set_label("Fraction of hour free")

    _savefig("heatmap_last12m_month_hour_fraction_ending_2024")
    plt.show()

    pivot_out = pivot.copy()
    pivot_out.index = pivot_out.index.strftime("%Y-%b")
    pivot_out.to_csv("heatmap_last12m_month_hour_fraction_ending_2024.csv")
    _hdr("Monthly × Hour fraction free (ending 2024, 0–1)")
    _print_table(pivot_out, max_rows=12)

def plot_heatmap_year_monthly(half_hour_price: pd.Series, year: int):
    tz = getattr(half_hour_price.index, "tz", None)
    start = pd.Timestamp(f"{year}-01-01", tz=tz)
    end_excl = pd.Timestamp(f"{year+1}-01-01", tz=tz)

    mask = (half_hour_price.index >= start) & (half_hour_price.index < end_excl)
    m = (half_hour_price.loc[mask] <= 0).astype(int)
    hour_frac = m.resample("1h").mean()

    df = pd.DataFrame({
        "month": hour_frac.index.tz_localize(None).to_period("M"),
        "hour":  hour_frac.index.hour,
        "free_frac": hour_frac.values
    })
    pivot = (df.pivot_table(index="month", columns="hour",
                            values="free_frac", aggfunc="mean").sort_index())

    data_vals = pivot.values.astype(float)
    if np.all(np.isnan(data_vals)):
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanmin(data_vals)); vmax = float(np.nanmax(data_vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=_figsize_wide())
    im = ax.imshow(pivot, aspect="auto", origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)

    ax.set_title(f"Price ≤ £0: Hour-of-day Fraction — {year}")
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Month")
    ax.set_xticks(list(range(0, 24, 3)))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 3)])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([p.strftime("%b %Y") for p in pivot.index])

    cbar = plt.colorbar(im, ax=ax)
    ticks = [vmin] if vmin == vmax else np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t*100:.0f}%" for t in ticks])
    cbar.set_label("Fraction of hour free")

    _savefig(f"heatmap_{year}_month_hour_fraction")
    plt.show()

def plot_month_broken_bars(half_hour_price: pd.Series, year: int, month: int):
    tz = getattr(half_hour_price.index, "tz", None)
    start = pd.Timestamp(year=year, month=month, day=1, tz=tz)
    end_excl = (start + pd.offsets.MonthEnd(1)) + pd.Timedelta(days=1)
    mask = (half_hour_price.index >= start) & (half_hour_price.index < end_excl)
    m = (half_hour_price <= 0).loc[mask].astype(int)

    if m.empty:
        print(f"No data for {start:%Y-%m}."); 
        return

    idx = m.index
    vals = m.values
    change = np.diff(vals.astype(int), prepend=0, append=0)
    starts = np.where(change == 1)[0]
    ends   = np.where(change == -1)[0]

    if len(starts) == 0:
        print(f"No free intervals (price ≤ £0) detected in {start:%B %Y}.")
        return

    day_to_spans: Dict[dt.date, List[Tuple[float, float]]] = {}
    for s, e in zip(starts, ends):
        span_idx = idx[s:e]
        d = span_idx[0].date()
        start_of_day = pd.Timestamp(d, tz=tz)
        x0 = (span_idx[0] - start_of_day).total_seconds() / 3600.0
        width = (span_idx[-1] - span_idx[0] + pd.Timedelta(minutes=30)).total_seconds() / 3600.0
        day_to_spans.setdefault(d, []).append((x0, width))

    days_sorted = sorted(day_to_spans.keys())
    plt.figure(figsize=(12, 0.3*len(days_sorted) + 2), dpi=300)
    for i, d in enumerate(days_sorted):
        plt.broken_barh(day_to_spans[d], (i-0.4, 0.8))
    plt.yticks(range(len(days_sorted)), [d.day for d in days_sorted])
    plt.xlim(0, 24)
    plt.xlabel("Hour of day (UTC)")
    plt.ylabel("Day of month")
    plt.title(f"Exact Free Intervals — {start:%B %Y}")
    plt.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.7)
    plt.tight_layout()
    plt.show()

# === Seasonal / diurnal features and stats =================================
def _season_from_month(m: int) -> str:
    return ("DJF" if m in (12, 1, 2) else
            "MAM" if m in (3, 4, 5) else
            "JJA" if m in (6, 7, 8) else
            "SON")

def build_feature_frame(half_hour_price: pd.Series) -> pd.DataFrame:
    s = half_hour_price.copy()
    df = pd.DataFrame({"price": s})
    df.index.name = "ts_utc"
    df["free"] = (df["price"] <= 0).astype(int)
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["weekend"] = (df["weekday"] >= 5).astype(int)
    df["month"] = df.index.month
    df["season"] = [_season_from_month(m) for m in df["month"]]
    return df

def seasonal_diurnal_stats_and_plots(half_hour_price: pd.Series):
    df = build_feature_frame(half_hour_price)

    overall_pct = df["free"].mean() * 100.0
    last12_end = df.index.max()
    last12_start = pd.Timestamp("2024-01-01", tz=last12_end.tz)
    last12 = df.loc[last12_start:last12_end]
    last12_pct = last12["free"].mean() * 100.0

    by_season = df.groupby("season")["free"].mean().reindex(["DJF","MAM","JJA","SON"]) * 100.0
    by_weekend = df.groupby("weekend")["free"].mean().rename({0:"Weekday",1:"Weekend"}) * 100.0
    by_hour = df.groupby("hour")["free"].mean() * 100.0

    neg_depth = (-df.loc[df["price"] < 0, "price"]).dropna()
    depth_stats = {
        "count_neg_halfhours": int(neg_depth.size),
        "median_depth_£/MWh": float(np.nanmedian(neg_depth)) if neg_depth.size else np.nan,
        "p10_depth": float(np.nanpercentile(neg_depth, 10)) if neg_depth.size else np.nan,
        "p1_depth":  float(np.nanpercentile(neg_depth, 1)) if neg_depth.size else np.nan,
    }

    _hdr("Headline shares of half-hours with price ≤ £0")
    print(f"Calendar 2024: {_pct(overall_pct)}")
    print(f"Last 12 months to 2024-12-31: {_pct(last12_pct)}")

    _hdr("Seasonality — % of half-hours ≤ £0 (2024)")
    season_tbl = pd.DataFrame({"%≤£0": by_season.round(2)})
    season_tbl.to_csv("free_share_by_season_2024.csv")
    _print_table(season_tbl)

    _hdr("Weekend vs Weekday — % of half-hours ≤ £0 (2024)")
    wk_tbl = pd.DataFrame({"%≤£0": by_weekend.round(2)})
    wk_tbl.to_csv("free_share_weekend_vs_weekday_2024.csv")
    _print_table(wk_tbl)

    _hdr("Hour of day — % of half-hours ≤ £0 (2024)")
    hour_tbl = pd.DataFrame({"hour": by_hour.index, "%≤£0": by_hour.values.round(2)}).set_index("hour")
    hour_tbl.to_csv("free_share_by_hour_2024.csv")
    _print_table(hour_tbl, max_rows=30)

    _hdr("Negative price depth (|£/MWh|) among free half-hours — 2024")
    depth_tbl = pd.DataFrame([depth_stats])
    depth_tbl.to_csv("negative_price_depth_stats_2024.csv", index=False)
    _print_table(depth_tbl)

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(4.2, 5.6), constrained_layout=True)
    axA.bar(by_season.index, by_season.values, color=PALETTE["orange"])
    axA.set_ylabel("% of half-hours ≤ £0")
    axA.set_xlabel("Season")
    axA.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
    axA.set_title("Probability of ≤ £0 by season (2024)")

    axB.bar(by_weekend.index, by_weekend.values, color=PALETTE["purple"])
    axB.set_ylabel("% of half-hours ≤ £0")
    axB.set_title("Probability of ≤ £0: weekend vs weekday (2024)")

    _savefig("share_le0_season_and_daytype_stacked_2024"); plt.show()

    h = by_hour.index.to_numpy(dtype=float)
    y = by_hour.values.astype(float)
    
    # weights = number of half-hours contributing to each hourly mean
    w = (df.groupby("hour")["free"]
           .size()
           .reindex(range(24), fill_value=0)
           .to_numpy(dtype=float))
    w = w[h.astype(int)]  # align to h
    
    fig, ax = plt.subplots(figsize=_figsize_single())
    ax.plot(
        h, y,
        marker="x",
        markersize=3.5,
        markeredgewidth=1.2,
        linestyle="none",
        label="Observed",
        color=PALETTE["blue"]
    )
    
    label_used = ""
    try:
        if not _HAS_LOWESS:
            raise RuntimeError("LOWESS not available")
        # Try LOESS with weights → if TypeError (older statsmodels), retry without.
        try:
            x_fit, y_fit = _circular_loess(h, y, frac=0.12, n_out=240, it=1, weights=w)
        except TypeError as e:
            print(f"[loess] 'weights' unsupported in this statsmodels build: {e} — retrying without weights.")
            x_fit, y_fit = _circular_loess(h, y, frac=0.12, n_out=240, it=1, weights=None)
        label_used = "LOESS (cyclic)"
    except Exception as e:
        print(f"[loess] Falling back to circular moving average: {type(e).__name__}: {e}")
        x_fit, y_fit = _circular_moving_average(h, y, k=3, n_out=240)
        label_used = "Circular MA"
    
    ax.plot(x_fit, y_fit, linewidth=1.2, label=label_used, color=PALETTE["red"])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(PercentFormatter(100))  # show 0–100-scale values as %
    ax.set_ylim(0, max(6.0, float(np.nanmax(y)) * 1.15))  # a little headroom
    ax.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlim(0, 23)
    ax.set_ylim(bottom=0)  # percentages cannot be negative
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("% of half-hours ≤ £0")
    ax.set_title("Probability of ≤ £0 by hour of day (2024)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    
    _savefig("share_le0_by_hour_2024"); plt.show()

    if _HAS_SM:
        X = df[["month","hour","weekend"]].copy()
        X["weekend"] = X["weekend"].astype(int)
        X = pd.get_dummies(X, columns=["month","hour"], drop_first=True)
        X = sm.add_constant(X, has_constant="add")
        y = df["free"].astype(int)
        try:
            model = sm.Logit(y, X, missing="drop")
            res = model.fit(disp=False, maxiter=200)
            with open("logit_freepower_summary_2024.txt", "w") as f:
                f.write(res.summary().as_text())
            _hdr("Logistic regression saved")
            print("Wrote: logit_freepower_summary_2024.txt")
        except Exception as e:
            print(f"[logit] Skipped (fit error: {type(e).__name__}: {e})")
    else:
        print("[logit] statsmodels not found; skipping logistic regression.")

# ========================== MAIN GLUE ==========================
if __name__ == "__main__":
    print("Script started — building 2024 datasets…")

    # 1) Carbon Intensity (load or fetch) — 2024
    ci = load_or_fetch_ci(START, END)

    # 2) MID prices — 2024
    mid = fetch_mid_ytd(START, END)

    # 3) Merge on exact 30-min timestamps — 2024
    merged = pd.merge(ci, mid, on="ts_utc", how="inner").sort_values("ts_utc")
    merged = merged[(merged["ts_utc"].dt.date >= START) & (merged["ts_utc"].dt.date <= END)]
    merged.to_csv("UK_halfhour_price_and_ci_2024.csv", index=False)

    half_hour_price = (
        merged.set_index("ts_utc")["mid_price_gbp_per_mwh"]
        .sort_index()
    )
    half_hour_price = half_hour_price[~half_hour_price.index.duplicated(keep="first")]

    half_hour_ci = (
        merged.set_index("ts_utc")["ci_g_per_kwh"]
        .sort_index()
    )
    half_hour_ci = half_hour_ci[~half_hour_ci.index.duplicated(keep="first")]

    daily_price = half_hour_price.resample("1D").mean()
    daily_ci    = half_hour_ci.resample("1D").mean()

    _hdr("Merge summary (2024)")
    print(f"Merged rows: {len(merged):,}")
    print(f"Range: {merged['ts_utc'].min()} → {merged['ts_utc'].max()}")
    print("Saved merged CSV -> UK_halfhour_price_and_ci_2024.csv")

    # 4) Ofgem price cap overlays — 2024
    cap_hh = cap_timeseries(START, END, freq="30min")
    cap_d  = cap_timeseries(START, END, freq="1D")

    # 5) Plots/tables — 2024
    plot_minutes_free_per_month(half_hour_price, END)
    plot_year_hourly_profile(half_hour_price, half_hour_ci, 2024)
    plot_full_month_heatmaps(half_hour_price, START, END)
    plot_heatmap_last12m_monthly(half_hour_price, END)
    plot_heatmap_year_monthly(half_hour_price, 2024)
    seasonal_diurnal_stats_and_plots(half_hour_price)

    # 6) Frequency (1-minute) + coupling visuals — 2024 only
    print("\nFetching NESO grid frequency 1-minute series (Jan–Dec 2024)…")
    try:
        freq_1min = fetch_frequency_jan2024_to_dec2024_minute(chunk_days=15)
        price_1min = half_hour_price.resample("1min").ffill().rename("price_gbp_per_mwh")
        ci_1min    = half_hour_ci.resample("1min").ffill().rename("ci_g_per_kwh")
        freq_1min  = freq_1min.rename(columns={"frequency": "grid_frequency_hz"})
        combined_1min = pd.concat([price_1min, ci_1min, freq_1min], axis=1).sort_index()
        combined_1min = combined_1min.loc["2024-01-01":"2024-12-31 23:59:59"]
        save_df_parquet_or_csv(combined_1min, "UK_price_CI_frequency_1min_2024.parquet")

        # Hourly medians + frequency deviation
        df = combined_1min.copy()
        df["freq_dev_mhz"] = (df["grid_frequency_hz"] - 50.0) * 1000.0
        price_h = df["price_gbp_per_mwh"].resample("1h").median()
        freq_h  = df["freq_dev_mhz"].resample("1h").median()

        # Aggregate to daily medians for visibility and smooth with a 7-day rolling median
        price_d = df["price_gbp_per_mwh"].resample("1D").median()
        freq_d  = df["freq_dev_mhz"].resample("1D").median()
        price_smooth = price_d.rolling(7, center=True, min_periods=3).median()
        freq_smooth  = freq_d.rolling(7, center=True, min_periods=3).median()
        
        fig, ax1 = plt.subplots(figsize=_figsize_wide())
        ax2 = ax1.twinx()
        
        ax1.plot(price_h.index, price_h.values, linewidth=0.3, alpha=0.25, color=PALETTE["blue"])
        ax2.plot(freq_h.index,  freq_h.values,  linewidth=0.3, alpha=0.25, color=PALETTE["red"])
        
        # Main trend lines (clear and readable)
        l1, = ax1.plot(
        price_smooth.index, price_smooth.values,
        linewidth=1.6, color=PALETTE["blue"],
        label="Price\n(7-day median \n of daily medians)"
        )
        
        l2, = ax2.plot(
            freq_smooth.index, freq_smooth.values,
            linewidth=1.6, color=PALETTE["orange"],
            label="Frequency deviation\n(7-day median \n of daily medians)"
        )
        
        # Axes labels & sensible ranges (tweak if your data extend beyond these)
        ax1.set_ylabel("Price (£/MWh)")
        ax2.set_ylabel("Frequency deviation (mHz)")
        ax1.set_ylim(-50, 300)
        ax2.set_ylim(-120, 160)
        ax1.set_xlabel("Time (UTC)")
        first_m = price_h.index.min().tz_localize(None).to_period("M").to_timestamp()
        last_m  = price_h.index.max().tz_localize(None).to_period("M").to_timestamp()
        ax1.set_xlim(first_m, last_m)
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%b"))
        ax1.xaxis.set_minor_locator(mdates.MonthLocator())
        plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", bbox_to_anchor=(1.18, 1))
        ax1.set_title("Price vs Grid Frequency (hourly medians, 2024)")
        _savefig("price_vs_frequency_hourly_medians_2024"); plt.show()

        # Hexbin density (1-min)
        sel = df[["price_gbp_per_mwh","freq_dev_mhz"]].dropna()
        if not sel.empty:
            fig, ax = plt.subplots(figsize=_figsize_single())
            hb = ax.hexbin(sel["freq_dev_mhz"], sel["price_gbp_per_mwh"], gridsize=80, mincnt=5)
            ax.set_xlabel("Frequency deviation (mHz)")
            ax.set_ylabel("Price (£/MWh)")
            ax.set_title("Price vs Frequency deviation (1-min density) — wide, 2024")
            cb = plt.colorbar(hb, ax=ax); cb.set_label("Minutes")
            ax.set_xlim(-300, 300); ax.set_ylim(-100, 100)
            _savefig("hexbin_price_vs_frequency_wide_2024"); plt.show()

            fig, ax = plt.subplots(figsize=_figsize_single())
            hb = ax.hexbin(sel["freq_dev_mhz"], sel["price_gbp_per_mwh"], gridsize=80, mincnt=5)
            ax.set_xlabel("Frequency deviation (mHz)")
            ax.set_ylabel("Price (£/MWh)")
            ax.set_title("Price vs Frequency deviation (1-min density) — focused, 2024")
            cb = plt.colorbar(hb, ax=ax); cb.set_label("Minutes")
            ax.set_xlim(-250, 250); ax.set_ylim(-40, 110)
            _savefig("hexbin_price_vs_frequency_focused_2024"); plt.show()
           
            
        # === Price by frequency deviation decile (box & violin) =====================
        sel = df[["price_gbp_per_mwh", "freq_dev_mhz"]].dropna()
        if not sel.empty:
            # (Optional) cap extreme prices so a few spikes don't squash the boxes
            sel = sel[sel["price_gbp_per_mwh"].between(-100, 300)]
        
            # Deciles of frequency deviation (mHz)
            sel["freq_decile"] = pd.qcut(sel["freq_dev_mhz"], 10, labels=False)
        
            # Gather price arrays per decile, in order Q1..Q10
            groups = [
                sel.loc[sel["freq_decile"] == i, "price_gbp_per_mwh"].to_numpy()
                for i in range(10)
            ]
        
            # ---- Box plot (recommended) ----
            fig, ax = plt.subplots(figsize=_figsize_single())
            ax.boxplot(
                groups,
                showfliers=False,     # hides extreme outliers for cleaner boxes
                whis=[5, 95],         # whiskers span 5–95th percentiles
            )
            ax.set_xticklabels([f"Q{i}" for i in range(1, 11)])
            ax.set_xlabel("Frequency deviation decile (mHz, Q1=lowest)")
            ax.set_ylabel("Price (£/MWh)")
            ax.set_title("Price by frequency deviation decile (1-min data, 2024)")
            _savefig("price_by_freq_decile_box_2024"); plt.show()
        
            # ---- Violin plot (optional companion) ----
            fig, ax = plt.subplots(figsize=_figsize_single())
            vp = ax.violinplot(groups, showmeans=True, showextrema=False)
            # light styling so it matches your palette
            for body in vp["bodies"]:
                body.set_facecolor(PALETTE["orange"])
                body.set_edgecolor("none")
                body.set_alpha(0.6)
            vp["cmeans"].set_linewidth(1.0)
        
            ax.set_xticks(range(1, 11))
            ax.set_xticklabels([f"Q{i}" for i in range(1, 11)])
            ax.set_xlabel("Frequency deviation decile (mHz, Q1=lowest)")
            ax.set_ylabel("Price (£/MWh)")
            ax.set_title("Price by frequency deviation decile — violin (1-min, 2024)")
            _savefig("price_by_freq_decile_violin_2024"); plt.show()


        corr_df = pd.concat([price_h.rename("price"), freq_h.rename("freq_mhz")], axis=1).dropna()
        if not corr_df.empty:
            month_key = corr_df.index.tz_localize(None).to_period("M")
            corr_monthly = corr_df.groupby(month_key).apply(lambda x: x.corr().iloc[0,1]).rename("corr")
            corr_monthly.index = corr_monthly.index.strftime("%Y-%b")
            corr_monthly.to_csv("price_vs_frequency_corr_monthly_2024.csv")
            _hdr("Monthly correlation: price vs frequency deviation (hourly medians, 2024)")
            _print_table(pd.DataFrame(corr_monthly), max_rows=24)
    except Exception as e:
        print(f"[frequency 2024] Skipped (error: {type(e).__name__}: {e})")

    # 7) Domestic usage & DR: assumed UK profile (SSEN-independent) — 2024
    print("\nUsing assumed UK domestic profile (no SSEN needed)…")
    def _season_from_month(m: int) -> str:
        return ("DJF" if m in (12, 1, 2) else
                "MAM" if m in (3, 4, 5) else
                "JJA" if m in (6, 7, 8) else
                "SON")

    ANNUAL_KWH_PER_HOUSEHOLD = float(os.getenv("ANNUAL_KWH_PER_HOUSEHOLD", "2700"))
    HOUSEHOLDS = int(os.getenv("HOUSEHOLDS", "28600000"))
    ASSUMED_DAILY_KWH_PER_METER = ANNUAL_KWH_PER_HOUSEHOLD / 365.25
    ASSUMED_METERS = HOUSEHOLDS
    ASSUMED_WEEKEND_FACTOR = float(os.getenv("ASSUMED_WEEKEND_FACTOR", "1.06"))
    ASSUMED_SEASON_FACTORS = {
        "DJF": float(os.getenv("ASSUMED_SEASON_FACTOR_DJF", "1.20")),
        "MAM": float(os.getenv("ASSUMED_SEASON_FACTOR_MAM", "1.00")),
        "JJA": float(os.getenv("ASSUMED_SEASON_FACTOR_JJA", "0.90")),
        "SON": float(os.getenv("ASSUMED_SEASON_FACTOR_SON", "1.05")),
    }

    _HOURLY_REL = np.array([
        0.80, 0.70, 0.60, 0.60, 0.70, 0.90, 1.40, 1.60, 1.20, 1.00, 0.90, 0.90,
        0.90, 0.90, 0.95, 1.10, 1.60, 2.10, 2.20, 2.00, 1.60, 1.20, 1.00, 0.90
    ], dtype=float)
    _HALFHOUR_REL = np.repeat(_HOURLY_REL, 2)
    _HALFHOUR_REL = _HALFHOUR_REL / _HALFHOUR_REL.sum()

    def _halfhour_idx_within_day(idx: pd.DatetimeIndex) -> np.ndarray:
        return (idx.hour * 2 + (idx.minute >= 30).astype(int)).astype(int)

    def build_assumed_domestic_series(
        halfhour_index: pd.DatetimeIndex,
        meters: int = ASSUMED_METERS,
        daily_kwh_per_meter: float = ASSUMED_DAILY_KWH_PER_METER,
        weekend_factor: float = ASSUMED_WEEKEND_FACTOR,
        season_factors: Dict[str, float] = ASSUMED_SEASON_FACTORS,
    ) -> pd.DataFrame:
        if halfhour_index.tz is None:
            halfhour_index = halfhour_index.tz_localize("UTC")

        df = pd.DataFrame(index=halfhour_index)
        df.index.name = "ts_utc"

        season = [_season_from_month(m) for m in df.index.month]
        s_fac = np.array([season_factors.get(s, 1.0) for s in season], dtype=float)
        w_fac = np.where(df.index.weekday >= 5, weekend_factor, 1.0).astype(float)

        pos = _halfhour_idx_within_day(df.index)
        base_rel = _HALFHOUR_REL[pos]

        per_meter_kwh = daily_kwh_per_meter * base_rel * s_fac * w_fac
        df["kwh"] = per_meter_kwh * meters
        df["meter_count"] = meters

        df["kwh"] = df["kwh"].fillna(0.0)
        df["meter_count"] = df["meter_count"].fillna(meters).astype(int)

        out = df.reset_index()
        return out[["ts_utc", "kwh", "meter_count"]]

    def build_domestic_hour_profile(ssen_df: pd.DataFrame) -> pd.DataFrame:
        if ssen_df is None or ssen_df.empty:
            _hdr("Domestic hour-of-day profile (kWh per meter)")
            print("(no rows)")
            return pd.DataFrame(columns=["kwh_per_meter_mean", "kwh_per_meter_median", "obs"])

        df = ssen_df.copy()
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts_utc"])
        df["hour"] = df["ts_utc"].dt.hour
        df["kwh_per_meter"] = df["kwh"] / df["meter_count"].replace(0, np.nan)

        grp = df.groupby("hour")["kwh_per_meter"]
        out = pd.DataFrame({
            "kwh_per_meter_mean": grp.mean(),
            "kwh_per_meter_median": grp.median(),
            "obs": grp.size()
        }).reset_index().set_index("hour").sort_index()

        out.to_csv("ssen_domestic_hour_profile_2024.csv")
        _hdr("Domestic hour-of-day profile (kWh per meter) — 2024")
        _print_table(out)
        return out

    def evaluate_dr_simple(ssen_df: pd.DataFrame,
                           price_hh: pd.Series,
                           ci_hh: pd.Series,
                           share: float = DR_SHIFTABLE_SHARE,
                           ci_clean_threshold: float = 100.0,
                           cap_moved_kwh: Optional[float] = None,
                           segment_name: Optional[str] = None) -> pd.DataFrame:
        if ssen_df.empty:
            _hdr("DR: no usage data available; skipping.")
            return pd.DataFrame()

        s = ssen_df.set_index("ts_utc").sort_index()
        s = s.reindex(price_hh.index).interpolate(limit_direction="both")
        ci_aligned = ci_hh.reindex(price_hh.index).interpolate(limit_direction="both")

        kwh = s["kwh"].fillna(0.0)
        price = price_hh.ffill()
        ci = ci_aligned.ffill()

        mwh = kwh / 1000.0
        baseline_cost_gbp = float((mwh * price).sum())
        baseline_co2_kg = float((kwh * ci / 1000.0).sum())

        mask_sink = (price <= 0) & (ci <= ci_clean_threshold)
        sink_energy = float(kwh[mask_sink].sum())
        source_energy = float(kwh[~mask_sink].sum())

        shiftable = share * source_energy
        moved = min(shiftable, sink_energy)
        if cap_moved_kwh is not None:
            moved = min(moved, float(cap_moved_kwh))

        avg_price_source = float(price[~mask_sink].mean(skipna=True))
        avg_price_sink   = float(price[mask_sink].mean(skipna=True)) if mask_sink.any() else 0.0
        avg_ci_source = float(ci[~mask_sink].mean(skipna=True))
        avg_ci_sink   = float(ci[mask_sink].mean(skipna=True)) if mask_sink.any() else float(ci.min())

        savings_cost = (moved/1000.0) * (avg_price_source - avg_price_sink)
        savings_co2  = (moved) * (avg_ci_source - avg_ci_sink) / 1000.0

        post_cost = baseline_cost_gbp - savings_cost
        post_co2  = baseline_co2_kg - savings_co2

        res = pd.DataFrame([{
            "segment_name": segment_name or "",
            "baseline_cost_gbp": baseline_cost_gbp,
            "post_dr_cost_gbp": post_cost,
            "cost_savings_gbp": savings_cost,
            "baseline_co2_kg": baseline_co2_kg,
            "post_dr_co2_kg": post_co2,
            "co2_savings_kg": savings_co2,
            "share_shiftable": share,
            "ci_clean_threshold_g_per_kwh": ci_clean_threshold,
            "source_energy_kwh": source_energy,
            "sink_energy_kwh": sink_energy,
            "moved_energy_kwh": moved
        }])
        res.to_csv("dr_summary_2024.csv", index=False)

        _hdr("DR simple model — summary (2024)")
        res["cost_savings_pct"] = 100.0 * res["cost_savings_gbp"] / res["baseline_cost_gbp"]
        res["co2_savings_pct"]  = 100.0 * res["co2_savings_kg"]  / res["baseline_co2_kg"]
        summary = res.copy()
        summary["cost_savings_pct"] = summary["cost_savings_pct"].map("{:.1f}%".format)
        summary["co2_savings_pct"]  = summary["co2_savings_pct"].map("{:.1f}%".format)
        summary["baseline_cost_gbp"] = summary["baseline_cost_gbp"].map("£{:,.0f}".format)
        summary["post_dr_cost_gbp"]  = summary["post_dr_cost_gbp"].map("£{:,.0f}".format)
        summary["cost_savings_gbp"]  = summary["cost_savings_gbp"].map("£{:,.0f}".format)
        summary["baseline_co2_kg"]   = summary["baseline_co2_kg"].map("{:,.0f} kg".format)
        summary["post_dr_co2_kg"]    = summary["post_dr_co2_kg"].map("{:,.0f} kg".format)
        summary["co2_savings_kg"]    = summary["co2_savings_kg"].map("{:,.0f} kg".format)

        print(summary.to_string(index=False))
        return res

    def run_supplier_mix_analysis(ssen_df: pd.DataFrame,
                                  price_hh: pd.Series,
                                  ci_hh: pd.Series,
                                  segments: List[dict],
                                  ci_clean_threshold: float = 100.0,
                                  behavioural_share: float = 1.0) -> pd.DataFrame:
        if ssen_df.empty:
            _hdr("Supplier mix DR — no usage data")
            return pd.DataFrame()

        total_kwh = float(ssen_df["kwh"].sum())
        rows = []

        for seg in segments:
            name = seg["name"]
            w = float(seg["customer_share"])
            cap_pct = float(seg["max_shift_pct"])
            cap_kwh = w * total_kwh * cap_pct

            res = evaluate_dr_simple(
                ssen_df, price_hh, ci_hh,
                share=behavioural_share,
                ci_clean_threshold=ci_clean_threshold,
                cap_moved_kwh=cap_kwh,
                segment_name=name
            ).iloc[0].to_dict()

            scale_cols = [
                "baseline_cost_gbp","post_dr_cost_gbp","cost_savings_gbp",
                "baseline_co2_kg","post_dr_co2_kg","co2_savings_kg",
                "source_energy_kwh","sink_energy_kwh","moved_energy_kwh"
            ]
            for c in scale_cols:
                res[c] = float(res[c]) * w

            res["customer_share"] = w
            res["max_shift_pct"] = cap_pct
            rows.append(res)

        out = pd.DataFrame(rows)
        out["cost_savings_pct"] = 100.0 * out["cost_savings_gbp"] / out["baseline_cost_gbp"]
        out["co2_savings_pct"]  = 100.0 * out["co2_savings_kg"]  / out["baseline_co2_kg"]

        totals = {
            "segment_name": "TOTAL",
            "customer_share": out["customer_share"].sum(),
            "max_shift_pct": None,
        }
        for c in ["baseline_cost_gbp","post_dr_cost_gbp","cost_savings_gbp",
                  "baseline_co2_kg","post_dr_co2_kg","co2_savings_kg",
                  "source_energy_kwh","sink_energy_kwh","moved_energy_kwh"]:
            totals[c] = out[c].sum()
        totals["cost_savings_pct"] = 100.0 * totals["cost_savings_gbp"] / max(totals["baseline_cost_gbp"], 1e-9)
        totals["co2_savings_pct"]  = 100.0 * totals["co2_savings_kg"]  / max(totals["baseline_co2_kg"], 1e-9)

        out = pd.concat([out, pd.DataFrame([totals])], ignore_index=True)
        out.to_csv("dr_segments_summary_2024.csv", index=False)

        _hdr("Supplier mix DR — segments summary (2024)")
        disp = out.copy()
        money_cols = ["baseline_cost_gbp","post_dr_cost_gbp","cost_savings_gbp"]
        disp[money_cols] = disp[money_cols].applymap(lambda x: f"£{x:,.0f}" if pd.notna(x) else "—")
        mass_cols = ["baseline_co2_kg","post_dr_co2_kg","co2_savings_kg"]
        disp[mass_cols] = disp[mass_cols].applymap(lambda x: f"{x:,.0f} kg" if pd.notna(x) else "—")
        energy_cols = ["source_energy_kwh","sink_energy_kwh","moved_energy_kwh"]
        disp[energy_cols] = disp[energy_cols].applymap(lambda x: f"{x:,.0f} kWh" if pd.notna(x) else "—")
        for c in ["customer_share","max_shift_pct","cost_savings_pct","co2_savings_pct"]:
            if c in disp.columns:
                disp[c] = disp[c].apply(lambda v: "—" if (v is None or pd.isna(v)) else f"{float(v):.1f}%"
                                        if c.endswith("_pct") else f"{100.0*float(v):.1f}%")
        _print_table(disp, max_rows=50)
        return out

    ssen_df = build_assumed_domestic_series(half_hour_price.index)
    save_df_parquet_or_csv(ssen_df, "assumed_domestic_usage_2024.parquet")
    build_domestic_hour_profile(ssen_df)
    evaluate_dr_simple(
        ssen_df,
        half_hour_price,
        half_hour_ci,
        share=DR_SHIFTABLE_SHARE,
        ci_clean_threshold=100.0
    )

    segments = [
        {"name": "No storage + no daytime EV",   "customer_share": 0.89, "max_shift_pct": 0.10},
        {"name": "Thermal storage + automation", "customer_share": 0.10, "max_shift_pct": 0.20},
        {"name": "Battery installed + EV",       "customer_share": 0.01, "max_shift_pct": 0.40},
    ]
    run_supplier_mix_analysis(
        ssen_df,
        half_hour_price,
        half_hour_ci,
        segments=segments,
        ci_clean_threshold=100.0,
        behavioural_share=1.0,
    )

    print("Done (2024 freeze).")
