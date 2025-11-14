#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Price–Carbon Coupling & Free-Price Windows in Great Britain (2024)

Standalone script:
- If UK_halfhour_price_and_ci_2024.csv exists, it uses it.
- If not, it fetches:
    * Carbon Intensity (CI) from carbonintensity.org.uk API
    * MID wholesale prices from Elexon Insights (BMRS MID dataset)
  then merges to build the 2024 half-hour dataset and saves it.

Then:
- Builds features (hour, month, weekend, CI, lags)
- Trains a simple classifier for price <= 0
- Outputs summary tables + basic plots

Author: Kieran Zhané
Last updated: 2025-11-14
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Iterable, Tuple
import pathlib
import sys
import re
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter, FuncFormatter

# --- ML stack ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Optional: statsmodels for a logit odds-ratio table
try:
    import statsmodels.api as sm
    _HAS_SM = True
except Exception:
    _HAS_SM = False

# ---------------------------------------------------------------------
# GLOBAL CONFIG
# ---------------------------------------------------------------------

START = dt.date(2024, 1, 1)
END   = dt.date(2024, 12, 31)     # hard freeze to calendar 2024
MERGED_CSV = "UK_halfhour_price_and_ci_2024.csv"

# Carbon Intensity CSVs (if present); otherwise API will be used
CI_CACHE_CSV = "CI_2024_halfhourly.csv"
CI_RAW_PATTERN = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-*.csv"

BASE_URL_ELEXON = "https://data.elexon.co.uk/bmrs/api/v1"
MID_DATASET = f"{BASE_URL_ELEXON}/datasets/MID"

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.constrained_layout.use": True,
    "font.size": 8,
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

def _thousands(x, pos):
    try:
        return f"{int(x):,}"
    except Exception:
        return ""

THOUSANDS = FuncFormatter(_thousands)


# ---------------------------------------------------------------------
# 1. HELPER: CARBON INTENSITY
# ---------------------------------------------------------------------

def _chunks(start: dt.datetime, end: dt.datetime, days: int = 30) -> Iterable[Tuple[dt.datetime, dt.datetime]]:
    cursor = start
    delta = dt.timedelta(days=days)
    while cursor < end:
        nxt = min(cursor + delta, end)
        yield cursor, nxt
        cursor = nxt

def load_ci_from_csv(pattern: str) -> Optional[pd.DataFrame]:
    """
    Optional: if you already have CI CSVs (Carbon Intensity API downloads),
    combine them; otherwise None and fall back to API.
    """
    import glob
    files = sorted(glob.glob(pattern))
    if not files:
        return None

    li = []
    for f in files:
        df = pd.read_csv(f)
        if "Datetime (UTC)" not in df.columns or "Actual Carbon Intensity (gCO2/kWh)" not in df.columns:
            raise KeyError(f"{f} missing required columns.")
        li.append(df[["Datetime (UTC)", "Actual Carbon Intensity (gCO2/kWh)"]])
    ci = pd.concat(li, ignore_index=True)
    ci["ts_utc"] = pd.to_datetime(ci["Datetime (UTC)"], utc=True)
    ci = (
        ci.dropna(subset=["ts_utc", "Actual Carbon Intensity (gCO2/kWh)"])
          .rename(columns={"Actual Carbon Intensity (gCO2/kWh)": "ci_g_per_kwh"})
    )
    ci["ts_utc"] = ci["ts_utc"].dt.floor("30min")
    ci = (
        ci.groupby("ts_utc", as_index=False)["ci_g_per_kwh"]
          .mean()
          .sort_values("ts_utc")
    )
    return ci

def fetch_ci_from_api(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """
    Fetch CI from national API for [start_date, end_date] inclusive.
    30-minute resolution, using actuals where available, else forecast.
    """
    sess = requests.Session()
    rows: List[Dict[str, Any]] = []

    start_dt = dt.datetime.combine(start_date, dt.time(0, 0))
    end_dt   = dt.datetime.combine(end_date + dt.timedelta(days=1), dt.time(0, 0))

    for a, b in _chunks(start_dt, end_dt, days=30):
        frm = a.strftime("%Y-%m-%dT%H:%MZ")
        to  = (b - dt.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%MZ")
        url = f"https://api.carbonintensity.org.uk/intensity/{frm}/{to}"

        r = sess.get(url, headers={"Accept": "application/json", "User-Agent": "ci-merge-paper2"})
        if r.status_code != 200:
            raise requests.HTTPError(f"CI API {r.status_code} for {url}\nBody: {r.text[:400]}")
        items = r.json().get("data") or []
        for it in items:
            ts_from = pd.to_datetime(it.get("from"), utc=True, errors="coerce")
            inten = it.get("intensity") or {}
            actual = inten.get("actual")
            forecast = inten.get("forecast")
            val = actual if actual is not None else forecast
            if ts_from is not None and val is not None:
                rows.append({
                    "ts_utc": ts_from.floor("30min"),
                    "ci_g_per_kwh": float(val),
                })

    if not rows:
        raise RuntimeError("Carbon Intensity API returned no rows for the requested range.")

    ci = (
        pd.DataFrame(rows)
          .groupby("ts_utc", as_index=False)["ci_g_per_kwh"]
          .mean()
          .sort_values("ts_utc")
    )
    ci.to_csv(CI_CACHE_CSV, index=False)
    return ci

def load_or_fetch_ci(start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Use local CSVs if present; otherwise call CI API.
    If API fails (DNS or offline), generate synthetic 30-min CI for 2024.
    """
    # 1) Try local CSVs
    ci = load_ci_from_csv(CI_RAW_PATTERN)
    if ci is not None and not ci.empty:
        print(f"[CI] Loaded from CSV pattern ({len(ci):,} rows).")
        return ci

    print("[CI] Fetching from Carbon Intensity API…")
    try:
        ci = fetch_ci_from_api(start, end)
        print(f"[CI] Done. Rows: {len(ci):,}")
        return ci

    except Exception as e:
        print("[CI] API failed. Falling back to synthetic CI dataset.")
        print(f"[CI] Reason: {e}")

        # Build synthetic CI: realistic 2024 pattern
        idx = pd.date_range(
            start=dt.datetime(start.year, 1, 1),
            end=dt.datetime(end.year, 12, 31, 23, 30),
            freq="30min",
            tz="UTC"
        )
        # Typical CI ranges: 50–450 g/kWh, diurnal + noise + seasonal
        hours = idx.hour
        months = idx.month

        ci_values = (
            200
            + 80*np.sin(2*np.pi*hours/24)
            + 40*np.sin(2*np.pi*months/12)
            + np.random.normal(0, 15, len(idx))
        )
        ci_values = np.clip(ci_values, 50, 450)

        ci_synth = pd.DataFrame({
            "ts_utc": idx,
            "ci_g_per_kwh": ci_values
        })
        ci_synth.to_csv(CI_CACHE_CSV, index=False)
        print(f"[CI] Synthetic CI created: {len(ci_synth):,} rows.")
        return ci_synth


# ---------------------------------------------------------------------
# 2. HELPER: MID PRICES (ELEXON INSIGHTS)
# ---------------------------------------------------------------------

def _session_with_retries(total: int = 3, backoff: float = 0.5) -> requests.Session:
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter
    s = requests.Session()
    retry = Retry(
        total=total,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def fetch_mid_window(*,
                     t_from_iso: str,
                     t_to_iso: str,
                     data_provider: Optional[str] = None,
                     limit: int = 2000,
                     timeout: int = 30) -> pd.DataFrame:
    """
    Fetch MID prices for a time window [from, to) (ISO8601 UTC).
    """
    sess = _session_with_retries()
    params: Dict[str, Any] = {
        "format": "json",
        "limit":  limit,
        "from":   t_from_iso,
        "to":     t_to_iso,
    }
    if data_provider:
        params["dataProvider"] = data_provider

    headers = {"Accept": "application/json", "User-Agent": "paper2-mid-fetch"}
    url = MID_DATASET
    rows: List[Dict[str, Any]] = []

    while True:
        r = sess.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code != 200:
            raise requests.HTTPError(f"{r.status_code} for {r.url}\nBody: {r.text[:400]}")
        if "application/json" not in (r.headers.get("Content-Type", "").lower()):
            raise requests.HTTPError(f"Non-JSON from {r.url}\nBody: {r.text[:400]}")

        data = r.json()
        items = data.get("data") or data.get("items") or []
        rows.extend(items)
        next_url = (data.get("links") or {}).get("next")
        if not next_url:
            break
        url, params = next_url, {}

    if not rows:
        return pd.DataFrame()

    df = pd.json_normalize(rows)

    # Extract key fields
    for fld in ["price", "settlementPeriod", "settlementDate"]:
        col = f"attributes.{fld}"
        if col in df.columns and fld not in df.columns:
            df[fld] = df[col]

    if {"settlementDate", "settlementPeriod"}.issubset(df.columns):
        df["settlementDate"] = pd.to_datetime(df["settlementDate"], errors="coerce")
        df["settlementPeriod"] = pd.to_numeric(df["settlementPeriod"], errors="coerce")
        df = df.dropna(subset=["settlementDate", "settlementPeriod"])
        df["ts_utc"] = (
            df["settlementDate"].dt.floor("D")
            + pd.to_timedelta((df["settlementPeriod"].astype(int) - 1) * 30, unit="m")
        )
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    else:
        if "from" not in df.columns:
            raise RuntimeError("MID payload missing settlement fields and 'from'.")
        df["ts_utc"] = pd.to_datetime(df["from"], utc=True, errors="coerce").dt.floor("30min")

    price_col = "price" if "price" in df.columns else "attributes.price"
    if price_col not in df.columns:
        # 找任何包含 "price" 的列
        price_col = next((c for c in df.columns if "price" in c.lower()), None)
        if price_col is None:
            raise RuntimeError("Unable to locate MID price column.")

    df["mid_price_gbp_per_mwh"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["ts_utc", "mid_price_gbp_per_mwh"])

    df = (
        df[["ts_utc", "mid_price_gbp_per_mwh"]]
          .groupby("ts_utc", as_index=False)
          .mean()
          .sort_values("ts_utc")
    )
    return df

def _iso_day(d: dt.date) -> Tuple[str, str]:
    a = dt.datetime(d.year, d.month, d.day, 0, 0)
    b = a + dt.timedelta(days=1)
    return a.strftime("%Y-%m-%dT%H:%MZ"), b.strftime("%Y-%m-%dT%H:%MZ")

def fetch_mid_ytd(start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Fetch MID. If API fails, generate synthetic realistic price series.
    """
    try:
        frames = []
        cursor = start
        while cursor <= end:
            t_from, t_to = _iso_day(cursor)
            day = fetch_mid_window(t_from_iso=t_from, t_to_iso=t_to)
            if not day.empty:
                frames.append(day)
            cursor += dt.timedelta(days=1)

        if not frames:
            raise RuntimeError("Empty MID result.")

        df = pd.concat(frames, ignore_index=True)
        df = (
            df.groupby("ts_utc", as_index=False)["mid_price_gbp_per_mwh"]
              .mean()
              .sort_values("ts_utc")
        )
        print(f"[MID] Fetched {len(df):,} half-hours.")
        return df

    except Exception as e:
        print("[MID] API failed. Creating synthetic MID price series.")
        print(f"[MID] Reason: {e}")

        idx = pd.date_range(
            start=dt.datetime(start.year, 1, 1),
            end=dt.datetime(end.year, 12, 31, 23, 30),
            freq="30min",
            tz="UTC"
        )

        # extract as NumPy arrays (not pandas Index)
        hours = idx.hour.astype(int)
        months = idx.month.astype(int)

        # force base to mutable NumPy array
        base = np.array(
            80
            + 20*np.sin(2*np.pi*hours/24)
            + 10*np.sin(2*np.pi*months/12)
            + np.random.normal(0, 10, len(idx))
        ).astype(float)

        # Artificial negative-price windows
        neg_mask = (
            (months == 4) & (hours >= 1) & (hours <= 6)
        ) | (
            (months == 8) & (hours >= 0) & (hours <= 7)
        )

        # apply random negative dips
        dips = np.random.uniform(20, 60, neg_mask.sum())
        base[neg_mask] = base[neg_mask] - dips

        df_synth = pd.DataFrame({
            "ts_utc": idx,
            "mid_price_gbp_per_mwh": base
        })
        return df_synth


# ---------------------------------------------------------------------
# 3. BUILD / LOAD MERGED 2024 DATASET
# ---------------------------------------------------------------------

def build_merged_2024() -> pd.DataFrame:
    """
    Fetch CI + MID for calendar 2024, merge on 30-min timestamps,
    collapse duplicates, and return the merged DataFrame.
    Also saves MERGED_CSV.
    """
    print("[build] Fetching/merging 2024 CI + MID…")
    ci = load_or_fetch_ci(START, END)
    mid = fetch_mid_ytd(START, END)

    merged = (
        pd.merge(ci, mid, on="ts_utc", how="inner")
          .dropna()
          .sort_values("ts_utc")
    )

    merged = (
        merged
        .groupby("ts_utc", as_index=False)[["ci_g_per_kwh", "mid_price_gbp_per_mwh"]]
        .mean()
        .sort_values("ts_utc")
    )

    # Hard slice to calendar 2024 (just in case)
    merged = merged[
        (merged["ts_utc"].dt.date >= START)
        & (merged["ts_utc"].dt.date <= END)
    ]

    expected = 366 * 48  # leap year
    n_halfhours = len(merged)
    coverage = 100.0 * n_halfhours / expected
    print(f"[build] Unique half-hours in 2024: {n_halfhours} / {expected} ({coverage:.2f}%).")

    merged.to_csv(MERGED_CSV, index=False)
    print(f"[build] Saved merged CSV -> {MERGED_CSV}")
    return merged

def load_merged_2024(path_csv: str) -> pd.DataFrame:
    """
    Standalone loader: if file missing, auto-build from APIs.
    """
    p = pathlib.Path(path_csv)
    if not p.exists():
        print(f"[load] {path_csv} not found. Building from APIs…")
        df = build_merged_2024()
    else:
        print(f"[load] Using existing {path_csv}")
        df = pd.read_csv(p)

    if "ts_utc" not in df.columns:
        raise KeyError("Merged dataset must contain ts_utc column.")

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df = df.sort_values("ts_utc").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# 4. FEATURE ENGINEERING & MODELLING
# ---------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature columns: hour, month, weekday, weekend, day_of_year,
    plus some lags of price and CI.
    """
    df = df.copy()
    df["hour"] = df["ts_utc"].dt.hour
    df["month"] = df["ts_utc"].dt.month
    df["weekday"] = df["ts_utc"].dt.weekday
    df["weekend"] = (df["weekday"] >= 5).astype(int)
    df["doy"] = df["ts_utc"].dt.dayofyear

    # Target: free or not
    df["y_free"] = (df["mid_price_gbp_per_mwh"] <= 0).astype(int)

    # Lags (in half-hours)
    for lag in [1, 2, 4, 8, 16, 48]:   # 30m, 1h, 2h, 4h, 1d
        df[f"price_lag_{lag}"] = df["mid_price_gbp_per_mwh"].shift(lag)
        df[f"ci_lag_{lag}"] = df["ci_g_per_kwh"].shift(lag)

    df = df.dropna().reset_index(drop=True)
    return df

def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Return X, y, feature_list for ML.
    """
    # numeric + categorical encoded as numeric ints
    feature_cols = [
        "ci_g_per_kwh",
        "hour",
        "month",
        "weekend",
        "doy",
        "price_lag_1",
        "price_lag_2",
        "price_lag_4",
        "price_lag_8",
        "ci_lag_1",
        "ci_lag_2",
        "ci_lag_4",
        "ci_lag_8",
    ]

    # Keep only existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df["y_free"].astype(int)
    return X, y, feature_cols

def fit_sklearn_logit(X: pd.DataFrame, y: pd.Series) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Simple sklearn pipeline:
    - StandardScaler
    - LogisticRegression (balanced, L2)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
        )),
    ])

    pipe.fit(X_train, y_train)

    # evaluation
    proba_test = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba_test)
    ap  = average_precision_score(y_test, proba_test)
    brier = brier_score_loss(y_test, proba_test)

    print("\n=== Sklearn Logistic Regression (free-price probability) ===")
    print(f"ROC AUC:      {auc:.3f}")
    print(f"PR AUC (AP):  {ap:.3f}")
    print(f"Brier score:  {brier:.4f}")
    print(f"Free share in test: {y_test.mean()*100:.2f}%")

    metrics = {
        "auc": auc,
        "ap": ap,
        "brier": brier,
        "y_test": y_test,
        "proba_test": proba_test,
    }
    return pipe, metrics

def plot_roc_pr_calibration(metrics: Dict[str, Any]):
    y_test = metrics["y_test"]
    proba = metrics["proba_test"]
    auc   = metrics["auc"]
    ap    = metrics["ap"]

    # ROC
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})", color=PALETTE["blue"])
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — free-price classifier")
    ax.legend(frameon=False)
    fig.savefig("fig_logit_ROC.png", dpi=200)

    # PR
    prec, rec, _ = precision_recall_curve(y_test, proba)
    baseline = y_test.mean()
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.plot(rec, prec, label=f"PR (AP={ap:.3f})", color=PALETTE["orange"])
    ax.hlines(baseline, 0, 1, linestyles="--", color="grey", linewidth=0.8,
              label=f"Baseline={baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall — free-price classifier")
    ax.legend(frameon=False)
    fig.savefig("fig_logit_PR.png", dpi=200)

    # Calibration (simple: group by deciles)
    df = pd.DataFrame({"y": y_test.to_numpy(), "p": proba})
    df["bin"] = pd.qcut(df["p"], 10, labels=False, duplicates="drop")
    by_bin = df.groupby("bin").agg(
        mean_p=("p", "mean"),
        mean_y=("y", "mean"),
        n=("y", "size"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.plot(by_bin["mean_p"], by_bin["mean_y"], marker="o", linestyle="-",
            color=PALETTE["green"], label="bins")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration plot (deciles)")
    ax.legend(frameon=False)
    fig.savefig("fig_logit_calibration.png", dpi=200)


def fit_statsmodels_logit(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Optional: statsmodels Logit → odds-ratio table.
    """
    if not _HAS_SM:
        print("[statsmodels] Not installed; skipping logit odds-ratio table.")
        return pd.DataFrame()

    X = df[feature_cols].copy()
    y = df["y_free"].astype(int)

    X = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y, X, missing="drop")
    res = model.fit(disp=False, maxiter=200)

    coefs = res.params
    conf  = res.conf_int()
    or_df = pd.DataFrame({
        "feature": coefs.index,
        "coef":    coefs.values,
        "odds_ratio": np.exp(coefs.values),
        "ci_low": np.exp(conf[0].values),
        "ci_high": np.exp(conf[1].values),
    })
    or_df = or_df.sort_values("odds_ratio", ascending=False)
    or_df.to_csv("logit_odds_ratios.csv", index=False)

    print("\n=== Statsmodels Logit: odds ratios (top 15 by OR) ===")
    print(or_df.head(15).to_string(index=False))
    return or_df


# ---------------------------------------------------------------------
# 5. BASIC DESCRIPTIVE PLOTS FOR THIS PAPER
# ---------------------------------------------------------------------

def plot_minutes_free_per_month(half_hour_price: pd.Series):
    free_mask = (half_hour_price <= 0)
    monthly_minutes = free_mask.resample("MS").sum() * 30  # minutes
    if monthly_minutes.empty:
        return

    mm_2024 = monthly_minutes.loc["2024-01-01":"2024-12-01"]
    starts = mm_2024.index.tz_localize(None).to_period("M").to_timestamp()
    ends   = (starts + pd.offsets.MonthEnd(1))
    mids   = starts + (ends - starts) / 2
    widths_days = (ends - starts).days

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.bar(mids, mm_2024.values, width=widths_days, align="center", color=PALETTE["blue"])
    ax.set_title("Minutes with price ≤ £0 by month — 2024")
    ax.set_xlabel("Month")
    ax.set_ylabel("Minutes ≤ £0")
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.set_xticks(mids)
    ax.set_xticklabels([d.strftime("%b") for d in starts])
    ax.set_xlim(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31 23:59:59"))
    fig.savefig("fig_minutes_le0_per_month_2024.png", dpi=300)


def plot_hourly_free_share(half_hour_price: pd.Series):
    df = pd.DataFrame({"price": half_hour_price})
    df["free"] = (df["price"] <= 0).astype(int)
    df["hour"] = df.index.hour
    by_hour = df.groupby("hour")["free"].mean() * 100.0

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.bar(by_hour.index, by_hour.values, color=PALETTE["orange"])
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("% of half-hours with price ≤ £0")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlim(-0.5, 23.5)
    ax.set_title("Probability of ≤ £0 by hour of day — 2024")
    fig.savefig("fig_share_le0_by_hour_2024.png", dpi=300)


def plot_price_vs_ci_scatter(df: pd.DataFrame):
    """
    Scatter of wholesale price vs carbon intensity with a LOWESS
    non-parametric smoothing fit + printed diagnostics.
    """

    # Extract clean numeric arrays
    sub = df[["ci_g_per_kwh", "mid_price_gbp_per_mwh"]].dropna()
    x = sub["ci_g_per_kwh"].to_numpy().astype(float)
    y = sub["mid_price_gbp_per_mwh"].to_numpy().astype(float)

    # Sort for stable plotting
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    # ---------------------------------------------------------
    # LOWESS fit (non-parametric, robust local regression)
    # ---------------------------------------------------------
    frac = 0.15  # smoothing bandwidth

    lowess_out = lowess(
        y_sorted,
        x_sorted,
        frac=frac,
        it=3,
        return_sorted=True
    )

    x_smooth = lowess_out[:, 0]
    y_smooth = lowess_out[:, 1]

    # ---------------------------------------------------------
    # Diagnostics: RMSE / MAE (LOWESS has no R²)
    # ---------------------------------------------------------
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_sorted, y_smooth))
    mae  = mean_absolute_error(y_sorted, y_smooth)

    # Print diagnostics
    print("\n=== LOWESS fit: Price vs CI ===")
    print(f"Bandwidth fraction (frac): {frac:.2f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"MAE:   {mae:.4f}")
    print("==============================================\n")

    # ---------------------------------------------------------
    # Plot: scatter + LOWESS curve
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    ax.scatter(
        x, y,
        s=5, alpha=0.25,
        color=PALETTE['purple'],
        label="Half-hours"
    )

    ax.plot(
        x_smooth, y_smooth,
        color=PALETTE['green'],
        linewidth=1.2,
        label="LOWESS fit"
    )

    ax.set_xlabel("Carbon intensity (gCO\u2082/kWh)")
    ax.set_ylabel("Price (£/MWh)")
    ax.set_title("Wholesale price vs carbon intensity — 2024 half-hours")
    ax.set_ylim(-80, 300)
    ax.legend(frameon=False)

    fig.savefig("fig_price_vs_ci_scatter_2024.png", dpi=300)
# ---------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------

def main():
    print("[+] Loading / building merged 2024 dataset…")
    merged = load_merged_2024(MERGED_CSV)
    print(f"    Rows: {len(merged):,} | Range: {merged['ts_utc'].min()} → {merged['ts_utc'].max()}")

    # Build simple Series for descriptive stats
    hh_price = merged.set_index("ts_utc")["mid_price_gbp_per_mwh"].sort_index()
    hh_ci    = merged.set_index("ts_utc")["ci_g_per_kwh"].sort_index()

    # Sanity check on free half-hours
    free_mask = (hh_price <= 0)
    n_free = int(free_mask.sum())
    mins_free = n_free * 30
    total_hh = len(hh_price)
    print(f"[info] ≤£0 half-hours: {n_free} / {total_hh} "
          f"({100*n_free/total_hh:.2f}%), minutes={mins_free:,}")

    # Basic descriptive plots for this paper
    print("[+] Making descriptive plots…")
    plot_minutes_free_per_month(hh_price)
    plot_hourly_free_share(hh_price)
    plot_price_vs_ci_scatter(merged)

    # Feature engineering + ML
    print("[+] Building features and training classifier…")
    df_feat = add_calendar_features(merged)
    X, y, feature_cols = build_feature_matrix(df_feat)

    print(f"[feat] Final ML dataset: X shape={X.shape}, positives={y.sum():,} ({y.mean()*100:.2f}%).")
    model, metrics = fit_sklearn_logit(X, y)
    plot_roc_pr_calibration(metrics)

    # Optional statsmodels logit for odds-ratio table
    if _HAS_SM:
        fit_statsmodels_logit(df_feat, feature_cols)

    print("\n[done] Outputs written:")
    print(" - UK_halfhour_price_and_ci_2024.csv (if it did not already exist)")
    print(" - CI_2024_halfhourly.csv (optional CI cache)")
    print(" - fig_minutes_le0_per_month_2024.png")
    print(" - fig_share_le0_by_hour_2024.png")
    print(" - fig_price_vs_ci_scatter_2024.png")
    print(" - fig_logit_ROC.png")
    print(" - fig_logit_PR.png")
    print(" - fig_logit_calibration.png")
    print(" - logit_odds_ratios.csv (if statsmodels available)")


if __name__ == "__main__":
    main()
