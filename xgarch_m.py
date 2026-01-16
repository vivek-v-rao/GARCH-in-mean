"""
Fit (G)ARCH and GJR-GARCH models (optionally with GARCH-in-mean) to daily returns.

- Reads daily prices from a CSV using pandas
- Computes simple or log returns (toggle)
- Fits a list of volatility models (toggle)
- Prints a final table with:
    * GARCH-M parameter estimate and t-stat (kappa, if enabled)
    * Other GARCH parameter estimates (no t-stats)
    * If error_dist == "t", also prints nu (dof) before status
"""

from __future__ import annotations

import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from arch.univariate import ARCHInMean, ConstantMean, GARCH, Normal, StudentsT


# -----------------------
# user settings
# -----------------------

input_csv = "prices.csv"               # set path here (or pass as 1st CLI arg)
date_col = "Date"                      # set to None if no date column
price_cols: Optional[List[str]] = ["SPY", "EFA", "EEM", "TLT", "IEF", "LQD"] # None # None -> all columns except date_col

use_log_returns = True                 # True -> log returns, False -> simple returns
return_scale = 100.0                   # arch is often happier with returns in percent

models_to_fit = ["garch", "gjr"]       # allowed: "garch", "gjr"

p = 1                                  # arch order (lags of squared shocks) in the GARCH variance equation
q = 1                                  # garch order (lags of conditional variance) in the GARCH variance equation
o_gjr = 1                              # asymmetry order for GJR (usually 1)

use_garch_in_mean = True               # True -> estimate kappa, False -> standard mean model
archm_form = "var"                     # "vol" | "var" | "log" or a nonzero float

error_dist = "t"                       # "normal" or "t"
fit_cov_type = "robust"                # "robust" or "classic"

# fit controls
show_warnings = False
fit_disp = "off"
fit_update_freq = 0


# -----------------------
# helpers
# -----------------------

def load_prices(path: str, date_col: Optional[str], price_cols: Optional[List[str]]) -> pd.DataFrame:
    df = pd.read_csv(path)

    if date_col is not None and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)
        df = df.sort_index()

    if price_cols is None:
        cols = list(df.columns)
        if date_col is not None and date_col in cols:
            cols.remove(date_col)
        price_cols = cols

    df = df.loc[:, price_cols].copy()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop columns that are entirely missing/non-numeric
    df = df.dropna(axis=1, how="all")
    return df


def compute_returns(prices: pd.DataFrame, use_log: bool) -> pd.DataFrame:
    if use_log:
        rets = np.log(prices).diff()
    else:
        rets = prices.pct_change()
    return rets


def make_volatility_model(model_name: str, p: int, q: int, o_gjr: int) -> Tuple[str, GARCH]:
    name = model_name.strip().lower()
    if name == "garch":
        return "garch", GARCH(p=p, o=0, q=q)
    if name == "gjr":
        return "gjr", GARCH(p=p, o=o_gjr, q=q)
    raise ValueError(f"unknown model_name: {model_name!r}")


def make_distribution(dist_name: str):
    d = dist_name.strip().lower()
    if d in ("normal", "gaussian"):
        return Normal()
    if d in ("t", "student", "studentst", "students_t"):
        return StudentsT()
    raise ValueError(f"unknown error_dist: {dist_name!r}")


def is_student_t(dist_name: str) -> bool:
    d = dist_name.strip().lower()
    return d in ("t", "student", "studentst", "students_t")


def fit_arch_model(y: pd.Series, vol, use_archm: bool):
    dist = make_distribution(error_dist)

    if use_archm:
        am = ARCHInMean(y, volatility=vol, distribution=dist, form=archm_form)
    else:
        am = ConstantMean(y, volatility=vol, distribution=dist)

    res = am.fit(
        update_freq=fit_update_freq,
        disp=fit_disp,
        cov_type=fit_cov_type,
        show_warning=show_warnings,
    )
    return res


def safe_get(params: pd.Series, keys: List[str]) -> float:
    for k in keys:
        if k in params.index:
            return float(params[k])
    return np.nan


def find_archm_param_name(params_index: List[str]) -> Optional[str]:
    # The in-mean coefficient in arch is typically named "kappa"
    preferred = ["kappa", "lambda"]
    lower_map = {k.lower(): k for k in params_index}

    for cand in preferred:
        if cand in lower_map:
            return lower_map[cand]

    # fallback heuristics
    for k in params_index:
        lk = k.lower()
        if "kappa" in lk:
            return k
        if ("arch" in lk) and ("mean" in lk):
            return k

    return None


# -----------------------
# main
# -----------------------

def main() -> int:
    path = input_csv
    if len(sys.argv) >= 2:
        path = sys.argv[1]

    if not show_warnings:
        warnings.filterwarnings("ignore")

    prices = load_prices(path, date_col=date_col, price_cols=price_cols)
    if prices.empty:
        print("no usable price columns found")
        return 1
    print(f"prices file: {path}")

    if isinstance(prices.index, pd.DatetimeIndex) and prices.index.notna().any():
        dmin = prices.index.min()
        dmax = prices.index.max()
        print(f"date range: {dmin.strftime('%Y-%m-%d')} to {dmax.strftime('%Y-%m-%d')}")
    else:
        print("date range: (no datetime index)")
    print("log returns?", use_log_returns)
    print("return scaling:", return_scale, end="\n\n")

    rets = compute_returns(prices, use_log=use_log_returns)
    rets = rets.dropna(how="all")

    rows: List[Dict[str, object]] = []

    for series_name in rets.columns:
        y0 = rets[series_name].dropna()
        if y0.size < 50:
            continue

        y = (return_scale * y0).astype(float)

        for model_name in models_to_fit:
            model_label, vol = make_volatility_model(model_name, p=p, q=q, o_gjr=o_gjr)

            row: Dict[str, object] = {
                "series": series_name,
                "model": model_label,
                "nobs": int(y.shape[0]),
            }

            try:
                res = fit_arch_model(y, vol, use_archm=use_garch_in_mean)

                params = res.params
                tvals = getattr(res, "tvalues", None)

                archm_name = find_archm_param_name(list(params.index))
                if use_garch_in_mean and archm_name is not None:
                    row["archm"] = float(params[archm_name])
                    row["archm_t"] = float(tvals[archm_name]) if tvals is not None else np.nan
                else:
                    row["archm"] = np.nan
                    row["archm_t"] = np.nan

                # other parameters: values only (no t-stats)
                row["mu"] = safe_get(params, ["mu", "Const", "const"])
                row["omega"] = safe_get(params, ["omega"])
                row["alpha1"] = safe_get(params, ["alpha[1]", "alpha1"])
                row["beta1"] = safe_get(params, ["beta[1]", "beta1"])
                row["gamma1"] = safe_get(params, ["gamma[1]", "gamma1"])  # only for GJR; NaN otherwise

                # dof for Student's t
                if is_student_t(error_dist):
                    row["nu"] = safe_get(params, ["nu"])
                else:
                    row["nu"] = np.nan

                row["status"] = "ok"

            except Exception as e:
                row["archm"] = np.nan
                row["archm_t"] = np.nan
                row["mu"] = np.nan
                row["omega"] = np.nan
                row["alpha1"] = np.nan
                row["beta1"] = np.nan
                row["gamma1"] = np.nan
                row["nu"] = np.nan
                row["status"] = f"fail: {type(e).__name__}"

            rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        print("no models were fit (not enough data?)")
        return 1

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 100)

    # column order: put nu before status only when error_dist is t
    col_order = ["series", "model", "nobs", "archm", "archm_t", "mu", "omega", "alpha1", "gamma1", "beta1"]
    if is_student_t(error_dist):
        col_order += ["nu"]
    col_order += ["status"]

    for c in col_order:
        if c not in out.columns:
            out[c] = np.nan
    out = out[col_order]

    float_cols = ["archm", "archm_t", "mu", "omega", "alpha1", "gamma1", "beta1", "nu"]
    for c in float_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    print(out.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
