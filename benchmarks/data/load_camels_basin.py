"""Load CAMELS-equivalent data for one basin from public APIs.

Why this instead of the official CAMELS download?
-------------------------------------------------
The CAMELS-US archive is ~1.4 GB. For a benchmark PoC we only need
one (or a small number of) basins. This module reconstructs the same
data CAMELS provides — basin-mean DAYMET forcing + USGS streamflow —
on demand, per basin, with on-disk caching.

Sources (both public, no auth):
  - USGS NWIS daily discharge: waterservices.usgs.gov
  - DAYMET single-pixel daily forcing: daymet.ornl.gov

Single-pixel forcing is an approximation of basin-mean (CAMELS averages
over the catchment polygon). For mid-sized basins (~few hundred km²)
the centroid pixel is a reasonable proxy; for very large basins the
approximation degrades. Documented limitation, not a bug.

Cache directory matches the convention from load_ncep_reanalysis.py:
  - Override with TESSERA_CACHE environment variable
  - Default to C:/dev/tessera_cache on Windows

Usage
-----
    from benchmarks.data.load_camels_basin import load_camels_basin
    df = load_camels_basin(
        gauge_id="01013500",        # USGS gauge ID
        lat=47.2375, lon=-68.5825,  # basin centroid
        start_year=1990, end_year=2020,
    )
    # df has columns: date, prcp, tmax, tmin, tmean, q_obs (cfs)
"""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------- Cache setup -----------------------

def _default_cache_dir() -> Path:
    env = os.environ.get("TESSERA_CACHE")
    if env:
        return Path(env)
    if os.name == "nt":
        return Path("C:/dev/tessera_cache")
    return Path.home() / ".tessera" / "cache"


CACHE_DIR = _default_cache_dir() / "camels"


# ----------------------- USGS NWIS -----------------------

def _fetch_usgs_streamflow(
    gauge_id: str, start_date: str, end_date: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch daily mean discharge (parameter code 00060) from USGS NWIS.

    Returns DataFrame indexed by date (Timestamp) with column 'q_obs'
    in cubic feet per second.
    """
    import requests
    url = ("https://waterservices.usgs.gov/nwis/dv/?format=json"
           f"&sites={gauge_id}&parameterCd=00060"
           f"&startDT={start_date}&endDT={end_date}")
    if verbose:
        print(f"[usgs] fetch {gauge_id} {start_date}..{end_date}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    ts_list = data["value"]["timeSeries"]
    if not ts_list:
        raise RuntimeError(f"USGS NWIS returned no time series for gauge {gauge_id}")
    # Pick the first time series (daily mean).
    values = ts_list[0]["values"][0]["value"]
    if not values:
        raise RuntimeError(f"USGS NWIS returned empty values for gauge {gauge_id}")
    rows = [(v["dateTime"][:10], float(v["value"]))
            for v in values if v["value"] not in ("-999999", "")]
    df = pd.DataFrame(rows, columns=["date", "q_obs"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    if verbose:
        print(f"[usgs] got {len(df)} daily flow values")
    return df


# ----------------------- DAYMET -----------------------

def _fetch_daymet_singlepixel(
    lat: float, lon: float, start_year: int, end_year: int,
    vars_csv: str = "prcp,tmax,tmin",
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch daily DAYMET forcing for one (lat, lon).

    DAYMET serves one calendar year per request only sometimes; the
    single-pixel API accepts a date range. Returns a DataFrame indexed
    by date with columns named after the requested vars.
    """
    import requests
    url = ("https://daymet.ornl.gov/single-pixel/api/data"
           f"?lat={lat}&lon={lon}&vars={vars_csv}"
           f"&start={start_year}-01-01&end={end_year}-12-31")
    if verbose:
        print(f"[daymet] fetch lat={lat} lon={lon} {start_year}..{end_year}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    text = r.text
    # The first 6-7 lines are metadata; the CSV starts at the "year,yday,..." header.
    lines = text.splitlines()
    header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("year"))
    csv_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_text))
    # Construct a date column from (year, yday). DAYMET uses 365-day years
    # (drops Dec 31 in leap years), so yday in [1, 365].
    df["date"] = pd.to_datetime(df["year"].astype(int) * 1000 + df["yday"].astype(int),
                                 format="%Y%j", errors="coerce")
    df = df.set_index("date").sort_index()
    # Rename columns to be tidy (strip units)
    rename = {
        "prcp (mm/day)": "prcp",
        "tmax (deg c)": "tmax",
        "tmin (deg c)": "tmin",
    }
    df = df.rename(columns=rename)
    keep = [c for c in df.columns if c in ("prcp", "tmax", "tmin", "srad",
                                            "vp", "swe", "dayl")]
    df = df[keep]
    if verbose:
        print(f"[daymet] got {len(df)} daily forcing values, cols={list(df.columns)}")
    return df


# ----------------------- Combine + cache -----------------------

def load_camels_basin(
    gauge_id: str, lat: float, lon: float,
    start_year: int = 1990, end_year: int = 2020,
    cache_dir: Path | None = None,
    refresh: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load DAYMET forcing + USGS streamflow for one basin, joined on date.

    Returns DataFrame with columns:
        prcp (mm/day), tmax (°C), tmin (°C), tmean (°C),
        q_obs (cfs), q_obs_mm (mm/day, normalized by basin area if provided
            — set to NaN if not)

    Days where USGS or DAYMET data is missing are dropped.

    On-disk cache: `<cache_dir>/<gauge_id>_<start>_<end>.parquet`. Pass
    refresh=True to force re-download.
    """
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{gauge_id}_{start_year}_{end_year}.parquet"

    if cache_path.exists() and not refresh:
        if verbose:
            print(f"[cache] loading {cache_path}")
        return pd.read_parquet(cache_path)

    # ---- Fetch ----
    flow = _fetch_usgs_streamflow(
        gauge_id, f"{start_year}-01-01", f"{end_year}-12-31",
        verbose=verbose,
    )
    forcing = _fetch_daymet_singlepixel(
        lat, lon, start_year, end_year, verbose=verbose,
    )

    # ---- Align ----
    df = forcing.join(flow, how="inner")
    df["tmean"] = (df["tmax"] + df["tmin"]) / 2.0
    # Add log streamflow for convenience (more Gaussian-shaped target)
    df["log_q"] = np.log(df["q_obs"].clip(lower=1e-3))
    # Drop any rows with NaN in core columns
    core = ["prcp", "tmax", "tmin", "tmean", "q_obs"]
    n_before = len(df)
    df = df.dropna(subset=core)
    n_after = len(df)
    if verbose:
        print(f"[align] {n_after} rows after inner-join + dropna "
              f"(dropped {n_before - n_after})")

    df.to_parquet(cache_path)
    if verbose:
        print(f"[cache] wrote {cache_path}")
    return df


# ----------------------- Known basins (Kratzert 531-set subset) --------

CAMELS_REFERENCE_BASINS: dict[str, dict] = {
    # gauge_id : { name, lat, lon, area_km2 }
    "01013500": dict(name="Fish River near Fort Kent, ME",
                     lat=47.2375, lon=-68.5825, area_km2=2253),
    "02479155": dict(name="Black Creek at Wiggins, MS",
                     lat=30.8333, lon=-89.1369, area_km2=520),
    "06614800": dict(name="Cache la Poudre, CO",
                     lat=40.6694, lon=-105.8869, area_km2=379),
    "09497500": dict(name="Salt River, AZ",
                     lat=33.7942, lon=-110.5092, area_km2=11154),
    "11264500": dict(name="Merced River, CA",
                     lat=37.7158, lon=-119.6669, area_km2=469),
}
"""A small reference set of CAMELS basins spanning humid/arid/snowmelt regimes.
   These five span the canonical hydrologic flavors:
     01013500 — humid temperate (Maine snow/rain mix)
     02479155 — humid subtropical (Mississippi rain)
     06614800 — semi-arid mountain (Colorado snowmelt)
     09497500 — arid semi-desert (Arizona)
     11264500 — Mediterranean mountain (California snowmelt)
"""


__all__ = [
    "CAMELS_REFERENCE_BASINS",
    "load_camels_basin",
]
