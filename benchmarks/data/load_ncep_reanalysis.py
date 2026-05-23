"""Load a gridded atmospheric field from NCEP/NCAR Reanalysis 2.

Why NCEP/NCAR Reanalysis 2 (and not ERA5)?
------------------------------------------
ERA5 is the gold-standard atmospheric reanalysis but requires a Copernicus
CDS account + API key (`pip install cdsapi`, `~/.cdsapirc` with UID/key).
NCEP/NCAR Reanalysis 2 is similar in nature (global daily reanalysis on a
~1.9° Gaussian grid, since 1979) but is hosted PUBLICLY on NOAA PSL with no
auth — one HTTP GET per yearly netCDF (~3 MB compressed). Perfect for a
reproducible benchmark.

If you have CDS credentials, swap `load_ncep_t2m` for an ERA5 retrieve;
the rest of the benchmark code works on the (T, Y, X) array unchanged.

Data attribution
----------------
NCEP-DOE AMIP-II Reanalysis (R-2). Data provided by the NOAA Physical
Sciences Laboratory at https://psl.noaa.gov/data/gridded/data.ncep.reanalysis2.html

Usage
-----
    from benchmarks.data.load_ncep_reanalysis import load_ncep_t2m
    field, lats, lons, days = load_ncep_t2m(year=2023, lat_min=25, lat_max=50,
                                            lon_min=235, lon_max=290)
    # field shape: (T_days, Y_lats, X_lons), in Kelvin
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np

# Cache directory.
# NOTE: must be on an ASCII path. The tessera repo lives behind a Windows
# junction (C:\dev\tessera) backing a OneDrive path with CJK characters;
# Python's import machinery canonicalises __file__ / os.getcwd() through
# the junction, so we can't reliably derive an ASCII path from __file__.
# Default to C:\dev\tessera_cache (sibling of the junction); override with
# the TESSERA_CACHE environment variable for other systems / Unix.
def _default_cache_dir() -> Path:
    env = os.environ.get("TESSERA_CACHE")
    if env:
        return Path(env)
    if os.name == "nt":
        return Path("C:/dev/tessera_cache")
    return Path.home() / ".tessera" / "cache"


CACHE_DIR = _default_cache_dir()


def _download_ncep_year(
    year: int,
    variable: str = "air.2m.gauss",
    cache_dir: Path | None = None,
    verbose: bool = True,
) -> Path:
    """Download one year of NCEP Reanalysis 2 daily data, caching to disk.

    Parameters
    ----------
    year : int
        Calendar year (1979 onward).
    variable : str
        NCEP filename stem. Common choices:
          - 'air.2m.gauss' (2-m air temperature, K)
          - 'prate.sfc.gauss' (surface precipitation rate, kg/m²/s)
          - 'uwnd.10m.gauss' (10-m zonal wind, m/s)
          - 'vwnd.10m.gauss' (10-m meridional wind, m/s)
          - 'slp' (sea-level pressure, Pa) — on regular 2.5° grid, not gauss
    cache_dir : Path | None
        Where to cache downloaded files. Default: `_cache/` next to this file.

    Returns
    -------
    Path to the cached netCDF file.
    """
    import requests

    if cache_dir is None:
        cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{variable}.{year}.nc"
    local = cache_dir / fname
    if local.exists() and local.stat().st_size > 0:
        if verbose:
            print(f"[ncep] cached: {local} ({local.stat().st_size / 1e6:.1f} MB)")
        return local

    # NCEP R2 daily mean on Gaussian grid
    if "gauss" in variable:
        url = ("https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis2/"
               f"Dailies/gaussian_grid/{fname}")
    else:
        url = ("https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis2/"
               f"Dailies/surface/{fname}")

    if verbose:
        print(f"[ncep] downloading {url}")
    r = requests.get(url, timeout=60, stream=True)
    r.raise_for_status()
    with open(local, "wb") as f:
        for chunk in r.iter_content(chunk_size=64 * 1024):
            f.write(chunk)
    if verbose:
        print(f"[ncep] saved {local} ({local.stat().st_size / 1e6:.1f} MB)")
    return local


def load_ncep_t2m(
    year: int = 2023,
    lat_min: float = 25.0,
    lat_max: float = 50.0,
    lon_min: float = 235.0,
    lon_max: float = 290.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load 2-m air temperature on a regional grid.

    Defaults give a CONUS box (roughly Florida → Maine → Washington state).

    NCEP longitudes are 0-360 (e.g. CONUS is 235-290). Latitudes run
    NORTH-to-SOUTH (90, ..., -90) on the Gaussian grid — this loader
    re-orders to south→north so the array is geographically intuitive.

    Returns
    -------
    field : np.ndarray of shape (T, Y, X)
        Temperature in Kelvin.
    lats  : np.ndarray of shape (Y,)
        Latitudes (south to north).
    lons  : np.ndarray of shape (X,)
        Longitudes (west to east, in NCEP 0-360 convention).
    days  : np.ndarray of shape (T,)
        Day-of-year (1-365 or 366).
    """
    from netCDF4 import Dataset  # type: ignore

    path = _download_ncep_year(year, "air.2m.gauss", verbose=verbose)
    with Dataset(path, "r") as ds:
        air = ds.variables["air"]
        lats_full = ds.variables["lat"][:].astype(np.float64)
        lons_full = ds.variables["lon"][:].astype(np.float64)
        time = ds.variables["time"][:].astype(np.float64)

        # NCEP lats are north→south. Find the subset indices.
        lat_mask = (lats_full >= lat_min) & (lats_full <= lat_max)
        lon_mask = (lons_full >= lon_min) & (lons_full <= lon_max)
        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]
        if len(lat_idx) == 0 or len(lon_idx) == 0:
            raise ValueError(
                f"Empty lat/lon subset: lat∈[{lat_min},{lat_max}] "
                f"lon∈[{lon_min},{lon_max}]; available lat range "
                f"{lats_full.min():.1f}..{lats_full.max():.1f}, "
                f"lon {lons_full.min():.1f}..{lons_full.max():.1f}")

        # Slice and load. NCEP `air` has 4 dims (time, level, lat, lon) with
        # a degenerate level axis of length 1 — squeeze it out.
        if air.ndim == 4:
            sub = air[:, 0, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
        else:
            sub = air[:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
        sub = np.asarray(sub, dtype=np.float64)
        lats = lats_full[lat_idx[0]:lat_idx[-1]+1]
        lons = lons_full[lon_idx[0]:lon_idx[-1]+1]

        # Reorient lats to south→north
        if lats[0] > lats[-1]:
            lats = lats[::-1]
            sub = sub[:, ::-1, :]

    # NCEP time is "hours since 1800-01-01"; convert to 1..N day index
    days = np.arange(1, sub.shape[0] + 1, dtype=np.float64)
    if verbose:
        print(f"[ncep] loaded {sub.shape} (T_days, lats={len(lats)}, lons={len(lons)})")
        print(f"[ncep] lat range {lats[0]:.1f}..{lats[-1]:.1f}, "
              f"lon range {lons[0]:.1f}..{lons[-1]:.1f}")
        print(f"[ncep] T mean = {sub.mean():.1f} K, std = {sub.std():.1f} K")
    return sub, lats, lons, days


if __name__ == "__main__":
    field, lats, lons, days = load_ncep_t2m(year=2023)
    print(f"\nField shape: {field.shape}")
    print(f"T range: {field.min():.1f} - {field.max():.1f} K  "
          f"({field.min()-273.15:.1f} - {field.max()-273.15:.1f} C)")
