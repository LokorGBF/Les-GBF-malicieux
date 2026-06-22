import pathlib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import binned_statistic_2d

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False


def smooth_annual_data(monthly_values: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """
    Transforme 12 valeurs mensuelles en 365 valeurs journalières lissées
    """
    days_per_month = np.array(
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    )

    daily_discontinuous = np.repeat(monthly_values, days_per_month, axis=0)

    return gaussian_filter1d(
        daily_discontinuous,
        sigma=sigma,
        mode="wrap",
        axis=0
    )


def load_albedo_series(csv_dir: pathlib.Path, pattern: str = "albedo{:02d}.csv"):
    """
    Charge les 12 fichiers CSV d'albédo de surface.

    Sortie :
    - monthly_albedo : shape (12, nlat, nlon)
    - latitudes
    - longitudes
    """
    if not csv_dir.exists():
        raise FileNotFoundError(f"Dossier d'albédo introuvable : {csv_dir}")

    latitudes, longitudes, cubes = None, None, []

    for month in range(1, 13):
        file_path = csv_dir / pattern.format(month)
        df = pd.read_csv(file_path)

        if latitudes is None:
            latitudes = df["Latitude/Longitude"].astype(float).to_numpy()
            longitudes = df.columns[1:].astype(float).to_numpy()

        cube_month = df.set_index("Latitude/Longitude").to_numpy(dtype=float)
        cubes.append(cube_month)

    print("Albédo de surface chargé.")
    return np.stack(cubes, axis=0), latitudes, longitudes


def load_and_grid_rzsm_data(csv_path: pathlib.Path):
    """
    Charge le fichier CSV d'humidité du sol RZSM et le met sur une grille régulière.

    Sortie :
    - RZSM_GRID : matrice 2D
    - lat_bins
    - lon_bins
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier RZSM introuvable : {csv_path}")

    df = pd.read_csv(csv_path)
    df["lon"] = ((df["lon"] + 180) % 360) - 180

    grid_res = 1.0
    lon_bins = np.arange(-180, 180 + grid_res, grid_res)
    lat_bins = np.arange(-90, 90 + grid_res, grid_res)

    statistic, _, _, _ = binned_statistic_2d(
        x=df["lon"],
        y=df["lat"],
        values=df["RZSM"],
        statistic="mean",
        bins=[lon_bins, lat_bins],
    )

    print("Humidité du sol RZSM chargée.")
    return statistic.T, lat_bins, lon_bins


def load_monthly_cloud_albedo_from_ceres(
    ceres_file_path: pathlib.Path,
    lat_deg=None,
    lon_deg=None,
    return_full_map: bool = False
):
    """
    Charge l'albédo des nuages depuis le fichier CERES NetCDF.

    Si return_full_map=True : renvoie la carte complète mois/lat/lon.
    Sinon : renvoie les 12 valeurs mensuelles pour un point donné.
    """
    if not XARRAY_AVAILABLE:
        print("xarray indisponible : albédo nuages = 0.")
        return np.zeros(12)

    if not ceres_file_path.exists():
        raise FileNotFoundError(f"Fichier CERES introuvable : {ceres_file_path}")

    with xr.open_dataset(ceres_file_path, decode_times=True) as ds:
        ds.load()

        ds = ds.assign_coords(
            lon=(((ds.lon + 180) % 360) - 180)
        ).sortby("lon")

        toa_sw_all = ds["toa_sw_all_mon"]
        toa_sw_clr = ds["toa_sw_clr_c_mon"]
        solar_in = ds["solar_mon"]

        cloud_albedo = xr.where(
            solar_in > 1e-6,
            (toa_sw_all - toa_sw_clr) / solar_in,
            0.0
        )

        cloud_albedo_monthly = cloud_albedo.groupby("time.month").mean(
            dim="time",
            skipna=True
        )

        if return_full_map:
            print("Albédo des nuages CERES chargé.")
            return cloud_albedo_monthly

        values = cloud_albedo_monthly.sel(
            lat=lat_deg,
            lon=lon_deg,
            method="nearest"
        ).to_numpy()

    if len(values) != 12:
        values = np.pad(values, (0, 12 - len(values)), mode="edge")

    return values


def upscale_grid(data_lowres, lat_low, lon_low, lat_target, lon_target):
    """
    Interpole une grille 2D vers une nouvelle grille latitude-longitude.

    Entrées :
    - data_lowres : shape (nlat_low, nlon_low)
    - lat_low, lon_low : coordonnées source
    - lat_target, lon_target : coordonnées cible 1D

    Sortie :
    - data_target : shape (len(lat_target), len(lon_target))
    """
    interp = RegularGridInterpolator(
        (lat_low, lon_low),
        data_lowres,
        bounds_error=False,
        fill_value=None
    )

    points = np.array(np.meshgrid(lat_target, lon_target, indexing="ij"))
    points = np.moveaxis(points, 0, -1)

    return interp(points)


def create_continent_finder(shapefile_path: pathlib.Path):
    """
    Crée une fonction continent_finder(lat, lon).
    """
    if not GEOPANDAS_AVAILABLE:
        print("GeoPandas indisponible : tous les points seront traités comme Océan.")
        return lambda lat, lon: "Océan"

    try:
        world = gpd.read_file(shapefile_path).to_crs(epsg=4326)
    except Exception as e:
        print(f"Impossible de charger le shapefile : {e}")
        return lambda lat, lon: "Océan"

    def find_continent_for_point(lat: float, lon: float) -> str:
        point = Point(lon, lat)
        valid_world = world[world.geometry.notna()]

        for _, row in valid_world.iterrows():
            if row["geometry"].contains(point):
                return row["CONTINENT"]

        return "Océan"

    return find_continent_for_point