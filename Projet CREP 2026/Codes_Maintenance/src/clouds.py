import numpy as np
from src.config import RESSOURCES_DIR, CONSTANT_CLOUD_ALBEDO
from src.data_loader import load_monthly_cloud_albedo_from_ceres, smooth_annual_data


def prepare_cloud_albedo(grid: dict, use_cloud_albedo: bool = True):
    """
    Renvoie l'albédo des nuages sur 365 jours.

    Sortie :
    cloud_albedo_daily.shape = (365, N_THETA, N_PHI)
    """

    n_theta, n_phi = grid["lat_grid"].shape

    if not use_cloud_albedo:
        print("Albédo des nuages constant = 0.0")
        return np.full((365, n_theta, n_phi), CONSTANT_CLOUD_ALBEDO)

    ceres_path = (
        RESSOURCES_DIR
        / "albedo"
        / "CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc"
    )

    ceres_full = load_monthly_cloud_albedo_from_ceres(
        ceres_path,
        lat_deg=None,
        lon_deg=None,
        return_full_map=True
    )

    monthly_cloud_grid = ceres_full.sel(
        lat=grid["lat"],
        lon=grid["lon"],
        method="nearest"
    ).to_numpy()

    cloud_daily = smooth_annual_data(monthly_cloud_grid, sigma=15.0)

    print("Albédo des nuages préparé :", cloud_daily.shape) # matrice
    return cloud_daily