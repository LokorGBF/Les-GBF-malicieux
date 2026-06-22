import numpy as np
from src.config import RESSOURCES_DIR, CONSTANT_SURFACE_ALBEDO
from src.data_loader import load_albedo_series, smooth_annual_data, upscale_grid


def prepare_surface_albedo(grid: dict, use_variable_albedo: bool = True):
    """
    Renvoie l'albédo du sol sur 365 jours.

    Sortie :
    albedo_daily.shape = (365, N_THETA, N_PHI)
    """

    n_theta, n_phi = grid["lat_grid"].shape

    if not use_variable_albedo:
        print("Albédo de surface constant = 0.30 utilisé.")
        return np.full((365, n_theta, n_phi), CONSTANT_SURFACE_ALBEDO)

    albedo_dir = RESSOURCES_DIR / "albedo"

    monthly_albedo_lowres, lat_low, lon_low = load_albedo_series(albedo_dir)

    monthly_albedo_grid = np.array([
        upscale_grid(
            monthly_albedo_lowres[m],
            lat_low,
            lon_low,
            grid["lat"],
            grid["lon"]
        )
        for m in range(12)
    ])

    albedo_daily = smooth_annual_data(monthly_albedo_grid, sigma=15.0)

    print("Albédo de surface préparé :", albedo_daily.shape) # matrice
    return albedo_daily