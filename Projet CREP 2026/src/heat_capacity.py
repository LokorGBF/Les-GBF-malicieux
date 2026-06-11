import numpy as np
from src.config import RESSOURCES_DIR, CONSTANT_HEAT_CAPACITY
from src.constants import (
    RHO_WATER,
    RHO_BULK,
    CP_DRY_SOIL,
    CP_WATER,
    CP_ICE,
    ACTIVE_LAYER_DEPTH,
)
from src.data_loader import load_and_grid_rzsm_data, upscale_grid


def compute_cp_from_rzsm(rzsm: np.ndarray) -> np.ndarray:
    """
    Calcule cp en kJ.kg-1.K-1 depuis l'humidité du sol RZSM.
    """
    is_ice = np.isclose(rzsm, 0.9)

    rzsm_clipped = np.clip(rzsm, 1e-6, 0.999)

    w = (RHO_WATER * rzsm_clipped) / (
        RHO_BULK * (1 - rzsm_clipped) + RHO_WATER * rzsm_clipped
    )

    cp = CP_DRY_SOIL + w * (CP_WATER - CP_DRY_SOIL)

    return np.where(is_ice, CP_ICE, cp)


def prepare_heat_capacity(grid: dict, use_variable_capacity: bool = True):
    """
    Renvoie C[i,j] en J.m-2.K-1.

    Sortie :
    C.shape = (N_THETA, N_PHI)
    """

    n_theta, n_phi = grid["lat_grid"].shape

    if not use_variable_capacity:
        print("Capacité thermique constante utilisée.")
        return np.full((n_theta, n_phi), CONSTANT_HEAT_CAPACITY)

    rzsm_path = RESSOURCES_DIR / "Cp_humidity" / "average_rzsm_tout.csv"

    rzsm_grid_low, lat_bins, lon_bins = load_and_grid_rzsm_data(rzsm_path)

    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    lon_centers = 0.5 * (lon_bins[:-1] + lon_bins[1:])

    rzsm_grid = upscale_grid(
        rzsm_grid_low,
        lat_centers,
        lon_centers,
        grid["lat"],
        grid["lon"]
    )

    rzsm_grid = np.nan_to_num(rzsm_grid, nan=0.2)

    cp_kj = compute_cp_from_rzsm(rzsm_grid)

    C = (cp_kj * 1000.0) * RHO_BULK * ACTIVE_LAYER_DEPTH

    print("Capacité thermique préparée :", C.shape)
    return C