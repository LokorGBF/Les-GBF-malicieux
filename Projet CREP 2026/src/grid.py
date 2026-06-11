import numpy as np
from src.constants import EARTH_RADIUS

# Crée le maillage de la Terre


def create_spherical_grid(n_theta: int, n_phi: int) -> dict:
    """
    Crée une grille sphérique régulière en latitude-longitude.

    theta : colatitude en radians, avec theta = 0 au pôle Nord.
    phi   : longitude en radians.

    Retourne aussi lat/lon en degrés, plus faciles à utiliser avec les données.
    """

    # On définit les bords des cellules en latitude et longitude
    lat_edges_deg = np.linspace(-90, 90, n_theta + 1)
    lon_edges_deg = np.linspace(-180, 180, n_phi + 1)

    # Centres des cellules
    lat_deg = 0.5 * (lat_edges_deg[:-1] + lat_edges_deg[1:])
    lon_deg = 0.5 * (lon_edges_deg[:-1] + lon_edges_deg[1:])

    lat_grid, lon_grid = np.meshgrid(lat_deg, lon_deg, indexing="ij")

    # Passage aux coordonnées sphériques
    theta_grid = np.radians(90.0 - lat_grid)
    phi_grid = np.radians(lon_grid)


    return {
        "lat": lat_deg,
        "lon": lon_deg,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
        "theta_grid": theta_grid,
        "phi_grid": phi_grid,
    }


def print_grid_info(grid: dict) -> None:
    print("lat_grid :", grid["lat_grid"].shape)
    print("lon_grid :", grid["lon_grid"].shape)
    print("theta_grid :", grid["theta_grid"].shape)
    print("phi_grid :", grid["phi_grid"].shape)