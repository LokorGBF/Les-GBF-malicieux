import numpy as np
from src.config import RESSOURCES_DIR
from src.constants import (
    DELTA_HVAP,
    RHO_LIQUID_WATER,
    SECONDS_PER_YEAR,
    EVAPORATION_M_PER_YEAR,
)
from src.data_loader import create_continent_finder

# Calcule et renvoie la perte d'énergie par évaporation


def evaporation_flux_from_zone(zone: str) -> float:
    """
    Renvoie le flux latent moyen en W.m-2 pour une zone.
    """
    evap_m_per_year = EVAPORATION_M_PER_YEAR.get(
        zone,
        EVAPORATION_M_PER_YEAR["Océan"]
    )

    evap_m_per_sec = evap_m_per_year / SECONDS_PER_YEAR

    return DELTA_HVAP * RHO_LIQUID_WATER * evap_m_per_sec


def prepare_evaporation_base(grid: dict, use_evaporation: bool = True):
    """
    Prépare une matrice q_base[i,j] en W.m-2.

    Cette matrice est ensuite utilisée dans le bilan d'énergie.
    """

    n_theta, n_phi = grid["lat_grid"].shape

    if not use_evaporation:
        print("Évaporation désactivée.")
        return np.zeros((n_theta, n_phi))

    shapefile_path = RESSOURCES_DIR / "map" / "ne_110m_admin_0_countries.shp"
    continent_finder = create_continent_finder(shapefile_path)

    q_base = np.zeros((n_theta, n_phi))

    for i in range(n_theta):
        for j in range(n_phi):
            lat = grid["lat_grid"][i, j]
            lon = grid["lon_grid"][i, j]

            zone = continent_finder(lat, lon)

            if lat > 75:
                q_base[i, j] = 0.0
            else:
                q_base[i, j] = evaporation_flux_from_zone(zone)

    print("Évaporation préparée :", q_base.shape)
    return q_base


# Ancien code : l’évaporation avait un signe jour/nuit qui pouvait donner une contribution la nuit

def compute_evaporation_flux(q_base, daylight_mask):
    """
    Version simple :
    l'évaporation est une perte positive surtout lorsque la surface reçoit du Soleil.

    Sortie :
    P_evaporation en W.m-2.
    """

    return q_base * daylight_mask