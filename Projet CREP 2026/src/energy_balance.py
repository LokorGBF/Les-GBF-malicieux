import numpy as np

from src.time_solar import get_day_and_hour_utc, cos_incidence_grid
from src.solar_flux import compute_solar_absorbed
from src.infrared import compute_surface_infrared
from src.greenhouse import compute_greenhouse_flux
from src.evaporation import compute_evaporation_flux
from src.convection import compute_convection_flux

# Calcule le bilan total d'énergie


def compute_energy_balance(T, t_sec, grid, params, options):
    """
    Calcule P_net[i,j] en W.m-2 pour toute la grille.

    Sortie :
    - P_net : matrice W.m-2
    - details : dictionnaire avec chaque flux séparé
    """

    P_net = np.zeros_like(T)
    details = {}

    day_index, hour_utc = get_day_and_hour_utc(t_sec)
    day_number = day_index + 1

    cos_i = cos_incidence_grid(
        grid["lat_grid"],
        grid["lon_grid"],
        day_number,
        hour_utc
    )

    daylight_mask = (cos_i > 0).astype(float)

    # Albédo du sol et des nuages
    surface_albedo = params["surface_albedo_daily"][day_index]
    cloud_albedo = params["cloud_albedo_daily"][day_index]

    # 1. Soleil
    if options["solar"]:
        P_solar = compute_solar_absorbed(
            cos_i,
            surface_albedo,
            cloud_albedo
        )
        P_net += P_solar
        details["solar"] = P_solar
    else:
        details["solar"] = np.zeros_like(T)

    # 2. Effet de serre / IR atmosphère
    if options["greenhouse"]:
        P_greenhouse = compute_greenhouse_flux(
            T,
            use_co2=options.get("co2", False)
        )
        P_net += P_greenhouse
        details["greenhouse"] = P_greenhouse
    else:
        details["greenhouse"] = np.zeros_like(T)

    # 3. IR surface
    if options["surface_infrared"]:
        P_ir = compute_surface_infrared(T)
        P_net -= P_ir
        details["surface_infrared"] = P_ir
    else:
        details["surface_infrared"] = np.zeros_like(T)

    # 4. Évaporation
    if options["evaporation"]:
        P_evap = compute_evaporation_flux(
            params["evaporation_base"],
            daylight_mask
        )
        P_net -= P_evap
        details["evaporation"] = P_evap
    else:
        details["evaporation"] = np.zeros_like(T)

    # 5. Convection
    if options["convection"]:
        P_conv = compute_convection_flux(T)
        P_net -= P_conv
        details["convection"] = P_conv
    else:
        details["convection"] = np.zeros_like(T)

    details["P_net"] = P_net

    return P_net, details