import numpy as np

from src.time_solar import get_day_and_hour_utc, cos_incidence_grid
from src.solar_flux import compute_solar_absorbed
from src.infrared import compute_surface_infrared
from src.greenhouse import compute_greenhouse_flux
from src.evaporation import compute_evaporation_flux
from src.convection import compute_convection_flux


def compute_energy_balance(T, t_sec, grid, params, options, return_details: bool = False):
    """
    Calcule P_net[i,j] en W.m-2 pour toute la grille.

    Si return_details=True:
    retourne (P_net, details)
    Sinon:
    retourne seulement P_net (plus rapide, moins d'allocations).
    """
    P_net = np.zeros_like(T, dtype=float)
    details = {} if return_details else None

    day_index, hour_utc = get_day_and_hour_utc(t_sec)
    day_number = day_index + 1

    cos_i = cos_incidence_grid(
        grid["lat_grid"],
        grid["lon_grid"],
        day_number,
        hour_utc
    )
    daylight_mask = (cos_i > 0).astype(float)

    surface_albedo = params["surface_albedo_daily"][day_index]
    cloud_albedo = params["cloud_albedo_daily"][day_index]

    if options["solar"]:
        P_solar = compute_solar_absorbed(cos_i, surface_albedo, cloud_albedo)
        P_net += P_solar
        if return_details:
            details["solar"] = P_solar

    if options["greenhouse"]:
        P_greenhouse = compute_greenhouse_flux(
            T,
            use_co2=options.get("co2", False)
        )
        P_net += P_greenhouse
        if return_details:
            details["greenhouse"] = P_greenhouse

    if options["surface_infrared"]:
        P_ir = compute_surface_infrared(T)
        P_net -= P_ir
        if return_details:
            details["surface_infrared"] = P_ir

    if options["evaporation"]:
        P_evap = compute_evaporation_flux(params["evaporation_base"], daylight_mask)
        P_net -= P_evap
        if return_details:
            details["evaporation"] = P_evap

    if options["convection"]:
        P_conv = compute_convection_flux(T)
        P_net -= P_conv
        if return_details:
            details["convection"] = P_conv

    if return_details:
        details["P_net"] = P_net
        return P_net, details

    return P_net