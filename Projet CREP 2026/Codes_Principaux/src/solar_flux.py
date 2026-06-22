import numpy as np
from src.constants import SOLAR_CONSTANT

# Calcule la puissance solaire absorbée par la surface


def compute_solar_absorbed(cos_incidence, surface_albedo, cloud_albedo):
    """
    Calcule la puissance solaire absorbée par la surface.

    Sortie :
    P_solar_abs.shape = (N_THETA, N_PHI)
    Unité : W.m-2
    """

    return SOLAR_CONSTANT * cos_incidence * (1 - cloud_albedo) * (1 - surface_albedo)