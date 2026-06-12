import numpy as np


def compute_convection_flux(T_surface, T_air=None, h: float = 10.0):
    """
    Flux de convection.

    Version de départ :
    si T_air n'est pas fourni, on renvoie 0.

    Plus tard :
    P_conv = h * (T_surface - T_air)
    """

    if T_air is None:
        return np.zeros_like(T_surface)

    return h * (T_surface - T_air)