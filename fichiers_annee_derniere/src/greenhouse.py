import numpy as np
from src.constants import SIGMA, T_ATM

# Calcule l'effet de serre (simplifié) car température atmosphère constante


def compute_greenhouse_flux(T, use_co2: bool = False, co2_ppm: float = 420.0):
    """
    Flux infrarouge descendant de l'atmosphère vers la surface.

    Version de base :
    atmosphère à température radiative constante T_ATM.

    Sortie :
    matrice W.m-2 de même taille que T.
    """

    base_flux = SIGMA * T_ATM**4
    P_greenhouse = np.full_like(T, base_flux)

    if use_co2:
        # Placeholder pour l'ajout de CO2
        pass

    return P_greenhouse