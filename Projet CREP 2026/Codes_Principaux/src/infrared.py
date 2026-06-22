from src.constants import SIGMA

# Calcule l'infrarouge émis par la surface


def compute_surface_infrared(T):
    """
    Puissance infrarouge émise par la surface.

    Entrée :
    T en K.

    Sortie :
    W.m-2
    """
    return SIGMA * T**4 # loi de Stefan-Boltzmann