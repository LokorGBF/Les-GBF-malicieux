import numpy as np
from math import pi


def get_day_and_hour_utc(t_sec: float) -> tuple[int, float]:
    """
    Convertit un temps en secondes en jour de l'année et heure UTC.

    Retour :
    - day_index : entre 0 et 364
    - hour_utc : entre 0 et 24
    """
    day_index = int(t_sec // 86400) % 365
    hour_utc = (t_sec / 3600.0) % 24.0
    return day_index, hour_utc


def solar_declination(day_number: int) -> float:
    """
    Déclinaison solaire en radians.

    day_number doit être entre 1 et 365.
    """
    return np.radians(23.44) * np.sin(2 * pi * (284 + day_number) / 365)


def solar_hour_grid(hour_utc: float, lon_grid_deg):
    """
    Calcule l'heure solaire locale pour chaque longitude.
    """
    return (hour_utc + lon_grid_deg / 15.0) % 24.0


def cos_incidence_grid(lat_grid_deg, lon_grid_deg, day_number: int, hour_utc: float):
    """
    Calcule cos(angle d'incidence solaire) pour toute la grille.

    Sortie :
    matrice de taille (N_THETA, N_PHI), valeurs entre 0 et 1.
    """
    lat_rad = np.radians(lat_grid_deg)
    delta = solar_declination(day_number)

    hour_local = solar_hour_grid(hour_utc, lon_grid_deg)
    H = np.radians(15.0 * (hour_local - 12.0))

    cos_i = (
        np.sin(lat_rad) * np.sin(delta)
        + np.cos(lat_rad) * np.cos(delta) * np.cos(H)
    )

    return np.maximum(cos_i, 0.0)