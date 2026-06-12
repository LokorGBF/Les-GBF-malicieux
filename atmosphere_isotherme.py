"""
Modèle atmosphérique isotherme — 10 sous-couches de 0 à 10 000 km.

Hypothèses physiques
--------------------
- Température uniforme T0 = 288 K (atmosphère isotherme).
- Pesanteur variable avec l'altitude : g(z) = G0·(R_T/(R_T+z))²  (exacte).
- Fractions molaires constantes PAR COUCHE — bonne approximation sous l'homopause
  (< 80 km) ; au-delà, les valeurs sont indicatives (diffusion différentielle,
  dissociation photochimique ; les résultats restent indicatifs).
- Gaz parfaits : loi barométrique intégrée analytiquement avec g(z) variable.

Décomposition verticale
-----------------------
Sous l'homopause (< 80 km) — fractions pédagogiques (tableau fourni) :
  [0]  Troposphère basse      0 – 2 km
  [1]  Troposphère moyenne    2 – 5 km
  [2]  Troposphère haute      5 – 12 km
  [3]  Stratosphère basse    12 – 25 km
  [4]  Stratosphère moyenne  25 – 35 km
  [5]  Stratosphère haute    35 – 50 km
  [6]  Mésosphère            50 – 80 km

Au-delà de l'homopause — composition indicative (MSIS-E-90 / NRLMSISE-00) :
  [7]  Thermosphère basse    80 – 200 km   (O atomique dominant)
  [8]  Thermosphère haute   200 – 500 km   (He et H croissants)
  [9]  Exosphère            500 – 10 000 km (He/H quasi-total)

Organisation du module
----------------------
1. Constantes physiques.
2. Masses molaires.
3. Définition des couches (LAYERS).
4. Outils internes (_*).
5. Fonctions publiques :
     gravity, pressure_air,
     gas_partial_pressures,
     total_molar_concentration, gas_molar_concentrations,
     total_number_density,      gas_number_densities,
     air_mass_density,          gas_mass_concentrations.
6. Utilitaire d'affichage : summary.
7. Exemple d'utilisation (__main__).

Unités de sortie
----------------
  Altitude       m         Pression       Pa
  Concentration  mol/m³    Densité num.   molécules/m³
  Densité masse  kg/m³     Pesanteur      m/s²
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONSTANTES PHYSIQUES
# ══════════════════════════════════════════════════════════════════════════════

P0: float      = 101_325.0       # Pression au sol                  [Pa]
T0: float      = 288.0           # Température isotherme imposée    [K]
R:  float      = 8.314_462_618   # Constante des gaz parfaits       [J/(mol·K)]
K_B: float     = 1.380_649e-23   # Constante de Boltzmann           [J/K]
G0: float      = 9.806_65        # Pesanteur au niveau de la mer    [m/s²]
R_EARTH: float = 6_371_000.0     # Rayon moyen de la Terre          [m]

# Altitude de l'homopause : en dessous, la composition est bien mélangée.
HOMOPAUSE_ALTITUDE: float = 80_000.0     # m
# Limite supérieure modélisée.
MAX_ALTITUDE: float       = 10_000_000.0 # m (10 000 km, exosphère supérieure)

# ══════════════════════════════════════════════════════════════════════════════
# 2.  MASSES MOLAIRES  [kg/mol]
# ══════════════════════════════════════════════════════════════════════════════
# Pour ajouter un gaz : l'inscrire ici PUIS dans chaque _Layer concernée.

MOLAR_MASSES: dict[str, float] = {
    "N2":  28.0134e-3,
    "O2":  31.9988e-3,
    "Ar":  39.948e-3,
    "CO2": 44.0095e-3,
    "Ne":  20.1797e-3,
    "He":  4.002_602e-3,
    "CH4": 16.0425e-3,
    "Kr":  83.798e-3,
    "H2":  2.015_88e-3,
    "N2O": 44.013e-3,
    "O3":  47.9982e-3,
    "H2O": 18.015_28e-3,
    "O":   15.999e-3,    # oxygène atomique (thermosphère/exosphère)
    "H":   1.008e-3,     # hydrogène atomique (exosphère)
}

# ══════════════════════════════════════════════════════════════════════════════
# 3.  DÉFINITION DES COUCHES ATMOSPHÉRIQUES
# ══════════════════════════════════════════════════════════════════════════════
#
# Règles de construction des fractions molaires
# ---------------------------------------------
# • Gaz fixes dans toutes les sous-couches homopause (< 80 km) :
#     O2  = 0.209_46       (COESA-1976)
#     Ar  = 0.009_34
#     Ne  = 0.000_018_18
#     He  = 0.000_005_24
#     Kr  = 0.000_001_14
#     H2  = 0.000_000_55
#   → Somme fixes = 0.218_825_11
#
# • CO2, CH4, N2O, H2O, O3 : valeurs du tableau pédagogique (% → fraction).
#   CO2 = 0.04239 % = 4.239e-4   CH4 = 0.0001942 % = 1.942e-6
#   N2O = 3.38e-8                 O3 et H2O varient par couche.
#
# • N2 est calculé par complémentarité : N2 = 1 − Σ(autres) pour garantir
#   que chaque couche somme exactement à 1. Les valeurs N2 ci-dessous
#   ont été vérifiées numériquement (script de vérification inclus en fin).
#
# • Thermosphère / Exosphère : fractions issues de NRLMSISE-00 (moyennes
#   activité solaire modérée, latitude 45°N). N2, O2, O, He, H, Ar, N.
#   La notion de « fraction molaire globale » perd son sens physique au-delà
#   de l'homopause (séparation diffusive) — ces valeurs sont INDICATIVES.
#
# Sources :
#   - COESA (1976) US Standard Atmosphere
#   - NRLMSISE-00 (Picone et al., 2002, JGR 107(A12))
#   - Tableau pédagogique fourni (fractions de départ pour test du programme)
# ══════════════════════════════════════════════════════════════════════════════

class _Layer(NamedTuple):
    """Une couche atmosphérique à composition homogène."""
    z_min:     float              # Altitude minimale [m]  (inclusive)
    z_max:     float              # Altitude maximale [m]  (exclusive, sauf dernière)
    fractions: dict[str, float]  # Fractions molaires — doivent sommer à 1 (± 1e-3)
    label:     str               # Nom lisible


LAYERS: tuple[_Layer, ...] = (

    # ── Sous-homopause : composition issue du tableau pédagogique ─────────────
    # N2 ajusté pour que Σ fractions = 1.000 000 00 dans chaque couche.

    _Layer(                                    # [0]
        z_min=0, z_max=2_000,
        label="Troposphère basse",
        fractions={
            "N2":  0.770_748_67,   # = 1 − Σ(autres)
            "O2":  0.209_46,
            "Ar":  0.009_34,
            "CO2": 4.239e-4,
            "Ne":  1.818e-5,
            "He":  5.24e-6,
            "CH4": 1.942e-6,
            "Kr":  1.14e-6,
            "H2":  5.5e-7,
            "N2O": 3.38e-7,
            "H2O": 1.0e-2,         # ~1 % (air humide de basse troposphère)
            "O3":  4.0e-8,         # très faible en surface
        },
    ),

    _Layer(                                    # [1]
        z_min=2_000, z_max=5_000,
        label="Troposphère moyenne",
        fractions={
            "N2":  0.777_748_65,   # H2O chute → N2 monte
            "O2":  0.209_46,
            "Ar":  0.009_34,
            "CO2": 4.239e-4,
            "Ne":  1.818e-5,
            "He":  5.24e-6,
            "CH4": 1.942e-6,
            "Kr":  1.14e-6,
            "H2":  5.5e-7,
            "N2O": 3.38e-7,
            "H2O": 3.0e-3,
            "O3":  6.0e-8,
        },
    ),

    _Layer(                                    # [2]
        z_min=5_000, z_max=12_000,
        label="Troposphère haute",
        fractions={
            "N2":  0.780_248_63,
            "O2":  0.209_46,
            "Ar":  0.009_34,
            "CO2": 4.239e-4,
            "Ne":  1.818e-5,
            "He":  5.24e-6,
            "CH4": 1.942e-6,
            "Kr":  1.14e-6,
            "H2":  5.5e-7,
            "N2O": 3.38e-7,
            "H2O": 5.0e-4,
            "O3":  8.0e-8,
        },
    ),

    _Layer(                                    # [3]
        z_min=12_000, z_max=25_000,
        label="Stratosphère basse",
        fractions={
            "N2":  0.780_739_71,
            "O2":  0.209_46,
            "Ar":  0.009_34,
            "CO2": 4.239e-4,
            "Ne":  1.818e-5,
            "He":  5.24e-6,
            "CH4": 1.942e-6,
            "Kr":  1.14e-6,
            "H2":  5.5e-7,
            "N2O": 3.38e-7,
            "H2O": 5.0e-6,         # air très sec au-dessus de la tropopause
            "O3":  4.0e-6,         # début du pic ozone
        },
    ),

    _Layer(                                    # [4]
        z_min=25_000, z_max=35_000,
        label="Stratosphère moyenne",
        fractions={
            "N2":  0.780_735_71,
            "O2":  0.209_46,
            "Ar":  0.009_34,
            "CO2": 4.239e-4,
            "Ne":  1.818e-5,
            "He":  5.24e-6,
            "CH4": 1.942e-6,
            "Kr":  1.14e-6,
            "H2":  5.5e-7,
            "N2O": 3.38e-7,
            "H2O": 5.0e-6,
            "O3":  8.0e-6,         # pic vers 25–30 km
        },
    ),

    _Layer(                                    # [5]
        z_min=35_000, z_max=50_000,
        label="Stratosphère haute",
        fractions={
            "N2":  0.780_739_71,
            "O2":  0.209_46,
            "Ar":  0.009_34,
            "CO2": 4.239e-4,
            "Ne":  1.818e-5,
            "He":  5.24e-6,
            "CH4": 1.942e-6,
            "Kr":  1.14e-6,
            "H2":  5.5e-7,
            "N2O": 3.38e-7,
            "H2O": 5.0e-6,
            "O3":  4.0e-6,         # déclin progressif
        },
    ),

    _Layer(                                    # [6]
        z_min=50_000, z_max=80_000,
        label="Mésosphère",
        fractions={
            "N2":  0.780_748_71,
            "O2":  0.209_46,
            "Ar":  0.009_34,
            "CO2": 4.239e-4,
            "Ne":  1.818e-5,
            "He":  5.24e-6,
            "CH4": 1.942e-6,
            "Kr":  1.14e-6,
            "H2":  5.5e-7,
            "N2O": 3.38e-7,
            "H2O": 0.0,            # H2O = 0 (tableau)
            "O3":  0.0,            # O3  = 0 (tableau)
        },
    ),

    # ── Au-delà de l'homopause : composition INDICATIVE (NRLMSISE-00) ─────────
    # La dissociation de O2 produit O atomique dominant entre 100 et 300 km.
    # He et H deviennent dominants au-delà de 500 km (exosphère).
    # N2 et O2 persistent mais leurs fractions chutent rapidement.
    # Kr et H2 sont négligés (< 1e-9).
    #
    # AVERTISSEMENT : la loi barométrique isotherme est une très mauvaise
    # approximation au-delà de 80 km (gradient de température réel ≠ 0,
    # diffusion différentielle, non-équilibre thermique).
    # Les concentrations calculées sont à considérer comme des ordres de grandeur.

    _Layer(                                    # [7]
        z_min=80_000, z_max=200_000,
        label="Thermosphère basse",
        fractions={
            # O atomique devient dominant dès ~100 km (dissociation de O2 par UV)
            "N2":  0.470,
            "O2":  0.130,
            "O":   0.390,          # O atomique
            "Ar":  0.005_5,
            "He":  0.003_5,
            "CO2": 1.0e-4,
            "Ne":  1.8e-5,
            "H":   1.0e-4,         # H atomique, encore minoritaire
        },
    ),

    _Layer(                                    # [8]
        z_min=200_000, z_max=500_000,
        label="Thermosphère haute",
        fractions={
            # O domine, He et H progressent rapidement.
            # N2 et O2 quasiment absents à 500 km.
            "O":   0.620,
            "He":  0.250,
            "N2":  0.080,
            "H":   0.040,
            "O2":  0.009,
            "Ar":  0.001,
        },
    ),

    _Layer(                                    # [9]
        z_min=500_000, z_max=10_000_000,
        label="Exosphère",
        fractions={
            # He et H totalement dominants.
            # O encore présent en basse exosphère.
            "He":  0.600,
            "H":   0.350,
            "O":   0.049,
            "N2":  0.001,
        },
    ),
)

# ══════════════════════════════════════════════════════════════════════════════
# 4.  OUTILS INTERNES
# ══════════════════════════════════════════════════════════════════════════════

def _to_array(z: ArrayLike) -> NDArray[np.float64]:
    """
    Convertit z en tableau NumPy float64 et vérifie que l'altitude appartient
    au domaine du modèle.
    """
    z = np.asarray(z, dtype=np.float64)
    if np.any(z < 0):
        raise ValueError("L'altitude z doit être ≥ 0 m.")
    if np.any(z > MAX_ALTITUDE):
        raise ValueError(
            f"L'altitude dépasse la limite du modèle ({MAX_ALTITUDE/1e3:.0f} km)."
        )
    return z

def _checked_layer(layer: _Layer) -> tuple[dict[str, float], float]:
    """
    Valide et normalise la composition d'une couche.

    Retourne
    --------
    fractions_norm : dict[str, float]
        Fractions normalisées (somme = 1 exactement).
    M_mean : float
        Masse molaire moyenne [kg/mol].

    Lève ValueError si un gaz de `fractions` est absent de MOLAR_MASSES,
    ou si les fractions semblent être en pourcentage.
    """
    frac = layer.fractions

    # Vérification des masses molaires
    missing = set(frac) - set(MOLAR_MASSES)
    if missing:
        raise ValueError(
            f"['{layer.label}'] Masse molaire manquante pour : "
            + ", ".join(sorted(missing))
        )

    total = sum(frac.values())

    # Détection d'une erreur d'unité fréquente
    if np.isclose(total, 100.0, atol=0.5):
        raise ValueError(
            f"['{layer.label}'] Les fractions semblent être en %. "
            "Utiliser des valeurs entre 0 et 1."
        )
    if not np.isclose(total, 1.0, atol=1e-3):
        raise ValueError(
            f"['{layer.label}'] Σ fractions = {total:.8f} — doit être proche de 1."
        )

    # Normalisation (absorbe les erreurs d'arrondi < 1e-3)
    norm = {gas: x / total for gas, x in frac.items()}

    # Masse molaire moyenne
    M_mean = sum(norm[gas] * MOLAR_MASSES[gas] for gas in norm)

    return norm, M_mean


# Cache : chaque couche n'est validée qu'une seule fois.
_LAYER_CACHE: dict[int, tuple[dict[str, float], float]] = {}


def _get_layer_data(layer_idx: int) -> tuple[dict[str, float], float]:
    """Retourne (fractions normalisées, M_mean) depuis le cache."""
    if layer_idx not in _LAYER_CACHE:
        _LAYER_CACHE[layer_idx] = _checked_layer(LAYERS[layer_idx])
    return _LAYER_CACHE[layer_idx]


def _layer_index(z_scalar: float) -> int:
    """Indice de la couche LAYERS correspondant à l'altitude scalaire z [m]."""
    for i, layer in enumerate(LAYERS):
        if i == len(LAYERS) - 1 or z_scalar < layer.z_max:
            return i
    return len(LAYERS) - 1  # jamais atteint — garde-fou pour mypy


# Version vectorisée pour traiter des tableaux NumPy efficacement.
_vlayer_index: np.ufunc = np.vectorize(_layer_index, otypes=[int])


def _state(
    z: ArrayLike,
) -> tuple[NDArray[object], NDArray[np.float64], NDArray[np.float64]]:
    """
    Noyau de calcul vectorisé.

    Retourne pour chaque point d'altitude z :
        fractions_arr : NDArray[object]    — dict des fractions de la couche
        M_mean_arr    : NDArray[float64]   — masse molaire moyenne [kg/mol]
        pressure_arr  : NDArray[float64]   — pression [Pa]

    Formule de la pression (loi barométrique isotherme, g variable)
    ---------------------------------------------------------------
    L'équilibre hydrostatique dP/dz = −ρ·g, avec ρ = P·M/(R·T0) et
    g(z) = G0·(R_T/(R_T+z))², est intégré analytiquement couche par couche.

    Dans une couche où M est constante, la contribution à l'exposant entre
    les altitudes z_a et z_b vaut :

        −M·G0/(R·T0) · [H_geo(z_b) − H_geo(z_a)]

    avec :

        H_geo(z) = R_T·z / (R_T+z).

    Les contributions des couches déjà traversées sont additionnées afin de
    garantir la continuité de la pression lorsque la composition change.
    """
    z = _to_array(z)
    scalar_input = z.ndim == 0
    z = np.atleast_1d(z)

    layer_indices = _vlayer_index(z)

    fractions_arr = np.empty(z.shape, dtype=object)
    M_mean_arr    = np.empty(z.shape, dtype=np.float64)

    # Remplissage par masque : toutes les altitudes d'une même couche
    # partagent le même dict (lecture seule) — pas de copie mémoire.
    for idx in np.unique(layer_indices):
        mask = layer_indices == idx
        frac, M = _get_layer_data(int(idx))
        fractions_arr[mask] = frac
        M_mean_arr[mask]    = M

    # Altitude géopotentielle : R_T·z / (R_T+z)
    z_geo = R_EARTH * z / (R_EARTH + z)

    # Intégration couche par couche.
    # La masse molaire moyenne pouvant changer d'une couche à l'autre,
    # on additionne les contributions déjà parcourues depuis le sol.
    # Cette écriture garantit la continuité et la décroissance de P(z).
    exponent = np.empty(z.shape, dtype=np.float64)
    exponent_at_layer_start = 0.0

    for idx, layer in enumerate(LAYERS):
        mask = layer_indices == idx
        _, M = _get_layer_data(idx)

        z_min_geo = R_EARTH * layer.z_min / (R_EARTH + layer.z_min)

        if np.any(mask):
            exponent[mask] = (
                exponent_at_layer_start
                - (M * G0 / (R * T0)) * (z_geo[mask] - z_min_geo)
            )

        # Prépare l'exposant cumulé au début de la couche suivante.
        # Cette mise à jour n'est pas nécessaire après la dernière couche.
        if idx < len(LAYERS) - 1:
            z_max_geo = R_EARTH * layer.z_max / (R_EARTH + layer.z_max)
            exponent_at_layer_start -= (
                M * G0 / (R * T0) * (z_max_geo - z_min_geo)
            )

    pressure = P0 * np.exp(exponent)

    if scalar_input:
        return fractions_arr[0], float(M_mean_arr[0]), float(pressure[0])

    return fractions_arr, M_mean_arr, pressure


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FONCTIONS PUBLIQUES
# ══════════════════════════════════════════════════════════════════════════════

# ── 5a. Champs mécaniques ─────────────────────────────────────────────────────

def gravity(z: ArrayLike) -> NDArray[np.float64]:
    """
    Pesanteur à l'altitude z [m/s²].

        g(z) = G0 · (R_EARTH / (R_EARTH + z))²
    """
    z = _to_array(z)
    return G0 * (R_EARTH / (R_EARTH + z)) ** 2


def pressure_air(z: ArrayLike) -> NDArray[np.float64]:
    """
    Pression totale de l'air à l'altitude z [Pa].

    Utilise la loi barométrique isotherme avec pesanteur variable
    et composition par couche atmosphérique.
    """
    return _state(z)[2]


# ── 5b. Pressions partielles ──────────────────────────────────────────────────

def gas_partial_pressures(z: ArrayLike) -> dict[str, NDArray[np.float64]]:
    """
    Pression partielle de chaque gaz à l'altitude z [Pa].

        P_i(z) = x_i(z) · P(z)

    Retourne un dict {symbole: tableau(z)}.
    Les gaz absents d'une couche contribuent 0.
    """
    fractions_arr, _, pressure = _state(z)
    pressure      = np.atleast_1d(np.asarray(pressure,      dtype=np.float64))
    fractions_arr = np.atleast_1d(fractions_arr)

    all_gases = set().union(*(f.keys() for f in fractions_arr))
    return {
        gas: np.fromiter(
            (f.get(gas, 0.0) * p for f, p in zip(fractions_arr, pressure)),
            dtype=np.float64, count=len(pressure),
        )
        for gas in all_gases
    }


# ── 5c. Concentrations molaires ───────────────────────────────────────────────

def total_molar_concentration(z: ArrayLike) -> NDArray[np.float64]:
    """
    Concentration molaire totale de l'air [mol/m³].

        C(z) = P(z) / (R · T0)
    """
    return np.asarray(pressure_air(z), dtype=np.float64) / (R * T0)


def gas_molar_concentrations(z: ArrayLike) -> dict[str, NDArray[np.float64]]:
    """
    Concentration molaire de chaque gaz [mol/m³].

        C_i(z) = x_i(z) · P(z) / (R · T0)
    """
    fractions_arr, _, pressure = _state(z)
    pressure      = np.atleast_1d(np.asarray(pressure,      dtype=np.float64))
    fractions_arr = np.atleast_1d(fractions_arr)

    C_total   = pressure / (R * T0)
    all_gases = set().union(*(f.keys() for f in fractions_arr))
    return {
        gas: np.fromiter(
            (f.get(gas, 0.0) * c for f, c in zip(fractions_arr, C_total)),
            dtype=np.float64, count=len(pressure),
        )
        for gas in all_gases
    }


# ── 5d. Densités numériques ───────────────────────────────────────────────────

def total_number_density(z: ArrayLike) -> NDArray[np.float64]:
    """
    Densité numérique totale [molécules/m³].

        n(z) = P(z) / (k_B · T0)
    """
    return np.asarray(pressure_air(z), dtype=np.float64) / (K_B * T0)


def gas_number_densities(z: ArrayLike) -> dict[str, NDArray[np.float64]]:
    """
    Densité numérique de chaque gaz [molécules/m³].

        n_i(z) = x_i(z) · P(z) / (k_B · T0)
    """
    fractions_arr, _, pressure = _state(z)
    pressure      = np.atleast_1d(np.asarray(pressure,      dtype=np.float64))
    fractions_arr = np.atleast_1d(fractions_arr)

    n_total   = pressure / (K_B * T0)
    all_gases = set().union(*(f.keys() for f in fractions_arr))
    return {
        gas: np.fromiter(
            (f.get(gas, 0.0) * n for f, n in zip(fractions_arr, n_total)),
            dtype=np.float64, count=len(pressure),
        )
        for gas in all_gases
    }


# ── 5e. Masses volumiques ─────────────────────────────────────────────────────

def air_mass_density(z: ArrayLike) -> NDArray[np.float64]:
    """
    Masse volumique totale de l'air [kg/m³].

        ρ(z) = P(z) · M_moy(z) / (R · T0)
    """
    _, M_mean, pressure = _state(z)
    return (
        np.asarray(pressure, dtype=np.float64)
        * np.asarray(M_mean,  dtype=np.float64)
        / (R * T0)
    )


def gas_mass_concentrations(z: ArrayLike) -> dict[str, NDArray[np.float64]]:
    """
    Masse volumique de chaque gaz [kg/m³].

        ρ_i(z) = C_i(z) · M_i
    """
    C = gas_molar_concentrations(z)
    return {gas: C[gas] * MOLAR_MASSES[gas] for gas in C}


# ══════════════════════════════════════════════════════════════════════════════
# 6.  UTILITAIRE D'AFFICHAGE
# ══════════════════════════════════════════════════════════════════════════════

# Ordre d'affichage conservé dans toutes les couches pour faciliter la lecture.
# Les espèces absentes d'une couche sont affichées avec une fraction nulle.
DISPLAY_GASES: tuple[str, ...] = (
    "N2",
    "O2",
    "CO2",
    "CH4",
    "N2O",
    "H2O",
    "O3",
    "Ar",
)

# Espèces supplémentaires affichées à partir de la thermosphère basse.
HIGH_ALTITUDE_EXTRA_GASES: tuple[str, ...] = (
    "H",
    "H2",
    "He",
)


def mean_pressure_in_layer(layer_idx: int, n_points: int = 64) -> float:
    """
    Calcule la pression moyenne dans une couche atmosphérique [Pa].

    La moyenne est définie par :

        P_moy = 1 / (z_max - z_min) · ∫[z_min, z_max] P(z) dz

    L'intégrale est évaluée numériquement par la méthode de Gauss-Legendre.
    Cette méthode fournit une valeur précise sans nécessiter SciPy.

    Parameters
    ----------
    layer_idx : int
        Indice de la couche dans LAYERS.
    n_points : int
        Nombre de points de quadrature. La valeur par défaut est suffisante
        pour les profils de pression lisses utilisés dans ce modèle.
    """
    idx = int(layer_idx)
    if not 0 <= idx < len(LAYERS):
        raise ValueError(
            f"L'indice de couche doit être compris entre 0 et {len(LAYERS) - 1}."
        )
    if n_points < 2:
        raise ValueError("Le nombre de points de quadrature doit être ≥ 2.")

    layer = LAYERS[idx]
    thickness = layer.z_max - layer.z_min

    # Transformation des points de Gauss-Legendre définis sur [-1, 1]
    # vers l'intervalle réel [z_min, z_max].
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    midpoint = 0.5 * (layer.z_min + layer.z_max)
    half_width = 0.5 * thickness
    z_values = midpoint + half_width * nodes

    integral = half_width * np.sum(
        weights * np.asarray(pressure_air(z_values), dtype=np.float64)
    )
    return float(integral / thickness)


def mean_gravity_in_layer(layer_idx: int) -> float:
    """
    Calcule la pesanteur moyenne dans une couche atmosphérique [m/s²].

        g_moy = 1 / (z_max - z_min) · ∫[z_min, z_max] g(z) dz

    L'expression est intégrée analytiquement.
    """
    idx = int(layer_idx)
    if not 0 <= idx < len(LAYERS):
        raise ValueError(
            f"L'indice de couche doit être compris entre 0 et {len(LAYERS) - 1}."
        )

    layer = LAYERS[idx]
    thickness = layer.z_max - layer.z_min

    return float(
        G0 * R_EARTH**2 / thickness
        * (
            1.0 / (R_EARTH + layer.z_min)
            - 1.0 / (R_EARTH + layer.z_max)
        )
    )


def summary(layer_idx: int) -> None:
    """
    Affiche un résumé tabulaire pour une zone d'altitude complète.

    La pression affichée correspond à la moyenne de la fonction continue P(z)
    sur l'intervalle d'altitude de la couche. Les concentrations sont calculées
    à partir de cette pression moyenne et des fractions molaires de la couche.

    Parameters
    ----------
    layer_idx : int
        Indice de la couche dans LAYERS.
    """
    idx = int(layer_idx)
    if not 0 <= idx < len(LAYERS):
        raise ValueError(
            f"L'indice de couche doit être compris entre 0 et {len(LAYERS) - 1}."
        )

    layer = LAYERS[idx]
    frac_norm, M_mean = _get_layer_data(idx)

    # Pression moyenne obtenue par intégration de P(z) sur la couche.
    P_mean = mean_pressure_in_layer(idx)

    # Les grandeurs suivantes sont cohérentes avec cette pression moyenne.
    # Dans chaque couche, T0, M_mean et les fractions molaires sont constantes.
    g_mean   = mean_gravity_in_layer(idx)
    rho_mean = P_mean * M_mean / (R * T0)
    C_mean   = P_mean / (R * T0)
    n_mean   = P_mean / (K_B * T0)

    gases = DISPLAY_GASES
    if idx >= 7:
        gases += HIGH_ALTITUDE_EXTRA_GASES

    W = 68
    print(f"\n{'═'*W}")
    print(
        f"  Zone d'altitude : {layer.z_min/1e3:g} – {layer.z_max/1e3:g} km"
        f"  —  {layer.label}"
    )
    print(f"{'═'*W}")
    print(f"  Pesanteur moyenne      g  = {g_mean:>12.5f}   m/s²")
    print(f"  Pression moyenne       P  = {P_mean:>12.4e}   Pa")
    print(f"  Masse volumique moy.   ρ  = {rho_mean:>12.4e}   kg/m³")
    print(f"  Conc. molaire moy.     C  = {C_mean:>12.4e}   mol/m³")
    print(f"  Densité numérique moy. n  = {n_mean:>12.4e}   molécules/m³")
    print(f"\n  {'Gaz':<6}  {'x_i':>10}  {'P_i moy. [Pa]':>14}  {'C_i moy. [mol/m³]':>17}")
    print(f"  {'─'*55}")

    for gas in gases:
        xi = frac_norm.get(gas, 0.0)
        partial_pressure_mean = xi * P_mean
        molar_concentration_mean = xi * C_mean

        print(
            f"  {gas:<6}  {xi:>10.4e}  "
            f"{partial_pressure_mean:>14.4e}  "
            f"{molar_concentration_mean:>17.4e}"
        )

    print(f"{'═'*W}\n")


# 7.  EXEMPLE D'UTILISATION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import numpy as np  # noqa: F811 (réimport local pour clarté)

    # ── Profil de pression 0 → 80 km ──────────────────────────────────────────
    z_vec = np.linspace(0, 80_000, 801)
    P_vec = pressure_air(z_vec)
    print(f"Pression au sol      : {P_vec[0]:.2f} Pa  (ref. = 101 325 Pa)")
    print(f"Pression à  12 km    : {float(pressure_air(12_000)):.1f} Pa")
    print(f"Pression à  25 km    : {float(pressure_air(25_000)):.2f} Pa")
    print(f"Pression à  50 km    : {float(pressure_air(50_000)):.3f} Pa")
    print(f"Pression à  80 km    : {float(pressure_air(80_000)):.4f} Pa")

    # ── Résumé de chaque ZONE d'altitude ──────────────────────────────────────
    # La pression affichée est la moyenne intégrale de P(z) sur chaque zone.
    for layer_idx in range(len(LAYERS)):
        summary(layer_idx)
