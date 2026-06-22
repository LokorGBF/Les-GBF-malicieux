"""
Modèle atmosphérique isotherme — version courte corrigée.

- atmosphère isotherme : T0 = 288 K ;
- g(z) = G0 * (R_EARTH / (R_EARTH + z))² ;
- pression intégrée couche par couche ;
- fractions molaires constantes par couche ;
- pression moyenne par couche par quadrature de Gauss-Legendre.

Fonction principale :
    gas_info_at_altitude(z, gas)

Elle demande une altitude z et un gaz, puis renvoie directement :
- la couche atmosphérique ;
- la fraction molaire xi ;
- la pression totale ;
- la pression partielle du gaz ;
- la concentration molaire du gaz ;
- la densité numérique du gaz ;
- la concentration massique du gaz.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

# =============================================================================
# Constantes
# =============================================================================

P0 = 101_325.0
T0 = 288.0
R = 8.314_462_618
K_B = 1.380_649e-23
G0 = 9.806_65
R_EARTH = 6_371_000.0

MAX_ALTITUDE = 10_000_000.0

MOLAR_MASSES = {
    "N2": 28.0134e-3,
    "O2": 31.9988e-3,
    "Ar": 39.948e-3,
    "CO2": 44.0095e-3,
    "Ne": 20.1797e-3,
    "He": 4.002_602e-3,
    "CH4": 16.0425e-3,
    "Kr": 83.798e-3,
    "H2": 2.015_88e-3,
    "N2O": 44.013e-3,
    "O3": 47.9982e-3,
    "H2O": 18.015_28e-3,
    "O": 15.999e-3,
    "H": 1.008e-3,
}

GAS_ORDER = ("N2", "O2", "CO2", "CH4", "N2O", "H2O", "O3", "Ar",
             "H", "H2", "He", "O", "Ne", "Kr")


# =============================================================================
# Couches atmosphériques
# =============================================================================

class Layer(NamedTuple):
    z_min: float
    z_max: float
    fractions: dict[str, float]
    label: str


TRACE_GASES = {
    "O2": 0.209_46,
    "Ar": 0.009_34,
    "CO2": 4.239e-4,
    "Ne": 1.818e-5,
    "He": 5.24e-6,
    "CH4": 1.942e-6,
    "Kr": 1.14e-6,
    "H2": 5.5e-7,
    "N2O": 3.38e-7,
}


def lower_layer(z_min: float, z_max: float, label: str, n2: float, h2o: float, o3: float) -> Layer:
    return Layer(z_min, z_max, {"N2": n2, **TRACE_GASES, "H2O": h2o, "O3": o3}, label)


LAYERS = (
    lower_layer(0, 2_000, "Troposphère basse", 0.770_748_67, 1.0e-2, 4.0e-8),
    lower_layer(2_000, 5_000, "Troposphère moyenne", 0.777_748_65, 3.0e-3, 6.0e-8),
    lower_layer(5_000, 12_000, "Troposphère haute", 0.780_248_63, 5.0e-4, 8.0e-8),
    lower_layer(12_000, 25_000, "Stratosphère basse", 0.780_739_71, 5.0e-6, 4.0e-6),
    lower_layer(25_000, 35_000, "Stratosphère moyenne", 0.780_735_71, 5.0e-6, 8.0e-6),
    lower_layer(35_000, 50_000, "Stratosphère haute", 0.780_739_71, 5.0e-6, 4.0e-6),
    lower_layer(50_000, 80_000, "Mésosphère", 0.780_748_71, 0.0, 0.0),
    Layer(
        80_000, 200_000,
        {"N2": 0.470, "O2": 0.130, "O": 0.390, "Ar": 0.005_5,
         "He": 0.003_5, "CO2": 1.0e-4, "Ne": 1.8e-5, "H": 1.0e-4},
        "Thermosphère basse",
    ),
    Layer(
        200_000, 500_000,
        {"O": 0.620, "He": 0.250, "N2": 0.080, "H": 0.040,
         "O2": 0.009, "Ar": 0.001},
        "Thermosphère haute",
    ),
    Layer(
        500_000, 10_000_000,
        {"He": 0.600, "H": 0.350, "O": 0.049, "N2": 0.001},
        "Exosphère",
    ),
)

LAYER_Z_MAX = np.array([layer.z_max for layer in LAYERS], dtype=float)


# =============================================================================
# Outils internes
# =============================================================================

def _as_valid_array(z: ArrayLike) -> NDArray[np.float64]:
    z = np.asarray(z, dtype=np.float64)
    if np.any(z < 0):
        raise ValueError("L'altitude z doit être ≥ 0 m.")
    if np.any(z > MAX_ALTITUDE):
        raise ValueError(f"L'altitude dépasse la limite du modèle ({MAX_ALTITUDE/1e3:.0f} km).")
    return z


def _geopotential_altitude(z: ArrayLike) -> NDArray[np.float64]:
    z = np.asarray(z, dtype=np.float64)
    return R_EARTH * z / (R_EARTH + z)


def _layer_indices(z: NDArray[np.float64]) -> NDArray[np.int_]:
    return np.searchsorted(LAYER_Z_MAX, z, side="right").clip(max=len(LAYERS) - 1)


def _ordered_gases(gases) -> tuple[str, ...]:
    gas_set = set(gases)
    ordered = [gas for gas in GAS_ORDER if gas in gas_set]
    ordered += sorted(gas_set - set(ordered))
    return tuple(ordered)


def available_gases() -> tuple[str, ...]:
    """Liste de toutes les espèces présentes dans le code."""
    return _ordered_gases(MOLAR_MASSES)


def normalize_gas_name(gas: str) -> str:
    """Accepte par exemple 'co2', 'CO2', 'he', 'He' et renvoie le nom canonique."""
    gas_clean = gas.strip()
    for known_gas in MOLAR_MASSES:
        if gas_clean.upper() == known_gas.upper():
            return known_gas
    raise ValueError(
        f"Gaz inconnu : {gas!r}. Gaz disponibles : {', '.join(available_gases())}"
    )


@lru_cache(maxsize=None)
def _layer_data(layer_idx: int) -> tuple[dict[str, float], float]:
    layer = LAYERS[layer_idx]

    missing = set(layer.fractions) - set(MOLAR_MASSES)
    if missing:
        raise ValueError(f"[{layer.label}] Masse molaire manquante pour : {', '.join(sorted(missing))}")

    total = sum(layer.fractions.values())
    if not np.isclose(total, 1.0, atol=1e-3):
        raise ValueError(f"[{layer.label}] Σ fractions = {total:.8f} — doit être proche de 1.")

    fractions = {gas: x / total for gas, x in layer.fractions.items()}
    mean_molar_mass = sum(fractions[gas] * MOLAR_MASSES[gas] for gas in fractions)
    return fractions, mean_molar_mass


def _state(z: ArrayLike) -> tuple[NDArray[object], NDArray[np.float64], NDArray[np.float64]]:
    z_arr = _as_valid_array(z)
    scalar_input = z_arr.ndim == 0
    z_arr = np.atleast_1d(z_arr)

    indices = _layer_indices(z_arr)
    fractions_arr = np.empty(z_arr.shape, dtype=object)
    mean_molar_mass_arr = np.empty(z_arr.shape, dtype=np.float64)

    for idx in np.unique(indices):
        fractions, mean_molar_mass = _layer_data(int(idx))
        mask = indices == idx
        fractions_arr[mask] = fractions
        mean_molar_mass_arr[mask] = mean_molar_mass

    z_geo = _geopotential_altitude(z_arr)
    exponent = np.empty_like(z_arr)
    exponent_start = 0.0

    for idx, layer in enumerate(LAYERS):
        _, mean_molar_mass = _layer_data(idx)
        z_min_geo = _geopotential_altitude(layer.z_min)
        mask = indices == idx

        if np.any(mask):
            exponent[mask] = (
                exponent_start
                - mean_molar_mass * G0 / (R * T0) * (z_geo[mask] - z_min_geo)
            )

        if idx < len(LAYERS) - 1:
            z_max_geo = _geopotential_altitude(layer.z_max)
            exponent_start -= mean_molar_mass * G0 / (R * T0) * (z_max_geo - z_min_geo)

    pressure = P0 * np.exp(exponent)

    if scalar_input:
        return fractions_arr[0], float(mean_molar_mass_arr[0]), float(pressure[0])
    return fractions_arr, mean_molar_mass_arr, pressure


def layer_index_at_altitude(z: float) -> int:
    """Indice de la couche contenant l'altitude z."""
    z_arr = _as_valid_array(float(z))
    return int(_layer_indices(np.atleast_1d(z_arr))[0])


# =============================================================================
# Fonctions de base
# =============================================================================

def gravity(z: ArrayLike) -> NDArray[np.float64]:
    z = _as_valid_array(z)
    return G0 * (R_EARTH / (R_EARTH + z)) ** 2


def pressure_air(z: ArrayLike) -> NDArray[np.float64]:
    return _state(z)[2]


def total_molar_concentration(z: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(pressure_air(z), dtype=np.float64) / (R * T0)


def total_number_density(z: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(pressure_air(z), dtype=np.float64) / (K_B * T0)


def air_mass_density(z: ArrayLike) -> NDArray[np.float64]:
    _, mean_molar_mass, pressure = _state(z)
    return np.asarray(pressure) * np.asarray(mean_molar_mass) / (R * T0)


# =============================================================================
# Fonctions réutilisables par espèce
# =============================================================================

def gas_info_at_altitude(z: float, gas: str) -> dict[str, float | str]:
    """
    Renvoie toutes les infos utiles pour un gaz à une altitude donnée.

    Exemple :
        info = gas_info_at_altitude(3500, "CO2")
        print(info["partial_pressure_Pa"])
    """
    gas = normalize_gas_name(gas)
    z = float(z)

    idx = layer_index_at_altitude(z)
    layer = LAYERS[idx]
    fractions, _ = _layer_data(idx)

    xi = float(fractions.get(gas, 0.0))
    P = float(pressure_air(z))
    C_total = P / (R * T0)

    return {
        "gas": gas,
        "altitude_m": z,
        "altitude_km": z / 1000,
        "layer_index": idx,
        "layer_label": layer.label,
        "layer_z_min_m": layer.z_min,
        "layer_z_max_m": layer.z_max,
        "molar_mass_kg_mol": MOLAR_MASSES[gas],
        "xi": xi,
        "gravity_m_s2": float(gravity(z)),
        "total_pressure_Pa": P,
        "partial_pressure_Pa": xi * P,
        "total_molar_concentration_mol_m3": C_total,
        "molar_concentration_mol_m3": xi * C_total,
        "number_density_molecules_m3": xi * P / (K_B * T0),
        "mass_concentration_kg_m3": xi * C_total * MOLAR_MASSES[gas],
    }


def all_gases_info_at_altitude(z: float) -> dict[str, dict[str, float | str]]:
    """Même calcul, mais pour toutes les espèces chimiques définies dans le code."""
    return {gas: gas_info_at_altitude(z, gas) for gas in available_gases()}


# Anciennes fonctions conservées pour compatibilité.
def species_values_at_altitude(z: ArrayLike, gases: tuple[str, ...] | None = None) -> dict[str, dict[str, NDArray[np.float64]]]:
    fractions_arr, _, pressure = _state(z)
    pressure = np.atleast_1d(np.asarray(pressure, dtype=np.float64))
    fractions_arr = np.atleast_1d(fractions_arr)
    C_total = pressure / (R * T0)

    if gases is None:
        gases = _ordered_gases(set().union(*(f.keys() for f in fractions_arr)))

    return {
        gas: {
            "xi": np.array([f.get(gas, 0.0) for f in fractions_arr], dtype=np.float64),
            "partial_pressure_Pa": np.array([f.get(gas, 0.0) * p for f, p in zip(fractions_arr, pressure)]),
            "molar_concentration_mol_m3": np.array([f.get(gas, 0.0) * c for f, c in zip(fractions_arr, C_total)]),
        }
        for gas in gases
    }


def gas_partial_pressures(z: ArrayLike) -> dict[str, NDArray[np.float64]]:
    values = species_values_at_altitude(z)
    return {gas: data["partial_pressure_Pa"] for gas, data in values.items()}


def gas_molar_concentrations(z: ArrayLike) -> dict[str, NDArray[np.float64]]:
    values = species_values_at_altitude(z)
    return {gas: data["molar_concentration_mol_m3"] for gas, data in values.items()}


def gas_number_densities(z: ArrayLike) -> dict[str, NDArray[np.float64]]:
    values = species_values_at_altitude(z)
    return {gas: data["molar_concentration_mol_m3"] * (R / K_B) for gas, data in values.items()}


def gas_mass_concentrations(z: ArrayLike) -> dict[str, NDArray[np.float64]]:
    values = species_values_at_altitude(z)
    return {gas: data["molar_concentration_mol_m3"] * MOLAR_MASSES[gas] for gas, data in values.items()}


# =============================================================================
# Interface simple : demande altitude + gaz
# =============================================================================

def print_gas_info(info: dict[str, float | str]) -> None:
    """Affiche proprement le résultat de gas_info_at_altitude()."""
    print("\n" + "═" * 68)
    print(f"Gaz demandé      : {info['gas']}")
    print(f"Altitude         : {info['altitude_km']:.3f} km")
    print(
        f"Couche           : {info['layer_label']} "
        f"({info['layer_z_min_m']/1000:g} – {info['layer_z_max_m']/1000:g} km)"
    )
    print("─" * 68)
    print(f"Masse molaire              = {info['molar_mass_kg_mol']:.6e} kg/mol")
    print(f"Fraction molaire xi        = {info['xi']:.6e}")
    print(f"Pesanteur g                = {info['gravity_m_s2']:.6e} m/s²")
    print(f"Pression totale P          = {info['total_pressure_Pa']:.6e} Pa")
    print(f"Pression partielle Pi      = {info['partial_pressure_Pa']:.6e} Pa")
    print(f"Concentration totale C     = {info['total_molar_concentration_mol_m3']:.6e} mol/m³")
    print(f"Concentration molaire Ci   = {info['molar_concentration_mol_m3']:.6e} mol/m³")
    print(f"Densité numérique ni       = {info['number_density_molecules_m3']:.6e} molécules/m³")
    print(f"Concentration massique rhoi= {info['mass_concentration_kg_m3']:.6e} kg/m³")
    print("═" * 68 + "\n")


def ask_gas_info() -> dict[str, float | str]:
    """Demande l'altitude et le gaz, puis renvoie les infos du gaz."""
    print("Gaz disponibles :", ", ".join(available_gases()))
    z = float(input("Altitude z en mètres : ").replace(",", "."))
    gas = input("Gaz recherché : ")
    info = gas_info_at_altitude(z, gas)
    print_gas_info(info)
    return info


if __name__ == "__main__":
    ask_gas_info()
