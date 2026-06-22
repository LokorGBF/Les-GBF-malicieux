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

from __future__ import annotations                      # Permet d'utiliser des annotations de types plus modernes.

from functools import lru_cache                         # Sert à mémoriser certains calculs répétitifs.
from typing import NamedTuple                           # Sert à créer des objets simples avec des champs nommés.

import numpy as np                                      # Bibliothèque pour les calculs numériques.
from numpy.typing import ArrayLike, NDArray             # Types utilisés pour clarifier les entrées/sorties numpy.


# =============================================================================
# Constantes
# =============================================================================
# Ces valeurs servent de base physique au modèle atmosphérique.

P0 = 101_325.0                                          # Pression au niveau de la mer, en pascals.
T0 = 288.0                                              # Température supposée constante : modèle isotherme.
R = 8.314_462_618                                      # Constante des gaz parfaits, en J·mol⁻¹·K⁻¹.
K_B = 1.380_649e-23                                    # Constante de Boltzmann, en J·K⁻¹.
G0 = 9.806_65                                          # Pesanteur moyenne au niveau de la mer, en m·s⁻².
R_EARTH = 6_371_000.0                                  # Rayon moyen de la Terre, en mètres.

MAX_ALTITUDE = 10_000_000.0                            # Limite maximale du modèle : 10 000 km.


# Masses molaires des espèces chimiques, en kg/mol.
MOLAR_MASSES = {
    "N2": 28.0134e-3,                                  # Diazote.
    "O2": 31.9988e-3,                                  # Dioxygène.
    "Ar": 39.948e-3,                                   # Argon.
    "CO2": 44.0095e-3,                                 # Dioxyde de carbone.
    "Ne": 20.1797e-3,                                  # Néon.
    "He": 4.002_602e-3,                                # Hélium.
    "CH4": 16.0425e-3,                                 # Méthane.
    "Kr": 83.798e-3,                                   # Krypton.
    "H2": 2.015_88e-3,                                 # Dihydrogène.
    "N2O": 44.013e-3,                                  # Protoxyde d'azote.
    "O3": 47.9982e-3,                                  # Ozone.
    "H2O": 18.015_28e-3,                               # Vapeur d'eau.
    "O": 15.999e-3,                                    # Oxygène atomique.
    "H": 1.008e-3,                                     # Hydrogène atomique.
}


# Ordre d'affichage préféré des gaz.
GAS_ORDER = (
    "N2", "O2", "CO2", "CH4", "N2O", "H2O", "O3", "Ar",
    "H", "H2", "He", "O", "Ne", "Kr"
)


# =============================================================================
# Couches atmosphériques
# =============================================================================
# Le modèle découpe l'atmosphère en couches.
# Chaque couche possède :
# - une altitude minimale ;
# - une altitude maximale ;
# - des fractions molaires ;
# - un nom.

class Layer(NamedTuple):
    z_min: float                                        # Altitude minimale de la couche, en mètres.
    z_max: float                                        # Altitude maximale de la couche, en mètres.
    fractions: dict[str, float]                         # Fractions molaires des gaz dans la couche.
    label: str                                          # Nom de la couche atmosphérique.


# Gaz dont la fraction molaire reste presque constante dans les basses couches.
TRACE_GASES = {
    "O2": 0.209_46,                                     # Fraction molaire du dioxygène.
    "Ar": 0.009_34,                                     # Fraction molaire de l'argon.
    "CO2": 4.239e-4,                                   # Fraction molaire du CO2.
    "Ne": 1.818e-5,                                    # Fraction molaire du néon.
    "He": 5.24e-6,                                     # Fraction molaire de l'hélium.
    "CH4": 1.942e-6,                                   # Fraction molaire du méthane.
    "Kr": 1.14e-6,                                     # Fraction molaire du krypton.
    "H2": 5.5e-7,                                      # Fraction molaire du dihydrogène.
    "N2O": 3.38e-7,                                    # Fraction molaire du protoxyde d'azote.
}


def lower_layer(z_min: float, z_max: float, label: str, n2: float, h2o: float, o3: float) -> Layer:
    return Layer(
        z_min,                                          # Début de la couche.
        z_max,                                          # Fin de la couche.
        {"N2": n2, **TRACE_GASES, "H2O": h2o, "O3": o3}, # Composition de la couche.
        label,                                          # Nom de la couche.
    )


# Liste complète des couches utilisées par le modèle.
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


# Tableau contenant les altitudes maximales des couches.
# Il permet de retrouver rapidement dans quelle couche se trouve une altitude donnée.
LAYER_Z_MAX = np.array([layer.z_max for layer in LAYERS], dtype=float)


# =============================================================================
# Outils internes
# =============================================================================
# Ces fonctions sont utilisées par le modèle mais ne sont pas forcément appelées
# directement par l'utilisateur.


def _as_valid_array(z: ArrayLike) -> NDArray[np.float64]:
    z = np.asarray(z, dtype=np.float64)                  # Convertit l'altitude en tableau numpy.
    if np.any(z < 0):                                    # Vérifie que l'altitude n'est pas négative.
        raise ValueError("L'altitude z doit être ≥ 0 m.")
    if np.any(z > MAX_ALTITUDE):                         # Vérifie que l'altitude reste dans le modèle.
        raise ValueError(f"L'altitude dépasse la limite du modèle ({MAX_ALTITUDE/1e3:.0f} km).")
    return z                                             # Renvoie l'altitude validée.


def _geopotential_altitude(z: ArrayLike) -> NDArray[np.float64]:
    z = np.asarray(z, dtype=np.float64)                  # Convertit l'altitude en tableau numpy.
    return R_EARTH * z / (R_EARTH + z)                   # Altitude géopotentielle tenant compte de g(z).


def _layer_indices(z: NDArray[np.float64]) -> NDArray[np.int_]:
    return np.searchsorted(LAYER_Z_MAX, z, side="right").clip(max=len(LAYERS) - 1)
                                                             # Renvoie l'indice de la couche correspondant à chaque altitude.


def _ordered_gases(gases) -> tuple[str, ...]:
    gas_set = set(gases)                                  # Supprime les doublons.
    ordered = [gas for gas in GAS_ORDER if gas in gas_set] # Place les gaz connus dans l'ordre choisi.
    ordered += sorted(gas_set - set(ordered))             # Ajoute les éventuels autres gaz à la fin.
    return tuple(ordered)                                 # Renvoie une liste immuable.


def available_gases() -> tuple[str, ...]:
    """Liste de toutes les espèces présentes dans le code."""
    return _ordered_gases(MOLAR_MASSES)                   # Tous les gaz définis dans MOLAR_MASSES.


def normalize_gas_name(gas: str) -> str:
    """Accepte par exemple 'co2', 'CO2', 'he', 'He' et renvoie le nom canonique."""
    gas_clean = gas.strip()                               # Supprime les espaces avant/après le nom du gaz.
    for known_gas in MOLAR_MASSES:                        # Parcourt les gaz connus.
        if gas_clean.upper() == known_gas.upper():        # Compare sans tenir compte des majuscules.
            return known_gas                              # Renvoie le nom officiel du gaz.
    raise ValueError(
        f"Gaz inconnu : {gas!r}. Gaz disponibles : {', '.join(available_gases())}"
    )


@lru_cache(maxsize=None)
def _layer_data(layer_idx: int) -> tuple[dict[str, float], float]:
    layer = LAYERS[layer_idx]                             # Récupère la couche demandée.

    missing = set(layer.fractions) - set(MOLAR_MASSES)    # Vérifie si une masse molaire manque.
    if missing:
        raise ValueError(f"[{layer.label}] Masse molaire manquante pour : {', '.join(sorted(missing))}")

    total = sum(layer.fractions.values())                 # Somme des fractions molaires de la couche.
    if not np.isclose(total, 1.0, atol=1e-3):              # Vérifie que la somme est proche de 1.
        raise ValueError(f"[{layer.label}] Σ fractions = {total:.8f} — doit être proche de 1.")

    fractions = {gas: x / total for gas, x in layer.fractions.items()}
                                                             # Normalise les fractions pour que la somme fasse 1.
    mean_molar_mass = sum(fractions[gas] * MOLAR_MASSES[gas] for gas in fractions)
                                                             # Calcule la masse molaire moyenne de l'air dans la couche.
    return fractions, mean_molar_mass                     # Renvoie la composition et la masse molaire moyenne.


def _state(z: ArrayLike) -> tuple[NDArray[object], NDArray[np.float64], NDArray[np.float64]]:
    z_arr = _as_valid_array(z)                            # Vérifie que l'altitude est valide.
    scalar_input = z_arr.ndim == 0                        # Retient si l'utilisateur a donné une seule valeur.
    z_arr = np.atleast_1d(z_arr)                          # Transforme en tableau pour simplifier les calculs.

    indices = _layer_indices(z_arr)                       # Trouve la couche de chaque altitude.
    fractions_arr = np.empty(z_arr.shape, dtype=object)   # Prépare un tableau pour les compositions.
    mean_molar_mass_arr = np.empty(z_arr.shape, dtype=np.float64)
                                                             # Prépare un tableau pour les masses molaires moyennes.

    for idx in np.unique(indices):                         # Parcourt seulement les couches présentes.
        fractions, mean_molar_mass = _layer_data(int(idx)) # Récupère les données de la couche.
        mask = indices == idx                              # Sélectionne les altitudes appartenant à cette couche.
        fractions_arr[mask] = fractions                    # Affecte la composition correspondante.
        mean_molar_mass_arr[mask] = mean_molar_mass        # Affecte la masse molaire moyenne correspondante.

    z_geo = _geopotential_altitude(z_arr)                  # Convertit z en altitude géopotentielle.
    exponent = np.empty_like(z_arr)                        # Prépare l'exposant de la loi barométrique.
    exponent_start = 0.0                                   # Valeur de départ de l'exposant au niveau de la mer.

    for idx, layer in enumerate(LAYERS):                   # Intègre la pression couche par couche.
        _, mean_molar_mass = _layer_data(idx)              # Masse molaire moyenne de la couche.
        z_min_geo = _geopotential_altitude(layer.z_min)    # Altitude géopotentielle au début de la couche.
        mask = indices == idx                              # Altitudes situées dans cette couche.

        if np.any(mask):                                   # Si au moins une altitude est dans cette couche.
            exponent[mask] = (
                exponent_start
                - mean_molar_mass * G0 / (R * T0) * (z_geo[mask] - z_min_geo)
            )                                             # Loi barométrique isotherme intégrée dans la couche.

        if idx < len(LAYERS) - 1:                          # Prépare le départ de la couche suivante.
            z_max_geo = _geopotential_altitude(layer.z_max)
            exponent_start -= mean_molar_mass * G0 / (R * T0) * (z_max_geo - z_min_geo)

    pressure = P0 * np.exp(exponent)                       # Pression totale obtenue à partir de l'exposant.

    if scalar_input:                                       # Si l'utilisateur avait donné une seule altitude.
        return fractions_arr[0], float(mean_molar_mass_arr[0]), float(pressure[0])
    return fractions_arr, mean_molar_mass_arr, pressure    # Sinon, renvoie des tableaux.


def layer_index_at_altitude(z: float) -> int:
    """Indice de la couche contenant l'altitude z."""
    z_arr = _as_valid_array(float(z))                      # Vérifie l'altitude.
    return int(_layer_indices(np.atleast_1d(z_arr))[0])    # Renvoie l'indice de la couche.


# =============================================================================
# Fonctions de base
# =============================================================================
# Ces fonctions donnent les grandeurs physiques générales de l'air.


def gravity(z: ArrayLike) -> NDArray[np.float64]:
    z = _as_valid_array(z)                                 # Vérifie l'altitude.
    return G0 * (R_EARTH / (R_EARTH + z)) ** 2             # Pesanteur qui diminue avec l'altitude.


def pressure_air(z: ArrayLike) -> NDArray[np.float64]:
    return _state(z)[2]                                    # Pression totale de l'air à l'altitude z.


def total_molar_concentration(z: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(pressure_air(z), dtype=np.float64) / (R * T0)
                                                             # Concentration molaire totale : C = P / RT.


def total_number_density(z: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(pressure_air(z), dtype=np.float64) / (K_B * T0)
                                                             # Densité numérique totale : n = P / kBT.


def air_mass_density(z: ArrayLike) -> NDArray[np.float64]:
    _, mean_molar_mass, pressure = _state(z)                # Récupère masse molaire moyenne et pression.
    return np.asarray(pressure) * np.asarray(mean_molar_mass) / (R * T0)
                                                             # Masse volumique de l'air : rho = P M / RT.


# =============================================================================
# Fonctions réutilisables par espèce
# =============================================================================
# Ces fonctions calculent les grandeurs pour un gaz précis.


def gas_info_at_altitude(z: float, gas: str) -> dict[str, float | str]:
    """
    Renvoie toutes les infos utiles pour un gaz à une altitude donnée.

    Exemple :
        info = gas_info_at_altitude(3500, "CO2")
        print(info["partial_pressure_Pa"])
    """
    gas = normalize_gas_name(gas)                          # Transforme le nom du gaz en nom officiel.
    z = float(z)                                           # Convertit l'altitude en nombre réel.

    idx = layer_index_at_altitude(z)                       # Trouve l'indice de la couche atmosphérique.
    layer = LAYERS[idx]                                    # Récupère la couche correspondante.
    fractions, _ = _layer_data(idx)                        # Récupère les fractions molaires de la couche.

    xi = float(fractions.get(gas, 0.0))                    # Fraction molaire du gaz ; 0 si absent de la couche.
    P = float(pressure_air(z))                             # Pression totale à cette altitude.
    C_total = P / (R * T0)                                 # Concentration molaire totale de l'air.

    return {
        "gas": gas,                                        # Nom du gaz.
        "altitude_m": z,                                   # Altitude en mètres.
        "altitude_km": z / 1000,                           # Altitude en kilomètres.
        "layer_index": idx,                                # Numéro de la couche.
        "layer_label": layer.label,                        # Nom de la couche.
        "layer_z_min_m": layer.z_min,                      # Début de la couche.
        "layer_z_max_m": layer.z_max,                      # Fin de la couche.
        "molar_mass_kg_mol": MOLAR_MASSES[gas],            # Masse molaire du gaz.
        "xi": xi,                                          # Fraction molaire du gaz.
        "gravity_m_s2": float(gravity(z)),                 # Pesanteur à cette altitude.
        "total_pressure_Pa": P,                            # Pression totale.
        "partial_pressure_Pa": xi * P,                     # Pression partielle : Pi = xi × P.
        "total_molar_concentration_mol_m3": C_total,       # Concentration molaire totale.
        "molar_concentration_mol_m3": xi * C_total,        # Concentration molaire du gaz.
        "number_density_molecules_m3": xi * P / (K_B * T0),# Densité numérique du gaz.
        "mass_concentration_kg_m3": xi * C_total * MOLAR_MASSES[gas],
                                                             # Concentration massique du gaz.
    }


def all_gases_info_at_altitude(z: float) -> dict[str, dict[str, float | str]]:
    """Même calcul, mais pour toutes les espèces chimiques définies dans le code."""
    return {gas: gas_info_at_altitude(z, gas) for gas in available_gases()}
                                                             # Calcule les infos pour tous les gaz disponibles.


# Anciennes fonctions conservées pour compatibilité.
# Elles permettent encore d'utiliser d'anciens scripts basés sur ce code.
def species_values_at_altitude(z: ArrayLike, gases: tuple[str, ...] | None = None) -> dict[str, dict[str, NDArray[np.float64]]]:
    fractions_arr, _, pressure = _state(z)                 # Récupère fractions et pression.
    pressure = np.atleast_1d(np.asarray(pressure, dtype=np.float64))
                                                             # Convertit la pression en tableau.
    fractions_arr = np.atleast_1d(fractions_arr)           # Convertit les fractions en tableau.
    C_total = pressure / (R * T0)                           # Concentration molaire totale.

    if gases is None:                                      # Si aucun gaz n'est demandé.
        gases = _ordered_gases(set().union(*(f.keys() for f in fractions_arr)))
                                                             # Utilise tous les gaz présents dans les couches.

    return {
        gas: {
            "xi": np.array([f.get(gas, 0.0) for f in fractions_arr], dtype=np.float64),
                                                             # Fraction molaire du gaz.
            "partial_pressure_Pa": np.array([f.get(gas, 0.0) * p for f, p in zip(fractions_arr, pressure)]),
                                                             # Pression partielle du gaz.
            "molar_concentration_mol_m3": np.array([f.get(gas, 0.0) * c for f, c in zip(fractions_arr, C_total)]),
                                                             # Concentration molaire du gaz.
        }
        for gas in gases
    }


def gas_partial_pressures(z: ArrayLike) -> dict[str, NDArray[np.float64]]:
    values = species_values_at_altitude(z)                 # Calcule les valeurs par espèce.
    return {gas: data["partial_pressure_Pa"] for gas, data in values.items()}
                                                             # Extrait seulement les pressions partielles.


def gas_molar_concentrations(z: ArrayLike) -> dict[str, NDArray[np.float64]]:
    values = species_values_at_altitude(z)                 # Calcule les valeurs par espèce.
    return {gas: data["molar_concentration_mol_m3"] for gas, data in values.items()}
                                                             # Extrait seulement les concentrations molaires.


def gas_number_densities(z: ArrayLike) -> dict[str, NDArray[np.float64]]:
    values = species_values_at_altitude(z)                 # Calcule les valeurs par espèce.
    return {gas: data["molar_concentration_mol_m3"] * (R / K_B) for gas, data in values.items()}
                                                             # Convertit mol/m³ en molécules/m³.


def gas_mass_concentrations(z: ArrayLike) -> dict[str, NDArray[np.float64]]:
    values = species_values_at_altitude(z)                 # Calcule les valeurs par espèce.
    return {gas: data["molar_concentration_mol_m3"] * MOLAR_MASSES[gas] for gas, data in values.items()}
                                                             # Convertit les concentrations molaires en kg/m³.


# =============================================================================
# Interface simple : demande altitude + gaz
# =============================================================================
# Cette dernière partie sert à lancer le programme directement depuis un terminal.


def print_gas_info(info: dict[str, float | str]) -> None:
    """Affiche proprement le résultat de gas_info_at_altitude()."""
    print("\n" + "═" * 68)                               # Ligne de séparation.
    print(f"Gaz demandé      : {info['gas']}")            # Gaz étudié.
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
    print("═" * 68 + "\n")                              # Ligne de fin d'affichage.


def ask_gas_info() -> dict[str, float | str]:
    """Demande l'altitude et le gaz, puis renvoie les infos du gaz."""
    print("Gaz disponibles :", ", ".join(available_gases())) # Affiche les gaz utilisables.
    z = float(input("Altitude z en mètres : ").replace(",", "."))
                                                             # Demande l'altitude et accepte les virgules.
    gas = input("Gaz recherché : ")                          # Demande le gaz à étudier.
    info = gas_info_at_altitude(z, gas)                      # Calcule toutes les informations.
    print_gas_info(info)                                     # Affiche le résultat.
    return info                                              # Renvoie aussi les données au programme.


if __name__ == "__main__":                                  # S'exécute seulement si le fichier est lancé directement.
    ask_gas_info()                                          # Lance l'interface interactive.

