"""
modele_atmosphere_modulaire.py

Modèle radiatif beta, volontairement simple mais proprement structuré.
But : pouvoir remplacer plus tard les moles et/ou les épaisseurs optiques
par un fichier CSV, une API, ou un autre module Python.

Ce fichier modélise une colonne atmosphérique découpée en couches.
Chaque couche contient :
- une géométrie dx, dy, dz ;
- une composition en moles ;
- une température ;
- une énergie interne ;
- éventuellement des épaisseurs optiques externes.

Le rayonnement est un spectre F(lambda), pas un simple nombre.
Unité utilisée pour le spectre : W.m^-2.um^-1.

Dépendances :
    pip install numpy matplotlib

Auteur : version beta pédagogique.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Any
import csv
import json
import math
import urllib.request

import numpy as np


# ============================================================
# Constantes physiques
# ============================================================

N_A = 6.022_140_76e23          # mol^-1
R = 8.314_462_618              # J.mol^-1.K^-1
h = 6.626_070_15e-34           # J.s
c = 299_792_458.0              # m.s^-1
k_B = 1.380_649e-23            # J.K^-1
SIGMA_SB = 5.670_374_419e-8    # W.m^-2.K^-4


# Capacités thermiques molaires à volume constant, valeurs simplifiées.
# J.mol^-1.K^-1
DEFAULT_CV_MOLAR = {
    "N2": 20.8,
    "O2": 20.8,
    "Ar": 12.47,
    "CO2": 28.8,
    "H2O": 27.0,
    "O3": 30.0,
    "CH4": 27.0,
    "N2O": 30.0,
    "Ne": 12.47,
    "He": 12.47,
}

# Fractions molaires beta, air sec + quelques traces.
# H2O est traité à part car très variable.
DEFAULT_DRY_MOLE_FRACTIONS = {
    "N2": 0.78084,
    "O2": 0.20946,
    "Ar": 0.00934,
    "CO2": 425e-6,
    "Ne": 18.2e-6,
    "He": 5.24e-6,
    "CH4": 1.9e-6,
    "N2O": 0.335e-6,
}


# ============================================================
# Fonctions utilitaires
# ============================================================

def trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def gaussian(x: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    """Gaussienne simple pour créer des bandes d'absorption beta."""
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def planck_flux_per_um(wavelength_um: np.ndarray, temperature_K: float) -> np.ndarray:
    """
    Flux spectral hémisphérique d'un corps noir : pi * B_lambda.

    Retour : W.m^-2.um^-1.
    wavelength_um : longueur d'onde en micromètres.
    """
    wl_m = wavelength_um * 1e-6
    T = max(float(temperature_K), 1e-9)

    # B_lambda en W.m^-2.sr^-1.m^-1
    a = 2.0 * h * c**2 / wl_m**5
    b = h * c / (wl_m * k_B * T)

    # Protection overflow : exp(>700) devient inf en double précision.
    b = np.clip(b, 1e-12, 700)
    B_lambda_m = a / (np.exp(b) - 1.0)

    # Flux hémisphérique = pi * radiance.
    flux_per_m = math.pi * B_lambda_m

    # Conversion par micromètre : d(lambda_m) = 1e-6 d(lambda_um)
    return flux_per_m * 1e-6


def interp_like(x_new: np.ndarray, x_old: np.ndarray, y_old: np.ndarray) -> np.ndarray:
    """Interpolation linéaire de y_old(x_old) vers x_new."""
    return np.interp(x_new, x_old, y_old, left=0.0, right=0.0)


# ============================================================
# Classe Spectrum
# ============================================================

@dataclass
class Spectrum:
    """
    Spectre radiatif.

    wavelengths_um : tableau des longueurs d'onde en micromètres.
    flux_w_m2_um   : flux spectral en W.m^-2.um^-1.
    name           : nom optionnel.
    """

    wavelengths_um: np.ndarray
    flux_w_m2_um: np.ndarray
    name: str = "spectre"

    def __post_init__(self) -> None:
        self.wavelengths_um = np.asarray(self.wavelengths_um, dtype=float)
        self.flux_w_m2_um = np.asarray(self.flux_w_m2_um, dtype=float)

        if self.wavelengths_um.ndim != 1:
            raise ValueError("wavelengths_um doit être un tableau 1D.")
        if self.flux_w_m2_um.ndim != 1:
            raise ValueError("flux_w_m2_um doit être un tableau 1D.")
        if len(self.wavelengths_um) != len(self.flux_w_m2_um):
            raise ValueError("wavelengths_um et flux_w_m2_um doivent avoir la même taille.")
        if np.any(np.diff(self.wavelengths_um) <= 0):
            raise ValueError("wavelengths_um doit être strictement croissant.")

    def copy(self, name: Optional[str] = None) -> "Spectrum":
        return Spectrum(
            wavelengths_um=self.wavelengths_um.copy(),
            flux_w_m2_um=self.flux_w_m2_um.copy(),
            name=self.name if name is None else name,
        )

    def integrate(self) -> float:
        """Flux total en W.m^-2."""
        return trapz(self.flux_w_m2_um, self.wavelengths_um)

    def scaled_to_total_flux(self, target_flux_w_m2: float, name: Optional[str] = None) -> "Spectrum":
        """Renormalise le spectre pour que son intégrale vaille target_flux_w_m2."""
        total = self.integrate()
        if total <= 0:
            raise ValueError("Impossible de renormaliser un spectre de flux total nul.")
        return Spectrum(
            self.wavelengths_um,
            self.flux_w_m2_um * (target_flux_w_m2 / total),
            name=name or self.name,
        )

    def resampled(self, new_wavelengths_um: np.ndarray, name: Optional[str] = None) -> "Spectrum":
        """Rééchantillonne le spectre sur une nouvelle grille."""
        return Spectrum(
            new_wavelengths_um,
            interp_like(new_wavelengths_um, self.wavelengths_um, self.flux_w_m2_um),
            name=name or self.name,
        )

    @classmethod
    def beta_solar(
        cls,
        wavelengths_um: np.ndarray,
        solar_constant_w_m2: float = 1361.0,
        sun_temperature_K: float = 5778.0,
        name: str = "Soleil beta",
    ) -> "Spectrum":
        """
        Spectre solaire beta : corps noir à 5778 K, renormalisé.

        solar_constant_w_m2 = 1361 W/m2 au sommet de l'atmosphère.
        Pour une moyenne planétaire, utiliser plutôt 1361/4 ≈ 340 W/m2.
        """
        raw = planck_flux_per_um(wavelengths_um, sun_temperature_K)
        sp = cls(wavelengths_um, raw, name=name)
        return sp.scaled_to_total_flux(solar_constant_w_m2, name=name)

    @classmethod
    def blackbody_surface(
        cls,
        wavelengths_um: np.ndarray,
        temperature_K: float,
        name: str = "corps noir",
    ) -> "Spectrum":
        """Spectre thermique d'une surface noire à la température donnée."""
        return cls(wavelengths_um, planck_flux_per_um(wavelengths_um, temperature_K), name=name)


# ============================================================
# Classe AtmosphereCell
# ============================================================

@dataclass
class AtmosphereCell:
    """
    Petite couche/cellule d'atmosphère.

    composition_mol : dictionnaire {gaz: quantité en moles dans le cube}.
    Les moles pourront venir plus tard d'un CSV, d'une API, d'un autre module, etc.

    Si tau_abs_external et tau_sca_external sont fournis, ils remplacent les
    épaisseurs optiques beta pour cette couche.
    """

    dx: float
    dy: float
    dz: float
    composition_mol: Dict[str, float] = field(default_factory=dict)
    temperature_K: float = 288.0
    altitude_m: float = 0.0
    name: str = "cellule"

    aerosol_tau_550: float = 0.0
    aerosol_alpha: float = 1.3
    cloud_tau_gray: float = 0.0

    tau_wavelengths_um_external: Optional[np.ndarray] = None
    tau_abs_external: Optional[np.ndarray] = None
    tau_sca_external: Optional[np.ndarray] = None

    U_J: Optional[float] = None

    def __post_init__(self) -> None:
        self.dx = float(self.dx)
        self.dy = float(self.dy)
        self.dz = float(self.dz)
        self.temperature_K = float(self.temperature_K)
        self.altitude_m = float(self.altitude_m)

        if self.dx <= 0 or self.dy <= 0 or self.dz <= 0:
            raise ValueError("dx, dy et dz doivent être strictement positifs.")

        self.composition_mol = {str(k): float(v) for k, v in self.composition_mol.items()}

        if self.U_J is None:
            self.U_J = self.heat_capacity_J_K() * self.temperature_K

    @property
    def area_m2(self) -> float:
        return self.dx * self.dy

    @property
    def volume_m3(self) -> float:
        return self.dx * self.dy * self.dz

    def total_moles(self) -> float:
        return float(sum(max(v, 0.0) for v in self.composition_mol.values()))

    def heat_capacity_J_K(self, cv_molar: Optional[Dict[str, float]] = None) -> float:
        """Capacité thermique totale de la cellule, en J/K."""
        cv = DEFAULT_CV_MOLAR if cv_molar is None else cv_molar
        C = 0.0
        for gas, n_mol in self.composition_mol.items():
            C += max(n_mol, 0.0) * cv.get(gas, 20.8)
        return max(C, 1e-12)

    def refresh_internal_energy_from_temperature(self) -> None:
        """Recalcule U à partir de T. Utile après modification de composition."""
        self.U_J = self.heat_capacity_J_K() * self.temperature_K

    def refresh_temperature_from_internal_energy(self) -> None:
        """Recalcule T à partir de U."""
        self.temperature_K = max(float(self.U_J) / self.heat_capacity_J_K(), 1e-9)

    def set_composition_mol(self, composition_mol: Dict[str, float], keep_temperature: bool = True) -> None:
        """
        Remplace la composition.
        keep_temperature=True : conserve T et recalcule U.
        keep_temperature=False : conserve U et recalcule T.
        """
        self.composition_mol = {str(k): float(v) for k, v in composition_mol.items()}
        if keep_temperature:
            self.refresh_internal_energy_from_temperature()
        else:
            self.refresh_temperature_from_internal_energy()

    def set_external_optical_depth(
        self,
        wavelengths_um: Iterable[float],
        tau_abs: Iterable[float],
        tau_sca: Optional[Iterable[float]] = None,
    ) -> None:
        """
        Injecte des épaisseurs optiques externes pour cette cellule.

        tau_abs : absorption.
        tau_sca : diffusion/scattering. Si None, mis à 0.
        """
        wl = np.asarray(list(wavelengths_um), dtype=float)
        ta = np.asarray(list(tau_abs), dtype=float)
        ts = np.zeros_like(ta) if tau_sca is None else np.asarray(list(tau_sca), dtype=float)

        if len(wl) != len(ta) or len(wl) != len(ts):
            raise ValueError("wavelengths, tau_abs et tau_sca doivent avoir la même taille.")
        if np.any(np.diff(wl) <= 0):
            raise ValueError("La grille wavelength externe doit être croissante.")

        self.tau_wavelengths_um_external = wl
        self.tau_abs_external = np.maximum(ta, 0.0)
        self.tau_sca_external = np.maximum(ts, 0.0)


# ============================================================
# Fournisseurs d'épaisseur optique
# ============================================================

class OpticalDepthProvider:
    """Interface minimale pour fournir tau_abs(lambda) et tau_sca(lambda)."""

    def get_tau(self, cell: AtmosphereCell, wavelengths_um: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class BetaOpticalDepthProvider(OpticalDepthProvider):
    """
    Fournisseur beta.

    Il calcule tau à partir des moles + sections efficaces fictives.
    Ce n'est PAS un modèle spectroscopique réel.
    Le but est de faire tourner correctement l'architecture.
    """

    def __init__(self, absorption_scale: float = 1.0, scattering_scale: float = 1.0):
        self.absorption_scale = float(absorption_scale)
        self.scattering_scale = float(scattering_scale)

        # Bandes beta : (centre_um, largeur_um, sigma_pic_m2_par_molecule)
        # Ce sont des valeurs fictives calibrées pour donner des effets visibles.
        self.absorption_bands: Dict[str, List[Tuple[float, float, float]]] = {
            "CO2": [
                (2.0, 0.12, 1.0e-25),
                (2.7, 0.18, 3.0e-25),
                (4.3, 0.22, 2.0e-23),
                (15.0, 1.80, 6.0e-24),
            ],
            "H2O": [
                (0.94, 0.05, 5.0e-27),
                (1.13, 0.07, 8.0e-27),
                (1.40, 0.12, 2.0e-26),
                (1.90, 0.18, 5.0e-26),
                (2.70, 0.35, 5.0e-25),
                (6.30, 0.80, 3.0e-24),
                (20.0, 8.00, 7.0e-25),
            ],
            "O3": [
                (0.25, 0.05, 1.0e-21),
                (0.60, 0.18, 1.0e-25),
                (9.60, 0.80, 1.5e-22),
            ],
            "O2": [
                (0.69, 0.02, 2.0e-28),
                (0.76, 0.02, 5.0e-28),
            ],
            "CH4": [
                (3.30, 0.25, 2.0e-23),
                (7.70, 0.70, 1.0e-23),
            ],
            "N2O": [
                (4.50, 0.30, 1.0e-23),
                (7.80, 0.60, 1.0e-23),
            ],
        }

        # Sections efficaces Rayleigh beta à 0.55 um.
        self.rayleigh_sigma_550 = {
            "N2": 5.1e-31,
            "O2": 4.8e-31,
            "Ar": 4.5e-31,
            "CO2": 1.1e-30,
            "H2O": 2.5e-31,
        }

    def _external_or_none(
        self, cell: AtmosphereCell, wavelengths_um: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if cell.tau_abs_external is None:
            return None
        if cell.tau_wavelengths_um_external is None:
            raise ValueError(f"{cell.name}: tau externe sans grille wavelength.")

        tau_abs = interp_like(
            wavelengths_um,
            cell.tau_wavelengths_um_external,
            np.asarray(cell.tau_abs_external, dtype=float),
        )

        if cell.tau_sca_external is None:
            tau_sca = np.zeros_like(tau_abs)
        else:
            tau_sca = interp_like(
                wavelengths_um,
                cell.tau_wavelengths_um_external,
                np.asarray(cell.tau_sca_external, dtype=float),
            )
        return np.maximum(tau_abs, 0.0), np.maximum(tau_sca, 0.0)

    def get_tau(self, cell: AtmosphereCell, wavelengths_um: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        external = self._external_or_none(cell, wavelengths_um)
        if external is not None:
            return external

        wl = np.asarray(wavelengths_um, dtype=float)
        tau_abs = np.zeros_like(wl)
        tau_sca = np.zeros_like(wl)

        area = cell.area_m2
        if area <= 0:
            raise ValueError("Aire de cellule non positive.")

        for gas, n_mol in cell.composition_mol.items():
            n_molecules_column = max(n_mol, 0.0) * N_A / area  # molécules.m^-2

            # Absorption beta.
            for center, width, sigma_peak in self.absorption_bands.get(gas, []):
                sigma_lambda = gaussian(wl, center, width, sigma_peak)
                tau_abs += n_molecules_column * sigma_lambda * self.absorption_scale

            # Diffusion Rayleigh beta.
            if gas in self.rayleigh_sigma_550:
                sigma_550 = self.rayleigh_sigma_550[gas]
                sigma_rayleigh = sigma_550 * (0.55 / np.maximum(wl, 1e-9)) ** 4
                tau_sca += n_molecules_column * sigma_rayleigh * self.scattering_scale

        # Aérosols : tau(lambda) = tau_550 * (lambda/0.55)^(-alpha)
        if cell.aerosol_tau_550 > 0:
            tau_sca += cell.aerosol_tau_550 * (np.maximum(wl, 1e-9) / 0.55) ** (-cell.aerosol_alpha)

        # Nuage gris : extinction non spectrale. Ici on le met surtout en diffusion.
        if cell.cloud_tau_gray > 0:
            tau_sca += cell.cloud_tau_gray

        return np.maximum(tau_abs, 0.0), np.maximum(tau_sca, 0.0)


# ============================================================
# Fournisseurs externes : CSV/API/callback
# ============================================================

class MoleProvider:
    """Interface minimale pour fournir une composition en moles."""

    def get_composition_for_layer(self, layer_index: int, cell: AtmosphereCell) -> Dict[str, float]:
        raise NotImplementedError


class CsvMoleProvider(MoleProvider):
    """
    Lit un CSV de moles.

    Format large recommandé :
        layer,N2,O2,Ar,CO2,H2O,O3,CH4,N2O
        0,3200,860,38,1.7,40,0,0.006,0.001
        1,2500,670,29,1.3,20,0,0.004,0.001

    Le nom de colonne 'layer' peut être changé.
    """

    def __init__(self, csv_path: str, layer_column: str = "layer"):
        self.csv_path = csv_path
        self.layer_column = layer_column
        self.data: Dict[int, Dict[str, float]] = {}
        self._load()

    def _load(self) -> None:
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV vide ou sans en-tête.")
            if self.layer_column not in reader.fieldnames:
                raise ValueError(f"Colonne '{self.layer_column}' absente du CSV.")

            for row in reader:
                idx = int(row[self.layer_column])
                comp: Dict[str, float] = {}
                for key, value in row.items():
                    if key == self.layer_column or value is None or value == "":
                        continue
                    comp[key] = float(value)
                self.data[idx] = comp

    def get_composition_for_layer(self, layer_index: int, cell: AtmosphereCell) -> Dict[str, float]:
        if layer_index not in self.data:
            raise KeyError(f"Aucune composition CSV pour la couche {layer_index}.")
        return dict(self.data[layer_index])


class CsvOpticalDepthProvider(OpticalDepthProvider):
    """
    Lit un CSV d'épaisseurs optiques.

    Format recommandé :
        layer,wavelength_um,tau_abs,tau_sca
        0,0.20,0.01,0.20
        0,0.21,0.01,0.18
        ...
        1,0.20,0.02,0.15

    Chaque couche peut avoir sa propre grille en lambda.
    """

    def __init__(
        self,
        csv_path: str,
        layer_column: str = "layer",
        wavelength_column: str = "wavelength_um",
        tau_abs_column: str = "tau_abs",
        tau_sca_column: str = "tau_sca",
    ):
        self.csv_path = csv_path
        self.layer_column = layer_column
        self.wavelength_column = wavelength_column
        self.tau_abs_column = tau_abs_column
        self.tau_sca_column = tau_sca_column
        self.data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._load()

    def _load(self) -> None:
        by_layer: Dict[int, List[Tuple[float, float, float]]] = {}
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV vide ou sans en-tête.")
            for col in [self.layer_column, self.wavelength_column, self.tau_abs_column]:
                if col not in reader.fieldnames:
                    raise ValueError(f"Colonne '{col}' absente du CSV.")

            for row in reader:
                idx = int(row[self.layer_column])
                wl = float(row[self.wavelength_column])
                ta = float(row[self.tau_abs_column])
                ts = float(row.get(self.tau_sca_column, 0.0) or 0.0)
                by_layer.setdefault(idx, []).append((wl, ta, ts))

        for idx, values in by_layer.items():
            values = sorted(values, key=lambda t: t[0])
            wl = np.array([v[0] for v in values], dtype=float)
            ta = np.array([v[1] for v in values], dtype=float)
            ts = np.array([v[2] for v in values], dtype=float)
            self.data[idx] = (wl, np.maximum(ta, 0.0), np.maximum(ts, 0.0))

    def get_tau(self, cell: AtmosphereCell, wavelengths_um: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # On suppose que le nom de couche peut contenir son indice à la fin : couche_0, couche_1...
        # Sinon, AtmosphereColumn utilisera le wrapper LayerIndexedOpticalProvider ci-dessous.
        raise RuntimeError(
            "CsvOpticalDepthProvider doit être utilisé via LayerIndexedOpticalProvider "
            "pour connaître l'indice de couche."
        )

    def get_tau_for_layer(self, layer_index: int, wavelengths_um: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if layer_index not in self.data:
            raise KeyError(f"Aucune épaisseur optique CSV pour la couche {layer_index}.")
        wl_old, ta_old, ts_old = self.data[layer_index]
        ta = interp_like(wavelengths_um, wl_old, ta_old)
        ts = interp_like(wavelengths_um, wl_old, ts_old)
        return np.maximum(ta, 0.0), np.maximum(ts, 0.0)


class CallbackMoleProvider(MoleProvider):
    """
    Fournisseur de moles basé sur une fonction Python.

    callback(layer_index, cell) -> dict {gaz: moles}
    """

    def __init__(self, callback: Callable[[int, AtmosphereCell], Dict[str, float]]):
        self.callback = callback

    def get_composition_for_layer(self, layer_index: int, cell: AtmosphereCell) -> Dict[str, float]:
        return dict(self.callback(layer_index, cell))


class ApiJsonMoleProvider(MoleProvider):
    """
    Fournisseur API très simple.

    L'API doit retourner du JSON, par exemple :
    {
        "layers": {
            "0": {"N2": 3200, "O2": 860, "CO2": 1.7},
            "1": {"N2": 2500, "O2": 670, "CO2": 1.3}
        }
    }

    Pour éviter une dépendance à requests, on utilise urllib.
    """

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        self.url = url
        self.headers = headers or {}
        self.layers: Dict[int, Dict[str, float]] = {}
        self._download()

    def _download(self) -> None:
        req = urllib.request.Request(self.url, headers=self.headers)
        with urllib.request.urlopen(req, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        raw_layers = payload.get("layers", payload)
        self.layers = {int(k): {str(g): float(n) for g, n in v.items()} for k, v in raw_layers.items()}

    def get_composition_for_layer(self, layer_index: int, cell: AtmosphereCell) -> Dict[str, float]:
        if layer_index not in self.layers:
            raise KeyError(f"Aucune composition API pour la couche {layer_index}.")
        return dict(self.layers[layer_index])


# ============================================================
# Diagnostics
# ============================================================

@dataclass
class LayerRadiativeDiagnostic:
    layer_index: int
    layer_name: str
    altitude_m: float
    temperature_before_K: float
    temperature_after_K: float
    flux_in_w_m2: float
    flux_out_w_m2: float
    absorbed_flux_w_m2: float
    scattered_flux_w_m2: float
    emitted_flux_total_w_m2: float
    delta_U_J: float
    mean_tau_abs: float
    mean_tau_sca: float


# ============================================================
# Colonne atmosphérique
# ============================================================

@dataclass
class AtmosphereColumn:
    """
    Atmosphère complète sous forme de colonne verticale.

    cells : liste des couches, du haut vers le bas ou du bas vers le haut selon usage.
    Ici, pour un rayonnement solaire descendant, on utilisera de préférence :
        couche 0 = sommet de l'atmosphère,
        dernière couche = proche du sol.
    """

    cells: List[AtmosphereCell]
    wavelengths_um: np.ndarray
    optical_provider: OpticalDepthProvider = field(default_factory=BetaOpticalDepthProvider)
    name: str = "colonne atmosphérique"

    def __post_init__(self) -> None:
        self.wavelengths_um = np.asarray(self.wavelengths_um, dtype=float)
        if np.any(np.diff(self.wavelengths_um) <= 0):
            raise ValueError("wavelengths_um doit être strictement croissant.")

    def apply_mole_provider(self, provider: MoleProvider, keep_temperature: bool = True) -> None:
        """Remplace les compositions des couches via CSV/API/callback."""
        for i, cell in enumerate(self.cells):
            comp = provider.get_composition_for_layer(i, cell)
            cell.set_composition_mol(comp, keep_temperature=keep_temperature)

    def apply_csv_moles(self, csv_path: str, layer_column: str = "layer", keep_temperature: bool = True) -> None:
        provider = CsvMoleProvider(csv_path, layer_column=layer_column)
        self.apply_mole_provider(provider, keep_temperature=keep_temperature)

    def apply_csv_optical_depths(
        self,
        csv_path: str,
        layer_column: str = "layer",
        wavelength_column: str = "wavelength_um",
        tau_abs_column: str = "tau_abs",
        tau_sca_column: str = "tau_sca",
    ) -> None:
        """
        Injecte les épaisseurs optiques du CSV directement dans chaque cellule.
        Après ça, BetaOpticalDepthProvider les utilisera automatiquement.
        """
        provider = CsvOpticalDepthProvider(
            csv_path,
            layer_column=layer_column,
            wavelength_column=wavelength_column,
            tau_abs_column=tau_abs_column,
            tau_sca_column=tau_sca_column,
        )
        for i, cell in enumerate(self.cells):
            wl, ta, ts = provider.data[i]
            cell.set_external_optical_depth(wl, ta, ts)

    def radiative_step_downward(
        self,
        incoming: Spectrum,
        dt_s: float,
        mu: float = 1.0,
        include_thermal_emission: bool = True,
        include_scattering_as_loss: bool = True,
    ) -> Tuple[Spectrum, List[LayerRadiativeDiagnostic]]:
        """
        Fait traverser un spectre descendant dans la colonne.

        dt_s : durée du pas de temps.
        mu   : cos(angle zénithal). mu=1 rayon vertical, mu=0.5 rayon incliné.
        include_thermal_emission : ajoute l'émission thermique de chaque couche au flux sortant.
        include_scattering_as_loss : si True, la diffusion retire du flux direct.

        Retour : spectre sortant + diagnostics par couche.
        """
        if dt_s < 0:
            raise ValueError("dt_s doit être positif.")
        mu = max(float(mu), 1e-6)

        spectrum = incoming.resampled(self.wavelengths_um, name=f"{incoming.name} rééchantillonné")
        diagnostics: List[LayerRadiativeDiagnostic] = []

        for i, cell in enumerate(self.cells):
            T_before = cell.temperature_K
            flux_in_total = spectrum.integrate()

            tau_abs, tau_sca = self.optical_provider.get_tau(cell, self.wavelengths_um)
            tau_abs = np.maximum(tau_abs, 0.0)
            tau_sca = np.maximum(tau_sca, 0.0)

            if include_scattering_as_loss:
                tau_ext = tau_abs + tau_sca
            else:
                tau_ext = tau_abs

            transmission = np.exp(-tau_ext / mu)
            flux_direct_out = spectrum.flux_w_m2_um * transmission

            removed_spectral = np.maximum(spectrum.flux_w_m2_um - flux_direct_out, 0.0)
            tau_ext_safe = np.maximum(tau_ext, 1e-30)
            absorbed_spectral = removed_spectral * (tau_abs / tau_ext_safe)
            scattered_spectral = removed_spectral * (tau_sca / tau_ext_safe) if include_scattering_as_loss else np.zeros_like(removed_spectral)

            absorbed_flux = trapz(absorbed_spectral, self.wavelengths_um)
            scattered_flux = trapz(scattered_spectral, self.wavelengths_um)

            # Émission thermique de la couche.
            # Émissivité spectrale beta : epsilon(lambda) = 1 - exp(-tau_abs).
            # On suppose une émission des deux faces de la couche.
            if include_thermal_emission:
                emissivity = 1.0 - np.exp(-tau_abs)
                one_side_blackbody = planck_flux_per_um(self.wavelengths_um, cell.temperature_K)
                emitted_total_spectral = 2.0 * emissivity * one_side_blackbody
                emitted_total_flux = trapz(emitted_total_spectral, self.wavelengths_um)

                # La moitié est ajoutée au flux descendant, l'autre moitié part vers le haut.
                emitted_down_spectral = 0.5 * emitted_total_spectral
            else:
                emitted_total_spectral = np.zeros_like(self.wavelengths_um)
                emitted_down_spectral = np.zeros_like(self.wavelengths_um)
                emitted_total_flux = 0.0

            out_spectral = flux_direct_out + emitted_down_spectral
            out_spectral = np.maximum(out_spectral, 0.0)

            # Bilan énergétique de la couche.
            # Diffusion pure : elle enlève du flux direct mais ne chauffe pas directement.
            delta_U = cell.area_m2 * dt_s * (absorbed_flux - emitted_total_flux)
            cell.U_J = float(cell.U_J) + delta_U
            cell.refresh_temperature_from_internal_energy()

            spectrum = Spectrum(self.wavelengths_um, out_spectral, name=f"après {cell.name}")
            flux_out_total = spectrum.integrate()

            diagnostics.append(
                LayerRadiativeDiagnostic(
                    layer_index=i,
                    layer_name=cell.name,
                    altitude_m=cell.altitude_m,
                    temperature_before_K=T_before,
                    temperature_after_K=cell.temperature_K,
                    flux_in_w_m2=flux_in_total,
                    flux_out_w_m2=flux_out_total,
                    absorbed_flux_w_m2=absorbed_flux,
                    scattered_flux_w_m2=scattered_flux,
                    emitted_flux_total_w_m2=emitted_total_flux,
                    delta_U_J=delta_U,
                    mean_tau_abs=float(np.mean(tau_abs)),
                    mean_tau_sca=float(np.mean(tau_sca)),
                )
            )

        return spectrum, diagnostics

    def run(
        self,
        incoming: Spectrum,
        dt_s: float,
        n_steps: int = 1,
        mu: float = 1.0,
        include_thermal_emission: bool = True,
        include_scattering_as_loss: bool = True,
    ) -> Tuple[Spectrum, List[List[LayerRadiativeDiagnostic]]]:
        """Exécute plusieurs pas de temps."""
        spectrum = incoming
        all_diags: List[List[LayerRadiativeDiagnostic]] = []
        for _ in range(n_steps):
            spectrum, diags = self.radiative_step_downward(
                incoming=incoming,
                dt_s=dt_s,
                mu=mu,
                include_thermal_emission=include_thermal_emission,
                include_scattering_as_loss=include_scattering_as_loss,
            )
            all_diags.append(diags)
        return spectrum, all_diags

    @classmethod
    def beta_earth_column(
        cls,
        n_layers: int = 20,
        dz_m: float = 500.0,
        dx_m: float = 1.0,
        dy_m: float = 1.0,
        wavelengths_um: Optional[np.ndarray] = None,
        surface_temperature_K: float = 288.0,
        top_temperature_K: float = 220.0,
        P0_Pa: float = 101_325.0,
        scale_height_m: float = 8_400.0,
        water_surface_fraction: float = 0.01,
        water_scale_height_m: float = 2_000.0,
    ) -> "AtmosphereColumn":
        """
        Construit une colonne beta avec composition calculée par gaz parfait.

        IMPORTANT : dans cette fonction, la couche 0 est en haut de l'atmosphère,
        et la dernière couche est proche du sol, ce qui est pratique pour un
        rayonnement solaire descendant.
        """
        if wavelengths_um is None:
            wavelengths_um = np.linspace(0.2, 50.0, 2500)

        total_height = n_layers * dz_m
        cells_bottom_to_top: List[AtmosphereCell] = []

        for j in range(n_layers):
            z_mid = (j + 0.5) * dz_m
            frac = z_mid / max(total_height, 1e-9)
            T = surface_temperature_K + (top_temperature_K - surface_temperature_K) * frac
            P = P0_Pa * math.exp(-z_mid / scale_height_m)
            V = dx_m * dy_m * dz_m
            n_total = P * V / (R * T)

            # Vapeur d'eau très simplifiée : décroît vite avec l'altitude.
            x_h2o = water_surface_fraction * math.exp(-z_mid / water_scale_height_m)
            x_h2o = min(max(x_h2o, 0.0), 0.05)
            n_h2o = n_total * x_h2o
            n_dry = n_total - n_h2o

            comp = {gas: n_dry * x for gas, x in DEFAULT_DRY_MOLE_FRACTIONS.items()}
            if n_h2o > 0:
                comp["H2O"] = n_h2o

            # Ozone beta : bosse vers 25 km.
            ozone_fraction = 8e-6 * math.exp(-0.5 * ((z_mid - 25_000.0) / 7_000.0) ** 2)
            comp["O3"] = n_total * ozone_fraction

            cell = AtmosphereCell(
                dx=dx_m,
                dy=dy_m,
                dz=dz_m,
                composition_mol=comp,
                temperature_K=T,
                altitude_m=z_mid,
                name=f"couche_{j:02d}_{z_mid:.0f}m",
            )
            cells_bottom_to_top.append(cell)

        # Pour un flux descendant : on inverse, haut -> bas.
        cells_top_to_bottom = list(reversed(cells_bottom_to_top))
        for i, cell in enumerate(cells_top_to_bottom):
            cell.name = f"couche_{i:02d}_z{cell.altitude_m:.0f}m"

        return cls(
            cells=cells_top_to_bottom,
            wavelengths_um=np.asarray(wavelengths_um, dtype=float),
            optical_provider=BetaOpticalDepthProvider(),
            name="colonne beta Terre",
        )


# ============================================================
# Affichage / export diagnostics
# ============================================================

def diagnostics_to_dicts(diags: List[LayerRadiativeDiagnostic]) -> List[Dict[str, Any]]:
    """Convertit les diagnostics en dictionnaires simples."""
    return [d.__dict__.copy() for d in diags]


def print_diagnostics(diags: List[LayerRadiativeDiagnostic], max_layers: Optional[int] = None) -> None:
    """Affiche un résumé lisible."""
    shown = diags if max_layers is None else diags[:max_layers]
    print("\nDiagnostics par couche :")
    print(
        "idx | altitude(m) | T_avant -> T_apres (K) | "
        "Flux in -> out (W/m2) | absorbe | diffuse | emis | dU(J)"
    )
    for d in shown:
        print(
            f"{d.layer_index:3d} | "
            f"{d.altitude_m:10.0f} | "
            f"{d.temperature_before_K:7.2f} -> {d.temperature_after_K:7.2f} | "
            f"{d.flux_in_w_m2:8.2f} -> {d.flux_out_w_m2:8.2f} | "
            f"{d.absorbed_flux_w_m2:7.2f} | "
            f"{d.scattered_flux_w_m2:7.2f} | "
            f"{d.emitted_flux_total_w_m2:7.2f} | "
            f"{d.delta_U_J: .3e}"
        )


def save_diagnostics_csv(diags: List[LayerRadiativeDiagnostic], path: str) -> None:
    """Sauvegarde les diagnostics dans un CSV."""
    rows = diagnostics_to_dicts(diags)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_spectra(incoming: Spectrum, outgoing: Spectrum) -> None:
    """Trace les spectres si matplotlib est installé."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib n'est pas installé. Fais : pip install matplotlib")
        return

    plt.figure()
    plt.plot(incoming.wavelengths_um, incoming.flux_w_m2_um, label="entrant")
    plt.plot(outgoing.wavelengths_um, outgoing.flux_w_m2_um, label="sortant")
    plt.xlabel("Longueur d'onde (um)")
    plt.ylabel("Flux spectral (W m$^{-2}$ um$^{-1}$)")
    plt.title("Spectre entrant / sortant")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_temperature_profile(column: AtmosphereColumn) -> None:
    """Trace le profil vertical de température."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib n'est pas installé. Fais : pip install matplotlib")
        return

    z = np.array([cell.altitude_m for cell in column.cells])
    T = np.array([cell.temperature_K for cell in column.cells])

    plt.figure()
    plt.plot(T, z / 1000.0, marker="o")
    plt.xlabel("Température (K)")
    plt.ylabel("Altitude (km)")
    plt.title("Profil vertical de température")
    plt.grid(True)
    plt.show()


# ============================================================
# Exemple d'utilisation
# ============================================================

def example_basic() -> None:
    """Exemple autonome avec données beta."""

    # Grille spectrale : du proche UV à l'infrarouge thermique.
    wavelengths = np.linspace(0.2, 50.0, 2500)

    # Flux solaire au sommet de l'atmosphère.
    # Pour une colonne locale en plein soleil : 1361 W/m2.
    # Pour moyenne planétaire : 1361/4.
    incoming = Spectrum.beta_solar(wavelengths, solar_constant_w_m2=1361.0)

    # Colonne atmosphérique beta.
    atmosphere = AtmosphereColumn.beta_earth_column(
        n_layers=24,
        dz_m=1000.0,
        dx_m=1.0,
        dy_m=1.0,
        wavelengths_um=wavelengths,
    )

    # Pas de temps : 60 secondes.
    outgoing, all_diags = atmosphere.run(
        incoming=incoming,
        dt_s=60.0,
        n_steps=1,
        mu=1.0,
        include_thermal_emission=True,
        include_scattering_as_loss=True,
    )

    diags = all_diags[-1]

    print("=== Résumé global ===")
    print(f"Flux entrant total : {incoming.integrate():.2f} W/m2")
    print(f"Flux sortant total : {outgoing.integrate():.2f} W/m2")
    print(f"Flux perdu net     : {incoming.integrate() - outgoing.integrate():.2f} W/m2")

    print_diagnostics(diags, max_layers=8)
    save_diagnostics_csv(diags, "diagnostics_radiatifs.csv")

    plot_spectra(incoming, outgoing)
    plot_temperature_profile(atmosphere)


def example_replace_moles_with_csv() -> None:
    """
    Exemple d'utilisation CSV pour les moles.

    Décommente et adapte le chemin quand ton collègue fournira le fichier.

    Format attendu :
        layer,N2,O2,Ar,CO2,H2O,O3
        0,1000,270,12,0.5,0.1,0.001
        1,1200,320,14,0.6,0.2,0.001
    """
    wavelengths = np.linspace(0.2, 50.0, 2500)
    atmosphere = AtmosphereColumn.beta_earth_column(n_layers=10, dz_m=1000.0, wavelengths_um=wavelengths)

    # atmosphere.apply_csv_moles("moles_par_couche.csv")

    incoming = Spectrum.beta_solar(wavelengths, solar_constant_w_m2=1361.0)
    outgoing, all_diags = atmosphere.run(incoming, dt_s=60.0)
    print(outgoing.integrate())


def example_replace_tau_with_csv() -> None:
    """
    Exemple d'utilisation CSV pour les épaisseurs optiques.

    Format attendu :
        layer,wavelength_um,tau_abs,tau_sca
        0,0.20,0.01,0.10
        0,0.21,0.01,0.09
        ...
        1,0.20,0.02,0.08
    """
    wavelengths = np.linspace(0.2, 50.0, 2500)
    atmosphere = AtmosphereColumn.beta_earth_column(n_layers=10, dz_m=1000.0, wavelengths_um=wavelengths)

    # atmosphere.apply_csv_optical_depths("tau_par_couche.csv")

    incoming = Spectrum.beta_solar(wavelengths, solar_constant_w_m2=1361.0)
    outgoing, all_diags = atmosphere.run(incoming, dt_s=60.0)
    print(outgoing.integrate())


if __name__ == "__main__":
    example_basic()
