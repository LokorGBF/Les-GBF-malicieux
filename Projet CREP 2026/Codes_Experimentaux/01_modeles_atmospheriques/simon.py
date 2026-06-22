import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import matplotlib.pyplot as plt
from hapi import *

from atmosphere_isotherme import gas_info_at_altitude

# ----------------------------------------------------------------------------------------------------------------------

# ===================
# BLACKBODY RADIATION
# ===================

def planck_function(lambda_wavelength, T):
    h = 6.62607015e-34      # Planck's constant, J*s
    c = 2.998e8             # Speed of light, m/s
    kB = 1.380649e-23       # Boltzmann's constant, J/K
    term1 = (2 * h * c**2) / lambda_wavelength**5
    term2 = np.exp((h * c) / (lambda_wavelength * kB * T)) - 1
    return term1 / term2

# ----------------------------------------------------------------------------------------------------------------------

# ================
# ATMOSPHERE MODEL
# ================

# --- Profil ISA intégré (hydrostatique, g variable avec l'altitude) ---
# Intègre simultanément :
#   dT/dz = kISA(z)                       (gradients standard ISA par couche)
#   dP/dz = -(Mair * g(z) / (R * T)) * P  (équilibre hydrostatique)
# avec g(z) = G * Mterre / (RT + z)^2.
# Calculé une seule fois au chargement, puis temperature_ISA / pressure_ISA
# interpolent T(z) et P(z) (acceptent scalaire ou tableau, comme np.interp).

def build_ISA_profile(z_top=85e3, N=10000):
    G = 6.67e-11        # Gravitational constant, m^3/(kg*s^2)
    Mair = 29e-3        # Molar mass of air, kg/mol
    R = 8.314           # Gas constant, J/(mol*K)
    Mterre = 5.97e24    # Earth mass, kg
    RT = 6378e3         # Earth radius, m

    Tsol = 288.0        # Surface temperature, K
    Psol = 1.013e5      # Surface pressure, Pa

    def kISA(z):
        if 0 <= z < 11e3:
            return -6.5e-3
        elif z < 20e3:
            return 0.0
        elif z < 32e3:
            return 1.0e-3
        elif z < 47e3:
            return 2.8e-3
        elif z < 51e3:
            return 0.0
        elif z < 71e3:
            return -2.8e-3
        elif z <= 85e3:
            return -2.0e-3
        else:
            return 0.0

    Z = np.linspace(0, z_top, N)
    dz = Z[1] - Z[0]

    T = np.zeros(N)
    P = np.zeros(N)
    T[0] = Tsol
    P[0] = Psol

    for i in range(N - 1):
        z = Z[i]
        g = G * Mterre / (RT + z)**2
        T[i + 1] = T[i] + kISA(z) * dz
        P[i + 1] = P[i] - (Mair * g / (R * T[i])) * P[i] * dz

    return Z, T, P

# Profil calculé une seule fois
_Z_ISA, _T_ISA, _P_ISA = build_ISA_profile()

def temperature_ISA(z):
    return np.interp(z, _Z_ISA, _T_ISA)

def pressure_ISA(z):
    return np.interp(z, _Z_ISA, _P_ISA)


# --- Anciens modèles (conservés) ---

def pressure_barometric(z):
    P0 = 101325     # Pressure at sea level in Pa
    H = 8500        # Scale height in m
    return P0 * np.exp(-z / H)

def temperature_uniform(z):
    T0 = 288.2
    return T0 * np.ones_like(z)

def temperature_simple(z):
    T0 = 288.2     # Temperature at sea level in K
    z_trop = 11000  # Tropopause height in m
    Gamma = -0.0065 # Temperature gradient in K/m
    T_trop = T0 + Gamma * z_trop
    return np.piecewise(z, [z < z_trop, z >= z_trop],
                        [lambda z: T0 + Gamma * z,
                         lambda z: T_trop])

def temperature_US1976(z):
    z_km = z/1000  # Convert altitude to km for easier comparisons

    # Troposphere (0 to 11 km)
    T0 = 288.15
    z_trop = 11

    # Tropopause (11 to 20 km)
    T_tropopause = 216.65
    z_tropopause = 20

    # Stratosphere 1 (20 to 32 km)
    T_strat1 = T_tropopause
    z_strat1 = 32

    # Stratosphere 2 (32 to 47 km)
    T_strat2 = 228.65
    z_strat2 = 47

    # Stratopause (47 to 51 km)
    T_stratopause = 270.65
    z_stratopause = 51

    # Mesosphere 1 (51 to 71 km)
    T_meso1 = T_stratopause
    z_meso1 = 71

    # Mesosphere 2 (71 to ...)
    T_meso2 = 214.65

    return np.piecewise(z_km,
                        [z_km < z_trop,
                         (z_km >= z_trop) & (z_km < z_tropopause),
                         (z_km >= z_tropopause) & (z_km < z_strat1),
                         (z_km >= z_strat1) & (z_km < z_strat2),
                         (z_km >= z_strat2) & (z_km < z_stratopause),
                         (z_km >= z_stratopause) & (z_km < z_meso1),
                         z_km >= z_meso1],
                        [lambda z: T0 - 6.5 * z,
                         lambda z: T_tropopause,
                         lambda z: T_strat1 + 1 * (z - z_tropopause),
                         lambda z: T_strat2 + 2.8 * (z - z_strat1),
                         lambda z: T_stratopause,
                         lambda z: T_meso1 - 2.8 * (z - z_stratopause),
                         lambda z: T_meso2 - 2 * (z - z_meso1)])


# ==> CHOOSE HERE THE TEMPERATURE MODEL
def temperature(z):
    return temperature_ISA(z)

# ==> CHOOSE HERE THE PRESSURE MODEL
def pressure(z):
    return pressure_ISA(z)

def air_number_density(z):
    kB = 1.380649e-23  # Boltzmann's constant, J/K
    return pressure(z) / (kB * temperature(z))

# ----------------------------------------------------------------------------------------------------------------------

# ===========================================
# ÉPAISSEUR OPTIQUE D'UNE COUCHE  (HITRAN/hapi)
# ===========================================
#
# Utilitaire indépendant de la simu deux flux : calcule l'épaisseur optique
# spectrale et effective d'un gaz sur une couche [z_min, z_max] à partir des
# raies HITRAN et du profil ISA ci-dessus (build_ISA_profile).

HITRAN_IDS = {
    "CO2": (2, 1),
    "H2O": (1, 1),
    "O3":  (3, 1),
    "N2O": (4, 1),
    "CH4": (6, 1),
    "O2":  (7, 1),
}

SPECTRAL_BANDS = {
    "CO2": (600.0, 750.0),
    "H2O": (200.0, 800.0),
    "O3":  (980.0, 1100.0),
    "N2O": (500.0, 800.0),
    "CH4": (1200.0, 1400.0),
    "O2":  (7500.0, 8000.0),
}

DNU = 0.05
K_B = 1.380_649e-23


def epaisseur_optique(gas, z_min, z_max, N_z=500, plot=False, verbose=False):
    """
    Épaisseur optique spectrale et effective d'un gaz sur la couche
    [z_min, z_max], via les raies HITRAN et le profil ISA (build_ISA_profile).

    Retourne : nu [cm^-1], lambda_um [µm], tau (spectrale), tau_eff (scalaire).
    """

    gas = gas.strip().upper()

    if gas not in HITRAN_IDS:
        raise ValueError(f"Gaz {gas!r} non supporté. Choisir parmi : {list(HITRAN_IDS.keys())}")
    if z_min < 0 or z_max < 0:
        raise ValueError("Les altitudes doivent être positives.")
    if z_max <= z_min:
        raise ValueError("Il faut que z_max > z_min.")
    if z_max > 85_000:
        raise ValueError("Avec ce modèle ISA, l'altitude maximale est 85 000 m.")

    mol_id, iso_id = HITRAN_IDS[gas]
    NU_MIN, NU_MAX = SPECTRAL_BANDS[gas]

    # ---- Profil sur [z_min, z_max] (même intégration ISA que la simu) ----
    Z_atm, T_atm, P_atm_profile = build_ISA_profile()

    Z_grid = np.linspace(z_min, z_max, N_z)
    delta_z = z_max - z_min

    T_grid = np.interp(Z_grid, Z_atm, T_atm)
    P_grid = np.interp(Z_grid, Z_atm, P_atm_profile)

    xi_grid = np.zeros(N_z)
    for i, z in enumerate(Z_grid):
        info = gas_info_at_altitude(z, gas)
        xi_grid[i] = info["xi"]

    # Densité numérique : n(z) = xi(z) * P(z) / (k_B * T(z))
    n_grid = xi_grid * P_grid / (K_B * T_grid)

    # Moyennes sur la couche
    P_mean = np.trapz(P_grid, Z_grid) / delta_z
    T_mean = np.trapz(T_grid, Z_grid) / delta_z
    xi_mean = np.trapz(xi_grid, Z_grid) / delta_z
    P_mean_atm = P_mean / 101325.0

    # Colonne moléculaire
    colonne = np.trapz(n_grid, Z_grid)

    # ---- Données HITRAN et section efficace ----
    table_name = f"{gas}_band"
    db_begin("hitran_data")
    fetch(table_name, mol_id, iso_id, NU_MIN, NU_MAX)

    nu, sigma_cm2 = absorptionCoefficient_Voigt(
        Components=((mol_id, iso_id),),
        SourceTables=table_name,
        Environment={"p": P_mean_atm, "T": T_mean},
        OmegaRange=[NU_MIN, NU_MAX],
        OmegaStep=DNU,
        HITRAN_units=True,
        GammaL="gamma_air"
    )

    nu = np.array(nu)
    sigma_cm2 = np.array(sigma_cm2)
    sigma_m2 = sigma_cm2 * 1e-4
    lambda_um = 1e4 / nu

    # ---- Épaisseur optique spectrale ----
    tau = sigma_m2 * colonne

    # ---- Épaisseur optique effective (pondérée Planck) ----
    transmittance = np.where(tau > 700, 0.0, np.exp(-tau))

    c2 = 1.438776877  # cm.K
    poids_planck = nu**3 / np.expm1(c2 * nu / T_mean)

    T_bande = np.trapz(poids_planck * transmittance, nu) / np.trapz(poids_planck, nu)
    tau_eff = -np.log(T_bande) if T_bande > 0 else np.inf

    # ---- Affichage ----
    if verbose:
        print("\n" + "=" * 55)
        print("  RÉSULTATS")
        print("=" * 55)
        print(f"Gaz                    : {gas}")
        print(f"Bande spectrale        : {NU_MIN}–{NU_MAX} cm⁻¹")
        print(f"Couche                 : {z_min:.0f} → {z_max:.0f} m")
        print(f"Épaisseur de couche    : {delta_z:.0f} m")
        print(f"xi moyen               : {xi_mean:.4e}")
        print(f"T moyenne              : {T_mean:.2f} K")
        print(f"P moyenne              : {P_mean:.2f} Pa")
        print(f"Colonne                : {colonne:.4e} molécules/m²")
        print(f"Transmittance moyenne  : {T_bande:.4e}")
        print(f"Épaisseur optique eff. : {tau_eff:.4f}")
        print("=" * 55)

        i_max = np.argmax(sigma_m2)
        print(f"\nPic d'absorption :")
        print(f"  ν     = {nu[i_max]:.2f} cm⁻¹")
        print(f"  λ     = {lambda_um[i_max]:.3f} µm")
        print(f"  σ_max = {sigma_cm2[i_max]:.3e} cm²/molécule")
        print(f"  τ_max = {tau[i_max]:.3e}")

    # ---- Graphique ----
    if plot:
        plt.figure()
        plt.semilogy(lambda_um, tau, color="darkorange")
        plt.gca().invert_xaxis()
        plt.xlabel("λ (µm)")
        plt.ylabel("τ")
        plt.title(
            f"Épaisseur optique — {gas}\n"
            f"couche {z_min:.0f}–{z_max:.0f} m | Tmoy={T_mean:.1f} K | τ_eff={tau_eff:.3f}"
        )
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return nu, lambda_um, tau, tau_eff

# ----------------------------------------------------------------------------------------------------------------------

# ==============
# CO2 ABSORPTION
# ==============

def cross_section_CO2(wavelength):
    LAMBDA_0 = 15.0e-6  # Band center in m
    exponent = -22.5 - 24 * np.abs((wavelength - LAMBDA_0) / LAMBDA_0)
    sigma = 10 ** exponent
    return sigma

# ----------------------------------------------------------------------------------------------------------------------

# ===========================================
# RADIATIVE TRANSFER SIMULATION  -  TWO STREAM
# ===========================================
#
# Modèle à deux flux conforme au schéma : chaque couche k émet dans les DEUX
# directions et absorbe une fraction du flux qui la traverse.
#
#   émissivité de la couche :   epsilon_k(lambda) = 1 - exp(-tau_k(lambda))
#   (loi de Kirchhoff : absorptivité = émissivité)
#
#   émission par direction :    P_k,émis(lambda) = epsilon_k(lambda) * pi * B(lambda, T_k) * dlambda
#   absorption :                P_k,absorbé(lambda) = epsilon_k(lambda) * P_reçu(lambda)
#
# Flux MONTANT (de la surface vers le haut, on remonte couche par couche) :
#   P_k,émis vers haut = P_(k-1),émis vers haut - P_k,absorbé + P_k,émis
#                      = (1 - epsilon_k) * P_(k-1),haut + P_k,émis
#
# Flux DESCENDANT (du sommet vers le sol, on descend couche par couche) :
#   P_k,émis vers bas  = P_(k+1),émis vers bas  - P_k,absorbé + P_k,émis
#                      = (1 - epsilon_k) * P_(k+1),bas  + P_k,émis
#
# Conditions aux limites : surface = corps noir (montant), espace = 0 (descendant).
# À T fixé, les deux flux sont indépendants : on fait deux passes séparées.

def simulate_radiative_transfer(CO2_fraction, z_max = 80000, delta_z = 10, lambda_min = 0.1e-6, lambda_max = 100e-6, delta_lambda = 0.01e-6):

    # Altitude and wavelength grids
    z_range = np.arange(0, z_max, delta_z)
    lambda_range = np.arange(lambda_min, lambda_max, delta_lambda)

    # Initialize arrays
    upward_flux       = np.zeros((len(z_range), len(lambda_range)))
    downward_flux     = np.zeros((len(z_range), len(lambda_range)))
    optical_thickness = np.zeros((len(z_range), len(lambda_range)))

    # Boundary condition : surface flux (corps noir, epsilon = 1) for all wavelengths
    earth_flux = np.pi * planck_function(lambda_range, temperature(0)) * delta_lambda
    print(f"Total earth surface flux in wavelength range: {earth_flux.sum():.2f} W/m^2")

    # -------- PASSE MONTANTE : surface -> sommet --------
    flux_in = earth_flux
    for i, z in enumerate(z_range):

        # Densité de CO2 et épaisseur optique de la couche
        n_CO2 = air_number_density(z) * CO2_fraction
        optical_thickness[i, :] = cross_section_CO2(lambda_range) * n_CO2 * delta_z

        # Émissivité de la couche et émission propre (par direction)
        epsilon     = 1 - np.exp(-optical_thickness[i, :])
        emitted_flux = epsilon * np.pi * planck_function(lambda_range, temperature(z)) * delta_lambda

        # P_k,haut = P_(k-1),haut - absorbé + émis = (1 - eps) * flux_in + émis
        upward_flux[i, :] = (1 - epsilon) * flux_in + emitted_flux

        flux_in = upward_flux[i, :]

    print(f"Total outgoing flux at the top of the atmosphere: {upward_flux[-1, :].sum():.2f} W/m^2")

    # -------- PASSE DESCENDANTE : sommet -> surface --------
    # Pas de rayonnement IR entrant depuis l'espace
    flux_in = np.zeros(len(lambda_range))
    for i in range(len(z_range) - 1, -1, -1):
        z = z_range[i]

        epsilon     = 1 - np.exp(-optical_thickness[i, :])
        emitted_flux = epsilon * np.pi * planck_function(lambda_range, temperature(z)) * delta_lambda

        # P_k,bas = P_(k+1),bas - absorbé + émis = (1 - eps) * flux_in + émis
        downward_flux[i, :] = (1 - epsilon) * flux_in + emitted_flux

        flux_in = downward_flux[i, :]

    print(f"Total downwelling flux at the surface (contre-rayonnement): {downward_flux[0, :].sum():.2f} W/m^2")

    return lambda_range, z_range, upward_flux, downward_flux, optical_thickness

# ----------------------------------------------------------------------------------------------------------------------

# MAIN

CO2_fraction = 280e-6
lambda_range, z_range, upward_flux, downward_flux, optical_thickness = simulate_radiative_transfer(CO2_fraction)
CO2_fraction *= 4

lambda_range, z_range, upward_flux2, downward_flux2, optical_thickness2 = simulate_radiative_transfer(CO2_fraction)

delta_lambda = lambda_range[1] - lambda_range[0]

# === Spectre au sommet de l'atmosphère (flux MONTANT) ===
plt.figure(figsize=(14, 9))
# Corps noir à la température de surface et à 216 K (haute atmosphère)
plt.plot(1e6 * lambda_range, np.pi * planck_function(lambda_range, temperature(0)) / 1e6, '--k')
plt.plot(1e6 * lambda_range, np.pi * planck_function(lambda_range, 216) / 1e6, '--k')

plt.plot(1e6 * lambda_range, upward_flux[-1, :]  / delta_lambda / 1e6, '-g', label='280 ppm')
plt.plot(1e6 * lambda_range, upward_flux2[-1, :] / delta_lambda / 1e6, '-r', label='560 ppm')# à changer
plt.fill_between(1e6 * lambda_range,
                 upward_flux[-1, :]  / delta_lambda / 1e6,
                 upward_flux2[-1, :] / delta_lambda / 1e6,
                 color='yellow', alpha=0.9)
plt.xlabel("Longueur d'onde (μm)")
plt.ylabel("Luminance spectrale (W/m²/μm/sr)")
plt.title("Flux montant au sommet de l'atmosphère")
plt.xlim(0, 50)
plt.ylim(0, 30)
plt.legend()
plt.grid(True)

# === Contre-rayonnement atmosphérique au sol (flux DESCENDANT) - nouveau ===
plt.figure(figsize=(14, 9))
plt.plot(1e6 * lambda_range, np.pi * planck_function(lambda_range, temperature(0)) / 1e6, '--k')
plt.plot(1e6 * lambda_range, downward_flux[0, :]  / delta_lambda / 1e6, '-g', label='280 ppm')
plt.plot(1e6 * lambda_range, downward_flux2[0, :] / delta_lambda / 1e6, '-r', label='560 ppm') # POIUR LA L2GENDE à changer
plt.fill_between(1e6 * lambda_range,
                 downward_flux[0, :]  / delta_lambda / 1e6,
                 downward_flux2[0, :] / delta_lambda / 1e6,
                 color='yellow', alpha=0.9)
plt.xlabel("Longueur d'onde (μm)")
plt.ylabel("Luminance spectrale (W/m²/μm/sr)")
plt.title("Contre-rayonnement atmosphérique reçu au sol")
plt.xlim(0, 50)
plt.ylim(0, 30)
plt.legend()
plt.grid(True)

plt.show()

# Exemple d'appel de la fonction épaisseur optique (décommenter au besoin) :
# nu, lambda_um, tau, tau_eff = epaisseur_optique("CO2", 0, 2000, plot=True, verbose=True)
# ----------------------------------------------------------------------------------------------------------------------
