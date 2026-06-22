import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import matplotlib.pyplot as plt
from hapi import *

from atmosphere_isotherme import gas_info_at_altitude


#Profil atmosphérique plus précis : T(z) et P(z)


def AtmTetP():
    # Constantes physiques
    G = 6.67e-11
    Mair = 29e-3
    R = 8.314
    Mterre = 5.97e24
    RT = 6378e3

    # Conditions au sol
    Tsol = 288.0
    Psol = 1.013e5

    # Gradient thermique ISA
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

    # Discrétisation altitude
    N = 10000
    Z = np.linspace(0, 85e3, N)
    dz = Z[1] - Z[0]

    T = np.zeros(N)
    P = np.zeros(N)

    T[0] = Tsol
    P[0] = Psol

    for i in range(N - 1):
        z = Z[i]
        g = G * Mterre / (RT + z)**2

        dT_dz = kISA(z)
        dP_dz = -(Mair * g / (R * T[i])) * P[i]

        T[i + 1] = T[i] + dT_dz * dz
        P[i + 1] = P[i] + dP_dz * dz

    return Z, T, P


#Paramètres spectraux fixes


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


#Demande des paramètres


print("=" * 55)
print("  Calcul de l'épaisseur optique (bandes IR)")
print("=" * 55)
print(f"Gaz disponibles : {', '.join(HITRAN_IDS.keys())}")

gas = input("\nGaz [ex: CO2] : ").strip().upper()
z_min = float(input("Altitude minimale de la couche [m, ex: 0] : "))
z_max = float(input("Altitude maximale de la couche [m, ex: 2000] : "))

if gas not in HITRAN_IDS:
    raise ValueError(f"Gaz {gas!r} non supporté. Choisir parmi : {list(HITRAN_IDS.keys())}")

if z_min < 0 or z_max < 0:
    raise ValueError("Les altitudes doivent être positives.")

if z_max <= z_min:
    raise ValueError("Il faut que z_max > z_min.")

# Important : ce nouveau profil n'est défini que jusqu'à 85 km
if z_max > 85_000:
    raise ValueError("Avec ce modèle ISA, l'altitude maximale est 85 000 m.")

mol_id, iso_id = HITRAN_IDS[gas]
NU_MIN, NU_MAX = SPECTRAL_BANDS[gas]


# Construction du profil atmosphérique sur [z_min, z_max]


Z_atm, T_atm, P_atm_profile = AtmTetP()

N_z = 500
Z_grid = np.linspace(z_min, z_max, N_z)
delta_z = z_max - z_min

# Interpolation de T(z) et P(z) sur la couche choisie
T_grid = np.interp(Z_grid, Z_atm, T_atm)
P_grid = np.interp(Z_grid, Z_atm, P_atm_profile)

# xi(z) du gaz via ton code atmosphérique
xi_grid = np.zeros(N_z)

for i, z in enumerate(Z_grid):
    info = gas_info_at_altitude(z, gas)
    xi_grid[i] = info["xi"]

# Densité numérique recalculée avec le NOUVEAU profil de pression
# n(z) = xi(z) * P(z) / (k_B * T(z))
n_grid = xi_grid * P_grid / (K_B * T_grid)

# Moyennes sur la couche
P_mean = np.trapezoid(P_grid, Z_grid) / delta_z
T_mean = np.trapezoid(T_grid, Z_grid) / delta_z
xi_mean = np.trapezoid(xi_grid, Z_grid) / delta_z
P_mean_atm = P_mean / 101325.0

# Colonne moléculaire
colonne = np.trapezoid(n_grid, Z_grid)

print(f"\n→ Gaz            : {gas}")
print(f"→ Couche         : {z_min:.0f} → {z_max:.0f} m")
print(f"→ xi moyen       : {xi_mean:.4e}")
print(f"→ T moyenne      : {T_mean:.2f} K")
print(f"→ P moyenne      : {P_mean:.2f} Pa ({P_mean_atm:.5f} atm)")
print(f"→ Colonne        : {colonne:.4e} molécules/m²")


# Données HITRAN et section efficace


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


#  Épaisseur optique spectrale


tau = sigma_m2 * colonne


#  Épaisseur optique effective


transmittance = np.where(tau > 700, 0.0, np.exp(-tau))

c2 = 1.438776877  # cm.K
poids_planck = nu**3 / np.expm1(c2 * nu / T_mean)

T_bande = np.trapezoid(poids_planck * transmittance, nu) / np.trapezoid(poids_planck, nu)
tau_eff = -np.log(T_bande) if T_bande > 0 else np.inf


#  Résultats


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

#  Graphique unique : épaisseur optique


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