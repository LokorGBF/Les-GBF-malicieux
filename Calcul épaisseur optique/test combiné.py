import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import matplotlib.pyplot as plt
from hapi import *

# =====================================================
# Coller ici tout le code atmosphérique
# (constantes, LAYERS, fonctions, etc.)
# =====================================================

# [Le code atmosphérique complet est importé tel quel]
# Pour l'exemple, on suppose qu'il est dans atmo.py :
from atmosphere_isotherme import gas_info_at_altitude, pressure_air, LAYERS, K_B, T0

# =====================================================
# Paramètres spectraux fixes
# =====================================================

# Table de correspondance gaz -> (molecule_id, isotopologue) HITRAN
HITRAN_IDS = {
    "CO2" : (2, 1),
    "H2O" : (1, 1),
    "O3"  : (3, 1),
    "N2O" : (4, 1),
    "CH4" : (6, 1),
    "O2"  : (7, 1),
}

# Bandes spectrales par gaz (cm^-1)
SPECTRAL_BANDS = {
    "CO2" : (600.0,  750.0),   # bande 15 µm
    "H2O" : (200.0,  800.0),
    "O3"  : (980.0, 1100.0),   # bande 9.6 µm
    "N2O" : (500.0,  800.0),
    "CH4" : (1200.0, 1400.0),
    "O2"  : (7500.0, 8000.0),
}

DNU    = 0.05    # pas spectral cm^-1
K_B    = 1.380_649e-23
T0     = 288.0   # température isotherme du modèle atmosphérique

# =====================================================
# 1. Demande des paramètres
# =====================================================

print("=" * 55)
print("  Calcul de l'épaisseur optique (bandes IR)")
print("=" * 55)
print(f"Gaz disponibles : {', '.join(HITRAN_IDS.keys())}")

gas      = input("\nGaz [ex: CO2] : ").strip().upper()
z_target = float(input("Altitude maximale de la couche [m, ex: 2000] : "))

if gas not in HITRAN_IDS:
    raise ValueError(f"Gaz {gas!r} non supporté. Choisir parmi : {list(HITRAN_IDS.keys())}")

mol_id, iso_id = HITRAN_IDS[gas]
NU_MIN, NU_MAX = SPECTRAL_BANDS[gas]

# =====================================================
# 2. Profil atmosphérique via le code atmosphérique
#    Grille d'altitude entre 0 et z_target
# =====================================================

N_z    = 500
Z_grid = np.linspace(0.0, z_target, N_z)

# Pression totale à chaque altitude (Pa)
P_grid = np.array([pressure_air(z) for z in Z_grid])

# xi et densité numérique du gaz à chaque altitude
xi_grid = np.zeros(N_z)
n_grid  = np.zeros(N_z)   # molécules/m³

for i, z in enumerate(Z_grid):
    info       = gas_info_at_altitude(z, gas)
    xi_grid[i] = info["xi"]
    n_grid[i]  = info["number_density_molecules_m3"]

# Pression et xi moyens sur la couche (pour HAPI)
P_mean     = np.trapz(P_grid, Z_grid) / z_target
xi_mean    = np.trapz(xi_grid, Z_grid) / z_target
P_mean_atm = P_mean / 101_325.0

# Colonne moléculaire intégrée : ∫ n(z) dz  [molécules/m²]
colonne = np.trapz(n_grid, Z_grid)

print(f"\n→ Gaz            : {gas}")
print(f"→ Couche         : 0 → {z_target:.0f} m")
print(f"→ xi moyen       : {xi_mean:.4e}")
print(f"→ P moyenne      : {P_mean:.2f} Pa  ({P_mean_atm:.5f} atm)")
print(f"→ Colonne        : {colonne:.4e} molécules/m²")

# =====================================================
# 3. Données HITRAN et section efficace
# =====================================================

table_name = f"{gas}_band"
db_begin("hitran_data")
fetch(table_name, mol_id, iso_id, NU_MIN, NU_MAX)

nu, sigma_cm2 = absorptionCoefficient_Voigt(
    Components   = ((mol_id, iso_id),),
    SourceTables = table_name,
    Environment  = {"p": P_mean_atm, "T": T0},
    OmegaRange   = [NU_MIN, NU_MAX],
    OmegaStep    = DNU,
    HITRAN_units = True,
    GammaL       = "gamma_air"
)

nu        = np.array(nu)
sigma_cm2 = np.array(sigma_cm2)
sigma_m2  = sigma_cm2 * 1e-4       # cm²/molécule → m²/molécule
lambda_um = 1e4 / nu               # cm^-1 → µm

# =====================================================
# 4. Épaisseur optique spectrale
#    tau(nu) = sigma(nu) * colonne
# =====================================================

tau = sigma_m2 * colonne

# =====================================================
# 5. Épaisseur optique effective (moyenne Planck)
# =====================================================

transmittance = np.where(tau > 700, 0.0, np.exp(-tau))

c2           = 1.438_776_877        # cm·K
poids_planck = nu**3 / np.expm1(c2 * nu / T0)

T_bande  = np.trapz(poids_planck * transmittance, nu) / np.trapz(poids_planck, nu)
tau_eff  = -np.log(T_bande) if T_bande > 0 else np.inf

# =====================================================
# 6. Résultats
# =====================================================

print("\n" + "=" * 55)
print("  RÉSULTATS")
print("=" * 55)
print(f"Gaz                    : {gas}")
print(f"Bande spectrale        : {NU_MIN}–{NU_MAX} cm⁻¹")
print(f"Couche                 : 0 → {z_target:.0f} m")
print(f"xi moyen               : {xi_mean:.4e}")
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

# =====================================================
# 7. Graphique unique : épaisseur optique
# =====================================================

plt.figure()
plt.semilogy(lambda_um, tau, color="darkorange")
plt.gca().invert_xaxis()
plt.xlabel("λ (µm)")
plt.ylabel("τ")
plt.title(
    f"Épaisseur optique — {gas}\n"
    f"xi = {xi_mean:.2e} | couche 0–{z_target:.0f} m | τ_eff = {tau_eff:.3f}"
)
plt.grid(True)
plt.tight_layout()
plt.show()