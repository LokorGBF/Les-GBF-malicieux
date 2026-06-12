import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hapi import *

# =========================
# 1. Paramètres spectraux HITRAN
# =========================

# Bande du CO2 autour de 15 µm
# 15 µm ≈ 667 cm^-1
NU_MIN = 600.0   # cm^-1
NU_MAX = 750.0   # cm^-1
DNU = 0.05       # cm^-1

# Température et pression de la couche atmosphérique
T = 288.0        # K
P_atm = 1.0      # atm
P0 = 101325.0    # Pa

# =========================
# 2. Paramètres de la couche atmosphérique
# =========================

L_m = 2000.0     # hauteur de la couche : 2 km

# Concentration du CO2
xi_CO2 = 420e-6  # 420 ppm

# Constantes
g = 9.81
R = 8.314
kB = 1.380649e-23

# Masse molaire moyenne de l'air
# On l'utilise car le CO2 est supposé bien mélangé dans l'air.
M_air = 0.02897  # kg/mol

# Pour retrouver exactement ton ancien modèle, remplace M_air par M_CO2.
# M_CO2 = 0.044  # kg/mol

# =========================
# 3. Fonction n_CO2(z)
# =========================

def n_CO2_z(z):
    """
    Densité moléculaire du CO2 à l'altitude z.

    Retourne n_CO2(z) en molécules/m^3.
    """
    P_z = P0 * np.exp(-(M_air * g * z) / (R * T))
    return xi_CO2 * P_z / (kB * T)

# Grille d'altitude de 0 à 2 km
z = np.linspace(0, L_m, 1000)

# Densité moléculaire du CO2 sur la couche
n_CO2 = n_CO2_z(z)

# Intégrale de n_CO2(z) dz
# Unité : molécules/m^2
colonne_CO2 = np.trapz(n_CO2, z)

print("\nColonne moléculaire de CO2 sur 2 km :")
print(f"∫ n_CO2(z) dz = {colonne_CO2:.3e} molécules/m²")

# =========================
# 4. Récupération de sigma avec HITRAN
# =========================

db_begin("hitran_data")

# CO2 :
# 2 = molécule CO2 dans HITRAN
# 1 = isotopologue principal
fetch("CO2_15um", 2, 1, NU_MIN, NU_MAX)

# Calcul de la section efficace avec profil de Voigt
nu, sigma_cm2 = absorptionCoefficient_Voigt(
    Components=((2, 1),),
    SourceTables="CO2_15um",
    Environment={"p": P_atm, "T": T},
    OmegaRange=[NU_MIN, NU_MAX],
    OmegaStep=DNU,
    HITRAN_units=True,
    GammaL="gamma_air"
)

nu = np.array(nu)
sigma_cm2 = np.array(sigma_cm2)

# Conversion cm²/molécule -> m²/molécule
sigma_m2 = sigma_cm2 * 1e-4

# Conversion nombre d'onde -> longueur d'onde
lambda_um = 1e4 / nu

# =========================
# 5. Calcul de l'épaisseur optique
# =========================

# Formule : tau_lambda = sigma_lambda * ∫ n_CO2(z) dz
tau = sigma_m2 * colonne_CO2

# Transmittance associée
transmittance = np.exp(-tau)

# =========================
# 6. Valeurs maximales
# =========================

i_max = np.argmax(sigma_m2)

print("\nMaximum de section efficace dans la bande étudiée :")
print(f"Nombre d'onde = {nu[i_max]:.2f} cm^-1")
print(f"Longueur d'onde = {lambda_um[i_max]:.2f} µm")
print(f"sigma = {sigma_cm2[i_max]:.3e} cm²/molécule")
print(f"sigma = {sigma_m2[i_max]:.3e} m²/molécule")

print("\nÉpaisseur optique correspondante sur 2 km :")
print(f"tau = {tau[i_max]:.3e}")
print(f"Transmittance = {transmittance[i_max]:.3e}")

# =========================
# 7. Graphiques
# =========================

plt.figure()
plt.plot(z, n_CO2)
plt.xlabel("Altitude z (m)")
plt.ylabel("n_CO₂(z) (molécules/m³)")
plt.title("Densité moléculaire du CO₂ entre 0 et 2 km")
plt.grid(True)
plt.show()

plt.figure()
plt.semilogy(lambda_um, sigma_m2)
plt.gca().invert_xaxis()
plt.xlabel("Longueur d'onde λ (µm)")
plt.ylabel("Section efficace σ (m²/molécule)")
plt.title("Section efficace du CO₂ autour de 15 µm")
plt.grid(True)
plt.show()

plt.figure()
plt.semilogy(lambda_um, tau)
plt.gca().invert_xaxis()
plt.xlabel("Longueur d'onde λ (µm)")
plt.ylabel("Épaisseur optique τ")
plt.title("Épaisseur optique du CO₂ sur une couche de 2 km")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(lambda_um, transmittance)
plt.gca().invert_xaxis()
plt.xlabel("Longueur d'onde λ (µm)")
plt.ylabel("Transmittance")
plt.title("Transmittance du CO₂ sur une couche de 2 km")
plt.grid(True)
plt.show()

# =========================
# 8. Sauvegarde des résultats
# =========================

df = pd.DataFrame({
    "nombre_onde_cm-1": nu,
    "longueur_onde_um": lambda_um,
    "sigma_cm2_par_molecule": sigma_cm2,
    "sigma_m2_par_molecule": sigma_m2,
    "tau_sur_2km": tau,
    "transmittance_sur_2km": transmittance
})

df.to_csv("tau_CO2_2km_HITRAN.csv", index=False)

print("\nFichier créé : tau_CO2_2km_HITRAN.csv")