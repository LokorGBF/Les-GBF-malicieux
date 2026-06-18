import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hapi import *

# =====================================================
# 1. Paramètres spectraux HITRAN
# =====================================================

# Bande du CO2 autour de 15 µm
# 15 µm ≈ 667 cm^-1
NU_MIN = 600.0   # cm^-1
NU_MAX = 750.0   # cm^-1
DNU = 0.05       # cm^-1

# Température et pression utilisées pour calculer les raies
T_atm = 288.0    # K
P_atm = 1.0      # atm
P0 = 101325.0    # Pa

# =====================================================
# 2. Paramètres de la couche d'atmosphère
# =====================================================

L_m = 2000.0       # hauteur de la couche : 2 km
xi_CO2 = 420e-6    # concentration CO2 : 420 ppm

# Constantes
g = 9.81
R = 8.314
kB = 1.380649e-23

# Masse molaire moyenne de l'air
# CO2 supposé bien mélangé dans l'air
M_air = 0.02897  # kg/mol

# =====================================================
# 3. Densité moléculaire n_CO2(z)
# =====================================================

def n_CO2_z(z):
    """
    Calcule la densité moléculaire du CO2 à l'altitude z.

    z : altitude en m
    retourne n_CO2(z) en molécules/m^3
    """
    P_z = P0 * np.exp(-(M_air * g * z) / (R * T_atm))
    return xi_CO2 * P_z / (kB * T_atm)

# Grille d'altitude entre 0 et 2 km
z = np.linspace(0, L_m, 1000)

# Densité de CO2 sur la couche
n_CO2 = n_CO2_z(z)

# Intégrale de n_CO2(z) dz
# Unité : molécules/m^2
colonne_CO2 = np.trapz(n_CO2, z)

print("\nColonne moléculaire de CO2 sur 2 km :")
print(f"∫ n_CO2(z) dz = {colonne_CO2:.3e} molécules/m²")

# =====================================================
# 4. Récupération des données HITRAN du CO2
# =====================================================

db_begin("hitran_data")

# CO2 :
# 2 = molécule CO2 dans HITRAN
# 1 = isotopologue principal
fetch("CO2_15um", 2, 1, NU_MIN, NU_MAX)

# =====================================================
# 5. Calcul de la section efficace sigma_lambda
# =====================================================

nu, sigma_cm2 = absorptionCoefficient_Voigt(
    Components=((2, 1),),
    SourceTables="CO2_15um",
    Environment={"p": P_atm, "T": T_atm},
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

# =====================================================
# 6. Calcul de tau_lambda sur toute la bande
# =====================================================

# Formule :
# tau_lambda = sigma_lambda * ∫ n_CO2(z) dz
tau = sigma_m2 * colonne_CO2

# Transmittance spectrale :
# T_lambda = exp(-tau_lambda)
# Pour éviter les problèmes numériques avec exp(-8000),
# on force à 0 quand tau est trop grand.
transmittance = np.where(tau > 700, 0.0, np.exp(-tau))

# Absorbance spectrale :
# A_lambda = 1 - T_lambda
absorbance = 1.0 - transmittance

# =====================================================
# 7. Moyenne sur toute la bande d'absorption
# =====================================================

# Température d'émission thermique terrestre
T_emit = 288.0  # K

# Constante c2 = h c / kB en cm.K
c2 = 1.438776877

# Poids de Planck en nombre d'onde
# B_nu ~ nu^3 / (exp(c2*nu/T)-1)
poids_planck = nu**3 / np.expm1(c2 * nu / T_emit)

# Transmittance moyenne pondérée sur la bande
T_bande = np.trapz(poids_planck * transmittance, nu) / np.trapz(poids_planck, nu)

# Absorbance moyenne pondérée sur la bande
A_bande = np.trapz(poids_planck * absorbance, nu) / np.trapz(poids_planck, nu)

# Épaisseur optique effective associée
# T_bande = exp(-tau_eff)
if T_bande > 0:
    tau_eff = -np.log(T_bande)
else:
    tau_eff = np.inf

print("\nRésultats moyens sur toute la bande 600-750 cm^-1 :")
print(f"Transmittance moyenne T_bande = {T_bande:.4e}")
print(f"Absorbance moyenne A_bande = {A_bande:.4e}")
print(f"Épaisseur optique effective tau_eff = {tau_eff:.4e}")

# =====================================================
# 8. Valeur maximale seulement pour information
# =====================================================

i_max = np.argmax(sigma_m2)

print("\nMaximum de sigma dans la bande, seulement pour information :")
print(f"Nombre d'onde = {nu[i_max]:.2f} cm^-1")
print(f"Longueur d'onde = {lambda_um[i_max]:.2f} µm")
print(f"sigma_max = {sigma_cm2[i_max]:.3e} cm²/molécule")
print(f"sigma_max = {sigma_m2[i_max]:.3e} m²/molécule")
print(f"tau_max = {tau[i_max]:.3e}")
print(f"Transmittance au maximum = {transmittance[i_max]:.3e}")

# =====================================================
# 9. Graphiques
# =====================================================

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
plt.title("Épaisseur optique du CO₂ sur 2 km")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(lambda_um, transmittance)
plt.gca().invert_xaxis()
plt.xlabel("Longueur d'onde λ (µm)")
plt.ylabel("Transmittance Tλ")
plt.title("Transmittance spectrale du CO₂ sur 2 km")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(lambda_um, absorbance)
plt.gca().invert_xaxis()
plt.xlabel("Longueur d'onde λ (µm)")
plt.ylabel("Absorbance Aλ")
plt.title("Absorbance spectrale du CO₂ sur 2 km")
plt.grid(True)
plt.show()

# =====================================================
# 10. Sauvegarde des résultats
# =====================================================

df = pd.DataFrame({
    "nombre_onde_cm-1": nu,
    "longueur_onde_um": lambda_um,
    "sigma_cm2_par_molecule": sigma_cm2,
    "sigma_m2_par_molecule": sigma_m2,
    "tau_lambda_sur_2km": tau,
    "transmittance_lambda_sur_2km": transmittance,
    "absorbance_lambda_sur_2km": absorbance,
    "poids_planck": poids_planck
})

df.to_csv("bande_CO2_2km_HITRAN.csv", index=False)

print("\nFichier créé : bande_CO2_2km_HITRAN.csv")