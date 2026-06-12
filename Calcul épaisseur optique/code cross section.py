import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hapi import *

# =========================
# 1. Paramètres spectraux
# =========================

# Bande du CO2 autour de 15 µm
# 15 µm ≈ 667 cm^-1
NU_MIN = 600.0   # cm^-1
NU_MAX = 750.0   # cm^-1
DNU = 0.05       # cm^-1

# Température et pression utilisées pour le profil de raie
# Elles influencent la largeur des raies spectrales
T = 288.0        # K
P_atm = 1.0      # atm

# =========================
# 2. Téléchargement des données HITRAN
# =========================

db_begin("hitran_data")

# CO2 :
# 2 = molécule CO2 dans HITRAN
# 1 = isotopologue principal
fetch("CO2_15um", 2, 1, NU_MIN, NU_MAX)

# =========================
# 3. Calcul de la section efficace sigma
# =========================

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
# 4. Valeur maximale de sigma
# =========================

i_max = np.argmax(sigma_m2)

print("\nSection efficace maximale du CO2 dans la bande 600-750 cm^-1 :")
print(f"Nombre d'onde = {nu[i_max]:.2f} cm^-1")
print(f"Longueur d'onde = {lambda_um[i_max]:.2f} µm")
print(f"sigma = {sigma_cm2[i_max]:.3e} cm²/molécule")
print(f"sigma = {sigma_m2[i_max]:.3e} m²/molécule")

# =========================
# 5. Graphique de sigma
# =========================

plt.figure()
plt.semilogy(lambda_um, sigma_m2)
plt.gca().invert_xaxis()
plt.xlabel("Longueur d'onde λ (µm)")
plt.ylabel("Section efficace σ (m²/molécule)")
plt.title("Section efficace d'absorption du CO₂ autour de 15 µm")
plt.grid(True)
plt.show()

# =========================
# 6. Sauvegarde des résultats
# =========================

df = pd.DataFrame({
    "nombre_onde_cm-1": nu,
    "longueur_onde_um": lambda_um,
    "sigma_cm2_par_molecule": sigma_cm2,
    "sigma_m2_par_molecule": sigma_m2
})

df.to_csv("cross_section_CO2_HITRAN.csv", index=False)

print("\nFichier créé : cross_section_CO2_HITRAN.csv")