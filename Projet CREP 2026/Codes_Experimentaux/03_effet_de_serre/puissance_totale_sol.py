import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# Récupération des valeurs de température et de pression qui corresponde à notre modèle 
import code_atmosphere_T_et_P as T_P

T_profil,P_profil,Z_grille = T_P.AtmTetP()

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

def pressure(z):
    P0 = 101325     # Pressure at sea level in Pa
    H = 8500        # Scale height in m
    return P0 * np.exp(-z / H)

def pressure_variable(z): # nouvelle manière d'avoir la pression selon le code_atmosphere_T_et_P.py
    return np.interp(z,Z_grille,P_profil)

def temperature_var(z): # nouvelle manière d'avoir la température selon le code_atmosphere_T_et_P.py
   
    return np.interp(z,Z_grille,T_profil)

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
    
    return temperature_var(z)

def air_number_density(z):
    kB = 1.380649e-23  # Boltzmann's constant, J/K
    return pressure_variable(z) / (kB * temperature(z))

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

 

    return lambda_range, z_range, upward_flux, downward_flux, optical_thickness

# ----------------------------------------------------------------------------------------------------------------------

# ======================================================================================================================
# MAIN : ÉVOLUTION DU FLUX REÇU AU SOL EN FONCTION DE LA CONCENTRATION EN CO2
# ======================================================================================================================

#  Configuration des paramètres physiques de base
CO2_fraction = 280e-6            # Concentration de référence en CO2 (280 ppm = ère préindustrielle)
puissance_recu_soleil = 1361 / 4 # Flux solaire moyen absorbé par unité de surface sur la sphère terrestre (W/m²)

# Définition des scénarios climatiques (facteurs multiplicateurs de la concentration de base)
factors = np.arange(0.5, 2, 0.5)
flux_sol_selon_frCO2 = []        # Liste pour accumuler le contre-rayonnement infrarouge reçu au sol

#  Boucle de simulation du transfert radiatif
for factor in factors:
    # Calcul de la concentration actuelle pour cette itération
    current_CO2 = CO2_fraction * factor
    
    # Exécution du modèle (calcul des flux montants et descendants dans l'atmosphère)
    lambda_range, z_range, upward_flux, downward_flux, optical_thickness = simulate_radiative_transfer(current_CO2)
    delta_lambda = lambda_range[1] - lambda_range[0] # Conservé si modification future de la grille spectrale
    
    # Extraction du flux infrarouge descendant qui atteint le SOL (altitude z = 0, d'où l'indice 0)
    # On somme l'énergie sur l'ensemble du spectre des longueurs d'onde
    flux_sol_selon_frCO2.append(downward_flux[0, :].sum())  

# Conversion en tableau NumPy pour permettre les opérations mathématiques directes
flux_sol_selon_frCO2 = np.array(flux_sol_selon_frCO2)

#  Calcul du bilan total reçu au sol et construction du graphique
# Flux total = Contre-rayonnement infrarouge de l'atmosphère + Rayonnement visible du Soleil
flux_total_sol = flux_sol_selon_frCO2 + puissance_recu_soleil

# Tracé de la courbe finale (Conversion de la fraction de CO2 en ppm sur l'axe X)
plt.plot(factors * CO2_fraction * 1e6, flux_total_sol, '-o', color='tab:red')
plt.xlabel("CO₂ (ppm)")
plt.ylabel("Flux reçu par le sol (W/m²)")
plt.title("Impact de la concentration en CO₂ sur le flux total absorbé au sol")
plt.grid(True)

# Affichage de la fenêtre graphique
plt.show()
# ----------------------------------------------------------------------------------------------------------------------