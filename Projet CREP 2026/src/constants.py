import numpy as np

# Contient toutes les constantes :

# Constantes physiques
SOLAR_CONSTANT = 1361.0          # W.m-2
SIGMA = 5.670374419e-8           # W.m-2.K-4
T_ATM = 223.15                   # K
EARTH_RADIUS = 6.371e6           # m
ACTIVE_LAYER_DEPTH = 0.5         # m

# Capacité thermique / humidité du sol
RHO_WATER = 1000.0               # kg.m-3
RHO_BULK = 1300.0                # kg.m-3

CP_DRY_SOIL = 0.8                # kJ.kg-1.K-1
CP_WATER = 4.187                 # kJ.kg-1.K-1
CP_ICE = 2.09                    # kJ.kg-1.K-1

# Évaporation / chaleur latente
DELTA_HVAP = 2453000             # J.kg-1
RHO_LIQUID_WATER = 1000          # kg.m-3
SECONDS_PER_YEAR = 365.25 * 24 * 3600

# Taux moyens d'évaporation par zone, en m/an
EVAPORATION_M_PER_YEAR = {
    "Europe": 0.49,
    "North America": 0.47,
    "South America": 0.94,
    "Oceania": 0.41,
    "Africa": 0.58,
    "Asia": 0.37,
    "Océan": 1.40,
    "Antarctica": 0.0,
}