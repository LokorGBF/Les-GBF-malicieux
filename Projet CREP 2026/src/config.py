from pathlib import Path

# Configuration de la simulation

# Dossiers
ROOT_DIR = Path(__file__).resolve().parents[1]
RESSOURCES_DIR = ROOT_DIR / "ressources"
RESULTS_DIR = ROOT_DIR / "results"
SIMULATIONS_DIR = RESULTS_DIR / "simulations"
FIGURES_DIR = RESULTS_DIR / "figures"

# Grille sphérique : paramètres
N_THETA = 140
N_PHI = 280

# Temps
DAYS = 365       
DT = 1800.0   # 30 min 

# Température initiale
INITIAL_TEMPERATURE = 288.15  # 15°C

# Valeurs constantes si certains effets sont désactivés
CONSTANT_SURFACE_ALBEDO = 0.30
CONSTANT_CLOUD_ALBEDO = 0.0
CONSTANT_HEAT_CAPACITY = 2.0e6  # J.m-2.K-1

# Effets activables / désactivables
OPTIONS = {
    "solar": True,
    "surface_albedo": True,
    "cloud_albedo": True,
    "surface_infrared": True,
    "greenhouse": True,
    "evaporation": True,
    "convection": False,
    "co2": False,
}

# Sauvegarde
SAVE_RESULTS = True
RESULT_FILE = SIMULATIONS_DIR / "T_history.npy"