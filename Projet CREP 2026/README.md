# Projet CREP 2026 — Modélisation climatique mondiale

> Projet mené dans le cadre de **CREP**

---

## Objectif

Modéliser l'évolution du climat terrestre au fil du temps, en repartant des bases posées par les groupes des années précédentes. L'apport principal de cette édition est la **modélisation du transfert radiatif atmosphérique** : découpe de l'atmosphère en couches, influence des gaz à effet de serre (CO₂ et autres) sur le bilan énergétique de surface.

---

## Contenu du dépôt

### `Codes_Maintenance/`
Code de simulation finalisé, structuré pour la production de résultats.

- `src/` — modules Python du modèle principal
- `results/figures/` — sorties graphiques de la simulation
- `ressources/` — données et références propres au modèle principa
- `main.py` — point d'entrée de la simulation
- `bilan_puissance_spectral.py` — calcul du bilan de puissance spectrale

### `Codes_Modeles/`
Scripts d'exploration et de développement, organisés par thème physique.

- `01_modeles_atmospheriques/` — premiers modèles de structure verticale de l'atmosphère
- `02_epaisseur_optique/` — calcul de l'épaisseur optique des couches atmosphériques
- `03_effet_de_serre/` — modélisation de l'absorption infrarouge par les GES
- `04_rayonnement_solaire/` — traitement du rayonnement solaire incident

### `ressources/`
Données et documentation communes à l'ensemble du projet.

- `SpectresHitran/` — spectres d'absorption issus de la base de données HITRAN
- `donnée/` — données climatiques de référence (réanalyses, mesures)

### `synthese.pdf`
Code source LaTeX du document de synthèse final.

---

## Installation

### Prérequis

Python 3.10+ et les bibliothèques suivantes :

```
numpy
matplotlib
scipy
tqdm
pandas
xarray
netCDF4
geopandas
shapely
```

### Mise en place

```bash
# Cloner le dépôt
git clone https://github.com/LokorGBF/Les-GBF-malicieux.git
cd "Les-GBF-malicieux/Projet CREP 2026"

# Créer un environnement virtuel (recommandé)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# Installer les dépendances
pip install numpy matplotlib scipy tqdm pandas xarray netCDF4 geopandas shapely
```

> **Note :** `geopandas` et `netCDF4` peuvent nécessiter des dépendances système (GDAL, HDF5). En cas de problème, consulter leurs documentations officielles ou utiliser `conda`.

### Lancer la simulation

```bash
cd Codes_Principaux
python main.py
```
