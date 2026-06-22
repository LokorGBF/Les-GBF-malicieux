# Fichier README du dossier `src`

Récapitulatif du dossier `src`

Ce dossier contient l'ensemble des **fichiers sources principaux** utilisés pour faire fonctionner le modèle climatique.  
Chaque fichier correspond à une partie précise du modèle : données, grille terrestre, flux solaire, albédo, effet de serre, bilan d’énergie, évolution de la température ou affichage des résultats.

## Fichiers du dossier `src`

- `__init__.py` : permet au dossier `src` d'être reconnu comme un module Python importable dans le reste du projet.

- `albedo.py` : prépare les **valeurs d’albédo terrestre** utilisées dans le modèle, soit à partir de données réelles, soit à partir d’une valeur constante.

- `clouds.py` : prépare les **valeurs d’albédo des nuages** utilisées dans le modèle, soit à partir de données réelles, soit à partir d’une valeur constante.

- `config.py` : permet de **choisir les paramètres de la simulation** ; c’est le fichier à modifier pour changer la taille de la grille, la durée, le pas de temps, la température initiale ou les options activées.

- `constants.py` : contient toutes les **constantes physiques** utilisées dans les calculs (constante solaire, constante de Stefan-Boltzmann, rayon de la Terre...).

- `convection.py` : calcule les **échanges d’énergie par convection** entre la surface terrestre et l’atmosphère. Cette partie n'a pas été implémentée cette année.

- `data_loader.py` : charge et prépare les **données externes** nécessaires au modèle, comme l’albédo, l’humidité du sol, les données CERES ou les cartes géographiques.

- `energy_balance.py` : calcule le **bilan d’énergie global** reçu ou perdu par chaque case du maillage terrestre.

- `evaporation.py` : calcule la **perte d’énergie liée à l’évaporation**, en fonction des zones géographiques et de l’énergie solaire reçue.

- `greenhouse.py` : modélise l’**effet de serre** en calculant le rayonnement infrarouge renvoyé vers la surface par l’atmosphère. Ce fichier contient un place_holder pour le C02 (non implémenté cette année).

- `grid.py` : crée la **grille latitude-longitude** utilisée pour découper la surface de la Terre.

- `heat_capacity.py` : calcule ou prépare la **capacité thermique du sol** (à partir des données d’humidité).

- `infrared.py` : calcule le **rayonnement infrarouge émis par la surface terrestre** en fonction de sa température.

- `model.py` : contient la **boucle principale de simulation** qui fait évoluer la température de la Terre au cours du temps.

- `plots_2d.py` : génère les **cartes et graphiques** permettant de visualiser les résultats de la simulation. 

- `solar_flux.py` : calcule le **flux solaire absorbé** par la surface terrestre en tenant compte de la position du Soleil, de l’albédo du sol et des nuages.

- `time_solar.py` : calcule les **paramètres solaires temporels**, comme le jour de l’année, l’heure solaire locale, la déclinaison solaire et l’angle d’incidence du Soleil.

## Dossier `__pycache__`

- `__pycache__` : dossier généré automatiquement par Python lors de l’exécution du code, il ne fait pas partie du code source principal.
