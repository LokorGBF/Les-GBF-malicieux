# Fichier README des codes principaux

Récapitulatif du dossier `Codes_Principaux`

## Fichier _main.py_

- lance la **simulation climatique principale**, crée la **grille terrestre**, appelle le modèle, sauvegarde les résultats, affiche le **planisphère** (température mondiale à la fin de la simulation) et une **courbe de température** pour une localisation donnée. Afin de modifier la latitude et longitude de l'emplacement pour la courbe de température, il suffit de les modifier ici : plot_temperature_curve(T_history, grid, lat_target=48.8566, lon_target=2.3522).

## Fichier _bilan_puissance_spectral.py_

- à remplir

## Dossier _src_

- contient l’ensemble des **fichiers sources principaux** du modèle climatique.
- le fichier `config.py` est le fichier où l’on choisit les **paramètres de la simulation** ; c’est donc là qu’il faut aller pour modifier la durée, la grille, le pas de temps, la température initiale ou les options activées.

## Dossier _ressources_

- contient les **données nécessaires au fonctionnement du modèle** (valeurs d’albédo, humidité, cartes géographiques).

## Dossier _results_

- contient les **résultats générés** par les simulations : figures, cartes, fichiers de température produits lors de l’exécution du modèle.

## Dossier _figures_

- contient les **images produites par le programme**, comme les cartes de température ou les graphiques issus de la simulation.
