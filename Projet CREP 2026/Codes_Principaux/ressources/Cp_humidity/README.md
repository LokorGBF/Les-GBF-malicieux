# Fichier README des ressources

Récapitulatif du dossier `Cp_humidity`

## Fichier __average_rzsm_tout.csv__

- `average_rzsm_tout.csv` : **découpe la Terre en petites surfaces** de 0,25° de latitude et de longitude et y associe une **valeur de l'humidité**.

## Fichier __ZZ_cp.py__

- `ZZ_cp.py` : script Python qui calcule la valeur de la **capacité thermique du sol** en fonction de la valeur de l'humidité (lue dans le fichier CSV). Cette donnée est ensuite utilisée dans le modèle pour déterminer comment la surface terrestre absorbe ou évacue la puissance thermique.