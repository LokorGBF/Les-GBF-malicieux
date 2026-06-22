# Fichier README des ressources

Récapitulatif du dossier `albedo`

## Dossier _albedo_

- `albedo01.csv` ... `albedo12.csv` : correspondent à **janvier ... decembre**. Chaque fichier a sur la première ligne l'ensemble des latitudes et sur la première colonne l'ensemble des longitudes. L'intersection correspond à **l'albedo moyen de chaque mois en 2023** de cette coordonnée GPS. Cette base de données a été créée grâce aux fichiers `construcion_csv.py` et `remplissage_csv.py`.

- `CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc` : fichier de données satellitaires brutes au format **NetCDF**. Il contient les mesures radiatives de la Terre (nécessaires pour l'albédo) et nécessite l'installation de bibliothèques Python spécifiques (comme **`netCDF4`** et **`scipy`**) pour pouvoir être lu et exploité.