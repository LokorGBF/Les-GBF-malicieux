# Fichier README des ressources

Récapitulatif du dossier `data`

## Dossier __data__

- `ne_10m_coastline.shp`, `ne_10m_coastline.dbf` et `ne_10m_coastline.shx` : ensemble de fichiers cartographiques (format Shapefile) permettant de **délimiter les côtes et les continents** sur nos modélisations. Ces fichiers sont interdépendants :
  - `.shp` : stocke la **géométrie** exacte (les coordonnées GPS des tracés côtiers de la Terre).
  - `.dbf` : stocke les **attributs et données** associés sous forme de base de données.
  - `.shx` : sert d'**index** pour faire le lien rapidement entre la géométrie et la base de données lors de l'exécution du code.
