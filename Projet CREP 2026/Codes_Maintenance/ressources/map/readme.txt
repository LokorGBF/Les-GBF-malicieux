# Fichier README des ressources

Récapitulatif du dossier `map`

## Dossier _map_

- `ne_110m_admin_0_countries.*` : ensemble de fichiers cartographiques (format Shapefile) permettant
 d'obtenir le **nom des pays et des continents** pour chaque coordonnée terrestre (utile, par exemple, pour attribuer une **constante de chaleur** spécifique à chaque région). Cet ensemble est composé de plusieurs fichiers interdépendants :
  - `.shp` : contient la **géométrie** (le tracé exact des frontières de tous les pays de la Terre).
  - `.dbf` : contient la **base de données** des attributs (les noms des pays, continents, codes 
  géographiques).
  - `.shx` : sert d'**index** de position pour lier rapidement la géométrie aux données de la base.
  - `.prj` : définit le **système de coordonnées** et la projection cartographique de la carte.
  - `.cpg` : précise l'**encodage** des caractères (pour lire correctement les accents dans les noms 
  de pays).
