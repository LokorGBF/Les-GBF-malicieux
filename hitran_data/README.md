# hitran_data
 
Données d'absorption spectrale issues de la base [HITRAN](https://hitran.org/) (*High-resolution Transmission*), utilisées pour le calcul de l'épaisseur optique des gaz atmosphériques.
 
## Contenu
 
Chaque gaz est représenté par deux fichiers :
 
| Fichier | Description |
|---|---|
| `*.data` | Raies d'absorption : nombre d'onde, intensité, largeur, etc. |
| `*.header` | Métadonnées associées (unités, isotopologue, source) |
 
Gaz disponibles : **CH₄**, **CO₂**, **H₂O**, **N₂O**, **O₃**.
 
## Source
 
Données téléchargées depuis [hitran.org](https://hitran.org/). Pour mettre à jour ou étendre la sélection de gaz, utiliser l'interface en ligne et exporter au format `.data`/`.header`.
