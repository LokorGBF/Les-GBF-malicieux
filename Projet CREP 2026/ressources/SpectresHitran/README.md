# Fichier README des ressources

Récapitulatif du dossier `ressources`.

Ce dossier regroupe les fichiers nécessaires aux différents modèles utilisés dans le projet : humidité du sol, albédo, découpage géographique, constantes associées aux continents, données pré-calculées, ainsi que les spectres d’absorption des gaz atmosphériques.

## Dossier _SpectreHitran_

Ce dossier contient les **spectres d’absorption infrarouge** de plusieurs gaz atmosphériques, générés à partir de la base de données **HITRAN** et affichés sous forme de graphiques.

Les graphiques montrent l’**absorbance** en fonction de la **fréquence** exprimée en `cm⁻¹`. L’axe du haut indique aussi la longueur d’onde correspondante en `µm`.

Ces fichiers servent à identifier les zones du rayonnement infrarouge terrestre qui sont absorbées par les différents gaz de l’atmosphère. Ils sont utiles pour étudier l’effet de serre, car la Terre émet principalement dans l’infrarouge.

### Paramètres utilisés dans les spectres individuels

Pour les spectres individuels :

- fraction molaire : `X = 0.01`
- température : `T = 280 K`
- pression : `P = 1 atm`
- longueur du trajet optique : `L = 1 cm`

Ces paramètres permettent de comparer les gaz dans des conditions identiques.

### Fichiers du dossier

- `spectraplot_H2O.png` : spectre d’absorption de la **vapeur d’eau**.
  - La vapeur d’eau absorbe fortement dans certaines zones de l’infrarouge, notamment aux faibles fréquences du graphique.
  - C’est un gaz très important dans l’effet de serre, car son absorption est large et intense.

- `spectraplot_CO2.png` : spectre d’absorption du **dioxyde de carbone**.
  - Le CO₂ présente une bande d’absorption importante autour de `650–700 cm⁻¹`, correspondant à une longueur d’onde proche de `15 µm`.
  - Cette zone est très importante car elle se situe dans le domaine d’émission infrarouge de la Terre.

- `spectraplot_CH4.png` : spectre d’absorption du **méthane**.
  - Le CH₄ absorbe principalement autour de `1200–1400 cm⁻¹`, soit environ `7–8 µm`.
  - Même s’il est moins abondant que le CO₂, il absorbe fortement dans certaines bandes précises.

- `spectraplot_O3.png` : spectre d’absorption de l’**ozone**.
  - L’ozone possède une bande d’absorption marquée autour de `1000–1100 cm⁻¹`, soit environ `9–10 µm`.
  - Cette zone se trouve dans la fenêtre infrarouge atmosphérique, donc l’ozone peut influencer la sortie du rayonnement terrestre vers l’espace.

- `spectraplot_N2.png` : spectre d’absorption du **diazote**.
  - Le N₂ absorbe très peu dans l’infrarouge.
  - L’échelle d’absorbance est extrêmement faible, ce qui montre que le diazote contribue très peu directement à l’effet de serre.

- `spectraplot_O2.png` : spectre d’absorption du **dioxygène**.
  - Le O₂ absorbe aussi très peu dans l’infrarouge.
  - Comme pour le N₂, son absorbance est très faible, donc il participe peu directement à l’effet de serre.

- `Spectre_complet_gazs_et_terre.png` : graphique comparatif regroupant les spectres de plusieurs gaz et le **spectre d’émission terrestre**.
  - Il permet de comparer directement les bandes d’absorption de `H2O`, `CO2`, `CH4`, `N2`, `O2` et `O3`.
  - La courbe rouge représente le **spectre d’émission de la Terre**, approximé pour une température d’environ `287 K`.
  - Ce fichier permet de voir quels gaz absorbent dans les zones où la Terre émet le plus d’énergie infrarouge.

### Utilité du dossier dans le projet

Le dossier `SpectreHitran` permet de relier les propriétés spectrales des gaz à leur rôle climatique.

Il sert notamment à :

- visualiser les bandes d’absorption infrarouge des principaux gaz atmosphériques ;
- comparer l’importance relative des gaz dans l’absorption du rayonnement terrestre ;
- repérer les zones de recouvrement entre l’émission de la Terre et l’absorption des gaz ;
- justifier le rôle majeur de certains gaz comme `H2O`, `CO2`, `CH4` et `O3` dans l’effet de serre ;
- montrer que `N2` et `O2`, bien que très abondants dans l’atmosphère, absorbent très peu directement dans l’infrarouge.

---

## Fichiers

### Rapport

- `données sourcées.pdf` : fichier qui regroupe des données utiles pour les modèles.
