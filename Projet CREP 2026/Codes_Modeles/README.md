# Fichier README des codes des modèles

<details>
<summary>Récapitulatif du dossier <code>Codes_Modeles</code></summary>

Le dossier `Codes_Modeles` regroupe les principaux programmes utilisés pour construire les modèles physiques du projet CREP 2026.

Ces codes permettent notamment de :

- modéliser l’évolution de la température et de la pression atmosphérique avec l’altitude ;
- calculer la composition de l’atmosphère en fonction de l’altitude ;
- estimer l’épaisseur optique de certaines couches atmosphériques ;
- étudier l’effet de serre à partir de la concentration de certains gaz ;
- modéliser le rayonnement solaire reçu par la Terre.

</details>

---

Le dossier est organisé en plusieurs sous-dossiers, chacun correspondant à une partie du modèle climatique global.


<details>
<summary> 01_modeles_atmospheriques</summary>

Ce dossier contient les programmes de base permettant de décrire l’atmosphère en fonction de l’altitude.

Il sert principalement à calculer les grandeurs atmosphériques fondamentales comme la température, la pression, la fraction molaire des gaz, la pression partielle et la concentration des espèces chimiques.

Ces données sont ensuite réutilisées dans les autres parties du projet, notamment pour le calcul de l’épaisseur optique et l’étude de l’effet de serre.

</details>

---

<details>
<summary>02_epaisseur_optique</summary>

Ce dossier contient les codes permettant de calculer l’**épaisseur optique** de l’atmosphère.

L’épaisseur optique mesure la capacité d’une couche atmosphérique à absorber ou atténuer un rayonnement.  
Dans ce projet, elle sert à quantifier l’effet des gaz atmosphériques sur le rayonnement infrarouge ou solaire.

Plus l’épaisseur optique est élevée, plus la couche atmosphérique absorbe fortement le rayonnement.

</details>

---

<details>
<summary>Prérequis importants</summary>

Avant d’exécuter le script `calcul_epaisseur_optique_final.py`, il faut respecter les consignes suivantes :

- exécuter au moins une fois le fichier `installation_hapi.py` ;
- vérifier que le fichier `atmosphere_isotherme.py` est bien dans le même dossier ;
- utiliser une version compatible de NumPy.

Pour installer une ancienne version de NumPy compatible avec le code, utiliser la commande suivante :

```bash
pip install "numpy<2.0"
</details>

---

<details>
<summary>03_effet_de_serre</summary>

Ce dossier regroupe les programmes utilisés pour étudier l’effet de serre.

L’objectif est de comprendre comment certains gaz atmosphériques, en particulier le CO2, influencent le flux radiatif émis par l’atmosphère et reçu par le sol.

Ce dossier permet donc de faire le lien entre la composition atmosphérique et le bilan énergétique de la Terre.

</details>

---

<details>
<summary>04_rayonnement_solaire</summary>

Ce dossier contient les programmes liés au rayonnement solaire.

Il sert à modéliser l’énergie solaire reçue par la Terre et à étudier comment cette énergie est répartie, absorbée ou transmise dans le système Terre-atmosphère.

Cette partie est complémentaire de l’étude de l’effet de serre : le rayonnement solaire correspond à l’énergie entrante, tandis que le rayonnement infrarouge terrestre correspond à une partie de l’énergie sortante.

</details>



