## Programme __AtmosphereModel1.py__

- Script définissant les **modèles d'atmosphère** (profils de température, pression, et densité en fonction de l'altitude) utilisés comme base pour les simulations radiatives.

## Programme __bilan_puissance_spectral.py__

- Simule le **transfert radiatif infrarouge** dans une atmosphère multicouche.
- Permet de modéliser l'impact de la concentration en **CO₂** sur l'effet de serre et de générer les graphiques des flux lumineux (flux sortant au sommet de l'atmosphère et flux reçu au sol).

## Programme __code_atmosphere_T_et_P.py__

- Permet de calculer la **température** et la **pression** de l'atmosphère en fonction de l'altitude, en se basant sur le modèle d'atmosphère standard **ISA**.

## Fichier __Documentation_GetSolRay.docx__

- Fichier de documentation complet détaillant le fonctionnement, les paramètres et les formules mathématiques utilisés dans le calcul du rayonnement solaire.

## Programme __effet_de_serre_ébauche.py__

- Script préliminaire permettant de calculer et de dessiner le **flux émis par l'atmosphère** en fonction de la fraction molaire du **CO₂**.

## Programme __Functions.py__

- Fichier regroupant l'ensemble des **fonctions mathématiques et utilitaires** partagées, destinées à être appelées par les autres scripts du projet pour éviter la répétition de code.

## Programme __GetSolRay.py__

- Script principal permettant de calculer et de modéliser le **rayonnement solaire incident** reçu par la Terre, en prenant en compte les différents paramètres orbitaux et atmosphériques.

## Programme __VariableClass.py__

- Script orienté objet contenant les **classes et structures de données** qui regroupent les variables globales, les constantes physiques et les paramètres de configuration de la simulation.

```
