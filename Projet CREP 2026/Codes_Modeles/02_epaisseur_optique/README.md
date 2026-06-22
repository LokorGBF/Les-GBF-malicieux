# Fichier README de l'épaisseur optique

Récapitulatif du dossier `02_epaisseur_optique`

## Prérequis importants

Avant d'exécuter le script `calcul_epaisseur_optique_final.py`, veuillez respecter les consignes suivantes :
- **Installation de HAPI :** Assurez-vous d'avoir exécuté au moins une fois le fichier `installation_hapi.py` sur votre machine.
- **Dépendance :** Le code `atmosphere_isotherme.py` doit obligatoirement se trouver dans le même dossier.
- **Version de NumPy :** Il est nécessaire d'utiliser une **ancienne version** de NumPy pour garantir la compatibilité du code. Pour l'installer, exécutez la commande suivante dans votre terminal :
  ```bash
  pip install "numpy<2.0"
  ```

## Programme __installation_hapi__

- Permet de charger la base de données spectrale **HITRAN**.

## Programme __calcul_epaisseur_optique_final__

- Permet de calculer **l'épaisseur optique** en fonction des gaz, de l'épaisseur de la couche atmosphérique et de la longueur d'onde.

## Programme __atmosphere_isotherme__

- Permet de calculer la **fraction molaire**, la **pression partielle** et la **concentration** d'un gaz en fonction de l'altitude, à partir d'un modèle d'atmosphère isotherme.

---

## Documentation et Ressources

- **`equation_pression_atmosphere.pdf`** : Document détaillant la modélisation mathématique et les équations régissant la pression atmosphérique en fonction de l'altitude.
- **`epaisseur_optique.pdf` / `Epaisseur_optique (5).pdf`** : Fichiers explicatifs détaillant la théorie physique de l'épaisseur optique et son application dans notre modèle de simulation.
- **`code épaiseur latex.txt`** : Fichier regroupant les formules mathématiques au format LaTeX utilisées pour les calculs d'épaisseur optique, prêtes à être intégrées dans le document de synthèse final.
