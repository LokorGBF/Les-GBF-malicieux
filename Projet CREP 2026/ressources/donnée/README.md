# ⚡ ***Les-GBF-malicieux***
## **Projet CREPES 2026 : Réalisation d'un modèle de prévision climatique mondiale**

### Présentation du projet

Notre objectif est de modéliser le climat terrestre mondial au fur et à mesure du temps, en réutilisant et améliorant les modèles des années passées, notamment en y ajoutant la modélisation de la composition atmosphérique en différentes couches.

---

## Objectifs de modélisation

Notre travail s'est divisé en plusieurs phases clés :

* **Phase 1 : Maintenance et tri des données des années passées**
  * [...]
* **Phase 2 : L'impact atmosphérique**
  * Découpe de l'atmosphère en différentes couches.
  * Modélisation de l'influence de la concentration en CO₂ sur la puissance surfacique reçue.
  * Élargissement à l'influence des autres gaz atmosphériques.
* **Phase 3 : Création d'un modèle qui nous donne des valeurs en fonction des données initiales et du temps**
  * Comparaison des prédictions de notre modèle avec les mesures expérimentales.

---

## Structure du projet

```bash
Les-GBF-malicieux/
├── Codes/                 # Tous nos scripts Python et le code source de la simulation
├── Donnees/               # Fichiers de données lourds (isolés dans un dossier séparé)
├── Ressources/            # Documentation, sources, schémas et le reste des données
├── .gitignore             # Liste des fichiers/dossiers ignorés par Git (ex: .venv, .vscode)
├── synthese.pdf           # Notre document de synthèse final
└── README.md              # Ce fichier
```

---

## Installation et Utilisation

### 1. Prérequis

Assurez-vous d'avoir installé Python 3 et les bibliothèques suivantes :
* `numpy`
* `path`
* `matplotlib`
* `scipy`
* `tqdm`
* `pandas`
* `xarray`
* `netCDF4`
* `geopandas`
* `shapely`

### 2. Utiliser un environnement virtuel (Recommandé)

Si l'installation des bibliothèques rencontre des problèmes (comme des conflits de versions avec d'autres projets sur votre machine), il est fortement conseillé de créer un **environnement virtuel**. Cela permet d'isoler le projet.


1. **Lancer le fichier principale :**

   ```bash
   python [Nom_du_script_principal_si_y'en_a_un_jsp_peut_etre_pas_on_verra.py]

