\# Fichier README de l'épaisseur optique



Récapitulatif du dossier 02\_epaisseur\_optique



&#x20;

\## Programme installation\_hapi



\-Permet de charger la base de donnée HITRAN



\## Programme calcul\_epaisseur\_optique\_final



\-Permet de donner l'épaisseur optique en fonction des gaz, de la taille de la couche, de la longueur d'onde



Avant d'exécuter "calcul\_epaisseur\_optique\_final.py", assurez-vous d'avoir
exécuté au moins une fois "installation\_hapi.py" sur votre PC.

Il faut avoir le code "atmosphere\_isotherme.py" dans le même dossier.

Il vous faudra aussi avoir une ANCIENNE VERSION de Numpy, pour être sûr
d'avoir la version adéquate de Numpy, exécuter la commande :  pip install "numpy<2.0"

Le code prend en entrée le type de gaz et les coordoonées z_min et z_max de la couche souhaitée.
Il renvoie le tau_effectif et un graphique de tau en fonction de lambda.
\# Fichier README du modèle atmosphèrique



\## Programme atmosphere\_isotherme



\-Permet de donner la fraction molaire, la pression partielle et la concentration d'un gaz en fonction de l'altitude à partir d'un modèle d'atmosphère isotherme

