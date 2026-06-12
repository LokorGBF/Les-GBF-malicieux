#Recupère la table astm e-490 pour avoir les données de l'irradiance solaire en fonction de la longueur d'onde
#Source : https://www.nlr.gov/grid/solar-resource/spectra-astm-e490 , National Laboratory of the Rockies
# Colone A : Longueur d'onde (µm) (119.5nm (UV-V) à 1000µm (Radio-Electrique))
# Colone B : Irradiance solaire (W/m²/µm)

from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd # Si Erreur avec avec Excel .xls, faire dans le terminal : pip install pandas xlrd
import matplotlib.pyplot as plt
from Functions import wavelength_to_rgb, rgb_255_to_mpl

# ========================= Téléchargement du fichier =========================

ASTM_FILENAME = "e490_00a_amo.xls"
ASTM_URL = "https://www.nlr.gov/media/docs/libraries/grid/e490_00a_amo.xls?sfvrsn=ce97914b_1"

# Recherche du fichier dans le dossier local du projet
def find_project_root(start: Path | None = None) -> Path:
    """
    Essaie de trouver la racine du projet Python.
    Cherche en remontant jusqu'à trouver un marqueur de projet.
    """
    if start is None:
        start = Path.cwd()

    start = start.resolve()

    markers = [
        "pyproject.toml",
        "requirements.txt",
        "setup.py",
        ".git",
    ]

    for folder in [start, *start.parents]:
        if any((folder / marker).exists() for marker in markers):
            return folder

    # Si aucun marqueur trouvé, on prend le dossier courant
    return start

def get_astm_e490_file(project_root: Path | None = None) -> Path:
    """
    Retourne le chemin vers e490_00a_amo.xls.

    Ordre :
    1) racine du projet
    2) dossier data/
    3) recherche récursive dans tout le projet
    4) téléchargement dans data/
    """
    if project_root is None:
        project_root = find_project_root()

    project_root = project_root.resolve()

    # 1. Cherche à la racine du projet
    file_at_root = project_root / ASTM_FILENAME
    if file_at_root.exists():
        print(f"Fichier trouvé à la racine : {file_at_root}")
        return file_at_root

    # 2. Cherche dans data/
    data_dir = project_root / "data"
    file_in_data = data_dir / ASTM_FILENAME
    if file_in_data.exists():
        print(f"Fichier trouvé dans data/ : {file_in_data}")
        return file_in_data

    # 3. Cherche partout dans le projet
    matches = list(project_root.rglob(ASTM_FILENAME))

    if matches:
        found_file = matches[0]
        print(f"Fichier trouvé ailleurs dans le projet : {found_file}")
        return found_file

    # 4. Sinon télécharge dans data/
    print("Fichier introuvable dans le projet.")
    print("Téléchargement du fichier ASTM E-490...")

    data_dir.mkdir(parents=True, exist_ok=True)

    urlretrieve(ASTM_URL, file_in_data)

    print(f"Fichier téléchargé ici : {file_in_data}")

    return file_in_data

# Lecture du fichier et extraction des données des E-lamda

def read_astm_e490_data(file_path: Path, printed: bool = False, graph: bool = False, xlim_nm: tuple[float, float] = (250, 1200)) -> pd.DataFrame:
    """
    Lit le fichier ASTM E-490 et retourne un DataFrame avec les longueurs d'onde en nm et les irradiances en W/m²/nm.
    Renvoie un DataFrame avec les colonnes "lambda_nm" et "E_nm".

    Paramètres :
    - file_path : chemin vers le fichier e490_00a_amo.xls
    - printed : affiche les premières lignes du DataFrame
    - graph : affiche un graphique du spectre solaire
    - xlim_nm : tuple (min, max) pour limiter l'affichage du graphique
    """

    astm_path = get_astm_e490_file()
    df = pd.read_excel(astm_path, sheet_name="NewAM0")


    lambda_µm = df.iloc[:, 0]
    E_µm = df.iloc[:, 1]
    # µm -> nm
    lambda_nm = lambda_µm * 1000
    E_nm = E_µm / 1000

    if printed:
        print(df.head())

    if graph:
        mask = (lambda_nm >= xlim_nm[0]) & (lambda_nm <= xlim_nm[1])

        lambda_plot = lambda_nm[mask]
        E_plot = E_nm[mask]

        colors = [
            rgb_255_to_mpl(wavelength_to_rgb(lam))
            for lam in lambda_plot
        ]

        plt.figure(figsize=(10, 5))

        # Courbe fine pour voir la forme générale
        plt.plot(lambda_plot, E_plot, linewidth=0.8, alpha=0.4)

        # Points colorés selon la longueur d'onde
        plt.scatter(lambda_plot, E_plot, c=colors, s=10)

        # Zones UV / visible / IR
        plt.axvspan(xlim_nm[0], 380, alpha=0.08, label="UV")
        plt.axvspan(380, 780, alpha=0.08, label="Visible")
        plt.axvspan(780, xlim_nm[1], alpha=0.08, label="IR proche")

        plt.axvline(380, linestyle="--", linewidth=0.8)
        plt.axvline(780, linestyle="--", linewidth=0.8)

        plt.xlabel("Longueur d'onde λ (nm)")
        plt.ylabel("Irradiance solaire $E_λ$ (W/m²/nm)")
        plt.title("Spectre solaire ASTM E-490 — UV, visible et IR proche")

        plt.xlim(xlim_nm)
        plt.grid(True)
        plt.legend()
        plt.show()

    return pd.DataFrame({"lambda_nm": lambda_nm, "E_nm": E_nm})

read_astm_e490_data(get_astm_e490_file(), printed=True, graph=True)