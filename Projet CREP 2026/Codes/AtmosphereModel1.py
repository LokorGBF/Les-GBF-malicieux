import numpy as np
import pandas as pd


N_A = 6.022_140_76e23  # mol^-1


class CubeAtmo:
    """
    Cube atmospherique simple.

    Hypotheses :
    - compositionMole contient des quantites en moles dans le cube.
    - coefOptique contient des coefficients par matiere et longueur d'onde en nm.

    Par defaut, on suppose que coefOptique[matiere][lambda_nm] est une section
    efficace sigma(lambda) en m2/molecule.
    """

    def __init__(self, dx: float, dy: float, dz: float, compositionMole: dict, coefOptique: dict):
        self.dx = float(dx)
        self.dy = float(dy)
        self.dz = float(dz)

        self.surface = self.dx * self.dy
        self.volume = self.dx * self.dy * self.dz

        self.compositionMole = compositionMole
        self.coefOptique = coefOptique

        # self.Eabs[matiere][lambda_nm] = energie absorbee en J
        self.Eabs = {}

        # self.Pabs[matiere][lambda_nm] = puissance absorbee en W
        self.Pabs = {}

    def get_colonnes_rayon(self, rayon: pd.DataFrame):
        col_lambda = "Wavelength, microns"

        if col_lambda not in rayon.columns:
            raise KeyError(f"Colonne manquante : {col_lambda}")

        if "E-490 W/m2/micron" in rayon.columns:
            col_flux = "E-490 W/m2/micron"
        elif "E-490 W/m2/microne" in rayon.columns:
            col_flux = "E-490 W/m2/microne"
        else:
            raise KeyError("Colonne de flux manquante : E-490 W/m2/micron")

        return col_lambda, col_flux

    def get_coef_interpole(self, matiere: str, lamda_nm: float) -> float:
        """Renvoie le coefficient optique interpole k(lambda)."""

        if matiere not in self.coefOptique:
            return 0.0

        coef_matiere = self.coefOptique[matiere]

        if len(coef_matiere) == 0:
            return 0.0

        lambdas = np.array(sorted(coef_matiere.keys()), dtype=float)
        coefs = np.array([coef_matiere[l] for l in lambdas], dtype=float)

        if lamda_nm < lambdas[0] or lamda_nm > lambdas[-1]:
            return 0.0

        return float(np.interp(lamda_nm, lambdas, coefs))

    def get_tau_matiere(self, matiere: str, lamda_nm: float) -> float:
        """
        Calcule l'epaisseur optique tau d'une matiere.

        Cas utilise ici :
        coefOptique = sigma(lambda), section efficace en m2/molecule.

        tau = sigma(lambda) * colonne_molecules
        colonne_molecules = nombre de molecules / surface
        """

        if matiere not in self.compositionMole:
            return 0.0

        if matiere not in self.coefOptique:
            return 0.0

        n_mol = float(self.compositionMole[matiere])
        sigma = self.get_coef_interpole(matiere, lamda_nm)

        N_molecules = n_mol * N_A
        colonne_molecules = N_molecules / self.surface

        tau = sigma * colonne_molecules

        return float(tau)

    def passageRayon_dt(self, rayon: pd.DataFrame):
        """
        Fait passer un rayon dans le cube.

        Entree :
        - rayon : DataFrame avec :
            "Wavelength, microns" en micrometres
            "E-490 W/m2/micron" en W/m2/micrometre

        Sorties :
        - new_rayon : DataFrame du rayon apres absorption
        - Pabs_matiere_lamda : dictionnaire des puissances absorbees

        Unites :
        - Pabs en W
        - lambda stocke en nm
        """

        col_lambda, col_flux = self.get_colonnes_rayon(rayon)

        new_rayon = rayon.copy()

        lambda_um_array = np.array(rayon[col_lambda], dtype=float)
        E_um_array = np.array(rayon[col_flux], dtype=float)

        lambda_nm_array = lambda_um_array * 1000.0
        E_nm_array = E_um_array / 1000.0

        if len(lambda_nm_array) > 1:
            delta_lambda_nm_array = np.gradient(lambda_nm_array)
        else:
            delta_lambda_nm_array = np.array([1.0])

        Eout_um_list = []
        Pout_total_W_list = []
        Pabs_total_W_list = []
        tau_total_list = []

        Pabs_matiere_lamda = {}

        for lamda_nm, E_nm, delta_lambda_nm in zip(
            lambda_nm_array,
            E_nm_array,
            delta_lambda_nm_array
        ):
            Pin_W = E_nm * self.surface * delta_lambda_nm

            tau_par_matiere = {}
            tau_total = 0.0

            for matiere in self.compositionMole.keys():
                tau = self.get_tau_matiere(matiere, lamda_nm)

                if tau > 0:
                    tau_par_matiere[matiere] = tau
                    tau_total += tau

            transmission = np.exp(-tau_total)

            Pout_W = Pin_W * transmission
            Pabs_total_W = Pin_W - Pout_W

            for matiere, tau in tau_par_matiere.items():
                if tau_total > 0:
                    Pabs_W = Pabs_total_W * tau / tau_total
                else:
                    Pabs_W = 0.0

                if matiere not in Pabs_matiere_lamda:
                    Pabs_matiere_lamda[matiere] = {}

                Pabs_matiere_lamda[matiere][lamda_nm] = Pabs_W

            if self.surface > 0 and delta_lambda_nm > 0:
                Eout_nm = Pout_W / (self.surface * delta_lambda_nm)
            else:
                Eout_nm = 0.0

            Eout_um = Eout_nm * 1000.0

            Eout_um_list.append(Eout_um)
            Pout_total_W_list.append(Pout_W)
            Pabs_total_W_list.append(Pabs_total_W)
            tau_total_list.append(tau_total)

        new_rayon[col_flux] = Eout_um_list
        new_rayon["Pout_total_W"] = Pout_total_W_list
        new_rayon["Pabs_total_W"] = Pabs_total_W_list
        new_rayon["tau_total"] = tau_total_list

        self.Pabs = Pabs_matiere_lamda

        return new_rayon, Pabs_matiere_lamda

    def Eabs_apresRayon(self, rayon: pd.DataFrame, dt_s: float):
        """
        Calcule l'energie absorbee apres passage du rayon pendant dt_s secondes.

        E = P * dt
        """

        new_rayon, Pabs_matiere_lamda = self.passageRayon_dt(rayon)

        Eabs_matiere_lamda = {}

        for matiere, dico_lambda in Pabs_matiere_lamda.items():
            Eabs_matiere_lamda[matiere] = {}

            for lamda_nm, Pabs_W in dico_lambda.items():
                Eabs_J = Pabs_W * dt_s
                Eabs_matiere_lamda[matiere][lamda_nm] = Eabs_J

        self.Eabs = Eabs_matiere_lamda

        return new_rayon, Eabs_matiere_lamda


def passage_atmosphere(atmosphere: list, rayon: pd.DataFrame, dt_s: float):
    """Fait passer un rayon a travers une liste de cubes atmospheriques."""

    rayon_courant = rayon.copy()

    for cube in atmosphere:
        rayon_courant, _ = cube.Eabs_apresRayon(rayon_courant, dt_s)

    return rayon_courant


def somme_absorption_atmosphere(atmosphere: list):
    """
    Additionne l'energie et la puissance absorbees par toutes les couches.

    Retour :
    - Eabs_total[matiere][lambda_nm] = energie absorbee totale en J
    - Pabs_total[matiere][lambda_nm] = puissance absorbee totale en W
    """

    Eabs_total = {}
    Pabs_total = {}

    for cube in atmosphere:
        for matiere, dico_lambda in cube.Eabs.items():
            if matiere not in Eabs_total:
                Eabs_total[matiere] = {}

            for lamda_nm, Eabs_J in dico_lambda.items():
                if lamda_nm not in Eabs_total[matiere]:
                    Eabs_total[matiere][lamda_nm] = 0.0

                Eabs_total[matiere][lamda_nm] += Eabs_J

        for matiere, dico_lambda in cube.Pabs.items():
            if matiere not in Pabs_total:
                Pabs_total[matiere] = {}

            for lamda_nm, Pabs_W in dico_lambda.items():
                if lamda_nm not in Pabs_total[matiere]:
                    Pabs_total[matiere][lamda_nm] = 0.0

                Pabs_total[matiere][lamda_nm] += Pabs_W

    return Eabs_total, Pabs_total


def extraire_spectre_nm(rayon: pd.DataFrame):
    """
    Convertit un rayon au format ASTM en tableaux utiles pour les graphes.

    Retour :
    - lambda_nm : longueurs d'onde en nm
    - E_nm : irradiance spectrale en W/m2/nm
    """

    col_lambda = "Wavelength, microns"

    if col_lambda not in rayon.columns:
        raise KeyError(f"Colonne manquante : {col_lambda}")

    if "E-490 W/m2/micron" in rayon.columns:
        col_flux = "E-490 W/m2/micron"
    elif "E-490 W/m2/microne" in rayon.columns:
        col_flux = "E-490 W/m2/microne"
    else:
        raise KeyError("Colonne de flux manquante : E-490 W/m2/micron")

    lambda_nm = np.array(rayon[col_lambda], dtype=float) * 1000.0
    E_nm = np.array(rayon[col_flux], dtype=float) / 1000.0

    return lambda_nm, E_nm


def puissance_entrante_par_lambda(rayon: pd.DataFrame, surface: float):
    """
    Calcule la puissance entrante par bande spectrale.

    Retour :
    - lambda_nm
    - Pin_W : puissance entrante par bande en W
    """

    lambda_nm, E_nm = extraire_spectre_nm(rayon)

    if len(lambda_nm) > 1:
        delta_lambda_nm = np.gradient(lambda_nm)
    else:
        delta_lambda_nm = np.array([1.0])

    Pin_W = E_nm * surface * delta_lambda_nm

    return lambda_nm, Pin_W
