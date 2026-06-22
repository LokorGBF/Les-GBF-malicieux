import numpy as np
import matplotlib.pyplot as plt
def AtmTetP() :

    # Constantes physiques
    G = 6.67e-11          # Constante gravitationnelle (m^3.kg^-1.s^-2)
    Mair = 29e-3          # Masse molaire de l'air (kg/mol)
    R = 8.314             # Constante des gaz parfaits (J/(K.mol))
    Mterre = 5.97e24      # Masse de la Terre (kg)
    RT = 6378e3           # Rayon de la Terre (m)
    g0 = 9.81             # Pesanteur constante du modèle isotherme (m/s^2)

    # Conditions au sol
    Tsol = 288            # Température au sol (K)
    Psol = 1.013e5        # Pression au sol (Pa)


    # Gradient thermique ISA
    def kISA(z):
        if 0 <= z < 11e3:
            return -6.5e-3
        elif z < 20e3:
            return 0
        elif z < 32e3:
            return 1.0e-3
        elif z < 47e3:
            return 2.8e-3
        elif z < 51e3:
            return 0
        elif z < 71e3:
            return -2.8e-3
        elif z <= 85e3:

            return -2.0e-3
        else:
            return 0

    # Discrétisation de l'altitude
    N = 10000
    Z = np.linspace(0, 85e3, N)

    # Pas d'intégration
    dz = Z[1] - Z[0]

    # Tableaux de stockage
    T = np.zeros(N)
    P = np.zeros(N)

    # Conditions initiales
    T[0] = Tsol
    P[0] = Psol


    # Méthode d'Euler
    for i in range(N - 1):
        g = G*Mterre/(RT+Z[i])**2
        dT = kISA(Z[i])

        dP = -(Mair * g / R) * P[i] / T[i]

        T[i + 1] = T[i] + dT * dz
        P[i + 1] = P[i] + dP * dz

    # Modèle isotherme
    def PisoT(z, T0=288):
        return Psol * np.exp(-Mair * g0 * z / (R * T0))


    return (T,P,Z)