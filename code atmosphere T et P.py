import numpy as np
import matplotlib.pyplot as plt
def AtmTetP() :

    G= 6.67e-11         # m/s²
    Mair = 29e-3      # kg/mol
    R = 8.314         # J/(K.mol)
    Mterre = 5.97e24
    RT= 6378e3

    Tsol = 288        # K
    Psol = 1.013e5    # Pa


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



   
    return (T,P)
z= int(input("z"))
T,P=AtmTetP()
print(T[z],P[z])

