import numpy as np
import matplotlib.pyplot as plt


def AtmTetP():

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

    # Gradient thermique ISA : dT/dz
    def kISA(z):

        if 0 <= z < 11e3:
            return -6.5e-3

        elif z < 20e3:
            return 0.0

        elif z < 32e3:
            return 1.0e-3

        elif z < 47e3:
            return 2.8e-3

        elif z < 51e3:
            return 0.0

        elif z < 71e3:
            return -2.8e-3

        elif z <= 85e3:
            return -2.0e-3

        else:
            return 0.0

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

    # Résolution du modèle avec température variable
    # par la méthode d'Euler
    for i in range(N - 1):

        z = Z[i]

        # Pesanteur variable avec l'altitude
        g = G * Mterre / (RT + z)**2

        # Équations différentielles
        dT_dz = kISA(z)

        dP_dz = -(Mair * g / (R * T[i])) * P[i]

        # Méthode d'Euler
        T[i + 1] = T[i] + dT_dz * dz
        P[i + 1] = P[i] + dP_dz * dz

    # Modèle isotherme :
    # P(z) = P0 exp(-Mair*g0*z/(R*T0))
    P_iso = Psol * np.exp(
        -Mair * g0 * Z / (R * Tsol)
    )

    return Z, T, P, P_iso


# Calcul des différents modèles
Z, T, P, P_iso = AtmTetP()


# Comparaison des deux modèles de pression
plt.figure(figsize=(8, 6))

plt.plot(
    P,
    Z / 1000,
    label="Modèle ISA : T(z) et g(z) variables"
)

plt.plot(
    P_iso,
    Z / 1000,
    "--",
    label="Modèle isotherme : T = T0 et g = 9,81 m/s²"
)

plt.xlabel("Pression P (Pa)")
plt.ylabel("Altitude z (km)")
plt.title("Comparaison des deux modèles de pression atmosphérique")
plt.grid()
plt.legend()
plt.show()


# Comparaison avec une échelle logarithmique
plt.figure(figsize=(8, 6))

plt.semilogx(
    P,
    Z / 1000,
    label="Modèle ISA : T(z) et g(z) variables"
)

plt.semilogx(
    P_iso,
    Z / 1000,
    "--",
    label="Modèle isotherme : T = T0 et g = 9,81 m/s²"
)

plt.xlabel("Pression P (Pa)")
plt.ylabel("Altitude z (km)")
plt.title("Comparaison des pressions en échelle logarithmique")
plt.grid()
plt.legend()
plt.show()


# Affichage de la température du modèle ISA
plt.figure(figsize=(8, 6))

plt.plot(T, Z / 1000)

plt.xlabel("Température T (K)")
plt.ylabel("Altitude z (km)")
plt.title("Évolution de la température dans le modèle ISA")
plt.grid()
plt.show()