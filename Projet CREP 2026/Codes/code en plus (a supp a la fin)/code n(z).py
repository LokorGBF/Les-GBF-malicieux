import numpy as np
import matplotlib.pyplot as plt

# Constantes
T = 288                    # K
g = 9.81                   # m/s^2
R = 8.314                  # J/(mol*K)
kb = 1.380649e-23          # J/K
P0 = 101325                # Pa

def n_z(M, z, xi):
    """
    Calcule la densité moléculaire n_i(z)

    Paramètres :
    M  : masse molaire du gaz (kg/mol)
    z  : altitude (m)
    xi : fraction molaire (ex: 420 ppm = 420e-6)

    Retour :
    n_i(z) en molécules/m^3
    """
    return (np.exp(-(M*g*z)/(R*T)) * xi * P0) / (kb * T)



M_CO2 = 0.044             
xi_CO2 = 420e-6         

z = np.linspace(0, 20000, 500)  # de 0 à 20 km

n_CO2 = n_z(M_CO2, z, xi_CO2)

# Graphique
plt.plot(z, n_CO2)
plt.ylabel("n_CO2 (molécules/m^3)")
plt.xlabel("Altitude (m)")

plt.title("Densité moléculaire du CO2 en fonction de l'altitude")
plt.show()