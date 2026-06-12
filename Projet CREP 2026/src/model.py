import numpy as np
from tqdm import tqdm

from src.energy_balance import compute_energy_balance

# Temporalité du modèle = boucle temporelle

def initial_temperature(grid: dict, base_temperature: float):
    """
    Température initiale uniforme sur toute la planète.
    base_temperature est en kelvins.
    """

    n_theta, n_phi = grid["lat_grid"].shape
    return np.full((n_theta, n_phi), base_temperature)


def run_model(grid, params, options, days: int, dt: float, base_temperature: float):
    """
    Lance la simulation.

    Sortie :
    T_history.shape = (N_steps + 1, N_THETA, N_PHI)
    """

    n_theta, n_phi = grid["lat_grid"].shape
    n_steps = int(days * 24 * 3600 / dt)

    T_history = np.zeros((n_steps + 1, n_theta, n_phi), dtype=np.float32)

    T_history[0] = initial_temperature(grid, base_temperature)

    for n in tqdm(range(n_steps), desc="Simulation temporelle"):
        t_sec = n * dt

        T_current = T_history[n].astype(float)

        P_net, _ = compute_energy_balance(
            T_current,
            t_sec,
            grid,
            params,
            options
        )

        C = params["heat_capacity"]

        T_next = T_current + dt * P_net / C

        # Sécurité numérique simple
        T_next = np.clip(T_next, 150.0, 350.0)

        T_history[n + 1] = T_next.astype(np.float32)

    return T_history