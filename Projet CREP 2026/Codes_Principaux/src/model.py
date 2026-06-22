import numpy as np
from tqdm import tqdm

from src.energy_balance import compute_energy_balance


def initial_temperature(grid: dict, base_temperature: float):
    """
    Température initiale uniforme sur toute la planète.
    """
    n_theta, n_phi = grid["lat_grid"].shape
    return np.full((n_theta, n_phi), base_temperature, dtype=np.float64)


def run_model(
    grid,
    params,
    options,
    days: int,
    dt: float,
    base_temperature: float,
    history_stride: int = 1,
):
    """
    Lance la simulation.

    history_stride = 1   -> tous les pas sauvegardés
    history_stride = 48  -> ~1 pas/jour si dt=1800s
    """
    n_theta, n_phi = grid["lat_grid"].shape
    n_steps = int(days * 24 * 3600 / dt)

    stride = max(1, int(history_stride))

    n_store = (n_steps // stride) + 1
    if n_steps % stride != 0:
        n_store += 1  # on garde aussi l'état final

    T_history = np.empty((n_store, n_theta, n_phi), dtype=np.float32)

    T_current = initial_temperature(grid, base_temperature)
    T_next = np.empty_like(T_current)
    C = params["heat_capacity"]

    store_idx = 0
    T_history[store_idx] = T_current

    for n in tqdm(range(n_steps), desc="Simulation temporelle"):
        t_sec = n * dt

        P_net = compute_energy_balance(
            T_current,
            t_sec,
            grid,
            params,
            options,
            return_details=False,
        )

        np.add(T_current, dt * P_net / C, out=T_next)
        np.clip(T_next, 150.0, 350.0, out=T_next)

        step = n + 1
        if (step % stride == 0) or (step == n_steps):
            store_idx += 1
            T_history[store_idx] = T_next

        T_current, T_next = T_next, T_current

    return T_history[: store_idx + 1]