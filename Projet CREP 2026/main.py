import numpy as np

# Lance la simulation
from src.config import (
    N_THETA,
    N_PHI,
    DAYS,
    DT,
    INITIAL_TEMPERATURE,
    OPTIONS,
    SAVE_RESULTS,
    RESULT_FILE,
    SIMULATIONS_DIR,
)

from src.grid import create_spherical_grid, print_grid_info
from src.albedo import prepare_surface_albedo
from src.clouds import prepare_cloud_albedo
from src.heat_capacity import prepare_heat_capacity
from src.evaporation import prepare_evaporation_base
from src.model import run_model
from src.plots_2d import plot_temperature_map, plot_temperature_curve


def main():
    print("=== Création de la grille sphérique ===")
    grid = create_spherical_grid(N_THETA, N_PHI)
    print_grid_info(grid)

    print("\n=== Préparation des paramètres ===")

    surface_albedo_daily = prepare_surface_albedo(
        grid,
        use_variable_albedo=OPTIONS["surface_albedo"]
    )

    cloud_albedo_daily = prepare_cloud_albedo(
        grid,
        use_cloud_albedo=OPTIONS["cloud_albedo"]
    )

    heat_capacity = prepare_heat_capacity(
        grid,
        use_variable_capacity=OPTIONS["heat_capacity"]
    )

    evaporation_base = prepare_evaporation_base(
        grid,
        use_evaporation=OPTIONS["evaporation"]
    )

    params = {
        "surface_albedo_daily": surface_albedo_daily,
        "cloud_albedo_daily": cloud_albedo_daily,
        "heat_capacity": heat_capacity,
        "evaporation_base": evaporation_base,
    }

    print("\n=== Lancement du modèle ===")
    T_history = run_model(
        grid=grid,
        params=params,
        options=OPTIONS,
        days=DAYS,
        dt=DT,
        base_temperature=INITIAL_TEMPERATURE
    )

    print("T_history :", T_history.shape)
    print("Max finale :", T_history[-1].max() - 273.15, "°C")
    print("Min finale :", T_history[-1].min() - 273.15, "°C")
    print("Moyenne finale :", T_history[-1].mean() - 273.15, "°C")
    print("Présence de NaN :", np.isnan(T_history).any())

    if SAVE_RESULTS:
        SIMULATIONS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(RESULT_FILE, T_history)
        print(f"Résultats sauvegardés : {RESULT_FILE}")

    print("\n=== Affichage ===")
    plot_temperature_map(T_history[-1], grid, filename="temperature_finale.png")
    plot_temperature_curve(T_history, grid, lat_target=48.8566, lon_target=2.3522)


if __name__ == "__main__":
    main()