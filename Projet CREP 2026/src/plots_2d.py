import matplotlib.pyplot as plt
import numpy as np
from src.config import FIGURES_DIR


def plot_temperature_map(T, grid, filename="temperature_map.png"):
    """
    Affiche et sauvegarde une carte de température en °C.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.imshow(
        T - 273.15,
        origin="lower",
        extent=[-180, 180, -90, 90],
        cmap="inferno",
        vmin=-50,
        vmax=50,
        aspect="auto",
    )
    plt.colorbar(label="Température de surface (°C)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Température de surface")
    plt.grid(alpha=0.3)

    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Carte sauvegardée : {output_path}")


def plot_temperature_curve(
    T_history,
    grid,
    lat_target=48,
    lon_target=2,
    filename="temperature_curve.png",
    history_stride: int = 1,
):
    """
    Trace la température dans le temps pour la case la plus proche.
    """
    from src.config import DT

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    lat = grid["lat"]
    lon = grid["lon"]
    i = abs(lat - lat_target).argmin()
    j = abs(lon - lon_target).argmin()

    T_series = T_history[:, i, j] - 273.15
    time_hours = np.arange(len(T_series)) * DT * max(1, int(history_stride)) / 3600.0

    plt.figure(figsize=(10, 5))
    plt.plot(time_hours, T_series)
    plt.xlabel("Temps (heures)")
    plt.ylabel("Température (°C)")
    plt.title(f"Température proche de lat={lat[i]:.1f}, lon={lon[j]:.1f}")
    plt.grid(alpha=0.3)

    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Courbe sauvegardée : {output_path}")