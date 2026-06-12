class petitCubeAtmo:
    def __init__(self, dx: int, dy: int, dz: int, compositionMole: dict):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.compositionMole = compositionMole # Composition de l'air en pourcentage
        self.temperature = 0 # Température en K
        self.U = 0 #energie interne en J
        
        
def rayonmentApresCouche(atmosphere: list, rayonnementEntrant: float) -> float:
    """
    Calcule le rayonnement après avoir traversé une couche d'atmosphère.

    param:
    -atmosphere : liste de petitCubeAtmo représentant la couche d'atmosphère
    -rayonnementEntrant : rayonnement entrant en W/m2

    return:
    -rayonnementSortant : rayonnement sortant en W/m2
    """
    rayonnementSortant = rayonnementEntrant
    for cube in atmosphere:
        # Calcul de l'absorption du rayonnement par le cube d'atmosphère
        absorption = sum(cube.compositionMole.values()) * 0.1 # Absorption proportionnelle à la composition de l'air
        rayonnementSortant *= (1 - absorption) # Le rayonnement sortant est réduit par l'absorption
    return rayonnementSortant
atmosphere = [petitCubeAtmo(1, 1, 1, {"N2": 1.61e+04, "O2": 4.34e+03, "Ar": 9.34e+02}) for _ in range(10)] # Atmosphère composée de 10 petit cube de 1m3, avec une composition en pourcentage de l'air

