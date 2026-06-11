def wavelength_to_rgb(lambda_nm: float, gamma: float = 0.8) -> tuple[int, int, int]:
    """
    Convertit une longueur d'onde visible en couleur RGB.

    Paramètre :
        lambda_nm : longueur d'onde en nanomètres, typiquement entre 380 et 780 nm
        gamma : correction gamma pour un rendu visuel plus naturel

    Retour :
        tuple (R, G, B), chaque valeur entre 0 et 255
    """

    if lambda_nm < 380 or lambda_nm > 780:
        return (0, 0, 0)  # invisible pour l'œil humain

    if 380 <= lambda_nm < 440:
        r = -(lambda_nm - 440) / (440 - 380)
        g = 0.0
        b = 1.0

    elif 440 <= lambda_nm < 490:
        r = 0.0
        g = (lambda_nm - 440) / (490 - 440)
        b = 1.0

    elif 490 <= lambda_nm < 510:
        r = 0.0
        g = 1.0
        b = -(lambda_nm - 510) / (510 - 490)

    elif 510 <= lambda_nm < 580:
        r = (lambda_nm - 510) / (580 - 510)
        g = 1.0
        b = 0.0

    elif 580 <= lambda_nm < 645:
        r = 1.0
        g = -(lambda_nm - 645) / (645 - 580)
        b = 0.0

    else:  # 645 <= lambda_nm <= 780
        r = 1.0
        g = 0.0
        b = 0.0

    # Atténuation aux bords du visible
    if 380 <= lambda_nm < 420:
        factor = 0.3 + 0.7 * (lambda_nm - 380) / (420 - 380)
    elif 420 <= lambda_nm < 700:
        factor = 1.0
    else:  # 700 <= lambda_nm <= 780
        factor = 0.3 + 0.7 * (780 - lambda_nm) / (780 - 700)

    def correct(color: float) -> int:
        if color == 0:
            return 0
        return round(255 * ((color * factor) ** gamma))

    R = correct(r)
    G = correct(g)
    B = correct(b)

    return (R, G, B)

def rgb_255_to_mpl(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """
    Convertit un RGB classique 0-255 en RGB utilisable par matplotlib 0-1.
    Exemple : (255, 128, 0) -> (1.0, 0.502, 0.0)
    """
    return tuple(v / 255 for v in rgb)