import math

import numpy as np


def calcular_pi_dav(potencias, min_amostras=2):
    """
    Calcula o PI estimado do DAV a partir da variacao da potencia simulada
    da bomba em uma janela temporal.

    PI = (Pmax - Pmin) / Pavg

    Este valor e uma aproximacao de simulacao inspirada no principio usado em
    dispositivos como o HeartMate 3. Ele nao reproduz algoritmo proprietario
    nem usa pressao arterial, PAS, PAD, PAM ou pressao de pulso.
    """

    if potencias is None:
        valores = np.array([], dtype=float)
    else:
        valores = np.array(list(potencias), dtype=float)

    valores = valores[np.isfinite(valores)]
    valores = valores[valores >= 0]

    if len(valores) < int(min_amostras):
        return {
            "pi": None,
            "potencia_max": None,
            "potencia_min": None,
            "potencia_media": None,
            "potencia_std": None,
            "n_amostras": int(len(valores)),
            "mensagem": "amostras insuficientes para calcular PI estimado do DAV",
        }

    potencia_media = float(np.mean(valores))

    if potencia_media <= 0 or math.isclose(potencia_media, 0.0):
        return {
            "pi": None,
            "potencia_max": float(np.max(valores)),
            "potencia_min": float(np.min(valores)),
            "potencia_media": potencia_media,
            "potencia_std": float(np.std(valores)),
            "n_amostras": int(len(valores)),
            "mensagem": "potencia media invalida para calcular PI estimado do DAV",
        }

    potencia_max = float(np.max(valores))
    potencia_min = float(np.min(valores))

    return {
        "pi": float((potencia_max - potencia_min) / potencia_media),
        "potencia_max": potencia_max,
        "potencia_min": potencia_min,
        "potencia_media": potencia_media,
        "potencia_std": float(np.std(valores)),
        "n_amostras": int(len(valores)),
        "mensagem": None,
    }


def calcular_pi_dav_janela(potencias, dt=1.0, janela_s=15.0, min_amostras=2):
    """
    Calcula o PI estimado do DAV usando a janela final da serie de potencia.

    O numero de amostras da janela e definido pela resolucao temporal real
    informada em dt, evitando assumir que uma amostra equivale sempre a 1 s.
    """

    if potencias is None:
        serie = []
    else:
        serie = list(potencias)

    dt = float(dt)
    janela_s = float(janela_s)

    if dt <= 0:
        raise ValueError("dt deve ser positivo para a janela de PI estimado do DAV.")

    amostras_janela = max(1, int(math.ceil(janela_s / dt)))
    janela = serie[-amostras_janela:]
    resultado = calcular_pi_dav(janela, min_amostras=min_amostras)
    resultado["janela_s"] = janela_s
    resultado["dt"] = dt
    resultado["amostras_janela"] = int(amostras_janela)
    resultado["potencias_janela"] = [
        float(v)
        for v in np.array(janela, dtype=float)
        if np.isfinite(v) and float(v) >= 0
    ]
    return resultado
