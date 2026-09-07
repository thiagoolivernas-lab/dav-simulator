import numpy as np


def calcular_mae(y_real, y_pred):

    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)

    return float(
        np.mean(
            np.abs(y_real - y_pred)
        )
    )


def calcular_rmse(y_real, y_pred):

    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)

    return float(
        np.sqrt(
            np.mean(
                (y_real - y_pred) ** 2
            )
        )
    )


def calcular_r2(y_real, y_pred):

    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum(
        (y_real - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_real - np.mean(y_real)) ** 2
    )

    if ss_tot == 0:
        return 0

    return float(
        1 - ss_res / ss_tot
    )

def avaliar_picos_r(
    picos_reais,
    picos_detectados,
    fs=125,
    tolerancia_ms=100
):
    """
    Avalia detecção de picos R.

    Um pico detectado é considerado correto se estiver
    a até tolerancia_ms de um pico real.
    """

    picos_reais = np.asarray(picos_reais, dtype=int)
    picos_detectados = np.asarray(picos_detectados, dtype=int)

    tolerancia_amostras = int(
        (tolerancia_ms / 1000) * fs
    )

    usados_reais = set()
    erros_ms = []

    tp = 0
    fp = 0

    for pd in picos_detectados:

        if len(picos_reais) == 0:
            fp += 1
            continue

        distancias = np.abs(
            picos_reais - pd
        )

        idx = int(np.argmin(distancias))

        menor_distancia = distancias[idx]

        if (
            menor_distancia <= tolerancia_amostras
            and idx not in usados_reais
        ):
            tp += 1
            usados_reais.add(idx)

            erro_ms = (
                menor_distancia / fs
            ) * 1000

            erros_ms.append(
                float(erro_ms)
            )

        else:
            fp += 1

    fn = len(picos_reais) - tp

    sensibilidade = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    precisao = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    f1 = (
        2 * precisao * sensibilidade
        / (precisao + sensibilidade)
        if (precisao + sensibilidade) > 0
        else 0
    )

    erro_medio_ms = (
        float(np.mean(erros_ms))
        if erros_ms
        else None
    )

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "sensibilidade": round(float(sensibilidade), 4),
        "precisao": round(float(precisao), 4),
        "f1": round(float(f1), 4),
        "erro_medio_ms": (
            round(erro_medio_ms, 2)
            if erro_medio_ms is not None
            else None
        ),
    }