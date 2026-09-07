import math

import numpy as np


def _serie(historico, chave):
    return np.array(historico.get(chave, []), dtype=float)


def calcular_metricas_controle(historico, dt=1.0):
    """
    Calcula metricas classicas de desempenho de controle.

    Equacoes:
    e[k] = PAM_alvo[k] - PAM_estimada[k]
    e_ss = |media(PAM_alvo_final) - media(PAM_estimada_final)|
    MAE = mean(|e|)
    RMSE = sqrt(mean(e^2))
    IAE = sum(|e| dt)
    ISE = sum(e^2 dt)
    ITAE = sum(t |e| dt)
    esforco = sum(|RPM_aplicada[k] - RPM_aplicada[k-1]|)
    saturacao = 100 * ciclos_saturados / ciclos_totais
    """

    pam = _serie(historico, "pam_estimada")
    alvo = _serie(historico, "pam_alvo")
    erro = _serie(historico, "erro")
    rpm = _serie(historico, "rpm_aplicada")
    saturacao = np.array(historico.get("saturacao_rpm", []), dtype=bool)

    if len(pam) == 0 or len(alvo) == 0:
        return {
            "mensagem": "Historico insuficiente para calcular metricas.",
            "erro_regime": None,
            "mae": None,
            "rmse": None,
            "rise_time": None,
            "settling_time_2pct": None,
            "settling_time_1mmhg": None,
            "overshoot_pct": None,
            "iae": None,
            "ise": None,
            "itae": None,
            "esforco_controle": None,
            "percentual_saturacao": None,
        }

    n = min(len(pam), len(alvo))
    pam = pam[:n]
    alvo = alvo[:n]
    erro = alvo - pam if len(erro) != n else erro[:n]

    janela_final = max(1, int(math.ceil(n * 0.10)))
    pam_final_media = float(np.mean(pam[-janela_final:]))
    alvo_final_media = float(np.mean(alvo[-janela_final:]))
    erro_regime = abs(alvo_final_media - pam_final_media)

    mae = float(np.mean(np.abs(erro)))
    rmse = float(np.sqrt(np.mean(erro ** 2)))
    tempo = np.arange(n, dtype=float) * float(dt)
    iae = float(np.sum(np.abs(erro)) * dt)
    ise = float(np.sum(erro ** 2) * dt)
    itae = float(np.sum(tempo * np.abs(erro)) * dt)

    rpm = rpm[:n] if len(rpm) >= n else rpm
    esforco = (
        float(np.sum(np.abs(np.diff(rpm))))
        if len(rpm) > 1
        else 0.0
    )
    percentual_saturacao = (
        float(np.mean(saturacao[:n]) * 100.0)
        if len(saturacao) >= n
        else 0.0
    )

    alvo_inicial = float(alvo[0])
    alvo_final = alvo_final_media
    pam_inicial = float(pam[0])
    delta = alvo_final - alvo_inicial

    rise_time = None
    overshoot_pct = None

    if abs(delta) > 1e-9:
        y10 = pam_inicial + 0.10 * delta
        y90 = pam_inicial + 0.90 * delta

        if delta > 0:
            idx10 = np.where(pam >= y10)[0]
            idx90 = np.where(pam >= y90)[0]
            pam_extremo = float(np.max(pam))
            overshoot_pct = max(0.0, (pam_extremo - alvo_final) / abs(delta) * 100.0)
        else:
            idx10 = np.where(pam <= y10)[0]
            idx90 = np.where(pam <= y90)[0]
            pam_extremo = float(np.min(pam))
            overshoot_pct = max(0.0, (alvo_final - pam_extremo) / abs(delta) * 100.0)

        if len(idx10) and len(idx90):
            rise_time = float(max(0.0, (idx90[0] - idx10[0]) * dt))

    def settling_time(banda):
        dentro = np.abs(pam - alvo_final) <= banda
        for i in range(n):
            if np.all(dentro[i:]):
                return float(i * dt)
        return None

    banda_2pct = max(abs(alvo_final) * 0.02, 1e-9)

    return {
        "mensagem": None,
        "erro_regime": float(erro_regime),
        "erro_regime_final": float(abs(float(alvo[-1]) - float(pam[-1]))),
        "mae": mae,
        "rmse": rmse,
        "rise_time": rise_time,
        "settling_time_2pct": settling_time(banda_2pct),
        "settling_time_1mmhg": settling_time(1.0),
        "overshoot_pct": overshoot_pct,
        "iae": iae,
        "ise": ise,
        "itae": itae,
        "esforco_controle": esforco,
        "percentual_saturacao": percentual_saturacao,
        "amostras": int(n),
    }
