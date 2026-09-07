import numpy as np


EPS_B = 1e-8
RPM_STD_MIN = 1.0


def _modelo_inconclusivo(mensagem, a=None, b=None, c=0.0, r2=None, rmse=None):
    b_val = None if b is None else float(b)
    return {
        "mensagem": mensagem,
        "a": None if a is None else float(a),
        "b": b_val,
        "c": float(c),
        "r2": r2,
        "rmse": rmse,
        "pam0": None,
        "rpm0": None,
        "epsilon_b": EPS_B,
        "b_abs": None if b_val is None else abs(b_val),
        "b_formatado": "-" if b_val is None else f"{b_val:.8e}",
        "a_formatado": "-" if a is None else f"{float(a):.8f}",
        "identificacao_conclusiva": False,
        "classificacao_identificacao": "IDENTIFICAÇÃO INCONCLUSIVA",
    }


def identificar_modelo_local(historico, epsilon_b=EPS_B):
    """
    Identifica localmente a dinamica incremental da planta.

    A identificacao e feita em torno do ponto de operacao:
    dPAM[k+1] = a * dPAM[k] + b * dRPM[k]

    com:
    dPAM = PAM - PAM0
    dRPM = RPM - RPM0

    O modelo equivalente com offset tambem e reportado:
    PAM[k+1] = a * PAM[k] + b * RPM[k] + c
    c = PAM0 - a * PAM0 - b * RPM0
    """

    pam = np.array(historico.get("pam_estimada", []), dtype=float)
    rpm = np.array(historico.get("rpm_aplicada", []), dtype=float)

    n = min(len(pam), len(rpm))
    if n < 4:
        return _modelo_inconclusivo(
            "Historico insuficiente para identificar modelo local.",
        )

    pam = pam[:n]
    rpm = rpm[:n]
    rpm_std = float(np.std(rpm[:-1]))
    rpm_span = float(np.max(rpm[:-1]) - np.min(rpm[:-1]))

    if rpm_std < RPM_STD_MIN or rpm_span <= 0:
        return _modelo_inconclusivo(
            "Identificacao inconclusiva: entrada RPM sem excitacao suficiente.",
        )

    pam0 = float(np.mean(pam[:-1]))
    rpm0 = float(np.mean(rpm[:-1]))
    dpam = pam - pam0
    drpm = rpm - rpm0

    y = dpam[1:n]
    X = np.column_stack([
        dpam[:n - 1],
        drpm[:n - 1],
    ])

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    erro = y - pred
    ss_res = float(np.sum(erro ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(erro ** 2)))

    a = float(coef[0])
    b = float(coef[1])
    c = float(pam0 - a * pam0 - b * rpm0)
    conclusiva = abs(b) >= float(epsilon_b)

    mensagem = None
    classificacao = "CONCLUSIVA"
    if not conclusiva:
        mensagem = (
            "Identificacao inconclusiva: |b| < epsilon_b. "
            "A entrada RPM nao excitou a PAM de forma mensuravel neste ensaio."
        )
        classificacao = "IDENTIFICAÇÃO INCONCLUSIVA"

    return {
        "mensagem": mensagem,
        "a": a,
        "b": b,
        "c": c,
        "r2": r2,
        "rmse": rmse,
        "pam0": pam0,
        "rpm0": rpm0,
        "epsilon_b": float(epsilon_b),
        "b_abs": abs(b),
        "b_formatado": f"{b:.8e}",
        "a_formatado": f"{a:.8f}",
        "identificacao_conclusiva": conclusiva,
        "classificacao_identificacao": classificacao,
    }


def analisar_estabilidade_local(
    a,
    b,
    kp,
    epsilon_b=EPS_B,
    identificacao_conclusiva=True,
):
    """
    Analisa estabilidade local do modelo identificado.

    Modelo incremental:
    dPAM[k+1] = a dPAM[k] + b dRPM[k]
    dRPM[k+1] = dRPM[k] - Kp dPAM[k]

    Acl = [[a, b], [-Kp, 1]]
    """

    if not identificacao_conclusiva:
        return {
            "mensagem": (
                "IDENTIFICAÇÃO INCONCLUSIVA: falta de excitacao mensuravel "
                "da entrada. A estabilidade nao deve ser declarada."
            ),
            "autovalores": [],
            "modulos": [],
            "raio_espectral": None,
            "estavel": None,
            "classificacao": "IDENTIFICAÇÃO INCONCLUSIVA",
        }

    if a is None or b is None:
        return {
            "mensagem": "Modelo local indisponivel para analise de estabilidade.",
            "autovalores": [],
            "modulos": [],
            "raio_espectral": None,
            "estavel": None,
            "classificacao": "indisponivel",
        }

    if abs(float(b)) < float(epsilon_b):
        return {
            "mensagem": (
                "IDENTIFICAÇÃO INCONCLUSIVA: |b| < epsilon_b ou falta de "
                "excitacao da entrada. A estabilidade nao deve ser declarada."
            ),
            "autovalores": [],
            "modulos": [],
            "raio_espectral": None,
            "estavel": None,
            "classificacao": "IDENTIFICAÇÃO INCONCLUSIVA",
        }

    acl = np.array(
        [
            [float(a), float(b)],
            [-float(kp), 1.0],
        ],
        dtype=float,
    )
    autovalores = np.linalg.eigvals(acl)
    modulos = np.abs(autovalores)
    rho = float(np.max(modulos))
    estavel = bool(np.all(modulos < 1.0))

    if estavel and rho < 0.85:
        classificacao = "ESTAVEL"
    elif estavel:
        classificacao = "ESTAVEL COM BAIXA MARGEM"
    else:
        classificacao = "INSTAVEL"

    return {
        "mensagem": None,
        "autovalores": [
            {
                "real": float(np.real(v)),
                "imag": float(np.imag(v)),
            }
            for v in autovalores
        ],
        "modulos": [float(v) for v in modulos],
        "raio_espectral": rho,
        "estavel": estavel,
        "classificacao": classificacao,
    }


def mapa_estabilidade_kp(a, b, kp_min=1, kp_max=50, epsilon_b=EPS_B):
    resultados = []

    if a is None or b is None or abs(float(b)) < float(epsilon_b):
        return {
            "resultados": resultados,
            "kp_max_estavel": None,
            "kp_menor_raio": None,
            "mensagem": "Mapa Kp indisponivel por identificacao inconclusiva.",
        }

    for kp in range(int(kp_min), int(kp_max) + 1):
        estabilidade = analisar_estabilidade_local(
            a,
            b,
            kp,
            epsilon_b=epsilon_b,
            identificacao_conclusiva=True,
        )
        resultados.append({
            "kp": kp,
            "raio_espectral": estabilidade["raio_espectral"],
            "estavel": estabilidade["estavel"],
            "autovalores": estabilidade["autovalores"],
        })

    estaveis = [r for r in resultados if r["estavel"]]
    kp_max_estavel = max([r["kp"] for r in estaveis]) if estaveis else None
    kp_menor_raio = min(
        resultados,
        key=lambda r: r["raio_espectral"]
        if r["raio_espectral"] is not None
        else float("inf"),
    )["kp"] if resultados else None

    return {
        "resultados": resultados,
        "kp_max_estavel": kp_max_estavel,
        "kp_menor_raio": kp_menor_raio,
        "mensagem": None,
    }
