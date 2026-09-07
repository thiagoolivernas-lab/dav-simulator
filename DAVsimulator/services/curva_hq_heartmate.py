import numpy as np


CURVAS_HQ = {
    3000: [(0,30),(1,25),(2,20),(3,12),(4,5)],

    3500: [(0,40),(1,35),(2,30),(3,22),(4,12),(5,3)],

    4000: [(0,55),(1,50),(2,45),(3,35),(4,25),(5,12)],

    4500: [(0,75),(1,70),(2,62),(3,52),(4,40),(5,25),(6,10)],

    5000: [(0,95),(1,90),(2,82),(3,72),(4,60),(5,45),(6,28),(7,10)],

    5500: [(0,110),(1,105),(2,97),(3,87),(4,75),(5,58),(6,40),(7,18)],

    6000: [(0,125),(1,120),(2,112),(3,102),(4,90),(5,72),(6,52),(7,28),(8,5)],

    6500: [(0,145),(1,140),(2,132),(3,120),(4,106),(5,88),(6,68),(7,42),(8,14)],

    7000: [(0,165),(1,160),(2,150),(3,138),(4,122),(5,102),(6,78),(7,50),(8,20)],
}


def gerar_curva_por_afinidade(rpm_alvo, rpm_ref=7000):
    """
    Gera curva H-Q extrapolada por leis de afinidade de bombas rotodinamicas.

    Equacoes usadas apenas para simulacao:
    Q2 = Q1 * (N2 / N1)
    DeltaP2 = DeltaP1 * (N2 / N1)^2
    P2 = P1 * (N2 / N1)^3

    As curvas geradas por esta funcao nao devem ser tratadas como dados
    experimentais ou fornecidos por fabricante.
    """

    fator = float(rpm_alvo) / float(rpm_ref)
    return [
        (
            float(q) * fator,
            float(delta_p) * (fator ** 2),
        )
        for q, delta_p in CURVAS_HQ[int(rpm_ref)]
    ]


def obter_curvas_hq(modo_curva="validado"):
    curvas = dict(CURVAS_HQ)

    if modo_curva == "extrapolado":
        for rpm in (7500, 8000, 8500, 9000):
            curvas[rpm] = gerar_curva_por_afinidade(rpm)

    return curvas


def normalizar_rpm_curva(rpm, modo_curva="validado"):
    rpm = float(rpm)

    if modo_curva == "validado":
        return float(np.clip(rpm, 3000, 7000)), "validada"

    if modo_curva == "extrapolado":
        return float(np.clip(rpm, 3000, 9000)), (
            "extrapolada" if rpm > 7000 else "validada"
        )

    raise ValueError("modo_curva deve ser 'validado' ou 'extrapolado'")


def _fluxo_na_curva(rpm_curva, delta_p, curvas=None):
    curvas = curvas or CURVAS_HQ
    pontos = curvas[rpm_curva]

    fluxos = np.array([p[0] for p in pontos], dtype=float)
    pressoes = np.array([p[1] for p in pontos], dtype=float)

    # Como pressão cai quando fluxo sobe, invertemos para interpolar.
    pressoes_inv = pressoes[::-1]
    fluxos_inv = fluxos[::-1]

    if delta_p > max(pressoes):
        return 0.0

    if delta_p < min(pressoes):
        return float(max(fluxos))

    return float(
        np.interp(
            delta_p,
            pressoes_inv,
            fluxos_inv
        )
    )

def calcular_intersecao_bomba_sistema(
    rpm,
    r_eff,
    q_min=0,
    q_max=8,
    passo=0.01,
    modo_curva="validado"
):
    rpm, _ = normalizar_rpm_curva(rpm, modo_curva=modo_curva)
    r_eff = float(r_eff)

    melhor_q = None
    melhor_dp = None
    menor_erro = None

    q = q_min

    while q <= q_max:

        # pressão exigida pelo sistema cardiovascular
        dp_sistema = r_eff * q

        # pressão que a bomba consegue gerar nesse fluxo
        fluxo_teste = estimar_fluxo_hq(
            rpm=rpm,
            delta_p=dp_sistema,
            modo_curva=modo_curva,
        )

        erro = abs(fluxo_teste - q)

        if menor_erro is None or erro < menor_erro:
            menor_erro = erro
            melhor_q = q
            melhor_dp = dp_sistema

        q += passo

    return {
        "fluxo": float(melhor_q),
        "delta_p": float(melhor_dp),
        "erro_intersecao": float(menor_erro),
    }


def estimar_fluxo_hq(
    rpm,
    delta_p,
    modo_curva="validado"
):
    rpm, _ = normalizar_rpm_curva(rpm, modo_curva=modo_curva)
    delta_p = float(delta_p)

    curvas = obter_curvas_hq(modo_curva)
    rpms = sorted(curvas.keys())

    if rpm <= rpms[0]:
        return _fluxo_na_curva(rpms[0], delta_p, curvas=curvas)

    if rpm >= rpms[-1]:
        return _fluxo_na_curva(rpms[-1], delta_p, curvas=curvas)

    rpm_inf = max(r for r in rpms if r <= rpm)
    rpm_sup = min(r for r in rpms if r >= rpm)

    if rpm_inf == rpm_sup:
        return _fluxo_na_curva(rpm_inf, delta_p, curvas=curvas)

    q_inf = _fluxo_na_curva(rpm_inf, delta_p, curvas=curvas)
    q_sup = _fluxo_na_curva(rpm_sup, delta_p, curvas=curvas)

    peso = (rpm - rpm_inf) / (rpm_sup - rpm_inf)

    return float(
        q_inf + peso * (q_sup - q_inf)
    )


def estimar_potencia_hm3(
    rpm,
    fluxo
):
    rpm = float(rpm)
    fluxo = float(fluxo)

    fator_rpm = rpm / 5000

    potencia = (
        3.5
        + 0.45 * fluxo
        + 1.2 * (fator_rpm ** 3)
    )

    return float(potencia)


def estimar_eficiencia_hm3(
    fluxo
):
    fluxo = float(fluxo)

    eficiencia = 0.65 * np.exp(
        -((fluxo - 5.0) ** 2) / 10
    )

    return float(eficiencia)


def calcular_ponto_operacao_hm3(
    rpm,
    delta_p=None,
    r_eff=None,
    modo_curva="validado"
):
    rpm_original = float(rpm)
    rpm, faixa_hq = normalizar_rpm_curva(
        rpm_original,
        modo_curva=modo_curva,
    )

    if r_eff is not None:

        intersecao = calcular_intersecao_bomba_sistema(
            rpm=rpm,
            r_eff=r_eff,
            modo_curva=modo_curva,
        )

        fluxo = intersecao["fluxo"]
        delta_p = intersecao["delta_p"]

    else:

        if delta_p is None:
            delta_p = 90

        fluxo = estimar_fluxo_hq(
            rpm=rpm,
            delta_p=delta_p,
            modo_curva=modo_curva,
        )

    potencia = estimar_potencia_hm3(
        rpm=rpm,
        fluxo=fluxo
    )

    eficiencia = estimar_eficiencia_hm3(
        fluxo=fluxo
    )

    return {
        "rpm": float(rpm),
        "rpm_solicitado": float(rpm_original),
        "modo_curva": modo_curva,
        "faixa_hq": faixa_hq,
        "delta_p": float(delta_p),
        "fluxo": float(fluxo),
        "potencia": float(potencia),
        "eficiencia": float(eficiencia),
    }
