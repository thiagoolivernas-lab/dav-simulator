import math


def validar_tau(tau_base, tau_atual=None):
    tau = tau_atual if tau_atual is not None else tau_base

    if tau is None:
        raise ValueError("tau_base/tau_atual deve ser informado.")

    tau = float(tau)

    if tau <= 0:
        raise ValueError("tau deve ser positivo para a dinâmica Windkessel.")

    return tau


def calcular_p_eq_windkessel(r_eff, q_dav, q_nativo=0.0):
    """
    Calcula a pressão de equilíbrio do modelo cardiovascular simplificado.

    Unidades adotadas no simulador:
    R_eff: mmHg/(L/min)
    Q: L/min
    P_eq: mmHg

    Nesta versão, Q_nativo permanece zero por ausência de modelo validado de
    fluxo cardíaco nativo. A assinatura permite acrescentá-lo futuramente.
    """

    q_total = float(q_dav) + float(q_nativo)
    return float(r_eff) * q_total


def atualizar_pam_windkessel(
    pam_atual,
    p_eq,
    tau_base,
    tau_atual=None,
    dt=1.0,
    razao_maxima=0.2,
):
    """
    Integra dP/dt = (P_eq - P) / tau por Euler subdividido.

    tau representa a constante vascular agregada tau = R*C. A complacência C
    não é estimada separadamente nesta versão.
    """

    tau = validar_tau(tau_base, tau_atual=tau_atual)
    dt = float(dt)

    if dt <= 0:
        raise ValueError("dt deve ser positivo para a dinâmica Windkessel.")

    passos = max(1, int(math.ceil(dt / (float(razao_maxima) * tau))))
    dt_sub = dt / passos
    pam = float(pam_atual)
    p_eq = float(p_eq)

    for _ in range(passos):
        pam += (dt_sub / tau) * (p_eq - pam)

    return {
        "pam": float(pam),
        "tau": float(tau),
        "dt": dt,
        "p_eq": p_eq,
        "subpassos": int(passos),
        "dt_sobre_tau": float(dt_sub / tau),
    }
