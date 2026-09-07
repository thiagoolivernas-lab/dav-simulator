from .controle_dav import controlar_rpm
from .curva_hq_heartmate import calcular_ponto_operacao_hm3
from .modelo_cardiovascular import (
    atualizar_pam_windkessel,
    calcular_p_eq_windkessel,
)


def executar_pipeline_dav(
    rpm_atual,
    pam_alvo,
    r_eff,
    tau_base,
    tau_atual=None,
    fluxo_base=3.0,
    pam_base=122.5,
    kp=20,
    pam_min=65,
    pam_max=140,
    modo_curva="validado",
    pam_atual=None,
    dt=1.0,
):
    ponto_bomba = calcular_ponto_operacao_hm3(
        rpm=rpm_atual,
        r_eff=r_eff,
        modo_curva=modo_curva,
    )

    fluxo = ponto_bomba["fluxo"]
    p_eq = calcular_p_eq_windkessel(
        r_eff=r_eff,
        q_dav=fluxo,
    )
    dinamica = atualizar_pam_windkessel(
        pam_atual=pam_base if pam_atual is None else pam_atual,
        p_eq=p_eq,
        tau_base=tau_base,
        tau_atual=tau_atual,
        dt=dt,
    )

    pam_estimada = dinamica["pam"]

    controle = controlar_rpm(
        pam_estimada=pam_estimada,
        pam_alvo=pam_alvo,
        rpm_atual=rpm_atual,
        kp=kp,
        rpm_max=9000 if modo_curva == "extrapolado" else 7000,
        pam_min=pam_min,
        pam_max=pam_max
    )

    return {
        "rpm_atual": rpm_atual,
        "fluxo_estimado": fluxo,
        "pam_estimada": pam_estimada,
        "p_eq": p_eq,
        "tau": dinamica["tau"],
        "dt": dinamica["dt"],
        "subpassos": dinamica["subpassos"],
        "dt_sobre_tau": dinamica["dt_sobre_tau"],
        "pam_alvo": pam_alvo,
        "rpm_sugerido": controle["rpm_sugerido"],
        "erro": controle["erro"],
        "delta_p": ponto_bomba["delta_p"],
        "potencia": ponto_bomba["potencia"],
        "eficiencia": ponto_bomba["eficiencia"],
        "modo_curva": modo_curva,
        "faixa_hq": ponto_bomba.get("faixa_hq"),
    }


def simular_controle_dav(
    rpm_inicial,
    pam_alvo,
    r_eff,
    tau_base,
    tau_atual=None,
    fluxo_base=3.0,
    pam_base=110,
    kp=20,
    ciclos=30,
    pam_min=65,
    pam_max=130,
    modo_curva="validado",
    pam_inicial=None,
):
    """
    Simula várias iterações do controle DAV.

    A cada ciclo:
    RPM -> curva H-Q -> Q_DAV -> P_eq -> dinâmica Windkessel -> controle
    """

    rpm = float(rpm_inicial)
    pam_atual = float(pam_inicial if pam_inicial is not None else pam_base)

    historico = {
        "ciclos": [],
        "tempo": [],
        "rpm": [],
        "fluxo": [],
        "pam": [],
        "pam_alvo": [],
        "erro": [],
        "potencia": [],
        "eficiencia": [],
        "delta_p": [],
        "r_eff": [],
        "tau": [],
        "p_eq": [],
        "subpassos": [],

    }

    for i in range(ciclos):

        
        ponto_bomba = calcular_ponto_operacao_hm3(
            rpm=rpm,
            r_eff=r_eff,
            modo_curva=modo_curva,
        )

        fluxo = ponto_bomba["fluxo"]
        p_eq = calcular_p_eq_windkessel(
            r_eff=r_eff,
            q_dav=fluxo,
        )
        dinamica = atualizar_pam_windkessel(
            pam_atual=pam_atual,
            p_eq=p_eq,
            tau_base=tau_base,
            tau_atual=tau_atual,
            dt=1.0,
        )
        pam_atual = dinamica["pam"]

        controle = controlar_rpm(
            pam_estimada=pam_atual,
            pam_alvo=pam_alvo,
            rpm_atual=rpm,
            kp=kp,
            rpm_max=9000 if modo_curva == "extrapolado" else 7000,
            pam_min=pam_min,
            pam_max=pam_max
        )

        historico["ciclos"].append(i + 1)
        historico["tempo"].append(float(i))
        historico["rpm"].append(float(rpm))
        historico["fluxo"].append(float(fluxo))
        historico["pam"].append(float(pam_atual))
        historico["pam_alvo"].append(float(pam_alvo))
        historico["erro"].append(float(controle["erro"]))
        historico["r_eff"].append(float(r_eff))
        historico["tau"].append(float(dinamica["tau"]))
        historico["p_eq"].append(float(p_eq))
        historico["subpassos"].append(int(dinamica["subpassos"]))

        historico["potencia"].append(
            float(ponto_bomba["potencia"])
        )

        historico["eficiencia"].append(
            float(ponto_bomba["eficiencia"])
        )

        historico["delta_p"].append(
            float(ponto_bomba["delta_p"])
        )

        rpm = float(
            controle["rpm_sugerido"]
        )

        
    return historico
