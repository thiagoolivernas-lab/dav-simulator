import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .analise_estabilidade_dav import (
    EPS_B,
    analisar_estabilidade_local,
    identificar_modelo_local,
    mapa_estabilidade_kp,
)
from .calibracao import calcular_r_eff_dinamica
from .controle_dav import controlar_rpm
from .curva_hq_heartmate import calcular_ponto_operacao_hm3
from .fuzzy_dav import calcular_demanda_fuzzy
from .metricas_controle import calcular_metricas_controle
from .modelo_cardiovascular import (
    atualizar_pam_windkessel,
    calcular_p_eq_windkessel,
)


CENARIOS_FISIOLOGICOS = {
    "repouso": {"fc": 75, "imu": 0.10, "spo2": 98, "temp": 36.5},
    "esforco_leve": {"fc": 95, "imu": 0.35, "spo2": 97, "temp": 36.8},
    "esforco_moderado": {"fc": 115, "imu": 0.60, "spo2": 96, "temp": 37.1},
    "esforco_intenso": {"fc": 135, "imu": 0.85, "spo2": 95, "temp": 37.4},
    "hipoxemia": {"fc": 118, "imu": 0.12, "spo2": 89, "temp": 37.6},
}


def _hrv_por_fc(fc):
    return float(max(12.0, 109.5 - float(fc) * 0.75))


def _protecao_pam(pam_estimada, pam_min, pam_max, rpm_sugerida, rpm_atual):
    if pam_estimada > pam_max and rpm_sugerida > rpm_atual:
        return "bloqueio_aumento_rpm_pam_alta"

    if pam_estimada < pam_min and rpm_sugerida < rpm_atual:
        return "bloqueio_reducao_rpm_pam_baixa"

    return ""


def executar_malha_fechada_controle(
    pam_base=122.5,
    pam_alvo_serie=None,
    r_eff=24.0,
    tau=0.8,
    fluxo_base=3.0,
    rpm_inicial=5000.0,
    kp=20.0,
    ciclos=100,
    dt=1.0,
    rpm_min=3000.0,
    rpm_max=7000.0,
    pam_min=65.0,
    pam_max=140.0,
    demanda_serie=None,
    modo_curva="validado",
    cenario="basal",
):
    """
    Executa a malha fechada DAV ciclo a ciclo.

    Planta simplificada:
    Q[k] = ponto_operacao_HQ(RPM[k], R_eff)
    P_eq[k] = R_eff * Q[k]
    PAM[k+1] = PAM[k] + dt/tau * (P_eq[k] - PAM[k])

    Controlador:
    erro[k] = PAM_alvo[k] - PAM[k]
    RPM_sugerida[k] = RPM[k] + Kp * erro[k]
    RPM_aplicada[k+1] = saturacao(RPM_sugerida[k], RPM_min, RPM_max)
    """

    if pam_alvo_serie is None:
        pam_alvo_serie = [float(pam_base)] * int(ciclos)

    ciclos = int(min(ciclos, len(pam_alvo_serie)))
    pam_alvo_serie = [float(v) for v in pam_alvo_serie[:ciclos]]

    if demanda_serie is None:
        demanda_serie = [0.0] * ciclos

    pam_atual = float(pam_base)
    rpm_atual = float(rpm_inicial)

    historico = {
        "ciclo": [],
        "tempo": [],
        "cenario": [],
        "pam_alvo": [],
        "pam_estimada": [],
        "erro": [],
        "rpm_atual": [],
        "rpm_sugerida": [],
        "rpm_aplicada": [],
        "q_dav": [],
        "delta_p": [],
        "potencia": [],
        "eficiencia": [],
        "r_eff": [],
        "tau": [],
        "p_eq": [],
        "subpassos_integracao": [],
        "demanda": [],
        "saturacao_rpm": [],
        "protecao_pam": [],
        "faixa_hq": [],
    }

    for k in range(ciclos):
        pam_alvo = pam_alvo_serie[k]
        demanda = float(demanda_serie[k]) if k < len(demanda_serie) else 0.0

        ponto = calcular_ponto_operacao_hm3(
            rpm=rpm_atual,
            r_eff=r_eff,
            modo_curva=modo_curva,
        )
        q_dav = float(ponto["fluxo"])
        p_eq = calcular_p_eq_windkessel(
            r_eff=r_eff,
            q_dav=q_dav,
        )
        dinamica = atualizar_pam_windkessel(
            pam_atual=pam_atual,
            p_eq=p_eq,
            tau_base=tau,
            dt=dt,
        )
        pam_atual = dinamica["pam"]

        controle = controlar_rpm(
            pam_estimada=pam_atual,
            pam_alvo=pam_alvo,
            rpm_atual=rpm_atual,
            kp=kp,
            rpm_min=rpm_min,
            rpm_max=rpm_max,
            pam_min=pam_min,
            pam_max=pam_max,
        )
        rpm_sugerida = float(controle["rpm_sugerido"])
        rpm_aplicada = float(np.clip(rpm_sugerida, rpm_min, rpm_max))
        saturacao = abs(rpm_aplicada - rpm_sugerida) > 1e-9
        protecao = _protecao_pam(
            pam_atual,
            pam_min,
            pam_max,
            rpm_sugerida,
            rpm_atual,
        )

        historico["ciclo"].append(k + 1)
        historico["tempo"].append(float(k * dt))
        historico["cenario"].append(cenario)
        historico["pam_alvo"].append(float(pam_alvo))
        historico["pam_estimada"].append(float(pam_atual))
        historico["erro"].append(float(pam_alvo - pam_atual))
        historico["rpm_atual"].append(float(rpm_atual))
        historico["rpm_sugerida"].append(rpm_sugerida)
        historico["rpm_aplicada"].append(rpm_aplicada)
        historico["q_dav"].append(q_dav)
        historico["delta_p"].append(float(ponto["delta_p"]))
        historico["potencia"].append(float(ponto["potencia"]))
        historico["eficiencia"].append(float(ponto["eficiencia"]))
        historico["r_eff"].append(float(r_eff))
        historico["tau"].append(float(dinamica["tau"]))
        historico["p_eq"].append(float(p_eq))
        historico["subpassos_integracao"].append(int(dinamica["subpassos"]))
        historico["demanda"].append(float(demanda))
        historico["saturacao_rpm"].append(bool(saturacao))
        historico["protecao_pam"].append(protecao)
        historico["faixa_hq"].append(ponto.get("faixa_hq", ""))

        rpm_atual = rpm_aplicada

    return historico


def executar_ensaio_degrau_controle(
    pam_base=122.5,
    delta=5.0,
    ciclo_degrau=20,
    ciclos=120,
    **kwargs,
):
    """
    Ensaio de degrau da referencia:
    PAM_alvo[k] = PAM_base, k < ciclo_degrau
    PAM_alvo[k] = PAM_base + delta, k >= ciclo_degrau
    """

    pam_alvo = [
        float(pam_base) if k < int(ciclo_degrau)
        else float(pam_base + delta)
        for k in range(int(ciclos))
    ]

    return executar_malha_fechada_controle(
        pam_base=pam_base,
        pam_alvo_serie=pam_alvo,
        ciclos=ciclos,
        **kwargs,
    )


def executar_ensaio_perturbacao_local(
    pam_base=122.5,
    r_eff=24.0,
    tau=0.8,
    fluxo_base=3.0,
    rpm0=5000.0,
    perturbacoes=None,
    ciclos_por_perturbacao=12,
    modo_curva="validado",
    dt=1.0,
):
    """
    Excita explicitamente a planta com pequenos degraus de RPM.

    As demais condicoes ficam constantes. Assim, a identificacao de b mede a
    influencia local real da entrada RPM sobre a resposta transitoria da PAM.
    """

    if perturbacoes is None:
        perturbacoes = [-200, -100, 100, 200]

    rpm_min = 3000.0
    rpm_max = 9000.0 if modo_curva == "extrapolado" else 7000.0
    rpm_serie = []
    rpm_serie.extend([float(rpm0)] * 10)
    for pert in perturbacoes:
        rpm_serie.extend(
            [float(np.clip(rpm0 + pert, rpm_min, rpm_max))]
            * int(ciclos_por_perturbacao)
        )
        rpm_serie.extend([float(rpm0)] * 6)

    pam_atual = float(pam_base)
    historico = {
        "ciclo": [],
        "tempo": [],
        "cenario": [],
        "pam_alvo": [],
        "pam_estimada": [],
        "erro": [],
        "rpm_atual": [],
        "rpm_sugerida": [],
        "rpm_aplicada": [],
        "q_dav": [],
        "delta_p": [],
        "potencia": [],
        "eficiencia": [],
        "r_eff": [],
        "tau": [],
        "p_eq": [],
        "subpassos_integracao": [],
        "demanda": [],
        "saturacao_rpm": [],
        "protecao_pam": [],
        "faixa_hq": [],
    }

    for k, rpm in enumerate(rpm_serie):
        ponto = calcular_ponto_operacao_hm3(
            rpm=rpm,
            r_eff=r_eff,
            modo_curva=modo_curva,
        )
        q_dav = float(ponto["fluxo"])
        p_eq = calcular_p_eq_windkessel(
            r_eff=r_eff,
            q_dav=q_dav,
        )
        dinamica = atualizar_pam_windkessel(
            pam_atual=pam_atual,
            p_eq=p_eq,
            tau_base=tau,
            dt=dt,
        )
        pam_atual = dinamica["pam"]

        historico["ciclo"].append(k + 1)
        historico["tempo"].append(float(k * dt))
        historico["cenario"].append("perturbacao_local")
        historico["pam_alvo"].append(float(pam_base))
        historico["pam_estimada"].append(float(pam_atual))
        historico["erro"].append(float(pam_base - pam_atual))
        historico["rpm_atual"].append(float(rpm))
        historico["rpm_sugerida"].append(float(rpm))
        historico["rpm_aplicada"].append(float(rpm))
        historico["q_dav"].append(q_dav)
        historico["delta_p"].append(float(ponto["delta_p"]))
        historico["potencia"].append(float(ponto["potencia"]))
        historico["eficiencia"].append(float(ponto["eficiencia"]))
        historico["r_eff"].append(float(r_eff))
        historico["tau"].append(float(dinamica["tau"]))
        historico["p_eq"].append(float(p_eq))
        historico["subpassos_integracao"].append(int(dinamica["subpassos"]))
        historico["demanda"].append(0.0)
        historico["saturacao_rpm"].append(False)
        historico["protecao_pam"].append("")
        historico["faixa_hq"].append(ponto.get("faixa_hq", ""))

    return historico


def identificar_estabilidade_por_perturbacao(
    pam_base=122.5,
    r_eff=24.0,
    tau=0.8,
    fluxo_base=3.0,
    kp=20.0,
    rpm0=5000.0,
    modo_curva="validado",
):
    historico = executar_ensaio_perturbacao_local(
        pam_base=pam_base,
        r_eff=r_eff,
        tau=tau,
        fluxo_base=fluxo_base,
        rpm0=rpm0,
        perturbacoes=[-200, -100, 100, 200],
        modo_curva=modo_curva,
    )
    modelo = identificar_modelo_local(historico, epsilon_b=EPS_B)
    estabilidade = analisar_estabilidade_local(
        modelo.get("a"),
        modelo.get("b"),
        kp,
        epsilon_b=modelo.get("epsilon_b", EPS_B),
        identificacao_conclusiva=modelo.get("identificacao_conclusiva", False),
    )
    return {
        "historico": historico,
        "modelo_local": modelo,
        "estabilidade": estabilidade,
    }


def executar_cenario_controle(
    nome,
    estado,
    pam_base=122.5,
    r_eff_base=24.0,
    tau_base=0.8,
    fluxo_base=3.0,
    kp=20.0,
    modo_curva="validado",
    ciclos=120,
):
    fc = float(estado["fc"])
    hrv = float(estado.get("hrv", _hrv_por_fc(fc)))
    imu = float(estado["imu"])
    spo2 = float(estado["spo2"])
    temp = float(estado["temp"])

    fuzzy = calcular_demanda_fuzzy(
        fc_val=fc,
        hrv_val=hrv,
        imu_val=imu,
        spo2_val=spo2,
        temp_val=temp,
        PAM_base=pam_base,
        delta_PAM=15,
    )
    r_eff = calcular_r_eff_dinamica(
        r_eff_base=r_eff_base,
        fc=fc,
        hrv=hrv,
        imu=imu,
        spo2=spo2,
        temp=temp,
    )
    pam_alvo = float(fuzzy["PAM_alvo"])
    demanda = float(fuzzy["demanda"])

    historico = executar_ensaio_degrau_controle(
        pam_base=pam_base,
        delta=pam_alvo - pam_base,
        ciclo_degrau=20,
        ciclos=ciclos,
        r_eff=r_eff,
        tau=tau_base,
        fluxo_base=fluxo_base,
        kp=kp,
        rpm_min=3000,
        rpm_max=9000 if modo_curva == "extrapolado" else 7000,
        modo_curva=modo_curva,
        demanda_serie=[demanda] * ciclos,
        cenario=nome,
    )
    metricas = calcular_metricas_controle(historico)
    analise_local = identificar_estabilidade_por_perturbacao(
        pam_base=pam_base,
        r_eff=r_eff,
        tau=tau_base,
        fluxo_base=fluxo_base,
        kp=kp,
        modo_curva=modo_curva,
    )

    return {
        "nome": nome,
        "estado": {
            "fc": fc,
            "hrv": hrv,
            "imu": imu,
            "spo2": spo2,
            "temp": temp,
        },
        "fuzzy": fuzzy,
        "r_eff": r_eff,
        "pam_inicial": pam_base,
        "pam_alvo": pam_alvo,
        "historico": historico,
        "metricas": metricas,
        "modelo_local": analise_local["modelo_local"],
        "estabilidade": analise_local["estabilidade"],
        "perturbacao_local": analise_local["historico"],
    }


def executar_teste_multicenario(**kwargs):
    resultados = []
    for nome, estado in CENARIOS_FISIOLOGICOS.items():
        resultados.append(
            executar_cenario_controle(nome, estado, **kwargs)
        )
    return resultados


def executar_teste_robustez(
    pam_base=122.5,
    r_eff_base=24.0,
    tau_base=0.8,
    fluxo_base=3.0,
    kp=20.0,
    modo_curva="validado",
):
    variacoes = []

    for fator in [0.8, 0.9, 1.0, 1.1, 1.2]:
        variacoes.append(("R_eff", fator, r_eff_base * fator, tau_base, fluxo_base))

    for fator in [0.8, 0.9, 1.0, 1.1, 1.2]:
        variacoes.append(("tau", fator, r_eff_base, tau_base * fator, fluxo_base))

    resultados = []
    for nome_param, fator, r_eff, tau, fluxo in variacoes:
        hist = executar_ensaio_degrau_controle(
            pam_base=pam_base,
            delta=10,
            ciclo_degrau=20,
            ciclos=100,
            r_eff=r_eff,
            tau=tau,
            fluxo_base=fluxo,
            kp=kp,
            rpm_min=3000,
            rpm_max=9000 if modo_curva == "extrapolado" else 7000,
            modo_curva=modo_curva,
            cenario=f"robustez_{nome_param}_{fator}",
        )
        metricas = calcular_metricas_controle(hist)
        analise_local = identificar_estabilidade_por_perturbacao(
            pam_base=pam_base,
            r_eff=r_eff,
            tau=tau,
            fluxo_base=fluxo,
            kp=kp,
            modo_curva=modo_curva,
        )
        resultados.append({
            "cenario": "degrau_10",
            "parametro": nome_param,
            "fator": fator,
            "metricas": metricas,
            "modelo_local": analise_local["modelo_local"],
            "estabilidade": analise_local["estabilidade"],
            "perturbacao_local": analise_local["historico"],
        })

    return resultados


def historico_para_linhas(historico, prefixo=None):
    chaves = list(historico.keys())
    quantidade = len(historico.get("ciclo", []))
    linhas = []
    for i in range(quantidade):
        linha = {}
        if prefixo:
            linha.update(prefixo)
        for chave in chaves:
            valor = historico[chave]
            if isinstance(valor, list) and i < len(valor):
                linha[chave] = valor[i]
        linhas.append(linha)
    return linhas


def salvar_resultados_validacao(resultado, pasta_saida):
    pasta = Path(pasta_saida)
    pasta.mkdir(parents=True, exist_ok=True)
    csv_path = pasta / "validacao_controle.csv"
    json_path = pasta / "validacao_controle.json"

    linhas = []
    data_hora = datetime.now().isoformat(timespec="seconds")

    for ensaio in resultado.get("ensaios_degrau", []):
        linhas.extend(historico_para_linhas(
            ensaio["historico"],
            {
                "data_hora": data_hora,
                "tipo": "degrau",
                "cenario": ensaio["nome"],
            },
        ))

    for cenario in resultado.get("multicenario", []):
        linhas.extend(historico_para_linhas(
            cenario["historico"],
            {
                "data_hora": data_hora,
                "tipo": "multicenario",
                "cenario": cenario["nome"],
            },
        ))

    if linhas:
        campos = sorted({chave for linha in linhas for chave in linha.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as arquivo:
            writer = csv.DictWriter(arquivo, fieldnames=campos)
            writer.writeheader()
            writer.writerows(linhas)

    resumo = {
        "data_hora": data_hora,
        "parametros": resultado.get("parametros"),
        "ensaios_degrau_metricas": [
            {
                "nome": e["nome"],
                "metricas": e["metricas"],
                "modelo_local": e["modelo_local"],
                "estabilidade": e["estabilidade"],
            }
            for e in resultado.get("ensaios_degrau", [])
        ],
        "multicenario": [
            {
                "nome": c["nome"],
                "estado": c["estado"],
                "r_eff": c["r_eff"],
                "pam_alvo": c["pam_alvo"],
                "metricas": c["metricas"],
                "modelo_local": c["modelo_local"],
                "estabilidade": c["estabilidade"],
            }
            for c in resultado.get("multicenario", [])
        ],
        "robustez": resultado.get("robustez", []),
        "mapa_kp": resultado.get("mapa_kp"),
    }

    with json_path.open("w", encoding="utf-8") as arquivo:
        json.dump(resumo, arquivo, ensure_ascii=False, indent=2)

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "linhas_csv": len(linhas),
    }


def executar_validacao_dinamica_controle(
    pam_base=122.5,
    r_eff_base=24.0,
    tau_base=0.8,
    fluxo_base=3.0,
    kp=20.0,
    modo_curva="validado",
    pasta_saida=None,
):
    rpm_max = 9000 if modo_curva == "extrapolado" else 7000
    parametros = {
        "kp": float(kp),
        "rpm_min": 3000,
        "rpm_max": rpm_max,
        "pam_inicial": float(pam_base),
        "pam_alvo": float(pam_base + 10),
        "modo_curva": modo_curva,
        "faixa_hq": "extrapolada" if modo_curva == "extrapolado" else "validada",
        "r_eff_base": float(r_eff_base),
        "tau_base": float(tau_base),
        "fluxo_base": float(fluxo_base),
    }

    analise_base = identificar_estabilidade_por_perturbacao(
        pam_base=pam_base,
        r_eff=r_eff_base,
        tau=tau_base,
        fluxo_base=fluxo_base,
        kp=kp,
        modo_curva=modo_curva,
    )

    ensaios = []
    for delta in (5, 10, -5):
        hist = executar_ensaio_degrau_controle(
            pam_base=pam_base,
            delta=delta,
            ciclo_degrau=20,
            ciclos=120,
            r_eff=r_eff_base,
            tau=tau_base,
            fluxo_base=fluxo_base,
            kp=kp,
            rpm_min=3000,
            rpm_max=rpm_max,
            modo_curva=modo_curva,
            cenario=f"degrau_{delta:+g}",
        )
        metricas = calcular_metricas_controle(hist)
        ensaios.append({
            "nome": f"Degrau {delta:+g} mmHg",
            "delta": delta,
            "historico": hist,
            "metricas": metricas,
            "modelo_local": analise_base["modelo_local"],
            "estabilidade": analise_base["estabilidade"],
        })

    perturbacao = analise_base["historico"]
    modelo_perturbacao = analise_base["modelo_local"]
    estabilidade_perturbacao = analise_base["estabilidade"]
    mapa_kp = mapa_estabilidade_kp(
        modelo_perturbacao.get("a"),
        modelo_perturbacao.get("b"),
        kp_min=1,
        kp_max=50,
        epsilon_b=modelo_perturbacao.get("epsilon_b", EPS_B),
    )

    multicenario = executar_teste_multicenario(
        pam_base=pam_base,
        r_eff_base=r_eff_base,
        tau_base=tau_base,
        fluxo_base=fluxo_base,
        kp=kp,
        modo_curva=modo_curva,
        ciclos=120,
    )
    robustez = executar_teste_robustez(
        pam_base=pam_base,
        r_eff_base=r_eff_base,
        tau_base=tau_base,
        fluxo_base=fluxo_base,
        kp=kp,
        modo_curva=modo_curva,
    )

    resultado = {
        "parametros": parametros,
        "ensaios_degrau": ensaios,
        "perturbacao_local": {
            "historico": perturbacao,
            "modelo_local": modelo_perturbacao,
            "estabilidade": estabilidade_perturbacao,
        },
        "mapa_kp": mapa_kp,
        "multicenario": multicenario,
        "robustez": robustez,
        "arquivos": None,
    }

    if pasta_saida:
        resultado["arquivos"] = salvar_resultados_validacao(
            resultado,
            pasta_saida,
        )

    return resultado
