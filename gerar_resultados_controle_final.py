import csv
import json
import os
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "meusistema.settings")

import django

django.setup()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from DAVsimulator.services.analise_estabilidade_dav import (
    EPS_B,
    analisar_estabilidade_local,
    identificar_modelo_local,
    mapa_estabilidade_kp,
)
from DAVsimulator.services.calibracao import calcular_r_eff_dinamica
from DAVsimulator.services.curva_hq_heartmate import (
    calcular_ponto_operacao_hm3,
    obter_curvas_hq,
)
from DAVsimulator.services.fuzzy_dav import calcular_demanda_fuzzy
from DAVsimulator.services.metricas_controle import calcular_metricas_controle
from DAVsimulator.services.validacao_controle_dav import (
    CENARIOS_FISIOLOGICOS,
    _hrv_por_fc,
    executar_ensaio_degrau_controle,
    executar_malha_fechada_controle,
    executar_teste_multicenario,
    identificar_estabilidade_por_perturbacao,
)
from gerar_resultados_dissertacao_final import (
    analisar_segmento,
    carregar_segmentos,
    fnum,
)


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "resultados_controle_final"
FIG_DIR = OUT_DIR / "figuras"
SEGMENTO = "3000003_0008"
KP = 20.0
RPM_INICIAL = 5000.0
RPM_MIN = 3000.0
RPM_MAX = 7000.0
MODO_CURVA = "validado"


def write_csv(path, linhas, campos=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if campos is None:
        campos = sorted({k for linha in linhas for k in linha.keys()})
    with path.open("w", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(linhas)
    return path


def historico_linhas(historico, extra=None):
    extra = extra or {}
    n = len(historico.get("ciclo", []))
    linhas = []
    for i in range(n):
        linha = dict(extra)
        for chave, valores in historico.items():
            if isinstance(valores, list) and i < len(valores):
                linha[chave] = valores[i]
        linhas.append(linha)
    return linhas


def carregar_calibracao_principal():
    _, segmentos = carregar_segmentos()
    segmento = next((s for s in segmentos if s.get("nome") == SEGMENTO), None)
    if segmento is None:
        raise RuntimeError(f"Segmento {SEGMENTO} nao encontrado.")

    analise = analisar_segmento(segmento)
    if not analise or not analise.get("calibracao"):
        raise RuntimeError(f"Nao foi possivel calibrar o segmento {SEGMENTO}.")

    return analise["calibracao"], analise


def metricas_linha(metricas, prefixo=None):
    linha = dict(prefixo or {})
    for chave in (
        "erro_regime",
        "erro_regime_final",
        "mae",
        "rmse",
        "rise_time",
        "settling_time_2pct",
        "settling_time_1mmhg",
        "overshoot_pct",
        "iae",
        "ise",
        "itae",
        "esforco_controle",
        "percentual_saturacao",
        "amostras",
    ):
        linha[chave] = fnum(metricas.get(chave), 9)
    linha["mensagem"] = metricas.get("mensagem") or ""
    return linha


def autovalor_linha(estabilidade):
    aut = estabilidade.get("autovalores") or []
    mod = estabilidade.get("modulos") or []
    return {
        "lambda1": json.dumps(aut[0], ensure_ascii=False) if len(aut) > 0 else "",
        "lambda2": json.dumps(aut[1], ensure_ascii=False) if len(aut) > 1 else "",
        "lambda1_mod": fnum(mod[0], 12) if len(mod) > 0 else "",
        "lambda2_mod": fnum(mod[1], 12) if len(mod) > 1 else "",
        "rho": fnum(estabilidade.get("raio_espectral"), 12),
        "estavel": estabilidade.get("estavel"),
        "classificacao": estabilidade.get("classificacao"),
        "mensagem_estabilidade": estabilidade.get("mensagem") or "",
    }


def gerar_controle_e_degrau(cal):
    hist = executar_ensaio_degrau_controle(
        pam_base=cal["pam_base"],
        delta=10.0,
        ciclo_degrau=20,
        ciclos=120,
        r_eff=cal["r_eff"],
        tau=cal["tau_base"],
        fluxo_base=cal["fluxo_base"],
        rpm_inicial=RPM_INICIAL,
        kp=KP,
        rpm_min=RPM_MIN,
        rpm_max=RPM_MAX,
        modo_curva=MODO_CURVA,
        cenario="degrau_+10",
    )
    linhas = historico_linhas(hist, {
        "segmento": SEGMENTO,
        "ensaio": "degrau_+10",
        "pam_base_precalibracao": cal["pam_base"],
        "tau_base_precalibracao": cal["tau_base"],
        "q_base_precalibracao": cal["fluxo_base"],
        "r_eff_base_precalibracao": cal["r_eff"],
        "kp": KP,
    })
    write_csv(OUT_DIR / "controle_nominal.csv", linhas)

    metricas = calcular_metricas_controle(hist)
    met_linha = metricas_linha(metricas, {
        "segmento": SEGMENTO,
        "ensaio": "degrau_+10",
        "pam_base": cal["pam_base"],
        "pam_alvo_final": cal["pam_base"] + 10.0,
        "tau_base": cal["tau_base"],
        "q_base": cal["fluxo_base"],
        "r_eff_base": cal["r_eff"],
        "kp": KP,
        "rpm_inicial": RPM_INICIAL,
        "rpm_max_observada": fnum(max(hist["rpm_aplicada"]), 9),
        "rpm_final": fnum(hist["rpm_aplicada"][-1], 9),
        "q_final": fnum(hist["q_dav"][-1], 9),
        "pam_final": fnum(hist["pam_estimada"][-1], 9),
        "saturacao_ocorreu": any(hist["saturacao_rpm"]),
        "atingiu_limite_rpm": max(hist["rpm_aplicada"]) >= RPM_MAX - 1e-9
        or min(hist["rpm_aplicada"]) <= RPM_MIN + 1e-9,
        "faixa_hq_usada": ",".join(sorted(set(hist["faixa_hq"]))),
    })
    write_csv(OUT_DIR / "metricas_controle.csv", [met_linha])
    return hist, metricas, met_linha


def gerar_identificacao_e_estabilidade(cal):
    analise = identificar_estabilidade_por_perturbacao(
        pam_base=cal["pam_base"],
        r_eff=cal["r_eff"],
        tau=cal["tau_base"],
        fluxo_base=cal["fluxo_base"],
        kp=KP,
        rpm0=RPM_INICIAL,
        modo_curva=MODO_CURVA,
    )
    hist = analise["historico"]
    modelo = analise["modelo_local"]
    estabilidade = analise["estabilidade"]

    resumo = {
        "tipo": "resumo_modelo",
        "segmento": SEGMENTO,
        "a": fnum(modelo.get("a"), 12),
        "a_formatado": modelo.get("a_formatado"),
        "b": fnum(modelo.get("b"), 12),
        "b_formatado": modelo.get("b_formatado"),
        "c": fnum(modelo.get("c"), 12),
        "r2": fnum(modelo.get("r2"), 12),
        "rmse": fnum(modelo.get("rmse"), 12),
        "pam0": fnum(modelo.get("pam0"), 12),
        "rpm0": fnum(modelo.get("rpm0"), 12),
        "epsilon_b": modelo.get("epsilon_b"),
        "identificacao_conclusiva": modelo.get("identificacao_conclusiva"),
        "classificacao_identificacao": modelo.get("classificacao_identificacao"),
        "mensagem": modelo.get("mensagem") or "",
        **autovalor_linha(estabilidade),
    }
    linhas = [resumo]
    linhas.extend(historico_linhas(hist, {"tipo": "dados_perturbacao", "segmento": SEGMENTO}))
    write_csv(OUT_DIR / "identificacao_local.csv", linhas)

    mapa = mapa_estabilidade_kp(
        modelo.get("a"),
        modelo.get("b"),
        kp_min=1,
        kp_max=100,
        epsilon_b=modelo.get("epsilon_b", EPS_B),
    )
    linhas_kp = []
    for item in mapa.get("resultados", []):
        aut = item.get("autovalores") or []
        mod = [abs(complex(v["real"], v["imag"])) for v in aut]
        linhas_kp.append({
            "kp": item["kp"],
            "lambda1_real": fnum(aut[0]["real"], 12) if len(aut) > 0 else "",
            "lambda1_imag": fnum(aut[0]["imag"], 12) if len(aut) > 0 else "",
            "lambda2_real": fnum(aut[1]["real"], 12) if len(aut) > 1 else "",
            "lambda2_imag": fnum(aut[1]["imag"], 12) if len(aut) > 1 else "",
            "lambda1_mod": fnum(mod[0], 12) if len(mod) > 0 else "",
            "lambda2_mod": fnum(mod[1], 12) if len(mod) > 1 else "",
            "rho": fnum(item.get("raio_espectral"), 12),
            "estavel": item.get("estavel"),
            "classificacao": "ESTAVEL" if item.get("estavel") else "INSTAVEL",
        })
    write_csv(OUT_DIR / "estabilidade_kp.csv", linhas_kp)
    return analise, mapa, linhas_kp


def gerar_multicenario(cal):
    resultados = executar_teste_multicenario(
        pam_base=cal["pam_base"],
        r_eff_base=cal["r_eff"],
        tau_base=cal["tau_base"],
        fluxo_base=cal["fluxo_base"],
        kp=KP,
        modo_curva=MODO_CURVA,
        ciclos=120,
    )
    linhas = []
    for item in resultados:
        hist = item["historico"]
        metricas = item["metricas"]
        estabilidade = item["estabilidade"]
        estado = item["estado"]
        fuzzy = item["fuzzy"]
        linhas.append({
            "cenario": item["nome"],
            "fc": estado["fc"],
            "hrv": estado["hrv"],
            "imu": estado["imu"],
            "spo2": estado["spo2"],
            "temperatura": estado["temp"],
            "d_fuzzy": fnum(fuzzy["demanda"], 9),
            "classe_fuzzy": fuzzy.get("classe"),
            "alerta_clinico": "alerta" in str(fuzzy.get("classe", "")).lower(),
            "pam_base": fnum(cal["pam_base"], 9),
            "pam_alvo": fnum(item["pam_alvo"], 9),
            "r_eff_utilizada": fnum(item["r_eff"], 9),
            "r_eff_modulacao": fnum(item["r_eff"] / cal["r_eff"], 9),
            "tau": fnum(cal["tau_base"], 9),
            "rpm_final": fnum(hist["rpm_aplicada"][-1], 9),
            "rpm_max": fnum(max(hist["rpm_aplicada"]), 9),
            "atingiu_limite_rpm": max(hist["rpm_aplicada"]) >= RPM_MAX - 1e-9
            or min(hist["rpm_aplicada"]) <= RPM_MIN + 1e-9,
            "q_dav_final": fnum(hist["q_dav"][-1], 9),
            "pam_final": fnum(hist["pam_estimada"][-1], 9),
            "erro_estacionario": fnum(metricas["erro_regime"], 9),
            "rmse": fnum(metricas["rmse"], 9),
            "overshoot": fnum(metricas["overshoot_pct"], 9),
            "saturacao": fnum(metricas["percentual_saturacao"], 9),
            "faixa_hq": ",".join(sorted(set(hist["faixa_hq"]))),
            **autovalor_linha(estabilidade),
        })
    write_csv(OUT_DIR / "controle_multicenario.csv", linhas)
    return resultados, linhas


def gerar_robustez(cal):
    variacoes = []
    for fator in [0.8, 0.9, 1.0, 1.1, 1.2]:
        variacoes.append(("R_eff", fator, cal["r_eff"] * fator, cal["tau_base"], cal["fluxo_base"]))
    for fator in [0.8, 0.9, 1.0, 1.1, 1.2]:
        variacoes.append(("tau", fator, cal["r_eff"], cal["tau_base"] * fator, cal["fluxo_base"]))
    for fator in [0.8, 1.0, 1.2]:
        variacoes.append(("Q_base", fator, cal["r_eff"], cal["tau_base"], cal["fluxo_base"] * fator))

    linhas = []
    for parametro, fator, r_eff, tau, fluxo_base in variacoes:
        hist = executar_ensaio_degrau_controle(
            pam_base=cal["pam_base"],
            delta=10.0,
            ciclo_degrau=20,
            ciclos=120,
            r_eff=r_eff,
            tau=tau,
            fluxo_base=fluxo_base,
            rpm_inicial=RPM_INICIAL,
            kp=KP,
            rpm_min=RPM_MIN,
            rpm_max=RPM_MAX,
            modo_curva=MODO_CURVA,
            cenario=f"robustez_{parametro}_{fator}",
        )
        metricas = calcular_metricas_controle(hist)
        analise = identificar_estabilidade_por_perturbacao(
            pam_base=cal["pam_base"],
            r_eff=r_eff,
            tau=tau,
            fluxo_base=fluxo_base,
            kp=KP,
            rpm0=RPM_INICIAL,
            modo_curva=MODO_CURVA,
        )
        linhas.append({
            "parametro": parametro,
            "fator": fator,
            "r_eff": fnum(r_eff, 9),
            "tau": fnum(tau, 9),
            "q_base": fnum(fluxo_base, 9),
            "pam_final": fnum(hist["pam_estimada"][-1], 9),
            "erro_estacionario": fnum(metricas["erro_regime"], 9),
            "rmse": fnum(metricas["rmse"], 9),
            "overshoot": fnum(metricas["overshoot_pct"], 9),
            "rpm_final": fnum(hist["rpm_aplicada"][-1], 9),
            "rpm_max": fnum(max(hist["rpm_aplicada"]), 9),
            "atingiu_limite_rpm": max(hist["rpm_aplicada"]) >= RPM_MAX - 1e-9
            or min(hist["rpm_aplicada"]) <= RPM_MIN + 1e-9,
            "saturacao": fnum(metricas["percentual_saturacao"], 9),
            "faixa_hq": ",".join(sorted(set(hist["faixa_hq"]))),
            **autovalor_linha(analise["estabilidade"]),
        })
    write_csv(OUT_DIR / "robustez_parametrica.csv", linhas)
    return linhas


def auditar_residuos():
    termos = [
        "122.5",
        "24.0",
        "r_eff=24",
        "0.8",
        "1.048",
        "15*(Q-Q_base)",
        "PAM_base + 15*(Q-Q_base)",
        "PAM_base + 15",
    ]
    arquivos = [BASE_DIR / "gerar_resultados_dissertacao_final.py"]
    arquivos.extend((BASE_DIR / "DAVsimulator").rglob("*.py"))
    linhas = []
    for arquivo in arquivos:
        texto = arquivo.read_text(encoding="utf-8", errors="ignore").splitlines()
        for num, linha in enumerate(texto, start=1):
            for termo in termos:
                if termo in linha:
                    linhas.append({
                        "termo": termo,
                        "arquivo": str(arquivo),
                        "linha": num,
                        "conteudo": linha.strip(),
                        "participa_resultados_controle_final": "nao; script usa calibracao recalculada do segmento 3000003_0008",
                    })
    write_csv(OUT_DIR / "auditoria_valores_residuais.csv", linhas)
    return linhas


def plotar_figuras(cal, hist, linhas_kp, multicenario_linhas):
    qs = np.linspace(0, 8, 120)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    curvas = obter_curvas_hq(MODO_CURVA)
    for rpm in [3000, 4000, 5000, 6000, 7000]:
        pontos = curvas[rpm]
        ax.plot(
            [p[0] for p in pontos],
            [p[1] for p in pontos],
            alpha=0.55,
            linewidth=1.2,
            label=f"{rpm} rpm",
        )
    sistema = cal["r_eff"] * qs
    ax.plot(qs, sistema, color="black", linestyle="--", linewidth=1.6, label="sistema cardiovascular")
    ponto = calcular_ponto_operacao_hm3(rpm=RPM_INICIAL, r_eff=cal["r_eff"], modo_curva=MODO_CURVA)
    ax.scatter([ponto["fluxo"]], [ponto["delta_p"]], color="#d62728", s=50, label="ponto inicial")
    ax.set_xlabel("Fluxo (L/min)")
    ax.set_ylabel("DeltaP (mmHg)")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "FIG_HQ_PONTO_OPERACAO.png", dpi=300)
    plt.close(fig)

    t = hist["tempo"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, hist["pam_alvo"], label="PAM alvo", color="#d62728", linestyle="--")
    axes[0].plot(t, hist["pam_estimada"], label="PAM estimada", color="#1f77b4")
    axes[0].set_ylabel("PAM (mmHg)")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.22)
    axes[1].plot(t, hist["erro"], color="#9467bd")
    axes[1].set_ylabel("Erro (mmHg)")
    axes[1].grid(True, alpha=0.22)
    axes[2].plot(t, hist["rpm_aplicada"], color="#2ca02c", label="RPM")
    axes[2].set_ylabel("RPM")
    axes[2].set_xlabel("Tempo (s)")
    axes[2].grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "FIG_CONTROLE_DINAMICO.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    if linhas_kp:
        ax.plot([x["kp"] for x in linhas_kp], [float(x["rho"]) for x in linhas_kp], color="#1f77b4")
    ax.axhline(1.0, color="#d62728", linestyle="--", label="limite de estabilidade")
    ax.set_xlabel("Kp")
    ax.set_ylabel("rho(Acl)")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "FIG_ESTABILIDADE_KP.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    nomes = [x["cenario"] for x in multicenario_linhas]
    axes[0].bar(nomes, [float(x["pam_alvo"]) for x in multicenario_linhas], alpha=0.55, label="PAM alvo")
    axes[0].plot(nomes, [float(x["pam_final"]) for x in multicenario_linhas], marker="o", color="#d62728", label="PAM final")
    axes[0].set_ylabel("PAM (mmHg)")
    axes[0].legend(frameon=False)
    axes[0].grid(True, axis="y", alpha=0.22)
    axes[1].bar(nomes, [float(x["d_fuzzy"]) for x in multicenario_linhas], color="#2ca02c")
    axes[1].set_ylabel("D fuzzy")
    axes[1].set_xlabel("Cenario")
    axes[1].grid(True, axis="y", alpha=0.22)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "FIG_CONTROLE_MULTICENARIO.png", dpi=300)
    plt.close(fig)


def gerar_relatorio(cal, met_linha, ident, mapa, multicenario_linhas, robustez_linhas, residuos):
    modelo = ident["modelo_local"]
    estabilidade = ident["estabilidade"]
    hipoxemia = next(x for x in multicenario_linhas if x["cenario"] == "hipoxemia")
    extrapolacao = any("extrapolada" in str(x.get("faixa_hq", "")) for x in multicenario_linhas + robustez_linhas)
    saturacao = (
        bool(met_linha["saturacao_ocorreu"])
        or bool(met_linha.get("atingiu_limite_rpm"))
        or any(
            float(x.get("saturacao") or 0) > 0 or bool(x.get("atingiu_limite_rpm"))
            for x in multicenario_linhas + robustez_linhas
        )
    )
    rel = []
    rel.append("# RELATORIO_CONTROLE_FINAL\n")
    rel.append("## Base individualizada\n")
    rel.append(f"- Segmento: `{SEGMENTO}`\n")
    rel.append(f"- PAM_base: {cal['pam_base']:.6f} mmHg\n")
    rel.append(f"- tau_base: {cal['tau_base']:.6f} s\n")
    rel.append(f"- Q_base: {cal['fluxo_base']:.6f} L/min\n")
    rel.append(f"- R_eff_base: {cal['r_eff']:.6f} mmHg/(L/min)\n")
    rel.append(f"- Kp: {KP:.6f}\n")
    rel.append("\n## Controle ativo\n")
    rel.append("- Realimentacao: `pam_estimada`, calculada pela dinamica Windkessel apos o ponto H-Q da bomba.\n")
    rel.append("- Atualizacao da PAM: `PAM[k+1] = PAM[k] + dt/tau * (P_eq[k] - PAM[k])`, com integracao de Euler subdividida.\n")
    rel.append("- Pressao de equilibrio: `P_eq = R_eff * (Q_DAV + Q_nativo)`. Na implementacao atual, `Q_nativo` existe na assinatura, mas permanece zero por ausencia de modelo validado.\n")
    rel.append("- Atualizacao da RPM: `RPM_sugerida[k] = RPM[k] + Kp * (PAM_alvo[k] - PAM_estimada[k])`, seguida de limites e protecoes de PAM.\n")
    rel.append("\n## Ensaio de degrau +10 mmHg\n")
    for chave in ("erro_regime", "mae", "rmse", "rise_time", "settling_time_2pct", "overshoot_pct", "iae", "ise", "itae", "esforco_controle", "rpm_max_observada", "saturacao_ocorreu"):
        rel.append(f"- {chave}: {met_linha.get(chave)}\n")
    rel.append("\n## Identificacao local e estabilidade\n")
    rel.append(f"- a: {modelo.get('a_formatado')}\n")
    rel.append(f"- b: {modelo.get('b_formatado')}\n")
    rel.append(f"- R2: {modelo.get('r2')}\n")
    rel.append(f"- RMSE: {modelo.get('rmse')}\n")
    rel.append(f"- rho(Acl): {estabilidade.get('raio_espectral')}\n")
    rel.append(f"- classificacao: {estabilidade.get('classificacao')}\n")
    rel.append("\n## Hipoxemia\n")
    rel.append(f"- Entrada: FC=118 bpm, HRV={hipoxemia['hrv']} ms, IMU=0.12, SpO2=89%, temperatura=37.6 C\n")
    rel.append(f"- D fuzzy: {hipoxemia['d_fuzzy']}\n")
    rel.append(f"- PAM alvo: {hipoxemia['pam_alvo']} mmHg\n")
    rel.append("\n## H-Q e saturacao\n")
    rel.append(f"- Houve extrapolacao H-Q nos resultados principais: {extrapolacao}\n")
    rel.append(f"- Houve saturacao/limite de RPM: {saturacao}\n")
    rel.append("\n## Valores residuais auditados\n")
    rel.append(f"- Ocorrencias encontradas no codigo/resultados antigos: {len(residuos)}. Ver `auditoria_valores_residuais.csv`.\n")
    rel.append("- Os resultados desta pasta foram gerados com a calibracao recalculada do segmento principal, nao com a bancada antiga.\n")
    rel.append("\n## Arquivos principais\n")
    for nome in [
        "controle_nominal.csv",
        "metricas_controle.csv",
        "identificacao_local.csv",
        "estabilidade_kp.csv",
        "controle_multicenario.csv",
        "robustez_parametrica.csv",
    ]:
        rel.append(f"- {nome}\n")
    (OUT_DIR / "RELATORIO_CONTROLE_FINAL.md").write_text("".join(rel), encoding="utf-8")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    cal, _ = carregar_calibracao_principal()

    hist, metricas, met_linha = gerar_controle_e_degrau(cal)
    ident, mapa, linhas_kp = gerar_identificacao_e_estabilidade(cal)
    multicenario, multicenario_linhas = gerar_multicenario(cal)
    robustez_linhas = gerar_robustez(cal)
    residuos = auditar_residuos()
    plotar_figuras(cal, hist, linhas_kp, multicenario_linhas)
    gerar_relatorio(cal, met_linha, ident, mapa, multicenario_linhas, robustez_linhas, residuos)

    resumo = {
        "saida": str(OUT_DIR),
        "segmento": SEGMENTO,
        "pam_base": cal["pam_base"],
        "tau_base": cal["tau_base"],
        "q_base": cal["fluxo_base"],
        "r_eff_base": cal["r_eff"],
        "kp": KP,
        "realimentacao": "pam_estimada",
        "q_nativo": "assinatura existe; valor padrao 0.0",
        "degrau_metricas": met_linha,
        "a": ident["modelo_local"].get("a"),
        "b": ident["modelo_local"].get("b"),
        "rho": ident["estabilidade"].get("raio_espectral"),
        "hipoxemia": next(x for x in multicenario_linhas if x["cenario"] == "hipoxemia"),
        "extrapolacao_hq": any("extrapolada" in str(x.get("faixa_hq", "")) for x in multicenario_linhas + robustez_linhas),
        "saturacao": (
            bool(met_linha["saturacao_ocorreu"])
            or bool(met_linha.get("atingiu_limite_rpm"))
            or any(
                float(x.get("saturacao") or 0) > 0 or bool(x.get("atingiu_limite_rpm"))
                for x in multicenario_linhas + robustez_linhas
            )
        ),
    }
    print("RESULTADOS_CONTROLE_FINAL", json.dumps(resumo, ensure_ascii=False))


if __name__ == "__main__":
    main()
