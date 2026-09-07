import argparse
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

from DAVsimulator.models import MIMICRecord
from DAVsimulator.services.analise_estabilidade_dav import (
    analisar_estabilidade_local,
    identificar_modelo_local,
    mapa_estabilidade_kp,
)
from DAVsimulator.services.calibracao import calcular_r_eff_dinamica
from DAVsimulator.services.curva_hq_heartmate import calcular_ponto_operacao_hm3
from DAVsimulator.services.fuzzy_dav import (
    DEMANDA_CTRL,
    SKFUZZY_DISPONIVEL,
    avaliar_fatores_clinicos,
    calcular_demanda_fallback,
    calcular_demanda_fuzzy,
)
from DAVsimulator.services.hemodinamica import (
    estimar_tau_batimento,
    segmentar_abp_por_picos_r,
)
from DAVsimulator.services.metricas_controle import calcular_metricas_controle
from DAVsimulator.services.modelo_cardiovascular import (
    atualizar_pam_windkessel,
    calcular_p_eq_windkessel,
)
from DAVsimulator.services.validacao_controle_dav import (
    CENARIOS_FISIOLOGICOS,
    _hrv_por_fc,
    executar_malha_fechada_controle,
    executar_teste_multicenario,
)
from DAVsimulator.services.wfdb_parser import (
    ler_waveform_canais,
    listar_segmentos_extraidos,
)
from DAVsimulator.views import (
    _analisar_abp,
    _analisar_ecg,
    _calibrar_paciente,
    _estimar_tau_e_pam,
)


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "resultados_dissertacao_final"
FIG_DIR = OUT_DIR / "figuras"
SEGMENTO_PRINCIPAL = "3000003_0008"


def fnum(valor, casas=6):
    if valor is None:
        return ""
    try:
        if isinstance(valor, float) and (np.isnan(valor) or np.isinf(valor)):
            return ""
        return round(float(valor), casas)
    except Exception:
        return valor


def write_csv(nome, linhas, campos=None):
    path = OUT_DIR / nome
    if campos is None:
        campos = sorted({k for linha in linhas for k in linha.keys()})
    with path.open("w", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(linhas)
    return path


def write_csv_path(path, linhas, campos=None):
    if campos is None:
        campos = sorted({k for linha in linhas for k in linha.keys()})
    with path.open("w", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(linhas)
    return path


def fuzzy_detalhado(fc, hrv, imu, spo2, temp, pam_base):
    fatores, alerta = avaliar_fatores_clinicos(fc, hrv, imu, spo2, temp)

    if SKFUZZY_DISPONIVEL and DEMANDA_CTRL is not None:
        from skfuzzy import control as ctrl

        sim = ctrl.ControlSystemSimulation(DEMANDA_CTRL)
        sim.input["fc"] = float(fc)
        sim.input["hrv"] = float(hrv)
        sim.input["imu"] = float(imu)
        sim.input["spo2"] = float(spo2)
        sim.input["temp"] = float(temp)
        sim.compute()
        d_inicial = float(sim.output.get("demanda", 0.0))
        metodo = "scikit-fuzzy"
    else:
        d_inicial = calcular_demanda_fallback(fc, hrv, imu, spo2, temp)
        metodo = "fallback"

    d_final = max(d_inicial, 0.88) if alerta else d_inicial
    d_final = float(np.clip(d_final, 0, 1))

    fuzzy = calcular_demanda_fuzzy(
        fc_val=fc,
        hrv_val=hrv,
        imu_val=imu,
        spo2_val=spo2,
        temp_val=temp,
        PAM_base=pam_base,
        delta_PAM=15,
    )

    return {
        "metodo": metodo,
        "d_inicial": d_inicial,
        "d_sem_pi": d_inicial,
        "alerta_clinico": alerta,
        "d_final": d_final,
        "d_pipeline": fuzzy["demanda"],
        "pam_alvo": fuzzy["PAM_alvo"],
        "classe": fuzzy["classe"],
        "fatores": "; ".join(fatores),
    }


def carregar_segmentos():
    registro = MIMICRecord.objects.first()
    if not registro or not registro.pasta_extraida:
        raise RuntimeError("Nenhum registro MIMIC extraido encontrado.")
    return registro, listar_segmentos_extraidos(registro.pasta_extraida)


def analisar_segmento(segmento, max_amostras=5000):
    dados = ler_waveform_canais(
        segmento["arquivo_dat"],
        max_amostras=max_amostras,
        inicio=0,
    )
    fs = dados["info"]["fs"]
    canais = dados["canais"]
    if "II" not in canais or "ABP" not in canais:
        return None

    ecg = _analisar_ecg(canais, fs)
    abp = _analisar_abp(canais, fs, ecg["metricas_ecg"])
    tau_pam = _estimar_tau_e_pam(canais, ecg["resultado_picos"], fs)
    calibracao = _calibrar_paciente(
        tau_pam["tau_medio"],
        tau_pam["tau_std"],
        tau_pam["pam_batimentos"],
        ecg["metricas_ecg"],
    )

    if not ecg["resultado_picos"] or not tau_pam["pam_batimentos"] or not calibracao:
        return None

    return {
        "dados": dados,
        "fs": fs,
        "canais": canais,
        "ecg": ecg,
        "abp": abp,
        "tau_pam": tau_pam,
        "calibracao": calibracao,
    }


def gerar_precalibracao(segmentos):
    linhas = []
    analises = {}
    for segmento in segmentos:
        try:
            analise = analisar_segmento(segmento)
        except Exception as exc:
            print("ERRO_SEGMENTO", segmento.get("nome"), exc)
            continue
        if not analise:
            continue
        analises[segmento["nome"]] = analise

        tau_pam = analise["tau_pam"]
        ecg = analise["ecg"]
        cal = analise["calibracao"]
        linhas.append({
            "segmento": segmento["nome"],
            "amostras": len(analise["canais"]["ABP"]),
            "fs": analise["fs"],
            "picos_r": ecg["resultado_picos"]["quantidade"],
            "intervalos_rr": max(0, ecg["resultado_picos"]["quantidade"] - 1),
            "fc": fnum(ecg["metricas_ecg"]["fc"]),
            "hrv": fnum(ecg["metricas_ecg"]["rmssd"]),
            "ciclos_abp_rr_analisados": len(segmentar_abp_por_picos_r(
                analise["canais"]["ABP"],
                ecg["resultado_picos"]["picos"],
                fs=analise["fs"],
            )),
            "ciclos_validos_tau": len(tau_pam["taus"]),
            "pas_media": fnum(np.mean(tau_pam["pas_batimentos"])),
            "pad_media": fnum(np.mean(tau_pam["pad_batimentos"])),
            "pam_base": fnum(cal["pam_base"]),
            "tau_base": fnum(cal["tau_base"]),
            "tau_std": fnum(cal["tau_std"]),
            "q_base": fnum(cal["fluxo_base"]),
            "r_eff_base": fnum(cal["r_eff"]),
        })

    write_csv("precalibracao.csv", linhas, [
        "segmento", "amostras", "fs", "picos_r", "intervalos_rr",
        "fc", "hrv", "ciclos_abp_rr_analisados", "ciclos_validos_tau",
        "pas_media", "pad_media", "pam_base", "tau_base", "tau_std",
        "q_base", "r_eff_base",
    ])
    return analises, linhas


def gerar_tau_por_ciclo(nome_segmento, analise):
    linhas = []
    tau_pam = analise["tau_pam"]
    for i, resultado_tau in enumerate(tau_pam["resultados_tau"]):
        linhas.append({
            "segmento": nome_segmento,
            "ciclo": resultado_tau.get("batimento", i + 1),
            "tau": fnum(resultado_tau["tau"]),
            "r2": fnum(resultado_tau["r2"]),
            "pam": fnum(resultado_tau.get("pam")),
            "pas": fnum(resultado_tau.get("pas")),
            "pad": fnum(resultado_tau.get("pad")),
            "pp": fnum(resultado_tau.get("pp")),
            "idx_pico_abs": resultado_tau.get("idx_pico_abs", ""),
            "inicio_diastole": resultado_tau.get("inicio_diastole", ""),
            "inicio_ajuste": resultado_tau.get("inicio_ajuste", ""),
        })
    write_csv("tau_por_ciclo.csv", linhas)
    return linhas


def estatisticas_tau_bruto(analise):
    segmentos_rr = segmentar_abp_por_picos_r(
        analise["canais"]["ABP"],
        analise["ecg"]["resultado_picos"]["picos"],
        fs=analise["fs"],
    )
    candidatos = []
    for seg in segmentos_rr:
        resultado = estimar_tau_batimento(seg["abp"], fs=analise["fs"])
        if resultado and 0.05 < resultado["tau"] < 5 and resultado["r2"] > 0.70:
            candidatos.append(resultado["tau"])

    validos = np.array(analise["tau_pam"]["taus"], dtype=float)
    return {
        "quantidade_antes_filtro_robusto": len(candidatos),
        "quantidade_depois_filtro_robusto": len(validos),
        "media": fnum(np.mean(validos)) if len(validos) else "",
        "mediana": fnum(np.median(validos)) if len(validos) else "",
        "desvio_padrao": fnum(np.std(validos)) if len(validos) else "",
        "minimo": fnum(np.min(validos)) if len(validos) else "",
        "maximo": fnum(np.max(validos)) if len(validos) else "",
    }


def gerar_fuzzy_cenarios(pam_base):
    linhas = []
    for nome, estado in CENARIOS_FISIOLOGICOS.items():
        fc = float(estado["fc"])
        hrv = float(estado.get("hrv", _hrv_por_fc(fc)))
        detalhe = fuzzy_detalhado(
            fc, hrv, estado["imu"], estado["spo2"], estado["temp"], pam_base
        )
        linhas.append({
            "cenario": nome,
            "fc": fc,
            "hrv": hrv,
            "imu": estado["imu"],
            "spo2": estado["spo2"],
            "temperatura": estado["temp"],
            "d_inicial": fnum(detalhe["d_inicial"]),
            "d_sem_pi": fnum(detalhe["d_sem_pi"]),
            "alerta_clinico": detalhe["alerta_clinico"],
            "d_final": fnum(detalhe["d_final"]),
            "d_pipeline": fnum(detalhe["d_pipeline"]),
            "classe": detalhe["classe"],
            "pam_base": fnum(pam_base),
            "pam_alvo": fnum(detalhe["pam_alvo"]),
            "metodo": detalhe["metodo"],
            "fatores": detalhe["fatores"],
        })
    write_csv("fuzzy_cenarios.csv", linhas)
    return linhas


def gerar_ponto_operacao(cal, pam_alvo, rpm=5000, dt=1.0):
    ponto = calcular_ponto_operacao_hm3(
        rpm=rpm,
        r_eff=cal["r_eff"],
        modo_curva="validado",
    )
    p_eq = calcular_p_eq_windkessel(cal["r_eff"], ponto["fluxo"])
    dyn = atualizar_pam_windkessel(
        pam_atual=cal["pam_base"],
        p_eq=p_eq,
        tau_base=cal["tau_base"],
        dt=dt,
    )
    linha = {
        "condicao": "principal_baseline",
        "rpm": rpm,
        "r_eff": fnum(cal["r_eff"]),
        "tau": fnum(cal["tau_base"]),
        "q_dav": fnum(ponto["fluxo"]),
        "delta_p": fnum(ponto["delta_p"]),
        "p_eq": fnum(p_eq),
        "pam_antes": fnum(cal["pam_base"]),
        "pam_apos": fnum(dyn["pam"]),
        "pam_alvo": fnum(pam_alvo),
        "q_base": fnum(cal["fluxo_base"]),
        "faixa_hq": ponto.get("faixa_hq", ""),
    }
    write_csv("ponto_operacao.csv", [linha])
    return linha


def gerar_controle_nominal(cal, fuzzy_linha):
    pam_alvo = float(fuzzy_linha["pam_alvo"])
    historico = executar_malha_fechada_controle(
        pam_base=cal["pam_base"],
        pam_alvo_serie=[pam_alvo] * 120,
        r_eff=cal["r_eff"],
        tau=cal["tau_base"],
        fluxo_base=cal["fluxo_base"],
        rpm_inicial=5000.0,
        kp=20.0,
        ciclos=120,
        dt=1.0,
        rpm_min=3000.0,
        rpm_max=7000.0,
        modo_curva="validado",
        demanda_serie=[float(fuzzy_linha["d_pipeline"])] * 120,
        cenario="nominal_principal",
    )
    linhas = []
    n = len(historico["ciclo"])
    for i in range(n):
        linhas.append({
            "k": i,
            "tempo": historico["tempo"][i],
            "d": historico["demanda"][i],
            "pam_alvo": historico["pam_alvo"][i],
            "pam_estimada": historico["pam_estimada"][i],
            "erro": historico["erro"][i],
            "rpm": historico["rpm_aplicada"][i],
            "q_dav": historico["q_dav"][i],
            "p_eq": historico["p_eq"][i],
            "r_eff": historico["r_eff"][i],
            "tau": historico["tau"][i],
            "delta_p": historico["delta_p"][i],
            "saturacao": historico["saturacao_rpm"][i],
        })
    write_csv("controle_nominal.csv", linhas)

    metricas = calcular_metricas_controle(historico)
    linha_metricas = {
        "pam_base": fnum(cal["pam_base"]),
        "pam_inicial": fnum(cal["pam_base"]),
        "pam_alvo": fnum(pam_alvo),
        "tau_base": fnum(cal["tau_base"]),
        "q_base": fnum(cal["fluxo_base"]),
        "r_eff_base": fnum(cal["r_eff"]),
        "rpm_inicial": 5000.0,
        "kp": 20.0,
        "dt": 1.0,
        "rpm_min": 3000.0,
        "rpm_max": 7000.0,
        "erro_estacionario": fnum(metricas["erro_regime"]),
        "erro_final": fnum(metricas["erro_regime_final"]),
        "mae": fnum(metricas["mae"]),
        "rmse": fnum(metricas["rmse"]),
        "tempo_subida": fnum(metricas["rise_time"]),
        "tempo_acomodacao_2pct": fnum(metricas["settling_time_2pct"]),
        "overshoot": fnum(metricas["overshoot_pct"]),
        "iae": fnum(metricas["iae"]),
        "ise": fnum(metricas["ise"]),
        "itae": fnum(metricas["itae"]),
        "esforco_controle": fnum(metricas["esforco_controle"]),
        "rpm_min_observada": fnum(np.min(historico["rpm_aplicada"])),
        "rpm_max_observada": fnum(np.max(historico["rpm_aplicada"])),
        "q_min_observada": fnum(np.min(historico["q_dav"])),
        "q_max_observada": fnum(np.max(historico["q_dav"])),
        "saturacao_percentual": fnum(metricas["percentual_saturacao"]),
        "pam_final": fnum(historico["pam_estimada"][-1]),
        "rpm_final": fnum(historico["rpm_aplicada"][-1]),
        "q_final": fnum(historico["q_dav"][-1]),
        "p_eq_final": fnum(historico["p_eq"][-1]),
    }
    write_csv("metricas_controle.csv", [linha_metricas])
    return historico, linha_metricas


def gerar_identificacao(cal):
    historico = executar_malha_fechada_controle(
        pam_base=cal["pam_base"],
        pam_alvo_serie=[cal["pam_base"]] * 10
        + [cal["pam_base"] + 5] * 30
        + [cal["pam_base"] + 10] * 40,
        r_eff=cal["r_eff"],
        tau=cal["tau_base"],
        fluxo_base=cal["fluxo_base"],
        rpm_inicial=5000.0,
        kp=20.0,
        ciclos=80,
        dt=1.0,
        modo_curva="validado",
    )
    modelo = identificar_modelo_local(historico)
    estabilidade = analisar_estabilidade_local(
        modelo.get("a"),
        modelo.get("b"),
        20.0,
        identificacao_conclusiva=modelo.get("identificacao_conclusiva", False),
    )
    mapa = mapa_estabilidade_kp(modelo.get("a"), modelo.get("b"), kp_min=1, kp_max=50)
    linhas = [{
        "tipo": "nominal_kp20",
        "kp": 20.0,
        "a": fnum(modelo.get("a"), 10),
        "b": modelo.get("b_formatado"),
        "r2": fnum(modelo.get("r2")),
        "rmse": fnum(modelo.get("rmse")),
        "autovalores": json.dumps(estabilidade.get("autovalores", [])),
        "raio_espectral": fnum(estabilidade.get("raio_espectral")),
        "classificacao": estabilidade.get("classificacao"),
        "mensagem": modelo.get("mensagem") or estabilidade.get("mensagem") or "",
    }]
    for item in mapa.get("resultados", []):
        linhas.append({
            "tipo": "mapa_kp",
            "kp": item["kp"],
            "a": fnum(modelo.get("a"), 10),
            "b": modelo.get("b_formatado"),
            "r2": fnum(modelo.get("r2")),
            "rmse": fnum(modelo.get("rmse")),
            "autovalores": json.dumps(item.get("autovalores", [])),
            "raio_espectral": fnum(item.get("raio_espectral")),
            "classificacao": "ESTAVEL" if item.get("estavel") else "INSTAVEL",
            "mensagem": mapa.get("mensagem") or "",
        })
    write_csv("identificacao_local.csv", linhas)
    return linhas, modelo, estabilidade, mapa


def gerar_multicenario(cal):
    cenarios = executar_teste_multicenario(
        pam_base=cal["pam_base"],
        r_eff_base=cal["r_eff"],
        tau_base=cal["tau_base"],
        fluxo_base=cal["fluxo_base"],
        kp=20.0,
        modo_curva="validado",
        ciclos=120,
    )
    linhas = []
    for item in cenarios:
        hist = item["historico"]
        met = item["metricas"]
        est = item["estabilidade"]
        estado = item["estado"]
        linhas.append({
            "cenario": item["nome"],
            "fc": fnum(estado["fc"]),
            "hrv": fnum(estado["hrv"]),
            "imu": fnum(estado["imu"]),
            "spo2": fnum(estado["spo2"]),
            "temperatura": fnum(estado["temp"]),
            "d": fnum(item["fuzzy"]["demanda"]),
            "pam_alvo": fnum(item["pam_alvo"]),
            "r_eff": fnum(item["r_eff"]),
            "tau": fnum(cal["tau_base"]),
            "rpm_inicial": fnum(hist["rpm_aplicada"][0]),
            "rpm_final": fnum(hist["rpm_aplicada"][-1]),
            "q_dav_final": fnum(hist["q_dav"][-1]),
            "p_eq_final": fnum(hist["p_eq"][-1]),
            "pam_estimada_final": fnum(hist["pam_estimada"][-1]),
            "erro_estacionario": fnum(met["erro_regime"]),
            "mae": fnum(met["mae"]),
            "rmse": fnum(met["rmse"]),
            "overshoot": fnum(met["overshoot_pct"]),
            "settling_time": fnum(met["settling_time_2pct"]),
            "raio_espectral": fnum(est.get("raio_espectral")),
            "saturacao": fnum(met["percentual_saturacao"]),
            "classe": item["fuzzy"]["classe"],
            "alerta": item["fuzzy"]["alerta"],
        })
    write_csv("controle_multicenario.csv", linhas)
    return linhas


def plot_figuras(analises, principal, fuzzy_linhas, ponto, controle_hist, mult_linhas, ident_linhas):
    ecg = principal["canais"]["II"]
    picos = principal["ecg"]["resultado_picos"]["picos"]
    fs = principal["fs"]
    t = np.arange(len(ecg)) / fs

    plt.figure(figsize=(12, 4))
    plt.plot(t, ecg, label="ECG II")
    plt.scatter(np.array(picos) / fs, np.array(ecg)[picos], s=14, label="picos R")
    plt.xlabel("Tempo (s)")
    plt.ylabel("ECG")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig01_picos_r.png", dpi=180)
    plt.close()

    abp = principal["canais"]["ABP"]
    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(len(abp)) / fs, abp, label="ABP")
    for p in picos[:15]:
        plt.axvline(p / fs, color="tab:red", alpha=0.18)
    plt.xlabel("Tempo (s)")
    plt.ylabel("ABP (mmHg)")
    plt.title("ABP segmentada pelos picos R")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig02_abp_segmentacao_rr.png", dpi=180)
    plt.close()

    tau0 = principal["tau_pam"]["resultados_tau"][0]
    plt.figure(figsize=(8, 4))
    plt.plot(tau0["t"], tau0["diastole"], label="ABP real - regiao de ajuste")
    plt.plot(tau0["t"], tau0["ajuste"], label="curva exponencial")
    plt.xlabel("Tempo local (s)")
    plt.ylabel("Pressao (mmHg)")
    plt.title(f"Tau={tau0['tau']:.4f}s, R2={tau0['r2']:.4f}, ciclo={tau0.get('batimento')}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig03_ajuste_tau.png", dpi=180)
    plt.close()

    taus = principal["tau_pam"]["taus"]
    plt.figure(figsize=(8, 4))
    plt.hist(taus, bins=min(12, max(3, len(taus) // 2)), edgecolor="black")
    plt.xlabel("Tau (s)")
    plt.ylabel("Frequencia")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig04_distribuicao_tau.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.bar([x["cenario"] for x in fuzzy_linhas], [float(x["d_pipeline"]) for x in fuzzy_linhas])
    plt.ylabel("D")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig05_fuzzy_cenarios.png", dpi=180)
    plt.close()

    qs = np.linspace(0, 8, 80)
    rpm = float(ponto["rpm"])
    dps = []
    for q in qs:
        dps.append(float(ponto["r_eff"]) * q)
    plt.figure(figsize=(8, 4))
    plt.plot(qs, dps, label="sistema cardiovascular")
    plt.scatter([float(ponto["q_dav"])], [float(ponto["delta_p"])], color="red", label="ponto de operacao")
    plt.xlabel("Fluxo (L/min)")
    plt.ylabel("DeltaP (mmHg)")
    plt.title(f"Intersecao H-Q/sistema em {rpm:.0f} rpm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig06_ponto_operacao.png", dpi=180)
    plt.close()

    x = controle_hist["tempo"]
    plt.figure(figsize=(10, 4))
    plt.plot(x, controle_hist["pam_alvo"], label="PAM alvo")
    plt.plot(x, controle_hist["pam_estimada"], label="PAM estimada")
    plt.xlabel("Tempo (s)")
    plt.ylabel("PAM (mmHg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig07_pam_alvo_estimada.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(x, controle_hist["erro"], label="erro")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Erro (mmHg)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig08_erro_controle.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(x, controle_hist["rpm_aplicada"])
    plt.xlabel("Tempo (s)")
    plt.ylabel("RPM")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig09_rpm.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(x, controle_hist["q_dav"])
    plt.xlabel("Tempo (s)")
    plt.ylabel("Q_DAV (L/min)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig10_vazao_dav.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar([m["cenario"] for m in mult_linhas], [float(m["pam_estimada_final"]) for m in mult_linhas])
    plt.ylabel("PAM final (mmHg)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig11_comparacao_cenarios.png", dpi=180)
    plt.close()

    mapa = [x for x in ident_linhas if x["tipo"] == "mapa_kp" and x["raio_espectral"] != ""]
    if mapa:
        plt.figure(figsize=(8, 4))
        plt.plot([x["kp"] for x in mapa], [float(x["raio_espectral"]) for x in mapa])
        plt.axhline(1.0, color="red", linestyle="--")
        plt.xlabel("Kp")
        plt.ylabel("Raio espectral")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig12_estabilidade_kp.png", dpi=180)
        plt.close()


def gerar_figuras_precalibracao_dissertacao(nome_segmento, principal, tau_linhas, tau_stats):
    fs = float(principal["fs"])
    abp = np.asarray(principal["canais"]["ABP"], dtype=float)
    picos_r = [int(p) for p in principal["ecg"]["resultado_picos"]["picos"]]
    segmentos_rr = segmentar_abp_por_picos_r(abp, picos_r, fs=fs)
    aceitos = principal["tau_pam"]["resultados_tau"]

    if not segmentos_rr or not aceitos:
        raise RuntimeError("Nao ha ciclos RR/tau aceitos para gerar as figuras de pre-calibracao.")

    ciclos_aceitos = {int(item.get("batimento")) for item in aceitos}
    tau_base = float(principal["calibracao"]["tau_base"])
    escolhido = min(aceitos, key=lambda item: abs(float(item["tau"]) - tau_base))
    ciclo_escolhido = int(escolhido.get("batimento"))
    segmento_escolhido = next(
        seg for seg in segmentos_rr if int(seg["indice"]) == ciclo_escolhido
    )

    inicio_janela = max(0, int(segmento_escolhido["inicio"]) - int(2.5 * fs))
    fim_janela = min(len(abp), int(segmento_escolhido["fim"]) + int(2.5 * fs))
    r_set = {p for p in picos_r if inicio_janela <= p < fim_janela}

    dados_abp = []
    for idx_abs in range(inicio_janela, fim_janela):
        ciclo = next(
            (
                seg for seg in segmentos_rr
                if int(seg["inicio"]) <= idx_abs < int(seg["fim"])
            ),
            None,
        )
        ciclo_idx = int(ciclo["indice"]) if ciclo else ""
        em_decaimento = False
        if ciclo_idx == ciclo_escolhido:
            ajuste_ini_abs = int(segmento_escolhido["inicio"]) + int(escolhido["inicio_ajuste"])
            ajuste_fim_abs = ajuste_ini_abs + len(escolhido["diastole"])
            em_decaimento = ajuste_ini_abs <= idx_abs < ajuste_fim_abs
        dados_abp.append({
            "segmento": nome_segmento,
            "amostra": idx_abs,
            "tempo_s": fnum(idx_abs / fs, 6),
            "abp_mmhg": fnum(abp[idx_abs], 6),
            "pico_r_ecg": idx_abs in r_set,
            "ciclo_rr": ciclo_idx,
            "ciclo_aceito_tau": ciclo_idx in ciclos_aceitos if ciclo_idx != "" else False,
            "pico_sistolico_abp": bool(ciclo and idx_abs == int(ciclo["idx_pico_abs"])),
            "regiao_decaimento_ajuste": em_decaimento,
            "ciclo_exemplo_ajuste": ciclo_idx == ciclo_escolhido,
        })
    write_csv_path(FIG_DIR / "FIG_ABP_PRECALIBRACAO_dados.csv", dados_abp, [
        "segmento", "amostra", "tempo_s", "abp_mmhg", "pico_r_ecg",
        "ciclo_rr", "ciclo_aceito_tau", "pico_sistolico_abp",
        "regiao_decaimento_ajuste", "ciclo_exemplo_ajuste",
    ])

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(figsize=(11, 4.6))
    t = np.arange(inicio_janela, fim_janela) / fs
    ax.plot(t, abp[inicio_janela:fim_janela], color="#1f77b4", linewidth=1.5, label="ABP real")

    legenda_ciclo = False
    legenda_pas = False
    for seg in segmentos_rr:
        inicio = int(seg["inicio"])
        fim = int(seg["fim"])
        if fim < inicio_janela or inicio > fim_janela:
            continue
        if int(seg["indice"]) in ciclos_aceitos:
            ax.axvspan(
                max(inicio, inicio_janela) / fs,
                min(fim, fim_janela) / fs,
                color="#2ca02c",
                alpha=0.08,
                label="ciclo RR aceito" if not legenda_ciclo else None,
            )
            legenda_ciclo = True
        ax.axvline(inicio / fs, color="#d62728", alpha=0.28, linewidth=0.9)
        if inicio_janela <= int(seg["idx_pico_abs"]) < fim_janela:
            ax.scatter(
                [int(seg["idx_pico_abs"]) / fs],
                [abp[int(seg["idx_pico_abs"])]],
                color="#ff7f0e",
                s=28,
                zorder=4,
                label="PAS no ciclo RR" if not legenda_pas else None,
            )
            legenda_pas = True

    r_visiveis = [p for p in picos_r if inicio_janela <= p < fim_janela]
    ax.scatter(
        np.asarray(r_visiveis) / fs,
        abp[r_visiveis],
        marker="|",
        s=180,
        color="#d62728",
        linewidths=1.8,
        label="picos R do ECG",
        zorder=5,
    )

    ajuste_ini_abs = int(segmento_escolhido["inicio"]) + int(escolhido["inicio_ajuste"])
    ajuste_fim_abs = ajuste_ini_abs + len(escolhido["diastole"])
    ax.axvspan(
        ajuste_ini_abs / fs,
        ajuste_fim_abs / fs,
        color="#9467bd",
        alpha=0.16,
        label="regiao de decaimento",
    )
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("ABP (mmHg)")
    ax.legend(loc="best", frameon=False, ncol=2)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    tmp_abp = FIG_DIR / "FIG_ABP_PRECALIBRACAO.tmp.png"
    fig.savefig(tmp_abp, dpi=300)
    plt.close(fig)
    os.replace(tmp_abp, FIG_DIR / "FIG_ABP_PRECALIBRACAO.png")

    dados_tau = []
    for i, valor in enumerate(escolhido["diastole"]):
        dados_tau.append({
            "painel": "a",
            "segmento": nome_segmento,
            "ciclo": ciclo_escolhido,
            "tempo_local_s": fnum(escolhido["t"][i], 6),
            "abp_real_mmhg": fnum(valor, 6),
            "abp_ajustada_mmhg": fnum(escolhido["ajuste"][i], 6),
            "tau": fnum(escolhido["tau"], 9),
            "r2": fnum(escolhido["r2"], 9),
            "tau_base": fnum(tau_base, 9),
        })
    for linha in tau_linhas:
        dados_tau.append({
            "painel": "b",
            "segmento": nome_segmento,
            "ciclo": linha["ciclo"],
            "tempo_local_s": "",
            "abp_real_mmhg": "",
            "abp_ajustada_mmhg": "",
            "tau": linha["tau"],
            "r2": linha["r2"],
            "tau_base": fnum(tau_base, 9),
        })
    write_csv_path(FIG_DIR / "FIG_TAU_VALIDACAO_dados.csv", dados_tau, [
        "painel", "segmento", "ciclo", "tempo_local_s", "abp_real_mmhg",
        "abp_ajustada_mmhg", "tau", "r2", "tau_base",
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].scatter(escolhido["t"], escolhido["diastole"], color="#1f77b4", s=24, label="ABP real")
    axes[0].plot(escolhido["t"], escolhido["ajuste"], color="#d62728", linewidth=2, label="ajuste exponencial")
    axes[0].set_xlabel("Tempo local (s)")
    axes[0].set_ylabel("Pressao (mmHg)")
    axes[0].set_title("(a)")
    axes[0].text(
        0.03,
        0.05,
        f"ciclo {ciclo_escolhido}\ntau={float(escolhido['tau']):.3f} s\nR2={float(escolhido['r2']):.3f}",
        transform=axes[0].transAxes,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.22)

    ciclos = [int(linha["ciclo"]) for linha in tau_linhas]
    taus = [float(linha["tau"]) for linha in tau_linhas]
    r2s = [float(linha["r2"]) for linha in tau_linhas]
    axes[1].plot(ciclos, taus, marker="o", color="#1f77b4", label="tau por ciclo")
    axes[1].axhline(tau_base, color="#d62728", linestyle="--", linewidth=1.5, label="tau_base")
    axes[1].set_xlabel("Ciclo RR aceito")
    axes[1].set_ylabel("Tau (s)")
    axes[1].set_title("(b)")
    axes[1].grid(True, alpha=0.22)
    ax_r2 = axes[1].twinx()
    ax_r2.plot(ciclos, r2s, marker="s", color="#2ca02c", alpha=0.75, label="R2")
    ax_r2.set_ylabel("R2")
    linhas1, labels1 = axes[1].get_legend_handles_labels()
    linhas2, labels2 = ax_r2.get_legend_handles_labels()
    axes[1].legend(linhas1 + linhas2, labels1 + labels2, frameon=False, loc="best")
    fig.tight_layout()
    tmp_tau = FIG_DIR / "FIG_TAU_VALIDACAO.tmp.png"
    fig.savefig(tmp_tau, dpi=300)
    plt.close(fig)
    os.replace(tmp_tau, FIG_DIR / "FIG_TAU_VALIDACAO.png")

    taus_np = np.asarray(taus, dtype=float)
    r2_np = np.asarray(r2s, dtype=float)
    resumo = {
        "segmento": nome_segmento,
        "funcao_segmentacao": "segmentar_abp_por_picos_r",
        "funcao_tau": "estimar_tau_por_ciclos_rr via _estimar_tau_e_pam",
        "ciclos_analisados": len(segmentos_rr),
        "ciclos_inicialmente_aceitos": int(tau_stats["quantidade_antes_filtro_robusto"]),
        "ciclos_apos_filtro_final": len(tau_linhas),
        "tau_min": fnum(np.min(taus_np), 9),
        "tau_max": fnum(np.max(taus_np), 9),
        "tau_medio": fnum(np.mean(taus_np), 9),
        "tau_mediano": fnum(np.median(taus_np), 9),
        "tau_std": fnum(np.std(taus_np), 9),
        "r2_medio": fnum(np.mean(r2_np), 9),
        "r2_mediano": fnum(np.median(r2_np), 9),
        "tau_base": fnum(tau_base, 9),
        "ciclo_exemplo": ciclo_escolhido,
    }
    write_csv_path(FIG_DIR / "FIG_TAU_VALIDACAO_resumo.csv", [resumo])
    return resumo


def tabela_md(linhas, campos, limite=12):
    if not linhas:
        return "_sem dados_\n"
    linhas = linhas[:limite]
    out = ["|" + "|".join(campos) + "|", "|" + "|".join(["---"] * len(campos)) + "|"]
    for linha in linhas:
        out.append("|" + "|".join(str(linha.get(c, "")) for c in campos) + "|")
    return "\n".join(out) + "\n"


def gerar_relatorio(precal, tau_linhas, tau_stats, fuzzy_linhas, ponto, met_ctrl, ident_linhas, mult_linhas):
    principal_pre = next(x for x in precal if x["segmento"] == SEGMENTO_PRINCIPAL)
    hipoxemia = [x for x in fuzzy_linhas if "hipoxemia" in x["cenario"]]
    referencias_antigas = []
    for termo in ("R_eff = 24", "122.5", "PAM_base + 15", "PAM hibrida", "ML sem ABP"):
        referencias_antigas.append(f"- Verificar manualmente textos antigos contendo `{termo}`; os CSVs finais nao usam esses valores como resultado do segmento principal.")

    rel = []
    rel.append("# RELATORIO_RESULTADOS_FINAIS\n")
    rel.append("## A. Detecção de picos R\n")
    rel.append(tabela_md(precal, ["segmento", "amostras", "fs", "picos_r", "intervalos_rr", "fc", "hrv"], limite=20))
    rel.append("## B. Pré-calibração\n")
    rel.append(tabela_md(precal, ["segmento", "ciclos_validos_tau", "pas_media", "pad_media", "pam_base", "tau_base", "tau_std", "q_base", "r_eff_base"], limite=20))
    rel.append(f"\nSegmento principal `{SEGMENTO_PRINCIPAL}`: PAM_base={principal_pre['pam_base']} mmHg, tau_base={principal_pre['tau_base']} s, Q_base={principal_pre['q_base']} L/min, R_eff_base={principal_pre['r_eff_base']}.\n")
    rel.append("## C. Estimação de tau\n")
    rel.append(tabela_md(tau_linhas, ["ciclo", "tau", "r2", "pam", "pas", "pad", "inicio_ajuste"], limite=20))
    rel.append("\nEstatísticas: " + json.dumps(tau_stats, ensure_ascii=False) + "\n")
    rel.append("## D. Camada fuzzy\n")
    rel.append(tabela_md(fuzzy_linhas, ["cenario", "fc", "hrv", "imu", "spo2", "temperatura", "d_inicial", "d_sem_pi", "alerta_clinico", "d_final", "pam_alvo"], limite=20))
    if hipoxemia:
        rel.append("\nHipoxemia: " + "; ".join([f"{h['cenario']} usa FC={h['fc']}, HRV={h['hrv']}, SpO2={h['spo2']} e gera D={h['d_pipeline']}" for h in hipoxemia]) + ".\n")
    rel.append("## E. Ponto de operação do DAV\n")
    rel.append(tabela_md([ponto], ["rpm", "r_eff", "tau", "q_dav", "delta_p", "p_eq", "pam_antes", "pam_apos", "q_base"], limite=5))
    rel.append("## F. Controle nominal\n")
    rel.append(tabela_md([met_ctrl], ["pam_base", "pam_alvo", "tau_base", "q_base", "r_eff_base", "rpm_inicial", "kp", "erro_estacionario", "mae", "rmse", "pam_final", "rpm_final", "q_final"], limite=5))
    rel.append("## G. Identificação local e estabilidade\n")
    rel.append(tabela_md(ident_linhas, ["tipo", "kp", "a", "b", "r2", "rmse", "raio_espectral", "classificacao"], limite=20))
    rel.append("## H. Validação multicenário\n")
    rel.append(tabela_md(mult_linhas, ["cenario", "fc", "hrv", "imu", "spo2", "d", "pam_alvo", "r_eff", "tau", "rpm_final", "q_dav_final", "p_eq_final", "pam_estimada_final", "erro_estacionario", "rmse", "raio_espectral", "saturacao"], limite=20))
    rel.append("## I. Valores antigos que não devem mais ser utilizados\n")
    rel.extend(referencias_antigas)
    rel.append("\n")
    rel.append("Os resultados finais foram recalculados a partir do código atual; valores antigos de dissertação não foram reutilizados.\n")
    rel.append("## J. Observações metodológicas\n")
    rel.append("- Não existe regressão FC/HRV -> PAM no pipeline ativo.\n")
    rel.append("- Não existe rede neural ECG -> PAM no pipeline ativo.\n")
    rel.append("- Não existe fusão PAM Windkessel + PAM ML no pipeline ativo.\n")
    rel.append("- A complacência C não é estimada separadamente; tau é a constante vascular agregada.\n")
    rel.append("- Q_nativo não é modelado; a simulação usa Q_DAV como aproximação operacional.\n")
    rel.append("- R_eff_dinamica é modulação heurística de simulação, não estimativa validada de RVS.\n")
    rel.append("- A variável que realimenta o controlador é `pam_estimada` produzida pela dinâmica cardiovascular/Windkessel.\n")
    rel.append("\n## Respostas diretas\n")
    respostas = {
        "PAM_base_3000003_0008": principal_pre["pam_base"],
        "tau_base_3000003_0008": principal_pre["tau_base"],
        "Q_base": principal_pre["q_base"],
        "R_eff_base": principal_pre["r_eff_base"],
        "PAM_realimentacao_controlador": "historico['pam_estimada'] / resultado_dav['pam_estimada']",
        "PAM_alvo_nominal": met_ctrl["pam_alvo"],
        "PAM_final": met_ctrl["pam_final"],
        "RPM_final": met_ctrl["rpm_final"],
        "Q_DAV_final": met_ctrl["q_final"],
        "erro_estacionario": met_ctrl["erro_estacionario"],
        "RMSE_controle": met_ctrl["rmse"],
        "R_eff_24_no_resultado_final": "nao; pode existir como valor default em formularios/rotinas antigas, mas o resultado principal usa R_eff_base recalculado",
        "hipoxemia": hipoxemia[0] if hipoxemia else {},
        "referencia_PAM_hibrida_ML": "nao encontrada no pipeline ativo",
    }
    rel.append("```json\n" + json.dumps(respostas, ensure_ascii=False, indent=2) + "\n```\n")
    (OUT_DIR / "RELATORIO_RESULTADOS_FINAIS.md").write_text("".join(rel), encoding="utf-8")


def main():
    global OUT_DIR, FIG_DIR, SEGMENTO_PRINCIPAL

    parser = argparse.ArgumentParser(
        description="Gera resultados finais da arquitetura consolidada."
    )
    parser.add_argument(
        "--segmento",
        default=SEGMENTO_PRINCIPAL,
        help="Segmento principal usado para resultados detalhados.",
    )
    parser.add_argument(
        "--saida",
        default=None,
        help="Nome da pasta de saida dentro do projeto generico.",
    )
    args = parser.parse_args()

    SEGMENTO_PRINCIPAL = args.segmento
    if args.saida:
        OUT_DIR = BASE_DIR / args.saida
    elif args.segmento != "3000003_0008":
        OUT_DIR = BASE_DIR / f"resultados_dissertacao_final_{args.segmento}"
    FIG_DIR = OUT_DIR / "figuras"

    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    registro, segmentos = carregar_segmentos()
    analises, precal = gerar_precalibracao(segmentos)
    if SEGMENTO_PRINCIPAL not in analises:
        raise RuntimeError(f"Segmento principal {SEGMENTO_PRINCIPAL} nao encontrado/analisado.")

    principal = analises[SEGMENTO_PRINCIPAL]
    tau_linhas = gerar_tau_por_ciclo(SEGMENTO_PRINCIPAL, principal)
    tau_stats = estatisticas_tau_bruto(principal)
    cal = principal["calibracao"]
    fuzzy_linhas = gerar_fuzzy_cenarios(cal["pam_base"])
    nominal_fuzzy = fuzzy_detalhado(
        principal["ecg"]["metricas_ecg"]["fc"],
        principal["ecg"]["metricas_ecg"]["rmssd"],
        0.10,
        98,
        36.5,
        cal["pam_base"],
    )
    fuzzy_nominal_linha = {
        "pam_alvo": nominal_fuzzy["pam_alvo"],
        "d_pipeline": nominal_fuzzy["d_pipeline"],
    }
    ponto = gerar_ponto_operacao(cal, nominal_fuzzy["pam_alvo"])
    controle_hist, met_ctrl = gerar_controle_nominal(cal, fuzzy_nominal_linha)
    ident_linhas, modelo, estabilidade, mapa = gerar_identificacao(cal)
    mult_linhas = gerar_multicenario(cal)
    plot_figuras(analises, principal, fuzzy_linhas, ponto, controle_hist, mult_linhas, ident_linhas)
    resumo_figuras = gerar_figuras_precalibracao_dissertacao(
        SEGMENTO_PRINCIPAL,
        principal,
        tau_linhas,
        tau_stats,
    )
    gerar_relatorio(precal, tau_linhas, tau_stats, fuzzy_linhas, ponto, met_ctrl, ident_linhas, mult_linhas)

    print("RESULTADOS_GERADOS", OUT_DIR)
    print("PAM_base", met_ctrl["pam_base"])
    print("tau_base", met_ctrl["tau_base"])
    print("Q_base", met_ctrl["q_base"])
    print("R_eff_base", met_ctrl["r_eff_base"])
    print("PAM_alvo_nominal", met_ctrl["pam_alvo"])
    print("PAM_final", met_ctrl["pam_final"])
    print("RPM_final", met_ctrl["rpm_final"])
    print("Q_final", met_ctrl["q_final"])
    print("erro_estacionario", met_ctrl["erro_estacionario"])
    print("RMSE_controle", met_ctrl["rmse"])
    print("FIG_PRECALIBRACAO", FIG_DIR / "FIG_ABP_PRECALIBRACAO.png")
    print("FIG_TAU_VALIDACAO", FIG_DIR / "FIG_TAU_VALIDACAO.png")
    print("RESUMO_TAU_FIGURAS", json.dumps(resumo_figuras, ensure_ascii=False))


if __name__ == "__main__":
    main()
