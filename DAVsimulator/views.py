from django.shortcuts import render, redirect
from django.conf import settings
from .services.calibracao import (calibrar_paciente_pre_implante, calcular_r_eff_dinamica,)
from .services.pipeline_dav import (executar_pipeline_dav, simular_controle_dav,)
from .services.fuzzy_dav import calcular_demanda_fuzzy
from .services.curva_hq_heartmate import CURVAS_HQ
from .services.curva_hq_heartmate import calcular_ponto_operacao_hm3
from .services.validacao import avaliar_picos_r

import json
import numpy as np

from .models import MIMICRecord

from .forms import (
    SimulacaoDAVForm,
    SintoniaHemodinamicaForm,
    MIMICRecordUploadForm,
)

from .services.wfdb_parser import (
    listar_arquivos_zip,
    encontrar_headers,
    ler_header_principal,
    detectar_sinais,
    extrair_info_headers,
    extrair_zip_dataset,
    listar_segmentos_extraidos,
    ler_waveform_canais,
    ler_info_segmento,
    detectar_picos_r_simples,
)

from .services.hemodinamica import (
    calcular_metricas_abp,
    detectar_pulsos_abp,
    estimar_tau_por_ciclos_rr,
    reconstruir_pam_por_windkessel_exponencial,
)

from .services.ia_ecg import (
    detectar_indices_picos_r_ia,
    calcular_fc_hrv,
)
from .services.pi_dav import calcular_pi_dav_janela
from .services.validacao_controle_dav import executar_validacao_dinamica_controle


def home(request):

    resultado = None

    if request.method == "POST":

        form = SimulacaoDAVForm(request.POST)

        if form.is_valid():

            dados = form.cleaned_data

            fc = dados["fc"]
            hrv = dados["hrv"]
            spo2 = dados["spo2"]
            temperatura = dados["temperatura"]
            atividade = dados["atividade"]

            pam_basal = dados["pam_basal"]
            resistencia = dados["resistencia"]
            tau = dados["tau_base"]
            vazao_basal = dados["vazao_basal"]

            rpm_inicial = dados["rpm_inicial"]
            kp = dados["kp"]
            duracao = dados["duracao"]

            fuzzy = calcular_demanda_fuzzy(
                fc_val=fc,
                hrv_val=hrv,
                imu_val=atividade,
                spo2_val=spo2,
                temp_val=temperatura,
                PAM_base=pam_basal,
                delta_PAM=15,
            )

            pam_alvo = fuzzy["PAM_alvo"]

            resultado_dav = executar_pipeline_dav(
                rpm_atual=rpm_inicial,
                pam_alvo=pam_alvo,
                r_eff=resistencia,
                tau_base=tau,
                tau_atual=tau,
                fluxo_base=vazao_basal,
                pam_base=pam_basal,
                kp=kp,
                pam_max=140,
            )

            historico = simular_controle_dav(
                rpm_inicial=rpm_inicial,
                pam_alvo=pam_alvo,
                r_eff=resistencia,
                tau_base=tau,
                tau_atual=tau,
                fluxo_base=vazao_basal,
                pam_base=pam_basal,
                kp=kp,
                ciclos=duracao,
                pam_max=140,
            )

            resultado = {
                "demanda": round(float(fuzzy["demanda"]), 3),
                "pam_alvo": round(pam_alvo, 2),
                "erro": round(float(resultado_dav["erro"]), 2),
                "rpm_final": round(
                    historico["rpm"][-1] if historico["rpm"] else rpm_inicial,
                    0
                ),
                "tau": round(tau, 2),
                "vazao_final": round(
                    historico["fluxo"][-1] if historico["fluxo"] else 0,
                    2
                ),
                "tempo_json": json.dumps(
                    historico.get("tempo") or historico["ciclos"]
                ),
                "pam_json": json.dumps(historico["pam"]),
                "abp_simulada_json": json.dumps([]),
                "pam_alvo_json": json.dumps(
                    historico.get("pam_alvo") or [pam_alvo] * len(historico["pam"])
                ),
                "rpm_json": json.dumps(
                    historico["rpm"]
                ),
                "vazao_json": json.dumps(
                    historico["fluxo"]
                ),
                "p_eq_json": json.dumps(historico.get("p_eq", [])),
                "demonstrativo": True,
            }

    else:

        form = SimulacaoDAVForm()

    return render(
        request,
        "DAVsimulator/home.html",
        {
            "form": form,
            "resultado": resultado,
        },
    )


def sintonia_hemodinamica(request):

    resultado = None

    if request.method == "POST":

        form = SintoniaHemodinamicaForm(
            request.POST
        )

        if form.is_valid():

            texto_abp = form.cleaned_data["abp"]

            q = form.cleaned_data["vazao_q"]

            valores = (
                texto_abp
                .replace(",", " ")
                .replace(";", " ")
                .split()
            )

            abp = []

            for valor in valores:

                try:

                    abp.append(
                        float(valor)
                    )

                except ValueError:

                    pass

            if len(abp) > 5:

                pam_media = np.mean(abp)

                abp_max = np.max(abp)

                abp_min = np.min(abp)

                pulsatilidade = (
                    abp_max - abp_min
                )

                resistencia = (
                    pam_media / q
                )

                resultado = {
                    "pam_media": round(
                        float(pam_media),
                        2
                    ),
                    "abp_max": round(
                        float(abp_max),
                        2
                    ),
                    "abp_min": round(
                        float(abp_min),
                        2
                    ),
                    "pulsatilidade": round(
                        float(pulsatilidade),
                        2
                    ),
                    "tau": None,
                    "resistencia": round(
                        float(resistencia),
                        3
                    ),
                    "n_amostras": len(abp),
                    "aviso": (
                        "Tau nao foi estimado nesta tela: a metodologia oficial "
                        "exige ECG sincronizado para delimitar os ciclos RR e "
                        "ajustar a queda diastolica por ciclo."
                    ),
                }

    else:

        form = SintoniaHemodinamicaForm()

    return render(
        request,
        "DAVsimulator/sintonia.html",
        {
            "form": form,
            "resultado": resultado,
        },
    )


def dataset_manager(request):

    registros = (
        MIMICRecord.objects
        .all()
        .order_by("nome")
    )

    return render(
        request,
        "DAVsimulator/dataset_manager.html",
        {
            "registros": registros
        },
    )


def dataset_upload(request):

    if request.method == "POST":

        form = MIMICRecordUploadForm(
            request.POST,
            request.FILES
        )

        if form.is_valid():

            registro = form.save()

            if registro.arquivo_zip:

                caminho_zip = (
                    registro.arquivo_zip.path
                )

                pasta_extraida = (
                    extrair_zip_dataset(
                        caminho_zip,
                        settings.MEDIA_ROOT
                        / "dav"
                        / "datasets"
                        / "extraidos",
                        registro.record_id
                        or registro.pk
                    )
                )

                registro.pasta_extraida = (
                    pasta_extraida
                )

                arquivos = listar_arquivos_zip(
                    caminho_zip
                )

                headers = encontrar_headers(
                    arquivos
                )

                info_headers = (
                    extrair_info_headers(
                        caminho_zip,
                        headers
                    )
                )

                sinais = {
                    "ecg": False,
                    "abp": False,
                    "ppg": False,
                    "resp": False,
                }

                headers_lidos = []

                for header in headers:

                    texto_header = (
                        ler_header_principal(
                            caminho_zip,
                            header
                        )
                    )

                    sinais_detectados = (
                        detectar_sinais(
                            texto_header
                        )
                    )

                    sinais["ecg"] = (
                        sinais["ecg"]
                        or sinais_detectados["ecg"]
                    )

                    sinais["abp"] = (
                        sinais["abp"]
                        or sinais_detectados["abp"]
                    )

                    sinais["ppg"] = (
                        sinais["ppg"]
                        or sinais_detectados["ppg"]
                    )

                    sinais["resp"] = (
                        sinais["resp"]
                        or sinais_detectados["resp"]
                    )

                    headers_lidos.append(
                        header
                    )

                registro.possui_ecg = sinais["ecg"]
                registro.possui_abp = sinais["abp"]
                registro.possui_ppg = sinais["ppg"]
                registro.possui_resp = sinais["resp"]

                if info_headers["fs"]:

                    registro.frequencia_amostragem = (
                        info_headers["fs"]
                    )

                registro.observacoes = (
                    f"Arquivos encontrados: {len(arquivos)}\n"
                    f"Headers: {headers_lidos}\n"
                    f"Sinais WFDB: {info_headers['sinais']}\n"
                    f"FS: {info_headers['fs']}\n"
                    f"Sinais detectados: {sinais}"
                )

                registro.save()

            return redirect(
                "DAVsimulator:datasets"
            )

    else:

        form = MIMICRecordUploadForm()

    return render(
        request,
        "DAVsimulator/dataset_upload.html",
        {
            "form": form
        },
    )


def dataset_detail(request, pk):

    registro = MIMICRecord.objects.get(
        pk=pk
    )

    return render(
        request,
        "DAVsimulator/dataset_detail.html",
        {
            "registro": registro
        },
    )


def fuzzy(request):
    registro = MIMICRecord.objects.first()

    if not registro:
        return redirect("DAVsimulator:datasets")

    return redirect(
        "DAVsimulator:dataset_sinais",
        pk=registro.pk,
    )


def _parse_int_param(valor, padrao):
    if valor is None:
        return padrao

    try:
        return int(str(valor).replace(".", "").replace(",", ""))
    except (TypeError, ValueError):
        return padrao


def _selecionar_segmento(segmentos, nome_segmento):
    if not segmentos:
        return None

    if nome_segmento:
        for segmento in segmentos:
            if segmento["nome"] == nome_segmento:
                return segmento

    segmentos_ecg_abp = []

    for segmento in segmentos:
        try:
            info = ler_info_segmento(
                segmento["arquivo_dat"].replace(".dat", ".hea")
            )
        except Exception:
            continue

        nomes = [
            canal.get("nome")
            for canal in info.get("canais_info", [])
        ]

        if "ABP" in nomes and (
            "II" in nomes
            or "I" in nomes
            or "V" in nomes
        ):
            segmentos_ecg_abp.append(segmento)

    if segmentos_ecg_abp:
        if len(segmentos_ecg_abp) > 1:
            return segmentos_ecg_abp[1]

        return segmentos_ecg_abp[0]

    for segmento in segmentos:
        try:
            info = ler_info_segmento(
                segmento["arquivo_dat"].replace(".dat", ".hea")
            )
        except Exception:
            continue

        nomes = [
            canal.get("nome")
            for canal in info.get("canais_info", [])
        ]

        if "ABP" in nomes:
            return segmento

    return segmentos[0]


def _analisar_ecg(canais, fs):
    resultado = {
        "resultado_picos": None,
        "picos_json": json.dumps([]),
        "metricas_ecg": None,
        "prob_ia_json": None,
        "janelas_qrs_json": None,
    }

    if "II" not in canais:
        return resultado

    ecg_ii = canais["II"]

    if len(ecg_ii) == 5000:
        picos_ia, prob_ia, janelas_qrs = detectar_indices_picos_r_ia(
            ecg_ii,
            fs=fs,
            threshold=0.5,
        )

        resultado["metricas_ecg"] = calcular_fc_hrv(
            picos_ia,
            fs=fs,
        )

        resultado["prob_ia_json"] = json.dumps(
            prob_ia.tolist()
        )

        resultado["janelas_qrs_json"] = json.dumps(
            janelas_qrs
        )

        resultado["resultado_picos"] = {
            "metodo": "IA CNN",
            "picos": picos_ia.tolist(),
            "quantidade": len(picos_ia),
        }
    else:
        resultado_picos = detectar_picos_r_simples(
            ecg_ii,
            fs=fs,
        )
        resultado_picos["metodo"] = "Detector simples"
        resultado["resultado_picos"] = resultado_picos

    resultado["picos_json"] = json.dumps(
        resultado["resultado_picos"]["picos"]
    )

    return resultado


def _analisar_abp(canais, fs, metricas_ecg):
    resultado = {
        "metricas_abp": None,
        "resultado_abp": None,
        "metricas_fc_abp": None,
        "erro_fc": None,
        "pressao_pulso_normalizada": None,
    }

    if "ABP" not in canais:
        return resultado

    abp = canais["ABP"]

    resultado["metricas_abp"] = calcular_metricas_abp(abp)
    resultado["resultado_abp"] = detectar_pulsos_abp(
        abp,
        fs=fs,
    )

    if (
        resultado["resultado_abp"]
        and resultado["resultado_abp"].get("picos")
    ):
        resultado["metricas_fc_abp"] = calcular_fc_hrv(
            resultado["resultado_abp"]["picos"],
            fs=fs,
        )

    if metricas_ecg and resultado["metricas_fc_abp"]:
        resultado["erro_fc"] = abs(
            metricas_ecg["fc"]
            - resultado["metricas_fc_abp"]["fc"]
        )

    if resultado["metricas_abp"]:
        pam = float(resultado["metricas_abp"]["pam"])
        pp = float(resultado["metricas_abp"]["pas"]) - float(resultado["metricas_abp"]["pad"])
        resultado["pressao_pulso_normalizada"] = pp / pam if pam > 0 else None

    return resultado


def _estimar_tau_e_pam(canais, resultado_picos_ecg, fs):
    resultado = {
        "taus": [],
        "tau_medio": None,
        "tau_std": None,
        "resultados_tau": [],
        "pam_batimentos": [],
        "pam_modelo_bruta": [],
        "rr_batimentos": [],
        "pas_batimentos": [],
        "pad_batimentos": [],
        "pp_batimentos": [],
    }

    if (
        "ABP" not in canais
        or not resultado_picos_ecg
        or not resultado_picos_ecg.get("picos")
    ):
        return resultado

    estimativa = estimar_tau_por_ciclos_rr(
        canais["ABP"],
        resultado_picos_ecg["picos"],
        fs=fs,
    )

    for ciclo in estimativa["ciclos"]:
        resultado["taus"].append(ciclo["tau"])
        resultado["resultados_tau"].append(ciclo["resultado_tau"])
        resultado["pam_batimentos"].append(ciclo["pam"])
        resultado["pam_modelo_bruta"].append(ciclo["pam_modelo_bruta"])
        resultado["rr_batimentos"].append(ciclo["rr_s"])
        resultado["pas_batimentos"].append(ciclo["pas"])
        resultado["pad_batimentos"].append(ciclo["pad"])
        resultado["pp_batimentos"].append(ciclo["pp"])

    if resultado["taus"]:
        resultado["tau_medio"] = round(
            float(estimativa["tau_medio"]),
            3,
        )
        resultado["tau_std"] = round(
            float(estimativa["tau_std"]),
            3,
        )

    return resultado


def _ajustar_modelo_pam(
    taus,
    pam_batimentos,
    metricas_ecg,
    pam_modelo_bruta=None,
    pas_batimentos=None,
    pad_batimentos=None,
    pp_batimentos=None,
    picos_r=None,
    fs=None,
    split=None,
):
    resultado = {
        "pam_modelo_batimentos": [],
        "pam_reconstruida_abp": [],
        "tau_modelo": None,
        "offset_calibracao": None,
        "metodo_sem_abp": None,
    }

    if (
        len(taus) <= 2
        or len(pam_batimentos) <= 2
    ):
        return resultado

    if not pam_modelo_bruta:
        return resultado

    n = min(
        len(taus),
        len(pam_batimentos),
        len(pam_modelo_bruta),
    )

    if n <= 2:
        return resultado

    ajuste = reconstruir_pam_por_windkessel_exponencial(
        pam_batimentos[:n],
        pam_modelo_bruta[:n],
        split=split,
    )

    resultado["pam_modelo_batimentos"] = ajuste["predicoes"]
    resultado["pam_reconstruida_abp"] = [
        float(valor) for valor in pam_modelo_bruta[:n]
    ]
    resultado["tau_modelo"] = float(np.median(taus[:n]))
    resultado["offset_calibracao"] = float(ajuste["offset"])
    resultado["metodo_sem_abp"] = (
        "Windkessel exponencial calibrado por ABP; "
        "ECG usado apenas como referencia temporal RR"
    )

    return resultado


def _metricas_regressao(y_real, y_pred):
    y_real = np.array(y_real, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    if len(y_real) == 0 or len(y_pred) == 0:
        return {
            "mae": 0,
            "rmse": 0,
            "msre": 0,
            "mape": 0,
            "r2": 0,
        }

    erro = y_real - y_pred
    denominador = np.maximum(np.abs(y_real), 1e-8)
    ss_res = float(np.sum(erro ** 2))
    ss_tot = float(np.sum((y_real - np.mean(y_real)) ** 2))

    return {
        "mae": float(np.mean(np.abs(erro))),
        "rmse": float(np.sqrt(np.mean(erro ** 2))),
        "msre": float(np.mean((erro / denominador) ** 2)),
        "mape": float(np.mean(np.abs(erro / denominador)) * 100),
        "r2": float(1 - (ss_res / ss_tot)) if ss_tot else 0,
    }


def _formatar_metricas_observador_fases(resultado):
    if not resultado or not resultado.get("metricas"):
        return None

    metricas = resultado["metricas"]

    def bloco(nome):
        item = metricas.get(nome) or {}
        return {
            "mae": round(float(item["mae"]), 3) if item.get("mae") is not None else None,
            "rmse": round(float(item["rmse"]), 3) if item.get("rmse") is not None else None,
            "r2": round(float(item["r2"]), 4) if item.get("r2") is not None else None,
            "erro_max": round(float(item["erro_max"]), 3) if item.get("erro_max") is not None else None,
        }

    return {
        "pas": bloco("pas"),
        "pad": bloco("pad"),
        "pam_formula": bloco("pam_formula"),
        "pam_integral": bloco("pam_integral"),
        "total_ciclos": resultado["total_ciclos"],
        "validos": resultado["validos"],
        "rejeitados": resultado["rejeitados"],
    }


def _validar_modelo_pam_batimento(
    taus,
    pam_batimentos,
    metricas_ecg,
    pam_modelo_bruta=None,
    pas_batimentos=None,
    pad_batimentos=None,
    pp_batimentos=None,
    picos_r=None,
    fs=None,
):
    if len(taus) < 6 or len(pam_batimentos) < 6 or not pam_modelo_bruta:
        return None

    n = min(
        len(taus),
        len(pam_batimentos),
        len(pam_modelo_bruta),
    )
    split = max(3, int(n * 0.7))

    if n - split < 2:
        split = n - 2

    pam_array = np.array(pam_batimentos[:n], dtype=float)
    ajuste = reconstruir_pam_por_windkessel_exponencial(
        pam_array.tolist(),
        pam_modelo_bruta[:n],
        split=split,
    )

    pam_pred = ajuste["predicoes"][split:]
    pam_pred_basico = np.array(pam_modelo_bruta[:n], dtype=float)[split:]
    pam_real = pam_array[split:]
    metricas = _metricas_regressao(
        pam_real,
        pam_pred,
    )
    metricas_reconstrucao = _metricas_regressao(
        pam_real,
        pam_pred_basico,
    )

    return {
        "coef": [float(np.median(taus[:n])), float(ajuste["offset"])],
        "pam_real": pam_real.tolist(),
        "pam_pred": [float(valor) for valor in pam_pred],
        "pam_pred_basico": pam_pred_basico.tolist(),
        "metricas": metricas,
        "metricas_basico": metricas_reconstrucao,
        "n_treino": int(split),
        "n_teste": int(n - split),
        "modelo": "Windkessel exponencial calibrado",
        "metodo_sem_abp": (
            "ECG delimita RR; ABP calibra tau e offset da curva exponencial"
        ),
    }


def _calibrar_paciente(tau_medio, tau_std, pam_batimentos, metricas_ecg):
    if tau_medio is None or not pam_batimentos:
        return None

    fc_media = None
    hrv_rmssd = None

    if metricas_ecg:
        fc_media = metricas_ecg.get("fc")
        hrv_rmssd = metricas_ecg.get("rmssd")

    calibracao = calibrar_paciente_pre_implante(
        tau_medio=tau_medio,
        tau_std=tau_std,
        pam_media=float(np.mean(pam_batimentos)),
        fc_media=fc_media,
        hrv_rmssd=hrv_rmssd,
        fluxo_estimado=3.0,
    )

    if calibracao and calibracao.get("r_eff") is not None:
        ponto_base = calcular_ponto_operacao_hm3(
            rpm=5000,
            r_eff=calibracao["r_eff"],
        )
        calibracao["q_base_assumido"] = calibracao.get("fluxo_base")
        calibracao["q_dav_base"] = ponto_base["fluxo"]
        calibracao["delta_p_base"] = ponto_base["delta_p"]

    return calibracao


def _estado_fisiologico_simulado(request, calibracao_paciente):
    usar_slider = request.GET.get("usar_slider") == "1"

    fc_base = (
        calibracao_paciente.get("fc_base")
        if calibracao_paciente
        else 75
    )

    hrv_base = (
        calibracao_paciente.get("hrv_base")
        if calibracao_paciente
        else 40
    )

    if usar_slider:
        fc_simulada = float(request.GET.get("fc", fc_base))
        hrv_simulada = max(5, 120 - (fc_simulada * 0.9))
        imu_val = float(request.GET.get("imu", 0.1))
        spo2_val = float(request.GET.get("spo2", 98))
        temp_val = float(request.GET.get("temp", 36.5))
    else:
        fc_simulada = fc_base
        hrv_simulada = hrv_base
        imu_val = 0.1
        spo2_val = 98
        temp_val = 36.5

    return {
        "usar_slider": usar_slider,
        "fc_simulada": fc_simulada,
        "hrv_simulada": hrv_simulada,
        "imu_val": imu_val,
        "spo2_val": spo2_val,
        "temp_val": temp_val,
    }


def _avaliar_validade_calibracao(
    calibracao_paciente,
    estado,
    resultado_dav=None,
    resultado_fuzzy=None,
):
    if not calibracao_paciente:
        return None

    pontos = 0
    fatores = []

    fc_base = calibracao_paciente.get("fc_base")
    hrv_base = calibracao_paciente.get("hrv_base")
    fluxo_base = calibracao_paciente.get("fluxo_base")
    pam_base = calibracao_paciente.get("pam_base")

    fc_atual = float(estado.get("fc_simulada") or 0)
    hrv_atual = float(estado.get("hrv_simulada") or 0)
    spo2 = float(estado.get("spo2_val") or 98)
    temp = float(estado.get("temp_val") or 36.5)

    if fc_base:
        desvio_fc = abs(fc_atual - float(fc_base)) / max(float(fc_base), 1.0)
        if desvio_fc >= 0.35:
            pontos += 2
            fatores.append("FC muito distante da calibracao basal")
        elif desvio_fc >= 0.20:
            pontos += 1
            fatores.append("FC diferente da calibracao basal")

    if hrv_base and hrv_atual:
        desvio_hrv = abs(hrv_atual - float(hrv_base)) / max(float(hrv_base), 1.0)
        if desvio_hrv >= 0.60:
            pontos += 2
            fatores.append("HRV muito diferente da calibracao basal")
        elif desvio_hrv >= 0.35:
            pontos += 1
            fatores.append("HRV diferente da calibracao basal")

    if resultado_dav and fluxo_base:
        fluxo_atual = float(resultado_dav.get("fluxo_estimado") or 0)
        desvio_fluxo = abs(fluxo_atual - float(fluxo_base))
        if desvio_fluxo >= 1.0:
            pontos += 2
            fatores.append("Fluxo estimado muito distante do fluxo basal")
        elif desvio_fluxo >= 0.5:
            pontos += 1
            fatores.append("Fluxo estimado diferente do fluxo basal")

    if resultado_dav:
        rpm_atual = float(resultado_dav.get("rpm_atual") or 5000)
        rpm_sugerido = float(resultado_dav.get("rpm_sugerido") or rpm_atual)
        desvio_rpm = abs(rpm_sugerido - rpm_atual)
        if desvio_rpm >= 700:
            pontos += 2
            fatores.append("Controle sugeriu grande mudanca de RPM")
        elif desvio_rpm >= 300:
            pontos += 1
            fatores.append("Controle sugeriu mudanca relevante de RPM")

    if spo2 < 90:
        pontos += 2
        fatores.append("SpO2 baixa: estado fora da calibracao basal")
    elif spo2 < 94:
        pontos += 1
        fatores.append("SpO2 limiar")

    if temp >= 38.0:
        pontos += 2
        fatores.append("Temperatura alta")
    elif temp >= 37.5:
        pontos += 1
        fatores.append("Temperatura elevada")

    if resultado_fuzzy and resultado_fuzzy.get("alerta"):
        pontos += 1
        fatores.append("Fuzzy classificou alerta fisiologico")

    if pontos >= 5:
        status = "Recalibracao recomendada"
        classe = "danger"
        interpretacao = (
            "O estado atual esta distante da assinatura basal usada para treinar "
            "a estimativa de PAM. A reconstrucao calibrada deve ser interpretada com "
            "maior incerteza."
        )
    elif pontos >= 2:
        status = "Calibracao em observacao"
        classe = "warning"
        interpretacao = (
            "Ha mudancas fisiologicas relevantes em relacao ao estado calibrado. "
            "A estimativa segue utilizavel para simulacao, mas com cautela."
        )
    else:
        status = "Calibracao confiavel"
        classe = "success"
        interpretacao = (
            "O estado atual permanece proximo da assinatura hemodinamica basal "
            "usada na calibracao."
        )

    return {
        "status": status,
        "classe": classe,
        "pontos": pontos,
        "fatores": fatores,
        "interpretacao": interpretacao,
        "pam_base": round(float(pam_base), 3) if pam_base is not None else None,
        "fc_base": round(float(fc_base), 2) if fc_base is not None else None,
        "hrv_base": round(float(hrv_base), 2) if hrv_base is not None else None,
        "fluxo_base": (
            round(float(fluxo_base), 3)
            if fluxo_base is not None
            else None
        ),
    }


def _calcular_pi_dav_por_potencia(historico_dav, janela_s=15.0):
    """
    Calcula o PI estimado do DAV pela variacao da potencia simulada da bomba.

    A janela usa a resolucao temporal do historico disponivel. Este valor e
    apenas uma estimativa do modelo, nao uma leitura direta do HeartMate 3 real.
    """

    if not historico_dav or not historico_dav.get("potencia"):
        return None

    tempos = historico_dav.get("tempo") or []
    dt = 1.0
    if len(tempos) >= 2:
        diffs = np.diff(np.array(tempos, dtype=float))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if len(diffs):
            dt = float(np.median(diffs))

    resultado = calcular_pi_dav_janela(
        historico_dav["potencia"],
        dt=dt,
        janela_s=janela_s,
    )
    n_janela = int(resultado["amostras_janela"])

    def serie_janela(nome):
        valores = historico_dav.get(nome) or []
        return [
            round(float(v), 4)
            for v in valores[-n_janela:]
            if v is not None and np.isfinite(float(v))
        ]

    valor = resultado["pi"]
    interpretacao = (
        "PI estimado a partir da variacao da potencia simulada da bomba. "
        "A faixa 2-6 pode ser usada apenas como referencia contextual da "
        "literatura do HeartMate 3, pois este simulador nao mede rotor drive power."
    )

    return {
        "valor": round(float(valor), 3) if valor is not None else None,
        "classe": "PI estimado do DAV",
        "interpretacao": interpretacao,
        "potencia_max": (
            round(float(resultado["potencia_max"]), 3)
            if resultado["potencia_max"] is not None
            else None
        ),
        "potencia_min": (
            round(float(resultado["potencia_min"]), 3)
            if resultado["potencia_min"] is not None
            else None
        ),
        "potencia_media": (
            round(float(resultado["potencia_media"]), 3)
            if resultado["potencia_media"] is not None
            else None
        ),
        "potencia_std": (
            round(float(resultado["potencia_std"]), 6)
            if resultado["potencia_std"] is not None
            else None
        ),
        "potencias_janela": [
            round(float(v), 4)
            for v in resultado.get("potencias_janela", [])
        ],
        "rpm_janela": serie_janela("rpm"),
        "fluxo_janela": serie_janela("fluxo"),
        "delta_p_janela": serie_janela("delta_p"),
        "n_amostras": resultado["n_amostras"],
        "janela_s": resultado["janela_s"],
        "dt": resultado["dt"],
        "mensagem": resultado["mensagem"],
    }


def _executar_pipeline_fisiologico_dav(
    calibracao_paciente,
    estado,
    pam_operacional_inicial=None,
):
    resultado = {
        "resultado_fuzzy": None,
        "resultado_dav": None,
        "historico_dav": None,
        "ponto_futuro": {
            "fluxo": 0,
            "delta_p": 0,
        },
        "r_eff_dinamica": None,
        "acionar_sisc": False,
        "mensagem_sisc": None,
        "pam_operacional_inicial": None,
    }

    if not (
        calibracao_paciente
        and calibracao_paciente.get("r_eff") is not None
    ):
        return resultado

    resultado["resultado_fuzzy"] = calcular_demanda_fuzzy(
        fc_val=estado["fc_simulada"],
        hrv_val=estado["hrv_simulada"],
        imu_val=estado["imu_val"],
        spo2_val=estado["spo2_val"],
        temp_val=estado["temp_val"],
        PAM_base=calibracao_paciente.get("pam_base"),
        delta_PAM=15,
    )

    resultado["r_eff_dinamica"] = calcular_r_eff_dinamica(
        r_eff_base=calibracao_paciente["r_eff"],
        fc=estado["fc_simulada"],
        hrv=estado["hrv_simulada"],
        imu=estado["imu_val"],
        spo2=estado["spo2_val"],
        temp=estado["temp_val"],
    )

    if resultado["resultado_fuzzy"]["alerta"]:
        resultado["acionar_sisc"] = True
        resultado["mensagem_sisc"] = "ALERTA FISIOLOGICO: acionar SISC."

    pam_alvo = resultado["resultado_fuzzy"]["PAM_alvo"]
    pam_inicial_controle = (
        float(pam_operacional_inicial)
        if pam_operacional_inicial is not None
        else float(calibracao_paciente["pam_base"])
    )

    resultado["resultado_dav"] = executar_pipeline_dav(
        rpm_atual=5000,
        pam_alvo=pam_alvo,
        r_eff=resultado["r_eff_dinamica"],
        tau_base=calibracao_paciente["tau_base"],
        tau_atual=calibracao_paciente["tau_base"],
        fluxo_base=calibracao_paciente["fluxo_base"],
        pam_base=calibracao_paciente["pam_base"],
        pam_atual=pam_inicial_controle,
        pam_min=65,
        pam_max=140,
    )

    resultado["ponto_futuro"] = calcular_ponto_operacao_hm3(
        rpm=resultado["resultado_dav"]["rpm_sugerido"],
        r_eff=resultado["r_eff_dinamica"],
    )

    resultado["historico_dav"] = simular_controle_dav(
        rpm_inicial=5000,
        pam_alvo=pam_alvo,
        r_eff=resultado["r_eff_dinamica"],
        tau_base=calibracao_paciente["tau_base"],
        tau_atual=calibracao_paciente["tau_base"],
        kp=20,
        ciclos=30,
        pam_max=140,
        pam_base=calibracao_paciente["pam_base"],
        pam_inicial=pam_inicial_controle,
    )
    resultado["pam_operacional_inicial"] = pam_inicial_controle

    return resultado


def dataset_sinais(request, pk):
    registro = MIMICRecord.objects.get(pk=pk)

    inicio = _parse_int_param(
        request.GET.get("inicio", "0"),
        0,
    )
    amostras = _parse_int_param(
        request.GET.get("amostras", "5000"),
        5000,
    )
    segmento_selecionado = request.GET.get("segmento")

    segmentos = []
    segmento_atual = None
    info_segmento = None
    nomes_canais = []
    canais_json = None

    ecg = {
        "resultado_picos": None,
        "picos_json": json.dumps([]),
        "metricas_ecg": None,
        "prob_ia_json": None,
        "janelas_qrs_json": None,
    }
    abp = {
        "metricas_abp": None,
        "resultado_abp": None,
        "metricas_fc_abp": None,
        "erro_fc": None,
        "pressao_pulso_normalizada": None,
    }
    tau_pam = {
        "taus": [],
        "tau_medio": None,
        "tau_std": None,
        "resultados_tau": [],
        "pam_batimentos": [],
        "pam_modelo_bruta": [],
        "rr_batimentos": [],
        "pas_batimentos": [],
        "pad_batimentos": [],
        "pp_batimentos": [],
    }
    modelo_pam = {
        "pam_modelo_batimentos": [],
        "pam_reconstruida_abp": [],
        "tau_modelo": None,
        "offset_calibracao": None,
        "metodo_sem_abp": None,
    }
    metricas_modelo_pam = None
    estado = {
        "usar_slider": False,
        "fc_simulada": 75,
        "hrv_simulada": 40,
        "imu_val": 0.1,
        "spo2_val": 98,
        "temp_val": 36.5,
    }
    dav = {
        "resultado_fuzzy": None,
        "resultado_dav": None,
        "historico_dav": None,
        "ponto_futuro": {
            "fluxo": 0,
            "delta_p": 0,
        },
        "r_eff_dinamica": None,
        "acionar_sisc": False,
        "mensagem_sisc": None,
    }

    calibracao_paciente = None
    comparacao_pam_controle = None
    pi_dav = None

    if registro.pasta_extraida:
        segmentos = listar_segmentos_extraidos(
            registro.pasta_extraida
        )
        segmento_atual = _selecionar_segmento(
            segmentos,
            segmento_selecionado,
        )

    if segmento_atual:
        dados = ler_waveform_canais(
            segmento_atual["arquivo_dat"],
            max_amostras=amostras,
            inicio=inicio,
        )

        info_segmento = dados["info"]
        canais = dados["canais"]
        nomes_canais = list(canais.keys())
        canais_json = json.dumps(canais)
        fs = info_segmento["fs"]

        # 1. ECG: detecta picos R, FC e HRV.
        ecg = _analisar_ecg(
            canais,
            fs,
        )

        # 2. ABP: calcula PAS/PAD/PAM, pulsos e erro ECG x ABP.
        abp = _analisar_abp(
            canais,
            fs,
            ecg["metricas_ecg"],
        )

        # 3. Tau/PAM: segmenta batimentos e estima a assinatura vascular.
        tau_pam = _estimar_tau_e_pam(
            canais,
            ecg["resultado_picos"],
            fs,
        )

        # 4. Modelo local: ECG delimita RR e ABP calibra o ajuste exponencial.
        modelo_pam = _ajustar_modelo_pam(
            tau_pam["taus"],
            tau_pam["pam_batimentos"],
            ecg["metricas_ecg"],
            pam_modelo_bruta=tau_pam["pam_modelo_bruta"],
            pas_batimentos=tau_pam["pas_batimentos"],
            pad_batimentos=tau_pam["pad_batimentos"],
            pp_batimentos=tau_pam["pp_batimentos"],
            picos_r=ecg["resultado_picos"]["picos"] if ecg["resultado_picos"] else None,
            fs=fs,
        )

        if (
            tau_pam["pam_batimentos"]
            and modelo_pam["pam_modelo_batimentos"]
        ):
            n_metricas = min(
                len(tau_pam["pam_batimentos"]),
                len(modelo_pam["pam_modelo_batimentos"]),
            )
            metricas_modelo_pam = _metricas_regressao(
                tau_pam["pam_batimentos"][:n_metricas],
                modelo_pam["pam_modelo_batimentos"][:n_metricas],
            )
            erros_modelo = np.array(
                tau_pam["pam_batimentos"][:n_metricas],
                dtype=float,
            ) - np.array(
                modelo_pam["pam_modelo_batimentos"][:n_metricas],
                dtype=float,
            )
            metricas_modelo_pam["erro_medio"] = float(
                np.mean(erros_modelo)
            )
            metricas_modelo_pam["erro_max_abs"] = float(
                np.max(np.abs(erros_modelo))
            )

        # 5. Calibracao: transforma a janela em parametros de paciente.
        calibracao_paciente = _calibrar_paciente(
            tau_pam["tau_medio"],
            tau_pam["tau_std"],
            tau_pam["pam_batimentos"],
            ecg["metricas_ecg"],
        )

        # 6. Estado fisiologico: sliders ou valores basais do paciente.
        estado = _estado_fisiologico_simulado(
            request,
            calibracao_paciente,
        )

        pam_modelo_sem_abp = None
        if modelo_pam["pam_modelo_batimentos"]:
            pam_modelo_sem_abp = float(
                np.mean(modelo_pam["pam_modelo_batimentos"])
            )

        # 7. Fuzzy + DAV: demanda, PAM alvo, RPM, fluxo e curva HQ.
        dav = _executar_pipeline_fisiologico_dav(
            calibracao_paciente,
            estado,
            pam_operacional_inicial=pam_modelo_sem_abp,
        )

        pi_dav = _calcular_pi_dav_por_potencia(dav["historico_dav"])

        pam_real_referencia = None
        if tau_pam["pam_batimentos"]:
            pam_real_referencia = float(np.mean(tau_pam["pam_batimentos"]))
        elif abp["metricas_abp"]:
            pam_real_referencia = float(abp["metricas_abp"]["pam"])

        pam_alvo_fuzzy = (
            float(dav["resultado_fuzzy"]["PAM_alvo"])
            if dav["resultado_fuzzy"]
            else None
        )
        pam_estimada_bomba = (
            float(dav["resultado_dav"]["pam_estimada"])
            if dav["resultado_dav"]
            else None
        )

        if pam_real_referencia is not None:
            comparacao_pam_controle = {
                "pam_real_dataset": round(pam_real_referencia, 3),
                "pam_modelo_sem_abp": (
                    round(pam_modelo_sem_abp, 3)
                    if pam_modelo_sem_abp is not None
                    else None
                ),
                "pam_alvo_fuzzy": (
                    round(pam_alvo_fuzzy, 3)
                    if pam_alvo_fuzzy is not None
                    else None
                ),
                "pam_estimada_bomba": (
                    round(pam_estimada_bomba, 3)
                    if pam_estimada_bomba is not None
                    else None
                ),
                "pam_operacional_inicial": (
                    round(dav["pam_operacional_inicial"], 3)
                    if dav.get("pam_operacional_inicial") is not None
                    else None
                ),
                "delta_alvo_real": (
                    round(pam_alvo_fuzzy - pam_real_referencia, 3)
                    if pam_alvo_fuzzy is not None
                    else None
                ),
                "delta_modelo_real": (
                    round(pam_modelo_sem_abp - pam_real_referencia, 3)
                    if pam_modelo_sem_abp is not None
                    else None
                ),
                "delta_bomba_real": (
                    round(pam_estimada_bomba - pam_real_referencia, 3)
                    if pam_estimada_bomba is not None
                    else None
                ),
            }

    resultado_dav = dav["resultado_dav"]
    historico_dav = dav["historico_dav"]
    ponto_futuro = dav["ponto_futuro"]
    r_eff_dinamica = dav["r_eff_dinamica"]
    validade_calibracao = _avaliar_validade_calibracao(
        calibracao_paciente,
        estado,
        resultado_dav=resultado_dav,
        resultado_fuzzy=dav["resultado_fuzzy"],
    )

    return render(
        request,
        "DAVsimulator/dataset_sinais.html",
        {
            "registro": registro,
            "segmentos": segmentos,
            "canais_json": canais_json,
            "nomes_canais": nomes_canais,
            "info_segmento": info_segmento,

            "usar_slider": estado["usar_slider"],
            "fc_simulada": estado["fc_simulada"],
            "hrv_simulada": estado["hrv_simulada"],
            "imu_val": estado["imu_val"],
            "spo2_val": estado["spo2_val"],
            "temp_val": estado["temp_val"],

            "resultado_picos": ecg["resultado_picos"],
            "picos_json": ecg["picos_json"],
            "metricas_ecg": ecg["metricas_ecg"],

            "metricas_abp": abp["metricas_abp"],
            "resultado_abp": abp["resultado_abp"],
            "metricas_fc_abp": abp["metricas_fc_abp"],
            "erro_fc": abp["erro_fc"],
            "pressao_pulso_normalizada": abp["pressao_pulso_normalizada"],
            "pi_dav": pi_dav,

            "prob_ia_json": ecg["prob_ia_json"],
            "janelas_qrs_json": ecg["janelas_qrs_json"],

            "taus": tau_pam["taus"],
            "tau_medio": tau_pam["tau_medio"],
            "tau_std": tau_pam["tau_std"],
            "resultados_tau": tau_pam["resultados_tau"],
            "resultados_tau_json": json.dumps(
                tau_pam["resultados_tau"]
            ),

            "pam_batimentos_json": json.dumps(
                tau_pam["pam_batimentos"]
            ),
            "pam_modelo_batimentos_json": json.dumps(
                modelo_pam["pam_modelo_batimentos"]
            ),
            "pam_reconstruida_abp_json": json.dumps(
                modelo_pam["pam_reconstruida_abp"]
            ),
            "pas_batimentos_json": json.dumps(
                tau_pam["pas_batimentos"]
            ),
            "pad_batimentos_json": json.dumps(
                tau_pam["pad_batimentos"]
            ),
            "pp_batimentos_json": json.dumps(
                tau_pam["pp_batimentos"]
            ),

            "tau_modelo": (
                round(float(modelo_pam["tau_modelo"]), 3)
                if modelo_pam["tau_modelo"] is not None
                else None
            ),
            "offset_calibracao": (
                round(float(modelo_pam["offset_calibracao"]), 3)
                if modelo_pam["offset_calibracao"] is not None
                else None
            ),
            "tau_modelo_json": json.dumps(
                float(modelo_pam["tau_modelo"])
                if modelo_pam["tau_modelo"] is not None
                else None
            ),
            "offset_calibracao_json": json.dumps(
                float(modelo_pam["offset_calibracao"])
                if modelo_pam["offset_calibracao"] is not None
                else None
            ),
            "metricas_modelo_pam": (
                {
                    "mae": round(metricas_modelo_pam["mae"], 3),
                    "rmse": round(metricas_modelo_pam["rmse"], 3),
                    "msre": round(metricas_modelo_pam["msre"], 6),
                    "mape": round(metricas_modelo_pam["mape"], 2),
                    "r2": round(metricas_modelo_pam["r2"], 4),
                    "erro_medio": round(
                        metricas_modelo_pam["erro_medio"],
                        3,
                    ),
                    "erro_max_abs": round(
                        metricas_modelo_pam["erro_max_abs"],
                        3,
                    ),
                }
                if metricas_modelo_pam
                else None
            ),
            "observador_pressorico": [],
            "metricas_observador_pressorico": None,
            "observador_pressorico_json": json.dumps([]),
            "observador_windkessel_fases": [],
            "metricas_observador_windkessel_fases": None,
            "observador_windkessel_fases_json": json.dumps(None),

            "calibracao_paciente": calibracao_paciente,
            "resultado_dav": resultado_dav,
            "historico_dav_json": json.dumps(historico_dav),
            "resultado_fuzzy": dav["resultado_fuzzy"],
            "validade_calibracao": validade_calibracao,
            "comparacao_pam_controle": comparacao_pam_controle,
            "pam_alvo_controle_json": json.dumps(
                dav["resultado_fuzzy"]["PAM_alvo"]
                if dav["resultado_fuzzy"]
                else None
            ),
            "pam_estimada_bomba_json": json.dumps(
                resultado_dav["pam_estimada"]
                if resultado_dav
                else None
            ),

            "acionar_sisc": dav["acionar_sisc"],
            "mensagem_sisc": dav["mensagem_sisc"],

            "inicio": inicio,
            "amostras": amostras,
            "segmento_selecionado": (
                segmento_atual["nome"]
                if segmento_atual
                else None
            ),

            "ponto_hm3_json": json.dumps({
                "fluxo": (
                    resultado_dav.get("fluxo_estimado")
                    if resultado_dav
                    else 0
                ),
                "delta_p": (
                    resultado_dav.get("delta_p")
                    if resultado_dav
                    else 0
                ),
            }),
            "curvas_hq_json": json.dumps(CURVAS_HQ),
            "ponto_futuro_json": json.dumps({
                "fluxo": ponto_futuro["fluxo"],
                "delta_p": ponto_futuro["delta_p"],
            }),
            "historico_hq_json": json.dumps({
                "fluxo": (
                    historico_dav["fluxo"]
                    if historico_dav
                    else []
                ),
                "delta_p": (
                    historico_dav["delta_p"]
                    if historico_dav
                    else []
                ),
                "rpm": (
                    historico_dav["rpm"]
                    if historico_dav
                    else []
                ),
            }),
            "r_eff_dinamica_json": json.dumps(
                r_eff_dinamica or 0
            ),
        },
    )


def validacao_calibracao(request):
    registro = MIMICRecord.objects.first()
    segmentos = (
        listar_segmentos_extraidos(registro.pasta_extraida)
        if registro and registro.pasta_extraida
        else []
    )

    resultados = []
    pam_real_global = []
    pam_pred_global = []
    pam_pred_basico_global = []
    pam_real_inteira = []
    pam_pred_inteira = []

    for segmento in segmentos:
        try:
            dados = ler_waveform_canais(
                segmento["arquivo_dat"],
                max_amostras=5000,
                inicio=0,
            )

            fs = dados["info"]["fs"]
            canais = dados["canais"]

            if "II" not in canais or "ABP" not in canais:
                continue

            ecg = _analisar_ecg(canais, fs)
            abp = _analisar_abp(canais, fs, ecg["metricas_ecg"])
            tau_pam = _estimar_tau_e_pam(
                canais,
                ecg["resultado_picos"],
                fs,
            )

            validacao = _validar_modelo_pam_batimento(
                tau_pam["taus"],
                tau_pam["pam_batimentos"],
                ecg["metricas_ecg"],
                pam_modelo_bruta=tau_pam["pam_modelo_bruta"],
                pas_batimentos=tau_pam["pas_batimentos"],
                pad_batimentos=tau_pam["pad_batimentos"],
                pp_batimentos=tau_pam["pp_batimentos"],
                picos_r=ecg["resultado_picos"]["picos"] if ecg["resultado_picos"] else None,
                fs=fs,
            )

            modelo_inteiro = _ajustar_modelo_pam(
                tau_pam["taus"],
                tau_pam["pam_batimentos"],
                ecg["metricas_ecg"],
                pam_modelo_bruta=tau_pam["pam_modelo_bruta"],
                pas_batimentos=tau_pam["pas_batimentos"],
                pad_batimentos=tau_pam["pad_batimentos"],
                pp_batimentos=tau_pam["pp_batimentos"],
                picos_r=(
                    ecg["resultado_picos"]["picos"]
                    if ecg["resultado_picos"]
                    else None
                ),
                fs=fs,
            )

            calibracao = _calibrar_paciente(
                tau_pam["tau_medio"],
                tau_pam["tau_std"],
                tau_pam["pam_batimentos"],
                ecg["metricas_ecg"],
            )

            if not validacao or not calibracao:
                continue

            if modelo_inteiro["pam_modelo_batimentos"]:
                n_inteiro = min(
                    len(tau_pam["pam_batimentos"]),
                    len(modelo_inteiro["pam_modelo_batimentos"]),
                )
                pam_real_inteira.extend(
                    tau_pam["pam_batimentos"][:n_inteiro]
                )
                pam_pred_inteira.extend(
                    modelo_inteiro["pam_modelo_batimentos"][:n_inteiro]
                )

            pam_real_global.extend(validacao["pam_real"])
            pam_pred_global.extend(validacao["pam_pred"])
            pam_pred_basico_global.extend(validacao["pam_pred_basico"])

            metricas = validacao["metricas"]
            metricas_basico = validacao["metricas_basico"]

            resultados.append({
                "segmento": segmento["nome"],
                "fc": round(calibracao["fc_base"] or 0, 1),
                "hrv": round(calibracao["hrv_base"] or 0, 1),
                "pam_media": round(calibracao["pam_base"] or 0, 2),
                "tau": round(calibracao["tau_base"] or 0, 3),
                "tau_std": round(calibracao["tau_std"] or 0, 3),
                "r_eff": round(calibracao["r_eff"] or 0, 2),
                "n_treino": validacao["n_treino"],
                "n_teste": validacao["n_teste"],
                "mae": round(metricas["mae"], 3),
                "rmse": round(metricas["rmse"], 3),
                "msre": round(metricas["msre"], 6),
                "mape": round(metricas["mape"], 2),
                "r2": round(metricas["r2"], 4),
                "mae_basico": round(metricas_basico["mae"], 3),
                "rmse_basico": round(metricas_basico["rmse"], 3),
                "modelo": validacao["modelo"],
            })

        except Exception as exc:
            print("ERRO VALIDACAO CALIBRACAO:", segmento.get("nome"), exc)

    metricas_globais = _metricas_regressao(
        pam_real_global,
        pam_pred_global,
    )
    metricas_inteira = _metricas_regressao(
        pam_real_inteira,
        pam_pred_inteira,
    )

    return render(
        request,
        "DAVsimulator/validacao_calibracao.html",
        {
            "registro": registro,
            "resultados": resultados,
            "quantidade": len(resultados),
            "metricas_globais": {
                "mae": round(metricas_globais["mae"], 3),
                "rmse": round(metricas_globais["rmse"], 3),
                "msre": round(metricas_globais["msre"], 6),
                "mape": round(metricas_globais["mape"], 2),
                "r2": round(metricas_globais["r2"], 4),
            },
            "metricas_inteira": {
                "mae": round(metricas_inteira["mae"], 3),
                "rmse": round(metricas_inteira["rmse"], 3),
                "msre": round(metricas_inteira["msre"], 6),
                "mape": round(metricas_inteira["mape"], 2),
                "r2": round(metricas_inteira["r2"], 4),
            },
            "pam_real_json": json.dumps(pam_real_global),
            "pam_pred_json": json.dumps(pam_pred_global),
            "pam_pred_basico_json": json.dumps(pam_pred_basico_global),
            "pam_real_inteira_json": json.dumps(pam_real_inteira),
            "pam_pred_inteira_json": json.dumps(pam_pred_inteira),
        },
    )


def validacao_windkessel_dav(request):
    registro = MIMICRecord.objects.first()
    segmentos = (
        listar_segmentos_extraidos(registro.pasta_extraida)
        if registro and registro.pasta_extraida
        else []
    )

    resultados = []
    pam_real_global = []
    pam_pred_global = []
    r_global = []
    c_global = []
    tau_global = []

    for segmento in segmentos:
        try:
            dados = ler_waveform_canais(
                segmento["arquivo_dat"],
                max_amostras=5000,
                inicio=0,
            )

            fs = dados["info"]["fs"]
            canais = dados["canais"]

            if "II" not in canais or "ABP" not in canais:
                continue

            ecg = _analisar_ecg(canais, fs)
            abp = _analisar_abp(canais, fs, ecg["metricas_ecg"])
            tau_pam = _estimar_tau_e_pam(
                canais,
                ecg["resultado_picos"],
                fs,
            )

            if len(tau_pam["pam_batimentos"]) < 8:
                continue

            quantidade = len(tau_pam["pam_batimentos"])
            split = max(6, int(quantidade * 0.7))

            if quantidade - split < 2:
                split = quantidade - 2

            validacao = _validar_modelo_pam_batimento(
                tau_pam["taus"],
                tau_pam["pam_batimentos"],
                ecg["metricas_ecg"],
                pam_modelo_bruta=tau_pam["pam_modelo_bruta"],
                pas_batimentos=tau_pam["pas_batimentos"],
                pad_batimentos=tau_pam["pad_batimentos"],
                pp_batimentos=tau_pam["pp_batimentos"],
                picos_r=(
                    ecg["resultado_picos"]["picos"]
                    if ecg["resultado_picos"]
                    else None
                ),
                fs=fs,
            )

            calibracao = _calibrar_paciente(
                tau_pam["tau_medio"],
                tau_pam["tau_std"],
                tau_pam["pam_batimentos"][:split],
                ecg["metricas_ecg"],
            )

            if not validacao or not calibracao:
                continue

            pam_real = validacao["pam_real"]
            pam_pred = validacao["pam_pred"]
            metricas = validacao["metricas"]

            pam_real_global.extend(pam_real)
            pam_pred_global.extend(pam_pred)
            r_global.extend([calibracao["r_eff"]] * len(pam_real))
            tau_global.extend(tau_pam["taus"][split:split + len(pam_real)])

            resultados.append({
                "segmento": segmento["nome"],
                "n_treino": split,
                "n_teste": quantidade - split,
                "mae": round(metricas["mae"], 3),
                "rmse": round(metricas["rmse"], 3),
                "msre": round(metricas["msre"], 6),
                "mape": round(metricas["mape"], 2),
                "r2": round(metricas["r2"], 4),
                "r_medio": round(float(calibracao["r_eff"] or 0), 3),
                "c_medio": "-",
                "tau_medio": round(float(np.mean(tau_pam["taus"][split:])), 3),
                "q_bomba": round(float(calibracao.get("fluxo_base") or 0), 3),
                "q_nativo": "-",
                "metodo": validacao["metodo_sem_abp"],
            })

        except Exception as exc:
            print("ERRO VALIDACAO WINDKESSEL DAV:", segmento.get("nome"), exc)

    metricas_globais = _metricas_regressao(
        pam_real_global,
        pam_pred_global,
    )

    return render(
        request,
        "DAVsimulator/validacao_windkessel_dav.html",
        {
            "registro": registro,
            "resultados": resultados,
            "quantidade": len(resultados),
            "metricas_globais": {
                "mae": round(metricas_globais["mae"], 3),
                "rmse": round(metricas_globais["rmse"], 3),
                "msre": round(metricas_globais["msre"], 6),
                "mape": round(metricas_globais["mape"], 2),
                "r2": round(metricas_globais["r2"], 4),
            },
            "pam_real_json": json.dumps(pam_real_global),
            "pam_pred_json": json.dumps(pam_pred_global),
            "r_json": json.dumps(r_global),
            "c_json": json.dumps(c_global),
            "tau_json": json.dumps(tau_global),
        },
    )


def _parse_float_param(valor, padrao):
    if valor is None:
        return padrao

    try:
        return float(str(valor).replace(",", "."))
    except (TypeError, ValueError):
        return padrao


def validacao_sistema_controle(request):
    kp = _parse_float_param(request.GET.get("kp"), 20.0)
    pam_base = _parse_float_param(request.GET.get("pam_base"), 122.5)
    r_eff = _parse_float_param(request.GET.get("r_eff"), 24.0)
    tau = _parse_float_param(request.GET.get("tau"), 0.8)
    fluxo_base = _parse_float_param(request.GET.get("fluxo_base"), 3.0)
    modo_curva = request.GET.get("modo_curva", "validado")

    if modo_curva not in ("validado", "extrapolado"):
        modo_curva = "validado"

    pasta_saida = (
        settings.BASE_DIR
        / "DAVsimulator"
        / "resultados"
    )

    resultado = executar_validacao_dinamica_controle(
        pam_base=pam_base,
        r_eff_base=r_eff,
        tau_base=tau,
        fluxo_base=fluxo_base,
        kp=kp,
        modo_curva=modo_curva,
        pasta_saida=pasta_saida,
    )

    ensaio_principal = (
        resultado["ensaios_degrau"][1]
        if len(resultado["ensaios_degrau"]) > 1
        else resultado["ensaios_degrau"][0]
        if resultado["ensaios_degrau"]
        else None
    )
    historico_principal = (
        ensaio_principal["historico"]
        if ensaio_principal
        else {}
    )
    mapa_kp = resultado["mapa_kp"] or {"resultados": []}

    multicenario = []
    for item in resultado["multicenario"]:
        hist = item["historico"]
        multicenario.append({
            "nome": item["nome"],
            "demanda": round(float(item["fuzzy"]["demanda"]), 3),
            "pam_alvo": round(float(item["pam_alvo"]), 2),
            "pam_inicial": round(float(item["pam_inicial"]), 2),
            "r_eff": round(float(item["r_eff"]), 3),
            "rpm_inicial": round(float(hist["rpm_aplicada"][0]), 0),
            "rpm_final": round(float(hist["rpm_aplicada"][-1]), 0),
            "fluxo_inicial": round(float(hist["q_dav"][0]), 3),
            "fluxo_final": round(float(hist["q_dav"][-1]), 3),
            "erro_regime": item["metricas"]["erro_regime"],
            "rise_time": item["metricas"]["rise_time"],
            "settling_time": item["metricas"]["settling_time_2pct"],
            "overshoot": item["metricas"]["overshoot_pct"],
            "iae": item["metricas"]["iae"],
            "ise": item["metricas"]["ise"],
            "itae": item["metricas"]["itae"],
            "saturacao": item["metricas"]["percentual_saturacao"],
            "rho": item["estabilidade"]["raio_espectral"],
            "b_local": item["modelo_local"].get("b_formatado", "-"),
            "classificacao": item["estabilidade"]["classificacao"],
        })

    robustez = []
    for item in resultado["robustez"]:
        robustez.append({
            "cenario": item["cenario"],
            "parametro": item["parametro"],
            "fator": item["fator"],
            "erro_regime": item["metricas"]["erro_regime"],
            "overshoot": item["metricas"]["overshoot_pct"],
            "saturacao": item["metricas"]["percentual_saturacao"],
            "rho": item["estabilidade"]["raio_espectral"],
            "b_local": item["modelo_local"].get("b_formatado", "-"),
            "classificacao": item["estabilidade"]["classificacao"],
        })

    return render(
        request,
        "DAVsimulator/validacao_sistema_controle.html",
        {
            "resultado": resultado,
            "parametros": resultado["parametros"],
            "ensaios_degrau": resultado["ensaios_degrau"],
            "ensaio_principal": ensaio_principal,
            "metricas": (
                ensaio_principal["metricas"]
                if ensaio_principal
                else None
            ),
            "modelo_local": resultado["perturbacao_local"]["modelo_local"],
            "estabilidade": resultado["perturbacao_local"]["estabilidade"],
            "mapa_kp": mapa_kp,
            "multicenario": multicenario,
            "robustez": robustez,
            "historico_json": json.dumps(historico_principal),
            "mapa_kp_json": json.dumps(mapa_kp["resultados"]),
            "arquivos": resultado["arquivos"],
        },
    )


def validacao_controle(request):

    resultados = []
    fc_erros = []
    hrv_erros = []
    sensibilidade_lista = []
    precisao_lista = []
    f1_lista = []
    ecg_preview = None

    registro = MIMICRecord.objects.first()
    segmentos = listar_segmentos_extraidos(registro.pasta_extraida)

    for segmento in segmentos:

        try:
            dados = ler_waveform_canais(
                segmento["arquivo_dat"],
                max_amostras=5000,
                inicio=0
            )

            fs = dados["info"]["fs"]

            if "II" not in dados["canais"]:
                continue

            if len(dados["canais"]["II"]) != 5000:
                continue

            picos_ia, _, _ = detectar_indices_picos_r_ia(
                dados["canais"]["II"],
                fs=fs,
                threshold=0.5
            )

            metricas = calcular_fc_hrv(
                picos_ia,
                fs=fs
            )

            fc = metricas["fc"]
            hrv = metricas["rmssd"]

            resultado_abp = None
            metricas_abp_fc = None
            comparacao_picos = None
            erro_fc = None
            erro_hrv = None

            if "ABP" in dados["canais"]:
                resultado_abp = detectar_pulsos_abp(
                    dados["canais"]["ABP"],
                    fs=fs,
                )

                if resultado_abp and resultado_abp.get("picos"):
                    metricas_abp_fc = calcular_fc_hrv(
                        resultado_abp["picos"],
                        fs=fs,
                    )

                    erro_fc = abs(
                        fc - metricas_abp_fc["fc"]
                    )
                    erro_hrv = abs(
                        hrv - metricas_abp_fc["rmssd"]
                    )

                    fc_erros.append(erro_fc)
                    hrv_erros.append(erro_hrv)

                    comparacao_picos = avaliar_picos_r(
                        resultado_abp["picos"],
                        picos_ia,
                        fs=fs,
                        tolerancia_ms=300,
                    )

                    sensibilidade_lista.append(
                        comparacao_picos["sensibilidade"]
                    )
                    precisao_lista.append(
                        comparacao_picos["precisao"]
                    )
                    f1_lista.append(
                        comparacao_picos["f1"]
                    )

            if ecg_preview is None:
                ecg_preview = {
                    "segmento": segmento["nome"],
                    "ecg": dados["canais"]["II"],
                    "picos": picos_ia.tolist(),
                    "fs": fs,
                }

            resultado_fuzzy = calcular_demanda_fuzzy(
                fc_val=fc,
                hrv_val=hrv,
                imu_val=0.3,
                spo2_val=98,
                temp_val=36.5,
                PAM_base=110,
                delta_PAM=15,
            )

            resultado_dav = executar_pipeline_dav(
                rpm_atual=5000,
                pam_alvo=resultado_fuzzy["PAM_alvo"],
                r_eff=24,
                tau_base=0.40,
                tau_atual=0.40,
                fluxo_base=3.0,
                pam_base=110
            )

            historico = simular_controle_dav(
                rpm_inicial=5000,
                pam_alvo=resultado_fuzzy["PAM_alvo"],
                r_eff=24,
                tau_base=0.40,
                tau_atual=0.40,
                fluxo_base=3.0,
                pam_base=110,
                kp=20,
                ciclos=30,
                pam_max=140
            )

            rpm_final = historico["rpm"][-1]
            fluxo_final = historico["fluxo"][-1]
            pam_final = historico["pam"][-1]

            resultados.append({
                "segmento": segmento["nome"],
                "fc": round(fc, 1),
                "hrv": round(hrv, 1),
                "fc_abp": (
                    round(metricas_abp_fc["fc"], 1)
                    if metricas_abp_fc
                    else None
                ),
                "hrv_abp": (
                    round(metricas_abp_fc["rmssd"], 1)
                    if metricas_abp_fc
                    else None
                ),
                "erro_fc": (
                    round(erro_fc, 2)
                    if erro_fc is not None
                    else None
                ),
                "erro_hrv": (
                    round(erro_hrv, 2)
                    if erro_hrv is not None
                    else None
                ),
                "sensibilidade": (
                    comparacao_picos["sensibilidade"]
                    if comparacao_picos
                    else None
                ),
                "precisao": (
                    comparacao_picos["precisao"]
                    if comparacao_picos
                    else None
                ),
                "f1": (
                    comparacao_picos["f1"]
                    if comparacao_picos
                    else None
                ),
                "classe": resultado_fuzzy["classe"],
                "demanda": round(resultado_fuzzy["demanda"], 2),
                "pam_alvo": round(resultado_fuzzy["PAM_alvo"], 1),
                "rpm_final": round(rpm_final, 0),
                "fluxo_final": round(fluxo_final, 2),
                "pam_final": round(pam_final, 1),
            })

        except Exception as e:
            print(segmento.get("nome"), e)

    return render(
        request,
        "DAVsimulator/validacao_controle.html",
        {
            "resultados": resultados,
            "metricas_ecg": {
                "mae_fc": (
                    round(float(np.mean(fc_erros)), 3)
                    if fc_erros
                    else None
                ),
                "mae_hrv": (
                    round(float(np.mean(hrv_erros)), 3)
                    if hrv_erros
                    else None
                ),
                "sensibilidade_media": (
                    round(float(np.mean(sensibilidade_lista)), 4)
                    if sensibilidade_lista
                    else None
                ),
                "precisao_media": (
                    round(float(np.mean(precisao_lista)), 4)
                    if precisao_lista
                    else None
                ),
                "f1_medio": (
                    round(float(np.mean(f1_lista)), 4)
                    if f1_lista
                    else None
                ),
                "referencia": "Pulsos ABP usados como referencia fisiologica indireta",
            },
            "ecg_preview_json": json.dumps(ecg_preview or {}),
        }
    )
