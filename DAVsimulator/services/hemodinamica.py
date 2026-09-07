from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks

import numpy as np


# ==========================================================
# MÃ‰TRICAS ABP
# ==========================================================

def calcular_metricas_abp(sinal_abp):

    if not sinal_abp:
        return None

    sinal = np.array(sinal_abp)

    pas = float(np.max(sinal))

    pad = float(np.min(sinal))

    pam = (
        pas + 2 * pad
    ) / 3

    return {
        "pas": round(pas, 2),
        "pad": round(pad, 2),
        "pam": round(pam, 2),
    }


# ==========================================================
# DETECÃ‡ÃƒO DE PULSOS ABP
# ==========================================================

def detectar_pulsos_abp(
    sinal_abp,
    fs=125
):

    sinal = np.array(sinal_abp)

    peaks, _ = find_peaks(
        sinal,
        distance=int(fs * 0.4),
        prominence=5
    )

    pulsos = []

    for i in range(len(peaks) - 1):

        inicio = int(peaks[i])

        fim = int(peaks[i + 1])

        trecho = sinal[inicio:fim]

        if len(trecho) < 5:
            continue

        pas = float(np.max(trecho))

        pad = float(np.min(trecho))

        pam = (
            pas + 2 * pad
        ) / 3

        pulsos.append({
            "inicio": inicio,
            "fim": fim,
            "pas": round(pas, 2),
            "pad": round(pad, 2),
            "pam": round(pam, 2),
        })

    return {
        "picos": peaks.tolist(),
        "pulsos": pulsos,
    }


# ==========================================================
# SEGMENTAÃ‡ÃƒO ABP
# ==========================================================

def segmentar_batimentos_abp(
    abp,
    picos_r
):

    segmentos = []

    for i in range(len(picos_r) - 1):

        inicio = int(picos_r[i])

        fim = int(picos_r[i + 1])

        if fim <= inicio:
            continue

        trecho = np.array(
            abp[inicio:fim]
        )

        if len(trecho) < 20:
            continue

        segmentos.append({
            "inicio": inicio,
            "fim": fim,
            "abp": trecho.tolist(),
        })

    return segmentos


# ==========================================================
# MODELO EXPONENCIAL
# ==========================================================

def modelo_exponencial(
    t,
    p0,
    tau,
    c
):

    return (
        p0 * np.exp(-t / tau)
    ) + c


# ==========================================================
# ESTIMAÃ‡ÃƒO ROBUSTA DE TAU
# ==========================================================

def estimar_tau_batimento(
    abp,
    fs=125,
    usar_regiao_util=True
):

    abp = np.array(abp, dtype=float)

    if len(abp) < 15:
        return None

    idx_pico = int(np.argmax(abp))

    inicio_diastole = idx_pico + int(0.04 * fs)

    if inicio_diastole >= len(abp) - 8:
        return None

    diastole = abp[inicio_diastole:]
    inicio_ajuste_rel = inicio_diastole

    if len(diastole) < 8:
        return None

    if usar_regiao_util:

        n = len(diastole)

        inicio_util = int(0.10 * n)
        fim_util = int(0.90 * n)

        derivada = np.diff(diastole)
        minimo_pre_upstroke = None
        amostras_subida = 3

        for i in range(1, len(diastole) - amostras_subida):

            if (
                derivada[i - 1] <= 0
                and np.all(derivada[i:i + amostras_subida] > 0)
            ):
                minimo_pre_upstroke = i
                break

        if minimo_pre_upstroke is not None:
            fim_util = min(
                fim_util,
                minimo_pre_upstroke + 1
            )

        inicio_ajuste_rel = inicio_diastole + inicio_util
        diastole = diastole[inicio_util:fim_util]

    if len(diastole) < 8:
        return None

    t = np.arange(len(diastole)) / fs

    try:

        p0_inicial = float(diastole[0] - diastole[-1])
        c_inicial = float(diastole[-1])

        if p0_inicial <= 0:
            p0_inicial = 1.0

        popt, _ = curve_fit(
            modelo_exponencial,
            t,
            diastole,
            p0=[
                p0_inicial,
                0.5,
                c_inicial
            ],
            bounds=(
                [0.01, 0.05, 0],
                [300, 5.0, 300]
            ),
            maxfev=20000
        )

        p0_fit, tau_fit, c_fit = popt

        ajuste = modelo_exponencial(
            t,
            *popt
        )

        ss_res = np.sum(
            (diastole - ajuste) ** 2
        )

        ss_tot = np.sum(
            (diastole - np.mean(diastole)) ** 2
        )

        if ss_tot == 0:
            return None

        r2 = 1 - (ss_res / ss_tot)

        return {
            "tau": float(tau_fit),
            "p0": float(p0_fit),
            "c": float(c_fit),
            "r2": float(r2),
            "idx_pico": int(idx_pico),
            "inicio_diastole": int(inicio_diastole),
            "inicio_ajuste": int(inicio_ajuste_rel),
            "diastole": diastole.tolist(),
            "ajuste": ajuste.tolist(),
            "t": t.tolist(),
        }

    except Exception as e:

        print("ERRO TAU:", e)

        return None

    # ======================================================
    # REGIÃƒO ÃšTIL DA DIÃSTOLE
    # ======================================================

    if usar_regiao_util:

        n = len(diastole)

        inicio_util = int(0.2 * n)

        fim_util = int(0.8 * n)

        diastole = diastole[
            inicio_util:fim_util
        ]

    if len(diastole) < 10:
        return None

    t = np.arange(
        len(diastole)
    ) / fs

    try:

        popt, _ = curve_fit(
            modelo_exponencial,
            t,
            diastole,
            p0=[
                diastole[0],
                0.5,
                diastole[-1]
            ],
            maxfev=10000
        )

        p0_fit, tau_fit, c_fit = popt

        ajuste = modelo_exponencial(
            t,
            *popt
        )

        # ==================================================
        # RÂ²
        # ==================================================

        ss_res = np.sum(
            (diastole - ajuste) ** 2
        )

        ss_tot = np.sum(
            (
                diastole
                - np.mean(diastole)
            ) ** 2
        )

        if ss_tot == 0:
            return None

        r2 = 1 - (
            ss_res / ss_tot
        )

        return {
            "tau": float(tau_fit),
            "p0": float(p0_fit),
            "c": float(c_fit),
            "r2": float(r2),
            "diastole": diastole.tolist(),
            "ajuste": ajuste.tolist(),
            "t": t.tolist(),
        }

    except:

        return None


# ==========================================================
# ESTIMAÃ‡ÃƒO ROBUSTA DATASET
# ==========================================================

def estimar_tau_dataset(
    abp,
    picos_r,
    fs=125
):

    segmentos = segmentar_batimentos_abp(
        abp,
        picos_r
    )

    taus = []

    resultados = []

    for seg in segmentos:

        resultado = estimar_tau_batimento(
            seg["abp"],
            fs=fs,
            usar_regiao_util=True
        )

        if not resultado:
            continue

        tau = resultado["tau"]

        r2 = resultado["r2"]

        # filtro fisiolÃ³gico
        if (
            0.05 < tau < 5
            and r2 > 0.70
        ):

            taus.append(tau)

            resultados.append(resultado)

    if len(taus) == 0:

        return None

    return {
        "tau_medio": float(
            np.median(taus)
        ),
        "tau_std": float(
            np.std(taus)
        ),
        "taus": taus,
        "resultados": resultados,
    }


def segmentar_abp_por_picos_r(
    abp,
    picos_r,
    fs=125,
):
    """
    Segmenta a ABP usando os picos R do ECG como referencia temporal.

    Cada ciclo cardiaco fica delimitado por R_i -> R_{i+1}. O pico sistolico
    da ABP e procurado dentro desse intervalo, sem assumir que ele coincide
    temporalmente com o pico R.
    """

    if abp is None or picos_r is None:
        return []

    sinal = np.array(abp, dtype=float)
    picos = [
        int(p)
        for p in picos_r
        if 0 <= int(p) < len(sinal)
    ]

    segmentos = []
    atraso_minimo = int(0.04 * fs)
    tamanho_minimo = int(0.25 * fs)

    for i in range(len(picos) - 1):
        inicio = int(picos[i])
        fim = int(picos[i + 1])

        if fim <= inicio or fim - inicio < tamanho_minimo:
            continue

        trecho = sinal[inicio:fim]

        if len(trecho) < tamanho_minimo:
            continue

        inicio_busca_pico = min(atraso_minimo, max(len(trecho) - 1, 0))
        trecho_busca = trecho[inicio_busca_pico:]

        if len(trecho_busca) == 0:
            continue

        idx_pico_rel = int(inicio_busca_pico + np.argmax(trecho_busca))
        pas = float(np.max(trecho))
        pad = float(np.min(trecho))
        pam_media = float(np.mean(trecho))

        segmentos.append({
            "indice": len(segmentos) + 1,
            "inicio": inicio,
            "fim": fim,
            "rr_s": float((fim - inicio) / fs),
            "idx_pico_rel": idx_pico_rel,
            "idx_pico_abs": int(inicio + idx_pico_rel),
            "pas": pas,
            "pad": pad,
            "pam": pam_media,
            "pam_formula": float((pas + 2 * pad) / 3),
            "pp": float(pas - pad),
            "abp": trecho.tolist(),
        })

    return segmentos


def estimar_tau_por_ciclos_rr(
    abp,
    picos_r,
    fs=125,
):
    segmentos = segmentar_abp_por_picos_r(
        abp,
        picos_r,
        fs=fs,
    )

    candidatos = []

    for segmento in segmentos:
        resultado_tau = estimar_tau_batimento(
            segmento["abp"],
            fs=fs,
            usar_regiao_util=True,
        )

        if not resultado_tau:
            continue

        tau = resultado_tau.get("tau")
        r2 = resultado_tau.get("r2", 0)

        if not (tau and 0.05 < tau < 5 and r2 > 0.70):
            continue

        bruto_ajuste = resultado_tau.get("ajuste") or []
        pam_modelo_bruta = (
            float(np.mean(bruto_ajuste))
            if bruto_ajuste
            else float(segmento["pam"])
        )

        candidatos.append({
            **segmento,
            "tau": float(tau),
            "r2": float(r2),
            "pam_modelo_bruta": pam_modelo_bruta,
            "resultado_tau": {
                **resultado_tau,
                "batimento": int(segmento["indice"]),
                "pam": float(segmento["pam"]),
                "pas": float(segmento["pas"]),
                "pad": float(segmento["pad"]),
                "pp": float(segmento["pp"]),
                "idx_pico_abs": int(segmento["idx_pico_abs"]),
            },
        })

    if not candidatos:
        return {
            "taus": [],
            "tau_medio": None,
            "tau_std": None,
            "ciclos": [],
        }

    mediana = float(np.median([c["tau"] for c in candidatos]))
    ciclos = [
        c
        for c in candidatos
        if 0.6 * mediana <= c["tau"] <= 1.4 * mediana
    ]

    taus = [c["tau"] for c in ciclos]

    return {
        "taus": taus,
        "tau_medio": float(np.median(taus)) if taus else None,
        "tau_std": float(np.std(taus)) if taus else None,
        "ciclos": ciclos,
    }


def estimar_marcadores_pressoricos_observador(
    abp,
    picos_r,
    fs=125,
):
    """
    Estima PAS, PAD e PAM por ciclo para validar o observador.

    A ABP e usada apenas para calibracao/validacao do ciclo:
    - o ECG delimita R_i -> R_i+1;
    - a sistole e ajustada por P_sis(t) = a*t^2 + b*t + c,
      usando o trecho de subida do vale pre-sistolico ate o pico;
    - a diastole e ajustada por P_dia(t) = P0*exp(-t/tau) + C;
    - a PAM estimada e calculada por PAD + 1/3*(PAS - PAD).
    """

    segmentos = segmentar_abp_por_picos_r(
        abp,
        picos_r,
        fs=fs,
    )

    resultados = []

    for segmento in segmentos:
        trecho = np.array(segmento["abp"], dtype=float)
        idx_pico = int(segmento["idx_pico_rel"])

        if idx_pico < 2 or idx_pico >= len(trecho):
            continue

        pre_pico = trecho[:idx_pico + 1]
        idx_inicio_sistole = int(np.argmin(pre_pico))
        sistole = trecho[idx_inicio_sistole:idx_pico + 1]
        x_sistole = np.arange(len(sistole), dtype=float) / float(fs)

        if len(sistole) < 3:
            continue

        try:
            if len(sistole) >= 5:
                a_fit, b_fit, c_fit = np.polyfit(
                    x_sistole,
                    sistole,
                    2,
                )
                sistole_ajustada = np.polyval(
                    [a_fit, b_fit, c_fit],
                    x_sistole,
                )
                ss_res_sistole = float(
                    np.sum((sistole - sistole_ajustada) ** 2)
                )
                ss_tot_sistole = float(
                    np.sum((sistole - np.mean(sistole)) ** 2)
                )
                r2_sistole = (
                    float(1 - (ss_res_sistole / ss_tot_sistole))
                    if ss_tot_sistole
                    else 0.0
                )

                t_min = float(x_sistole[0])
                t_max = float(x_sistole[-1])
                vertice_valido = False

                if a_fit < 0:
                    t_pas = float(-b_fit / (2.0 * a_fit))
                    vertice_valido = (
                        t_min <= t_pas <= t_max
                        and r2_sistole > 0.70
                    )

                if vertice_valido:
                    x_pico = t_pas
                    modelo_sistole = "quadratico_vertice"
                    motivo_sistole = "vertice aceito"
                else:
                    x_pico = t_max
                    modelo_sistole = "quadratico_borda"
                    motivo_sistole = (
                        "vertice invalido; usada borda final da sistole"
                    )
            else:
                b_fit, c_fit = np.polyfit(x_sistole, sistole, 1)
                a_fit = 0.0
                r2_sistole = 0.0
                x_pico = float(x_sistole[-1])
                modelo_sistole = "linear_borda"
                motivo_sistole = "poucos pontos para ajuste quadratico"

            pas_estimada = float(
                (a_fit * (x_pico ** 2))
                + (b_fit * x_pico)
                + c_fit
            )
        except Exception:
            continue

        tau_resultado = estimar_tau_batimento(
            trecho,
            fs=fs,
            usar_regiao_util=True,
        )

        if not tau_resultado:
            continue

        tau = float(tau_resultado.get("tau"))
        r2_tau = float(tau_resultado.get("r2", 0))

        if not (0.05 < tau < 5 and r2_tau > 0.70):
            continue

        if not tau_resultado.get("ajuste"):
            continue

        inicio_ajuste = int(
            tau_resultado.get(
                "inicio_ajuste",
                tau_resultado.get("inicio_diastole", idx_pico),
            )
        )
        t_pad = max(
            0.0,
            float((len(trecho) - 1 - inicio_ajuste) / float(fs)),
        )
        pad_estimada = float(
            modelo_exponencial(
                np.array([t_pad], dtype=float),
                float(tau_resultado.get("p0")),
                tau,
                float(tau_resultado.get("c")),
            )[0]
        )
        pam_estimada = float(
            pad_estimada + ((pas_estimada - pad_estimada) / 3.0)
        )

        resultados.append({
            "batimento": int(segmento["indice"]),
            "pas_real": float(segmento["pas"]),
            "pas_estimada": pas_estimada,
            "pas_estimada_bruta": pas_estimada,
            "pad_real": float(segmento["pad"]),
            "pad_estimada": pad_estimada,
            "pad_estimada_bruta": pad_estimada,
            "pam_real": float(segmento["pam"]),
            "pam_formula_real": float(segmento["pam_formula"]),
            "pam_estimada": pam_estimada,
            "tau": tau,
            "r2_tau": r2_tau,
            "a": float(a_fit),
            "b": float(b_fit),
            "c": float(c_fit),
            "x": x_pico,
            "modelo_sistole": modelo_sistole,
            "motivo_sistole": motivo_sistole,
            "r2_sistole": r2_sistole,
            "t_pad": t_pad,
            "p0": float(tau_resultado.get("p0")),
            "p_infinito": float(tau_resultado.get("c")),
        })

    if resultados:
        offset_pas = float(np.median([
            item["pas_real"] - item["pas_estimada_bruta"]
            for item in resultados
        ]))
        offset_pad = float(np.median([
            item["pad_real"] - item["pad_estimada_bruta"]
            for item in resultados
        ]))

        for item in resultados:
            item["pas_estimada"] = float(
                item["pas_estimada_bruta"] + offset_pas
            )
            item["pad_estimada"] = float(
                item["pad_estimada_bruta"] + offset_pad
            )
            item["pam_estimada"] = float(
                item["pad_estimada"]
                + ((item["pas_estimada"] - item["pad_estimada"]) / 3.0)
            )
            item["offset_pas"] = offset_pas
            item["offset_pad"] = offset_pad

    return resultados


def _modelo_windkessel_fases(t, delta_t_em, t_s, tau_s, k, p0, tau_d, p_inf):
    t = np.array(t, dtype=float)
    p = np.full_like(t, float(p0), dtype=float)

    sistole = (t >= delta_t_em) & (t <= t_s)
    if np.any(sistole):
        ts = t[sistole] - delta_t_em
        p[sistole] = (
            p0 * np.exp(-ts / tau_s)
            + k * (1.0 - np.exp(-ts / tau_s))
        )

    pas_modelo = (
        p0 * np.exp(-(t_s - delta_t_em) / tau_s)
        + k * (1.0 - np.exp(-(t_s - delta_t_em) / tau_s))
    )

    diastole = t > t_s
    if np.any(diastole):
        td = t[diastole] - t_s
        p[diastole] = (
            p_inf
            + (pas_modelo - p_inf) * np.exp(-td / tau_d)
        )

    return p


def _metricas_serie(real, pred):
    real = np.array(real, dtype=float)
    pred = np.array(pred, dtype=float)
    n = min(len(real), len(pred))

    if n == 0:
        return {
            "mae": None,
            "rmse": None,
            "r2": None,
            "erro_max": None,
        }

    real = real[:n]
    pred = pred[:n]
    erro = real - pred
    ss_res = float(np.sum(erro ** 2))
    ss_tot = float(np.sum((real - np.mean(real)) ** 2))

    return {
        "mae": float(np.mean(np.abs(erro))),
        "rmse": float(np.sqrt(np.mean(erro ** 2))),
        "r2": float(1 - (ss_res / ss_tot)) if ss_tot else 0.0,
        "erro_max": float(np.max(np.abs(erro))),
    }


def estimar_observador_windkessel_fases(
    abp,
    picos_r,
    fs=125,
    r2_min=0.70,
):
    """
    Observador experimental por ciclo.

    Usa ABP apenas para calibracao/validacao dos parametros do ciclo.
    O ciclo continua delimitado por Ri -> Ri+1, usando o ECG como referencia.
    """

    segmentos = segmentar_abp_por_picos_r(
        abp,
        picos_r,
        fs=fs,
    )

    ciclos = []

    for segmento in segmentos:
        trecho = np.array(segmento["abp"], dtype=float)
        n = len(trecho)
        rr_s = float(segmento["rr_s"])

        base = {
            "ciclo": int(segmento["indice"]),
            "indice_R_inicial": int(segmento["inicio"]),
            "indice_R_final": int(segmento["fim"]),
            "RR_s": rr_s,
            "FC_bpm": float(60.0 / rr_s) if rr_s > 0 else None,
            "PAS_real": float(segmento["pas"]),
            "PAD_real": float(segmento["pad"]),
            "PAM_real": float(segmento["pam"]),
            "valido": False,
            "motivo_rejeicao": "",
        }

        if n < 20 or rr_s <= 0:
            ciclos.append({
                **base,
                "motivo_rejeicao": "numero insuficiente de amostras",
            })
            continue

        t = np.arange(n, dtype=float) / float(fs)
        p_min = float(np.min(trecho))
        p_max = float(np.max(trecho))
        p_inicio = float(trecho[0])
        p_fim = float(trecho[-1])

        delta_hi = min(0.20, max(0.031, rr_s * 0.35))
        ts_hi = min(rr_s * 0.55, rr_s - (1.0 / float(fs)))
        ts_lo = min(max(0.08, 0.04), ts_hi - 0.01)

        if ts_hi <= 0.06 or delta_hi <= 0.03:
            ciclos.append({
                **base,
                "motivo_rejeicao": "RR curto para limites fisiologicos",
            })
            continue

        delta0 = min(max(0.08, 0.03), delta_hi)
        idx_pico_t = float(segmento["idx_pico_rel"] / float(fs))
        ts0 = min(max(idx_pico_t, delta0 + 0.04, ts_lo), ts_hi)
        tau_s0 = 0.12
        tau_d0 = min(max(rr_s * 0.45, 0.10), 1.2)
        k0 = p_max + max(5.0, p_max - p_min)
        p0_0 = p_inicio
        p_inf0 = max(0.0, min(120.0, p_fim))

        lower = np.array([
            0.03,
            ts_lo,
            0.03,
            max(1.0, p_min - 20.0),
            max(0.0, p_min - 30.0),
            0.05,
            0.0,
        ])
        upper = np.array([
            delta_hi,
            ts_hi,
            1.0,
            p_max + 120.0,
            p_max + 30.0,
            3.0,
            120.0,
        ])
        x0 = np.array([
            delta0,
            ts0,
            tau_s0,
            k0,
            p0_0,
            tau_d0,
            p_inf0,
        ])
        x0 = np.minimum(np.maximum(x0, lower + 1e-6), upper - 1e-6)

        def residuo(params):
            delta_t_em, t_s, tau_s, k, p0, tau_d, p_inf = params

            if (
                tau_s <= 0
                or tau_d <= 0
                or t_s <= delta_t_em
                or t_s >= rr_s
            ):
                return np.ones_like(trecho) * 1e6

            return (
                _modelo_windkessel_fases(
                    t,
                    delta_t_em,
                    t_s,
                    tau_s,
                    k,
                    p0,
                    tau_d,
                    p_inf,
                )
                - trecho
            )

        try:
            ajuste = least_squares(
                residuo,
                x0,
                bounds=(lower, upper),
                max_nfev=3000,
            )
        except Exception as exc:
            ciclos.append({
                **base,
                "motivo_rejeicao": f"falha do otimizador: {exc}",
            })
            continue

        if not ajuste.success:
            ciclos.append({
                **base,
                "motivo_rejeicao": "otimizador nao convergiu",
            })
            continue

        delta_t_em, t_s, tau_s, k, p0, tau_d, p_inf = [
            float(v) for v in ajuste.x
        ]
        p_modelo = _modelo_windkessel_fases(
            t,
            delta_t_em,
            t_s,
            tau_s,
            k,
            p0,
            tau_d,
            p_inf,
        )
        pas_est = float(
            p0 * np.exp(-(t_s - delta_t_em) / tau_s)
            + k * (1.0 - np.exp(-(t_s - delta_t_em) / tau_s))
        )
        pad_est = float(
            p_inf
            + (pas_est - p_inf) * np.exp(-(rr_s - t_s) / tau_d)
        )
        pam_formula = float(
            pad_est + ((pas_est - pad_est) / 3.0)
        )
        integral = (
            np.trapezoid(p_modelo, t)
            if hasattr(np, "trapezoid")
            else np.trapz(p_modelo, t)
        )
        pam_integral = float(integral / rr_s)
        erro = trecho - p_modelo
        rmse_curva = float(np.sqrt(np.mean(erro ** 2)))
        ss_res = float(np.sum(erro ** 2))
        ss_tot = float(np.sum((trecho - np.mean(trecho)) ** 2))
        r2_curva = float(1 - (ss_res / ss_tot)) if ss_tot else 0.0

        motivos = []
        if tau_s <= 0:
            motivos.append("tau_s <= 0")
        if tau_d <= 0:
            motivos.append("tau_d <= 0")
        if t_s <= delta_t_em:
            motivos.append("t_s <= delta_t_em")
        if t_s >= rr_s:
            motivos.append("t_s fora do ciclo")
        if pas_est <= pad_est:
            motivos.append("PAS estimada <= PAD estimada")
        if r2_curva < r2_min:
            motivos.append("R2 da curva baixo")

        margem = 1e-4
        nomes = [
            "delta_t_em",
            "t_s",
            "tau_s",
            "K",
            "P0",
            "tau_d",
            "P_inf",
        ]
        for nome, valor, lo, hi in zip(nomes, ajuste.x, lower, upper):
            if abs(float(valor) - float(lo)) < margem or abs(float(valor) - float(hi)) < margem:
                motivos.append(f"{nome} proximo do limite")

        valido = not motivos

        ciclos.append({
            **base,
            "delta_t_em": delta_t_em,
            "t_s": t_s,
            "tau_s": tau_s,
            "tau_d": tau_d,
            "K": k,
            "P0": p0,
            "P_inf": p_inf,
            "PAS_estimada": pas_est,
            "erro_PAS": float(segmento["pas"] - pas_est),
            "PAD_estimada": pad_est,
            "erro_PAD": float(segmento["pad"] - pad_est),
            "PAM_estimada_formula": pam_formula,
            "PAM_estimada_integral": pam_integral,
            "erro_PAM_formula": float(segmento["pam"] - pam_formula),
            "erro_PAM_integral": float(segmento["pam"] - pam_integral),
            "R2_curva": r2_curva,
            "RMSE_curva": rmse_curva,
            "valido": valido,
            "motivo_rejeicao": "; ".join(motivos) if motivos else "",
            "tempo": [float(v) for v in t],
            "abp_real": [float(v) for v in trecho],
            "p_modelo": [float(v) for v in p_modelo],
        })

    validos = [c for c in ciclos if c.get("valido")]
    metricas = {}

    if validos:
        metricas = {
            "pas": _metricas_serie(
                [c["PAS_real"] for c in validos],
                [c["PAS_estimada"] for c in validos],
            ),
            "pad": _metricas_serie(
                [c["PAD_real"] for c in validos],
                [c["PAD_estimada"] for c in validos],
            ),
            "pam_formula": _metricas_serie(
                [c["PAM_real"] for c in validos],
                [c["PAM_estimada_formula"] for c in validos],
            ),
            "pam_integral": _metricas_serie(
                [c["PAM_real"] for c in validos],
                [c["PAM_estimada_integral"] for c in validos],
            ),
        }

    return {
        "ciclos": ciclos,
        "metricas": metricas,
        "total_ciclos": len(ciclos),
        "validos": len(validos),
        "rejeitados": len(ciclos) - len(validos),
    }


def reconstruir_pam_por_windkessel_exponencial(
    pam_real,
    pam_modelo_bruta,
    split=None,
):
    """
    Calibra apenas um offset entre a media real do ciclo e a media da curva
    exponencial ajustada na diastole. Nao usa FC, HRV nem ML.
    """

    real = np.array(pam_real, dtype=float)
    bruto = np.array(pam_modelo_bruta, dtype=float)

    n = min(len(real), len(bruto))

    if n == 0:
        return {
            "predicoes": [],
            "offset": 0.0,
            "n_calibracao": 0,
        }

    real = real[:n]
    bruto = bruto[:n]
    limite = int(split) if split is not None else n
    limite = max(1, min(limite, n))
    offset = float(np.mean(real[:limite] - bruto[:limite]))
    predicoes = bruto + offset

    return {
        "predicoes": [float(v) for v in predicoes],
        "offset": offset,
        "n_calibracao": int(limite),
    }


