import numpy as np


def calibrar_paciente_pre_implante(
    tau_medio,
    tau_std,
    pam_media,
    fc_media=None,
    hrv_rmssd=None,
    fluxo_estimado=None
):
    """
    Calibração inicial do paciente usando dados pré-implante.

    Entrada:
    - tau_medio: constante de tempo arterial estimada da ABP
    - tau_std: variabilidade da tau
    - pam_media: PAM real extraída da ABP invasiva
    - fc_media: frequência cardíaca média
    - hrv_rmssd: variabilidade RR
    - fluxo_estimado: opcional. Quando o fluxo nao vem de medicao real,
      deve ser interpretado como hipotese de simulacao para definir Q_base.

    Saída:
    - assinatura hemodinâmica do paciente
    """

    resultado = {
        "tau_base": float(tau_medio) if tau_medio is not None else None,
        "tau_std": float(tau_std) if tau_std is not None else None,
        "pam_base": float(pam_media) if pam_media is not None else None,
        "fc_base": float(fc_media) if fc_media is not None else None,
        "hrv_base": float(hrv_rmssd) if hrv_rmssd is not None else None,
        "fluxo_base": float(fluxo_estimado) if fluxo_estimado is not None else None,
        "r_eff": None,
    }

    if (
        fluxo_estimado is not None
        and fluxo_estimado > 0
        and pam_media is not None
    ):
        resultado["r_eff"] = float(pam_media / fluxo_estimado)

    return resultado


def resumir_calibracao_por_janela(
    taus,
    pam_batimentos,
    fc_media=None,
    hrv_rmssd=None
):
    """
    Resume uma janela do dataset em parâmetros médios.
    """

    if not taus or not pam_batimentos:
        return None

    return {
        "tau_medio": float(np.median(taus)),
        "tau_std": float(np.std(taus)),
        "pam_media": float(np.mean(pam_batimentos)),
        "pam_mediana": float(np.median(pam_batimentos)),
        "fc_media": float(fc_media) if fc_media is not None else None,
        "hrv_rmssd": float(hrv_rmssd) if hrv_rmssd is not None else None,
        "n_batimentos": len(taus),
    }

def calcular_r_eff_dinamica(
    r_eff_base,
    fc,
    hrv,
    imu,
    spo2,
    temp
):
    """
    Modula heuristicamente a resistência vascular efetiva para simulação.

    Esta função nao estima RVS validada a partir de FC/HRV. Ela apenas
    perturba o parametro basal para ensaiar diferentes estados fisiologicos.

    Ideia simulada:
    - exercício / atividade: resistência tende a cair
    - hipóxia / alerta: resistência tende a subir
    """

    r_eff = float(r_eff_base)

    fc = float(fc)
    hrv = float(hrv)
    imu = float(imu)
    spo2 = float(spo2)
    temp = float(temp)

    # Atividade física: vasodilatação periférica
    if imu >= 0.5 and fc >= 100:
        r_eff *= 0.85

    if imu >= 0.8 and fc >= 120:
        r_eff *= 0.75

    # HRV baixa sugere estresse autonômico
    if hrv < 30:
        r_eff *= 1.10

    # Hipóxia: resposta de alerta
    if spo2 < 92:
        r_eff *= 1.20

    if spo2 < 88:
        r_eff *= 1.35

    # Febre/temperatura elevada
    if temp >= 38:
        r_eff *= 1.10

    # limites de segurança
    r_eff = max(
        10,
        min(r_eff, 45)
    )

    return float(r_eff)
