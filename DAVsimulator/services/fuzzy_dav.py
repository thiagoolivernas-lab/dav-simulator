import numpy as np


# ============================================================
# FUZZY DAV — CAMADA DE DEMANDA FISIOLÓGICA
# ============================================================
# Entrada:
# FC, HRV, IMU, SpO2, Temperatura
#
# Saída:
# demanda fisiológica D entre 0 e 1
# classe clínica
# PAM alvo personalizada
# alerta fisiológico
#
# Observação:
# Esta camada NÃO controla diretamente RPM.
# Ela define a PAM alvo.
# O controle de RPM continua sendo feito pelo pipeline DAV.
# ============================================================


try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_DISPONIVEL = True
except Exception:
    fuzz = None
    ctrl = None
    SKFUZZY_DISPONIVEL = False


# ============================================================
# FUNÇÃO AUXILIAR — FALLBACK SEM SCIKIT-FUZZY
# ============================================================

def calcular_demanda_fallback(
    fc_val,
    hrv_val,
    imu_val,
    spo2_val,
    temp_val
):
    """
    Fallback simples caso scikit-fuzzy não esteja instalado.

    Mantém o sistema funcionando no Django mesmo sem a biblioteca fuzzy.
    """

    fc_val = float(np.clip(fc_val, 40, 180))
    hrv_val = float(np.clip(hrv_val, 0, 150))
    imu_val = float(np.clip(imu_val, 0, 1))
    spo2_val = float(np.clip(spo2_val, 80, 100))
    temp_val = float(np.clip(temp_val, 35, 40))

    demanda = 0.0

    # Movimento
    demanda += 0.35 * imu_val

    # FC
    if fc_val < 60:
        demanda += 0.05
    elif fc_val < 95:
        demanda += 0.15
    elif fc_val < 130:
        demanda += 0.35
    else:
        demanda += 0.60

    # HRV baixa sugere estresse fisiológico
    if hrv_val < 25:
        demanda += 0.20
    elif hrv_val < 60:
        demanda += 0.10

    # SpO2 baixa força alerta
    if spo2_val < 94:
        demanda += 0.50

    # Temperatura elevada
    if temp_val >= 38:
        demanda += 0.40
    elif temp_val >= 37.3:
        demanda += 0.20

    demanda = float(np.clip(demanda, 0, 1))

    return demanda


# ============================================================
# CONSTRUÇÃO DO SISTEMA FUZZY
# ============================================================

def construir_sistema_fuzzy():

    if not SKFUZZY_DISPONIVEL:
        return None

    fc = ctrl.Antecedent(
        np.arange(40, 181, 1),
        "fc"
    )

    hrv = ctrl.Antecedent(
        np.arange(0, 151, 1),
        "hrv"
    )

    imu = ctrl.Antecedent(
        np.arange(0, 1.01, 0.01),
        "imu"
    )

    spo2 = ctrl.Antecedent(
        np.arange(80, 101, 1),
        "spo2"
    )

    temp = ctrl.Antecedent(
        np.arange(35, 40.1, 0.1),
        "temp"
    )

    demanda = ctrl.Consequent(
        np.arange(0, 1.01, 0.01),
        "demanda"
    )

    # ========================================================
    # FUNÇÕES DE PERTINÊNCIA
    # ========================================================

    fc["baixa"] = fuzz.trapmf(
        fc.universe,
        [40, 40, 55, 65]
    )

    fc["normal"] = fuzz.trimf(
        fc.universe,
        [55, 75, 95]
    )

    fc["alta"] = fuzz.trimf(
        fc.universe,
        [85, 110, 130]
    )

    fc["muito_alta"] = fuzz.trapmf(
        fc.universe,
        [120, 135, 180, 180]
    )

    hrv["baixa"] = fuzz.trapmf(
        hrv.universe,
        [0, 0, 15, 25]
    )

    hrv["normal"] = fuzz.trimf(
        hrv.universe,
        [20, 40, 60]
    )

    hrv["alta"] = fuzz.trapmf(
        hrv.universe,
        [50, 70, 150, 150]
    )

    imu["baixa"] = fuzz.trapmf(
        imu.universe,
        [0, 0, 0.10, 0.25]
    )

    imu["moderada"] = fuzz.trimf(
        imu.universe,
        [0.15, 0.45, 0.70]
    )

    imu["alta"] = fuzz.trapmf(
        imu.universe,
        [0.6, 0.75, 1.0, 1.0]
    )

    spo2["baixa"] = fuzz.trapmf(
        spo2.universe,
        [80, 80, 90, 94]
    )

    spo2["normal"] = fuzz.trapmf(
        spo2.universe,
        [93, 96, 100, 100]
    )

    temp["normal"] = fuzz.trapmf(
        temp.universe,
        [35.0, 35.8, 36.8, 37.3]
    )

    temp["elevada"] = fuzz.trimf(
        temp.universe,
        [37.0, 37.6, 38.2]
    )

    temp["alta"] = fuzz.trapmf(
        temp.universe,
        [38.0, 38.5, 40.0, 40.0]
    )

    demanda["baixa"] = fuzz.trapmf(
        demanda.universe,
        [0, 0, 0.15, 0.30]
    )

    demanda["moderada"] = fuzz.trimf(
        demanda.universe,
        [0.25, 0.45, 0.65]
    )

    demanda["alta"] = fuzz.trimf(
        demanda.universe,
        [0.55, 0.75, 0.90]
    )

    demanda["critica"] = fuzz.trapmf(
        demanda.universe,
        [0.85, 0.95, 1.0, 1.0]
    )

    # ========================================================
    # REGRAS FUZZY
    # ========================================================

    rules = []

    # Repouso fisiológico
    rules.append(
        ctrl.Rule(
            imu["baixa"]
            & fc["normal"]
            & hrv["alta"]
            & spo2["normal"]
            & temp["normal"],
            demanda["baixa"]
        )
    )

    rules.append(
        ctrl.Rule(
            imu["baixa"]
            & fc["baixa"]
            & spo2["normal"]
            & temp["normal"],
            demanda["baixa"]
        )
    )

    # Movimento sem grande resposta cardíaca
    rules.append(
        ctrl.Rule(
            imu["moderada"]
            & fc["normal"]
            & spo2["normal"],
            demanda["moderada"]
        )
    )

    rules.append(
        ctrl.Rule(
            imu["alta"]
            & fc["normal"]
            & hrv["normal"]
            & spo2["normal"],
            demanda["moderada"]
        )
    )

    # Atividade com resposta fisiológica
    rules.append(
        ctrl.Rule(
            imu["alta"]
            & fc["alta"]
            & hrv["baixa"]
            & spo2["normal"],
            demanda["alta"]
        )
    )

    rules.append(
        ctrl.Rule(
            imu["alta"]
            & fc["alta"]
            & hrv["normal"]
            & spo2["normal"],
            demanda["moderada"]
        )
    )

    rules.append(
        ctrl.Rule(
            imu["alta"]
            & fc["muito_alta"]
            & spo2["normal"],
            demanda["alta"]
        )
    )

    # FC alta sem movimento compatível: SISC
    rules.append(
        ctrl.Rule(
            imu["baixa"]
            & fc["alta"]
            & hrv["baixa"],
            demanda["critica"]
        )
    )

    rules.append(
        ctrl.Rule(
            imu["baixa"]
            & fc["muito_alta"],
            demanda["critica"]
        )
    )

    # Hipoxemia: SISC
    rules.append(
        ctrl.Rule(
            spo2["baixa"],
            demanda["critica"]
        )
    )

    # Febre/estresse: SISC
    rules.append(
        ctrl.Rule(
            temp["alta"]
            & fc["alta"],
            demanda["critica"]
        )
    )

    rules.append(
        ctrl.Rule(
            temp["alta"]
            & fc["muito_alta"],
            demanda["critica"]
        )
    )

    # Cobertura mínima
    rules.append(
        ctrl.Rule(
            imu["baixa"]
            & spo2["normal"]
            & temp["normal"],
            demanda["baixa"]
        )
    )

    rules.append(
        ctrl.Rule(
            imu["moderada"]
            & spo2["normal"],
            demanda["moderada"]
        )
    )

    rules.append(
        ctrl.Rule(
            fc["muito_alta"],
            demanda["critica"]
        ) 
    )

    return ctrl.ControlSystem(rules)



# Sistema fuzzy global
DEMANDA_CTRL = construir_sistema_fuzzy()


# ============================================================
# CLASSIFICAÇÃO DA DEMANDA
# ============================================================

def classificar_demanda(D):

    if D < 0.30:
        return "repouso"

    if D < 0.60:
        return "atividade leve/moderada"

    if D < 0.85:
        return "atividade intensa"

    return "alerta fisiológico"



def avaliar_fatores_clinicos(
    fc_val,
    hrv_val,
    imu_val,
    spo2_val,
    temp_val,
):
    fatores = []
    alerta = False

    if spo2_val < 92:
        fatores.append("Hipoxemia importante")
        alerta = True
    elif spo2_val < 94:
        fatores.append("SpO2 limiar")

    if fc_val >= 135:
        fatores.append("FC muito alta")
        alerta = True
    elif fc_val >= 110 and imu_val < 0.25:
        fatores.append("FC alta sem movimento proporcional")
        alerta = True
    elif fc_val >= 100:
        fatores.append("FC elevada")

    if hrv_val < 20:
        fatores.append("HRV muito baixa")
        if fc_val >= 95:
            alerta = True
    elif hrv_val < 35:
        fatores.append("HRV baixa")

    if temp_val >= 38.0:
        fatores.append("Temperatura alta")
        if fc_val >= 95:
            alerta = True
    elif temp_val >= 37.3:
        fatores.append("Temperatura elevada")

    if imu_val >= 0.7:
        fatores.append("Atividade intensa")
    elif imu_val >= 0.3:
        fatores.append("Atividade moderada")

    return fatores, alerta


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def calcular_demanda_fuzzy(
    fc_val,
    hrv_val,
    imu_val,
    spo2_val,
    temp_val,
    PAM_base,
    delta_PAM=15,
):
    """
    Calcula a demanda fisiológica do paciente.

    Esta função é compatível com a arquitetura atual:

    PAM_base vem da calibração pré-implante.
    delta_PAM define quanto a PAM alvo pode subir.
    """

    fc_val = float(np.clip(fc_val, 40, 180))
    hrv_val = float(np.clip(hrv_val, 0, 150))
    imu_val = float(np.clip(imu_val, 0, 1))
    spo2_val = float(np.clip(spo2_val, 80, 100))
    temp_val = float(np.clip(temp_val, 35, 40))

    fatores, alerta_clinico = avaliar_fatores_clinicos(
        fc_val,
        hrv_val,
        imu_val,
        spo2_val,
        temp_val,
    )

    if SKFUZZY_DISPONIVEL and DEMANDA_CTRL is not None:

        sim = ctrl.ControlSystemSimulation(
            DEMANDA_CTRL
        )

        sim.input["fc"] = fc_val

        sim.input["hrv"] = hrv_val

        sim.input["imu"] = imu_val

        sim.input["spo2"] = spo2_val

        sim.input["temp"] = temp_val

        sim.compute()

        D = float(
            sim.output.get(
                "demanda",
                0.0
            )
        )

        metodo = "scikit-fuzzy"

    else:

        D = calcular_demanda_fallback(
            fc_val,
            hrv_val,
            imu_val,
            spo2_val,
            temp_val
        )

        metodo = "fallback"

    if alerta_clinico:
        D = max(D, 0.88)

    D = float(
        np.clip(D, 0, 1)
    )

    classe = classificar_demanda(D)

    alerta = alerta_clinico or classe == "alerta fisiológico"

    PAM_alvo = (
        float(PAM_base)
        + float(delta_PAM) * D
    )

    return {
        "demanda": D,
        "classe": classe,
        "PAM_alvo": float(PAM_alvo),
        "alerta": alerta,
        "fatores": fatores,
        "metodo": metodo,
        "entradas": {
            "fc": float(fc_val),
            "hrv": float(hrv_val),
            "imu": float(imu_val),
            "spo2": float(spo2_val),
            "temp": float(temp_val),
            "PAM_base": float(PAM_base),
            "delta_PAM": float(delta_PAM),
        }
    }
