def controlar_rpm(
    pam_estimada,
    pam_alvo,
    rpm_atual,
    kp=20,
    rpm_min=3000,
    rpm_max=9000,
    pam_min=65,
    pam_max=95
):
    erro = pam_alvo - pam_estimada

    rpm_sugerido = rpm_atual + kp * erro

    if pam_estimada > pam_max:
        rpm_sugerido = min(rpm_sugerido, rpm_atual)

    elif pam_estimada < pam_min:
        rpm_sugerido = max(rpm_sugerido, rpm_atual)

    rpm_sugerido = max(
        rpm_min,
        min(rpm_sugerido, rpm_max)
    )

    return {
        "erro": float(erro),
        "rpm_atual": float(rpm_atual),
        "rpm_sugerido": float(rpm_sugerido),
        "pam_estimada": float(pam_estimada),
        "pam_alvo": float(pam_alvo),
    }