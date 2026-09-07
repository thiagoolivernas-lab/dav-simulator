from django import template

register = template.Library()

# -------------------------
# Badges: Probabilidade
# -------------------------
@register.filter
def risk_prob_badge(value: str) -> str:
    return {
        "alta": "text-bg-danger",
        "media": "text-bg-warning",
        "baixa": "text-bg-success",
    }.get(value, "text-bg-secondary")


# -------------------------
# Badges: Impacto
# -------------------------
@register.filter
def risk_impact_badge(value: str) -> str:
    return {
        "alto": "text-bg-danger",
        "medio": "text-bg-warning",
        "baixo": "text-bg-success",
    }.get(value, "text-bg-secondary")


# -------------------------
# Badges: Status do risco
# -------------------------
@register.filter
def risk_status_badge(value: str) -> str:
    return {
        "identificado": "text-bg-secondary",
        "avaliado": "text-bg-info",
        "tratamento": "text-bg-warning",
        "mitigado": "text-bg-primary",
        "aceito": "text-bg-success",
        "ocorrido": "text-bg-danger",
        "encerrado": "text-bg-dark",
    }.get(value, "text-bg-secondary")


# -------------------------
# Cor da barra da categoria (choice key)
# -------------------------
@register.filter
def risk_cat_color(cat_key: str) -> str:
    return {
        "escopo": "#ef4444",
        "cronograma": "#f59e0b",
        "financeiro": "#22c55e",
        "operacional": "#0ea5e9",
        "estrategico": "#8b5cf6",
        "comunicacao": "#ec4899",
        "externo": "#64748b",
        "contratual": "#14b8a6",
        "qualidade": "#06b6d4",
        "seguranca": "#fb7185",
        "tecnico": "#3b82f6",
        "tecnologia": "#a855f7",
        "recursos": "#84cc16",
        "interfaces": "#f97316",
        "": "#9ca3af",  # sem categoria
        None: "#9ca3af",
    }.get(cat_key, "#9ca3af")


# -------------------------
# Mantive seu mit_badge (se ainda usar)
# -------------------------
@register.filter
def mit_badge(text: str) -> str:
    if not text or not str(text).strip():
        return "text-bg-warning"
    return "text-bg-success"

