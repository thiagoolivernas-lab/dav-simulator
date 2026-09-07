from decimal import Decimal, InvalidOperation
from django import template

register = template.Library()


def _to_decimal(value):
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


@register.filter
def br_money(value, decimals=2):
    """Formata Decimal como moeda pt-BR: 10100836.99 -> 'R$ 10.100.836,99' """
    d = _to_decimal(value)
    if d is None:
        return "R$ 0,00"
    decimals = int(decimals)
    q = d.quantize(Decimal("1." + ("0" * decimals)))
    s = f"{q:,.{decimals}f}"  # 10,100,836.99
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")  # 10.100.836,99
    return f"R$ {s}"


@register.filter
def br_number(value, decimals=2):
    """Formata número pt-BR sem símbolo: 1234.5 -> '1.234,50' """
    d = _to_decimal(value)
    if d is None:
        return "-"
    decimals = int(decimals)
    q = d.quantize(Decimal("1." + ("0" * decimals)))
    s = f"{q:,.{decimals}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


@register.filter
def br_percent(value, decimals=2):
    """
    Formata percentual.
    Se vier 0.1312 -> 13.12% (multiplica por 100)
    Se vier 13.12 -> 13.12% (não multiplica)
    """
    d = _to_decimal(value)
    if d is None:
        return "- %"

    # Heurística: se for <= 1, assume fração (0-1) e multiplica por 100
    if abs(d) <= 1:
        d = d * Decimal("100")

    decimals = int(decimals)
    q = d.quantize(Decimal("1." + ("0" * decimals)))
    s = f"{q:,.{decimals}f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s} %"
