# projeto/forms_base.py
from django import forms

def _bs_class(field: forms.Field) -> str:
    if isinstance(field.widget, forms.CheckboxInput):
        return "form-check-input"
    if isinstance(field.widget, (forms.Select, forms.SelectMultiple)):
        return "form-select"
    return "form-control"

class BootstrapModelForm(forms.ModelForm):
    """Aplica classes Bootstrap automaticamente em todos os campos."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for _, field in self.fields.items():
            css = _bs_class(field)
            field.widget.attrs["class"] = (
                (field.widget.attrs.get("class", "") + " " + css).strip()
            )

