from django import forms


class SimulacaoDAVForm(forms.Form):
    fc = forms.FloatField(label="Frequência cardíaca (bpm)", initial=80)
    hrv = forms.FloatField(label="HRV / RMSSD (ms)", initial=35)
    spo2 = forms.FloatField(label="SpO₂ (%)", initial=97)
    temperatura = forms.FloatField(label="Temperatura corporal (°C)", initial=36.7)
    atividade = forms.FloatField(label="Nível de atividade física", initial=0.3)

    pam_basal = forms.FloatField(label="PAM basal (mmHg)", initial=85)
    resistencia = forms.FloatField(
        label="R_eff simplificada",
        initial=18
    )

    tau_base = forms.FloatField(
        label="Tau vascular agregado (s)",
        initial=0.8,
        min_value=0.05
    )

    vazao_basal = forms.FloatField(label="Vazão basal Q (L/min)", initial=5)
    rpm_inicial = forms.FloatField(label="RPM inicial da bomba", initial=5200)
    kp = forms.FloatField(label="Ganho proporcional Kp", initial=20)

    duracao = forms.IntegerField(label="Duração da simulação (s)", initial=60)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for field in self.fields.values():
            field.widget.attrs.update({"class": "form-control"})

class SintoniaHemodinamicaForm(forms.Form):
    abp = forms.CharField(
        label="Sinal ABP",
        widget=forms.Textarea(
            attrs={
                "class": "form-control",
                "rows": 8,
                "placeholder": "Cole valores de ABP separados por vírgula, espaço ou quebra de linha. Ex: 82, 85, 90, 96..."
            }
        )
    )

    vazao_q = forms.FloatField(
        label="Q_base assumido/medido (L/min)",
        initial=5,
        widget=forms.NumberInput(attrs={"class": "form-control"})
    )

from .models import MIMICRecord


class MIMICRecordUploadForm(forms.ModelForm):

    class Meta:
        model = MIMICRecord

        fields = [
            "nome",
            "record_id",
            "arquivo_zip",
            "frequencia_amostragem",            
        ]

        widgets = {
            "nome": forms.TextInput(attrs={"class": "form-control"}),
            "record_id": forms.TextInput(attrs={"class": "form-control"}),
            "arquivo_zip": forms.ClearableFileInput(attrs={"class": "form-control"}),
            "frequencia_amostragem": forms.NumberInput(attrs={"class": "form-control"}),
            "observacoes": forms.Textarea(attrs={"class": "form-control", "rows": 3}),
        }

