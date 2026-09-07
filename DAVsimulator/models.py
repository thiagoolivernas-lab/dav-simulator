from django.db import models


class MIMICRecord(models.Model):

    nome = models.CharField(max_length=100)

    record_id = models.CharField(
        max_length=50,
        blank=True,
        null=True
    )

    diretorio = models.CharField(
        max_length=300,
        blank=True,
        null=True
    )

    arquivo_zip = models.FileField(
        upload_to="dav/datasets/zip/",
        blank=True,
        null=True
    )

    arquivo_hea = models.FileField(
        upload_to="dav/datasets/hea/",
        blank=True,
        null=True
    )

    arquivo_dat = models.FileField(
        upload_to="dav/datasets/dat/",
        blank=True,
        null=True
    )

    arquivo_numerics_hea = models.FileField(
        upload_to="dav/datasets/numerics_hea/",
        blank=True,
        null=True
    )

    arquivo_numerics_dat = models.FileField(
        upload_to="dav/datasets/numerics_dat/",
        blank=True,
        null=True
    )

    frequencia_amostragem = models.FloatField(default=125)

    possui_ecg = models.BooleanField(default=True)
    possui_abp = models.BooleanField(default=True)
    possui_ppg = models.BooleanField(default=True)
    possui_resp = models.BooleanField(default=False)

    observacoes = models.TextField(
        blank=True,
        null=True
    )

    pasta_extraida = models.CharField(
        max_length=500,
        blank=True,
        null=True
    )



    criado_em = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.nome} ({self.record_id or 'sem record_id'})"
