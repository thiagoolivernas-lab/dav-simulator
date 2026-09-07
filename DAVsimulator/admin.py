from django.contrib import admin
from .models import MIMICRecord


@admin.register(MIMICRecord)
class MIMICRecordAdmin(admin.ModelAdmin):

    list_display = (
        "nome",
        "record_id",
        "frequencia_amostragem",
        "possui_ecg",
        "possui_abp",
        "possui_ppg",
    )

    search_fields = (
        "nome",
        "record_id",
    )