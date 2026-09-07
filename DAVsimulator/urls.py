from django.urls import path
from . import views

app_name = "DAVsimulator"

urlpatterns = [
    path("", views.home, name="home"),

    path(
        "sintonia/",
        views.sintonia_hemodinamica,
        name="sintonia"
    ),

     path(
         "datasets/",
          views.dataset_manager,
          name="datasets"
     ),

     path("datasets/upload/",
           views.dataset_upload,
           name="dataset_upload"
     ),

     path("datasets/<int:pk>/",
           views.dataset_detail,
           name="dataset_detail"
     ),

     path("datasets/<int:pk>/sinais/",
           views.dataset_sinais,
           name="dataset_sinais"
     ),

     path(
          "fuzzy/",
          views.fuzzy,
          name="fuzzy"
     ),

     path(
          "validacao-calibracao/",
           views.validacao_calibracao,
           name="validacao_calibracao"

      ),

      path(
            "validacao-controle/",
            views.validacao_controle,
            name="validacao_controle"
            ),

      path(
            "validacao-sistema-controle/",
            views.validacao_sistema_controle,
            name="validacao_sistema_controle"
            ),

      path(
            "validacao-windkessel-dav/",
            views.validacao_windkessel_dav,
            name="validacao_windkessel_dav"
            ),

]
