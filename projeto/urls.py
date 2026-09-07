from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("logout/", views.logout_view, name="logout"),

    # rotas provisórias para o template não quebrar
    path("projetos/", views.placeholder, name="projeto_list"),
]