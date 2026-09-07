from django.shortcuts import render, redirect
from django.contrib.auth import logout


def dashboard(request):
    return render(request, "projeto/dashboard.html")


def logout_view(request):
    logout(request)
    return redirect("dashboard")


def placeholder(request):
    return render(request, "projeto/dashboard.html")