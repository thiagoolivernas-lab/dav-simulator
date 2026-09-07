import math

from django.test import SimpleTestCase

from .services.pi_dav import calcular_pi_dav, calcular_pi_dav_janela


class PiDavTests(SimpleTestCase):
    def test_calcula_pi_por_potencia(self):
        resultado = calcular_pi_dav([4.0, 5.0, 6.0])

        self.assertAlmostEqual(resultado["pi"], 0.4)
        self.assertEqual(resultado["potencia_max"], 6.0)
        self.assertEqual(resultado["potencia_min"], 4.0)
        self.assertEqual(resultado["potencia_media"], 5.0)
        self.assertEqual(resultado["n_amostras"], 3)

    def test_lista_vazia(self):
        resultado = calcular_pi_dav([])

        self.assertIsNone(resultado["pi"])
        self.assertEqual(resultado["n_amostras"], 0)

    def test_somente_nan(self):
        resultado = calcular_pi_dav([math.nan, float("nan")])

        self.assertIsNone(resultado["pi"])
        self.assertEqual(resultado["n_amostras"], 0)

    def test_potencia_media_zero(self):
        resultado = calcular_pi_dav([0.0, 0.0, 0.0])

        self.assertIsNone(resultado["pi"])
        self.assertEqual(resultado["potencia_media"], 0.0)

    def test_valores_negativos_invalidos(self):
        resultado = calcular_pi_dav([-1.0, 4.0, 6.0])

        self.assertAlmostEqual(resultado["pi"], 0.4)
        self.assertEqual(resultado["n_amostras"], 2)

    def test_janela_incompleta(self):
        resultado = calcular_pi_dav_janela([5.0], dt=1.0, janela_s=15.0)

        self.assertIsNone(resultado["pi"])
        self.assertEqual(resultado["n_amostras"], 1)
        self.assertEqual(resultado["amostras_janela"], 15)

    def test_serie_constante(self):
        resultado = calcular_pi_dav([5.0, 5.0, 5.0])

        self.assertEqual(resultado["pi"], 0.0)
