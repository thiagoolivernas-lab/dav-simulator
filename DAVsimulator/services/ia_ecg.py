from pathlib import Path
import numpy as np
import tensorflow as tf
from keras.saving import register_keras_serializable
from scipy.signal import find_peaks


@register_keras_serializable()
def _crop(tensors):
    """
    Função usada no modelo treinado para ajustar/cortar tensores.
    Precisa existir aqui porque o modelo .keras foi salvo com essa função.
    """
    x, ref = tensors
    ref_len = tf.shape(ref)[1]
    return x[:, :ref_len, :]


BASE_DIR = Path(__file__).resolve().parent.parent

MODELO_PATH = BASE_DIR / "modelos" / "cnn_ecg_treinado.keras"

modelo_ecg = tf.keras.models.load_model(
    MODELO_PATH,
    custom_objects={"_crop": _crop},
    compile=False,
)


def detectar_picos_r_ia(ecg):

    ecg = np.array(ecg, dtype=np.float32)

    if ecg.shape[0] != 5000:
        raise ValueError(
            f"O modelo espera 5000 amostras, recebeu {ecg.shape[0]}"
        )

    # normalização
    ecg = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)

    # reshape
    entrada = ecg.reshape(1, 5000, 1)

    # inferência
    prob = modelo_ecg.predict(entrada, verbose=0)[0]

    # remove dimensão extra
    prob = np.squeeze(prob)

    return prob

def detectar_indices_picos_r_ia(ecg, fs=125, threshold=0.8):

    from scipy.signal import find_peaks

    ecg = np.array(ecg)

    prob = detectar_picos_r_ia(ecg)

    # suavização da probabilidade
    kernel = np.ones(15) / 15

    prob_suave = np.convolve(
        prob,
        kernel,
        mode="same"
    )

    distancia_minima = int(0.4 * fs)

    # ativações da CNN
    ativacoes, props = find_peaks(
        prob_suave,
        height=threshold,
        distance=distancia_minima
    )

    picos_r_refinados = []

    janelas_qrs = []

    janela_antes = int(0.02 * fs)
    janela_depois = int(0.22 * fs)

    for p in ativacoes:

        inicio = max(0, p - janela_antes)
        fim = min(len(ecg), p + janela_depois)

        trecho = ecg[inicio:fim]

        if len(trecho) == 0:
            continue

        pico_local = np.argmax(trecho)
        pico_real = inicio + pico_local

        picos_r_refinados.append(
            int(pico_real)
        )

        largura_qrs = int(0.12 * fs)

        inicio_qrs = max(0, pico_real - largura_qrs // 2)
        fim_qrs = min(len(ecg), pico_real + largura_qrs // 2)

        janelas_qrs.append({
            "inicio": int(inicio_qrs),
            "fim": int(fim_qrs),
        })


    # FOR ACABA AQUI
    # A correção vem depois do for

    picos_r_refinados = corrigir_picos_perdidos(
        ecg,
        picos_r_refinados,
        fs=fs
    )

    return (
        np.array(picos_r_refinados),
        prob_suave,
        janelas_qrs
    )

def corrigir_picos_perdidos(
    ecg,
    picos_r,
    fs=125
):

    picos_r = sorted(list(picos_r))

    if len(picos_r) < 3:
        return np.array(picos_r)

    rr = np.diff(picos_r)

    rr_medio = np.median(rr)

    novos_picos = []

    for i in range(len(picos_r) - 1):

        p1 = picos_r[i]
        p2 = picos_r[i + 1]

        novos_picos.append(p1)

        intervalo = p2 - p1

        # intervalo muito grande
        if intervalo > 1.6 * rr_medio:

            inicio_busca = p1 + int(0.2 * fs)
            fim_busca = p2 - int(0.2 * fs)

            if fim_busca > inicio_busca:

                trecho = ecg[inicio_busca:fim_busca]

                if len(trecho) > 0:

                    pico_local = np.argmax(trecho)

                    pico_corrigido = (
                        inicio_busca + pico_local
                    )

                    novos_picos.append(
                        int(pico_corrigido)
                    )

    novos_picos.append(picos_r[-1])

    novos_picos = sorted(
        list(set(novos_picos))
    )

    return np.array(novos_picos)

def calcular_fc_hrv(picos_r, fs=125):

    if len(picos_r) < 2:

        return {
            "fc": 0,
            "rr_ms": [],
            "rmssd": 0,
        }

    rr = np.diff(picos_r)

    rr_ms = (rr / fs) * 1000

    fc = 60 / np.mean(rr / fs)

    if len(rr_ms) > 1:

        diff_rr = np.diff(rr_ms)

        rmssd = np.sqrt(
            np.mean(diff_rr ** 2)
        )

    else:

        rmssd = 0

    return {
        "fc": round(fc, 2),
        "rr_ms": rr_ms.tolist(),
        "rmssd": round(float(rmssd), 2),
    }