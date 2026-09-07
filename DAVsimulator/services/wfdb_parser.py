import zipfile


def listar_arquivos_zip(caminho_zip):

    with zipfile.ZipFile(caminho_zip, "r") as zip_ref:

        return zip_ref.namelist()


def encontrar_headers(arquivos):

    headers = []

    for arquivo in arquivos:

        if arquivo.endswith(".hea"):

            headers.append(arquivo)

    return headers


def ler_header_principal(caminho_zip, header_name):

    with zipfile.ZipFile(caminho_zip, "r") as zip_ref:

        with zip_ref.open(header_name) as arquivo:

            conteudo = arquivo.read().decode("utf-8")

    return conteudo


def detectar_sinais(header_texto):

    sinais = {
        "ecg": False,
        "abp": False,
        "ppg": False,
        "resp": False,
    }

    texto = header_texto.upper()

    sinais_ecg = [
        "ECG",
        "II",
        "III",
        "V",
        "AVL",
        "AVR",
        "AVF",
    ]

    sinais_abp = [
        "ABP",
        "ART",
    ]

    sinais_ppg = [
        "PLETH",
        "PPG",
    ]

    sinais_resp = [
        "RESP",
        "RESPIRATION",
    ]

    for s in sinais_ecg:

        if s in texto:
            sinais["ecg"] = True

    for s in sinais_abp:

        if s in texto:
            sinais["abp"] = True

    for s in sinais_ppg:

        if s in texto:
            sinais["ppg"] = True

    for s in sinais_resp:

        if s in texto:
            sinais["resp"] = True

    return sinais

def extrair_info_headers(caminho_zip, headers):

    info = {
        "headers": [],
        "sinais": [],
        "fs": None,
    }

    for header in headers:

        texto = ler_header_principal(
            caminho_zip,
            header
        )

        linhas = texto.splitlines()

        if not linhas:
            continue

        primeira_linha = linhas[0].split()

        if len(primeira_linha) >= 3:

            try:
                fs = float(primeira_linha[2])

                if fs > 0:
                    info["fs"] = fs

            except ValueError:
                pass

        for linha in linhas[1:]:

            partes = linha.split()

            if len(partes) >= 9:

                sinal = partes[-1]

                if sinal not in info["sinais"]:
                    info["sinais"].append(sinal)

        info["headers"].append(header)

    return info

import os


def extrair_zip_dataset(caminho_zip, destino_base, record_id):

    pasta_destino = os.path.join(
        destino_base,
        str(record_id)
    )

    os.makedirs(
        pasta_destino,
        exist_ok=True
    )

    with zipfile.ZipFile(caminho_zip, "r") as zip_ref:

        zip_ref.extractall(
            pasta_destino
        )

    return pasta_destino

def listar_segmentos_extraidos(pasta_extraida):

    import os

    segmentos = []

    for raiz, dirs, arquivos in os.walk(pasta_extraida):

        for arquivo in arquivos:

            if arquivo.endswith(".dat"):

                nome_base = arquivo.replace(".dat", "")

                # Ignora numerics por enquanto: 3000003n.dat
                if nome_base.endswith("n"):
                    continue

                caminho_dat = os.path.join(
                    raiz,
                    arquivo
                )

                caminho_hea = caminho_dat.replace(".dat", ".hea")

                if not os.path.exists(caminho_hea):
                    continue

                segmentos.append({
                    "nome": nome_base,
                    "arquivo_dat": caminho_dat,
                })

    return segmentos


import numpy as np


def ler_waveform_bruto(caminho_dat):

    with open(caminho_dat, "rb") as f:

        dados = np.fromfile(
            f,
            dtype=np.int16
        )

    return dados[:5000]

def encontrar_header_do_segmento(caminho_dat):

    caminho_hea = caminho_dat.replace(".dat", ".hea")

    return caminho_hea


def ler_info_segmento(caminho_hea):

    with open(caminho_hea, "r", encoding="utf-8", errors="ignore") as f:
        linhas = f.read().splitlines()

    primeira = linhas[0].split()

    nome = primeira[0]
    n_canais = int(primeira[1])
    fs = float(primeira[2])

    sinais = []
    canais_info = []

    for linha in linhas[1:]:

        partes = linha.split()

        if len(partes) >= 2:

            nome_sinal = partes[-1]

            ganho = 1
            baseline = 0
            unidade = ""

            try:
                ganho_unidade = partes[2]

                if "/" in ganho_unidade:
                    ganho_txt, unidade = ganho_unidade.split("/")
                    ganho = float(ganho_txt)
                else:
                    ganho = float(ganho_unidade)

            except Exception:
                pass

            try:
                baseline = int(partes[4])
            except Exception:
                baseline = 0

            sinais.append(nome_sinal)

            canais_info.append({
                "nome": nome_sinal,
                "ganho": ganho,
                "baseline": baseline,
                "unidade": unidade,
            })

    return {
        "nome": nome,
        "n_canais": n_canais,
        "fs": fs,
        "sinais": sinais,
        "canais_info": canais_info,
    }

def ler_dat_80(caminho_dat):

    import numpy as np

    with open(caminho_dat, "rb") as f:

        raw = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

    return raw.astype(np.int16)

def ler_waveform_canais(caminho_dat, max_amostras=5000, inicio=0):

    caminho_hea = encontrar_header_do_segmento(caminho_dat)

    info = ler_info_segmento(caminho_hea)

    n_canais = info["n_canais"]

    dados = ler_dat_80(caminho_dat)

    total_linhas = len(dados) // n_canais

    dados = dados[:total_linhas * n_canais]

    matriz = dados.reshape(
        total_linhas,
        n_canais
    )

    fim = inicio + max_amostras

    matriz = matriz[inicio:fim, :]

    canais = {}

    for i, canal_info in enumerate(info["canais_info"]):

        nome_sinal = canal_info["nome"]
        ganho = canal_info["ganho"]
        baseline = canal_info["baseline"]

        canal_bruto = matriz[:, i]

        if nome_sinal == "ABP":

            canal_fisico = canal_bruto / ganho

        else:

            canal_fisico = (canal_bruto - 128) / ganho

        canais[nome_sinal] = canal_fisico.tolist()

    return {
        "info": info,
        "canais": canais,
    }


def ler_dat_212(caminho_dat):

    import numpy as np

    with open(caminho_dat, "rb") as f:
        raw = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

    n_blocos = len(raw) // 3

    raw = raw[:n_blocos * 3]

    raw = raw.reshape(
        n_blocos,
        3
    )

    b0 = raw[:, 0].astype(np.int16)
    b1 = raw[:, 1].astype(np.int16)
    b2 = raw[:, 2].astype(np.int16)

    s1 = ((b1 & 0x0F) << 8) | b0
    s2 = (b2 << 4) | ((b1 & 0xF0) >> 4)

    s1 = np.where(
        s1 >= 2048,
        s1 - 4096,
        s1
    )

    s2 = np.where(
        s2 >= 2048,
        s2 - 4096,
        s2
    )

    dados = np.empty(
        n_blocos * 2,
        dtype=np.int16
    )

    dados[0::2] = s1
    dados[1::2] = s2

    return dados

def detectar_picos_r_simples(ecg, fs=125):

    import numpy as np

    ecg_array = np.array(ecg)

    picos = []

    distancia_minima = int(0.35 * fs)

    for i in range(1, len(ecg_array) - 1):

        if (
            ecg_array[i] > ecg_array[i - 1]
            and ecg_array[i] > ecg_array[i + 1]
            and ecg_array[i] > 10
        ):

            if not picos or (i - picos[-1]) > distancia_minima:

                picos.append(i)

    if len(picos) >= 2:

        rr = np.diff(picos) / fs

        fc_media = 60 / np.mean(rr)

        hrv_rmssd = np.sqrt(
            np.mean(
                np.diff(rr) ** 2
            )
        ) * 1000

    else:

        fc_media = None
        hrv_rmssd = None

    return {
        "picos": picos,
        "fc_media": round(fc_media, 2) if fc_media else None,
        "hrv_rmssd": round(hrv_rmssd, 2) if hrv_rmssd else None,
    }
