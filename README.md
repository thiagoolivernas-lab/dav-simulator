# DAV Simulator

Sistema acadêmico em Python/Django para simulação de dispositivo de assistência ventricular (DAV), análise de sinais fisiológicos e avaliação de controle hemodinâmico.

O projeto reúne detecção de picos R no ECG com uma CNN, análise de ABP, estimativa de parâmetros hemodinâmicos, demanda fisiológica por lógica fuzzy, curvas de bomba e simulação de controle de RPM com modelo cardiovascular Windkessel. É um protótipo de pesquisa, sem validação para uso clínico.

## Requisitos

- Python **3.13** (versão utilizada no ambiente original: 3.13.2).
- Git e Git LFS.
- Espaço para o ambiente TensorFlow, o modelo treinado (~305 MB) e os sinais importados.

As versões das dependências diretas do ambiente original estão em `requirements.txt`. O notebook de treinamento é material complementar; o arquivo de dependências destina-se à aplicação e aos scripts de resultados, não à reprodução integral do treinamento.

## Instalação no Windows (PowerShell)

Substitua a URL abaixo pela URL deste repositório:

```powershell
git lfs install
git clone <URL_DO_REPOSITORIO>
cd <PASTA_DO_REPOSITORIO>
git lfs pull
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe manage.py migrate
.\.venv\Scripts\python.exe manage.py runserver
```

Abra http://127.0.0.1:8000/dav/ no navegador. O painel geral fica em http://127.0.0.1:8000/.

No Linux/macOS, crie o ambiente com `python3.13 -m venv .venv`, use `.venv/bin/python` no lugar de `.\.venv\Scripts\python.exe` e copie a configuração com `cp .env.example .env`.

Para acessar `/admin/`, crie seu próprio usuário:

```powershell
.\.venv\Scripts\python.exe manage.py createsuperuser
```

O banco SQLite é criado pelas migrações. Usuários e registros da máquina original não são distribuídos.

## Modelo de IA e Git LFS

O arquivo `DAVsimulator/modelos/cnn_ecg_treinado.keras` é necessário na inicialização da aplicação e é versionado por Git LFS, conforme `.gitattributes`. Clone com Git e execute `git lfs pull` para obter o modelo completo. Se o carregamento do Keras falhar, confira se o arquivo tem aproximadamente 305 MB; um arquivo de poucas linhas é apenas o ponteiro LFS.

## Dados e uso

1. Acesse `/dav/datasets/upload/`.
2. Importe um ZIP de sinais no formato WFDB, com os arquivos `.hea` e `.dat` correspondentes, e informe a frequência de amostragem correta.
3. Acesse `/dav/datasets/`, selecione o registro e abra a análise de sinais.
4. Explore a simulação em `/dav/`, a sintonia em `/dav/sintonia/` e as páginas de validação pelo menu.

Os sinais locais MIMIC, uploads e banco de dados não estão incluídos. Para reproduzir as análises originais, obtenha os dados pela fonte autorizada e importe-os na sua instalação. O script `DAVsimulator/dataset/download_mimic.py` contém a referência ao registro `3000003` em `mimic3wdb/30/3000003`; ele lê o registro remoto e imprime informações, sem cadastrar dados no sistema.

## Verificação e resultados

```powershell
.\.venv\Scripts\python.exe manage.py check
.\.venv\Scripts\python.exe manage.py test
.\.venv\Scripts\python.exe gerar_resultados_controle_final.py
```

Os dois scripts de resultados dependem de um registro importado e dos segmentos esperados pelo experimento. Eles selecionam o primeiro registro do banco; para reproduzir o experimento original, importe o registro correspondente antes de executá-los. Consulte as opções com `python gerar_resultados_dissertacao_final.py --help`. Os relatórios, CSVs e figuras gerados ficam nas pastas `resultados_*`, ignoradas pelo Git.

## Organização

| Caminho | Conteúdo |
| --- | --- |
| `meusistema/` | Configurações e rotas Django |
| `DAVsimulator/` | Interface, modelos de dados, formulários e migrações |
| `DAVsimulator/services/` | Processamento de sinais, IA, fuzzy e controle |
| `DAVsimulator/modelos/` | CNN treinada, armazenada no Git LFS |
| `projeto/` | Painel e templates compartilhados |
| `gerar_resultados_*.py` | Geração de resultados dos experimentos |
| `CNN_ECG_ATUAL.ipynb` | Notebook complementar de treinamento |

## Publicação no GitHub

Crie um repositório vazio na sua conta, sem gerar README, licença ou `.gitignore` no site. Na pasta deste projeto:

```powershell
git init -b main
git lfs install --local
git add .
git status
git lfs ls-files
git commit -m "Prepara DAV Simulator para compartilhamento acadêmico"
git remote add origin <URL_DO_REPOSITORIO>
git push -u origin main
```

Revise a lista em `git status` antes do commit. O `.gitignore` exclui configurações privadas, banco, sinais, uploads, logs, ambiente virtual e resultados locais. Se o repositório for privado, conceda acesso ao professor nas configurações do GitHub.

Publicar o repositório compartilha o código; o professor executa o sistema localmente seguindo os passos acima. A configuração fornecida utiliza `DEBUG=True` para desenvolvimento local.
