# DAV Simulator

Sistema acadêmico em Python/Django para simulação de dispositivo de assistência ventricular (DAV), análise de sinais fisiológicos e avaliação de controle hemodinâmico.

O projeto reúne detecção de picos R no ECG com uma CNN, análise de ABP, estimativa de parâmetros hemodinâmicos, demanda fisiológica por lógica fuzzy, curvas de bomba e simulação de controle de RPM com modelo cardiovascular Windkessel. É um protótipo de pesquisa, sem validação para uso clínico.

## Requisitos

- Python **3.13** (versão utilizada no ambiente original: 3.13.2).
- Git e Git LFS.
- Espaço para o ambiente TensorFlow, o modelo treinado (~305 MB) e os sinais importados.

As versões das dependências diretas do ambiente original estão em `requirements.txt`. O notebook de treinamento é material complementar; o arquivo de dependências destina-se à aplicação e aos scripts de resultados, não à reprodução integral do treinamento.

## Instalação no Windows (PowerShell)

Antes dos comandos, instale o **Python 3.13** no computador. No instalador do Windows, marque **Add Python to PATH** e mantenha o Python Launcher (`py`) habilitado. Instale também Git e Git LFS. Feche e reabra o PowerShell após instalar.

Confira se o Python está disponível:

```powershell
py -3.13 --version
```

O comando deve mostrar `Python 3.13.x`. Depois, baixe o projeto e prepare o ambiente:

```powershell
git lfs install
git clone https://github.com/thiagoolivernas-lab/dav-simulator.git
cd dav-simulator
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

## Dados do PhysioNet e importação

Os sinais, uploads e banco de dados local não estão incluídos no repositório. Para reproduzir as análises, baixe o registro **3000003** da **MIMIC-III Waveform Database, versão 1.0**, diretamente na fonte:

[Baixar os sinais do registro 3000003 no PhysioNet](https://physionet.org/content/mimic3wdb/1.0/30/3000003/#files-panel)

### 1. Baixar os arquivos e preparar o ZIP

1. Abra o link acima e localize a lista de arquivos da pasta `3000003`, na seção **Files**. O comando geral de download mostrado na página aponta para a base inteira; para este projeto, baixe somente os arquivos desse registro.
2. Crie uma pasta chamada `3000003` no seu computador e salve nela os arquivos da lista, preservando seus nomes e extensões. Use a opção de download do arquivo; se o navegador exibir o conteúdo de um `.hea`, salve o arquivo original sem acrescentar `.txt`.
3. Inclua o cabeçalho `3000003.hea`, o arquivo `3000003_layout.hea`, todos os pares `.hea` e `.dat` dos segmentos `3000003_0001` até `3000003_0017`, os arquivos `3000003n.hea` e `3000003n.dat` e o arquivo `RECORDS`. Cada segmento precisa de seu `.hea` e do `.dat` de mesmo nome, na mesma pasta.
4. Compacte a pasta em formato **ZIP**, gerando `3000003.zip`. No Windows, clique com o botão direito na pasta e escolha a opção de compactar em ZIP (ou **Enviar para → Pasta compactada**, conforme a versão). O sistema aceita os arquivos dentro da pasta `3000003` no ZIP; não é necessário extraí-los antes do upload.

### 2. Importar no DAV Simulator

Com a instalação concluída e o servidor em execução (`python manage.py runserver` no ambiente virtual):

1. Abra [Upload de Dataset](http://127.0.0.1:8000/dav/datasets/upload/) ou acesse a lista de datasets e clique em **Enviar dataset**.
2. Preencha o formulário:

   | Campo | Valor |
   | --- | --- |
   | Nome | `MIMIC — registro 3000003` |
   | Record id | `3000003` |
   | Arquivo zip | Selecione `3000003.zip` |
   | Frequência amostragem | `125` Hz para os sinais de waveform desse registro |

3. Clique em **Salvar Dataset** e aguarde o processamento. O sistema extrai o ZIP, lê os cabeçalhos e cadastra os sinais encontrados.
4. Na [lista de datasets](http://127.0.0.1:8000/dav/datasets/), clique no nome do registro e depois em **Visualizar sinais**.
5. Selecione o segmento desejado na página de sinais. Para acompanhar o experimento padrão do script da dissertação, utilize `3000003_0008`; a análise alternativa utiliza `3000003_0010`.

Se nenhum segmento aparecer, confira se o ZIP contém os pares `.hea`/`.dat` com os nomes originais. A presença de um sinal pode variar entre segmentos; selecione um trecho com os canais necessários à análise.

### 3. Executar as análises

Explore a simulação em `/dav/`, a sintonia em `/dav/sintonia/` e as páginas de validação pelo menu. Para gerar os relatórios pelos scripts, siga a seção seguinte. Em uma instalação destinada à reprodução, importe o registro `3000003` primeiro, pois os scripts selecionam o primeiro registro do banco.

O script `DAVsimulator/dataset/download_mimic.py` apenas lê o registro remoto e imprime informações; ele não prepara o ZIP nem cadastra os sinais na aplicação.

**Fonte dos dados:** Moody, B., Moody, G., Villarroel, M., Clifford, G. D., & Silva, I. (2020). *MIMIC-III Waveform Database (version 1.0).* PhysioNet. [DOI: 10.13026/c2607m](https://doi.org/10.13026/c2607m). Consulte na página do PhysioNet os termos de uso e as referências solicitadas ao utilizar os dados em trabalhos acadêmicos.

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
