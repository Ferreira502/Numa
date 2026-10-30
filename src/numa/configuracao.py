# Caminhos e parametros usados pelo projeto

from pathlib import Path


# Parte da localizacao deste arquivo, sem depender da pasta aberta no terminal
RAIZ_PROJETO = Path(__file__).resolve().parents[2]
PASTA_DADOS_BRUTOS = RAIZ_PROJETO / "dados" / "brutos"
ARQUIVO_DADOS_PROCESSADOS = RAIZ_PROJETO / "dados" / "processados" / "dados.json"

FRACAO_TREINO = 0.9

# Parametros do Transformer pequeno
TAMANHO_CONTEXTO = 32
DIMENSAO_EMBEDDING = 64
NUM_CABECAS = 4
NUM_CAMADAS = 2
DROPOUT = 0.1
