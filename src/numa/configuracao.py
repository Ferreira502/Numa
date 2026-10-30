# Centraliza os caminhos e os parametros do modelo, treinamento e geracao
# Os outros modulos importam estes valores para usar a mesma configuracao

from pathlib import Path


# Parte da localizacao deste arquivo, sem depender da pasta aberta no terminal
RAIZ_PROJETO = Path(__file__).resolve().parents[2]
PASTA_DADOS_BRUTOS = RAIZ_PROJETO / "dados" / "brutos"
ARQUIVO_DADOS_PROCESSADOS = RAIZ_PROJETO / "dados" / "processados" / "conversas.json"
PASTA_BLOCOS = RAIZ_PROJETO / "dados" / "processados" / "conversas_blocos"
EXEMPLOS_POR_BLOCO = 1000

FRACAO_TREINO = 0.9

# Fontes usadas juntas na preparacao e limite de pares selecionados por fonte
FONTES_CONVERSAS = {
    "basicas": PASTA_DADOS_BRUTOS / "conversas_basicas",
    "openassistant": PASTA_DADOS_BRUTOS / "openassistant" / "pares_portugues.jsonl",
    "gpt4": PASTA_DADOS_BRUTOS / "gpt4_ptbr" / "textos",
}
LIMITE_CONVERSAS = 1000
MARCADOR_USUARIO = "Usuario: "
MARCADOR_RESPOSTA = "\nNuma: "
MARCADOR_FIM = "\n<FIM>\n"

# Parametros do Transformer pequeno
TAMANHO_CONTEXTO = 256
DIMENSAO_EMBEDDING = 64
NUM_CABECAS = 4
NUM_CAMADAS = 2
DROPOUT = 0.1

# Parametros de treinamento
DISPOSITIVO_TREINO = "auto"
TAMANHO_LOTE = 4
PASSOS_TREINO = 500
TAXA_APRENDIZADO = 0.001
INTERVALO_AVALIACAO = 50
LOTES_AVALIACAO = 10
SEMENTE = 42
ARQUIVO_CHECKPOINT = RAIZ_PROJETO / "modelos" / "numa_conversas.pt"

# Parametros de geracao de texto
CARACTERES_GERADOS = 200
TEMPERATURA = 0.8
