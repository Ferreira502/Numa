# Centraliza os caminhos e os parametros do modelo, treinamento e geracao
# Os outros modulos importam estes valores para usar a mesma configuracao

from pathlib import Path


# Parte da localizacao deste arquivo, sem depender da pasta aberta no terminal
RAIZ_PROJETO = Path(__file__).resolve().parents[2]
PASTA_DADOS_BRUTOS = RAIZ_PROJETO / "dados" / "brutos"
ARQUIVO_DADOS_PROCESSADOS = RAIZ_PROJETO / "dados" / "processados" / "conversas.json"
PASTA_BLOCOS = RAIZ_PROJETO / "dados" / "processados" / "conversas_blocos"
ARQUIVO_MANIFESTO = PASTA_BLOCOS / "manifesto.json"
PASTA_AJUSTE_CONVERSAS = RAIZ_PROJETO / "dados" / "processados" / "conversas_ajuste_256"
ARQUIVO_MANIFESTO_AJUSTE = PASTA_AJUSTE_CONVERSAS / "manifesto.json"
PASTA_CONTINUIDADE = RAIZ_PROJETO / "dados" / "processados" / "conversas_continuidade_256"
ARQUIVO_MANIFESTO_CONTINUIDADE = PASTA_CONTINUIDADE / "manifesto.json"
ARQUIVO_CHECKPOINT_CONTINUIDADE = RAIZ_PROJETO / "modelos" / "numa_conversas_historico_256.pt"
EXEMPLOS_POR_BLOCO = 1000
BLOCOS_EM_CACHE = 3

# Primeira etapa de aprendizado da linguagem, com pesos separados do chat
FONTES_LINGUAGEM = {"dom_casmurro": PASTA_DADOS_BRUTOS / "dom_casmurro.txt"}
PASTA_LINGUAGEM_BASE = RAIZ_PROJETO / "dados" / "processados" / "linguagem_limpa"
PASTA_LINGUAGEM = RAIZ_PROJETO / "dados" / "processados" / "linguagem_ampliada_filtrada"
ARQUIVO_MANIFESTO_LINGUAGEM = PASTA_LINGUAGEM / "manifesto.json"
# Cada corpus e arquitetura usam arquivos separados para preservar os experimentos anteriores
ARQUIVO_CHECKPOINT_LINGUAGEM = RAIZ_PROJETO / "modelos" / "numa_linguagem_ampliada_256.pt"
PACIENCIA_VALIDACAO = 10

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
# Mantem o padrao antigo para os experimentos salvos e define o novo contexto de conversa
CONTEXTO_CONVERSA_LONGA = 512
DIMENSAO_EMBEDDING = 256
NUM_CABECAS = 4
NUM_CAMADAS = 5
DROPOUT = 0.1

# Parametros de treinamento
DISPOSITIVO_TREINO = "auto"
TAMANHO_LOTE = 4
PASSOS_TREINO = 500
TAXA_APRENDIZADO = 0.001
TAXA_AJUSTE_CONVERSAS = 0.0001
INTERVALO_AVALIACAO = 50
LOTES_AVALIACAO = 10
SEMENTE = 42
ARQUIVO_CHECKPOINT = RAIZ_PROJETO / "modelos" / "numa_conversas_ajuste_256.pt"

# Parametros de geracao de texto
CARACTERES_GERADOS = 200
TEMPERATURA = 0.8
