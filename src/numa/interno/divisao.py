# Compartilha a normalizacao e a divisao estavel usadas pelos leitores e experimentos

import hashlib


# Uniformiza espacos e maiusculas somente para identificar perguntas e pares repetidos
def normalizar_texto(texto: str) -> str:
    return " ".join(texto.split()).casefold()


# Mantem a mesma pergunta na mesma divisao sem depender da ordem dos arquivos
def definir_divisao(pergunta: str, fracao_treino: float, semente: int) -> tuple[str, str]:
    normalizada = normalizar_texto(pergunta)
    grupo = hashlib.sha256(normalizada.encode("utf-8")).hexdigest()
    assinatura = hashlib.sha256(f"{semente}:{grupo}".encode("ascii")).digest()
    numero = int.from_bytes(assinatura[:8], "big")
    parte = "treino" if numero < int(fracao_treino * 2**64) else "validacao"
    return parte, grupo
