# Monta o historico usado tanto pelo chat quanto pelo treino de continuidade
# Mantem somente trocas completas que cabem no contexto, sem alterar os marcadores


# Inclui as trocas recentes completas antes da pergunta atual e do inicio da resposta
def montar_contexto(pergunta: str, historico: list[tuple[str, str]], marcadores: dict, tamanho_contexto: int) -> str:
    if not pergunta.strip():
        raise ValueError("Digite uma mensagem")
    contexto = f"{marcadores['usuario']}{pergunta}{marcadores['resposta']}"
    if len(contexto) > tamanho_contexto:
        limite = tamanho_contexto - len(marcadores["usuario"]) - len(marcadores["resposta"])
        raise ValueError(f"A mensagem deve ter no maximo {limite} caracteres para caber no contexto")
    for mensagem, resposta in reversed(historico):
        troca = f"{marcadores['usuario']}{mensagem}{marcadores['resposta']}{resposta}{marcadores['fim']}"
        if len(troca) + len(contexto) > tamanho_contexto:
            break
        contexto = troca + contexto
    return contexto
