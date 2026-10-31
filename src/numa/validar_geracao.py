# Avalia respostas completas na validacao sem consultar casos de teste nem alterar pesos
# A correspondencia com respostas aceitas e uma medida textual restrita, nao uma avaliacao semantica geral

import json
from pathlib import Path

from numa.chat import gerar_resposta, torch
from numa.contexto import montar_contexto
from numa.tokenizer import TokenizadorCaracteres


# Ignora apenas caixa, espacos repetidos e pontuacao final ao comparar respostas curtas
def normalizar_comparacao(texto: str) -> str:
    return " ".join(texto.casefold().split()).rstrip(".!?").strip()


# Confere se cada entrada pertence a validacao carregada e se sua referencia corresponde aos alvos
def carregar_casos_validacao(caminho: Path, dados: dict, validacao) -> list[dict]:
    if not isinstance(validacao, list):
        raise ValueError("A geracao durante o treino usa somente conjuntos pequenos no formato conversas_v1")
    tokenizador = TokenizadorCaracteres("".join(dados["vocabulario"]))
    prefixos = {tuple(r["ids"][:r["inicio_resposta"]].tolist()):
                tuple(r["ids"][r["inicio_resposta"]:].tolist()) for r in validacao}
    casos = [json.loads(linha) for linha in caminho.read_text(encoding="utf-8").splitlines() if linha.strip()]
    vistos = set()
    for caso in casos:
        if caso.get("divisao") != "validacao" or caso["id"] in vistos:
            raise ValueError("Use casos de validacao com identificadores unicos")
        vistos.add(caso["id"])
        contexto = montar_contexto(caso["pergunta"], caso["historico"], dados["marcadores"], dados["protocolo"]["contexto"])
        prefixo = tuple(tokenizador.codificar(contexto))
        referencia = tuple(tokenizador.codificar(caso["referencia"] + dados["marcadores"]["fim"]))
        if prefixos.get(prefixo) != referencia:
            raise ValueError("Um caso nao corresponde aos alvos da validacao deste dataset")
        aceitas = caso.get("respostas_aceitas", [caso["referencia"]])
        if not isinstance(aceitas, list) or not aceitas or any(not isinstance(r, str) or not normalizar_comparacao(r) for r in aceitas):
            raise ValueError("Respostas aceitas precisam ser textos nao vazios")
    if not casos:
        raise ValueError("A avaliacao de geracao precisa de casos")
    return casos


# Gera o historico e a resposta atual e conta correspondencias nas respostas aceitas escritas nos dados
@torch.inference_mode()
def avaliar_geracao(modelo, dados: dict, casos: list[dict]) -> dict:
    estava_treinando = modelo.training
    tokenizador = TokenizadorCaracteres("".join(dados["vocabulario"]))
    marcadores = dados["marcadores"]
    resultados = []
    try:
        modelo.eval()
        for caso in casos:
            gerador = torch.Generator().manual_seed(42)
            historico = []
            for pergunta, _ in caso["historico"]:
                contexto = montar_contexto(pergunta, historico, marcadores, modelo.tamanho_contexto)
                resposta = gerar_resposta(modelo, tokenizador, contexto, marcadores, 100, 0.5, gerador)
                if resposta:
                    historico.append((pergunta, resposta))
            contexto = montar_contexto(caso["pergunta"], historico, marcadores, modelo.tamanho_contexto)
            resposta = gerar_resposta(modelo, tokenizador, contexto, marcadores, 100, 0.5, gerador)
            aceitas = caso.get("respostas_aceitas", [caso["referencia"]])
            corresponde = normalizar_comparacao(resposta) in {normalizar_comparacao(r) for r in aceitas}
            resultados.append({"id": caso["id"], "historico_usado": historico, "resposta": resposta,
                               "referencia": caso["referencia"], "corresponde": corresponde,
                               "grupo_avaliacao": caso.get("grupo_avaliacao", "geral")})
    finally:
        modelo.train(estava_treinando)
    quantidade = sum(r["corresponde"] for r in resultados)
    grupos = {}
    for resultado in resultados:
        grupo = grupos.setdefault(resultado["grupo_avaliacao"], {"correspondencias": 0, "total": 0})
        grupo["total"] += 1
        grupo["correspondencias"] += int(resultado["corresponde"])
    for grupo in grupos.values():
        grupo["proporcao"] = grupo["correspondencias"] / grupo["total"]
    # Cada grupo recebe o mesmo peso para impedir que muitos casos curtos escondam falhas nos longos
    proporcao = sum(g["proporcao"] for g in grupos.values()) / len(grupos)
    return {"correspondencias": quantidade, "total": len(resultados), "proporcao": proporcao,
            "proporcao_global": quantidade / len(resultados), "grupos": grupos,
            "casos": resultados, "temperatura": 0.5, "semente": 42, "caracteres": 100,
            "criterio": "correspondencia_textual_balanceada_por_grupo_v2"}
