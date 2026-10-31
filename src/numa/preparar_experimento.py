# Prepara um experimento pequeno para verificar respostas e continuidade por assunto
# Separa conversas completas e preserva o vocabulario, os datasets e os pesos anteriores

import argparse
import hashlib
import json
from pathlib import Path

from numa.configuracao import PASTA_LINGUAGEM, RAIZ_PROJETO, TAMANHO_CONTEXTO
from numa.preparar_continuidade import codificar_continuacao
from numa.tokenizer import TokenizadorCaracteres


# Confere cada conversa antes de separar seus prefixos em exemplos supervisionados
def validar_conversa(conversa: dict) -> None:
    if conversa["divisao"] not in ("treino", "validacao", "teste"):
        raise ValueError("Divisao desconhecida")
    if not conversa["id"] or not conversa["assunto"] or not conversa["trocas"]:
        raise ValueError("Informe identificador, assunto e trocas")
    for troca in conversa["trocas"]:
        if len(troca) != 2 or any(not isinstance(texto, str) or not texto.strip() for texto in troca):
            raise ValueError("Cada troca precisa de pergunta e resposta nao vazias")
    if "fontes_turnos" in conversa:
        fontes = conversa["fontes_turnos"]
        if len(fontes) != len(conversa["trocas"]) or any(not isinstance(f, str) or not f for f in fontes):
            raise ValueError("Informe uma fonte nao vazia para cada turno")
    if "respostas_aceitas" in conversa:
        aceitas = conversa["respostas_aceitas"]
        if not isinstance(aceitas, list) or len(aceitas) != len(conversa["trocas"]):
            raise ValueError("Informe uma lista de respostas aceitas para cada turno")
        if any(not isinstance(lista, list) or not lista or any(not isinstance(r, str) or not r.strip() for r in lista) for lista in aceitas):
            raise ValueError("Cada lista de respostas aceitas precisa conter textos nao vazios")


# Codifica todos os turnos mantendo a conversa inteira na divisao indicada nos dados brutos
def preparar_experimento(origem: Path, destino: Path, tamanho_contexto: int = TAMANHO_CONTEXTO) -> dict:
    if type(tamanho_contexto) is not int or tamanho_contexto < 1:
        raise ValueError("O contexto precisa ser um inteiro positivo")
    if destino.exists():
        raise ValueError("Escolha uma pasta nova para preservar o experimento anterior")
    vocabulario = json.loads((PASTA_LINGUAGEM / "vocabulario.json").read_text(encoding="utf-8"))
    tokenizador = TokenizadorCaracteres("".join(vocabulario["vocabulario"]))
    registros = [json.loads(linha) for linha in origem.read_text(encoding="utf-8").splitlines() if linha.strip()]
    if not registros:
        raise ValueError("O arquivo de conversas esta vazio")
    exemplos = {"treino": [], "validacao": []}
    casos = {parte: [] for parte in ("treino", "validacao", "teste")}
    cobertura = {parte: {} for parte in casos}
    prefixos = {}
    identificadores = set()
    for conversa in registros:
        validar_conversa(conversa)
        if conversa["id"] in identificadores:
            raise ValueError("Identificador de conversa repetido")
        identificadores.add(conversa["id"])
        parte, assunto = conversa["divisao"], conversa["assunto"]
        cobertura[parte][assunto] = cobertura[parte].get(assunto, 0) + 1
        for indice in range(len(conversa["trocas"])):
            trocas = conversa["trocas"][:indice + 1]
            exemplo, motivo = codificar_continuacao(trocas, tokenizador, vocabulario["marcadores"], tamanho_contexto)
            if motivo:
                raise ValueError(f"{conversa['id']} turno {indice + 1}: {motivo}")
            # A mesma entrada nao pode receber respostas conflitantes nem aparecer em duas divisoes
            prefixo = tuple(exemplo["ids"][:exemplo["inicio_resposta"]])
            resposta = tuple(exemplo["ids"][exemplo["inicio_resposta"]:])
            if prefixo in prefixos and prefixos[prefixo] != (parte, resposta):
                raise ValueError("Entrada repetida entre divisoes ou com respostas diferentes")
            repetido = prefixo in prefixos
            prefixos[prefixo] = (parte, resposta)
            if repetido:
                continue
            caso = {"id": f"{conversa['id']}_{indice + 1}", "tipo": assunto,
                    "historico": trocas[:-1], "pergunta": trocas[-1][0], "referencia": trocas[-1][1],
                    "divisao": parte, "conversa": conversa["id"]}
            if "respostas_aceitas" in conversa:
                caso["respostas_aceitas"] = conversa["respostas_aceitas"][indice]
            casos[parte].append(caso)
            if parte in exemplos:
                fonte = conversa["fontes_turnos"][indice] if "fontes_turnos" in conversa else assunto
                exemplo.update({"fonte": fonte, "conversa": conversa["id"]})
                exemplos[parte].append(exemplo)
    assuntos = set(cobertura["treino"])
    if not assuntos or any(set(cobertura[parte]) != assuntos for parte in cobertura):
        raise ValueError("Todos os assuntos precisam de conversas em treino, validacao e teste")
    # Estes casos ficam fora dos exemplos desta etapa, mas os pesos vieram de treinamentos anteriores
    protocolo = {"cobertura_conversas": cobertura, "contexto": tamanho_contexto,
                 "sha256_origem": hashlib.sha256(origem.read_bytes()).hexdigest(),
                 "divisao": "conversas_inteiras_com_assuntos_em_todas_as_partes",
                 "limite": "Reserva desta etapa, sem garantia de ineditismo nos treinamentos anteriores",
                 "avaliacao": "Geracao completa com historico gerado e revisao humana de pertinencia",
                 "comparacao": "Compare os mesmos casos, temperatura, semente e limite de caracteres"}
    dados = {**vocabulario, "formato": "conversas_v1", **exemplos, "protocolo": protocolo}
    destino.mkdir(parents=True)
    (destino / "dados.json").write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    (destino / "protocolo.json").write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding="utf-8")
    for parte, lista in casos.items():
        texto = "".join(json.dumps(caso, ensure_ascii=False) + "\n" for caso in lista)
        (destino / f"testes_{parte}.jsonl").write_text(texto, encoding="utf-8")
    # Compara uma conversa de treino por assunto com formulacoes reservadas desta etapa
    primeiras = {}
    for caso in casos["treino"]:
        primeiras.setdefault(caso["tipo"], caso["conversa"])
    sondas = [caso for caso in casos["treino"] if caso["conversa"] == primeiras[caso["tipo"]]]
    sondas += casos["teste"]
    texto = "".join(json.dumps(caso, ensure_ascii=False) + "\n" for caso in sondas)
    (destino / "sondas.jsonl").write_text(texto, encoding="utf-8")
    return {"conversas": cobertura, "casos": {parte: len(lista) for parte, lista in casos.items()}}


# Prepara somente os dados e informa os arquivos que serao usados pelo treinamento separado
def principal() -> None:
    analisador = argparse.ArgumentParser(description="Prepara um experimento controlado sem treinar")
    analisador.add_argument("--origem", type=Path, default=RAIZ_PROJETO / "dados/brutos/experimento_controlado/dialogos.jsonl")
    analisador.add_argument("--saida", type=Path, default=RAIZ_PROJETO / "dados/processados/experimento_controlado_256")
    argumentos = analisador.parse_args()
    try:
        resumo = preparar_experimento(argumentos.origem, argumentos.saida)
    except (OSError, ValueError, KeyError, TypeError) as erro:
        raise SystemExit(f"Nao foi possivel preparar o experimento: {erro}") from erro
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    print("Pesos preservados, nenhum treinamento executado")
    print(f"Dados: {argumentos.saida / 'dados.json'}")


if __name__ == "__main__":
    principal()
