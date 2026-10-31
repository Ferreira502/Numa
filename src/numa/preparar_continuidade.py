# Prepara dialogos com historico e variantes de perguntas usando o mesmo formato do chat
# Mantem familias e arvores inteiras na mesma divisao e referencia os dados anteriores sem copia-los

import argparse
import gzip
import hashlib
import itertools
import json
import os
import uuid
from pathlib import Path

from numa.configuracao import (
    ARQUIVO_MANIFESTO_AJUSTE, FRACAO_TREINO, PASTA_CONTINUIDADE, RAIZ_PROJETO,
    SEMENTE, TAMANHO_CONTEXTO,
)
from numa.contexto import montar_contexto
from numa.preparar_blocos import definir_divisao, remover_area_temporaria
from numa.ampliar_linguagem import normalizar_resposta
from numa.tokenizer import TokenizadorCaracteres


# Expande as formulacoes de cada turno mantendo as respostas e o historico do mesmo dialogo
def iterar_guiados(caminho: Path):
    with caminho.open(encoding="utf-8") as entrada:
        for linha in entrada:
            registro = json.loads(linha)
            if not isinstance(registro["grupo"], str) or not registro["grupo"] or not registro["trocas"]:
                raise ValueError("Informe um grupo e as trocas do dialogo")
            variantes = []
            for troca in registro["trocas"]:
                if not troca["perguntas"] or not isinstance(troca["resposta"], str):
                    raise ValueError("Cada troca precisa de perguntas e resposta")
                variantes.append([(pergunta, troca["resposta"]) for pergunta in troca["perguntas"]])
            if len(variantes) > 5 or any(len(v) > 4 for v in variantes):
                raise ValueError("Limite cada dialogo a cinco turnos e quatro variantes por turno")
            for caminho_trocas in itertools.product(*variantes):
                yield "familia:" + registro["grupo"], list(caminho_trocas)


# Reconstroi caminhos humanos revisados em portugues sem escolher respostas de classificacao inferior
def iterar_openassistant(caminho: Path):
    mensagens = {}
    with gzip.open(caminho, "rt", encoding="utf-8") as entrada:
        for linha in entrada:
            m = json.loads(linha)
            if m.get("lang") not in ("pt", "pt-BR") or m.get("review_result") is not True or m.get("deleted") or m.get("synthetic"):
                continue
            if m.get("role") == "assistant" and m.get("rank") != 0:
                continue
            if m.get("role") in ("assistant", "prompter") and isinstance(m.get("text"), str):
                mensagens[m["message_id"]] = m
    for mensagem in mensagens.values():
        if mensagem["role"] != "assistant":
            continue
        cadeia = []
        atual = mensagem
        vistos = set()
        while atual is not None and atual["message_id"] not in vistos:
            vistos.add(atual["message_id"])
            cadeia.append(atual)
            pai = atual.get("parent_id")
            if pai is None:
                break
            atual = mensagens.get(pai)
        if not cadeia or cadeia[-1].get("parent_id") is not None or len(cadeia) < 4:
            continue
        cadeia.reverse()
        if len(cadeia) % 2 or any(m["role"] != ("prompter" if i % 2 == 0 else "assistant") for i, m in enumerate(cadeia)):
            continue
        trocas = [(cadeia[i]["text"], cadeia[i + 1]["text"]) for i in range(0, len(cadeia), 2)]
        yield trocas[0][0], trocas


# Converte um exemplo com historico sem cortar a resposta nem esconder o tema anterior
def codificar_continuacao(trocas: list[tuple[str, str]], tokenizador, marcadores: dict,
                          tamanho_contexto: int = TAMANHO_CONTEXTO):
    limpas = [(normalizar_resposta(q), normalizar_resposta(r)) for q, r in trocas]
    if any(not q or not r or "<FIM>" in q or "<FIM>" in r for q, r in limpas):
        return None, "invalido"
    historico = limpas[:-1]
    pergunta, resposta = limpas[-1]
    try:
        contexto = montar_contexto(pergunta, historico, marcadores, tamanho_contexto)
    except ValueError:
        return None, "contexto_longo"
    completo = "".join(marcadores["usuario"] + q + marcadores["resposta"] + r + marcadores["fim"] for q, r in historico)
    completo += marcadores["usuario"] + pergunta + marcadores["resposta"]
    if contexto != completo:
        return None, "historico_fora_contexto"
    texto = contexto + resposta + marcadores["fim"]
    if len(texto) > tamanho_contexto + 1:
        return None, "resposta_nao_cabe_com_historico"
    try:
        ids = tokenizador.codificar(texto)
    except ValueError:
        return None, "caractere_desconhecido"
    return {"ids": ids, "inicio_resposta": len(contexto), "turnos": len(limpas)}, None


# Escreve os exemplos pequenos em blocos compativeis com o leitor de conversas existente
def gravar_historicos(pasta: Path, exemplos: dict, vocabulario: bytes, resumos: dict) -> None:
    pasta.mkdir()
    (pasta / "vocabulario.json").write_bytes(vocabulario)
    partes = {}
    for parte in ("treino", "validacao"):
        (pasta / parte).mkdir()
        fontes = {}
        for fonte, registros in exemplos[parte].items():
            blocos = []
            for inicio in range(0, len(registros), 128):
                lote = registros[inicio:inicio + 128]
                relativo = Path(parte) / f"{fonte}_{len(blocos) + 1:05d}.jsonl"
                conteudo = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in lote).encode("utf-8")
                (pasta / relativo).write_bytes(conteudo)
                blocos.append({"arquivo": relativo.as_posix(), "exemplos": len(lote), "sha256": hashlib.sha256(conteudo).hexdigest()})
            fontes[fonte] = {"total": len(registros), "blocos": blocos}
        partes[parte] = {"fontes": fontes, "total": sum(info["total"] for info in fontes.values())}
    if any(partes[p]["total"] == 0 for p in partes):
        raise ValueError("A continuidade precisa de treino e validacao nao vazios")
    dados = json.loads(vocabulario)
    manifesto = {"formato": "conversas_blocos_v1", "esquema_exemplos": "conversas_completas_v2",
                 "arquivo_vocabulario": "vocabulario.json", "tamanho_vocabulario": len(dados["vocabulario"]),
                 "tamanho_contexto": TAMANHO_CONTEXTO, "exemplos_por_bloco": 128,
                 "fracao_treino": FRACAO_TREINO, "semente": SEMENTE, "fontes": resumos, "partes": partes,
                 "divisao": "familia_ou_raiz_inteira_v1", "validacao_guiada": "somente_continuacoes"}
    (pasta / "manifesto.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")


# Prepara dados novos e uma mistura que reutiliza os blocos anteriores sem modificar pesos
def preparar_continuidade(destino: Path, base: Path, guiados: Path, humanos: Path,
                          fontes_adicionais: dict | None = None, prefixos_reservados: set[str] | None = None) -> dict:
    destino, base = destino.resolve(), base.resolve()
    if destino.exists():
        raise ValueError("Escolha uma pasta de saida que ainda nao existe")
    anterior = json.loads(base.read_text(encoding="utf-8"))
    if anterior.get("formato") != "conversas_blocos_v1":
        raise ValueError("A base precisa ser o manifesto do ajuste anterior")
    caminho_vocab = (base.parent / anterior["arquivo_vocabulario"]).resolve()
    if not caminho_vocab.is_relative_to(base.parent) or not base.is_relative_to(RAIZ_PROJETO):
        raise ValueError("Referencia de dados fora do projeto")
    vocabulario = caminho_vocab.read_bytes()
    dados = json.loads(vocabulario)
    tokenizador = TokenizadorCaracteres("".join(dados["vocabulario"]))
    if list(tokenizador.caracteres) != dados["vocabulario"]:
        raise ValueError("A ordem dos IDs de referencia e invalida")
    fontes = {"dialogos_guiados": iterar_guiados(guiados), "openassistant_historico": iterar_openassistant(humanos)}
    if fontes_adicionais:
        if set(fontes) & set(fontes_adicionais):
            raise ValueError("Use nomes diferentes para as fontes adicionais")
        fontes.update(fontes_adicionais)
    exemplos = {parte: {fonte: [] for fonte in fontes} for parte in ("treino", "validacao")}
    resumos = {fonte: {"treino": 0, "validacao": 0, "continuacoes": 0, "descartes": {}} for fonte in fontes}
    vistos = set()
    partes_prefixos = {}
    for fonte, conversas in fontes.items():
        for raiz, trocas in conversas:
            parte, grupo = definir_divisao(raiz, FRACAO_TREINO, SEMENTE)
            # Os caminhos humanos ja terminam numa continuacao e as familias guiadas incluem cada prefixo
            finais = range(1, len(trocas) + 1) if fonte == "dialogos_guiados" else [len(trocas)]
            for fim in finais:
                if parte == "validacao" and fim == 1:
                    continue
                exemplo, motivo = codificar_continuacao(trocas[:fim], tokenizador, dados["marcadores"])
                if motivo:
                    resumos[fonte]["descartes"][motivo] = resumos[fonte]["descartes"].get(motivo, 0) + 1
                    continue
                prefixo = hashlib.sha256(json.dumps(exemplo["ids"][:exemplo["inicio_resposta"]]).encode("ascii")).hexdigest()
                if prefixos_reservados and prefixo in prefixos_reservados:
                    descartes = resumos[fonte]["descartes"]
                    descartes["reservado_para_teste"] = descartes.get("reservado_para_teste", 0) + 1
                    continue
                if prefixo in partes_prefixos and partes_prefixos[prefixo] != parte:
                    descartes = resumos[fonte]["descartes"]
                    descartes["prefixo_outra_divisao"] = descartes.get("prefixo_outra_divisao", 0) + 1
                    continue
                chave = hashlib.sha256(json.dumps(exemplo["ids"]).encode("ascii")).hexdigest()
                if chave in vistos:
                    continue
                vistos.add(chave)
                partes_prefixos[prefixo] = parte
                exemplo.update({"fonte": fonte, "grupo_pergunta": grupo})
                exemplos[parte][fonte].append(exemplo)
                resumos[fonte][parte] += 1
                resumos[fonte]["continuacoes"] += fim > 1
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporaria = destino.with_name(f".{destino.name}.preparando_{uuid.uuid4().hex}")
    temporaria.mkdir()
    try:
        gravar_historicos(temporaria / "historico", exemplos, vocabulario, resumos)
        relativo_base = os.path.relpath(base, destino).replace("\\", "/")
        mistura = {"formato": "conversas_mistas_v1", "conjuntos": [
            {"manifesto": relativo_base}, {"manifesto": "historico/manifesto.json"}],
            "fontes_novas": resumos, "descricao": "Pares anteriores e dialogos com historico, sem copiar os blocos antigos"}
        (temporaria / "manifesto.json").write_text(json.dumps(mistura, ensure_ascii=False, indent=2), encoding="utf-8")
        if destino.exists():
            raise ValueError("O destino mudou durante a preparacao")
        temporaria.rename(destino)
    except BaseException:
        if temporaria.exists():
            remover_area_temporaria(temporaria, destino)
        raise
    return resumos


# Executa apenas a preparacao e mostra quantas continuacoes cabem no contexto atual
def principal() -> None:
    analisador = argparse.ArgumentParser(description="Prepara o ajuste de continuidade sem treinar")
    analisador.add_argument("--saida", type=Path, default=PASTA_CONTINUIDADE)
    argumentos = analisador.parse_args()
    try:
        resumo = preparar_continuidade(argumentos.saida, ARQUIVO_MANIFESTO_AJUSTE,
                                       RAIZ_PROJETO / "dados/brutos/dialogos_continuos/conversas.jsonl",
                                       RAIZ_PROJETO / "dados/brutos/openassistant/original.messages.jsonl.gz")
    except (OSError, ValueError, KeyError, TypeError) as erro:
        raise SystemExit(f"Nao foi possivel preparar a continuidade: {erro}") from erro
    for fonte, info in resumo.items():
        print(f"{fonte}: {info['treino']} de treino | {info['validacao']} de validacao | {info['continuacoes']} continuacoes")
        print(f"  Descartes: {info['descartes']}")
    print("Pares anteriores reutilizados, pesos preservados e nenhum treinamento executado")


if __name__ == "__main__":
    principal()
