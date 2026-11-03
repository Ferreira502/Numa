# Prepara datasets TXT de conversas com o mesmo leitor para todas as novas etapas
# Preserva os exemplos anteriores, o vocabulario e a separacao das conversas reservadas

import hashlib
import json
import re
import unicodedata
import uuid
from pathlib import Path

from numa.configuracao import (
    PASTA_BASE_AUTOMATICA, PASTA_NOVOS_DATASETS, PASTA_DADOS_AUTOMATICOS, SEMENTE,
)
from numa.interno.contexto import montar_contexto
from numa.tokenizer import TokenizadorCaracteres


# Normaliza a pergunta inicial para manter suas variacoes de caixa na mesma divisao
def chave_conversa(pergunta: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", pergunta).casefold().split())


# Le dialogos completos, separados por linhas vazias, sem descartar seus turnos anteriores
def ler_txt(caminho: Path) -> list[list[tuple[str, str]]]:
    conversas = []
    falas = []
    padrao = re.compile(r"^(Usu(?:a|\u00e1)rio|Numa):\s*(.*)$", re.IGNORECASE)
    linhas = caminho.read_text(encoding="utf-8-sig").splitlines()
    for numero, linha in enumerate([*linhas, ""], 1):
        if not linha.strip():
            if falas:
                if len(falas) % 2:
                    raise ValueError(f"{caminho}:{numero}: falta uma resposta Numa antes de separar a conversa")
                conversas.append([(falas[i][1], falas[i + 1][1]) for i in range(0, len(falas), 2)])
                falas = []
            continue
        encontrada = padrao.match(linha.strip())
        if encontrada:
            papel = "numa" if encontrada[1].casefold() == "numa" else "usuario"
            esperado = "usuario" if len(falas) % 2 == 0 else "numa"
            if papel != esperado or not encontrada[2].strip():
                raise ValueError(f"{caminho}:{numero}: esperava {esperado} com texto nao vazio")
            falas.append([papel, encontrada[2].strip()])
        elif falas:
            falas[-1][1] += "\n" + linha.strip()
        else:
            raise ValueError(f"{caminho}:{numero}: inicie com Usuario: e responda com Numa:")
    if not conversas:
        raise ValueError(f"{caminho}: nenhum dialogo encontrado")
    return conversas


# Monta uma janela com turnos completos que comporte a pergunta e a resposta atual
def codificar_troca(trocas: list[tuple[str, str]], tokenizador, marcadores: dict, contexto: int) -> dict:
    pergunta, resposta = trocas[-1]
    if any("<FIM>" in texto for troca in trocas for texto in troca):
        raise ValueError("Nao inclua o marcador interno <FIM> nos textos")
    historico = list(trocas[:-1])
    while True:
        prefixo = montar_contexto(pergunta, historico, marcadores, contexto)
        texto = prefixo + resposta + marcadores["fim"]
        if len(texto) <= contexto + 1:
            return {"ids": tokenizador.codificar(texto), "inicio_resposta": len(prefixo)}
        if not historico:
            raise ValueError(f"Pergunta e resposta juntas ultrapassam o contexto de {contexto} caracteres")
        historico.pop(0)


# Prepara e reaproveita uma versao dos dados identificada pelo conteudo da base e dos TXT
def preparar_dados(pasta_txt: Path = PASTA_NOVOS_DATASETS, base: Path = PASTA_BASE_AUTOMATICA,
                    pasta_saida: Path = PASTA_DADOS_AUTOMATICOS) -> dict:
    if not pasta_txt.is_dir():
        raise ValueError(f"Crie a pasta de datasets: {pasta_txt}")
    arquivos = sorted(pasta_txt.rglob("*.txt"))
    nomes_base = ("dados.json", "validacao_geracao.jsonl", "testes_teste.jsonl")
    conteudos_base = {nome: (base / nome).read_bytes() for nome in nomes_base}
    assinatura = hashlib.sha256(b"numa_txt_v2")
    for nome, conteudo in conteudos_base.items():
        assinatura.update(nome.encode("utf-8") + b"\0" + conteudo)
    for arquivo in arquivos:
        assinatura.update(arquivo.relative_to(pasta_txt).as_posix().encode("utf-8") + b"\0" + arquivo.read_bytes())
    identificacao = assinatura.hexdigest()
    versao = identificacao[:16]
    destino = pasta_saida / versao
    if destino.exists():
        manifesto = json.loads((destino / "manifesto.json").read_text(encoding="utf-8"))
        if manifesto["assinatura"] != identificacao:
            raise ValueError("A pasta de cache nao corresponde aos datasets")
        return {**manifesto, "pasta": destino, "reutilizado": True}
    dados = json.loads(conteudos_base["dados.json"])
    contexto = dados["protocolo"]["contexto"]
    marcadores = dados["marcadores"]
    tokenizador = TokenizadorCaracteres("".join(dados["vocabulario"]))
    casos_validacao = [json.loads(l) for l in conteudos_base["validacao_geracao.jsonl"].decode("utf-8").splitlines() if l.strip()]
    casos_teste = [json.loads(l) for l in conteudos_base["testes_teste.jsonl"].decode("utf-8").splitlines() if l.strip()]
    prefixos = {}
    raizes = {}
    for parte in ("treino", "validacao", "teste"):
        registros = dados[parte] if parte != "teste" else [
            codificar_troca([*c["historico"], (c["pergunta"], c["referencia"])], tokenizador, marcadores, contexto)
            for c in casos_teste]
        for registro in registros:
            ids = registro["ids"]
            prefixo = tuple(ids[:registro["inicio_resposta"]])
            alvo = tuple(ids[registro["inicio_resposta"]:])
            if prefixo in prefixos and prefixos[prefixo][0] != parte:
                raise ValueError("A base possui uma entrada em divisoes diferentes")
            prefixos[prefixo] = (parte, alvo)
            texto = tokenizador.decodificar(ids)
            pergunta = texto[len(marcadores["usuario"]):].partition(marcadores["resposta"])[0]
            raizes.setdefault(chave_conversa(pergunta), set()).add(parte)
    # As escolhas anteriores acompanham as versoes para impedir que mover um TXT exponha um teste ao treino
    for caminho_manifesto in sorted(pasta_saida.glob("*/manifesto.json")):
        anterior = json.loads(caminho_manifesto.read_text(encoding="utf-8"))
        for chave, parte in anterior.get("divisoes_txt", {}).items():
            raizes.setdefault(chave, set()).add(parte)
    divisoes_txt = {}
    novas = {p: 0 for p in ("treino", "validacao", "teste")}
    for arquivo in arquivos:
        relativo = arquivo.relative_to(pasta_txt)
        partes_explicitas = {p for p in relativo.parts[:-1] if p in novas}
        if len(partes_explicitas) > 1:
            raise ValueError(f"{relativo}: nao misture pastas de treino, validacao e teste")
        explicita = next(iter(partes_explicitas), None)
        for numero, trocas in enumerate(ler_txt(arquivo), 1):
            chave = chave_conversa(trocas[0][0])
            conhecidas = raizes.get(chave, set())
            reservada = next((p for p in ("teste", "validacao", "treino") if p in conhecidas), None)
            resumo_raiz = hashlib.sha256(f"{SEMENTE}:{chave}".encode("utf-8")).hexdigest()
            parte = explicita or reservada or ("validacao" if int(resumo_raiz[:8], 16) % 10 == 0 else "treino")
            if reservada and parte != reservada:
                raise ValueError(f"{relativo}, conversa {numero}: pergunta inicial ja pertence a {reservada}")
            raizes.setdefault(chave, set()).add(parte)
            divisoes_txt[chave] = parte
            for indice in range(len(trocas)):
                try:
                    registro = codificar_troca(trocas[:indice + 1], tokenizador, marcadores, contexto)
                except ValueError as erro:
                    raise ValueError(f"{relativo}, conversa {numero}, turno {indice + 1}: {erro}") from erro
                prefixo = tuple(registro["ids"][:registro["inicio_resposta"]])
                alvo = tuple(registro["ids"][registro["inicio_resposta"]:])
                if prefixo in prefixos:
                    if prefixos[prefixo] != (parte, alvo):
                        raise ValueError(f"{relativo}, conversa {numero}: entrada ja existe com outra resposta ou divisao")
                    continue
                prefixos[prefixo] = (parte, alvo)
                novas[parte] += 1
                identificador = hashlib.sha256(repr((parte, prefixo, alvo)).encode("ascii")).hexdigest()[:20]
                fontes = "historico_longo" if registro["inicio_resposta"] > 256 else "novos_txt"
                registro.update({"fonte": fontes, "conversa": identificador})
                if parte != "teste":
                    dados[parte].append(registro)
                if parte != "treino":
                    # O historico do teste reproduz exatamente a janela usada na preparacao
                    corpo = tokenizador.decodificar(list(prefixo))
                    trechos = corpo.split(marcadores["fim"])
                    historico = [tuple(t[len(marcadores["usuario"]):].split(marcadores["resposta"], 1)) for t in trechos[:-1]]
                    caso = {"id": identificador, "tipo": "novos_txt", "divisao": parte,
                            "historico": historico, "pergunta": trocas[indice][0], "referencia": trocas[indice][1],
                            "grupo_avaliacao": "historico_longo" if registro["inicio_resposta"] > 256 else "novos_txt"}
                    (casos_validacao if parte == "validacao" else casos_teste).append(caso)
    dados["protocolo"] = {**dados["protocolo"], "assinatura_automatica": identificacao,
                          "divisao_txt": "Pergunta inicial inteira na mesma divisao, semente fixa",
                          "contexto": contexto}
    manifesto = {"versao": versao, "assinatura": identificacao, "arquivos_txt": len(arquivos),
                 "novos_exemplos": novas, "treino": len(dados["treino"]), "validacao": len(dados["validacao"]),
                 "teste": len(casos_teste), "contexto": contexto, "divisoes_txt": divisoes_txt}
    pasta_saida.mkdir(parents=True, exist_ok=True)
    temporaria = pasta_saida / (".preparando_" + uuid.uuid4().hex)
    temporaria.mkdir()
    try:
        (temporaria / "dados.json").write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
        for nome, casos in (("validacao_geracao.jsonl", casos_validacao), ("testes_teste.jsonl", casos_teste)):
            (temporaria / nome).write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in casos), encoding="utf-8")
        (temporaria / "manifesto.json").write_text(json.dumps(manifesto, indent=2), encoding="utf-8")
        temporaria.rename(destino)
    finally:
        if temporaria.exists():
            for arquivo in temporaria.iterdir():
                arquivo.unlink()
            temporaria.rmdir()
    return {**manifesto, "pasta": destino, "reutilizado": False}
