# Prepara textos continuos para aprender a linguagem antes do ajuste com conversas
# Separa trechos de treino e validacao e grava IDs binarios sem carregar o livro inteiro na RAM
# Remove a capa e o indice do livro conhecido mantendo o corte no texto original

import argparse
from array import array
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path

from numa.configuracao import (
    FONTES_LINGUAGEM, FRACAO_TREINO, MARCADOR_FIM, MARCADOR_RESPOSTA,
    MARCADOR_USUARIO, PASTA_BLOCOS, PASTA_LINGUAGEM_BASE, TAMANHO_CONTEXTO,
)
from numa.experimentos.preparar_blocos import remover_area_temporaria
from numa.tokenizer import TokenizadorCaracteres


# Percorre caracteres UTF-8 aos poucos, normalizando as quebras de linha do Windows
def iterar_texto(caminho: Path, inicio: int = 0, fim: int | None = None):
    posicao = 0
    with caminho.open(encoding="utf-8-sig") as entrada:
        while trecho := entrada.read(65536):
            limite = len(trecho) if fim is None else min(len(trecho), fim - posicao)
            selecionado = trecho[max(0, inicio - posicao):max(0, limite)]
            if selecionado:
                yield selecionado
            posicao += len(trecho)
            if fim is not None and posicao >= fim:
                break


# Localiza a prosa do livro conhecido sem remover partes de outros textos por suposicao
def localizar_prosa(fonte: str, caminho: Path, quantidade: int) -> tuple[int, int]:
    if fonte != "dom_casmurro":
        return 0, quantidade
    inicio = None
    fim = None
    posicao = 0
    with caminho.open(encoding="utf-8-sig") as entrada:
        for linha in entrada:
            if inicio is None and linha.strip() == "I":
                inicio = posicao
            elif inicio is not None and linha.strip() == "INDICE":
                fim = posicao
                break
            posicao += len(linha)
    if inicio is None or fim is None or inicio >= fim:
        raise ValueError("Nao foi possivel identificar o primeiro capitulo e o indice de Dom Casmurro")
    return inicio, fim


# Conta caracteres e identifica o conteudo de cada fonte sem manter os textos na memoria
def examinar_fontes(fontes: dict[str, Path], fracao_treino: float, divisao_anterior: dict | None = None):
    caracteres = set(MARCADOR_USUARIO + MARCADOR_RESPOSTA + MARCADOR_FIM)
    resumos = {}
    for fonte, caminho in fontes.items():
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", fonte):
            raise ValueError("Use nomes de fontes sem acentos e sem separadores de caminho")
        assinatura = hashlib.sha256()
        quantidade = 0
        for trecho in iterar_texto(caminho):
            quantidade += len(trecho)
            assinatura.update(trecho.encode("utf-8"))
        assinatura_original = assinatura.hexdigest()
        corte_original = int(quantidade * fracao_treino)
        if divisao_anterior is not None:
            anterior = divisao_anterior["fontes"][fonte]
            if assinatura_original != anterior.get("sha256_original", anterior["sha256_texto_normalizado"]):
                raise ValueError(f"A fonte {fonte} mudou, nao e seguro reutilizar a divisao anterior")
            corte_original = anterior.get("corte_original", anterior["corte_treino"])
        inicio, fim = localizar_prosa(fonte, caminho, quantidade)
        corte = corte_original - inicio
        if min(corte, fim - inicio - corte) <= TAMANHO_CONTEXTO:
            raise ValueError(f"O texto {fonte} e curto demais para gerar treino e validacao com este contexto")
        assinatura = hashlib.sha256()
        for trecho in iterar_texto(caminho, inicio, fim):
            caracteres.update(trecho)
            assinatura.update(trecho.encode("utf-8"))
        resumos[fonte] = {
            "caminho": str(caminho), "caracteres": fim - inicio, "corte_treino": corte,
            "sha256_texto_normalizado": assinatura.hexdigest(),
            "caracteres_originais": quantidade, "corte_original": corte_original,
            "inicio_util": inicio, "fim_util": fim, "sha256_original": assinatura_original,
        }
    return caracteres, resumos


# Escreve IDs de 32 bits em ordem little-endian para leitura mapeada pelo PyTorch
def gravar_ids(arquivo, texto: str, tokenizador: TokenizadorCaracteres) -> None:
    numeros = array("I", tokenizador.codificar(texto))
    if numeros.itemsize != 4:
        raise ValueError("Esta plataforma nao usa inteiros de 32 bits no formato esperado")
    if sys.byteorder != "little":
        numeros.byteswap()
    arquivo.write(numeros.tobytes())


# Grava duas sequencias separadas por fonte para impedir janelas cruzando treino e validacao
def gravar_fontes(fontes: dict[str, Path], pasta: Path, tokenizador: TokenizadorCaracteres, resumos: dict):
    partes = {parte: {"total": 0, "fontes": {}} for parte in ("treino", "validacao")}
    for parte in partes:
        (pasta / parte).mkdir()
    for fonte, caminho in fontes.items():
        corte = resumos[fonte]["corte_treino"]
        relativos = {parte: Path(parte) / f"{fonte}.bin" for parte in partes}
        assinatura = hashlib.sha256()
        posicao = 0
        with (pasta / relativos["treino"]).open("xb") as treino, (pasta / relativos["validacao"]).open("xb") as validacao:
            for trecho in iterar_texto(caminho, resumos[fonte]["inicio_util"], resumos[fonte]["fim_util"]):
                separacao = max(0, min(len(trecho), corte - posicao))
                gravar_ids(treino, trecho[:separacao], tokenizador)
                gravar_ids(validacao, trecho[separacao:], tokenizador)
                assinatura.update(trecho.encode("utf-8"))
                posicao += len(trecho)
        if posicao != resumos[fonte]["caracteres"] or assinatura.hexdigest() != resumos[fonte]["sha256_texto_normalizado"]:
            raise ValueError(f"O texto {fonte} mudou durante a preparacao")
        for parte, quantidade in (("treino", corte), ("validacao", posicao - corte)):
            arquivo = pasta / relativos[parte]
            with arquivo.open("rb") as entrada:
                assinatura_arquivo = hashlib.file_digest(entrada, "sha256").hexdigest()
            # Cada inicio de janela precisa deixar contexto mais um caractere disponivel
            janelas = quantidade - TAMANHO_CONTEXTO
            partes[parte]["fontes"][fonte] = {
                "arquivo": relativos[parte].as_posix(), "caracteres": quantidade,
                "total": janelas, "sha256": assinatura_arquivo,
            }
            partes[parte]["total"] += janelas
    return partes


# Monta os dados da primeira etapa e inclui o vocabulario de conversas quando ele estiver disponivel
def preparar_linguagem(fontes: dict[str, Path], destino: Path, referencia: Path | None = None,
                       fracao_treino: float = FRACAO_TREINO, divisao_anterior: Path | None = None) -> dict:
    destino = destino.resolve()
    if destino.exists():
        raise ValueError(f"A pasta ja existe, use --saida para escolher outra: {destino}")
    if not fontes or not 0 < fracao_treino < 1:
        raise ValueError("Informe fontes e uma fracao de treino entre 0 e 1")
    divisao = None
    if divisao_anterior is not None:
        divisao = json.loads(divisao_anterior.read_text(encoding="utf-8"))
        if divisao.get("formato") != "linguagem_blocos_v1":
            raise ValueError("A divisao anterior precisa ser um manifesto de linguagem")
        fracao_treino = divisao["fracao_treino"]
        if not 0 < fracao_treino < 1:
            raise ValueError("Fracao de treino invalida no manifesto anterior")
        referencia = (divisao_anterior.resolve().parent / divisao["arquivo_vocabulario"]).resolve()
        if not referencia.is_relative_to(divisao_anterior.resolve().parent):
            raise ValueError("O vocabulario anterior esta fora da pasta esperada")
    caracteres, resumos = examinar_fontes(fontes, fracao_treino, divisao)
    if referencia is not None:
        vocabulario_referencia = json.loads(referencia.read_text(encoding="utf-8"))["vocabulario"]
        if not isinstance(vocabulario_referencia, list) or any(not isinstance(c, str) or len(c) != 1 for c in vocabulario_referencia):
            raise ValueError("O vocabulario de referencia precisa conter caracteres individuais")
        if divisao is not None and (vocabulario_referencia != sorted(set(vocabulario_referencia)) or not caracteres <= set(vocabulario_referencia)):
            raise ValueError("Os caracteres mudaram, nao e seguro reutilizar os pesos anteriores")
        caracteres.update(vocabulario_referencia)
    tokenizador = TokenizadorCaracteres("".join(sorted(caracteres)))
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporaria = destino.with_name(f".{destino.name}.preparando_{uuid.uuid4().hex}")
    temporaria.mkdir()
    try:
        vocabulario = {
            "vocabulario": list(tokenizador.caracteres),
            "marcadores": {"usuario": MARCADOR_USUARIO, "resposta": MARCADOR_RESPOSTA, "fim": MARCADOR_FIM},
        }
        (temporaria / "vocabulario.json").write_text(json.dumps(vocabulario, ensure_ascii=False), encoding="utf-8")
        partes = gravar_fontes(fontes, temporaria, tokenizador, resumos)
        manifesto = {
            "formato": "linguagem_blocos_v1", "arquivo_vocabulario": "vocabulario.json",
            "tamanho_vocabulario": tokenizador.tamanho_vocabulario, "tamanho_contexto": TAMANHO_CONTEXTO,
            "tipo_ids": "int32_little_endian", "divisao": "corte_original_com_recorte_de_prosa_v2",
            "fracao_treino": fracao_treino, "fontes": resumos, "partes": partes,
            "divisao_anterior": str(divisao_anterior.resolve()) if divisao_anterior else None,
        }
        (temporaria / "manifesto.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")
        if temporaria.resolve().parent != destino.parent or destino.exists():
            raise ValueError("O destino mudou durante a preparacao")
        temporaria.rename(destino)
        return manifesto
    except BaseException:
        if temporaria.exists():
            remover_area_temporaria(temporaria, destino)
        raise


# Prepara os textos locais e mostra as quantidades sem executar treinamento
def principal() -> None:
    analisador_argumentos = argparse.ArgumentParser(description="Prepara a primeira etapa de aprendizado da linguagem")
    analisador_argumentos.add_argument("--saida", type=Path, default=PASTA_LINGUAGEM_BASE)
    analisador_argumentos.add_argument("--preservar-divisao", type=Path, help="Manifesto anterior para preservar a fronteira e os IDs dos pesos existentes")
    argumentos = analisador_argumentos.parse_args()
    referencia = PASTA_LINGUAGEM_BASE / "vocabulario.json"
    if not referencia.exists():
        referencia = PASTA_BLOCOS / "vocabulario.json"
    try:
        manifesto = preparar_linguagem(FONTES_LINGUAGEM, argumentos.saida,
                                      referencia=referencia if referencia.exists() else None,
                                      divisao_anterior=argumentos.preservar_divisao)
    except (OSError, ValueError, KeyError) as erro:
        raise SystemExit(f"Nao foi possivel preparar a linguagem: {erro}") from erro
    print(f"Vocabulario: {manifesto['tamanho_vocabulario']} caracteres")
    for parte in ("treino", "validacao"):
        quantidade = sum(fonte["caracteres"] for fonte in manifesto["partes"][parte]["fontes"].values())
        print(f"{parte}: {quantidade} caracteres | {manifesto['partes'][parte]['total']} janelas possiveis")
    print("Preparacao concluida, nenhum treinamento foi executado")


if __name__ == "__main__":
    principal()
