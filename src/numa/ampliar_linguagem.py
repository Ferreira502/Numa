# Amplia o corpus de linguagem com respostas em portugues ja disponiveis no projeto
# Preserva o vocabulario e os arquivos do livro e separa novos textos por pergunta
# Os filtros sao de formato e idioma aproximado, sem garantir veracidade ou qualidade editorial

import argparse
import hashlib
import json
import random
import re
import shutil
import sqlite3
import unicodedata
import uuid
from pathlib import Path

from numa.configuracao import FONTES_CONVERSAS, PASTA_LINGUAGEM, PASTA_LINGUAGEM_BASE, SEMENTE, TAMANHO_CONTEXTO
from numa.preparar_blocos import definir_divisao, remover_area_temporaria
from numa.preparar_conversas import iterar_pares, normalizar_texto
from numa.preparar_linguagem import gravar_ids
from numa.tokenizer import TokenizadorCaracteres


# Uniformiza sinais tipograficos e espacos sem remover acentos ou palavras desconhecidas
def normalizar_resposta(texto: str) -> str:
    texto = unicodedata.normalize("NFKC", texto)
    substituicoes = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
                    "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u2022": "-"}
    for anterior, novo in substituicoes.items():
        texto = texto.replace(anterior, novo)
    linhas = [re.sub(r"[ \t]+", " ", linha).strip() for linha in texto.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(linhas)).strip()


# Descarta casos evidentes de codigo, tarefas truncadas, caracteres novos e pouco texto corrido
def motivo_descarte(texto: str, caracteres: set[str], trechos_reservados: list[str]) -> str | None:
    if not 400 <= len(texto) <= 5000:
        return "tamanho"
    if not set(texto) <= caracteres:
        return "caracteres_desconhecidos"
    if re.search(r"```|https?://|<[/!a-zA-Z]|\\[a-zA-Z]|(?m:^\s*(?:def |import |class ))", texto):
        return "codigo_ou_marcacao"
    normalizado = normalizar_texto(texto)
    if re.search(r"(?:pergunta:|resposta:|rationale:|passo \d|texto acima|passagem|alternativa correta)", normalizado):
        return "formato_de_tarefa"
    if re.search(r"(?:frase fornecida|artigo fornecido|texto fornecido|situa\u00e7\u00e3o dada|contexto dado|"
                 r"contexto fornecido|de acordo com a senten\u00e7a|nesta frase|nesta senten\u00e7a|"
                 r"sou uma ia|sou um modelo de linguagem|como uma ia)", normalizado):
        return "dependente_de_contexto_ou_autodescricao"
    if re.match(r"^[A-D][.)]\s", texto) or not re.search(r"[.!?][\"')]*$", texto):
        return "resposta_incompleta_ou_alternativa"
    palavras = set(re.findall(r"[^\W\d_]+", normalizado))
    sinais_portugues = {"n\u00e3o", "uma", "voc\u00ea", "tamb\u00e9m", "s\u00e3o", "est\u00e1", "isso", "essa"}
    if not palavras & sinais_portugues or sum(c.isalpha() for c in texto) < len(texto) * 0.6:
        return "idioma_ou_pouca_prosa"
    if texto.count(".") + texto.count("!") + texto.count("?") < 2:
        return "poucas_frases"
    if any(trecho in normalizado for trecho in trechos_reservados):
        return "sobreposicao_com_validacao_do_livro"
    return None


# Aceita somente arquivos contidos na pasta do conjunto de referencia
def resolver_base(pasta: Path, relativo: str) -> Path:
    caminho = (pasta / relativo).resolve()
    if not caminho.is_relative_to(pasta) or not caminho.is_file():
        raise ValueError(f"Arquivo de referencia ausente ou fora da pasta: {relativo}")
    return caminho


# Reconstroi apenas a validacao pequena do livro para evitar inserir suas passagens nos novos textos
def reservar_validacao(pasta: Path, manifesto: dict, vocabulario: list[str]) -> list[str]:
    from array import array
    import sys

    trechos = []
    for info in manifesto["partes"]["validacao"]["fontes"].values():
        arquivo = resolver_base(pasta, info["arquivo"])
        # A base desta ampliacao e pequena e seus dados sao inteiros little-endian de 32 bits
        if arquivo.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("Use uma base com ate 8 MiB por fonte de validacao para esta verificacao")
        ids = array("I")
        if ids.itemsize != 4:
            raise ValueError("O formato de inteiros desta plataforma nao e compativel")
        ids.frombytes(arquivo.read_bytes())
        if sys.byteorder != "little":
            ids.byteswap()
        if any(i >= len(vocabulario) for i in ids):
            raise ValueError("A validacao de referencia contem IDs invalidos")
        texto = normalizar_texto("".join(vocabulario[i] for i in ids))
        trechos.extend(texto[i:i + 128] for i in range(0, len(texto) - 127, 128))
    return trechos


# Percorre as fontes existentes com ordem de arquivos reproduzivel e registra textos unicos em disco
def selecionar_textos(conexao, fontes: dict[str, Path], caracteres: set[str], reservados: list[str],
                      limite: int, fracao: float, semente: int) -> dict:
    conexao.execute("PRAGMA cache_size=-8192")
    conexao.execute("CREATE TABLE textos (assinatura TEXT PRIMARY KEY, grupo TEXT, parte TEXT, fonte TEXT, texto TEXT)")
    resumo = {}
    for fonte, caminho in fontes.items():
        info = {"examinados": 0, "selecionados": 0, "descartes": {}}
        resumo[fonte] = info
        arquivos = sorted(caminho.glob("*.txt")) if caminho.is_dir() else [caminho]
        random.Random(semente).shuffle(arquivos)
        for arquivo in arquivos:
            for pergunta, resposta in iterar_pares(arquivo):
                info["examinados"] += 1
                if not isinstance(pergunta, str) or not pergunta.strip() or not isinstance(resposta, str):
                    motivo = "par_invalido"
                else:
                    texto = normalizar_resposta(resposta)
                    motivo = motivo_descarte(texto, caracteres, reservados)
                if motivo is None:
                    parte, grupo = definir_divisao(pergunta, fracao, semente)
                    assinatura = hashlib.sha256(normalizar_texto(texto).encode("utf-8")).hexdigest()
                    cursor = conexao.execute("INSERT OR IGNORE INTO textos VALUES (?, ?, ?, ?, ?)",
                                             (assinatura, grupo, parte, fonte, texto))
                    if cursor.rowcount:
                        info["selecionados"] += 1
                    else:
                        motivo = "texto_repetido"
                if motivo is not None:
                    info["descartes"][motivo] = info["descartes"].get(motivo, 0) + 1
                if info["selecionados"] >= limite:
                    break
            if info["selecionados"] >= limite:
                break
        conexao.commit()
        print(f"{fonte}: {info['selecionados']} textos selecionados de {info['examinados']} pares examinados", flush=True)
    return resumo


# Grava textos completos em cada divisao, sem cortar uma mesma resposta entre treino e validacao
def gravar_ampliacao(conexao, pasta: Path, tokenizador: TokenizadorCaracteres) -> dict:
    partes = {}
    for parte in ("treino", "validacao"):
        relativo = Path(parte) / "portugues_atual.bin"
        quantidade = 0
        registros = 0
        with (pasta / relativo).open("xb") as saida, (pasta / f"textos_{parte}.jsonl").open("x", encoding="utf-8") as auditoria:
            cursor = conexao.execute("SELECT assinatura, grupo, fonte, texto FROM textos WHERE parte=? ORDER BY assinatura", (parte,))
            for assinatura, grupo, fonte, texto in cursor:
                trecho = texto + "\n\n"
                gravar_ids(saida, trecho, tokenizador)
                quantidade += len(trecho)
                registros += 1
                auditoria.write(json.dumps({"assinatura": assinatura, "grupo_pergunta": grupo,
                                           "fonte": fonte, "texto": texto}, ensure_ascii=False) + "\n")
        if quantidade <= TAMANHO_CONTEXTO:
            raise ValueError(f"Nao ha texto suficiente em {parte}")
        with (pasta / relativo).open("rb") as entrada:
            assinatura = hashlib.file_digest(entrada, "sha256").hexdigest()
        partes[parte] = {"arquivo": relativo.as_posix(), "caracteres": quantidade,
                        "total": quantidade - TAMANHO_CONTEXTO, "sha256": assinatura, "textos": registros}
    return partes


# Publica uma nova pasta mantendo os arquivos anteriores e os IDs exatamente iguais
def ampliar_linguagem(base: Path, destino: Path, fontes: dict[str, Path], limite: int = 10000) -> dict:
    base = base.resolve()
    destino = destino.resolve()
    if destino.exists() or limite < 1 or not fontes:
        raise ValueError("Escolha uma saida inexistente, fontes e um limite positivo")
    manifesto_original = (base / "manifesto.json").read_bytes()
    manifesto = json.loads(manifesto_original)
    if manifesto.get("formato") != "linguagem_blocos_v1" or manifesto["tipo_ids"] != "int32_little_endian":
        raise ValueError("Informe uma base de linguagem em inteiros de 32 bits")
    if manifesto["tamanho_contexto"] != TAMANHO_CONTEXTO or "portugues_atual" in manifesto["fontes"]:
        raise ValueError("Contexto incompativel ou a base ja foi ampliada")
    vocabulario_original = resolver_base(base, manifesto["arquivo_vocabulario"]).read_bytes()
    dados = json.loads(vocabulario_original)
    tokenizador = TokenizadorCaracteres("".join(dados["vocabulario"]))
    if list(tokenizador.caracteres) != dados["vocabulario"]:
        raise ValueError("A ordem dos IDs de referencia e invalida")
    reservados = reservar_validacao(base, manifesto, dados["vocabulario"])
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporaria = destino.with_name(f".{destino.name}.preparando_{uuid.uuid4().hex}")
    temporaria.mkdir()
    conexao = None
    try:
        (temporaria / "vocabulario.json").write_bytes(vocabulario_original)
        for parte in ("treino", "validacao"):
            (temporaria / parte).mkdir()
            for fonte, info in manifesto["partes"][parte]["fontes"].items():
                if not re.fullmatch(r"[a-zA-Z0-9_-]+", fonte):
                    raise ValueError("Nome de fonte de referencia invalido")
                arquivo = resolver_base(base, info["arquivo"])
                with arquivo.open("rb") as entrada:
                    if hashlib.file_digest(entrada, "sha256").hexdigest() != info["sha256"]:
                        raise ValueError("Um arquivo da base foi alterado")
                relativo = Path(parte) / f"{fonte}.bin"
                shutil.copyfile(arquivo, temporaria / relativo)
                info["arquivo"] = relativo.as_posix()
        indice = temporaria / "selecao.sqlite"
        conexao = sqlite3.connect(indice)
        resumo = selecionar_textos(conexao, fontes, set(dados["vocabulario"]), reservados,
                                  limite, manifesto["fracao_treino"], SEMENTE)
        partes_novas = gravar_ampliacao(conexao, temporaria, tokenizador)
        conexao.close()
        conexao = None
        indice.unlink()
        for parte, info in partes_novas.items():
            manifesto["partes"][parte]["fontes"]["portugues_atual"] = info
            manifesto["partes"][parte]["total"] += info["total"]
        manifesto["fontes"]["portugues_atual"] = {
            "caracteres": sum(info["caracteres"] for info in partes_novas.values()),
            "corte_treino": partes_novas["treino"]["caracteres"],
            "divisao": "sha256_pergunta_normalizada_v1", "semente": SEMENTE,
            "origens": {nome: str(caminho.resolve()) for nome, caminho in fontes.items()}, "selecao": resumo,
        }
        manifesto["divisao"] = "base_preservada_e_novas_perguntas_sha256_v1"
        manifesto["base_preservada"] = {"pasta": str(base), "sha256_manifesto": hashlib.sha256(manifesto_original).hexdigest()}
        (temporaria / "manifesto.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")
        if destino.exists() or temporaria.resolve().parent != destino.parent:
            raise ValueError("O destino mudou durante a preparacao")
        temporaria.rename(destino)
        return manifesto
    except BaseException:
        if conexao is not None:
            conexao.close()
        if temporaria.exists():
            remover_area_temporaria(temporaria, destino)
        raise


# Prepara uma amostra controlada das fontes locais sem baixar arquivos ou iniciar treinamento
def principal() -> None:
    analisador = argparse.ArgumentParser(description="Amplia os textos de linguagem com fontes locais")
    analisador.add_argument("--base", type=Path, default=PASTA_LINGUAGEM_BASE)
    analisador.add_argument("--saida", type=Path, default=PASTA_LINGUAGEM)
    analisador.add_argument("--limite", type=int, default=10000, help="Textos aceitos por fonte")
    argumentos = analisador.parse_args()
    fontes = {nome: FONTES_CONVERSAS[nome] for nome in ("openassistant", "gpt4")}
    try:
        manifesto = ampliar_linguagem(argumentos.base, argumentos.saida, fontes, argumentos.limite)
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as erro:
        raise SystemExit(f"Nao foi possivel ampliar a linguagem: {erro}") from erro
    for parte in ("treino", "validacao"):
        info = manifesto["partes"][parte]["fontes"]["portugues_atual"]
        print(f"{parte}: {info['textos']} novos textos | {info['caracteres']} novos caracteres")
    print("Base e vocabulario preservados, nenhum treinamento executado")


if __name__ == "__main__":
    principal()
