# Prepara as conversas em duas passagens usando um indice temporario no disco
# Grava blocos por fonte e por divisao, um vocabulario comum e o manifesto usado pelo treinamento

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import uuid
from pathlib import Path

from numa.configuracao import (
    EXEMPLOS_POR_BLOCO,
    FONTES_CONVERSAS,
    FRACAO_TREINO,
    MARCADOR_FIM,
    MARCADOR_RESPOSTA,
    MARCADOR_USUARIO,
    PASTA_BLOCOS,
    SEMENTE,
    TAMANHO_CONTEXTO,
)
from numa.experimentos.preparar_conversas import codificar_conversas, iterar_pares, normalizar_texto
from numa.tokenizer import TokenizadorCaracteres
from numa.interno.divisao import definir_divisao


# Filtra e registra os pares no SQLite, reunindo somente os caracteres do vocabulario na RAM
def catalogar_conversas(fontes: dict[str, Path], conexao, limite: int, fracao_treino: float, semente: int,
                       vocabulario_fixo: set[str] | None = None, incluir_longas: bool = False,
                       respostas_vistas: set[str] | None = None):
    # O import local evita um ciclo entre os preparadores que compartilham a divisao por pergunta
    if incluir_longas:
        from numa.experimentos.ampliar_linguagem import normalizar_resposta
    # O indice de centenas de milhares de pares cabe neste cache sem ocupar a RAM com os textos
    conexao.execute("PRAGMA cache_size = -65536")
    conexao.execute(
        "CREATE TABLE exemplos (chave TEXT PRIMARY KEY, pergunta TEXT NOT NULL, "
        "resposta TEXT NOT NULL, fonte TEXT NOT NULL, parte TEXT NOT NULL, grupo TEXT NOT NULL)"
    )
    caracteres = set()
    resumos = {}
    for fonte, caminho in fontes.items():
        resumo = {
            "caminho": str(caminho), "examinadas": 0, "selecionadas": 0,
            "longas": 0, "invalidas": 0, "repetidas": 0, "treino": 0, "validacao": 0,
            "longas_incluidas": 0, "caracteres_desconhecidos": 0, "validacao_vista_na_linguagem": 0,
        }
        print(f"Lendo fonte: {fonte}", flush=True)
        for pergunta, resposta in iterar_pares(caminho):
            resumo["examinadas"] += 1
            if resumo["examinadas"] % 50000 == 0:
                print(f"  {fonte}: {resumo['examinadas']} pares examinados", flush=True)
            if not isinstance(pergunta, str) or not isinstance(resposta, str):
                resumo["invalidas"] += 1
                continue
            pergunta, resposta = pergunta.strip(), resposta.strip()
            # A pergunta original mantem a mesma divisao usada na etapa de linguagem
            parte, grupo = definir_divisao(pergunta, fracao_treino, semente)
            if incluir_longas:
                pergunta, resposta = normalizar_resposta(pergunta), normalizar_resposta(resposta)
                if vocabulario_fixo is not None:
                    # Preserva as letras acentuadas quando apenas a forma minuscula e conhecida
                    pergunta = "".join(c.lower() if c not in vocabulario_fixo and c.lower() in vocabulario_fixo else c for c in pergunta)
                    resposta = "".join(c.lower() if c not in vocabulario_fixo and c.lower() in vocabulario_fixo else c for c in resposta)
            if not pergunta or not resposta or "<FIM>" in pergunta or "<FIM>" in resposta:
                resumo["invalidas"] += 1
                continue
            texto = f"{MARCADOR_USUARIO}{pergunta}{MARCADOR_RESPOSTA}{resposta}{MARCADOR_FIM}"
            if vocabulario_fixo is not None and not set(texto) <= vocabulario_fixo:
                resumo["caracteres_desconhecidos"] += 1
                continue
            if not incluir_longas and len(texto) > TAMANHO_CONTEXTO + 1:
                resumo["longas"] += 1
                continue
            assinatura_resposta = hashlib.sha256(normalizar_texto(resposta).encode("utf-8")).hexdigest()
            if parte == "validacao" and respostas_vistas and assinatura_resposta in respostas_vistas:
                resumo["validacao_vista_na_linguagem"] += 1
                continue
            par_normalizado = [normalizar_texto(pergunta), normalizar_texto(resposta)]
            chave = hashlib.sha256(json.dumps(par_normalizado, ensure_ascii=False).encode("utf-8")).hexdigest()
            cursor = conexao.execute(
                "INSERT OR IGNORE INTO exemplos VALUES (?, ?, ?, ?, ?, ?)",
                (chave, pergunta, resposta, fonte, parte, grupo),
            )
            if not cursor.rowcount:
                resumo["repetidas"] += 1
                continue
            caracteres.update(texto)
            resumo["selecionadas"] += 1
            if len(texto) > TAMANHO_CONTEXTO + 1:
                resumo["longas_incluidas"] += 1
            resumo[parte] += 1
            if resumo["selecionadas"] % 5000 == 0:
                conexao.commit()
            # Zero permite percorrer todos os exemplos disponiveis daquela fonte
            if limite and resumo["selecionadas"] >= limite:
                break
        conexao.commit()
        resumos[fonte] = resumo
        print(f"  Selecionadas: {resumo['selecionadas']} | longas: {resumo['longas']}", flush=True)
    conexao.execute("CREATE INDEX partes_fontes ON exemplos (parte, fonte)")
    conexao.commit()
    return caracteres, resumos


# Codifica somente um bloco por vez e registra os caminhos, contagens e assinaturas dos arquivos
def gravar_blocos(conexao, pasta: Path, tokenizador: TokenizadorCaracteres, fontes: dict, tamanho_bloco: int) -> dict:
    partes = {}
    for parte in ("treino", "validacao"):
        destino = pasta / parte
        destino.mkdir()
        resumo_parte = {"total": 0, "fontes": {}}
        for fonte in fontes:
            cursor = conexao.execute(
                "SELECT pergunta, resposta, grupo FROM exemplos WHERE parte = ? AND fonte = ? ORDER BY rowid",
                (parte, fonte),
            )
            blocos = []
            quantidade = 0
            while registros := cursor.fetchmany(tamanho_bloco):
                pares = [(pergunta, resposta) for pergunta, resposta, _ in registros]
                exemplos = codificar_conversas(pares, tokenizador)
                relativo = Path(parte) / f"{fonte}_{len(blocos) + 1:05d}.jsonl"
                arquivo = pasta / relativo
                with arquivo.open("x", encoding="utf-8", newline="\n") as saida:
                    for exemplo, registro in zip(exemplos, registros):
                        exemplo["fonte"] = fonte
                        exemplo["grupo_pergunta"] = registro[2]
                        saida.write(json.dumps(exemplo, ensure_ascii=False, separators=(",", ":")) + "\n")
                with arquivo.open("rb") as entrada:
                    assinatura = hashlib.file_digest(entrada, "sha256").hexdigest()
                blocos.append({"arquivo": relativo.as_posix(), "exemplos": len(exemplos), "sha256": assinatura})
                quantidade += len(exemplos)
            resumo_parte["fontes"][fonte] = {"total": quantidade, "blocos": blocos}
            resumo_parte["total"] += quantidade
        partes[parte] = resumo_parte
    return partes


# Remove apenas a area temporaria criada nesta execucao, depois de verificar seu destino
def remover_area_temporaria(pasta: Path, destino: Path) -> None:
    if pasta.is_symlink() or pasta.resolve().parent != destino.resolve().parent:
        raise ValueError("A pasta temporaria esta fora do destino esperado")
    if not pasta.name.startswith(f".{destino.name}.preparando_"):
        raise ValueError("O nome da pasta temporaria nao corresponde a esta preparacao")
    shutil.rmtree(pasta)


# Coordena as passagens e publica a pasta somente quando todos os blocos estao completos
def preparar_em_blocos(fontes: dict[str, Path], destino: Path, tamanho_bloco: int = EXEMPLOS_POR_BLOCO,
                      limite: int = 0, fracao_treino: float = FRACAO_TREINO, semente: int = SEMENTE,
                      referencia: Path | None = None, incluir_longas: bool = False,
                      respostas_vistas: set[str] | None = None) -> dict:
    destino = destino.resolve()
    if destino.exists():
        raise ValueError(f"A pasta ja existe, escolha outra com --saida: {destino}")
    if tamanho_bloco < 1 or limite < 0 or not 0 < fracao_treino < 1:
        raise ValueError("Use bloco positivo, limite maior ou igual a zero e fracao de treino entre 0 e 1")
    if not fontes or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", fonte) for fonte in fontes):
        raise ValueError("Os nomes das fontes devem usar letras sem acentos, numeros ou sublinhados")
    vocabulario_referencia = None
    if referencia is not None:
        dados_referencia = json.loads(referencia.read_text(encoding="utf-8"))
        vocabulario_referencia = dados_referencia["vocabulario"]
        if not isinstance(vocabulario_referencia, list) or not vocabulario_referencia or any(
            not isinstance(c, str) or len(c) != 1 for c in vocabulario_referencia
        ) or vocabulario_referencia != sorted(set(vocabulario_referencia)):
            raise ValueError("O vocabulario de referencia precisa conter caracteres unicos em ordem fixa")
        esperados = {"usuario": MARCADOR_USUARIO, "resposta": MARCADOR_RESPOSTA, "fim": MARCADOR_FIM}
        if dados_referencia["marcadores"] != esperados or not set("".join(esperados.values())) <= set(vocabulario_referencia):
            raise ValueError("Os marcadores nao correspondem aos pesos da etapa de linguagem")
    if incluir_longas and vocabulario_referencia is None:
        raise ValueError("Informe o vocabulario fixo para preparar o ajuste com conversas longas")
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporaria = destino.with_name(f".{destino.name}.preparando_{uuid.uuid4().hex}")
    temporaria.mkdir()
    conexao = None
    try:
        indice = temporaria / "indice.sqlite"
        conexao = sqlite3.connect(indice)
        # Este catalogo e temporario e reconstruivel, sem alterar as fontes originais
        conexao.execute("PRAGMA journal_mode = OFF")
        conexao.execute("PRAGMA synchronous = OFF")
        # Os pequenos indices de ordenacao ficam na RAM em vez de usar a pasta temporaria do disco C
        conexao.execute("PRAGMA temp_store = MEMORY")
        caracteres, resumos = catalogar_conversas(fontes, conexao, limite, fracao_treino, semente,
                                                 set(vocabulario_referencia) if vocabulario_referencia else None,
                                                 incluir_longas, respostas_vistas)
        # Uma preparacao muito pequena pode nao gerar exemplos nas duas partes pelo hash
        for parte in ("treino", "validacao"):
            if not sum(resumo[parte] for resumo in resumos.values()):
                raise ValueError(f"A parte {parte} ficou vazia, aumente a selecao ou ajuste a semente")
        tokenizador = TokenizadorCaracteres("".join(vocabulario_referencia or sorted(caracteres)))
        marcadores = {"usuario": MARCADOR_USUARIO, "resposta": MARCADOR_RESPOSTA, "fim": MARCADOR_FIM}
        vocabulario = {"vocabulario": list(tokenizador.caracteres), "marcadores": marcadores}
        (temporaria / "vocabulario.json").write_text(json.dumps(vocabulario, ensure_ascii=False), encoding="utf-8")
        partes = gravar_blocos(conexao, temporaria, tokenizador, fontes, tamanho_bloco)
        manifesto = {
            "formato": "conversas_blocos_v1", "arquivo_vocabulario": "vocabulario.json",
            "tamanho_vocabulario": tokenizador.tamanho_vocabulario, "tamanho_contexto": TAMANHO_CONTEXTO,
            "exemplos_por_bloco": tamanho_bloco, "limite_por_fonte": limite,
            "semente": semente, "fracao_treino": fracao_treino,
            "divisao": "sha256_pergunta_normalizada_v1", "fontes": resumos, "partes": partes,
            "esquema_exemplos": "conversas_completas_v2" if incluir_longas else "conversas_curtas_v1",
            "vocabulario_fixo": referencia is not None,
        }
        (temporaria / "manifesto.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")
        conexao.close()
        conexao = None
        indice.unlink()
        if temporaria.resolve().parent != destino.parent or destino.exists():
            raise ValueError("O destino mudou durante a preparacao")
        temporaria.rename(destino)
        return manifesto
    except BaseException:
        if conexao is not None:
            conexao.close()
        if temporaria.exists():
            remover_area_temporaria(temporaria, destino)
        raise


# Prepara todos os pares elegiveis por padrao e mostra o resumo sem iniciar treinamento
def principal() -> None:
    analisador_argumentos = argparse.ArgumentParser(description="Prepara conversas em blocos sem carregar o conjunto inteiro na RAM")
    analisador_argumentos.add_argument("--saida", type=Path, default=PASTA_BLOCOS)
    analisador_argumentos.add_argument("--tamanho-bloco", type=int, default=EXEMPLOS_POR_BLOCO)
    analisador_argumentos.add_argument("--limite", type=int, default=0, help="Pares por fonte, use zero para todos os elegiveis")
    analisador_argumentos.add_argument("--semente", type=int, default=SEMENTE)
    argumentos = analisador_argumentos.parse_args()
    try:
        manifesto = preparar_em_blocos(FONTES_CONVERSAS, argumentos.saida, argumentos.tamanho_bloco,
                                       argumentos.limite, semente=argumentos.semente)
    except (OSError, ValueError, sqlite3.Error) as erro:
        raise SystemExit(f"Nao foi possivel preparar os blocos: {erro}") from erro
    for fonte, resumo in manifesto["fontes"].items():
        print(f"{fonte}: {resumo['selecionadas']} selecionadas | treino: {resumo['treino']} | validacao: {resumo['validacao']}")
        print(f"  Longas: {resumo['longas']} | invalidas: {resumo['invalidas']} | repetidas: {resumo['repetidas']}")
    print(f"Treino: {manifesto['partes']['treino']['total']} exemplos")
    print(f"Validacao: {manifesto['partes']['validacao']['total']} exemplos")
    print(f"Vocabulario: {manifesto['tamanho_vocabulario']} caracteres")
    print(f"Blocos salvos em: {argumentos.saida.resolve()}")
    print("Preparacao concluida, dados prontos para o treinamento sem executa-lo agora")


if __name__ == "__main__":
    principal()
