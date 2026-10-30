# Reune as fontes de conversa, seleciona pares curtos e separa treino e validacao
# Salva IDs e o inicio de cada resposta para treinar a resposta inteira e seu marcador de fim

import argparse
import json
import random
import re
from pathlib import Path

from numa.configuracao import (
    ARQUIVO_DADOS_PROCESSADOS,
    FRACAO_TREINO,
    FONTES_CONVERSAS,
    LIMITE_CONVERSAS,
    MARCADOR_FIM,
    MARCADOR_RESPOSTA,
    MARCADOR_USUARIO,
    SEMENTE,
    TAMANHO_CONTEXTO,
)
from numa.tokenizer import TokenizadorCaracteres


# Extrai uma pergunta e sua resposta sem aceitar rotulos ambiguos no mesmo bloco
def extrair_par(bloco: str):
    if bloco.count(MARCADOR_RESPOSTA) != 1:
        return None, None
    pergunta, _, resposta = bloco.partition(MARCADOR_RESPOSTA)
    return pergunta.strip(), resposta.strip()


# Le o TXT linha por linha e guarda somente uma conversa por vez
def iterar_pares_txt(arquivo: Path):
    padrao = re.compile(r"^Usu(?:a|\u00e1)rio: ")
    linhas = []
    iniciado = False
    anterior_vazia = True
    with arquivo.open(encoding="utf-8-sig") as entrada:
        for linha in entrada:
            inicio = padrao.match(linha)
            if inicio and (not iniciado or anterior_vazia):
                if iniciado:
                    yield extrair_par("".join(linhas))
                linhas = [linha[inicio.end():]]
                iniciado = True
            elif iniciado:
                linhas.append(linha)
            elif linha.strip():
                raise ValueError(f"Formato de conversa nao reconhecido em {arquivo.name}")
            anterior_vazia = not linha.strip()
    if iniciado:
        yield extrair_par("".join(linhas))


# Percorre uma pasta TXT ou um arquivo TXT ou JSONL sem carregar todas as fontes na memoria
def iterar_pares(caminho: Path):
    if caminho.is_dir():
        arquivos = sorted(caminho.glob("*.txt"))
    elif caminho.is_file():
        arquivos = [caminho]
    else:
        raise ValueError(f"Fonte de conversas nao encontrada: {caminho}")
    if not arquivos:
        raise ValueError(f"Nenhum arquivo TXT encontrado em {caminho}")

    for arquivo in arquivos:
        if arquivo.suffix.lower() == ".jsonl":
            # O JSONL do OpenAssistant ja preserva os pares e seus identificadores
            with arquivo.open(encoding="utf-8-sig") as entrada:
                for numero, linha in enumerate(entrada, 1):
                    if not linha.strip():
                        continue
                    try:
                        registro = json.loads(linha)
                    except ValueError as erro:
                        raise ValueError(f"JSON invalido em {arquivo.name}, linha {numero}") from erro
                    if not isinstance(registro, dict):
                        yield None, None
                    else:
                        yield registro.get("pergunta"), registro.get("resposta")
        elif arquivo.suffix.lower() == ".txt":
            yield from iterar_pares_txt(arquivo)
        else:
            raise ValueError(f"Use arquivos TXT ou JSONL: {arquivo.name}")


# Filtra pares completos e usa um conjunto comum para remover repeticoes entre as fontes
def ler_conversas(caminho: Path, limite: int, pares_vistos: set | None = None) -> tuple[list[tuple[str, str]], dict]:
    if limite < 2:
        raise ValueError("Selecione pelo menos duas conversas para separar treino e validacao")
    conversas = []
    vistos = pares_vistos if pares_vistos is not None else set()
    resumo = {"pares_examinados": 0, "longos_descartados": 0, "invalidos_descartados": 0, "repetidos_descartados": 0}
    for pergunta, resposta in iterar_pares(caminho):
        resumo["pares_examinados"] += 1
        if not isinstance(pergunta, str) or not isinstance(resposta, str):
            resumo["invalidos_descartados"] += 1
            continue
        pergunta, resposta = pergunta.strip(), resposta.strip()
        if not pergunta or not resposta or "<FIM>" in pergunta or "<FIM>" in resposta:
            resumo["invalidos_descartados"] += 1
            continue
        texto_formatado = f"{MARCADOR_USUARIO}{pergunta}{MARCADOR_RESPOSTA}{resposta}{MARCADOR_FIM}"
        if len(texto_formatado) > TAMANHO_CONTEXTO + 1:
            resumo["longos_descartados"] += 1
            continue
        chave = (normalizar_texto(pergunta), normalizar_texto(resposta))
        if chave in vistos:
            resumo["repetidos_descartados"] += 1
            continue
        vistos.add(chave)
        conversas.append((pergunta, resposta))
        if len(conversas) == limite:
            break
    resumo["selecionadas"] = len(conversas)
    return conversas, resumo


# Uniformiza espacos e maiusculas somente para identificar perguntas e pares repetidos
def normalizar_texto(texto: str) -> str:
    return " ".join(texto.split()).casefold()


# Codifica cada par e registra onde a resposta comeca, incluindo seu marcador de fim
def codificar_conversas(conversas: list[tuple[str, str]], tokenizador: TokenizadorCaracteres, origens: dict | None = None) -> list[dict]:
    exemplos = []
    for pergunta, resposta in conversas:
        prefixo = f"{MARCADOR_USUARIO}{pergunta}{MARCADOR_RESPOSTA}"
        texto = f"{prefixo}{resposta}{MARCADOR_FIM}"
        exemplos.append({
            "ids": tokenizador.codificar(texto),
            "inicio_resposta": len(prefixo),
            "fonte": origens.get((pergunta, resposta), "personalizada") if origens else "personalizada",
        })
    return exemplos


# Remove pares repetidos e divide grupos de perguntas antes de criar os exemplos em IDs
def preparar_conversas(conversas: list[tuple[str, str]], fracao_treino: float = FRACAO_TREINO, origens: dict | None = None) -> dict:
    if not 0 < fracao_treino < 1:
        raise ValueError("A fracao de treino deve estar entre 0 e 1")
    grupos = {}
    pares_vistos = set()
    for pergunta, resposta in conversas:
        if len(f"{MARCADOR_USUARIO}{pergunta}{MARCADOR_RESPOSTA}{resposta}{MARCADOR_FIM}") > TAMANHO_CONTEXTO + 1:
            raise ValueError("Uma conversa ultrapassa o contexto, selecione exemplos mais curtos")
        chave_pergunta = normalizar_texto(pergunta)
        chave_par = (chave_pergunta, normalizar_texto(resposta))
        if chave_par not in pares_vistos:
            pares_vistos.add(chave_par)
            grupos.setdefault(chave_pergunta, []).append((pergunta, resposta))
    if len(grupos) < 2:
        raise ValueError("Sao necessarias pelo menos duas perguntas diferentes")

    # Perguntas iguais com respostas alternativas ficam juntas na mesma parte
    chaves = list(grupos)
    random.Random(SEMENTE).shuffle(chaves)
    corte = min(len(chaves) - 1, max(1, int(len(chaves) * fracao_treino)))
    treino = [par for chave in chaves[:corte] for par in grupos[chave]]
    validacao = [par for chave in chaves[corte:] for par in grupos[chave]]

    # O vocabulario comum apenas define os IDs dos caracteres presentes nos exemplos
    # As respostas de validacao nao deverao ser usadas para atualizar os pesos
    caracteres = set()
    for pergunta, resposta in treino + validacao:
        caracteres.update(f"{MARCADOR_USUARIO}{pergunta}{MARCADOR_RESPOSTA}{resposta}{MARCADOR_FIM}")
    tokenizador = TokenizadorCaracteres("".join(sorted(caracteres)))
    return {
        "formato": "conversas_v1",
        "vocabulario": list(tokenizador.caracteres),
        "marcadores": {
            "usuario": MARCADOR_USUARIO,
            "resposta": MARCADOR_RESPOSTA,
            "fim": MARCADOR_FIM,
        },
        "semente": SEMENTE,
        "pares_lidos": len(conversas),
        "duplicados_removidos": len(conversas) - len(pares_vistos),
        "treino": codificar_conversas(treino, tokenizador, origens),
        "validacao": codificar_conversas(validacao, tokenizador, origens),
    }


# Combina as fontes em ordem fixa e informa a origem de cada par selecionado
def reunir_fontes(fontes: dict[str, Path], limite: int):
    conversas = []
    origens = {}
    resumos = {}
    vistos = set()
    for nome, caminho in fontes.items():
        selecionadas, resumo = ler_conversas(caminho, limite, vistos)
        resumo["caminho"] = str(caminho)
        resumos[nome] = resumo
        conversas.extend(selecionadas)
        origens.update({par: nome for par in selecionadas})
    if not conversas:
        raise ValueError("Nenhuma conversa completa cabe no contexto atual")
    return conversas, origens, resumos


# Prepara as fontes combinadas ou uma pasta especifica e grava os dados sem executar treinamento
def principal() -> None:
    analisador_argumentos = argparse.ArgumentParser(description="Prepara pares completos de conversa para o Numa")
    analisador_argumentos.add_argument("--pasta", type=Path, help="Usa somente esta pasta em vez das tres fontes configuradas")
    analisador_argumentos.add_argument("--limite", type=int, default=LIMITE_CONVERSAS, help="Maximo de pares selecionados por fonte")
    analisador_argumentos.add_argument("--saida", type=Path, default=ARQUIVO_DADOS_PROCESSADOS)
    argumentos = analisador_argumentos.parse_args()
    try:
        fontes = {"personalizada": argumentos.pasta} if argumentos.pasta else FONTES_CONVERSAS
        conversas, origens, resumos = reunir_fontes(fontes, argumentos.limite)
        dados = preparar_conversas(conversas, origens=origens)
        # Conta a distribuicao final depois da separacao por grupos de perguntas
        for nome, resumo in resumos.items():
            for parte in ("treino", "validacao"):
                resumo[parte] = sum(exemplo["fonte"] == nome for exemplo in dados[parte])
        dados["selecao"] = {"limite_por_fonte": argumentos.limite, "fontes": resumos}
        argumentos.saida.parent.mkdir(parents=True, exist_ok=True)
        argumentos.saida.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    except (OSError, ValueError) as erro:
        raise SystemExit(f"Nao foi possivel preparar as conversas: {erro}") from erro

    for nome, resumo in resumos.items():
        print(f"{nome}: {resumo['selecionadas']} selecionadas | treino: {resumo['treino']} | validacao: {resumo['validacao']}")
        print(f"  Examinadas: {resumo['pares_examinados']} | longas: {resumo['longos_descartados']} | invalidas: {resumo['invalidos_descartados']} | repetidas: {resumo['repetidos_descartados']}")
    print(f"Total selecionado: {dados['pares_lidos']}")
    print(f"Duplicados removidos: {dados['duplicados_removidos']}")
    print(f"Conversas de treino: {len(dados['treino'])}")
    print(f"Conversas de validacao: {len(dados['validacao'])}")
    print(f"Vocabulario: {len(dados['vocabulario'])} caracteres")
    print(f"Dados salvos em: {argumentos.saida}")
    print("Para treinar, execute: PYTHONPATH=src python -m numa.treinar")


if __name__ == "__main__":
    principal()
