# Seleciona pares de pergunta e resposta e separa conversas completas para treino e validacao
# Salva IDs e o inicio de cada resposta para o futuro treinamento focado nas respostas

import argparse
import json
import random
import re
from pathlib import Path

from numa.configuracao import (
    ARQUIVO_DADOS_PROCESSADOS,
    FRACAO_TREINO,
    LIMITE_CONVERSAS,
    MARCADOR_FIM,
    MARCADOR_RESPOSTA,
    MARCADOR_USUARIO,
    PASTA_CONVERSAS,
    SEMENTE,
)
from numa.tokenizer import TokenizadorCaracteres


# Le pares dos lotes TXT ate o limite, mantendo cada pergunta ligada a sua resposta
def ler_conversas(pasta: Path, limite: int) -> list[tuple[str, str]]:
    if limite < 2:
        raise ValueError("Selecione pelo menos duas conversas para separar treino e validacao")
    arquivos = sorted(pasta.glob("*.txt"))
    if not arquivos:
        raise ValueError(f"Nenhum arquivo TXT encontrado em {pasta}")

    conversas = []
    # Aceita o rotulo Usuario com ou sem acento nos dados originais
    # Uma linha vazia seguida desse rotulo marca o inicio de um novo par
    padrao = r"(?:\A|\n\n)Usu(?:a|\u00e1)rio: "
    for arquivo in arquivos:
        texto = arquivo.read_text(encoding="utf-8-sig").strip()
        if not texto:
            continue
        blocos = re.split(padrao, texto)
        if blocos[0] or len(blocos) < 2:
            raise ValueError(f"Formato de conversa nao reconhecido em {arquivo.name}")
        for bloco in blocos[1:]:
            # Rejeita divisores ambiguos para nao ligar uma resposta a pergunta errada
            if bloco.count(MARCADOR_RESPOSTA) != 1:
                raise ValueError(f"Par sem resposta ou com rotulos ambiguos em {arquivo.name}")
            pergunta, _, resposta = bloco.partition(MARCADOR_RESPOSTA)
            pergunta, resposta = pergunta.strip(), resposta.strip()
            if not pergunta or not resposta:
                raise ValueError(f"Pergunta ou resposta vazia em {arquivo.name}")
            if "<FIM>" in pergunta or "<FIM>" in resposta:
                raise ValueError(f"O marcador reservado <FIM> aparece no conteudo de {arquivo.name}")
            conversas.append((pergunta, resposta))
            if len(conversas) == limite:
                return conversas
    if not conversas:
        raise ValueError("Os arquivos nao contem conversas")
    return conversas


# Uniformiza espacos e maiusculas somente para identificar perguntas e pares repetidos
def normalizar_texto(texto: str) -> str:
    return " ".join(texto.split()).casefold()


# Codifica cada par e registra onde a resposta comeca, incluindo seu marcador de fim
def codificar_conversas(conversas: list[tuple[str, str]], tokenizador: TokenizadorCaracteres) -> list[dict]:
    exemplos = []
    for pergunta, resposta in conversas:
        prefixo = f"{MARCADOR_USUARIO}{pergunta}{MARCADOR_RESPOSTA}"
        texto = f"{prefixo}{resposta}{MARCADOR_FIM}"
        exemplos.append({
            "ids": tokenizador.codificar(texto),
            "inicio_resposta": len(prefixo),
        })
    return exemplos


# Remove pares repetidos e divide grupos de perguntas antes de criar os exemplos em IDs
def preparar_conversas(conversas: list[tuple[str, str]], fracao_treino: float = FRACAO_TREINO) -> dict:
    if not 0 < fracao_treino < 1:
        raise ValueError("A fracao de treino deve estar entre 0 e 1")
    grupos = {}
    pares_vistos = set()
    for pergunta, resposta in conversas:
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
        "treino": codificar_conversas(treino, tokenizador),
        "validacao": codificar_conversas(validacao, tokenizador),
    }


# Seleciona os lotes, prepara as conversas e grava um arquivo separado dos textos comuns
def principal() -> None:
    analisador_argumentos = argparse.ArgumentParser(description="Prepara pares completos de conversa para o Numa")
    analisador_argumentos.add_argument("--pasta", type=Path, default=PASTA_CONVERSAS)
    analisador_argumentos.add_argument("--limite", type=int, default=LIMITE_CONVERSAS)
    analisador_argumentos.add_argument("--saida", type=Path, default=ARQUIVO_DADOS_PROCESSADOS)
    argumentos = analisador_argumentos.parse_args()
    try:
        conversas = ler_conversas(argumentos.pasta, argumentos.limite)
        dados = preparar_conversas(conversas)
        argumentos.saida.parent.mkdir(parents=True, exist_ok=True)
        argumentos.saida.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    except (OSError, ValueError) as erro:
        raise SystemExit(f"Nao foi possivel preparar as conversas: {erro}") from erro

    print(f"Pares lidos: {dados['pares_lidos']}")
    print(f"Duplicados removidos: {dados['duplicados_removidos']}")
    print(f"Conversas de treino: {len(dados['treino'])}")
    print(f"Conversas de validacao: {len(dados['validacao'])}")
    print(f"Vocabulario: {len(dados['vocabulario'])} caracteres")
    print(f"Dados salvos em: {argumentos.saida}")
    print("Para treinar, execute: PYTHONPATH=src python -m numa.treinar")


if __name__ == "__main__":
    principal()
