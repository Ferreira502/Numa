# Prepara todas as fontes locais de conversa para aproveitar os pesos de linguagem existentes
# Mantem os IDs, registra descartes e inclui respostas longas para treino em janelas

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from numa.configuracao import FONTES_CONVERSAS, PASTA_AJUSTE_CONVERSAS, PASTA_LINGUAGEM, SEMENTE
from numa.experimentos.preparar_blocos import preparar_em_blocos
from numa.experimentos.preparar_conversas import normalizar_texto


# Identifica respostas reservadas ao treino de linguagem para evitar reutiliza-las como validacao nova
def carregar_respostas_vistas(caminho: Path) -> set[str]:
    respostas = set()
    with caminho.open(encoding="utf-8") as entrada:
        for linha in entrada:
            registro = json.loads(linha)
            respostas.add(hashlib.sha256(normalizar_texto(registro["texto"]).encode("utf-8")).hexdigest())
    return respostas


# Processa os pares originais uma unica vez sem misturar copias antigas dos dados processados
def principal() -> None:
    analisador = argparse.ArgumentParser(description="Prepara o ajuste de conversas com vocabulario de linguagem")
    analisador.add_argument("--saida", type=Path, default=PASTA_AJUSTE_CONVERSAS)
    analisador.add_argument("--base-linguagem", type=Path, default=PASTA_LINGUAGEM)
    analisador.add_argument("--limite", type=int, default=0, help="Pares por fonte, zero processa todos")
    argumentos = analisador.parse_args()
    try:
        base = argumentos.base_linguagem.resolve()
        descricao = json.loads((base / "manifesto.json").read_text(encoding="utf-8"))
        if descricao.get("formato") != "linguagem_blocos_v1":
            raise ValueError("Informe a pasta de dados da etapa de linguagem")
        vocabulario = (base / descricao["arquivo_vocabulario"]).resolve()
        if not vocabulario.is_relative_to(base):
            raise ValueError("O vocabulario esta fora da pasta de linguagem")
        respostas_vistas = carregar_respostas_vistas(base / "textos_treino.jsonl")
        manifesto = preparar_em_blocos(FONTES_CONVERSAS, argumentos.saida, limite=argumentos.limite,
                                       fracao_treino=descricao["fracao_treino"], semente=SEMENTE,
                                       referencia=vocabulario, incluir_longas=True, respostas_vistas=respostas_vistas)
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as erro:
        raise SystemExit(f"Nao foi possivel preparar o ajuste: {erro}") from erro
    for fonte, resumo in manifesto["fontes"].items():
        print(f"{fonte}: {resumo['selecionadas']} pares | longos incluidos: {resumo['longas_incluidas']}")
        print(f"  Repetidos: {resumo['repetidas']} | invalidos: {resumo['invalidas']} | caracteres desconhecidos: {resumo['caracteres_desconhecidos']}")
    print(f"Treino: {manifesto['partes']['treino']['total']} pares | validacao: {manifesto['partes']['validacao']['total']} pares")
    print("Dados anteriores e pesos preservados, nenhum treinamento executado")


if __name__ == "__main__":
    principal()
