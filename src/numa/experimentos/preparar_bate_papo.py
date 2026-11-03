# Reune os dialogos naturais e os exemplos anteriores para revisar habilidades no mesmo treino
# Preserva as divisoes, o vocabulario e os arquivos de pesos existentes

import argparse
import json
from pathlib import Path

from numa.configuracao import RAIZ_PROJETO
from numa.experimentos.preparar_experimento import preparar_experimento


# Junta as conversas completas em um arquivo novo sem mover validacao ou teste para o treino
def preparar_bate_papo(destino: Path) -> dict:
    if destino.exists():
        raise ValueError("Escolha uma pasta nova para preservar os dados preparados")
    fontes = [RAIZ_PROJETO / "dados/brutos/conversas_variadas/dialogos_completos.jsonl",
              RAIZ_PROJETO / "dados/brutos/bate_papo_natural/dialogos.jsonl"]
    registros = []
    for fonte in fontes:
        registros.extend(json.loads(linha) for linha in fonte.read_text(encoding="utf-8").splitlines() if linha.strip())
    origem = destino.parent / (destino.name + "_origem.jsonl")
    if origem.exists():
        raise ValueError("O arquivo de origem combinado ja existe, preserve a preparacao anterior")
    origem.parent.mkdir(parents=True, exist_ok=True)
    with origem.open("x", encoding="utf-8") as arquivo:
        for registro in registros:
            arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
    resumo = preparar_experimento(origem, destino)
    return resumo


# Prepara somente a mistura pequena e informa sua cobertura sem executar treinamento
def principal() -> None:
    analisador = argparse.ArgumentParser(description="Prepara bate-papo com revisao dos exemplos anteriores")
    analisador.add_argument("--saida", type=Path, default=RAIZ_PROJETO / "dados/processados/bate_papo_revisao_256")
    argumentos = analisador.parse_args()
    try:
        resumo = preparar_bate_papo(argumentos.saida)
    except (OSError, ValueError, KeyError, TypeError) as erro:
        raise SystemExit(f"Nao foi possivel preparar o bate-papo: {erro}") from erro
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    print("Os exemplos anteriores sao revisados, mas isso nao garante ausencia de esquecimento")
    print("O contexto continua limitado a 256 caracteres, sem precisar limpar a cada troca")
    print("Pesos preservados, nenhum treinamento executado")


if __name__ == "__main__":
    principal()
