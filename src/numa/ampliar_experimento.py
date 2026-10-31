# Amplia os assuntos conhecidos com formulacoes escritas e continuacoes curtas
# Mantem as divisoes antigas, preserva os testes e prepara somente um novo conjunto de dados

import argparse
import json
from pathlib import Path

from numa.configuracao import RAIZ_PROJETO
from numa.preparar_experimento import preparar_experimento


# Combina somente as formulacoes de treino e as respostas definidas manualmente para cada assunto
def combinar_variacoes(temas: list[dict]) -> list[dict]:
    registros = []
    for tema in temas:
        for indice, pergunta in enumerate(tema["perguntas"]):
            for numero, seguimento in enumerate(tema["seguimentos"]):
                for variante, pergunta_seguinte in enumerate(seguimento["perguntas"]):
                    registros.append({"id": f"variacao_{tema['assunto']}_{indice}_{numero}_{variante}",
                                      "assunto": tema["assunto"], "divisao": "treino",
                                      "trocas": [[pergunta, tema["resposta"]],
                                                 [pergunta_seguinte, seguimento["resposta"]]]})
    return registros


# Reune os dados anteriores e novos em arquivos diferentes sem mover testes para treinamento
def ampliar_experimento(brutos: Path, destino: Path) -> dict:
    if destino.exists():
        raise ValueError("Escolha uma pasta de dados processados nova")
    origem = brutos / "dialogos_completos.jsonl"
    if origem.exists():
        raise ValueError("O arquivo combinado ja existe, preserve a preparacao anterior")
    anteriores = RAIZ_PROJETO / "dados/brutos/experimento_controlado/dialogos.jsonl"
    registros = [json.loads(l) for l in anteriores.read_text(encoding="utf-8").splitlines() if l.strip()]
    temas = json.loads((brutos / "variacoes.json").read_text(encoding="utf-8"))
    registros += combinar_variacoes(temas)
    registros += [json.loads(l) for l in (brutos / "adicionais.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    aceitas = json.loads((brutos / "respostas_aceitas.json").read_text(encoding="utf-8"))
    for registro in registros:
        if registro["divisao"] == "validacao":
            registro["respostas_aceitas"] = [list(dict.fromkeys([troca[1], *aceitas[registro["assunto"]][indice]]))
                                              for indice, troca in enumerate(registro["trocas"])]
    with origem.open("x", encoding="utf-8") as arquivo:
        for registro in registros:
            arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
    resumo = preparar_experimento(origem, destino)
    print("Variacoes combinadas nao sao conversas humanas independentes")
    return resumo


# Executa apenas a preparacao e informa a cobertura do novo conjunto
def principal() -> None:
    analisador = argparse.ArgumentParser(description="Amplia as conversas sem treinar ou mudar pesos")
    analisador.add_argument("--brutos", type=Path, default=RAIZ_PROJETO / "dados/brutos/conversas_variadas")
    analisador.add_argument("--saida", type=Path, default=RAIZ_PROJETO / "dados/processados/conversas_variadas_256")
    argumentos = analisador.parse_args()
    try:
        resumo = ampliar_experimento(argumentos.brutos, argumentos.saida)
    except (OSError, ValueError, KeyError, TypeError) as erro:
        raise SystemExit(f"Nao foi possivel ampliar as conversas: {erro}") from erro
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    print("Pesos anteriores preservados, nenhum treinamento executado")


if __name__ == "__main__":
    principal()
