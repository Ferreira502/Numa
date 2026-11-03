# Reune os exemplos anteriores e dialogos longos para treinar as novas posicoes do contexto
# Mantem cada conversa em sua divisao e preserva os testes ja reservados

import json
from numa.configuracao import CONTEXTO_CONVERSA_LONGA, RAIZ_PROJETO
from numa.experimentos.preparar_experimento import preparar_experimento


# Prepara a mistura com historicos completos de ate 512 caracteres sem alterar pesos
def principal() -> None:
    destino = RAIZ_PROJETO / "dados/processados/conversas_contexto_512"
    origem = RAIZ_PROJETO / "dados/brutos/contexto_512/dialogos_completos.jsonl"
    if destino.exists() or origem.exists():
        raise SystemExit("A preparacao de 512 ja existe, preserve os arquivos anteriores")
    fontes = [RAIZ_PROJETO / "dados/brutos/conversas_variadas/dialogos_completos.jsonl",
              RAIZ_PROJETO / "dados/brutos/bate_papo_natural/dialogos.jsonl",
              RAIZ_PROJETO / "dados/brutos/contexto_512/dialogos_longos.jsonl"]
    registros = []
    for fonte in fontes:
        registros.extend(json.loads(l) for l in fonte.read_text(encoding="utf-8").splitlines() if l.strip())
    with origem.open("x", encoding="utf-8") as arquivo:
        for registro in registros:
            arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
    resumo = preparar_experimento(origem, destino, CONTEXTO_CONVERSA_LONGA)
    # Usa os casos anteriores e o ultimo turno de cada dialogo longo para limitar o custo da geracao
    casos = [json.loads(l) for l in (destino / "testes_validacao.jsonl").read_text(encoding="utf-8").splitlines()]
    comprimentos = {r["id"]: len(r["trocas"]) for r in registros}
    selecionados = [c for c in casos if not c["conversa"].startswith("longo_") or
                    c["id"] == f"{c['conversa']}_{comprimentos[c['conversa']]}" ]
    with (destino / "validacao_geracao.jsonl").open("x", encoding="utf-8") as arquivo:
        for caso in selecionados:
            arquivo.write(json.dumps(caso, ensure_ascii=False) + "\n")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    print(f"Contexto: {CONTEXTO_CONVERSA_LONGA} | casos para geracao: {len(selecionados)}")
    print("Preparacao concluida, nenhum treinamento executado")


if __name__ == "__main__":
    principal()
