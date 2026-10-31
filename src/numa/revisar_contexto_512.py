# Amplia exercicios de memoria contextual mantendo revisao e divisoes anteriores
# Reserva formulacoes diferentes e aceita somente respostas alternativas escritas explicitamente

import argparse
import json
from pathlib import Path

from numa.configuracao import RAIZ_PROJETO, PASTA_LINGUAGEM, CONTEXTO_CONVERSA_LONGA
from numa.preparar_continuidade import codificar_continuacao
from numa.preparar_experimento import preparar_experimento
from numa.tokenizer import TokenizadorCaracteres


# Define formas curtas corretas para os casos antigos de validacao sem mudar os casos de teste
def alternativas_antigas() -> dict:
    return {"longo_papo_v1": ["Redes neurais.", "Sobre redes neurais."],
            "longo_nome_v1": ["Bia.", "Seu nome e Bia."],
            "longo_python_v1": ["Organizar arquivos.", "Voce queria organizar arquivos."],
            "longo_listas_v1": ["Bia e Carlos.", "Os nomes eram Bia e Carlos."],
            "longo_variaveis_v1": ["20.", "O valor era 20.", "idade valia 20."],
            "longo_rede_v1": ["Fotos de caes.", "Voce queria usar fotos de caes."],
            "longo_comandos_v1": ["/limpar e /sair.", "Eu citei /limpar e /sair."],
            "longo_mudanca_v1": ["Python.", "Sobre Python."]}


# Combina valores e distancias sem misturar formulacoes de treino, validacao e teste
def criar_dialogos(roteiros: dict, tokenizador, marcadores: dict) -> tuple[list[dict], int]:
    registros = []
    descartados = 0
    intercalacoes = roteiros["intercalacoes"]
    for roteiro in roteiros["roteiros"]:
        for parte in ("treino", "validacao", "teste"):
            aberturas = roteiro["aberturas_treino"] if parte == "treino" else [roteiro["abertura_" + parte]]
            perguntas = roteiro["perguntas_treino"] if parte == "treino" else [roteiro["pergunta_" + parte]]
            distancias = range(5) if parte == "treino" else (2, 4)
            for indice_valor, valor in enumerate(roteiro[parte]):
                respostas = [r.format(valor=valor) for r in roteiro["respostas"]]
                for variante, abertura in enumerate(aberturas):
                    for distancia in distancias:
                        pergunta = perguntas[variante % len(perguntas)]
                        trocas = [[abertura.format(valor=valor), roteiro["resposta_inicial"].format(valor=valor)]]
                        trocas += [list(troca) for troca in intercalacoes[:distancia]]
                        trocas += [[pergunta, respostas[variante % len(respostas)]]]
                        exemplo, motivo = codificar_continuacao(trocas, tokenizador, marcadores, CONTEXTO_CONVERSA_LONGA)
                        if motivo:
                            descartados += 1
                            continue
                        registro = {"id": f"memoria_{roteiro['tipo']}_{parte}_{indice_valor}_{variante}_{distancia}",
                                    "assunto": "memoria_contextual", "divisao": parte, "trocas": trocas,
                                    "respostas_aceitas": [[r] for _, r in trocas]}
                        registro["respostas_aceitas"][-1] = respostas
                        registros.append(registro)
    return registros, descartados


# Separa as fontes para revisar exemplos antigos e sortear alvos que realmente ultrapassam 256 posicoes
def definir_fontes(registro: dict, tokenizador, marcadores: dict, novo: bool) -> None:
    fontes = []
    for indice in range(len(registro["trocas"])):
        exemplo, motivo = codificar_continuacao(registro["trocas"][:indice + 1], tokenizador, marcadores, CONTEXTO_CONVERSA_LONGA)
        if motivo:
            raise ValueError(f"{registro['id']}: {motivo}")
        fonte = "historico_longo" if exemplo["inicio_resposta"] > 256 else ("conversa_contextual" if novo else "revisao")
        fontes.append(fonte)
    registro["fontes_turnos"] = fontes


# Prepara um conjunto separado e uma avaliacao com peso igual para os grupos curtos e longos
def revisar_contexto(destino: Path) -> dict:
    brutos = RAIZ_PROJETO / "dados/brutos/contexto_512_revisado"
    origem = brutos / "dialogos_completos.jsonl"
    if destino.exists() or origem.exists():
        raise ValueError("A revisao ja existe, preserve seus arquivos e escolha uma nova etapa")
    dados = json.loads((PASTA_LINGUAGEM / "vocabulario.json").read_text(encoding="utf-8"))
    tokenizador = TokenizadorCaracteres("".join(dados["vocabulario"]))
    marcadores = dados["marcadores"]
    anterior = RAIZ_PROJETO / "dados/brutos/contexto_512/dialogos_completos.jsonl"
    registros = [json.loads(l) for l in anterior.read_text(encoding="utf-8").splitlines() if l.strip()]
    alternativas = alternativas_antigas()
    for registro in registros:
        if registro["id"] in alternativas:
            registro["respostas_aceitas"] = [[r] for _, r in registro["trocas"]]
            registro["respostas_aceitas"][-1] += alternativas[registro["id"]]
        definir_fontes(registro, tokenizador, marcadores, novo=False)
    roteiros = json.loads((brutos / "roteiros.json").read_text(encoding="utf-8"))
    novos, descartados = criar_dialogos(roteiros, tokenizador, marcadores)
    for registro in novos:
        definir_fontes(registro, tokenizador, marcadores, novo=True)
    registros += novos
    with origem.open("x", encoding="utf-8") as arquivo:
        for registro in registros:
            arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
    resumo = preparar_experimento(origem, destino, CONTEXTO_CONVERSA_LONGA)
    casos = [json.loads(l) for l in (destino / "testes_validacao.jsonl").read_text(encoding="utf-8").splitlines()]
    anteriores = RAIZ_PROJETO / "dados/processados/conversas_contexto_512/validacao_geracao.jsonl"
    ids_anteriores = {json.loads(l)["id"] for l in anteriores.read_text(encoding="utf-8").splitlines()}
    finais = {f"{r['id']}_{len(r['trocas'])}" for r in novos if r["divisao"] == "validacao"}
    selecionados = []
    for caso in casos:
        if caso["id"] not in ids_anteriores | finais:
            continue
        exemplo, _ = codificar_continuacao(caso["historico"] + [[caso["pergunta"], caso["referencia"]]], tokenizador, marcadores, CONTEXTO_CONVERSA_LONGA)
        caso["grupo_avaliacao"] = "historico_longo" if exemplo["inicio_resposta"] > 256 else "curto"
        selecionados.append(caso)
    with (destino / "validacao_geracao.jsonl").open("x", encoding="utf-8") as arquivo:
        for caso in selecionados:
            arquivo.write(json.dumps(caso, ensure_ascii=False) + "\n")
    return {"preparacao": resumo, "conversas_novas": len(novos), "combinacoes_longas_descartadas": descartados,
            "casos_geracao": len(selecionados)}


# Executa somente a preparacao, mantendo o treinamento sob controle do usuario
def principal() -> None:
    analisador = argparse.ArgumentParser(description="Revisa os dados e a avaliacao de memoria contextual")
    analisador.add_argument("--saida", type=Path, default=RAIZ_PROJETO / "dados/processados/contexto_512_revisado")
    argumentos = analisador.parse_args()
    try:
        resumo = revisar_contexto(argumentos.saida)
    except (OSError, ValueError, KeyError, TypeError) as erro:
        raise SystemExit(f"Nao foi possivel revisar os dialogos: {erro}") from erro
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    print("As quantidades novas incluem combinacoes de roteiros, nao conversas humanas independentes")
    print("Pesos e arquivos anteriores preservados, nenhum treino executado")


if __name__ == "__main__":
    principal()
