# Amplia dialogos escritos para o projeto com respostas alternativas e mudancas de assunto
# Reserva entradas de teste antes da codificacao e preserva todos os pesos e conjuntos anteriores

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from numa.configuracao import ARQUIVO_MANIFESTO_AJUSTE, PASTA_LINGUAGEM, RAIZ_PROJETO, TAMANHO_CONTEXTO
from numa.contexto import montar_contexto
from numa.preparar_continuidade import preparar_continuidade
from numa.tokenizer import TokenizadorCaracteres


# Reune os temas anteriores e as alternativas escritas manualmente, sem inventar respostas por troca de palavras
def carregar_temas(originais: Path, ampliacao: Path) -> list[dict]:
    temas = {}
    with originais.open(encoding="utf-8") as entrada:
        for linha in entrada:
            r = json.loads(linha)
            for troca in r["trocas"]:
                troca["respostas"] = [troca.pop("resposta")]
            temas[r["grupo"]] = r
    vistos = set()
    with ampliacao.open(encoding="utf-8") as entrada:
        for linha in entrada:
            r = json.loads(linha)
            grupo = r["grupo"]
            if grupo in vistos:
                raise ValueError("Tema repetido na ampliacao")
            vistos.add(grupo)
            if "trocas" in r:
                if grupo in temas:
                    raise ValueError("Uma nova definicao nao pode substituir um tema anterior")
                temas[grupo] = r
            else:
                tema = temas[grupo]
                for i, nome in enumerate(("inicial", "seguimento")):
                    tema["trocas"][i]["perguntas"].append(r[f"pergunta_{nome}"])
                    tema["trocas"][i]["respostas"].append(r[f"resposta_{nome}"])
    for tema in temas.values():
        if len(tema["trocas"]) != 2:
            raise ValueError("Esta ampliacao usa exatamente duas trocas por tema")
        for troca in tema["trocas"]:
            if not 1 <= len(troca["perguntas"]) <= 4 or not 1 <= len(troca["respostas"]) <= 2:
                raise ValueError("Use ate quatro perguntas e duas respostas por troca")
            if any(not isinstance(t, str) or not t.strip() for t in troca["perguntas"] + troca["respostas"]):
                raise ValueError("As formulacoes precisam ser textos nao vazios")
    return list(temas.values())


# Combina formulacoes e respostas de uma troca, mantendo o significado definido no arquivo de dados
def opcoes(troca: dict):
    return list(itertools.product(troca["perguntas"], troca["respostas"]))


# Mostra a mesma pergunta de seguimento em assuntos diferentes e inclui formulacoes iniciais
def iterar_seguimentos(temas: list[dict]):
    for tema in temas:
        grupo = "familia:" + tema["grupo"]
        for primeira in opcoes(tema["trocas"][0]):
            yield grupo, [primeira]
            for segunda in opcoes(tema["trocas"][1]):
                yield grupo, [primeira, segunda]


# Ensina perguntas novas depois de saudacoes, mudancas de tema e mudancas apos uma continuacao
def iterar_mudancas(temas: list[dict]):
    identidade = next(t for t in temas if t["grupo"] == "numa_identidade")
    cumprimentos = [("oi", "Ola! Como posso ajudar?"), ("bom dia", "Bom dia! Como posso ajudar?")]
    for indice, tema in enumerate(temas):
        for cumprimento in cumprimentos:
            for pergunta in opcoes(tema["trocas"][0]):
                yield "saudacao:" + tema["grupo"], [cumprimento, pergunta]
        alvos = [identidade, temas[(indice + 1) % len(temas)], temas[(indice + 7) % len(temas)]]
        for alvo in alvos:
            if alvo["grupo"] == tema["grupo"]:
                continue
            grupo = "mudanca:" + tema["grupo"] + "->" + alvo["grupo"]
            for primeira in opcoes(tema["trocas"][0]):
                for segunda in opcoes(alvo["trocas"][0]):
                    yield grupo, [primeira, segunda]
        # As respostas permanecem curtas para que o tema anterior possa continuar visivel
        for r1 in tema["trocas"][0]["respostas"]:
            for r2 in tema["trocas"][1]["respostas"]:
                primeira = (tema["trocas"][0]["perguntas"][0], r1)
                segunda = (tema["trocas"][1]["perguntas"][0], r2)
                for nome in opcoes(identidade["trocas"][0]):
                    yield "mudanca_dupla:" + tema["grupo"], [primeira, segunda, nome]


# Le os cenarios reservados e identifica os prefixos que nao podem entrar no treino novo
def carregar_testes(caminho: Path, dados: dict) -> tuple[list[dict], set[str]]:
    tokenizador = TokenizadorCaracteres("".join(dados["vocabulario"]))
    testes = []
    reservados = set()
    with caminho.open(encoding="utf-8") as entrada:
        for linha in entrada:
            r = json.loads(linha)
            contexto = montar_contexto(r["pergunta"], r["historico"], dados["marcadores"], TAMANHO_CONTEXTO)
            ids = tokenizador.codificar(contexto)
            reservados.add(hashlib.sha256(json.dumps(ids).encode("ascii")).hexdigest())
            tokenizador.codificar(r["referencia"])
            testes.append(r)
    if len({r["id"] for r in testes}) != len(testes):
        raise ValueError("Os casos de teste precisam de identificadores unicos")
    return testes, reservados


# Prepara a ampliacao pequena e reutiliza os blocos de perguntas e respostas anteriores
def principal() -> None:
    analisador = argparse.ArgumentParser(description="Amplia e reserva testes de continuidade sem treinar")
    analisador.add_argument("--saida", type=Path, default=RAIZ_PROJETO / "dados/processados/conversas_revisadas_256")
    argumentos = analisador.parse_args()
    brutos = RAIZ_PROJETO / "dados/brutos"
    originais = brutos / "dialogos_continuos/conversas.jsonl"
    ampliacao = brutos / "dialogos_revisados/ampliacao.jsonl"
    arquivo_testes = brutos / "dialogos_revisados/testes.jsonl"
    try:
        temas = carregar_temas(originais, ampliacao)
        dados = json.loads((PASTA_LINGUAGEM / "vocabulario.json").read_text(encoding="utf-8"))
        testes, reservados = carregar_testes(arquivo_testes, dados)
        resumo = preparar_continuidade(argumentos.saida, ARQUIVO_MANIFESTO_AJUSTE, originais,
                                       brutos / "openassistant/original.messages.jsonl.gz",
                                       fontes_adicionais={"seguimentos_revisados": iterar_seguimentos(temas),
                                                          "mudancas_assunto": iterar_mudancas(temas)},
                                       prefixos_reservados=reservados)
        (argumentos.saida / "testes.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in testes), encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as erro:
        raise SystemExit(f"Nao foi possivel ampliar os dialogos: {erro}") from erro
    print(f"Temas escritos: {len(temas)} | casos reservados para teste: {len(testes)}")
    for fonte, info in resumo.items():
        print(f"{fonte}: treino {info['treino']} | validacao {info['validacao']} | continuacoes {info['continuacoes']}")
        print(f"  Descartes: {info['descartes']}")
    print("As quantidades incluem combinacoes das formulacoes, nao conversas humanas independentes")
    print("Pesos preservados, nenhum treinamento executado")


if __name__ == "__main__":
    principal()
