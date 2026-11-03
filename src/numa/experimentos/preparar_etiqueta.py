# Prepara agradecimentos, confirmacoes e despedidas em contextos variados
# Revisa os exemplos anteriores e mantem os casos de validacao e teste em suas divisoes

import argparse
import copy
import json
from pathlib import Path

from numa.configuracao import RAIZ_PROJETO, CONTEXTO_CONVERSA_LONGA
from numa.experimentos.preparar_experimento import preparar_experimento


# Reutiliza contextos completos da mesma divisao para ensinar como responder ao final de uma explicacao
def criar_etiqueta(anteriores: list[dict], roteiros: dict) -> list[dict]:
    mapa = {r["id"]: r for r in anteriores}
    bases = {
        "treino": ["python_t1", "listas_t1", "variaveis_t1", "rede_t1", "comandos_t1", "identidade_t1",
                   "natural_bate_papo_t2", "natural_identidade_t6"],
        "validacao": ["python_v1", "listas_v1", "identidade_v1", "natural_bate_papo_v1"],
        "teste": ["python_p1", "listas_p1", "identidade_p1", "natural_bate_papo_p1"],
    }
    # Este contexto inclui a sequencia que revelou a dificuldade com agradecimentos
    exemplo = {"id": "etiqueta_apresentacao_base", "assunto": "etiqueta_conversa", "divisao": "treino",
               "trocas": [["oi", "Ola! Como posso ajudar?"], ["qual seu nome?", "Eu sou a Numa."],
                          ["o que voce faz?", "Fui criada para estudar como redes neurais aprendem."]]}
    mapa[exemplo["id"]] = exemplo
    bases["treino"].append(exemplo["id"])
    novos = []
    for parte, identificadores in bases.items():
        for tipo, roteiro in roteiros.items():
            for numero, frase in enumerate(roteiro[parte]):
                # Tambem ensina a intencao quando a mensagem chega sem historico
                novos.append({"id": f"etiqueta_{parte}_{tipo}_{numero}_direta", "assunto": "etiqueta_conversa",
                              "divisao": parte, "trocas": [[frase, roteiro["resposta"]]],
                              "fontes_turnos": ["etiqueta"], "respostas_aceitas": [roteiro["aceitas"]]})
                for identificador in identificadores:
                    base = mapa[identificador]
                    if base["divisao"] != parte:
                        raise ValueError("O contexto nao pode mudar de divisao")
                    trocas = copy.deepcopy(base["trocas"])
                    aceitas = copy.deepcopy(base.get("respostas_aceitas", [[r] for _, r in trocas]))
                    fontes = list(base.get("fontes_turnos", ["revisao"] * len(trocas)))
                    trocas.append([frase, roteiro["resposta"]])
                    aceitas.append(roteiro["aceitas"])
                    fontes.append("etiqueta")
                    registro = {"id": f"etiqueta_{parte}_{tipo}_{numero}_{identificador}",
                                "assunto": "etiqueta_conversa", "divisao": parte,
                                "trocas": trocas, "fontes_turnos": fontes, "respostas_aceitas": aceitas}
                    novos.append(registro)
                    # Agradecer ou confirmar nao precisa encerrar a conversa
                    if parte == "treino" and tipo != "encerramento" and numero < 2 and identificador in bases["treino"][:4]:
                        continuacao = copy.deepcopy(registro)
                        continuacao["id"] += "_continua"
                        continuacao["trocas"].append(["agora quero saber seu nome", "Eu sou a Numa."])
                        continuacao["fontes_turnos"].append("etiqueta")
                        continuacao["respostas_aceitas"].append(["Eu sou a Numa.", "Eu sou a Numa, uma IA educacional."])
                        novos.append(continuacao)
    return novos


# Prepara um novo conjunto sem modificar os arquivos anteriores ou executar treinamento
def preparar_etiqueta(destino: Path) -> dict:
    brutos = RAIZ_PROJETO / "dados/brutos/etiqueta_conversa"
    origem = brutos / "dialogos_completos.jsonl"
    if destino.exists() or origem.exists():
        raise ValueError("A etapa de etiqueta ja existe, preserve seus arquivos")
    anterior = RAIZ_PROJETO / "dados/brutos/contexto_512_revisado/dialogos_completos.jsonl"
    registros = [json.loads(l) for l in anterior.read_text(encoding="utf-8").splitlines() if l.strip()]
    roteiros = json.loads((brutos / "roteiros.json").read_text(encoding="utf-8"))
    novos = criar_etiqueta(registros, roteiros)
    registros += novos
    with origem.open("x", encoding="utf-8") as arquivo:
        for registro in registros:
            arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
    resumo = preparar_experimento(origem, destino, CONTEXTO_CONVERSA_LONGA)
    anteriores_casos = RAIZ_PROJETO / "dados/processados/contexto_512_revisado/validacao_geracao.jsonl"
    antigos = {json.loads(l)["id"]: json.loads(l) for l in anteriores_casos.read_text(encoding="utf-8").splitlines()}
    finais = {f"{r['id']}_{len(r['trocas'])}" for r in novos if r["divisao"] == "validacao"}
    casos = [json.loads(l) for l in (destino / "testes_validacao.jsonl").read_text(encoding="utf-8").splitlines()]
    selecionados = []
    for caso in casos:
        if caso["id"] in antigos:
            caso["grupo_avaliacao"] = antigos[caso["id"]]["grupo_avaliacao"]
        elif caso["id"] in finais:
            caso["grupo_avaliacao"] = "etiqueta"
        else:
            continue
        selecionados.append(caso)
    with (destino / "validacao_geracao.jsonl").open("x", encoding="utf-8") as arquivo:
        for caso in selecionados:
            arquivo.write(json.dumps(caso, ensure_ascii=False) + "\n")
    return {"dados": resumo, "novas_conversas": len(novos), "casos_geracao": len(selecionados)}


# Informa os dados preparados para o usuario executar o ajuste no terminal
def principal() -> None:
    analisador = argparse.ArgumentParser(description="Prepara etiqueta de conversa com revisao dos exemplos anteriores")
    analisador.add_argument("--saida", type=Path, default=RAIZ_PROJETO / "dados/processados/etiqueta_conversa_512")
    argumentos = analisador.parse_args()
    try:
        resumo = preparar_etiqueta(argumentos.saida)
    except (OSError, ValueError, KeyError, TypeError) as erro:
        raise SystemExit(f"Nao foi possivel preparar a etiqueta: {erro}") from erro
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    print("Os dialogos incluem combinacoes de formulacoes escritas, nao conversas humanas independentes")
    print("Pesos e dados anteriores preservados, nenhum treinamento executado")


if __name__ == "__main__":
    principal()
