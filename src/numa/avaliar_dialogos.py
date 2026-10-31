# Gera respostas para casos reservados e salva uma comparacao reproduzivel para revisao humana
# As referencias ajudam a avaliar o comportamento, sem transformar correspondencia textual em acuracia

import argparse
import hashlib
import json
import math
from pathlib import Path

from numa.chat import carregar_modelo, gerar_resposta, torch
from numa.contexto import montar_contexto
from numa.configuracao import RAIZ_PROJETO


# Executa inferencia com um gerador local para comparar os mesmos casos antes e depois do treino
def avaliar_casos(modelo_path: Path, testes_path: Path, caracteres: int = 120, temperatura: float = 0.5,
                  semente: int = 42, limite: int = 0, historico_referencia: bool = False) -> dict:
    if caracteres < 1 or not math.isfinite(temperatura) or temperatura <= 0 or limite < 0:
        raise ValueError("Use caracteres positivos, temperatura positiva e limite nao negativo")
    modelo, tokenizador, marcadores = carregar_modelo(modelo_path)
    casos = [json.loads(linha) for linha in testes_path.read_text(encoding="utf-8").splitlines() if linha.strip()]
    resultados = []
    for caso in casos[:limite] if limite else casos:
        gerador = torch.Generator().manual_seed(semente)
        historico = list(caso["historico"]) if historico_referencia else []
        if not historico_referencia:
            for pergunta, _ in caso["historico"]:
                contexto = montar_contexto(pergunta, historico, marcadores, modelo.tamanho_contexto)
                resposta_anterior = gerar_resposta(modelo, tokenizador, contexto, marcadores, caracteres, temperatura, gerador)
                if resposta_anterior:
                    historico.append((pergunta, resposta_anterior))
        contexto = montar_contexto(caso["pergunta"], historico, marcadores, modelo.tamanho_contexto)
        resposta = gerar_resposta(modelo, tokenizador, contexto, marcadores, caracteres, temperatura, gerador)
        resultados.append({**caso, "historico_usado": historico, "resposta_gerada": resposta,
                           "trocas_retidas": contexto.count(marcadores["fim"])})
        print(f"\n{caso['id']} | {caso['tipo']}")
        for pergunta, anterior in historico:
            print(f"Historico: {pergunta} -> {anterior}")
        print(f"Pergunta: {caso['pergunta']}")
        print(f"Numa: {resposta}")
        print(f"Referencia para revisao: {caso['referencia']}", flush=True)
    return {"modelo": str(modelo_path.resolve()), "sha256_modelo": hashlib.sha256(modelo_path.read_bytes()).hexdigest(),
            "testes": str(testes_path.resolve()),
            "temperatura": temperatura, "semente": semente, "caracteres": caracteres, "casos": resultados,
            "modo_historico": "referencia" if historico_referencia else "gerado_pela_numa",
            "avaliacao": "Revisao humana de pertinencia, continuidade e mudanca de assunto"}


# Recebe os arquivos e produz um relatorio sem modificar os pesos ou iniciar treinamento
def principal() -> None:
    analisador = argparse.ArgumentParser(description="Testa dialogos reservados sem treinar")
    analisador.add_argument("--modelo", type=Path, default=RAIZ_PROJETO / "modelos/numa_conversas_historico_256.pt")
    analisador.add_argument("--testes", type=Path, default=RAIZ_PROJETO / "dados/processados/conversas_revisadas_256/testes.jsonl")
    analisador.add_argument("--saida", type=Path)
    analisador.add_argument("--caracteres", type=int, default=120)
    analisador.add_argument("--temperatura", type=float, default=0.5)
    analisador.add_argument("--semente", type=int, default=42)
    analisador.add_argument("--limite", type=int, default=0)
    analisador.add_argument("--historico-referencia", action="store_true", help="Usa respostas corretas no historico em vez de gera-las")
    argumentos = analisador.parse_args()
    if argumentos.saida and argumentos.saida.exists():
        analisador.error("Escolha outra saida para preservar o relatorio anterior")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    try:
        relatorio = avaliar_casos(argumentos.modelo, argumentos.testes, argumentos.caracteres,
                                  argumentos.temperatura, argumentos.semente, argumentos.limite, argumentos.historico_referencia)
        if argumentos.saida:
            argumentos.saida.parent.mkdir(parents=True, exist_ok=True)
            with argumentos.saida.open("x", encoding="utf-8") as saida:
                json.dump(relatorio, saida, ensure_ascii=False, indent=2)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as erro:
        raise SystemExit(f"Nao foi possivel avaliar os dialogos: {erro}") from erro


if __name__ == "__main__":
    principal()
