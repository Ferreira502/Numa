# Abre um bate-papo com os pesos treinados e usa o mesmo formato dos exemplos de conversa
# Mantem as trocas recentes no contexto e encerra a resposta ao gerar o marcador de fim

import argparse
import math
from pathlib import Path

import torch

from numa.configuracao import ARQUIVO_CHECKPOINT, CARACTERES_GERADOS, TEMPERATURA
from numa.modelo import ModeloNuma
from numa.tokenizer import TokenizadorCaracteres


# Reconstroi o modelo, o tokenizador e os marcadores do treinamento de conversas
def carregar_modelo(caminho: Path = ARQUIVO_CHECKPOINT):
    estado_salvo = torch.load(caminho, map_location="cpu", weights_only=True)
    if estado_salvo.get("formato") != "conversas_v1":
        raise ValueError("Este arquivo nao foi treinado para conversas, execute numa.treinar")
    vocabulario = estado_salvo["vocabulario"]
    tokenizador = TokenizadorCaracteres("".join(vocabulario))
    if list(tokenizador.caracteres) != vocabulario:
        raise ValueError("O vocabulario salvo nao corresponde a ordem esperada dos IDs")
    modelo = ModeloNuma(**estado_salvo["configuracao_modelo"])
    if tokenizador.tamanho_vocabulario != estado_salvo["configuracao_modelo"]["tamanho_vocabulario"]:
        raise ValueError("O vocabulario nao corresponde ao modelo salvo")
    marcadores = estado_salvo["marcadores"]
    for nome in ("usuario", "resposta", "fim"):
        if not isinstance(marcadores[nome], str) or not marcadores[nome]:
            raise ValueError("O checkpoint esta sem marcadores de conversa validos")
        tokenizador.codificar(marcadores[nome])
    modelo.load_state_dict(estado_salvo["estado_modelo"])
    modelo.eval()
    return modelo, tokenizador, marcadores


# Formata a pergunta e inclui as trocas recentes completas que couberem no contexto
def montar_contexto(pergunta: str, historico: list[tuple[str, str]], marcadores: dict, tamanho_contexto: int) -> str:
    if not pergunta.strip():
        raise ValueError("Digite uma mensagem")
    contexto = f"{marcadores['usuario']}{pergunta}{marcadores['resposta']}"
    if len(contexto) > tamanho_contexto:
        limite = tamanho_contexto - len(marcadores["usuario"]) - len(marcadores["resposta"])
        raise ValueError(f"A mensagem deve ter no maximo {limite} caracteres para caber no contexto")
    for mensagem, resposta in reversed(historico):
        troca = f"{marcadores['usuario']}{mensagem}{marcadores['resposta']}{resposta}{marcadores['fim']}"
        if len(troca) + len(contexto) > tamanho_contexto:
            break
        contexto = troca + contexto
    return contexto


# Gera apenas a resposta usando temperatura e encerra no marcador de fim ou na proxima fala
@torch.inference_mode()
def gerar_resposta(
    modelo: ModeloNuma,
    tokenizador: TokenizadorCaracteres,
    contexto: str,
    marcadores: dict,
    quantidade: int = CARACTERES_GERADOS,
    temperatura: float = TEMPERATURA,
) -> str:
    if not contexto or len(contexto) > modelo.tamanho_contexto:
        raise ValueError("O contexto deve ser nao vazio e caber no limite do modelo")
    if quantidade < 1 or not math.isfinite(temperatura) or temperatura <= 0:
        raise ValueError("A quantidade de caracteres e a temperatura devem ser positivas e finitas")
    modelo.eval()
    ids = torch.tensor([tokenizador.codificar(contexto)], dtype=torch.long)
    resposta = ""
    paradas = (marcadores["fim"], "\n" + marcadores["usuario"])
    for _ in range(quantidade):
        # A janela desliza quando a pergunta e a resposta ultrapassam o limite de contexto
        logits = modelo(ids[:, -modelo.tamanho_contexto:])[:, -1, :]
        probabilidades = torch.softmax(logits / temperatura, dim=-1)
        proximo = torch.multinomial(probabilidades, num_samples=1)
        ids = torch.cat((ids, proximo), dim=1)
        resposta += tokenizador.decodificar([proximo.item()])
        for parada in paradas:
            if resposta.endswith(parada):
                return resposta[:-len(parada)].strip()
    return resposta.strip()


# Carrega os pesos e recebe mensagens, mantendo o historico enquanto o chat estiver aberto
def principal() -> None:
    analisador_argumentos = argparse.ArgumentParser(description="Conversa com o modelo Numa treinado")
    analisador_argumentos.add_argument("--modelo", type=Path, default=ARQUIVO_CHECKPOINT)
    analisador_argumentos.add_argument("--caracteres", type=int, default=CARACTERES_GERADOS)
    analisador_argumentos.add_argument("--temperatura", type=float, default=TEMPERATURA)
    argumentos = analisador_argumentos.parse_args()
    if argumentos.caracteres < 1:
        analisador_argumentos.error("A quantidade de caracteres deve ser positiva")
    if not math.isfinite(argumentos.temperatura) or argumentos.temperatura <= 0:
        analisador_argumentos.error("A temperatura deve ser positiva e finita")
    if not argumentos.modelo.exists():
        raise SystemExit("Prepare as conversas e execute numa.treinar antes de abrir o chat")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    try:
        modelo, tokenizador, marcadores = carregar_modelo(argumentos.modelo)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as erro:
        raise SystemExit(f"Nao foi possivel carregar o modelo: {erro}") from erro

    historico = []
    print("Numa pronta para conversar com os pesos treinados")
    print("Digite /sair para encerrar ou /limpar para iniciar uma nova conversa")
    while True:
        try:
            pergunta = input("Voce: ").strip()
            if pergunta == "/sair":
                break
            if pergunta == "/limpar":
                historico.clear()
                print("Historico limpo")
                continue
            if not pergunta:
                continue
            contexto = montar_contexto(pergunta, historico, marcadores, modelo.tamanho_contexto)
            resposta = gerar_resposta(
                modelo, tokenizador, contexto, marcadores, argumentos.caracteres, argumentos.temperatura
            )
            if not resposta:
                print("O modelo encerrou a geracao sem produzir uma resposta")
                continue
            print("Numa:", resposta)
            historico.append((pergunta, resposta))
            # Limita a memoria do terminal, alem do limite aplicado ao contexto de cada resposta
            historico = historico[-modelo.tamanho_contexto:]
        except ValueError as erro:
            print(f"Nao foi possivel responder: {erro}")
        except (EOFError, KeyboardInterrupt):
            print()
            break


if __name__ == "__main__":
    principal()
