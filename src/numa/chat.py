# Abre um bate-papo com os pesos treinados e usa o mesmo formato dos exemplos de conversa
# Mantem as trocas recentes no contexto e encerra a resposta ao gerar o marcador de fim

import argparse
import math
import os
import shutil
import sys
import textwrap
import warnings
from pathlib import Path

# Este chat nao usa a integracao com NumPy, entao omite somente esse aviso opcional
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Failed to initialize NumPy: No module named 'numpy'.*", category=UserWarning)
    import torch

from numa.configuracao import ARQUIVO_CHECKPOINT, CARACTERES_GERADOS, TEMPERATURA
from numa.modelo import ModeloNuma
from numa.tokenizer import TokenizadorCaracteres
from numa.interno.contexto import montar_contexto


# Adiciona cores somente em terminais interativos que permitem essa formatacao
def colorir(texto: str, cor: str) -> str:
    if sys.stdout.isatty() and os.environ.get("TERM") != "dumb" and "NO_COLOR" not in os.environ:
        return f"\033[{cor}m{texto}\033[0m"
    return texto


# Limpa a tela interativa e apresenta o cabecalho de uma nova conversa
def exibir_cabecalho() -> None:
    if sys.stdout.isatty() and os.environ.get("TERM") != "dumb":
        print("\033[2J\033[H", end="", flush=True)
    print(colorir("NUMA", "1;36"))
    print(colorir("Chat experimental", "2"))
    print()
    print(colorir("/limpar  Nova conversa     /sair  Encerrar", "2"))
    print(colorir("-" * min(60, shutil.get_terminal_size().columns), "2"))


# Exibe a resposta com espacos e quebra de linhas ajustada a largura do terminal
def exibir_resposta(resposta: str) -> None:
    print()
    print(colorir("Numa", "1;36"))
    largura = max(20, min(80, shutil.get_terminal_size().columns - 2))
    for linha in resposta.splitlines():
        print(textwrap.fill(linha, width=largura, initial_indent="  ", subsequent_indent="  ", break_long_words=False, break_on_hyphens=False))


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


# Gera apenas a resposta usando temperatura e encerra no marcador de fim ou na proxima fala
@torch.inference_mode()
def gerar_resposta(
    modelo: ModeloNuma,
    tokenizador: TokenizadorCaracteres,
    contexto: str,
    marcadores: dict,
    quantidade: int = CARACTERES_GERADOS,
    temperatura: float = TEMPERATURA,
    gerador: torch.Generator | None = None,
) -> str:
    if not contexto or len(contexto) > modelo.tamanho_contexto:
        raise ValueError("O contexto deve ser nao vazio e caber no limite do modelo")
    if quantidade < 1 or not math.isfinite(temperatura) or temperatura <= 0:
        raise ValueError("A quantidade de caracteres e a temperatura devem ser positivas e finitas")
    modelo.eval()
    dispositivo = next(modelo.parameters()).device
    ids = torch.tensor([tokenizador.codificar(contexto)], dtype=torch.long, device=dispositivo)
    resposta = ""
    paradas = (marcadores["fim"], "\n" + marcadores["usuario"])
    for _ in range(quantidade):
        # A janela desliza quando a pergunta e a resposta ultrapassam o limite de contexto
        logits = modelo(ids[:, -modelo.tamanho_contexto:])[:, -1, :]
        probabilidades = torch.softmax(logits / temperatura, dim=-1)
        # O gerador local fica na CPU para comparar os mesmos sorteios entre avaliacoes
        amostra = probabilidades.cpu() if gerador is not None else probabilidades
        proximo = torch.multinomial(amostra, num_samples=1, generator=gerador).to(dispositivo)
        ids = torch.cat((ids, proximo), dim=1)
        resposta += tokenizador.decodificar([proximo.item()])
        for parada in paradas:
            if resposta.endswith(parada):
                return resposta[:-len(parada)].strip()
    return resposta.strip()


# Carrega os pesos e recebe mensagens, mantendo o historico enquanto o chat estiver aberto
def principal(argumentos_cli: list[str] | None = None) -> None:
    analisador_argumentos = argparse.ArgumentParser(description="Conversa com o modelo Numa treinado")
    analisador_argumentos.add_argument("--modelo", type=Path, default=ARQUIVO_CHECKPOINT)
    analisador_argumentos.add_argument("--caracteres", type=int, default=CARACTERES_GERADOS)
    analisador_argumentos.add_argument("--temperatura", type=float, default=TEMPERATURA)
    argumentos = analisador_argumentos.parse_args(argumentos_cli)
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
    exibir_cabecalho()
    while True:
        try:
            print()
            pergunta = input(colorir("Voce > ", "1;32")).strip()
            if pergunta == "/sair":
                print(colorir("\nConversa encerrada", "2"))
                break
            if pergunta == "/limpar":
                historico.clear()
                exibir_cabecalho()
                continue
            if not pergunta:
                continue
            contexto = montar_contexto(pergunta, historico, marcadores, modelo.tamanho_contexto)
            resposta = gerar_resposta(
                modelo, tokenizador, contexto, marcadores, argumentos.caracteres, argumentos.temperatura
            )
            if not resposta:
                print(colorir("\nO modelo encerrou a geracao sem produzir uma resposta", "33"))
                continue
            exibir_resposta(resposta)
            historico.append((pergunta, resposta))
            # Limita a memoria do terminal, alem do limite aplicado ao contexto de cada resposta
            historico = historico[-modelo.tamanho_contexto:]
        except ValueError as erro:
            print(colorir(f"\nNao foi possivel responder: {erro}", "33"))
        except (EOFError, KeyboardInterrupt):
            print()
            break


if __name__ == "__main__":
    principal()
