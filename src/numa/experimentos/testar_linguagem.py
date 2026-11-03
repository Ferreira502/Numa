# Testa os melhores pesos da etapa de linguagem continuando um texto digitado
# Executa somente inferencia, sem alterar pesos ou iniciar treinamento

import argparse
import math
import warnings
from pathlib import Path

# A geracao usa tensores diretamente e nao precisa da integracao opcional com NumPy
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Failed to initialize NumPy: No module named 'numpy'.*", category=UserWarning)
    import torch

from numa.configuracao import ARQUIVO_CHECKPOINT_LINGUAGEM, CARACTERES_GERADOS, SEMENTE, TEMPERATURA
from numa.interno.dados_treinamento import validar_vocabulario
from numa.modelo import ModeloNuma
from numa.tokenizer import TokenizadorCaracteres


# Reconstroi a rede usando a arquitetura e os IDs exatos do checkpoint de linguagem
def carregar_modelo(caminho: Path, dispositivo: torch.device):
    estado = torch.load(caminho, map_location="cpu", weights_only=True)
    if estado.get("formato") != "linguagem_v1":
        raise ValueError("Escolha um checkpoint da etapa de linguagem")
    validar_vocabulario(estado)
    tokenizador = TokenizadorCaracteres("".join(estado["vocabulario"]))
    configuracao = estado["configuracao_modelo"]
    if tokenizador.tamanho_vocabulario != configuracao["tamanho_vocabulario"]:
        raise ValueError("O vocabulario nao corresponde a arquitetura salva")
    modelo = ModeloNuma(**configuracao)
    modelo.load_state_dict(estado["estado_modelo"])
    modelo.to(dispositivo).eval()
    resumo = {nome: estado.get(nome) for nome in ("passo", "erro_validacao")}
    return modelo, tokenizador, resumo


# Preve um caractere por vez e mantem apenas a janela de contexto mais recente
@torch.inference_mode()
def gerar_texto(modelo: ModeloNuma, tokenizador: TokenizadorCaracteres, texto: str,
                quantidade: int = CARACTERES_GERADOS, temperatura: float = TEMPERATURA,
                semente: int = SEMENTE) -> str:
    if not texto.strip():
        raise ValueError("Digite o inicio de um texto")
    if quantidade < 1 or not math.isfinite(temperatura) or temperatura < 0:
        raise ValueError("Use quantidade positiva e temperatura finita maior ou igual a zero")
    dispositivo = next(modelo.parameters()).device
    ids = torch.tensor([tokenizador.codificar(texto)[-modelo.tamanho_contexto:]], device=dispositivo, dtype=torch.long)
    gerador = torch.Generator(device=dispositivo).manual_seed(semente)
    continuacao = []
    estava_treinando = modelo.training
    modelo.eval()
    try:
        for _ in range(quantidade):
            logits = modelo(ids)[:, -1, :]
            if temperatura == 0:
                # Argmax escolhe o caractere mais provavel sem sortear
                proximo = logits.argmax(dim=-1, keepdim=True)
            else:
                # A amostragem segue as probabilidades aprendidas pela rede
                probabilidades = torch.softmax(logits / temperatura, dim=-1)
                proximo = torch.multinomial(probabilidades, num_samples=1, generator=gerador)
            continuacao.append(proximo.item())
            ids = torch.cat((ids, proximo), dim=1)[:, -modelo.tamanho_contexto:]
    finally:
        modelo.train(estava_treinando)
    return tokenizador.decodificar(continuacao)


# Permite testar um inicio pela linha de comando ou varios inicios em uma sessao interativa
def principal() -> None:
    analisador_argumentos = argparse.ArgumentParser(description="Avalia a geracao da primeira etapa, sem treinamento")
    analisador_argumentos.add_argument("--modelo", type=Path, default=ARQUIVO_CHECKPOINT_LINGUAGEM)
    analisador_argumentos.add_argument("--texto", help="Inicio opcional, sem este argumento abre o teste interativo")
    analisador_argumentos.add_argument("--caracteres", type=int, default=CARACTERES_GERADOS)
    analisador_argumentos.add_argument("--temperatura", type=float, default=TEMPERATURA, help="Zero escolhe sempre o caractere mais provavel")
    analisador_argumentos.add_argument("--semente", type=int, default=SEMENTE)
    analisador_argumentos.add_argument("--dispositivo", choices=("auto", "cpu", "cuda"), default="auto")
    argumentos = analisador_argumentos.parse_args()
    if argumentos.caracteres < 1 or not math.isfinite(argumentos.temperatura) or argumentos.temperatura < 0:
        analisador_argumentos.error("Use caracteres positivos e temperatura finita maior ou igual a zero")
    if argumentos.dispositivo == "cuda" and not torch.cuda.is_available():
        analisador_argumentos.error("CUDA nao esta disponivel nesta instalacao")
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu") if argumentos.dispositivo == "auto" else torch.device(argumentos.dispositivo)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    try:
        modelo, tokenizador, resumo = carregar_modelo(argumentos.modelo, dispositivo)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as erro:
        raise SystemExit(f"Nao foi possivel abrir os pesos de linguagem: {erro}") from erro
    print(f"Teste de linguagem | {dispositivo} | pesos do passo {resumo['passo']}")
    print("O modelo continua textos, o ajuste para conversas sera uma etapa posterior")
    if argumentos.texto is None:
        print("Digite o inicio de um texto ou /sair para encerrar")
    while True:
        try:
            texto = argumentos.texto if argumentos.texto is not None else input("\nInicio > ")
            if argumentos.texto is None and texto.strip() == "/sair":
                break
            continuacao = gerar_texto(modelo, tokenizador, texto, argumentos.caracteres,
                                     argumentos.temperatura, argumentos.semente)
            print(f"\nTexto gerado:\n{texto}{continuacao}")
        except ValueError as erro:
            if argumentos.texto is not None:
                raise SystemExit(f"Nao foi possivel gerar o texto: {erro}") from erro
            print(f"Nao foi possivel gerar o texto: {erro}")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if argumentos.texto is not None:
            break


if __name__ == "__main__":
    principal()
