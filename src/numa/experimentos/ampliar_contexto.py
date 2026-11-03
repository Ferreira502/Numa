# Amplia somente a tabela de posicoes e cria um checkpoint de inicializacao sem treinar
# Copia todos os pesos existentes e preserva integralmente as posicoes ja aprendidas

import argparse
import hashlib
from pathlib import Path

from numa.chat import torch
from numa.configuracao import CONTEXTO_CONVERSA_LONGA, RAIZ_PROJETO
from numa.modelo import ModeloNuma


# Constroi um modelo maior no contexto e copia os parametros sem reutilizar o AdamW antigo
def ampliar_contexto(origem: Path, destino: Path, tamanho_contexto: int = CONTEXTO_CONVERSA_LONGA) -> dict:
    if destino.exists() or origem.resolve() == destino.resolve():
        raise ValueError("Escolha um checkpoint de destino novo")
    estado = torch.load(origem, map_location="cpu", weights_only=True)
    if estado.get("formato") != "conversas_v1":
        raise ValueError("A origem precisa ser um checkpoint de conversas")
    configuracao = dict(estado["configuracao_modelo"])
    anterior = configuracao["tamanho_contexto"]
    if type(tamanho_contexto) is not int or tamanho_contexto <= anterior:
        raise ValueError("O contexto novo precisa ser maior que o contexto original")
    configuracao["tamanho_contexto"] = tamanho_contexto
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        modelo = ModeloNuma(**configuracao)
    parametros = dict(estado["estado_modelo"])
    posicoes = modelo.embedding_posicao.weight.detach().clone()
    posicoes[:anterior].copy_(parametros["embedding_posicao.weight"])
    parametros["embedding_posicao.weight"] = posicoes
    modelo.load_state_dict(parametros, strict=True)
    ampliacao = {"origem": str(origem.resolve()), "sha256_origem": hashlib.sha256(origem.read_bytes()).hexdigest(),
                 "passo_origem": estado["passo"], "contexto_anterior": anterior, "contexto_novo": tamanho_contexto,
                 "posicoes_novas": "Inicializacao normal do PyTorch com semente 42, ainda sem treinamento"}
    novo = {"formato": "conversas_v1", "etapa": "inicializacao_contexto", "passo": 0,
            "configuracao_modelo": configuracao, "estado_modelo": modelo.state_dict(),
            "vocabulario": estado["vocabulario"], "marcadores": estado["marcadores"], "ampliacao_contexto": ampliacao}
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("xb") as arquivo:
        torch.save(novo, arquivo)
    return ampliacao


# Gera apenas a inicializacao para um futuro ajuste em outro arquivo de saida
def principal() -> None:
    analisador = argparse.ArgumentParser(description="Amplia o contexto preservando os pesos aprendidos")
    analisador.add_argument("--origem", type=Path, required=True)
    analisador.add_argument("--saida", type=Path, default=RAIZ_PROJETO / "modelos/numa_contexto_512_inicial.pt")
    argumentos = analisador.parse_args()
    try:
        info = ampliar_contexto(argumentos.origem, argumentos.saida)
    except (OSError, ValueError, KeyError, RuntimeError) as erro:
        raise SystemExit(f"Nao foi possivel ampliar o contexto: {erro}") from erro
    print(f"Contexto ampliado: {info['contexto_anterior']} -> {info['contexto_novo']}")
    print(f"Checkpoint de inicializacao: {argumentos.saida}")
    print("Pesos antigos preservados, novas posicoes ainda precisam de treino")


if __name__ == "__main__":
    principal()
