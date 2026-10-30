# Treina o Numa para responder mensagens usando entropia cruzada, backpropagation e AdamW
# Retoma o ultimo treinamento compativel e salva os melhores pesos para o chat

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from numa.configuracao import (
    ARQUIVO_CHECKPOINT,
    ARQUIVO_DADOS_PROCESSADOS,
    DIMENSAO_EMBEDDING,
    DROPOUT,
    INTERVALO_AVALIACAO,
    LOTES_AVALIACAO,
    NUM_CABECAS,
    NUM_CAMADAS,
    PASSOS_TREINO,
    SEMENTE,
    TAMANHO_CONTEXTO,
    TAMANHO_LOTE,
    TAXA_APRENDIZADO,
)
from numa.modelo import ModeloNuma


# Sorteia trechos dentro de conversas e exclui perguntas e preenchimentos do erro
def criar_lote(exemplos: list[dict], gerador: torch.Generator | None = None):
    entrada = torch.zeros((TAMANHO_LOTE, TAMANHO_CONTEXTO), dtype=torch.long)
    alvos = torch.full_like(entrada, -100)
    indices = torch.randint(len(exemplos), (TAMANHO_LOTE,), generator=gerador)
    for linha, indice in enumerate(indices.tolist()):
        exemplo = exemplos[indice]
        ids = exemplo["ids"]
        inicio_resposta = exemplo["inicio_resposta"]
        # O ultimo alvo sempre pertence a resposta ou ao marcador de fim
        ultimo_alvo = torch.randint(inicio_resposta, len(ids), (1,), generator=gerador).item()
        fim = ultimo_alvo + 1
        inicio = max(0, fim - TAMANHO_CONTEXTO - 1)
        comprimento = fim - inicio - 1
        entrada[linha, :comprimento] = ids[inicio:fim - 1]
        alvos_trecho = ids[inicio + 1:fim].clone()
        # Os alvos estao deslocados em um caractere em relacao a entrada
        posicoes_alvos = torch.arange(inicio + 1, fim)
        alvos_trecho[posicoes_alvos < inicio_resposta] = -100
        alvos[linha, :comprimento] = alvos_trecho
    # O preenchimento fica no final e nao afeta posicoes anteriores pela mascara causal
    return entrada, alvos


# Converte uma lista de IDs em tensor e rejeita valores fora do vocabulario
def validar_ids(valores: list[int], tamanho_vocabulario: int) -> torch.Tensor:
    if not isinstance(valores, list) or not valores or any(type(valor) is not int for valor in valores):
        raise ValueError("Os IDs devem ser uma lista nao vazia de numeros inteiros")
    ids = torch.tensor(valores, dtype=torch.long)
    if ids.min().item() < 0 or ids.max().item() >= tamanho_vocabulario:
        raise ValueError("Os IDs nao correspondem ao vocabulario salvo")
    return ids


# Valida o formato, os marcadores e a posicao das respostas antes do treinamento
def carregar_conversas(dados: dict) -> tuple[list[dict], list[dict]]:
    if dados.get("formato") != "conversas_v1":
        raise ValueError("Execute primeiro numa.preparar_conversas")
    vocabulario = dados["vocabulario"]
    if not isinstance(vocabulario, list) or not vocabulario or any(
        not isinstance(caractere, str) or len(caractere) != 1 for caractere in vocabulario
    ) or vocabulario != sorted(set(vocabulario)):
        raise ValueError("O vocabulario deve conter caracteres unicos em ordem fixa")
    marcadores = dados["marcadores"]
    mapa = {caractere: indice for indice, caractere in enumerate(vocabulario)}
    for nome in ("usuario", "resposta", "fim"):
        if not isinstance(marcadores[nome], str) or not marcadores[nome]:
            raise ValueError("Os marcadores de conversa devem ser textos nao vazios")
    ids_usuario = [mapa[caractere] for caractere in marcadores["usuario"]]
    ids_resposta = [mapa[caractere] for caractere in marcadores["resposta"]]
    ids_fim = [mapa[caractere] for caractere in marcadores["fim"]]
    partes = []
    for nome in ("treino", "validacao"):
        if not isinstance(dados[nome], list) or not dados[nome]:
            raise ValueError(f"A parte {nome} nao possui conversas")
        exemplos = []
        for exemplo in dados[nome]:
            valores = exemplo["ids"]
            ids = validar_ids(valores, len(vocabulario))
            inicio = exemplo["inicio_resposta"]
            if type(inicio) is not int or not len(ids_usuario) + len(ids_resposta) < inicio < len(ids) - len(ids_fim):
                raise ValueError("O inicio da resposta esta fora da conversa")
            if valores[:len(ids_usuario)] != ids_usuario or valores[inicio - len(ids_resposta):inicio] != ids_resposta:
                raise ValueError("O inicio da resposta nao corresponde aos marcadores salvos")
            if valores[-len(ids_fim):] != ids_fim:
                raise ValueError("A conversa esta sem o marcador de fim")
            exemplos.append({"ids": ids, "inicio_resposta": inicio})
        partes.append(exemplos)
    return partes[0], partes[1]


# Compara as previsoes com os caracteres da resposta, ignorando os alvos marcados com -100
def calcular_erro(modelo: ModeloNuma, entrada: torch.Tensor, alvos: torch.Tensor):
    logits = modelo(entrada)
    # A entropia cruzada recebe logits diretamente, sem aplicar softmax antes
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), alvos.reshape(-1), ignore_index=-100)


# Avalia trechos fixos de treino e validacao sem dropout e sem alterar os pesos
@torch.no_grad()
def avaliar(modelo: ModeloNuma, treino: list[dict], validacao: list[dict]) -> dict:
    estava_treinando = modelo.training
    modelo.eval()
    resultados = {}
    gerador = torch.Generator().manual_seed(SEMENTE + 1)
    for nome, exemplos in (("treino", treino), ("validacao", validacao)):
        soma_erros = 0.0
        quantidade_alvos = 0
        for _ in range(LOTES_AVALIACAO):
            entrada, alvos = criar_lote(exemplos, gerador)
            quantidade = (alvos != -100).sum().item()
            soma_erros += calcular_erro(modelo, entrada, alvos).item() * quantidade
            quantidade_alvos += quantidade
        resultados[nome] = soma_erros / quantidade_alvos
    modelo.train(estava_treinando)
    return resultados


# Grava primeiro um arquivo temporario para evitar deixar um checkpoint incompleto
def salvar_checkpoint(caminho: Path, estado: dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    torch.save(estado, temporario)
    temporario.replace(caminho)


# Restaura os pesos e o AdamW somente quando o vocabulario, a arquitetura e os marcadores combinam
def retomar_treinamento(caminho: Path, modelo: ModeloNuma, otimizador, dados: dict, configuracao: dict) -> dict:
    estado = torch.load(caminho, map_location="cpu", weights_only=True)
    if estado.get("formato") != "conversas_v1":
        raise ValueError("O checkpoint nao e de conversas, use --reiniciar para criar o novo modelo")
    if estado["vocabulario"] != dados["vocabulario"]:
        raise ValueError("O vocabulario mudou, use --reiniciar para treinar com os novos IDs")
    if estado["configuracao_modelo"] != configuracao or estado["marcadores"] != dados["marcadores"]:
        raise ValueError("A arquitetura ou os marcadores mudaram, use --reiniciar")
    modelo.load_state_dict(estado["estado_modelo"])
    otimizador.load_state_dict(estado["estado_otimizador"])
    # Permite ajustar a taxa na configuracao sem descartar o historico do AdamW
    for grupo in otimizador.param_groups:
        grupo["lr"] = TAXA_APRENDIZADO
    if "estado_aleatorio" in estado:
        torch.set_rng_state(estado["estado_aleatorio"])
    return estado


# Prepara o treinamento, retoma o progresso disponivel e executa mais passos de aprendizado
def principal() -> None:
    analisador_argumentos = argparse.ArgumentParser(description="Treina o Numa para conversar")
    analisador_argumentos.add_argument("--dados", type=Path, default=ARQUIVO_DADOS_PROCESSADOS)
    analisador_argumentos.add_argument("--saida", type=Path, default=ARQUIVO_CHECKPOINT)
    analisador_argumentos.add_argument("--passos", type=int, default=PASSOS_TREINO)
    analisador_argumentos.add_argument("--reiniciar", action="store_true")
    argumentos = analisador_argumentos.parse_args()
    if argumentos.passos < 1:
        analisador_argumentos.error("A quantidade de passos deve ser positiva")
    torch.manual_seed(SEMENTE)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    try:
        conteudo = argumentos.dados.read_bytes()
        assinatura = hashlib.sha256(conteudo).hexdigest()
        dados = json.loads(conteudo.decode("utf-8"))
        treino, validacao = carregar_conversas(dados)
    except (OSError, ValueError, KeyError, TypeError) as erro:
        raise SystemExit(f"Confira ou prepare novamente as conversas: {erro}") from erro

    configuracao = {
        "tamanho_vocabulario": len(dados["vocabulario"]),
        "tamanho_contexto": TAMANHO_CONTEXTO,
        "dimensao_embedding": DIMENSAO_EMBEDDING,
        "num_cabecas": NUM_CABECAS,
        "num_camadas": NUM_CAMADAS,
        "dropout": DROPOUT,
    }
    modelo = ModeloNuma(**configuracao)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=TAXA_APRENDIZADO, weight_decay=0.01)
    caminho_ultimo = argumentos.saida.with_name(argumentos.saida.stem + "_ultimo" + argumentos.saida.suffix)
    caminho_retomada = caminho_ultimo if caminho_ultimo.exists() else argumentos.saida
    estado_anterior = {}
    if not argumentos.reiniciar and caminho_retomada.exists():
        try:
            estado_anterior = retomar_treinamento(caminho_retomada, modelo, otimizador, dados, configuracao)
        except (OSError, ValueError, KeyError, RuntimeError) as erro:
            raise SystemExit(f"Nao foi possivel continuar o treinamento: {erro}") from erro
        print(f"Retomando os pesos do passo {estado_anterior['passo']}", flush=True)
    else:
        print("Iniciando um novo treinamento de conversas", flush=True)

    passo_inicial = estado_anterior.get("passo", 0)
    melhor_erro = float("inf")
    melhor_passo = passo_inicial
    # Se os dados mudaram, os erros de validacao antigos nao sao comparaveis
    if estado_anterior.get("assinatura_dados") == assinatura and argumentos.saida.exists():
        melhor_erro = estado_anterior.get("melhor_erro_validacao", estado_anterior["erro_validacao"])
        melhor_passo = estado_anterior.get("melhor_passo", passo_inicial)
    print(f"CPU | contexto: {TAMANHO_CONTEXTO} | lote: {TAMANHO_LOTE} | mais {argumentos.passos} passos", flush=True)
    print(f"Conversas: {len(treino)} de treino e {len(validacao)} de validacao", flush=True)

    for deslocamento in range(argumentos.passos + 1):
        passo = passo_inicial + deslocamento
        if deslocamento > 0:
            modelo.train()
            entrada, alvos = criar_lote(treino)
            otimizador.zero_grad(set_to_none=True)
            erro = calcular_erro(modelo, entrada, alvos)
            if not torch.isfinite(erro):
                raise SystemExit("Erro nao finito, confira a taxa de aprendizado e os dados")
            # Backpropagation calcula os gradientes e AdamW atualiza os parametros
            erro.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            otimizador.step()

        if deslocamento % INTERVALO_AVALIACAO == 0 or deslocamento == argumentos.passos:
            erros = avaliar(modelo, treino, validacao)
            print(f"Passo {passo:5d} | treino: {erros['treino']:.4f} | validacao: {erros['validacao']:.4f}", flush=True)
            melhorou = erros["validacao"] < melhor_erro
            if melhorou:
                melhor_erro = erros["validacao"]
                melhor_passo = passo
            estado = {
                "formato": "conversas_v1",
                "passo": passo,
                "estado_modelo": modelo.state_dict(),
                "estado_otimizador": otimizador.state_dict(),
                "estado_aleatorio": torch.get_rng_state(),
                "configuracao_modelo": configuracao,
                "vocabulario": dados["vocabulario"],
                "marcadores": dados["marcadores"],
                "assinatura_dados": assinatura,
                "erro_treino": erros["treino"],
                "erro_validacao": erros["validacao"],
                "melhor_erro_validacao": melhor_erro,
                "melhor_passo": melhor_passo,
            }
            if melhorou:
                salvar_checkpoint(argumentos.saida, estado)
            # O ultimo estado permite continuar mesmo se a validacao piorou depois do melhor passo
            salvar_checkpoint(caminho_ultimo, estado)

    print(f"Melhor validacao: {melhor_erro:.4f} no passo {melhor_passo}")
    print(f"Pesos usados pelo chat: {argumentos.saida}")
    print(f"Progresso para continuar treinando: {caminho_ultimo}")


if __name__ == "__main__":
    principal()
