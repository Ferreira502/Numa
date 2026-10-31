# Treina a linguagem ou as respostas usando entropia cruzada, backpropagation e AdamW
# Mantem dados e pesos separados por etapa e carrega somente os exemplos necessarios
# Avalia o aprendizado e permite parar quando a validacao deixa de melhorar

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from numa.configuracao import (
    ARQUIVO_CHECKPOINT,
    ARQUIVO_CHECKPOINT_LINGUAGEM,
    ARQUIVO_CHECKPOINT_CONTINUIDADE,
    ARQUIVO_MANIFESTO_AJUSTE,
    ARQUIVO_MANIFESTO_LINGUAGEM,
    ARQUIVO_MANIFESTO_CONTINUIDADE,
    DIMENSAO_EMBEDDING,
    DISPOSITIVO_TREINO,
    DROPOUT,
    INTERVALO_AVALIACAO,
    LOTES_AVALIACAO,
    NUM_CABECAS,
    NUM_CAMADAS,
    PACIENCIA_VALIDACAO,
    PASSOS_TREINO,
    SEMENTE,
    TAMANHO_CONTEXTO,
    TAMANHO_LOTE,
    TAXA_APRENDIZADO,
    TAXA_AJUSTE_CONVERSAS,
)
from numa.modelo import ModeloNuma
from numa.dados_treinamento import LeitorBlocos, ParteBlocos, carregar_conversas
from numa.dados_linguagem import LeitorLinguagem, ParteLinguagem
from numa.dados_mistos import LeitorMistura, ParteMistura
from numa.validar_geracao import carregar_casos_validacao, avaliar_geracao


# Escolhe CUDA quando disponivel ou permite selecionar a CPU explicitamente
def selecionar_dispositivo(preferencia: str = DISPOSITIVO_TREINO) -> torch.device:
    if preferencia not in ("auto", "cpu", "cuda"):
        raise ValueError("Use auto, cpu ou cuda como dispositivo")
    if preferencia == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA nao esta disponivel nesta instalacao, use --dispositivo cpu")
    if preferencia == "auto":
        preferencia = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(preferencia)


# Identifica as fontes para permitir sorteio e avaliacao sem favorecer o maior dataset
def agrupar_por_fonte(exemplos: list[dict]) -> dict[str, list[dict]]:
    if not exemplos:
        raise ValueError("Nao ha exemplos para formar um lote")
    grupos = {}
    for exemplo in exemplos:
        grupos.setdefault(exemplo.get("fonte", "personalizada"), []).append(exemplo)
    return grupos


# Sorteia a fonte e o exemplo, mascarando a pergunta somente na etapa de conversas
def criar_lote(exemplos, gerador: torch.Generator | None = None, fonte: str | None = None):
    entrada = torch.zeros((TAMANHO_LOTE, TAMANHO_CONTEXTO), dtype=torch.long)
    alvos = torch.full_like(entrada, -100)
    if isinstance(exemplos, (ParteBlocos, ParteLinguagem, ParteMistura)):
        selecionados = [exemplos.sortear(gerador, fonte) for _ in range(TAMANHO_LOTE)]
    else:
        grupos = agrupar_por_fonte(exemplos)
        fontes = [fonte] if fonte is not None else sorted(grupos)
        fontes_sorteadas = torch.randint(len(fontes), (TAMANHO_LOTE,), generator=gerador)
        selecionados = []
        for indice_fonte in fontes_sorteadas.tolist():
            grupo = grupos[fontes[indice_fonte]]
            indice = torch.randint(len(grupo), (1,), generator=gerador).item()
            selecionados.append(grupo[indice])
    for linha, exemplo in enumerate(selecionados):
        ids = exemplo["ids"]
        inicio_resposta = exemplo["inicio_resposta"]
        if len(ids) > TAMANHO_CONTEXTO + 1:
            raise ValueError("O exemplo nao cabe no contexto, prepare novamente os dados")
        comprimento = len(ids) - 1
        entrada[linha, :comprimento] = ids[:-1]
        alvos_trecho = ids[1:].clone()
        # O primeiro alvo da resposta fica uma posicao antes de seu indice no texto
        alvos_trecho[:inicio_resposta - 1] = -100
        alvos[linha, :comprimento] = alvos_trecho
    # O preenchimento fica no final e nao afeta posicoes anteriores pela mascara causal
    return entrada, alvos


# Abre o manifesto sem carregar todos os blocos ou aceita o JSON dos experimentos anteriores
def carregar_dados(caminho: Path):
    if caminho.is_dir():
        caminho = caminho / "manifesto.json"
    conteudo = caminho.read_bytes()
    descricao = json.loads(conteudo.decode("utf-8"))
    if descricao.get("formato") == "conversas_mistas_v1":
        leitor = LeitorMistura(caminho)
        return leitor.dados, ParteMistura(leitor, "treino"), ParteMistura(leitor, "validacao"), leitor.assinatura, leitor
    if descricao.get("formato") == "linguagem_blocos_v1":
        leitor = LeitorLinguagem(caminho)
        return leitor.dados, ParteLinguagem(leitor, "treino"), ParteLinguagem(leitor, "validacao"), leitor.assinatura, leitor
    if descricao.get("formato") == "conversas_blocos_v1":
        leitor = LeitorBlocos(caminho)
        return leitor.dados, ParteBlocos(leitor, "treino"), ParteBlocos(leitor, "validacao"), leitor.assinatura, leitor
    treino, validacao = carregar_conversas(descricao)
    return descricao, treino, validacao, hashlib.sha256(conteudo).hexdigest(), None


# Calcula a media do erro dos alvos validos e depois a media entre exemplos do lote
def calcular_erro(modelo: ModeloNuma, entrada: torch.Tensor, alvos: torch.Tensor):
    # Os exemplos ficam na RAM e somente o lote atual acompanha o modelo na GPU
    dispositivo = next(modelo.parameters()).device
    entrada = entrada.to(dispositivo)
    alvos = alvos.to(dispositivo)
    logits = modelo(entrada)
    # A entropia cruzada recebe logits diretamente, sem aplicar softmax antes
    erros = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), alvos.reshape(-1), ignore_index=-100, reduction="none"
    ).reshape_as(alvos)
    quantidades = (alvos != -100).sum(dim=1)
    if (quantidades == 0).any():
        raise ValueError("Cada exemplo precisa conter pelo menos um alvo")
    # Uma resposta longa tem o mesmo peso que uma curta no calculo final
    return (erros.sum(dim=1) / quantidades).mean()


# Avalia cada fonte separadamente e combina as medias com o mesmo peso por fonte
@torch.no_grad()
def avaliar(modelo: ModeloNuma, treino, validacao) -> dict:
    estava_treinando = modelo.training
    modelo.eval()
    resultados = {"fontes": {}}
    gerador = torch.Generator().manual_seed(SEMENTE + 1)
    for nome, exemplos in (("treino", treino), ("validacao", validacao)):
        erros_fontes = {}
        grupos = {fonte: exemplos for fonte in exemplos.fontes} if isinstance(exemplos, (ParteBlocos, ParteLinguagem, ParteMistura)) else agrupar_por_fonte(exemplos)
        for fonte, grupo in sorted(grupos.items()):
            soma_erros = 0.0
            for _ in range(LOTES_AVALIACAO):
                entrada, alvos = criar_lote(grupo, gerador, fonte=fonte)
                soma_erros += calcular_erro(modelo, entrada, alvos).item()
            erros_fontes[fonte] = soma_erros / LOTES_AVALIACAO
        resultados["fontes"][nome] = erros_fontes
        resultados[nome] = sum(erros_fontes.values()) / len(erros_fontes)
    modelo.train(estava_treinando)
    return resultados


# Grava primeiro um arquivo temporario para evitar deixar um checkpoint incompleto
def salvar_checkpoint(caminho: Path, estado: dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    torch.save(estado, temporario)
    temporario.replace(caminho)


# Restaura os pesos e o AdamW somente quando o vocabulario, a arquitetura e os marcadores combinam
def retomar_treinamento(caminho: Path, modelo: ModeloNuma, otimizador, dados: dict, configuracao: dict,
                        taxa_aprendizado: float = TAXA_APRENDIZADO) -> dict:
    estado = torch.load(caminho, map_location="cpu", weights_only=True)
    if estado.get("formato") != dados.get("formato_modelo", "conversas_v1"):
        raise ValueError("O checkpoint pertence a outra etapa, escolha outro caminho com --saida")
    if estado["vocabulario"] != dados["vocabulario"]:
        raise ValueError("O vocabulario mudou, use --reiniciar para treinar com os novos IDs")
    if estado["configuracao_modelo"] != configuracao or estado["marcadores"] != dados["marcadores"]:
        raise ValueError("A arquitetura ou os marcadores mudaram, use --reiniciar")
    modelo.load_state_dict(estado["estado_modelo"])
    otimizador.load_state_dict(estado["estado_otimizador"])
    # Permite ajustar a taxa na configuracao sem descartar o historico do AdamW
    for grupo in otimizador.param_groups:
        grupo["lr"] = taxa_aprendizado
    if "estado_aleatorio" in estado:
        torch.set_rng_state(estado["estado_aleatorio"])
    # CUDA usa um gerador separado, que tambem precisa ser restaurado para retomar
    dispositivo = next(modelo.parameters()).device
    if dispositivo.type == "cuda" and estado.get("estado_aleatorio_cuda") is not None:
        torch.cuda.set_rng_state(estado["estado_aleatorio_cuda"], device=dispositivo)
    return estado


# Copia os pesos da etapa anterior para o ajuste, mantendo o checkpoint original intacto
def inicializar_conversas(caminho: Path, modelo: ModeloNuma, dados: dict, configuracao: dict,
                          formato_origem: str = "linguagem_v1") -> dict:
    estado = torch.load(caminho, map_location="cpu", weights_only=True)
    if estado.get("formato") != formato_origem:
        raise ValueError(f"A inicializacao precisa usar um checkpoint no formato {formato_origem}")
    if estado["vocabulario"] != dados["vocabulario"] or estado["marcadores"] != dados["marcadores"]:
        raise ValueError("Os IDs ou marcadores mudaram, prepare o ajuste com o vocabulario do modelo de origem")
    if estado["configuracao_modelo"] != configuracao:
        raise ValueError("A arquitetura precisa ser igual a do modelo de origem")
    modelo.load_state_dict(estado["estado_modelo"])
    # O AdamW comeca um historico novo nesta etapa, mas os pesos sao preservados
    nome_passo = "passo_linguagem" if formato_origem == "linguagem_v1" else "passo_conversas"
    return {"arquivo": str(caminho.resolve()), nome_passo: estado["passo"], "formato_origem": formato_origem}


# Prepara o treinamento, retoma o progresso disponivel e executa mais passos de aprendizado
def principal() -> None:
    analisador_argumentos = argparse.ArgumentParser(description="Treina o Numa por etapas")
    analisador_argumentos.add_argument("--etapa", choices=("linguagem", "conversas", "continuidade"), default="conversas")
    analisador_argumentos.add_argument("--dados", type=Path)
    analisador_argumentos.add_argument("--saida", type=Path)
    analisador_argumentos.add_argument("--passos", type=int, default=PASSOS_TREINO)
    analisador_argumentos.add_argument("--paciencia", type=int, default=PACIENCIA_VALIDACAO,
                                      help="Avaliacoes sem melhora antes de parar, use zero para desativar")
    analisador_argumentos.add_argument("--reiniciar", action="store_true")
    analisador_argumentos.add_argument("--inicializar-de", type=Path,
                                      help="Pesos da etapa anterior usados somente na primeira execucao do novo ajuste")
    analisador_argumentos.add_argument("--taxa-aprendizado", type=float,
                                      help="Taxa do AdamW, por padrao o ajuste usa uma taxa menor")
    analisador_argumentos.add_argument("--revisao-linguagem", type=float,
                                      help="Fracao de lotes que revisam a linguagem, por padrao 0.1 no ajuste")
    analisador_argumentos.add_argument("--dispositivo", choices=("auto", "cpu", "cuda"), default=DISPOSITIVO_TREINO)
    analisador_argumentos.add_argument("--avaliar-geracao", type=Path,
                                      help="Casos de validacao para selecionar pesos por respostas completas aceitas")
    analisador_argumentos.add_argument("--intervalo-geracao", type=int, default=250)
    argumentos = analisador_argumentos.parse_args()
    if argumentos.passos < 1 or argumentos.paciencia < 0:
        analisador_argumentos.error("Use passos positivos e paciencia maior ou igual a zero")
    if argumentos.intervalo_geracao < INTERVALO_AVALIACAO or argumentos.intervalo_geracao % INTERVALO_AVALIACAO:
        analisador_argumentos.error("O intervalo de geracao deve ser um multiplo positivo do intervalo de avaliacao")
    linguagem = argumentos.etapa == "linguagem"
    continuidade = argumentos.etapa == "continuidade"
    if continuidade and argumentos.reiniciar:
        analisador_argumentos.error("A continuidade sempre aproveita pesos existentes, escolha outra saida para um novo experimento")
    if argumentos.inicializar_de and (linguagem or argumentos.reiniciar):
        analisador_argumentos.error("Use --inicializar-de somente no ajuste de conversas, sem --reiniciar")
    taxa_aprendizado = argumentos.taxa_aprendizado
    if taxa_aprendizado is None:
        taxa_aprendizado = TAXA_APRENDIZADO if linguagem else TAXA_AJUSTE_CONVERSAS
    if not math.isfinite(taxa_aprendizado) or taxa_aprendizado <= 0:
        analisador_argumentos.error("A taxa de aprendizado deve ser positiva e finita")
    revisao_linguagem = argumentos.revisao_linguagem
    if revisao_linguagem is None:
        revisao_linguagem = 0.0 if linguagem else 0.1
    if not math.isfinite(revisao_linguagem) or not 0 <= revisao_linguagem < 1 or (linguagem and revisao_linguagem):
        analisador_argumentos.error("Use revisao entre zero e um, somente no ajuste de conversas")
    dados_padrao = ARQUIVO_MANIFESTO_LINGUAGEM if linguagem else (ARQUIVO_MANIFESTO_CONTINUIDADE if continuidade else ARQUIVO_MANIFESTO_AJUSTE)
    saida_padrao = ARQUIVO_CHECKPOINT_LINGUAGEM if linguagem else (ARQUIVO_CHECKPOINT_CONTINUIDADE if continuidade else ARQUIVO_CHECKPOINT)
    argumentos.dados = argumentos.dados or dados_padrao
    argumentos.saida = argumentos.saida or saida_padrao
    if continuidade:
        origem_continuidade = (argumentos.inicializar_de or ARQUIVO_CHECKPOINT).resolve()
        saida_ultima = argumentos.saida.with_name(argumentos.saida.stem + "_ultimo" + argumentos.saida.suffix).resolve()
        if origem_continuidade in (argumentos.saida.resolve(), saida_ultima):
            analisador_argumentos.error("Use uma saida diferente para preservar os pesos de conversas originais")
    objetivo = "linguagem_autorregressiva_v1" if linguagem else "respostas_balanceadas_v1"
    try:
        dispositivo = selecionar_dispositivo(argumentos.dispositivo)
    except ValueError as erro:
        analisador_argumentos.error(str(erro))
    torch.manual_seed(SEMENTE)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    try:
        dados, treino, validacao, assinatura, leitor = carregar_dados(argumentos.dados)
    except (OSError, ValueError, KeyError, TypeError) as erro:
        raise SystemExit(f"Confira ou prepare novamente os dados: {erro}") from erro
    if isinstance(treino, ParteLinguagem) != linguagem:
        analisador_argumentos.error("A etapa escolhida nao corresponde aos dados, confira --etapa e --dados")
    casos_geracao = None
    assinatura_geracao = None
    if argumentos.avaliar_geracao:
        if linguagem:
            analisador_argumentos.error("Use avaliacao de dialogos somente no ajuste de conversas")
        try:
            casos_geracao = carregar_casos_validacao(argumentos.avaliar_geracao, dados, validacao)
            assinatura_geracao = hashlib.sha256(argumentos.avaliar_geracao.read_bytes()).hexdigest()
        except (OSError, ValueError, KeyError, TypeError) as erro:
            raise SystemExit(f"Confira os casos de validacao de geracao: {erro}") from erro
    treino_revisao = None
    assinatura_revisao = None
    if revisao_linguagem:
        try:
            dados_revisao, treino_revisao, _, assinatura_revisao, _ = carregar_dados(ARQUIVO_MANIFESTO_LINGUAGEM)
            if dados_revisao["vocabulario"] != dados["vocabulario"] or not isinstance(treino_revisao, ParteLinguagem):
                raise ValueError("A revisao precisa usar os mesmos IDs do ajuste de conversas")
        except (OSError, ValueError, KeyError, TypeError) as erro:
            raise SystemExit(f"Nao foi possivel preparar a revisao de linguagem: {erro}") from erro

    configuracao = {
        "tamanho_vocabulario": len(dados["vocabulario"]),
        "tamanho_contexto": TAMANHO_CONTEXTO,
        "dimensao_embedding": DIMENSAO_EMBEDDING,
        "num_cabecas": NUM_CABECAS,
        "num_camadas": NUM_CAMADAS,
        "dropout": DROPOUT,
    }
    # Move a rede antes de criar o AdamW para que os parametros usem o dispositivo correto
    modelo = ModeloNuma(**configuracao).to(dispositivo)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=taxa_aprendizado, weight_decay=0.01)
    caminho_ultimo = argumentos.saida.with_name(argumentos.saida.stem + "_ultimo" + argumentos.saida.suffix)
    caminho_retomada = caminho_ultimo if caminho_ultimo.exists() else argumentos.saida
    estado_anterior = {}
    inicializacao = None
    if not argumentos.reiniciar and caminho_retomada.exists():
        try:
            estado_anterior = retomar_treinamento(caminho_retomada, modelo, otimizador, dados, configuracao, taxa_aprendizado)
        except (OSError, ValueError, KeyError, RuntimeError) as erro:
            raise SystemExit(f"Nao foi possivel continuar o treinamento: {erro}") from erro
        print(f"Retomando os pesos do passo {estado_anterior['passo']}", flush=True)
        inicializacao = estado_anterior.get("inicializacao")
    elif not linguagem and not argumentos.reiniciar:
        try:
            origem_padrao = ARQUIVO_CHECKPOINT if continuidade else ARQUIVO_CHECKPOINT_LINGUAGEM
            formato_origem = "conversas_v1" if continuidade else "linguagem_v1"
            inicializacao = inicializar_conversas(argumentos.inicializar_de or origem_padrao,
                                                 modelo, dados, configuracao, formato_origem)
        except (OSError, ValueError, KeyError, RuntimeError) as erro:
            raise SystemExit(f"Nao foi possivel aproveitar os pesos de linguagem: {erro}") from erro
        etapa_origem = "conversas" if continuidade else "linguagem"
        passo_origem = inicializacao["passo_conversas" if continuidade else "passo_linguagem"]
        print(f"Iniciando ajuste de {argumentos.etapa} com os pesos de {etapa_origem} do passo {passo_origem}", flush=True)
    else:
        print(f"Iniciando um novo treinamento de {argumentos.etapa}", flush=True)

    passo_inicial = estado_anterior.get("passo", 0)
    if estado_anterior and estado_anterior.get("assinatura_geracao") != assinatura_geracao:
        analisador_argumentos.error("O criterio de selecao mudou, use outra saida para preservar a comparacao")
    melhor_erro = float("inf")
    melhor_passo = passo_inicial
    melhor_proporcao = -1.0
    resultado_geracao = None
    # Mudancas nos dados ou no calculo do erro exigem uma nova referencia de validacao
    if estado_anterior.get("assinatura_dados") == assinatura and argumentos.saida.exists() and estado_anterior.get("objetivo") == objetivo:
        melhor_erro = estado_anterior.get("melhor_erro_validacao", estado_anterior["erro_validacao"])
        melhor_passo = estado_anterior.get("melhor_passo", passo_inicial)
        melhor_proporcao = estado_anterior.get("melhor_proporcao_geracao", -1.0)
    nome_dispositivo = f"GPU: {torch.cuda.get_device_name(dispositivo)}" if dispositivo.type == "cuda" else "CPU"
    print(f"{nome_dispositivo} | contexto: {TAMANHO_CONTEXTO} | lote: {TAMANHO_LOTE} | mais {argumentos.passos} passos", flush=True)
    quantidade_parametros = sum(parametro.numel() for parametro in modelo.parameters())
    print(f"Arquitetura: {DIMENSAO_EMBEDDING} dimensoes | {NUM_CAMADAS} blocos | {quantidade_parametros} parametros", flush=True)
    print(f"Taxa de aprendizado: {taxa_aprendizado}", flush=True)
    unidade = "Janelas" if linguagem else "Conversas"
    print(f"{unidade}: {len(treino)} de treino e {len(validacao)} de validacao", flush=True)
    if linguagem:
        print("Texto continuo | todos os proximos caracteres participam do erro", flush=True)
    elif leitor is not None:
        print(f"Leitura em blocos | cache de ate {leitor.limite_cache} blocos na RAM", flush=True)
        if leitor.manifesto.get("esquema_exemplos") == "conversas_completas_v2":
            print("Respostas longas incluidas | treino em janelas | erro somente na resposta", flush=True)
    if not linguagem:
        print("Fontes com a mesma chance de sorteio e respostas com o mesmo peso no erro", flush=True)
    if revisao_linguagem:
        print(f"Revisao de linguagem em aproximadamente {revisao_linguagem:.0%} dos lotes", flush=True)
    if argumentos.paciencia:
        modalidade = "de geracao" if casos_geracao else "do erro"
        print(f"Parada apos {argumentos.paciencia} avaliacoes {modalidade} sem melhorar", flush=True)
    if casos_geracao:
        print(f"Selecao por respostas completas aceitas na validacao a cada {argumentos.intervalo_geracao} passos", flush=True)
        print("Correspondencia textual nao mede qualidade geral de conversa", flush=True)

    # A paciencia conta avaliacoes sem melhora dentro desta execucao
    avaliacoes_sem_melhora = 0
    inicio_execucao = time.perf_counter()
    for deslocamento in range(argumentos.passos + 1):
        passo = passo_inicial + deslocamento
        if deslocamento > 0:
            modelo.train()
            # A revisao reutiliza somente o treino da linguagem, nunca seus exemplos de validacao
            revisar = revisao_linguagem > 0 and torch.rand(()).item() < revisao_linguagem
            entrada, alvos = criar_lote(treino_revisao if revisar else treino)
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
            for fonte, erro_validacao in erros["fontes"]["validacao"].items():
                print(f"  Validacao {fonte}: {erro_validacao:.4f}", flush=True)
            mediu_selecao = not casos_geracao or deslocamento % argumentos.intervalo_geracao == 0 or deslocamento == argumentos.passos
            melhorou = erros["validacao"] < melhor_erro if not casos_geracao else False
            if casos_geracao and mediu_selecao:
                resultado_geracao = {**avaliar_geracao(modelo, dados, casos_geracao), "passo": passo}
                proporcao = resultado_geracao["proporcao"]
                melhorou = proporcao > melhor_proporcao or (proporcao == melhor_proporcao and erros["validacao"] < melhor_erro)
                print(f"Geracao na validacao: {resultado_geracao['correspondencias']}/{resultado_geracao['total']} respostas aceitas", flush=True)
                if melhorou:
                    melhor_proporcao = proporcao
            if melhorou:
                melhor_erro = erros["validacao"]
                melhor_passo = passo
                avaliacoes_sem_melhora = 0
            elif deslocamento > 0 and mediu_selecao:
                avaliacoes_sem_melhora += 1
            estado = {
                "formato": dados.get("formato_modelo", "conversas_v1"),
                "etapa": argumentos.etapa,
                "inicializacao": inicializacao,
                "revisao_linguagem": {"fracao": revisao_linguagem, "assinatura_dados": assinatura_revisao},
                "objetivo": objetivo,
                "passo": passo,
                "estado_modelo": modelo.state_dict(),
                "estado_otimizador": otimizador.state_dict(),
                "estado_aleatorio": torch.get_rng_state(),
                "estado_aleatorio_cuda": torch.cuda.get_rng_state(dispositivo) if dispositivo.type == "cuda" else None,
                "configuracao_modelo": configuracao,
                "vocabulario": dados["vocabulario"],
                "marcadores": dados["marcadores"],
                "assinatura_dados": assinatura,
                "formato_dados": leitor.manifesto["formato"] if leitor is not None else "conversas_v1",
                "arquivo_dados": str(argumentos.dados.resolve()),
                "erro_treino": erros["treino"],
                "erro_validacao": erros["validacao"],
                "erros_por_fonte": erros["fontes"],
                "melhor_erro_validacao": melhor_erro,
                "melhor_passo": melhor_passo,
                "assinatura_geracao": assinatura_geracao,
                "melhor_proporcao_geracao": melhor_proporcao,
                "avaliacao_geracao": resultado_geracao,
            }
            if melhorou:
                salvar_checkpoint(argumentos.saida, estado)
            # O ultimo estado permite continuar mesmo se a validacao piorou depois do melhor passo
            salvar_checkpoint(caminho_ultimo, estado)
            if argumentos.paciencia and avaliacoes_sem_melhora >= argumentos.paciencia:
                print("Treinamento encerrado por falta de melhora na validacao", flush=True)
                break

    if casos_geracao:
        print(f"Melhor correspondencia na geracao: {melhor_proporcao:.1%} no passo {melhor_passo}")
        print(f"Erro de validacao nesse checkpoint: {melhor_erro:.4f}")
    else:
        print(f"Melhor validacao: {melhor_erro:.4f} no passo {melhor_passo}")
    descricao_pesos = "Pesos da primeira etapa de linguagem" if linguagem else "Pesos usados pelo chat"
    print(f"{descricao_pesos}: {argumentos.saida}")
    print(f"Progresso para continuar treinando: {caminho_ultimo}")
    print(f"Tempo total com avaliacao e salvamento: {time.perf_counter() - inicio_execucao:.1f} segundos")


if __name__ == "__main__":
    principal()
