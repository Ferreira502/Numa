# Define a rede Transformer decoder com embeddings, atencao causal e camadas MLP
# Recebe IDs e retorna logits para prever o proximo caractere em cada posicao

import json

import torch
from torch import nn

from numa.configuracao import (
    ARQUIVO_DADOS_PROCESSADOS,
    DIMENSAO_EMBEDDING,
    DROPOUT,
    NUM_CABECAS,
    NUM_CAMADAS,
    TAMANHO_CONTEXTO,
)


class BlocoTransformer(nn.Module):
    # Combina atencao causal e uma rede neural multicamada

    # Cria as camadas de normalizacao, atencao causal, MLP e dropout de um bloco
    def __init__(self, dimensao_embedding: int, num_cabecas: int, dropout: float):
        super().__init__()
        # nn.Module registra as camadas e seus parametros para o futuro otimizador
        # LayerNorm normaliza os valores de cada posicao para estabilizar o calculo
        # Ela tambem aprende uma escala e um bias, inicializados em 1 e 0
        self.norma_atencao = nn.LayerNorm(dimensao_embedding)
        # Cada cabeca trabalha com dimensao_embedding / num_cabecas valores
        # Cada cabeca pode aprender relacoes diferentes entre posicoes do texto
        # A atencao possui matrizes treinaveis para consultas, chaves e valores
        # O PyTorch inicializa essas matrizes e a projecao de saida automaticamente
        # As pontuacoes de atencao sao calculadas para cada entrada
        # Elas dependem dos pesos treinaveis e do texto recebido
        self.atencao = nn.MultiheadAttention(
            dimensao_embedding, num_cabecas, dropout=dropout, batch_first=True
        )
        self.dropout_atencao = nn.Dropout(dropout)
        self.norma_mlp = nn.LayerNorm(dimensao_embedding)
        # Esta MLP e a rede multicamada presente dentro de cada bloco
        # nn.Linear faz a conta saida = entrada @ pesos.T + bias
        # A primeira Linear possui 4 * dimensao_embedding neuronios
        # Seus pesos formam uma matriz de dimensao_embedding por 4 * dimensao_embedding
        # Seus pesos e biases comecam aleatorios, seguindo a regra padrao do PyTorch
        # GELU aplica uma ativacao nao linear para aprender relacoes mais complexas
        # A segunda Linear reduz o resultado para dimensao_embedding valores novamente
        # Dropout zera parte dos valores durante o treino para ajudar a generalizar
        self.mlp = nn.Sequential(
            nn.Linear(dimensao_embedding, 4 * dimensao_embedding),
            nn.GELU(),
            nn.Linear(4 * dimensao_embedding, dimensao_embedding),
            nn.Dropout(dropout),
        )

    # Aplica a atencao e a MLP com conexoes residuais para transformar a sequencia
    def forward(self, x: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
        # forward descreve o caminho dos dados pela rede
        # Aqui x tem formato [lote, comprimento do texto, dimensao_embedding]
        normalizado = self.norma_atencao(x)
        # Usamos o proprio texto como origem das consultas, chaves e valores
        # As projecoes internas criam representacoes diferentes para cada papel
        # A mascara impede que uma posicao consulte caracteres futuros
        saida_atencao, _ = self.atencao(
            normalizado,
            normalizado,
            normalizado,
            attn_mask=mascara,
            need_weights=False,
        )
        # As somas preservam informacoes e ajudam os gradientes a atravessar a rede
        x = x + self.dropout_atencao(saida_atencao)
        return x + self.mlp(self.norma_mlp(x))


class ModeloNuma(nn.Module):
    # Recebe IDs e produz pontuacoes para o proximo caractere em cada posicao

    # Constroi os embeddings, os blocos Transformer, a saida e a mascara causal
    def __init__(
        self,
        tamanho_vocabulario: int,
        tamanho_contexto: int = TAMANHO_CONTEXTO,
        dimensao_embedding: int = DIMENSAO_EMBEDDING,
        num_cabecas: int = NUM_CABECAS,
        num_camadas: int = NUM_CAMADAS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        if tamanho_vocabulario < 1 or tamanho_contexto < 1 or num_camadas < 1:
            raise ValueError("Vocabulario, contexto e quantidade de camadas devem ser positivos")
        if num_cabecas < 1 or dimensao_embedding < 1 or dimensao_embedding % num_cabecas:
            raise ValueError("A dimensao do embedding deve ser positiva e divisivel pelo numero de cabecas")

        self.tamanho_contexto = tamanho_contexto
        # Embedding e uma tabela de valores aprendidos, sem bias
        # Esta tabela tem tamanho_vocabulario * dimensao_embedding parametros
        # Cada ID seleciona uma linha da tabela, sem dar significado numerico ao ID
        # Por padrao o PyTorch inicia esses valores com uma distribuicao normal
        # de media 0 e desvio padrao 1
        self.embedding_caractere = nn.Embedding(tamanho_vocabulario, dimensao_embedding)
        # A segunda tabela ensina a rede a distinguir a ordem dos caracteres
        # Esta tabela tem tamanho_contexto * dimensao_embedding parametros
        self.embedding_posicao = nn.Embedding(tamanho_contexto, dimensao_embedding)
        self.dropout = nn.Dropout(dropout)
        # Cada bloco possui seus proprios pesos e biases
        self.blocos = nn.ModuleList(
            BlocoTransformer(dimensao_embedding, num_cabecas, dropout)
            for _ in range(num_camadas)
        )
        self.norma_final = nn.LayerNorm(dimensao_embedding)
        # Esta camada gera uma pontuacao para cada caractere do vocabulario
        # Ela tem dimensao_embedding * tamanho_vocabulario pesos
        # A quantidade de neuronios e de biases e igual ao tamanho do vocabulario
        self.camada_saida = nn.Linear(dimensao_embedding, tamanho_vocabulario)

        # True bloqueia o acesso as posicoes futuras na atencao
        mascara = torch.triu(
            torch.ones(tamanho_contexto, tamanho_contexto, dtype=torch.bool), diagonal=1
        )
        # A mascara acompanha o modelo no dispositivo, mas nao e um peso treinavel
        self.register_buffer("mascara_causal", mascara, persistent=False)

    # Transforma um lote de IDs em logits respeitando o limite de contexto e a causalidade
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # Formato da entrada: quantidade de sequencias por quantidade de caracteres
        if ids.ndim != 2:
            raise ValueError("A entrada deve ter duas dimensoes: lote e sequencia")
        comprimento = ids.shape[1]
        if not 1 <= comprimento <= self.tamanho_contexto:
            raise ValueError(f"A sequencia deve ter entre 1 e {self.tamanho_contexto} IDs")

        posicoes = torch.arange(comprimento, device=ids.device)
        # Somamos a representacao do caractere com a representacao de sua posicao
        x = self.embedding_caractere(ids) + self.embedding_posicao(posicoes)
        x = self.dropout(x)
        mascara = self.mascara_causal[:comprimento, :comprimento]
        for bloco in self.blocos:
            x = bloco(x, mascara)

        # A saida contem logits, que sao pontuacoes ainda sem converter em probabilidades
        # Formato da saida: [lote, comprimento do texto, tamanho do vocabulario]
        # Na posicao 0 as pontuacoes tentam prever o caractere da posicao 1
        # No treino a entropia cruzada comparara esses logits com os IDs corretos
        # loss.backward() calculara os gradientes e o otimizador ajustara os parametros
        return self.camada_saida(self.norma_final(x))


# Executa uma passagem por uma conversa preparada e mostra os formatos e os parametros
def principal() -> None:
    if not ARQUIVO_DADOS_PROCESSADOS.exists():
        raise SystemExit("Execute primeiro a preparacao dos dados")
    dados = json.loads(ARQUIVO_DADOS_PROCESSADOS.read_text(encoding="utf-8"))
    # Criar o modelo inicializa seus parametros, sem aprender com o texto ainda
    # Uma nova execucao cria novos valores aleatorios, pois nao carregamos pesos salvos
    modelo = ModeloNuma(tamanho_vocabulario=len(dados["vocabulario"]))
    # Os colchetes externos criam um lote com uma unica sequencia limitada ao contexto
    ids = torch.tensor([dados["treino"][0]["ids"][:TAMANHO_CONTEXTO]], dtype=torch.long)
    # eval desativa o dropout para esta demonstracao
    modelo.eval()
    # no_grad dispensa o registro dos calculos para backpropagation
    # Este teste apenas calcula a saida, sem medir o erro nem atualizar parametros
    with torch.no_grad():
        logits = modelo(ids)

    # numel conta os valores das tabelas, matrizes e vetores treinaveis
    # O total de parametros inclui mais do que a quantidade de neuronios
    print(f"Parametros do modelo: {sum(parametro.numel() for parametro in modelo.parameters()):,}")
    print(f"Formato da entrada: {tuple(ids.shape)}")
    print(f"Formato da saida: {tuple(logits.shape)}")
    print("Os pesos foram inicializados aleatoriamente e ainda precisam de treinamento")


if __name__ == "__main__":
    principal()
