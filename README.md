# Numa

Numa e um projeto educacional de inteligencia artificial generativa, desenvolvido em Python com PyTorch, para estudar como uma rede neural aprende a produzir respostas a partir de exemplos de conversa

O modelo utiliza uma arquitetura Transformer decoder pequena, com pesos inicializados aleatoriamente e treinados do zero, e realiza a geracao localmente, sem chamadas a APIs de IA e sem usar um modelo pre-treinado como base

## Proposito

O objetivo e compreender o caminho entre uma mensagem escrita e uma resposta gerada: como o texto se torna uma representacao numerica, como a rede utiliza o contexto e como o treinamento modifica seus parametros

A evolucao da Numa e guiada por experimentos, observacao dos erros e comparacao entre perguntas conhecidas e perguntas que ficaram fora do treinamento

## Como funciona

A tokenizacao inicial atribui um identificador numerico a cada caractere, incluindo letras, espacos e pontuacao

Os embeddings transformam esses identificadores e suas posicoes em vetores aprendidos pela rede, enquanto os blocos Transformer combinam atencao causal e camadas neurais para relacionar as informacoes disponiveis no contexto

Ao receber uma mensagem, a Numa calcula as probabilidades do proximo caractere e repete esse processo para construir uma resposta, com a temperatura controlando a variacao dessa escolha

## Como aprende

O treinamento utiliza exemplos de perguntas e respostas, com atencao especial a previsao da resposta e de seu marcador de encerramento

A entropia cruzada mede o erro das previsoes, o backpropagation calcula os gradientes e o otimizador AdamW ajusta os pesos e biases

Os experimentos combinam conversas basicas, exemplos humanos do OpenAssistant e dados sinteticos em portugues derivados de respostas do GPT4

Esses textos sao usados como exemplos de treinamento, sem carregar os pesos dos modelos que os produziram

As fontes sao balanceadas durante o treinamento, e perguntas iguais permanecem na mesma parte da divisao entre treino e validacao

## Estagio atual

A Numa ja realiza treinamento, avaliacao, salvamento e retomada dos pesos, alem de interacao por um chat no terminal com contexto recente

Em experimentos controlados, o modelo conseguiu aprender respostas para exemplos conhecidos, mas as respostas a perguntas novas ainda apresentam erros, repeticoes e sequencias sem sentido

O erro numerico e acompanhado junto da qualidade do texto gerado, pois uma perda menor ou a reproducao de uma resposta conhecida, por si so, nao demonstra capacidade de conversar de forma geral

## Limites e evolucao

A Numa ainda e um modelo pequeno e experimental, com contexto limitado e forte tendencia a memorizar conjuntos reduzidos de exemplos, cujas respostas nao devem ser tratadas como uma fonte confiavel de informacao

O objetivo e ampliar gradualmente a coerencia, a capacidade de responder a novas formulacoes e a continuidade das conversas, investigando a qualidade dos dados, a arquitetura e o processo de treinamento

Alcancar a qualidade de assistentes como GPT ou Claude exige recursos de dados, computacao e engenharia muito maiores que os utilizados neste projeto
