# Numa

Numa e um projeto educacional de inteligência artificial generativa, desenvolvido em Python com PyTorch, para estudar como uma rede neural aprende a produzir respostas a partir de exemplos de conversa

O modelo utiliza uma arquitetura Transformer decoder pequena, com pesos inicializados aleatoriamente e treinados do zero, e realiza a geração localmente, sem chamadas a APIs de IA e sem usar um modelo pré-treinado como base

## Proposito

O objetivo e compreender o caminho entre uma mensagem escrita e uma resposta gerada: como o texto se torna uma representação numérica, como a rede utiliza o contexto e como o treinamento modifica seus parâmetros

A evolução da Numa e guiada por experimentos, observação dos erros e comparação entre perguntas conhecidas e perguntas que ficaram fora do treinamento

## Como funciona

A tokenizacao inicial atribui um identificador numérico a cada caractere, incluindo letras, espaços e pontuação

Os embeddings transformam esses identificadores e suas posições em vetores aprendidos pela rede, enquanto os blocos Transformer combinam atenção causal e camadas neurais para relacionar as informações disponíveis no contexto

Ao receber uma mensagem, a Numa calcula as probabilidades do próximo caractere e repete esse processo para construir uma resposta, com a temperatura controlando a variação dessa escolha

## Como aprende

O treinamento utiliza exemplos de perguntas e respostas, com atenção especial a previsão da resposta e de seu marcador de encerramento

A entropia cruzada mede o erro das previsões, o backpropagation calcula os gradientes e o otimizador AdamW ajusta os pesos e biases

Os experimentos combinam conversas básicas, exemplos humanos do OpenAssistant e dados sintéticos em português derivados de respostas do GPT4

Esses textos são usados como exemplos de treinamento, sem carregar os pesos dos modelos que os produziram

As fontes são balanceadas durante o treinamento, e perguntas iguais permanecem na mesma parte da divisão entre treino e validação

## Estagio atual

A Numa já realiza treinamento, avaliação, salvamento e retomada dos pesos, além de interação por um chat no terminal com contexto recente

Em experimentos controlados, o modelo conseguiu aprender respostas para exemplos conhecidos, mas as respostas a perguntas novas ainda apresentam erros, repetições e sequencias sem sentido

O erro numérico e acompanhado junto da qualidade do texto gerado, pois uma perda menor ou a reprodução de uma resposta conhecida, por si so, não demonstra capacidade de conversar de forma geral

## Limites e evolução

A Numa ainda e um modelo pequeno e experimental, com contexto limitado e forte tendencia a memorizar conjuntos reduzidos de exemplos, cujas respostas não devem ser tratadas como uma fonte confiável de informação

O objetivo e ampliar gradualmente a coerência, a capacidade de responder a novas formulações e a continuidade das conversas, investigando a qualidade dos dados, a arquitetura e o processo de treinamento
