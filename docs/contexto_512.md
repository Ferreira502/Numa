# Contexto de 512 caracteres

O novo modelo usa 512 posicoes sem mudar os cinco blocos, as quatro cabecas ou as representacoes de 256 valores
O tokenizador continua usando caracteres
Os experimentos antigos mantem seu contexto original e o novo treino le 512 do protocolo dos dados

## Pesos preservados

`numa.experimentos.ampliar_contexto` copiou `modelos/numa_conversas_variadas_256.pt` para `modelos/numa_contexto_512_inicial.pt`
Todos os pesos antigos e as primeiras 256 linhas de `embedding_posicao.weight` foram preservados exatamente
Somente as 256 linhas adicionais receberam valores iniciais novos, usando semente 42
A origem, seu SHA256 e seu passo de treinamento estao registrados no checkpoint de inicializacao
O AdamW comeca um estado novo porque a tabela de posicoes mudou de tamanho
Esse arquivo e usado com `--inicializar-de`, nao como retomada de um otimizador antigo

O total passou de 4.069.226 para 4.134.762 parametros, um aumento de 65.536 valores na tabela de posicoes
As previsoes para entradas de ate 256 posicoes ficaram identicas no teste em CPU
O teste de forward em CUDA com lote de quatro exemplos de 512 posicoes passou sem calcular gradientes
Nenhum treinamento foi executado nessa verificacao

## Dados preparados

`dados/brutos/contexto_512/dialogos_longos.jsonl` contem 24 dialogos escritos de cinco turnos
Eles incluem pedidos para recuperar um nome, valor, tarefa ou assunto citado anteriormente
Cada conversa inteira fica somente em treino, validacao ou teste

`numa.experimentos.preparar_contexto_512` reuniu esses dialogos, o bate-papo natural e as conversas variadas
O conjunto novo fica em `dados/processados/conversas_contexto_512`
Ele tem 498 exemplos de treino, 84 de validacao e 70 de teste
Todos os exemplos do conjunto de revisao anterior foram mantidos em suas divisoes
Ha 16 exemplos de treino cuja resposta comeca depois da posicao 256

`validacao_geracao.jsonl` seleciona os 44 casos anteriores de validacao e o ultimo turno dos oito dialogos longos de validacao
Isso limita o custo da geracao e mede tambem pedidos que dependem de historico maior
A correspondencia textual com respostas aceitas continua sendo um criterio restrito e exige revisao humana

## Proximo treino no Git Bash

Na raiz do projeto, com a `.venv` ativa:

```bash
PYTHONPATH=src python -m numa.treinar --etapa continuidade \
  --dados dados/processados/conversas_contexto_512/dados.json \
  --inicializar-de modelos/numa_contexto_512_inicial.pt \
  --saida modelos/numa_bate_papo_512.pt \
  --avaliar-geracao dados/processados/conversas_contexto_512/validacao_geracao.jsonl \
  --intervalo-geracao 500 \
  --dispositivo cuda --passos 2000 --paciencia 4 \
  --taxa-aprendizado 0.00005 --revisao-linguagem 0.05
```

O treino deve anunciar contexto 512, representacoes de 256 dimensoes e 4.134.762 parametros
Os pesos novos de posicao ainda precisam aprender com os dialogos longos
Aumentar o contexto permite consultar mais historico, mas nao garante que a rede use corretamente todas as mensagens
Mensagens antigas continuam saindo da janela quando a conversa ultrapassa o novo limite
Use `/limpar` quando quiser iniciar outra conversa

## Depois do treino

```bash
PYTHONPATH=src python -m numa.chat --modelo modelos/numa_bate_papo_512.pt --temperatura 0.5
```

O chat le o tamanho do contexto do checkpoint e mantem as trocas recentes que couberem
Os pesos de 256 e os dados anteriores permanecem disponiveis
