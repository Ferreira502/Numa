# Revisao da memoria contextual de 512 caracteres

O treino anterior tinha somente 16 exemplos cuja resposta comecava depois da posicao 256
O novo conjunto tem 376 exemplos desse tipo, sorteados como uma fonte propria
Isso aumenta a oportunidade de aprender as posicoes novas sem excluir os exemplos anteriores

## Dados e sorteio

`dados/brutos/contexto_512_revisado/roteiros.json` define exercicios de nomes, idades, listas, tarefas, leituras e assuntos anteriores
As formulacoes de abertura e de pergunta final sao separadas por treino, validacao e teste
Os valores, as respostas curtas e as distancias variam entre os exercicios
As quantidades incluem combinacoes de roteiros sinteticos e nao representam conversas humanas independentes
Duas combinacoes de teste foram descartadas por ultrapassar 512 caracteres, antes de avaliar qualquer resposta do modelo

`numa.experimentos.revisar_contexto_512` ja preparou `dados/processados/contexto_512_revisado`
O conjunto tem 1218 exemplos de treino, 168 de validacao e 148 casos de teste
Os exemplos anteriores continuam em suas divisoes e as entradas dos testes antigos foram preservadas
Somente as alternativas aceitas da validacao foram complementadas em arquivos novos

O sorteio escolhe com a mesma chance tres fontes de conversa:

- `revisao`: 482 exemplos anteriores com alvos nas primeiras 256 posicoes
- `conversa_contextual`: 360 exemplos novos com alvos nas primeiras 256 posicoes
- `historico_longo`: 376 exemplos com respostas iniciadas depois da posicao 256

O comando tambem reserva aproximadamente 5% dos lotes para revisao de linguagem
Isso reduz o foco exclusivo nos exemplos novos, mas nao garante ausencia de esquecimento

## Avaliacao

`validacao_geracao.jsonl` usa 44 casos curtos e 32 casos de historico longo
A selecao do checkpoint usa a media das proporcoes de correspondencia nos dois grupos, com peso igual
O total de correspondencias tambem aparece, mas nao decide sozinho quando os grupos possuem tamanhos diferentes
As respostas anteriores do historico sao geradas pela propria Numa

As alternativas aceitas sao textos explicitos para cada caso, por exemplo `20` e uma frase completa equivalente
Uma resposta como `120`, `nao e 20` ou `20 e 25` nao e aceita como `20`
Nao usamos busca de numeros ou palavras dentro de frases para declarar um acerto
A correspondencia continua limitada e pode rejeitar outras respostas corretas

`reavaliacao_antiga.json` mostra somente a recontagem das 52 respostas antigas ja salvas
Ela nao cobre os 24 novos casos de validacao e nao pode ser comparada como se fosse uma execucao dos 76 casos novos
Na recontagem antiga, o ultimo checkpoint recuperou corretamente o valor 20 em um caso longo

## Treinamento manual

Na raiz do projeto, com a `.venv` ativa no Git Bash:

```bash
PYTHONPATH=src python -m numa.treinar --etapa continuidade \
  --dados dados/processados/contexto_512_revisado/dados.json \
  --inicializar-de modelos/numa_bate_papo_512_ultimo.pt \
  --saida modelos/numa_memoria_revisada_512.pt \
  --avaliar-geracao dados/processados/contexto_512_revisado/validacao_geracao.jsonl \
  --intervalo-geracao 500 \
  --dispositivo cuda --passos 3000 --paciencia 4 \
  --taxa-aprendizado 0.00003 --revisao-linguagem 0.05
```

O ajuste parte dos pesos do passo 2000, preservando tambem o aprendizado das posicoes novas
O otimizador comeca um estado novo para este conjunto separado
As linhas `Geracao curto` e `Geracao historico_longo` mostram o comportamento de cada grupo
Os checkpoints anteriores permanecem preservados
Nao foi executado treinamento durante a preparacao ou os testes de carregamento

## Depois do treinamento

```bash
PYTHONPATH=src python -m numa.chat --modelo modelos/numa_memoria_revisada_512.pt --temperatura 0.5
```

Avalie tambem os casos reservados em `testes_teste.jsonl` com `numa.experimentos.avaliar_dialogos`
O contexto permanece em 512 caracteres e informacoes mais antigas continuam saindo da janela
Aumentar os exemplos e corrigir a selecao nao garante uma conversa geral ou memoria permanente
