# Experimento controlado de conversas

Este experimento verifica se a Numa consegue responder a perguntas ensinadas e usar o historico em poucos assuntos conhecidos

Os dialogos foram escritos manualmente para este teste educacional, sem gerar respostas durante o chat por consulta a uma tabela

## Dados

- `dados/brutos/experimento_controlado/dialogos.jsonl`: conversas editaveis com identificador, assunto, divisao e trocas de pergunta e resposta
- `src/numa/preparar_experimento.py`: verifica a cobertura, codifica os textos usando os mesmos IDs e separa os exemplos de cada conversa
- `dados/processados/experimento_controlado_256/dados.json`: 42 exemplos de treino e 14 de validacao, distribuidos entre sete assuntos
- `protocolo.json`: cobertura por assunto, identificacao do arquivo de origem e limites da avaliacao
- `testes_treino.jsonl`, `testes_validacao.jsonl` e `testes_teste.jsonl`: cenarios para gerar respostas completas e revisar o comportamento
- `sondas.jsonl`: 14 entradas destinadas ao treino e 14 reservadas para comparar os mesmos casos antes e depois
- `avaliacao_antes.json`: respostas do checkpoint anterior, sem atualizar os pesos

Cada conversa inteira fica em uma unica divisao e nenhum prefixo de entrada se repete entre as divisoes deste experimento
Todos os assuntos aparecem nas tres divisoes e todas as respostas cabem junto com seu historico no contexto atual
A mesma pergunta de seguimento aparece em assuntos diferentes para verificar se a resposta depende do historico

Os casos reservados nao entram no treino ou na escolha do melhor checkpoint desta etapa
Como aproveitamos pesos anteriores, nao ha garantia de que textos semelhantes nunca tenham aparecido antes
Este experimento mede um comportamento restrito e nao representa um teste de conversacao geral

## Treinamento manual

Execute no Git Bash na raiz do projeto com a `.venv` ativa
A preparacao ja foi executada e nao precisa ser repetida

```bash
PYTHONPATH=src python -m numa.treinar --etapa continuidade \
  --dados dados/processados/experimento_controlado_256/dados.json \
  --inicializar-de modelos/numa_conversas_revisadas_256.pt \
  --saida modelos/numa_experimento_256.pt \
  --dispositivo cuda --passos 1000 --paciencia 10 \
  --taxa-aprendizado 0.0001 --revisao-linguagem 0
```

Os pesos anteriores sao copiados para um ajuste separado e seus arquivos permanecem preservados
O experimento usa apenas estes exemplos pequenos para isolar o aprendizado das respostas
Isso nao garante preservar todas as habilidades do modelo ajustado, por isso mantemos o checkpoint anterior
Uma segunda execucao com a mesma saida retoma o ultimo estado do experimento

## Comparacao depois do treino

```bash
PYTHONPATH=src python -m numa.avaliar_dialogos \
  --modelo modelos/numa_experimento_256.pt \
  --testes dados/processados/experimento_controlado_256/sondas.jsonl \
  --caracteres 100 --temperatura 0.5 --semente 42 \
  --saida dados/processados/experimento_controlado_256/avaliacao_depois.json
```

Compare com `avaliacao_antes.json`, mantendo os mesmos casos e parametros
O avaliador gera tambem as respostas anteriores do historico, como acontece no chat real
As referencias servem para revisao humana e nao sao fornecidas ao modelo durante a geracao

Confira se a resposta atende a pergunta, se o exemplo corresponde ao assunto anterior e se uma pergunta nova permite mudar de assunto
Uma resposta pode estar correta usando palavras diferentes da referencia
Nao transforme correspondencia literal em acuracia nem trate o erro medio por caractere como qualidade de conversa
Se acertar somente os exemplos de treino, ainda falta generalizar
Se continuar errando os exemplos ensinados, investigue o ajuste antes de ampliar a base ou a rede

## Chat do experimento

```bash
PYTHONPATH=src python -m numa.chat --modelo modelos/numa_experimento_256.pt --temperatura 0.5
```

Comece com `o que e Python?` e depois `pode dar um exemplo?`
Em outra conversa, teste `o que e uma lista?` seguido da mesma pergunta de exemplo
As respostas devem mudar de acordo com o assunto
