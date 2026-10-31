# Ajuste de conversas variadas

Esta etapa continua o ultimo estado do experimento controlado, que ja respondeu corretamente aos exemplos ensinados
O objetivo e melhorar as reformulacoes, os pedidos de exemplo e as mudancas de assunto nos sete grupos conhecidos
Nao e um treinamento para conversar sobre qualquer assunto

## O que mudou

As formulacoes e respostas novas estao em `dados/brutos/conversas_variadas`
`variacoes.json` associa perguntas e seguimentos ao significado definido para cada assunto
`adicionais.jsonl` acrescenta validacao e seis conversas curtas de tres turnos
`respostas_aceitas.json` descreve alternativas corretas para comparar respostas curtas na validacao
`numa.ampliar_experimento` combina estas formulacoes com os registros antigos sem mover as divisoes anteriores

A preparacao ja foi executada em `dados/processados/conversas_variadas_256`
O conjunto tem 354 exemplos de treino, 28 de validacao e os mesmos 14 casos de teste reservados anteriormente
Os numeros de treino incluem combinacoes de formulacoes, nao conversas humanas independentes
As respostas completas cabem junto com o historico no contexto de 256 caracteres
Conversas mais longas continuam limitadas por esse contexto

## Treinamento pelo terminal

Na raiz do projeto e com a `.venv` ativa no Git Bash:

```bash
PYTHONPATH=src python -m numa.treinar --etapa continuidade \
  --dados dados/processados/conversas_variadas_256/dados.json \
  --inicializar-de modelos/numa_experimento_256_ultimo.pt \
  --saida modelos/numa_conversas_variadas_256.pt \
  --avaliar-geracao dados/processados/conversas_variadas_256/testes_validacao.jsonl \
  --intervalo-geracao 250 \
  --dispositivo cuda --passos 2000 --paciencia 6 \
  --taxa-aprendizado 0.00005 --revisao-linguagem 0
```

O treinamento copia os pesos anteriores e cria checkpoints novos
O checkpoint selecionado usa primeiro a proporcao de respostas aceitas na geracao da validacao
Empates usam o menor erro de validacao por caractere
Os checkpoints registram as respostas geradas, o passo, a semente e os parametros dessa avaliacao
O historico tambem e gerado pela Numa e nao recebe as respostas corretas anteriores
A paciencia conta somente avaliacoes de geracao, a cada 250 passos e no final da execucao
Sem `--avaliar-geracao`, o treinamento anterior continua selecionando pelo erro por caractere
Ao retomar esta etapa, mantenha o arquivo e os parametros do criterio de selecao

Esse criterio compara textos aceitos, ignorando caixa, espacos repetidos e pontuacao final
Ele pode rejeitar uma resposta correta escrita de outra forma e nao garante qualidade semantica ou conversa geral
Os casos de teste ficam fora dessa escolha
Os exemplos reservados sao desta etapa e nao tem garantia de ineditismo em todos os treinamentos passados

## Avaliacao depois de treinar

```bash
PYTHONPATH=src python -m numa.avaliar_dialogos \
  --modelo modelos/numa_conversas_variadas_256.pt \
  --testes dados/processados/conversas_variadas_256/testes_teste.jsonl \
  --caracteres 100 --temperatura 0.5 --semente 42 \
  --saida dados/processados/conversas_variadas_256/avaliacao_depois.json
```

Compare com `avaliacao_antes.json` na mesma pasta
Revise se os exemplos correspondem ao assunto anterior e se as mudancas de assunto sao atendidas
Uma resposta literalmente diferente da referencia pode estar correta

## Chat dos novos pesos

```bash
PYTHONPATH=src python -m numa.chat --modelo modelos/numa_conversas_variadas_256.pt --temperatura 0.5
```

Os arquivos de pesos anteriores permanecem preservados
Um ajuste concentrado pode alterar outras habilidades, por isso nao substituimos os checkpoints existentes
