# Agradecimentos, confirmacoes e encerramentos

Esta etapa trabalha a dificuldade observada depois de `ok, entendi muito obrigado`
Os dialogos incluem explicacoes de Python, listas, variaveis, redes neurais e a apresentacao da Numa
Tambem incluem agradecimentos seguidos de outra pergunta, para nao ensinar que todo agradecimento termina o assunto

## Preparacao

`dados/brutos/etiqueta_conversa/roteiros.json` contem as frases e as respostas aceitas por intencao
`numa.experimentos.preparar_etiqueta` ja reuniu os novos dialogos com os dados anteriores
As quantidades incluem combinacoes escritas e nao representam conversas humanas independentes
Os contextos usados pelos exemplos novos permanecem nas mesmas divisoes dos exemplos de origem
As frases de validacao e de teste sao diferentes das formulacoes novas de treino

O resultado fica em `dados/processados/etiqueta_conversa_512`
Ele tem 1417 exemplos de treino, 198 de validacao e 178 casos de teste
Os exemplos anteriores foram mantidos e os testes antigos continuam com suas entradas e referencias
A avaliacao de geracao usa 44 casos curtos, 32 de historico longo e 30 de etiqueta
Esses tres grupos recebem o mesmo peso na escolha do checkpoint

O chat continua gerando respostas pela rede neural, sem consultar estas frases como uma tabela de respostas
Os arquivos novos sao exemplos de treinamento
Despedidas como `tchau` ensinam uma resposta de encerramento e o comando `/sair` continua encerrando o programa

## Treinamento manual

Na raiz do projeto, com a `.venv` ativa no Git Bash:

```bash
PYTHONPATH=src python -m numa.treinar --etapa continuidade \
  --dados dados/processados/etiqueta_conversa_512/dados.json \
  --inicializar-de modelos/numa_memoria_revisada_512.pt \
  --saida modelos/numa_etiqueta_512.pt \
  --avaliar-geracao dados/processados/etiqueta_conversa_512/validacao_geracao.jsonl \
  --intervalo-geracao 500 \
  --dispositivo cuda --passos 1500 --paciencia 3 \
  --taxa-aprendizado 0.00002 --revisao-linguagem 0.05
```

O ajuste copia o checkpoint que estava sendo usado no chat e usa uma taxa menor
O numero de passo desse checkpoint pode ser zero na etapa anterior, mas os pesos ja vieram de outros treinamentos
Os checkpoints anteriores sao preservados e os novos arquivos usam o nome `numa_etiqueta_512`
O sorteio tambem revisa os exemplos anteriores e aproximadamente 5% dos lotes de treinamento revisam linguagem
Isso ajuda a preservar habilidades, sem garantir ausencia de esquecimento
Nenhum treinamento foi executado durante a preparacao

## Teste depois do treinamento

```bash
PYTHONPATH=src python -m numa.chat --modelo modelos/numa_etiqueta_512.pt --temperatura 0.5
```

Digite uma mensagem por vez, esperando a resposta:

```text
oi
qual seu nome?
o que voce faz?
ok, entendi muito obrigado
```

O agradecimento deve receber uma resposta pertinente, por exemplo `De nada! Se precisar, estou por aqui`
Depois teste uma formulacao reservada em uma nova conversa:

```text
/limpar
o que e Python?
pode dar um exemplo?
agradeco por ter explicado isso
```

Uma resposta pode estar correta usando palavras diferentes das alternativas aceitas
Confira tambem se a rede continua respondendo aos assuntos anteriores e recuperando informacoes do historico
O contexto continua limitado a 512 caracteres e a etapa nao garante conversa geral sobre qualquer assunto
