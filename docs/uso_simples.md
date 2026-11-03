# Usar a Numa com datasets TXT

Adicione os novos arquivos UTF-8 em `dados/brutos/novos`
Um exemplo ja foi criado em `dados/brutos/novos/exemplo.txt`
O fluxo tambem revisa a base preparada anteriormente, incluindo os agradecimentos
Nao e necessario criar um script Python para cada novo dataset

## Formato dos TXT

Use `Usuario:` e `Numa:` alternados, com uma pergunta e sua resposta correta
Mantenha os turnos da mesma conversa juntos e use uma linha vazia para separar conversas independentes

```text
Usuario: oi, podemos estudar um pouco?
Numa: Claro. Sobre o que voce quer conversar?
Usuario: primeiro quero falar de Python
Numa: Python e uma linguagem de programacao.
Usuario: obrigado, ja me ajudou
Numa: De nada! Se precisar, estou por aqui.

Usuario: hoje quero apenas conversar
Numa: Tudo bem. Como foi seu dia?
Usuario: consegui concluir uma tarefa
Numa: Que bom! O que voce conseguiu concluir?
```

Os arquivos desta pasta devem conter dialogos, com respostas escritas por voce ou revisadas
Este leitor nao transforma automaticamente um livro ou texto sem rotulos em uma conversa
Uma mensagem pode continuar nas linhas seguintes, mas uma linha vazia encerra a conversa
O leitor verifica a alternancia dos papeis, os caracteres conhecidos e o limite de contexto
As janelas preservam turnos completos recentes e a pergunta e resposta precisam caber nos 512 caracteres
Caracteres fora do vocabulario atual geram um erro com o arquivo e o turno, sem trocar os IDs ja aprendidos

## Executar no Git Bash

Na raiz do projeto:

```bash
source .venv/Scripts/activate
PYTHONPATH=src python -m numa
```

O comando prepara ou reaproveita os dados, executa ate 1000 passos adicionais e abre o chat
A parada por validacao pode encerrar o treinamento antes desse limite
A GPU e usada quando CUDA esta disponivel e o modo automatico usa CPU quando necessario
A cada nova execucao padrao ha mais treinamento, mesmo quando os TXT nao mudaram
Para abrir somente o chat:

```bash
PYTHONPATH=src python -m numa --chat
```

Para conferir os arquivos sem treinar:

```bash
PYTHONPATH=src python -m numa --preparar
```

Para mudar o limite de passos ou terminar sem abrir o chat:

```bash
PYTHONPATH=src python -m numa --passos 1500 --sem-chat
```

## Divisao e progresso

Os TXT da pasta principal recebem uma divisao estavel por pergunta inicial
Conversas com a mesma pergunta inicial ficam na mesma divisao, incluindo seus turnos posteriores
As divisoes anteriores acompanham as versoes dos dados para impedir que mover um arquivo exponha um teste ao treinamento
Se quiser escolher a divisao de uma nova conversa, use subpastas `treino`, `validacao` ou `teste` dentro de `dados/brutos/novos`
Uma pergunta ja reservada nao pode ser movida para outra divisao
O leitor tambem recusa entradas iguais com respostas conflitantes
Os testes continuam fora da atualizacao dos pesos e da selecao do checkpoint

Os dados preparados ficam em `dados/processados/automatico`, identificados pelo conteudo dos arquivos
Os pesos ficam em `modelos/automatico`, com arquivos separados por versao dos dados
O arquivo `ativo.json` aponta para o checkpoint selecionado para o chat
Com os mesmos arquivos, o treino retoma o ultimo estado e seu otimizador
Quando os TXT mudam, um novo ajuste aproveita os pesos selecionados anteriormente e inicia um novo estado do otimizador
As versoes anteriores permanecem disponiveis

Mantenha na pasta os TXT que deseja continuar revisando nos treinos seguintes
A revisao inclui os exemplos anteriores e lotes de linguagem, mas nao garante ausencia de esquecimento
Os scripts experimentais antigos continuam disponiveis para compatibilidade, e deixam de ser necessarios para adicionar novos TXT

## Organizacao do codigo

Na raiz de `src/numa`, `__main__.py` coordena a execucao, `chat.py` recebe as mensagens, `treinar.py` ajusta os pesos, `tokenizer.py` converte textos e `modelo.py` define a rede
`configuracao.py` centraliza os parametros e `__init__.py` identifica o pacote Python
Os leitores e as funcoes reutilizadas ficam em `interno`
As ferramentas das etapas anteriores ficam em `experimentos` e podem ser executadas com o prefixo `numa.experimentos`
Por exemplo, a antiga avaliacao manual agora usa `python -m numa.experimentos.avaliar_dialogos`
O fluxo cotidiano continua sendo `PYTHONPATH=src python -m numa`
