# Numa

Numa será um projeto educacional de geração de texto em Python. A ideia é treinar do zero um Transformer decoder pequeno e, aos poucos, transformá-lo em um assistente para conversa. Não usaremos API de IA nem pesos pré-treinados. O PyTorch fornecerá tensores, camadas e cálculo de gradientes.

Um modelo pequeno terá capacidade limitada. Alcançar a qualidade de sistemas como GPT ou Claude exige volumes de dados, computação e trabalho de engenharia muito maiores.

## Estrutura

| Caminho | Função futura |
| --- | --- |
| `src/numa/configuracao.py` | Parâmetros do modelo, treinamento e caminhos. |
| `src/numa/preparar_dados.py` | Leitura, preparo e divisão dos textos em treino e validação. |
| `src/numa/tokenizer.py` | Conversão de texto em números e vice-versa, inicialmente por caracteres. |
| `src/numa/modelo.py` | Transformer decoder com atenção causal. |
| `src/numa/treinar.py` | Treinamento, avaliação e salvamento dos pesos. |
| `src/numa/chat.py` | Carregamento do modelo e conversa no terminal. |
| `src/numa/__init__.py` | Marca `numa` como pacote Python. |
| `dados/brutos/` | Textos originais locais. |
| `dados/processados/` | Dados preparados localmente. |
| `modelos/` | Pesos e checkpoints locais. |
| `docs/` | Anotações e explicações futuras. |
| `requirements.txt` | Dependências Python necessárias. |
| `.gitignore` | Arquivos locais que não entram no versionamento. |

O tokenizer por caracteres está implementado. Os demais arquivos Python ainda estão vazios; ainda não há comandos para treinar ou conversar.

## Primeiro componente: tokenizer

O `TokenizadorCaracteres` cria um vocabulário a partir dos caracteres de um texto. `encode` converte o texto em uma lista de IDs; `decode` faz o caminho inverso. Um caractere ausente do vocabulário gera um erro claro. Para experimentar no Git Bash, na raiz do projeto:

```bash
PYTHONPATH=src python -c 'from numa.tokenizer import TokenizadorCaracteres; t = TokenizadorCaracteres("oi, Numa!"); ids = t.encode("oi, Numa!"); print(ids); print(t.decode(ids))'
```

O último resultado deve ser `oi, Numa!`. Para digitar seus próprios textos no terminal, execute:

```bash
python src/numa/tokenizer.py
```

Digite uma frase e pressione Enter. O programa mostra o vocabulário, os IDs e o texto reconstruído. Pressione Enter em uma linha vazia para sair. O vocabulário é criado novamente para cada frase; esta experiência ainda não treina o modelo nem conversa com uma IA.

## Preparação no Windows com Git Bash

1. Confira o Python com `py -V` e o Git com `git --version`. Se algum estiver ausente, instale-o pelos sites oficiais. O PyTorch atual oferece suporte a Python 3.14 no Windows de 64 bits; consulte [a página de instalação do PyTorch](https://pytorch.org/get-started/locally/) caso use outra versão. Nesta máquina, Python 3.14.6 de 64 bits e Git já estão instalados.
2. No VS Code, instale a extensão **Python**, da Microsoft (`ms-python.python`), na aba Extensions. Ela fornece seleção do interpretador, execução e depuração de Python. Depois, selecione o interpretador `Numa/.venv/Scripts/python.exe` com **Python: Select Interpreter**.
3. Na raiz do projeto, crie o ambiente virtual apenas se `.venv` ainda não existir: `py -m venv .venv`. Ative-o no Git Bash com `source .venv/Scripts/activate`. O ambiente separa as bibliotecas deste projeto das globais. Neste projeto, a `.venv` já foi criada; basta ativá-la.
4. Com o ambiente ativo, instale somente o PyTorch para CPU, caso ainda não esteja instalado: `python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu`. `torch` é o nome do pacote PyTorch; a variante CPU evita baixar bibliotecas CUDA desnecessárias neste início. Nesta máquina, PyTorch 2.15.0+cpu já foi instalado. Para usar GPU mais tarde, consulte o seletor no site do PyTorch.
5. Confira com `python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.rand(2, 2))"`. O caminho impresso deve apontar para `.venv`, seguido da versão do PyTorch e de uma matriz de números. Pode aparecer um aviso sobre NumPy ausente; a geração da matriz funciona sem ele, e não precisamos instalá-lo agora.

Git servirá para registrar a evolução do código. Ainda não é necessário inicializar um repositório; nenhum arquivo foi adicionado ou registrado em commit.
