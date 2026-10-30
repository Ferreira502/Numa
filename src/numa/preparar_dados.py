# Usa o tokenizer e salva o vocabulario junto dos IDs, 
# para reutilizarmos a mesma correspondencia entre letras e numeros depois

import json
from pathlib import Path

from numa.configuracao import (
    ARQUIVO_DADOS_PROCESSADOS,
    FRACAO_TREINO,
    PASTA_DADOS_BRUTOS,
)
from numa.tokenizer import TokenizadorCaracteres


def ler_textos(pasta: Path) -> str:
    # Junta os arquivos .txt da pasta em ordem alfabetica
    arquivos = sorted(pasta.glob("*.txt"))
    if not arquivos:
        raise ValueError(f"Coloque pelo menos um arquivo .txt em {pasta}")

    textos = []
    for arquivo in arquivos:
        # utf-8-sig tambem aceita arquivos UTF-8 com a marca BOM do Windows
        texto = arquivo.read_text(encoding="utf-8-sig")
        if texto.strip():
            textos.append(texto)

    if not textos:
        raise ValueError("Os arquivos .txt estão vazios ou contêm apenas espaços")

    # Duas quebras de linha separam o conteudo de arquivos diferentes
    return "\n\n".join(textos)


def preparar_dados(texto: str, fracao_treino: float = FRACAO_TREINO) -> dict:
    # Cria um vocabulario fixo e separa duas sequencias de IDs
    if not 0 < fracao_treino < 1:
        raise ValueError("A fração de treino precisa estar entre 0 e 1")

    tokenizer = TokenizadorCaracteres(texto)
    ids = tokenizer.encode(texto)
    # O corte indica onde terminam os 90 por cento usados para treino
    corte = int(len(ids) * fracao_treino)
    if corte < 2 or len(ids) - corte < 2:
        raise ValueError(
            "Texto muito curto: treino e validação precisam de pelo menos "
            "dois caracteres cada. Adicione mais texto"
        )

    # Mantemos a ordem do texto: o modelo aprendera a prever o proximo caractere
    # O vocabulario inclui os caracteres de ambas as partes, para que todos
    # possam ser representados; os pesos serao treinados apenas com o treino
    # Por enquanto estes IDs sao apenas dados, sem calculo de erro ou aprendizado
    # No treino usaremos sequencias deslocadas em uma posicao
    # Exemplo: entrada 'Num' e alvos 'uma' fazem a rede prever o proximo caractere
    return {
        "vocabulario": list(tokenizer.caracteres),
        "ids_treino": ids[:corte],
        "ids_validacao": ids[corte:],
    }


def main() -> None:
    # Prepara os textos e salva o resultado em um unico arquivo JSON
    try:
        texto = ler_textos(PASTA_DADOS_BRUTOS)
        dados = preparar_dados(texto)
        ARQUIVO_DADOS_PROCESSADOS.parent.mkdir(parents=True, exist_ok=True)
        ARQUIVO_DADOS_PROCESSADOS.write_text(
            json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except (OSError, ValueError) as erro:
        raise SystemExit(f"Não foi possível preparar os dados: {erro}") from erro

    print(f"Vocabulário: {len(dados['vocabulario'])} caracteres")
    print(f"Treino: {len(dados['ids_treino'])} IDs")
    print(f"Validação: {len(dados['ids_validacao'])} IDs")
    print(f"Dados salvos em: {ARQUIVO_DADOS_PROCESSADOS}")


if __name__ == "__main__":
    main()
