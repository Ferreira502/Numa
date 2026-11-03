# Executa a Numa em um fluxo unico: preparar TXT, treinar e abrir o chat
# Cada conjunto de dados tem seus checkpoints e as execucoes repetidas retomam o progresso

import argparse
import json
from pathlib import Path

from numa.configuracao import (
    RAIZ_PROJETO, PASTA_MODELOS_AUTOMATICOS, PASSOS_AUTOMATICOS,
)
from numa.interno.preparar_dados import preparar_dados


# Encontra os pesos do fluxo automatico ou a versao treinada mais recente das etapas anteriores
def modelo_disponivel() -> Path:
    ativo = PASTA_MODELOS_AUTOMATICOS / "ativo.json"
    if ativo.exists():
        caminho = (RAIZ_PROJETO / json.loads(ativo.read_text(encoding="utf-8"))["modelo"]).resolve()
        if caminho.is_relative_to(RAIZ_PROJETO) and caminho.is_file():
            return caminho
        raise ValueError("O checkpoint registrado como ativo nao foi encontrado")
    for nome in ("numa_etiqueta_512.pt", "numa_memoria_revisada_512.pt"):
        caminho = RAIZ_PROJETO / "modelos" / nome
        if caminho.is_file():
            return caminho
    raise ValueError("Nenhum checkpoint inicial de 512 foi encontrado na pasta modelos")


# Coordena o fluxo pelo terminal sem precisar de scripts diferentes para cada novo dataset
def principal(argumentos_cli: list[str] | None = None) -> None:
    analisador = argparse.ArgumentParser(description="Adiciona TXT, prepara dados, treina a Numa e abre o chat")
    modo = analisador.add_mutually_exclusive_group()
    modo.add_argument("--chat", action="store_true", help="Abre somente o chat, sem preparar ou treinar")
    modo.add_argument("--preparar", action="store_true", help="Somente confere e prepara os dados")
    analisador.add_argument("--sem-chat", action="store_true", help="Encerra depois do treinamento")
    analisador.add_argument("--passos", type=int, default=PASSOS_AUTOMATICOS)
    analisador.add_argument("--dispositivo", choices=("auto", "cpu", "cuda"), default="auto")
    argumentos = analisador.parse_args(argumentos_cli)
    if argumentos.passos < 1:
        analisador.error("Use um numero positivo de passos")
    try:
        if argumentos.chat:
            from numa.chat import principal as abrir_chat
            abrir_chat(["--modelo", str(modelo_disponivel()), "--temperatura", "0.5"])
            return
        info = preparar_dados()
        print(f"Dados: {info['treino']} de treino, {info['validacao']} de validacao e {info['teste']} de teste", flush=True)
        print(f"TXT encontrados: {info['arquivos_txt']} | contexto: {info['contexto']} | versao: {info['versao']}", flush=True)
        if argumentos.preparar:
            print(f"Preparacao concluida em {info['pasta']}, nenhum treinamento executado")
            return
        destino = PASTA_MODELOS_AUTOMATICOS / (info["versao"] + ".pt")
        from numa.treinar import principal as treinar
        print(f"Iniciando ate {argumentos.passos} passos adicionais de treinamento", flush=True)
        parametros = ["--etapa", "continuidade", "--dados", str(info["pasta"] / "dados.json"),
                      "--saida", str(destino), "--avaliar-geracao", str(info["pasta"] / "validacao_geracao.jsonl"),
                      "--intervalo-geracao", "500", "--passos", str(argumentos.passos), "--paciencia", "3",
                      "--taxa-aprendizado", "0.00002", "--revisao-linguagem", "0.05",
                      "--dispositivo", argumentos.dispositivo]
        ultimo = destino.with_name(destino.stem + "_ultimo.pt")
        if not destino.exists() and not ultimo.exists():
            parametros.extend(["--inicializar-de", str(modelo_disponivel())])
        treinar(parametros)
        if not destino.is_file():
            raise ValueError("O treinamento terminou sem produzir o checkpoint selecionado")
        PASTA_MODELOS_AUTOMATICOS.mkdir(parents=True, exist_ok=True)
        temporario = PASTA_MODELOS_AUTOMATICOS / "ativo.json.tmp"
        temporario.write_text(json.dumps({"modelo": destino.relative_to(RAIZ_PROJETO).as_posix(),
                                         "versao": info["versao"]}, indent=2), encoding="utf-8")
        temporario.replace(PASTA_MODELOS_AUTOMATICOS / "ativo.json")
        if not argumentos.sem_chat:
            from numa.chat import principal as abrir_chat
            abrir_chat(["--modelo", str(destino), "--temperatura", "0.5"])
    except (OSError, ValueError, KeyError, TypeError) as erro:
        raise SystemExit(f"Nao foi possivel executar a Numa: {erro}") from erro


if __name__ == "__main__":
    principal()
