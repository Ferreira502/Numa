# Compara os pesos existentes na mesma divisao de linguagem sem executar treinamento
# Pode salvar o melhor candidato reavaliado em outro arquivo, preservando os originais

import argparse
from pathlib import Path

from numa.experimentos.testar_linguagem import carregar_modelo, torch
from numa.configuracao import ARQUIVO_MANIFESTO_LINGUAGEM
from numa.treinar import avaliar, carregar_dados, salvar_checkpoint, selecionar_dispositivo


# Avalia os candidatos na mesma amostra fixa e preserva o otimizador do candidato escolhido
def avaliar_candidatos(caminhos: list[Path], dados: Path, dispositivo: torch.device, saida: Path | None = None) -> dict:
    if not caminhos or (saida is not None and saida.exists()):
        raise ValueError("Informe candidatos e escolha uma saida que ainda nao existe")
    descricao, treino, validacao, assinatura, leitor = carregar_dados(dados)
    if descricao.get("formato_modelo") != "linguagem_v1":
        raise ValueError("Informe dados da etapa de linguagem")
    resultados = []
    melhor = None
    for caminho in caminhos:
        modelo, tokenizador, resumo = carregar_modelo(caminho, dispositivo)
        if list(tokenizador.caracteres) != descricao["vocabulario"]:
            raise ValueError(f"Os IDs dos dados nao correspondem aos pesos de {caminho}")
        erros = avaliar(modelo, treino, validacao)
        resultado = {"arquivo": str(caminho.resolve()), "passo": resumo["passo"], "erros": erros}
        resultados.append(resultado)
        print(f"Passo {resumo['passo']} | treino: {erros['treino']:.4f} | validacao: {erros['validacao']:.4f}", flush=True)
        if melhor is None or erros["validacao"] < melhor["erros"]["validacao"]:
            melhor = resultado
    if saida is not None:
        estado = torch.load(melhor["arquivo"], map_location="cpu", weights_only=True)
        estado.update({
            "checkpoint_origem": melhor["arquivo"], "assinatura_dados": assinatura,
            "arquivo_dados": str(leitor.caminho), "formato_dados": leitor.manifesto["formato"],
            "erro_treino": melhor["erros"]["treino"], "erro_validacao": melhor["erros"]["validacao"],
            "erros_por_fonte": melhor["erros"]["fontes"],
            "melhor_erro_validacao": melhor["erros"]["validacao"], "melhor_passo": melhor["passo"],
        })
        salvar_checkpoint(saida, estado)
        print(f"Melhores pesos reavaliados salvos em: {saida.resolve()}")
    return {"candidatos": resultados, "melhor": melhor}


# Recebe os caminhos e compara os checkpoints sem atualizar os parametros da rede
def principal() -> None:
    analisador = argparse.ArgumentParser(description="Reavalia pesos de linguagem sem treinamento")
    analisador.add_argument("--dados", type=Path, default=ARQUIVO_MANIFESTO_LINGUAGEM)
    analisador.add_argument("--modelos", type=Path, nargs="+", required=True)
    analisador.add_argument("--saida", type=Path)
    analisador.add_argument("--dispositivo", choices=("auto", "cpu", "cuda"), default="auto")
    argumentos = analisador.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    try:
        avaliar_candidatos(argumentos.modelos, argumentos.dados,
                          selecionar_dispositivo(argumentos.dispositivo), argumentos.saida)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as erro:
        raise SystemExit(f"Nao foi possivel avaliar a linguagem: {erro}") from erro


if __name__ == "__main__":
    principal()
