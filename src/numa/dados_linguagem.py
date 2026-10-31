# Le janelas de texto continuo para prever todos os proximos caracteres
# Mapeia os arquivos no disco e copia para RAM apenas os exemplos sorteados

import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch

from numa.configuracao import BLOCOS_EM_CACHE, TAMANHO_CONTEXTO
from numa.dados_treinamento import validar_vocabulario


class LeitorLinguagem:
    # Valida os metadados antes de permitir acesso aos arquivos numericos
    def __init__(self, caminho: Path, limite_cache: int = BLOCOS_EM_CACHE):
        if limite_cache < 1 or sys.byteorder != "little":
            raise ValueError("Use cache positivo e uma plataforma little-endian")
        self.caminho = caminho.resolve()
        self.pasta = self.caminho.parent
        conteudo = self.caminho.read_bytes()
        self.manifesto = json.loads(conteudo.decode("utf-8"))
        if self.manifesto.get("formato") != "linguagem_blocos_v1" or self.manifesto["tipo_ids"] != "int32_little_endian":
            raise ValueError("Formato de linguagem nao reconhecido")
        if self.manifesto["tamanho_contexto"] != TAMANHO_CONTEXTO or not 0 < self.manifesto["fracao_treino"] < 1:
            raise ValueError("Confira o contexto e a divisao dos dados de linguagem")
        vocabulario = self.resolver_arquivo(self.manifesto["arquivo_vocabulario"]).read_bytes()
        self.dados = json.loads(vocabulario.decode("utf-8"))
        self.dados["formato_modelo"] = "linguagem_v1"
        validar_vocabulario(self.dados)
        if len(self.dados["vocabulario"]) != self.manifesto["tamanho_vocabulario"]:
            raise ValueError("O vocabulario nao corresponde ao manifesto")
        self.assinatura = hashlib.sha256(conteudo + b"\0" + vocabulario).hexdigest()
        self.limite_cache = limite_cache
        self._cache = OrderedDict()
        arquivos_vistos = set()
        for parte in ("treino", "validacao"):
            resumo = self.manifesto["partes"][parte]
            total = 0
            if set(resumo["fontes"]) != set(self.manifesto["fontes"]):
                raise ValueError("As fontes das duas partes precisam corresponder ao manifesto")
            for fonte, info in resumo["fontes"].items():
                arquivo = self.resolver_arquivo(info["arquivo"])
                if arquivo in arquivos_vistos:
                    raise ValueError("Treino e validacao precisam usar arquivos separados")
                arquivos_vistos.add(arquivo)
                quantidade = info["caracteres"]
                origem = self.manifesto["fontes"][fonte]
                esperada = origem["corte_treino"] if parte == "treino" else origem["caracteres"] - origem["corte_treino"]
                if type(quantidade) is not int or quantidade != esperada or quantidade <= TAMANHO_CONTEXTO:
                    raise ValueError("Quantidade de caracteres invalida")
                if info["total"] != quantidade - TAMANHO_CONTEXTO or arquivo.stat().st_size != quantidade * 4:
                    raise ValueError("O tamanho dos dados nao corresponde ao manifesto")
                total += info["total"]
            if total < 1 or total != resumo["total"]:
                raise ValueError("Quantidade total de janelas invalida")

    # Impede leitura de caminhos ausentes ou fora do conjunto preparado
    def resolver_arquivo(self, relativo: str) -> Path:
        caminho = (self.pasta / relativo).resolve()
        if not caminho.is_relative_to(self.pasta) or not caminho.is_file():
            raise ValueError(f"Arquivo ausente ou fora da pasta de linguagem: {relativo}")
        return caminho

    # Confere a integridade e abre o mapeamento sem converter todo o arquivo em uma lista
    def carregar_sequencia(self, parte: str, fonte: str) -> torch.Tensor:
        info = self.manifesto["partes"][parte]["fontes"][fonte]
        chave = info["arquivo"]
        if chave not in self._cache:
            while len(self._cache) >= self.limite_cache:
                self._cache.popitem(last=False)
            arquivo = self.resolver_arquivo(chave)
            with arquivo.open("rb") as entrada:
                if hashlib.file_digest(entrada, "sha256").hexdigest() != info["sha256"]:
                    raise ValueError(f"O arquivo de linguagem foi alterado: {chave}")
            self._cache[chave] = torch.from_file(str(arquivo), shared=False, size=info["caracteres"], dtype=torch.int32)
        self._cache.move_to_end(chave)
        return self._cache[chave]


class ParteLinguagem:
    # Representa uma divisao sem misturar caracteres do treino com os da validacao
    def __init__(self, leitor: LeitorLinguagem, parte: str):
        self.leitor = leitor
        self.parte = parte
        self.resumo = leitor.manifesto["partes"][parte]
        self.fontes = sorted(self.resumo["fontes"])

    # Conta as posicoes onde cabe uma janela completa de contexto mais o proximo caractere
    def __len__(self) -> int:
        return self.resumo["total"]

    # Sorteia uma janela e usa todos os caracteres seguintes como alvos de aprendizado
    def sortear(self, gerador: torch.Generator | None = None, fonte: str | None = None) -> dict:
        if fonte is None:
            fonte = self.fontes[torch.randint(len(self.fontes), (1,), generator=gerador).item()]
        if fonte not in self.fontes:
            raise ValueError(f"A fonte {fonte} nao existe em {self.parte}")
        inicio = torch.randint(self.resumo["fontes"][fonte]["total"], (1,), generator=gerador).item()
        ids = self.leitor.carregar_sequencia(self.parte, fonte)[inicio:inicio + TAMANHO_CONTEXTO + 1].long()
        if ids.min().item() < 0 or ids.max().item() >= len(self.leitor.dados["vocabulario"]):
            raise ValueError("Os IDs nao correspondem ao vocabulario da linguagem")
        # Inicio igual a um permite reaproveitar os lotes sem mascarar nenhum alvo
        return {"ids": ids, "inicio_resposta": 1, "fonte": fonte}
