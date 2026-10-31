# Reune conjuntos de conversas compativeis sem copiar os blocos ja processados
# Sorteia as fontes e delega a leitura aos caches dos conjuntos originais

import hashlib
import json
from pathlib import Path

import torch

from numa.configuracao import RAIZ_PROJETO
from numa.dados_treinamento import LeitorBlocos, ParteBlocos


class LeitorMistura:
    # Abre os manifestos declarados e exige os mesmos caracteres e marcadores em todos
    def __init__(self, caminho: Path):
        self.caminho = caminho.resolve()
        conteudo = self.caminho.read_bytes()
        self.manifesto = json.loads(conteudo)
        if self.manifesto.get("formato") != "conversas_mistas_v1":
            raise ValueError("O manifesto nao descreve uma mistura de conversas")
        self.leitores = []
        assinatura = hashlib.sha256(conteudo)
        fontes = set()
        caminhos = set()
        for conjunto in self.manifesto["conjuntos"]:
            origem = (self.caminho.parent / conjunto["manifesto"]).resolve()
            if not origem.is_relative_to(RAIZ_PROJETO) or origem in caminhos:
                raise ValueError("Referencia repetida ou fora do projeto")
            caminhos.add(origem)
            leitor = LeitorBlocos(origem)
            nomes = set(leitor.manifesto["fontes"])
            if nomes & fontes:
                raise ValueError("Cada fonte precisa ter um nome unico na mistura")
            fontes.update(nomes)
            if self.leitores and any(leitor.dados[k] != self.leitores[0].dados[k] for k in ("vocabulario", "marcadores")):
                raise ValueError("A mistura precisa preservar os mesmos IDs e marcadores")
            self.leitores.append(leitor)
            assinatura.update(leitor.assinatura.encode("ascii"))
        if not self.leitores:
            raise ValueError("A mistura esta vazia")
        self.dados = self.leitores[0].dados
        self.assinatura = assinatura.hexdigest()
        self.limite_cache = sum(leitor.limite_cache for leitor in self.leitores)


class ParteMistura:
    # Associa cada fonte ao seu conjunto sem misturar treino com validacao
    def __init__(self, leitor: LeitorMistura, parte: str):
        self.partes = [ParteBlocos(origem, parte) for origem in leitor.leitores]
        self.origens = {fonte: vista for vista in self.partes for fonte in vista.fontes}
        self.fontes = sorted(self.origens)

    # Soma os exemplos disponiveis mantendo a leitura sob demanda
    def __len__(self) -> int:
        return sum(len(vista) for vista in self.partes)

    # Sorteia uma fonte e usa o leitor correspondente para escolher o exemplo
    def sortear(self, gerador: torch.Generator | None = None, fonte: str | None = None) -> dict:
        if fonte is None:
            fonte = self.fontes[torch.randint(len(self.fontes), (1,), generator=gerador).item()]
        if fonte not in self.origens:
            raise ValueError("A fonte nao possui exemplos nesta divisao")
        return self.origens[fonte].sortear(gerador, fonte)
