# Valida os exemplos e le os blocos de treino e validacao sob demanda
# Mantem um cache limitado na RAM e sorteia exemplos sem favorecer blocos menores

import hashlib
import json
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import torch

from numa.configuracao import BLOCOS_EM_CACHE, TAMANHO_CONTEXTO
from numa.preparar_blocos import definir_divisao


# Confere a ordem dos caracteres e a presenca dos marcadores no vocabulario comum
def validar_vocabulario(dados: dict) -> dict:
    vocabulario = dados["vocabulario"]
    if not isinstance(vocabulario, list) or not vocabulario or any(
        not isinstance(caractere, str) or len(caractere) != 1 for caractere in vocabulario
    ) or vocabulario != sorted(set(vocabulario)):
        raise ValueError("O vocabulario deve conter caracteres unicos em ordem fixa")
    marcadores = dados["marcadores"]
    mapa = {caractere: indice for indice, caractere in enumerate(vocabulario)}
    for nome in ("usuario", "resposta", "fim"):
        marcador = marcadores[nome]
        if not isinstance(marcador, str) or not marcador:
            raise ValueError("Os marcadores devem ser textos nao vazios")
        if any(caractere not in mapa for caractere in marcador):
            raise ValueError("Um marcador contem caracteres ausentes do vocabulario")
    return {nome: [mapa[caractere] for caractere in marcadores[nome]] for nome in ("usuario", "resposta", "fim")}


# Converte IDs validos para um tensor na CPU, preservando a memoria da GPU para os lotes
def validar_ids(valores: list[int], tamanho_vocabulario: int) -> torch.Tensor:
    if not isinstance(valores, list) or not valores or any(type(valor) is not int for valor in valores):
        raise ValueError("Os IDs devem ser uma lista nao vazia de numeros inteiros")
    ids = torch.tensor(valores, dtype=torch.long)
    if ids.min().item() < 0 or ids.max().item() >= tamanho_vocabulario:
        raise ValueError("Os IDs nao correspondem ao vocabulario salvo")
    return ids


# Confere o contexto e o inicio da resposta e converte somente os registros recebidos
def validar_exemplos(registros: list[dict], dados: dict, fonte_esperada: str | None = None,
                     permitir_longas: bool = False) -> list[dict]:
    if not isinstance(registros, list) or not registros:
        raise ValueError("A parte dos dados nao possui conversas")
    marcadores = validar_vocabulario(dados)
    usuario, resposta, fim = (marcadores[nome] for nome in ("usuario", "resposta", "fim"))
    exemplos = []
    for registro in registros:
        valores = registro["ids"]
        ids = validar_ids(valores, len(dados["vocabulario"]))
        if not permitir_longas and len(ids) > TAMANHO_CONTEXTO + 1:
            raise ValueError("Uma conversa ultrapassa o contexto, prepare novamente os dados")
        inicio = registro["inicio_resposta"]
        if type(inicio) is not int or not len(usuario) + len(resposta) < inicio < len(ids) - len(fim):
            raise ValueError("O inicio da resposta esta fora da conversa")
        if valores[:len(usuario)] != usuario or valores[inicio - len(resposta):inicio] != resposta:
            raise ValueError("O inicio da resposta nao corresponde aos marcadores")
        if valores[-len(fim):] != fim:
            raise ValueError("A conversa esta sem o marcador de fim")
        fonte = registro.get("fonte", "personalizada")
        if not isinstance(fonte, str) or not fonte or (fonte_esperada is not None and fonte != fonte_esperada):
            raise ValueError("A fonte do exemplo nao corresponde ao bloco")
        exemplos.append({"ids": ids, "inicio_resposta": inicio, "fonte": fonte})
    return exemplos


# Mantem a leitura dos pequenos arquivos JSON usados nos experimentos anteriores
def carregar_conversas(dados: dict) -> tuple[list[dict], list[dict]]:
    if dados.get("formato") != "conversas_v1":
        raise ValueError("Formato de conversas nao reconhecido")
    return validar_exemplos(dados["treino"], dados), validar_exemplos(dados["validacao"], dados)


class BlocoConversasCompletas:
    # Guarda as linhas compactas e converte somente a conversa sorteada para tensor
    def __init__(self, linhas: list[bytes], dados: dict, fonte: str, parte: str, fracao: float, semente: int):
        self.linhas = linhas
        self.dados = dados
        self.fonte = fonte
        self.parte = parte
        self.fracao = fracao
        self.semente = semente

    # Informa a quantidade de pares sem decodificar os textos longos do bloco
    def __len__(self) -> int:
        return len(self.linhas)

    # Confere os IDs, os marcadores e a divisao do par recebido antes de usa-lo no treino
    def __getitem__(self, indice: int) -> dict:
        registro = json.loads(self.linhas[indice])
        exemplo = validar_exemplos([registro], self.dados, self.fonte, permitir_longas=True)[0]
        grupo = registro["grupo_pergunta"]
        if not isinstance(grupo, str) or len(grupo) != 64 or any(c not in "0123456789abcdef" for c in grupo):
            raise ValueError("O grupo da pergunta e invalido")
        assinatura = hashlib.sha256(f"{self.semente}:{grupo}".encode("ascii")).digest()
        prevista = "treino" if int.from_bytes(assinatura[:8], "big") < int(self.fracao * 2**64) else "validacao"
        if prevista != self.parte:
            raise ValueError("A pergunta esta na divisao errada")
        return exemplo


class LeitorBlocos:
    # Carrega apenas o manifesto e o vocabulario ao abrir o conjunto de dados
    def __init__(self, caminho: Path, limite_cache: int = BLOCOS_EM_CACHE):
        if limite_cache < 1:
            raise ValueError("O cache precisa permitir pelo menos um bloco")
        self.caminho = caminho.resolve()
        self.pasta = self.caminho.parent
        conteudo = self.caminho.read_bytes()
        self.manifesto = json.loads(conteudo.decode("utf-8"))
        if self.manifesto.get("formato") != "conversas_blocos_v1":
            raise ValueError("O arquivo nao e um manifesto de blocos")
        if self.manifesto["tamanho_contexto"] != TAMANHO_CONTEXTO:
            raise ValueError("O contexto mudou, prepare novamente os blocos")
        if not 0 < self.manifesto["fracao_treino"] < 1:
            raise ValueError("A fracao de treino do manifesto e invalida")
        vocabulario = self.resolver_arquivo(self.manifesto["arquivo_vocabulario"]).read_bytes()
        self.dados = json.loads(vocabulario.decode("utf-8"))
        self.dados["formato"] = "conversas_v1"
        validar_vocabulario(self.dados)
        if len(self.dados["vocabulario"]) != self.manifesto["tamanho_vocabulario"]:
            raise ValueError("O vocabulario nao corresponde ao manifesto")
        # A assinatura considera tambem os caracteres e marcadores efetivamente carregados
        self.assinatura = hashlib.sha256(conteudo + b"\0" + vocabulario).hexdigest()
        self.limite_cache = limite_cache
        self._cache = OrderedDict()
        self.acumulados = {}
        arquivos_vistos = set()
        for parte in ("treino", "validacao"):
            resumo = self.manifesto["partes"][parte]
            total_parte = 0
            for fonte, info in resumo["fontes"].items():
                acumulados = []
                total_fonte = 0
                for bloco in info["blocos"]:
                    arquivo = self.resolver_arquivo(bloco["arquivo"])
                    if arquivo in arquivos_vistos:
                        raise ValueError("Um arquivo foi repetido no manifesto")
                    arquivos_vistos.add(arquivo)
                    quantidade = bloco["exemplos"]
                    if type(quantidade) is not int or not 1 <= quantidade <= self.manifesto["exemplos_por_bloco"]:
                        raise ValueError("A quantidade de exemplos do bloco e invalida")
                    total_fonte += quantidade
                    acumulados.append(total_fonte)
                if total_fonte != info["total"] or total_fonte != self.manifesto["fontes"][fonte][parte]:
                    raise ValueError("As contagens por fonte do manifesto nao correspondem")
                self.acumulados[parte, fonte] = acumulados
                total_parte += total_fonte
            if total_parte != resumo["total"] or total_parte < 1:
                raise ValueError("As contagens da divisao do manifesto nao correspondem")

    # Aceita somente arquivos que permanecem dentro da pasta do conjunto preparado
    def resolver_arquivo(self, relativo: str) -> Path:
        caminho = (self.pasta / relativo).resolve()
        if not caminho.is_relative_to(self.pasta) or not caminho.is_file():
            raise ValueError(f"Arquivo ausente ou fora da pasta dos blocos: {relativo}")
        return caminho

    # Informa quantos blocos estao na RAM para acompanhar o limite do cache
    @property
    def blocos_em_memoria(self) -> int:
        return len(self._cache)

    # Le e valida um bloco, removendo o menos recente antes de carregar outro
    def carregar_bloco(self, parte: str, fonte: str, numero: int) -> list[dict] | BlocoConversasCompletas:
        bloco = self.manifesto["partes"][parte]["fontes"][fonte]["blocos"][numero]
        chave = bloco["arquivo"]
        if chave in self._cache:
            self._cache.move_to_end(chave)
            return self._cache[chave]
        while len(self._cache) >= self.limite_cache:
            self._cache.popitem(last=False)
        arquivo = self.resolver_arquivo(chave)
        if self.manifesto.get("esquema_exemplos") == "conversas_completas_v2":
            conteudo = arquivo.read_bytes()
            if hashlib.sha256(conteudo).hexdigest() != bloco["sha256"]:
                raise ValueError(f"O bloco foi alterado ou esta incompleto: {chave}")
            linhas = conteudo.splitlines()
            if len(linhas) != bloco["exemplos"] or any(not linha for linha in linhas):
                raise ValueError(f"A quantidade de conversas nao corresponde ao bloco: {chave}")
            exemplos = BlocoConversasCompletas(linhas, self.dados, fonte, parte,
                                              self.manifesto["fracao_treino"], self.manifesto["semente"])
            self._cache[chave] = exemplos
            return exemplos
        with arquivo.open("rb") as entrada:
            if hashlib.file_digest(entrada, "sha256").hexdigest() != bloco["sha256"]:
                raise ValueError(f"O bloco foi alterado ou esta incompleto: {chave}")
        registros = []
        with arquivo.open(encoding="utf-8") as entrada:
            for linha in entrada:
                if len(registros) >= bloco["exemplos"]:
                    raise ValueError(f"O bloco possui mais exemplos que o manifesto: {chave}")
                registros.append(json.loads(linha))
        if len(registros) != bloco["exemplos"]:
            raise ValueError(f"O bloco possui menos exemplos que o manifesto: {chave}")
        exemplos = validar_exemplos(registros, self.dados, fonte)
        marcadores = self.dados["marcadores"]
        vocabulario = self.dados["vocabulario"]
        for registro in registros:
            inicio = registro["inicio_resposta"]
            pergunta = "".join(vocabulario[i] for i in registro["ids"][len(marcadores["usuario"]):inicio - len(marcadores["resposta"])])
            prevista, grupo = definir_divisao(pergunta, self.manifesto["fracao_treino"], self.manifesto["semente"])
            if prevista != parte or grupo != registro["grupo_pergunta"]:
                raise ValueError(f"A pergunta esta na divisao errada: {chave}")
        self._cache[chave] = exemplos
        return exemplos


class ParteBlocos:
    # Representa treino ou validacao compartilhando o mesmo cache limitado de arquivos
    def __init__(self, leitor: LeitorBlocos, parte: str):
        self.leitor = leitor
        self.parte = parte
        self.resumo = leitor.manifesto["partes"][parte]
        self.fontes = sorted(fonte for fonte, info in self.resumo["fontes"].items() if info["total"] > 0)

    # Retorna a quantidade total de exemplos sem abrir os blocos
    def __len__(self) -> int:
        return self.resumo["total"]

    # Escolhe a fonte igualmente e um indice global que respeita os tamanhos reais dos blocos
    def sortear(self, gerador: torch.Generator | None = None, fonte: str | None = None) -> dict:
        if fonte is None:
            indice_fonte = torch.randint(len(self.fontes), (1,), generator=gerador).item()
            fonte = self.fontes[indice_fonte]
        elif fonte not in self.fontes:
            raise ValueError(f"A fonte {fonte} nao possui exemplos em {self.parte}")
        total = self.resumo["fontes"][fonte]["total"]
        indice = torch.randint(total, (1,), generator=gerador).item()
        acumulados = self.leitor.acumulados[self.parte, fonte]
        numero = bisect_right(acumulados, indice)
        inicio = acumulados[numero - 1] if numero else 0
        exemplo = self.leitor.carregar_bloco(self.parte, fonte, numero)[indice - inicio]
        if len(exemplo["ids"]) <= TAMANHO_CONTEXTO + 1:
            return exemplo
        # Sorteia um trecho da resposta, mantendo os caracteres anteriores como contexto
        tamanho_trecho = max(1, TAMANHO_CONTEXTO // 2)
        quantidade_alvos = len(exemplo["ids"]) - exemplo["inicio_resposta"]
        # Sorteia um alvo para que a ultima janela curta nao receba peso excessivo
        janela = torch.randint(quantidade_alvos, (1,), generator=gerador).item() // tamanho_trecho
        alvo_inicio = exemplo["inicio_resposta"] + janela * tamanho_trecho
        fim = min(len(exemplo["ids"]), alvo_inicio + tamanho_trecho)
        inicio = max(0, fim - TAMANHO_CONTEXTO - 1)
        return {"ids": exemplo["ids"][inicio:fim], "inicio_resposta": alvo_inicio - inicio, "fonte": fonte}
