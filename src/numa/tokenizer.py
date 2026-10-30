# Converte textos em IDs de caracteres e reconstrui textos a partir desses IDs
# Quando executado no terminal, demonstra a conversao sem salvar os resultados

class TokenizadorCaracteres:
    # Cria um vocabulario com os caracteres encontrados no texto fornecido

    # Cria o vocabulario em ordem fixa e associa cada caractere a um ID
    def __init__(self, texto: str) -> None:
        if not texto:
            raise ValueError("O texto usado para criar o vocabulario nao pode ser vazio")

        # A ordenacao mantem os mesmos IDs para o mesmo conjunto de caracteres
        self.caracteres = tuple(sorted(set(texto)))
        self._caractere_para_id = {
            caractere: indice for indice, caractere in enumerate(self.caracteres)
        }

    # Retorna a quantidade de caracteres conhecidos pelo tokenizador
    @property
    def tamanho_vocabulario(self) -> int:
        return len(self.caracteres)

    # Converte os caracteres do texto em IDs e rejeita caracteres desconhecidos
    def codificar(self, texto: str) -> list[int]:
        try:
            return [self._caractere_para_id[caractere] for caractere in texto]
        except KeyError as erro:
            raise ValueError(
                f"Caractere {erro.args[0]!r} nao existe no vocabulario"
            ) from erro

    # Converte IDs validos nos caracteres correspondentes para reconstruir o texto
    def decodificar(self, ids: list[int]) -> str:
        texto = []
        for indice in ids:
            if not isinstance(indice, int) or not 0 <= indice < self.tamanho_vocabulario:
                raise ValueError(f"ID {indice!r} nao existe no vocabulario")
            texto.append(self.caracteres[indice])
        return "".join(texto)


# Permite digitar textos e visualizar o vocabulario, os IDs e a reconstrucao
def principal() -> None:
    print("Digite um texto para ver seus IDs. Pressione Enter sem texto para sair")
    while True:
        try:
            texto = input("Texto: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not texto:
            break

        tokenizador = TokenizadorCaracteres(texto)
        ids = tokenizador.codificar(texto)
        print("Vocabulario:", dict(enumerate(tokenizador.caracteres)))
        print("IDs:", ids)
        print("Texto reconstruido:", tokenizador.decodificar(ids))


if __name__ == "__main__":
    principal()
