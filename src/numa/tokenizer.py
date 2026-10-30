# faz a conversao entre texto e IDs quando executado sozinho, 
# permite experimentar isso pelo terminal, mas não salva os resultados

class TokenizadorCaracteres:
    # Cria um vocabulario com os caracteres encontrados no texto fornecido

    def __init__(self, texto: str) -> None:
        if not texto:
            raise ValueError("O texto usado para criar o vocabulário não pode ser vazio")

        # A ordenacao mantem os mesmos IDs para o mesmo conjunto de caracteres
        self.caracteres = tuple(sorted(set(texto)))
        self._caractere_para_id = {
            caractere: indice for indice, caractere in enumerate(self.caracteres)
        }

    @property
    def tamanho_vocabulario(self) -> int:
        # Quantidade de caracteres que o tokenizer conhece
        return len(self.caracteres)

    def encode(self, texto: str) -> list[int]:
        # Converte cada caractere do texto em seu ID
        try:
            return [self._caractere_para_id[caractere] for caractere in texto]
        except KeyError as erro:
            raise ValueError(
                f"Caractere {erro.args[0]!r} não existe no vocabulário"
            ) from erro

    def decode(self, ids: list[int]) -> str:
        # Reconstroi um texto a partir dos IDs dos caracteres
        texto = []
        for indice in ids:
            if not isinstance(indice, int) or not 0 <= indice < self.tamanho_vocabulario:
                raise ValueError(f"ID {indice!r} não existe no vocabulário")
            texto.append(self.caracteres[indice])
        return "".join(texto)


def main() -> None:
    # Permite experimentar o tokenizer no terminal
    print("Digite um texto para ver seus IDs. Pressione Enter sem texto para sair")
    while True:
        try:
            texto = input("Texto: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not texto:
            break

        tokenizer = TokenizadorCaracteres(texto)
        ids = tokenizer.encode(texto)
        print("Vocabulário:", dict(enumerate(tokenizer.caracteres)))
        print("IDs:", ids)
        print("Texto reconstruído:", tokenizer.decode(ids))


if __name__ == "__main__":
    main()
