"""Gerador de Autômato Finito Determinístico (AFD) mínimo.

Trabalho de Linguagens Formais e Autômatos — Ciência da Computação, UFFS/Chapecó.

A partir de um arquivo com tokens e/ou Gramáticas Regulares, a aplicação gera um
AFD mínimo (livre de estados inalcançáveis e mortos), com estado de erro. Tudo em
um único arquivo, organizado nas seções:

    1. Estrutura de dados (classe Automato)
    2. Carga            -> arquivo -> AFND
    3. Determinização   -> AFND -> AFD (construção de subconjuntos)
    4. Minimização      -> remove estados inalcançáveis e mortos (sem classes de equivalência)
    5. Estado de erro   -> completa a função de transição
    6. Tabelas          -> formatação da saída
    7. CLI              -> interface de linha de comando

Uso:
    python afd.py <arquivo_entrada> [-o saida.txt] [-t palavra1 palavra2 ...]
"""

from __future__ import annotations

import argparse
import re
import string
import sys

# Símbolo de transição vazia (epsilon). Nas gramáticas regulares deste trabalho o
# epsilon só marca um não-terminal como final, então normalmente NÃO há transições
# em epsilon; ainda assim o suporte fica pronto (determinização com ε-fecho).
EPSILON = "ε"


# ====================================================================== #
# 1. Estrutura de dados
# ====================================================================== #
class Automato:
    """Representa tanto o AFND quanto o AFD em todas as etapas.

    Transições em dicionário: (estado, símbolo) -> conjunto de estados destino.
    Quando todo conjunto destino tem no máximo um elemento, o autômato é
    determinístico.
    """

    def __init__(self):
        self.estados: set[str] = set()
        self.alfabeto: set[str] = set()
        self.transicoes: dict[tuple[str, str], set[str]] = {}
        self.inicial: str | None = None
        self.finais: set[str] = set()
        # estado final -> conjunto de tokens/sentenças que ele reconhece
        self.rotulos: dict[str, set[str]] = {}

    # -- construção -- #
    def adicionar_estado(self, estado: str) -> None:
        self.estados.add(estado)

    def adicionar_transicao(self, origem: str, simbolo: str, destino: str) -> None:
        self.estados.add(origem)
        self.estados.add(destino)
        if simbolo != EPSILON:
            self.alfabeto.add(simbolo)
        self.transicoes.setdefault((origem, simbolo), set()).add(destino)

    def marcar_final(self, estado: str, *rotulos: str) -> None:
        self.estados.add(estado)
        self.finais.add(estado)
        for rotulo in rotulos:
            if rotulo:
                self.rotulos.setdefault(estado, set()).add(rotulo)

    # -- consulta -- #
    def destinos(self, estado: str, simbolo: str) -> set[str]:
        return self.transicoes.get((estado, simbolo), set())

    def eh_deterministico(self) -> bool:
        for (_, simbolo), destinos in self.transicoes.items():
            if simbolo == EPSILON or len(destinos) > 1:
                return False
        return True

    def simbolos_ordenados(self) -> list[str]:
        return sorted(self.alfabeto)

    def rotulo_de(self, estado: str) -> str:
        return ", ".join(sorted(self.rotulos.get(estado, set())))

    # -- simulação (executa o autômato sobre uma palavra) -- #
    def aceita(self, palavra: str) -> tuple[bool, set[str]]:
        """Retorna (aceita?, tokens_reconhecidos). Pressupõe autômato determinístico."""
        if self.inicial is None:
            return False, set()
        atual = self.inicial
        for ch in palavra:
            destinos = self.destinos(atual, ch)
            if not destinos:
                return False, set()
            atual = next(iter(destinos))
        return (atual in self.finais), set(self.rotulos.get(atual, set()))

    def copia(self) -> "Automato":
        novo = Automato()
        novo.estados = set(self.estados)
        novo.alfabeto = set(self.alfabeto)
        novo.transicoes = {chave: set(dest) for chave, dest in self.transicoes.items()}
        novo.inicial = self.inicial
        novo.finais = set(self.finais)
        novo.rotulos = {e: set(r) for e, r in self.rotulos.items()}
        return novo


# ====================================================================== #
# 2. Carga: arquivo -> AFND
# ====================================================================== #
# Regras (conforme o enunciado):
#   * um único estado inicial (S), compartilhado por todos os tokens/gramáticas;
#   * para um token, cada símbolo cria um NOVO estado destino (sem reuso); o último
#     estado é final e reconhece aquele token;
#   * em uma gramática, cada não-terminal vira um estado próprio da gramática;
#     a produção `ε` marca o não-terminal como final; `<X> ::= a<Y>` cria X --a--> Y;
#   * o rótulo (token reconhecido) de uma gramática é o nome do seu símbolo inicial.

NOME_INICIAL = "S"

# Letras para os demais estados do AFND, na ordem de criação (A, B, C, ...).
# A letra 'S' fica reservada ao estado inicial, por isso é omitida.
_LETRAS = "".join(c for c in string.ascii_uppercase if c != "S")

# Representações aceitas para epsilon dentro de uma produção.
_EPSILONS = {"ε", "&", "eps", "epsilon", "lambda", "λ", ""}

# Captura uma sequência de terminais seguida (opcionalmente) de um não-terminal ao
# final, ex.: 'a<A>' -> ('a', '<A>'); 'abc' -> ('abc', None).
_RE_ALTERNATIVA = re.compile(r"^([^<>]*)(<[^<>]+>)?$")

# Linha de gramática: começa com um não-terminal seguido de '::=', ex.: '<S> ::= ...'.
# (Distingue uma produção de um token que apenas contenha '::=', como o operador '::='.)
_RE_GRAMATICA = re.compile(r"^<[^<>]+>\s*::=")


class _Contador:
    """Gera nomes de estados novos na ordem de criação: A, B, C, ... Z, A1, B1, ..."""

    def __init__(self):
        self.i = 0

    def novo(self) -> str:
        ciclo, pos = divmod(self.i, len(_LETRAS))
        self.i += 1
        letra = _LETRAS[pos]
        return letra if ciclo == 0 else f"{letra}{ciclo}"


def carregar_arquivo(caminho: str) -> Automato:
    with open(caminho, "r", encoding="utf-8") as arq:
        return carregar(arq.readlines())


def carregar(linhas: list[str]) -> Automato:
    """Constrói o AFND a partir das linhas de entrada."""
    afnd = Automato()
    afnd.inicial = NOME_INICIAL
    afnd.adicionar_estado(NOME_INICIAL)
    contador = _Contador()

    # Remove linhas em branco, separadores (----, ====) e comentários (# ...).
    uteis = [ln.strip() for ln in linhas if not _eh_ignoravel(ln)]

    i = 0
    while i < len(uteis):
        if _eh_gramatica(uteis[i]):
            # Agrupa linhas consecutivas de gramática em um único bloco.
            bloco = []
            while i < len(uteis) and _eh_gramatica(uteis[i]):
                bloco.append(uteis[i])
                i += 1
            _carregar_gramatica(afnd, bloco, contador)
        else:
            _carregar_token(afnd, uteis[i], contador)
            i += 1
    return afnd


def _eh_gramatica(linha: str) -> bool:
    """Verdadeiro se a linha é uma produção de gramática ('<NT> ::= ...')."""
    return _RE_GRAMATICA.match(linha) is not None


def _eh_ignoravel(linha: str) -> bool:
    s = linha.strip()
    if s == "" or s.startswith("#"):
        return True
    # Linha separadora: 3+ caracteres feitos só de '-' ou '=' (ex.: ----, ====).
    # O limiar de comprimento evita descartar operadores curtos como '=', '==', '--'.
    return len(s) >= 3 and set(s) <= {"-", "="}


def _nome_nt(texto: str) -> str:
    """Extrai 'S' de '<S>'."""
    return texto.strip().lstrip("<").rstrip(">").strip()


def _carregar_token(afnd: Automato, palavra: str, contador: _Contador) -> None:
    """Cadeia de transições de um token literal (cada símbolo, um estado novo)."""
    atual = afnd.inicial
    for ch in palavra:
        destino = contador.novo()
        afnd.adicionar_transicao(atual, ch, destino)
        atual = destino
    afnd.marcar_final(atual, palavra)


def _carregar_gramatica(afnd: Automato, bloco: list[str], contador: _Contador) -> None:
    """Processa um bloco de produções (uma gramática regular)."""
    producoes: list[tuple[str, list[str]]] = []
    for linha in bloco:
        lhs, rhs = linha.split("::=", 1)
        producoes.append((_nome_nt(lhs), [alt.strip() for alt in rhs.split("|")]))

    simbolo_inicial = producoes[0][0]
    rotulo = simbolo_inicial  # token reconhecido = nome do símbolo inicial

    # O símbolo inicial usa o estado inicial global; os demais não-terminais
    # recebem estados novos, exclusivos desta gramática.
    mapa_nt: dict[str, str] = {simbolo_inicial: afnd.inicial}

    def estado_de(nt: str) -> str:
        if nt not in mapa_nt:
            mapa_nt[nt] = contador.novo()
        return mapa_nt[nt]

    for nt, alternativas in producoes:
        origem = estado_de(nt)
        for alt in alternativas:
            _processar_alternativa(afnd, origem, alt, estado_de, contador, rotulo)


def _processar_alternativa(afnd, origem, alt, estado_de, contador, rotulo) -> None:
    alt = alt.strip()

    if alt in _EPSILONS:  # produção epsilon -> 'origem' é final
        afnd.marcar_final(origem, rotulo)
        return

    casamento = _RE_ALTERNATIVA.match(alt)
    if casamento is None:
        raise ValueError(f"Produção inválida: {alt!r}")
    terminais, nt = casamento.group(1), casamento.group(2)

    destino_nt = estado_de(_nome_nt(nt)) if nt is not None else None

    if nt is not None and terminais == "":
        # <X> ::= <Y> (transição vazia entre não-terminais)
        afnd.adicionar_transicao(origem, EPSILON, destino_nt)
        return

    # Cadeia de terminais. Havendo não-terminal, o ÚLTIMO terminal leva ao estado
    # dele; caso contrário, a cadeia termina em um novo estado final.
    atual = origem
    for k, ch in enumerate(terminais):
        ultimo = k == len(terminais) - 1
        destino = destino_nt if (ultimo and nt is not None) else contador.novo()
        afnd.adicionar_transicao(atual, ch, destino)
        atual = destino
    if nt is None:
        afnd.marcar_final(atual, rotulo)


# ====================================================================== #
# 3. Determinização: AFND -> AFD (construção de subconjuntos)
# ====================================================================== #
def epsilon_fecho(afnd: Automato, conjunto: set[str]) -> set[str]:
    pilha = list(conjunto)
    fecho = set(conjunto)
    while pilha:
        estado = pilha.pop()
        for destino in afnd.destinos(estado, EPSILON):
            if destino not in fecho:
                fecho.add(destino)
                pilha.append(destino)
    return fecho


def mover(afnd: Automato, conjunto: set[str], simbolo: str) -> set[str]:
    resultado: set[str] = set()
    for estado in conjunto:
        resultado |= afnd.destinos(estado, simbolo)
    return resultado


def nome_subconjunto(conjunto) -> str:
    """Nome de um estado do AFD: a notação do subconjunto, ex.: {A,H}, {C,M}, {S}."""
    return "{" + ",".join(sorted(conjunto)) + "}"


def determinizar(afnd: Automato) -> Automato:
    afd = Automato()
    # O alfabeto do AFND já não contém epsilon (ver Automato.adicionar_transicao).
    afd.alfabeto = set(afnd.alfabeto)
    alfabeto = sorted(afd.alfabeto)

    inicial = frozenset(epsilon_fecho(afnd, {afnd.inicial}))
    afd.inicial = nome_subconjunto(inicial)

    pendentes = [inicial]
    visitados: set[frozenset[str]] = set()

    while pendentes:
        atual = pendentes.pop()
        if atual in visitados:
            continue
        visitados.add(atual)

        nome_atual = nome_subconjunto(atual)
        afd.adicionar_estado(nome_atual)

        # Estado final do AFD se contiver algum final do AFND; reúne os rótulos
        # de todos os finais contidos no subconjunto.
        finais_contidos = atual & afnd.finais
        if finais_contidos:
            rotulos: set[str] = set()
            for estado in finais_contidos:
                rotulos |= afnd.rotulos.get(estado, set())
            afd.marcar_final(nome_atual, *rotulos)

        for simbolo in alfabeto:
            destino = frozenset(epsilon_fecho(afnd, mover(afnd, atual, simbolo)))
            if not destino:
                continue  # célula vazia; tratada na etapa do estado de erro
            afd.adicionar_transicao(nome_atual, simbolo, nome_subconjunto(destino))
            if destino not in visitados:
                pendentes.append(destino)

    return afd


# ====================================================================== #
# 4. Minimização: remove estados inalcançáveis e mortos
# ====================================================================== #
# Conforme o enunciado, a minimização restringe-se a deixar o AFD "livre de
# estados inalcançáveis e mortos", SEM aplicar classes de equivalência.
#   * inalcançável: não é atingido a partir do estado inicial;
#   * morto: não é final e não alcança nenhum estado final.

def estados_alcancaveis(afd: Automato) -> set[str]:
    alcancaveis = {afd.inicial}
    pilha = [afd.inicial]
    while pilha:
        estado = pilha.pop()
        for simbolo in afd.alfabeto:
            for destino in afd.destinos(estado, simbolo):
                if destino not in alcancaveis:
                    alcancaveis.add(destino)
                    pilha.append(destino)
    return alcancaveis


def estados_vivos(afd: Automato) -> set[str]:
    """Estados que alcançam algum final (os demais são 'mortos')."""
    vivos = set(afd.finais)
    mudou = True
    while mudou:
        mudou = False
        for (origem, _simbolo), destinos in afd.transicoes.items():
            if origem not in vivos and (destinos & vivos):
                vivos.add(origem)
                mudou = True
    return vivos


def _restringir(afd: Automato, manter: set[str]) -> Automato:
    """Cópia do AFD contendo apenas os estados em `manter`."""
    novo = Automato()
    novo.inicial = afd.inicial
    novo.alfabeto = set(afd.alfabeto)
    for estado in manter:
        novo.adicionar_estado(estado)
        if estado in afd.finais:
            novo.marcar_final(estado, *afd.rotulos.get(estado, set()))
    for (origem, simbolo), destinos in afd.transicoes.items():
        if origem not in manter:
            continue
        for destino in destinos:
            if destino in manter:
                novo.adicionar_transicao(origem, simbolo, destino)
    return novo


def remover_inalcancaveis(afd: Automato) -> Automato:
    return _restringir(afd, estados_alcancaveis(afd))


def remover_mortos(afd: Automato) -> Automato:
    # O estado inicial é sempre preservado (se for morto, a linguagem é vazia).
    return _restringir(afd, estados_vivos(afd) | {afd.inicial})


def minimizar(afd: Automato) -> Automato:
    return remover_mortos(remover_inalcancaveis(afd))


# ====================================================================== #
# 5. Estado de erro: completa a função de transição
# ====================================================================== #
# Toda célula não mapeada passa a apontar para o estado de erro, que é um estado
# armadilha (não final): todas as suas transições permanecem nele próprio.

NOME_ERRO = "qErro"


def adicionar_estado_erro(afd: Automato, nome: str = NOME_ERRO) -> Automato:
    completo = afd.copia()
    completo.adicionar_estado(nome)
    for estado in list(completo.estados):
        for simbolo in completo.alfabeto:
            if not completo.destinos(estado, simbolo):
                completo.adicionar_transicao(estado, simbolo, nome)
    return completo


# ====================================================================== #
# 6. Tabelas de transição (saída em texto)
# ====================================================================== #
# Convenções: -> inicial; * final; - célula sem transição. A última coluna mostra
# o(s) token(s) reconhecido(s).

def _ordenar_estados(afd: Automato) -> list[str]:
    """Inicial primeiro, demais em ordem, e o estado de erro por último."""
    outros = sorted(e for e in afd.estados if e != afd.inicial and e != NOME_ERRO)
    ordenados = ([afd.inicial] if afd.inicial is not None else []) + outros
    if NOME_ERRO in afd.estados:
        ordenados.append(NOME_ERRO)
    return ordenados


def _rotulo_estado(afd: Automato, estado: str) -> str:
    prefixo = "->" if estado == afd.inicial else "  "
    marca_final = "*" if estado in afd.finais else " "
    return f"{prefixo}{marca_final}{estado}"


def formatar_tabela(afd: Automato, titulo: str = "") -> str:
    simbolos = afd.simbolos_ordenados()
    estados = _ordenar_estados(afd)
    tem_rotulo = bool(afd.rotulos)

    cabecalho = ["Estado"] + simbolos + (["token"] if tem_rotulo else [])

    linhas: list[list[str]] = []
    for estado in estados:
        linha = [_rotulo_estado(afd, estado)]
        for simbolo in simbolos:
            destinos = afd.destinos(estado, simbolo)
            linha.append(", ".join(sorted(destinos)) if destinos else "-")
        if tem_rotulo:
            linha.append(afd.rotulo_de(estado))
        linhas.append(linha)

    larguras = [len(c) for c in cabecalho]
    for linha in linhas:
        for j, celula in enumerate(linha):
            larguras[j] = max(larguras[j], len(celula))

    def formata(celulas: list[str]) -> str:
        return " | ".join(c.ljust(larguras[j]) for j, c in enumerate(celulas))

    partes: list[str] = []
    if titulo:
        partes.append(titulo)
        partes.append("=" * len(titulo))
    partes.append(formata(cabecalho))
    partes.append("-+-".join("-" * w for w in larguras))
    for linha in linhas:
        partes.append(formata(linha))
    return "\n".join(partes)


# ====================================================================== #
# 7. Interface de linha de comando
# ====================================================================== #
def processar(caminho: str):
    """Executa o pipeline completo e devolve os 4 autômatos das etapas."""
    afnd = carregar_arquivo(caminho)
    afd = determinizar(afnd)
    afd_min = minimizar(afd)
    afd_final = adicionar_estado_erro(afd_min)
    return afnd, afd, afd_min, afd_final


def montar_saida(afnd, afd, afd_min, afd_final) -> str:
    blocos = [
        formatar_tabela(afnd, "1) AFND - apos a carga"),
        formatar_tabela(afd, "2) AFD - apos a determinizacao"),
        formatar_tabela(afd_min, "3) AFD minimo - sem estados inalcancaveis e mortos"),
        formatar_tabela(afd_final, "4) AFD final - com estado de erro"),
    ]
    return "\n\n\n".join(blocos)


def montar_testes(afd_final, palavras: list[str]) -> str:
    linhas = ["Teste de reconhecimento (AFD final):", "-" * 36]
    for palavra in palavras:
        aceita, tokens = afd_final.aceita(palavra)
        if aceita:
            rotulo = ", ".join(sorted(tokens)) if tokens else "(sem rotulo)"
            linhas.append(f"  {palavra!r:15} ACEITA   -> token: {rotulo}")
        else:
            linhas.append(f"  {palavra!r:15} REJEITA")
    return "\n".join(linhas)


def _configura_utf8() -> None:
    # Garante a impressão de 'ε' e das setas '->' no Windows.
    for fluxo in (sys.stdout, sys.stderr):
        if hasattr(fluxo, "reconfigure"):
            try:
                fluxo.reconfigure(encoding="utf-8")
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    _configura_utf8()
    parser = argparse.ArgumentParser(
        description="Gera um AFD minimo com estado de erro a partir de tokens/gramaticas."
    )
    parser.add_argument("entrada", help="arquivo de entrada (tokens e/ou gramaticas regulares)")
    parser.add_argument("-o", "--saida", help="grava o resultado tambem em um arquivo")
    parser.add_argument(
        "-t", "--testar", nargs="*", metavar="PALAVRA",
        help="palavras para testar o reconhecimento no AFD final",
    )
    args = parser.parse_args(argv)

    try:
        afnd, afd, afd_min, afd_final = processar(args.entrada)
    except FileNotFoundError:
        print(f"Erro: arquivo nao encontrado: {args.entrada}", file=sys.stderr)
        return 1

    saida = montar_saida(afnd, afd, afd_min, afd_final)
    if args.testar:
        saida += "\n\n\n" + montar_testes(afd_final, args.testar)

    print(saida)
    if args.saida:
        with open(args.saida, "w", encoding="utf-8") as arq:
            arq.write(saida + "\n")
        print(f"\n[resultado gravado em {args.saida}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
