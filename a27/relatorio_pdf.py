"""Gera um PDF a partir de um Markdown simples.

Suporta: títulos (#, ##, ###), parágrafos, listas (-), tabelas (| a | b |),
imagens (![legenda](arquivo.png)), **negrito** e `código`.

Uso: python relatorio_pdf.py RELATORIO.md relatorio.pdf
"""

import html
import re
import sys
from pathlib import Path

from fpdf import FPDF

_FONTES_WINDOWS = {"": "arial.ttf", "B": "arialbd.ttf", "I": "ariali.ttf", "BI": "arialbi.ttf"}
_SEM_UNICODE = {"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-", "…": "...", "→": "->",
                "×": "x", "≤": "<=", "≥": ">=", "·": "-", "α": "alfa", "Σ": "soma", "≈": "~"}


def _fonte(pdf: FPDF) -> tuple[str, bool]:
    """Usa Arial (Unicode) quando existe no sistema; senão Helvetica com texto em latin-1."""
    pasta = Path("C:/Windows/Fonts")
    if all((pasta / arq).exists() for arq in _FONTES_WINDOWS.values()):
        for estilo, arq in _FONTES_WINDOWS.items():
            pdf.add_font("Arial", estilo, str(pasta / arq))
        return "Arial", False
    return "helvetica", True


def _inline(texto: str) -> str:
    texto = html.escape(texto, quote=False)
    texto = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", texto)
    return re.sub(r"`(.+?)`", r"<i>\1</i>", texto)


def _celula(texto: str) -> str:
    """Célula de tabela: o fpdf2 não aceita tags aninhadas em <td>, então fica texto puro."""
    return html.escape(re.sub(r"\*\*(.+?)\*\*|`(.+?)`", lambda m: m.group(1) or m.group(2), texto), quote=False)


def markdown_para_html(md: str, base: Path) -> str:
    partes, lista, tabela, paragrafo = [], [], [], []

    def fechar():
        if paragrafo:
            partes.append(f"<p>{_inline(' '.join(paragrafo))}</p>")
            paragrafo.clear()
        if lista:
            partes.append("<ul>" + "".join(f"<li>{_inline(i)}</li>" for i in lista) + "</ul>")
            lista.clear()
        if tabela:
            linhas = [row for row in tabela if not re.fullmatch(r"\|[\s:|-]+\|", row)]
            celulas = [[c.strip() for c in row.strip("|").split("|")] for row in linhas]
            cab = "".join(f"<th>{_celula(c)}</th>" for c in celulas[0])
            corpo = "".join("<tr>" + "".join(f"<td>{_celula(c)}</td>" for c in r) + "</tr>" for r in celulas[1:])
            partes.append(f'<table width="100%"><thead><tr>{cab}</tr></thead><tbody>{corpo}</tbody></table>')
            tabela.clear()

    for linha in md.splitlines():
        s = linha.strip()
        img = re.fullmatch(r"!\[(.*?)\]\((.+?)\)", s)
        if not s:
            fechar()
        elif s.startswith("#"):
            fechar()
            nivel = len(s) - len(s.lstrip("#"))
            partes.append(f"<h{nivel}>{_inline(s[nivel:].strip())}</h{nivel}>")
        elif img:
            fechar()
            partes.append(f'<center><img src="{(base / img.group(2)).as_posix()}" width="420"></center>')
            if img.group(1):
                partes.append(f"<p><i>{_inline(img.group(1))}</i></p>")
        elif s.startswith("|"):
            if paragrafo or lista:
                fechar()
            tabela.append(s)
        elif s.startswith("- "):
            if paragrafo or tabela:
                fechar()
            lista.append(s[2:])
        else:
            if tabela or lista:
                fechar()
            paragrafo.append(s)
    fechar()
    return "\n".join(partes)


def gerar_pdf(md_path, pdf_path, tamanho_fonte: int = 10, data=None) -> int:
    """Converte o Markdown em PDF e devolve o número de páginas.

    `data` (datetime com fuso) fixa a data de criação gravada no PDF; com ela, o mesmo Markdown gera sempre o mesmo arquivo.
    """
    md_path, pdf_path = Path(md_path), Path(pdf_path)
    pdf = FPDF(format="A4")
    if data is not None:
        pdf.set_creation_date(data)
    pdf.set_margins(18, 15, 18)
    pdf.set_auto_page_break(True, margin=15)
    familia, latin1 = _fonte(pdf)
    texto = md_path.read_text(encoding="utf-8")
    if latin1:
        for de, para in _SEM_UNICODE.items():
            texto = texto.replace(de, para)
        texto = texto.encode("latin-1", "replace").decode("latin-1")
    pdf.add_page()
    pdf.set_font(familia, size=tamanho_fonte)
    pdf.write_html(markdown_para_html(texto, md_path.parent),
                   tag_styles=None, font_family=familia)
    pdf.output(str(pdf_path))
    return pdf.pages_count


if __name__ == "__main__":
    print(gerar_pdf(sys.argv[1], sys.argv[2]), "página(s)")
