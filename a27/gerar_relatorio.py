"""Gera RELATORIO.md, a figura de comparação de ranks e relatorio.pdf (máx. 2 páginas) com os números do próprio app.

Uso: python gerar_relatorio.py   (carrega os modelos de embedding reais)
"""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

from healthsearch_app import (  # noqa: E402
    EXEMPLOS, GABARITO, MODELO_PADRAO, MODELOS, avaliar, avaliar_consultas, buscar, carregar_modelo, embeddings_documentos,
    separacao,
)
from relatorio_pdf import gerar_pdf  # noqa: E402

PASTA = Path(__file__).resolve().parent
DATA = datetime(2026, 9, 28, tzinfo=timezone.utc)
MOTORES = [("BM25", "rank_bm25", "#1f77b4"), ("Semântico", "rank_semantico", "#ff7f0e"), ("Híbrido RRF", "rank_rrf", "#2ca02c")]


def posicao_relevante(r, coluna, relevantes):
    """Melhor posição de um documento relevante; None se o BM25 não o recuperou (score 0)."""
    t = r["tabela"]
    linhas = t[t["id"].isin(relevantes)]
    if coluna == "rank_bm25":
        linhas = linhas[linhas["score_bm25"] > 0]
    return int(linhas[coluna].min()) if len(linhas) else None


def figura(modelo, E):
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    largura = 0.26
    for k, (nome, coluna, cor) in enumerate(MOTORES):
        for j, consulta in enumerate(EXEMPLOS):
            p = posicao_relevante(buscar(consulta, modelo, E), coluna, GABARITO[consulta][1])
            x = j + (k - 1) * largura
            if p is None:
                ax.bar(x, 7, largura, color="white", edgecolor=cor, hatch="//")
            else:
                ax.bar(x, p, largura, color=cor)
    ax.set_xticks(range(len(EXEMPLOS)), list(EXEMPLOS), fontsize=8)
    ax.set_ylabel("posição do 1º relevante")
    ax.set_ylim(0, 8)
    ax.set_yticks(range(1, 7))
    ax.legend(handles=[Patch(color=cor, label=nome) for nome, _, cor in MOTORES]
              + [Patch(facecolor="white", edgecolor="gray", hatch="//", label="não recuperado")],
              fontsize=7, ncol=4, loc="upper right")
    ax.set_title("Posição do 1º documento relevante (1 = topo; barra hachurada = BM25 não recuperou)", fontsize=9)
    plt.tight_layout()
    fig.savefig(PASTA / "comparacao_ranks.png", dpi=150)
    plt.close(fig)


def tabela_metricas(aval):
    linhas = ["| Consultas | Motor | P@1 | MRR | Recall@3 |", "|---|---|---|---|---|"]
    for (tipo, motor), v in aval.iterrows():
        linhas.append(f"| {tipo} | {motor} | " + " | ".join(f"{v[c]:.2f}".replace(".", ",") for c in ["P@1", "MRR", "Recall@3"]) + " |")
    return "\n".join(linhas)


def montar_markdown() -> str:
    modelo = carregar_modelo(MODELO_PADRAO)
    E = embeddings_documentos(modelo, MODELO_PADRAO)
    figura(modelo, E)
    aval = avaliar(modelo, E)
    # maior α que ainda dá o melhor MRR do híbrido
    melhor_alfa = max(np.round(np.linspace(0, 1, 11), 2), key=lambda a: (round(avaliar(modelo, E, alfa=a).loc[("todas", "Híbrido RRF"), "MRR"], 9), a))
    outro = [m for m in MODELOS if m != MODELO_PADRAO][0]
    m2 = carregar_modelo(outro)
    E2 = embeddings_documentos(m2, outro)
    aval2 = avaliar(m2, E2, outro)
    rr2 = avaliar_consultas(m2, E2, outro).pivot(index="consulta", columns="motor", values="RR")
    erros_sem = ", ".join(f"“{q}”" for q in GABARITO if rr2.at[q, "Semântico"] < 1)
    ganha = ", ".join(f"“{q}”" for q in GABARITO if rr2.at[q, "Híbrido RRF"] > rr2.at[q, "Semântico"])
    perde = ", ".join(f"“{q}”" for q in GABARITO if rr2.at[q, "Híbrido RRF"] < rr2.at[q, "Semântico"])
    cod = buscar("CÓD-ECG-12D", modelo, E)["tabela"]
    sep_bm25, sep_sem = separacao(cod["score_bm25"], {1, 6}), separacao(cod["cosseno"], {1, 6})
    mrr = lambda a, tipo, motor: f"{a.loc[(tipo, motor), 'MRR']:.2f}".replace(".", ",")
    return f"""# HealthSearch — Relatório Técnico

Laboratório Prático 05 · Tendências em Ciência da Computação · Prof. Me. Ricardo Roberto de Lima · **Equipe:** Victor (entrega individual) · {DATA:%d/%m/%Y}.

## 1. Arquitetura

Um único arquivo Streamlit (`healthsearch_app.py`). A lógica fica em funções puras, uma por fase, e a interface em `main()`.

| Fase | Implementação |
|---|---|
| 1. Corpus e pré-processamento | as 6 diretrizes em hardcode (ID, título, trecho); minúsculas e sem acentos, remoção de caracteres especiais (o hífen dos códigos fica) e das stopwords em português. Códigos entram inteiros e em partes: “ECG-12D” acha “CÓD-ECG-12D”. |
| 2. BM25 | `rank_bm25.BM25Okapi`, refeito a cada mudança dos sliders k1 (0 a 3, padrão 1,2) e b (0 a 1, padrão 0,75). Com k1 = 0 a biblioteca calcula 0/0 nos documentos sem o termo; o app soma termo a termo e usa o limite correto (0). |
| 3. Semântico | `sentence-transformers` com `{MODELO_PADRAO.split("/")[-1]}` (prefixos “query: ” e “passage: ”) e similaridade de cosseno. Alternativa na barra lateral: `{outro.split("/")[-1]}`. |
| 4. RRF | Score(D) = α/(60 + Rank_BM25) + (1 − α)/(60 + Rank_Sem), com ranks a partir de 1. Documentos empatados (por exemplo, todos com BM25 = 0) dividem a mesma posição, para ninguém ser favorecido pelo ID. |
| Interface | abas Léxico (BM25), Semântico, Híbrido RRF e Matriz Comparativa; sliders de k1, b e α; botões com as consultas dos pontos cegos; tempo de cada motor; métricas P@1, MRR e Recall@3 num gabarito de {len(GABARITO)} consultas; curva do MRR em função de α. |

## 2. Comparação de ranks

![Figura 1 — posição do 1º documento relevante nas consultas de demonstração (modelo {MODELO_PADRAO.split("/")[-1]}, α = 0,5).](comparacao_ranks.png)

- **Ponto cego do BM25:** em “ataque cardíaco”, “AVC” e “AAS 100mg” nenhuma diretriz contém os termos, e o BM25 não recupera nada. Em “pressão alta”, a palavra “alta” leva o BM25 à reanimação (“alta qualidade”). O semântico e o híbrido trazem o infarto, o AVC isquêmico e a crise hipertensiva.
- **Ponto cego do semântico:** em “CÓD-ECG-12D” ele acerta a ordem, mas dá notas parecidas a tudo: a separação entre os relevantes e o resto é de {sep_sem:.0%} da nota máxima, contra {sep_bm25:.0%} no BM25, que zera quem não tem o código.
- **Híbrido RRF:** cobre os dois casos. Quando os motores discordam em posições espelhadas, α = 0,5 gera empate (“pressão alta”); α menor dá preferência ao semântico.

## 3. Métricas de desempenho

Gabarito de {len(GABARITO)} consultas com as diretrizes relevantes marcadas à mão: metade na linguagem do plantão (sinônimos e siglas), metade com termos técnicos e códigos. Nos empates, vale a média sobre as ordens possíveis.

{tabela_metricas(aval)}

Com o e5, o semântico já acerta todo o gabarito, e o híbrido chega a MRR 1,0 com α ≤ {f"{melhor_alfa:.1f}".replace(".", ",")}. Com o `{outro.split("/")[-1]}`, o semântico cai para MRR {mrr(aval2, "todas", "Semântico")} (não põe um relevante em 1º em {erros_sem}). O híbrido fica com {mrr(aval2, "todas", "Híbrido RRF")}: o que ele ganha em {ganha} (o BM25 acerta) perde em {perde} (o BM25 erra). **Conclusão:** num corpus pequeno, o RRF funciona como seguro, porque nunca cai para o MRR {mrr(aval, "sinônimo ou sigla", "BM25")} do BM25 em sinônimos e mantém a precisão em códigos. Mas ele não supera o melhor motor sozinho quando os dois discordam só no topo.

## 4. Divisão de tarefas

| Tarefa | Responsável |
|---|---|
| Pré-processamento, BM25 e RRF | Victor |
| Motor semântico e escolha do modelo | Victor |
| Interface Streamlit e métricas | Victor |
| Relatório | Victor |
"""


if __name__ == "__main__":
    md = PASTA / "RELATORIO.md"
    md.write_text(montar_markdown(), encoding="utf-8")
    print(gerar_pdf(md, PASTA / "relatorio.pdf", data=DATA), "página(s)")
