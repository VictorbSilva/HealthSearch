"""HealthSearch — motor de busca híbrido (BM25 + busca semântica) com Reciprocal Rank Fusion.

Laboratório Prático 05 · Desafio Integrador. App Streamlit num único arquivo com as 4 fases:
  Fase 1: ingestão do corpus médico (as 6 diretrizes fixas) e pré-processamento: minúsculas,
          remoção de caracteres especiais e de stopwords em português;
  Fase 2: motor léxico Okapi BM25 (rank_bm25), com k1 e b ajustáveis na barra lateral;
  Fase 3: motor semântico: embeddings (sentence-transformers) e similaridade de cosseno;
  Fase 4: fusão RRF: Score(D) = α · 1/(k + Rank_BM25) + (1 − α) · 1/(k + Rank_Semântico), k = 60.

Rodar:  streamlit run healthsearch_app.py
O modelo semântico (intfloat/multilingual-e5-small, ≈ 470 MB) é baixado do Hugging Face na primeira
execução. Sem rede, defina HEALTHSEARCH_EMBEDDER=simulado: o motor semântico passa a usar a
simulação vetorial de VetorizadorSimulado (não entende sinônimos; serve só para abrir a interface).
"""

import hashlib
import itertools
import os
import re
import time
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

# ----------------------------------------------------------------------------- Fase 1: corpus (hardcode obrigatório)
DOCUMENTOS = [
    {"id": 1, "titulo": "Protocolo Emergência ECG",
     "conteudo": "Pacientes com dor precordial aguda e suspeita de síndrome coronariana devem realizar eletrocardiograma CÓD-ECG-12D em até 10 minutos."},
    {"id": 2, "titulo": "Guia de Farmacologia Cardíaca",
     "conteudo": "O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários reduz a mortalidade no infarto agudo do miocárdio."},
    {"id": 3, "titulo": "Diretriz de Hipertensão Arterial",
     "conteudo": "A crise hipertensiva severa requer administração de anti-hipertensivos venosos e monitoramento contínuo da pressão arterial na UTI."},
    {"id": 4, "titulo": "Manual de AVC Isquêmico",
     "conteudo": "O acidente vascular cerebral isquêmico agudo deve ser tratado com trombolíticos venosos em até quatro horas e meia do início dos sintomas."},
    {"id": 5, "titulo": "Protocolo de Reanimação RCR",
     "conteudo": "Parada cardiorrespiratória em adultos exige compressões torácicas contínuas de alta qualidade e desfibrilação precoce no código azul."},
    {"id": 6, "titulo": "Procedimentos de UTI Geral",
     "conteudo": "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, recomenda-se a monitorização cardíaca contínua por telemetria."},
]
IDS = [d["id"] for d in DOCUMENTOS]

# Stopwords do português, já sem acento (a normalização vem antes da remoção).
STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "da", "do", "das", "dos", "e", "em", "no", "na", "nos",
    "nas", "ao", "aos", "para", "pra", "por", "pelo", "pela", "pelos", "pelas", "com", "sem", "que", "se", "ou",
    "ate", "como", "mais", "muito", "seu", "sua", "seus", "suas", "ele", "ela", "eles", "elas", "este", "esta",
    "esse", "essa", "isso", "isto", "ja", "nao", "entre", "sobre", "ser", "foi", "sao", "tem", "ha", "me", "meu",
    "minha", "qual", "quando", "onde", "tambem", "so", "pode", "deve", "devem", "é", "e",
}

K_RRF = 60
K1_PADRAO, B_PADRAO, ALFA_PADRAO = 1.2, 0.75, 0.5
# modelo → prefixos (consulta, documento). Os modelos e5 foram treinados com "query: " e "passage: ".
MODELOS = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": ("", ""),
}
MODELO_PADRAO = "intfloat/multilingual-e5-small"

# Consultas de demonstração dos pontos cegos, com o que cada uma mostra.
EXEMPLOS = {
    "ataque cardíaco": "sinônimo: nenhuma diretriz usa essas palavras",
    "CÓD-ECG-12D": "código exato de exame",
    "pressão alta": "sinônimo com palavra ambígua (“alta”)",
    "AVC": "sigla que só aparece no título",
    "AAS 100mg": "sigla e dose de medicamento",
    "infarto": "termo que aparece literalmente",
}
# Julgamento de relevância (gabarito) usado nas métricas: consulta → (tipo, documentos relevantes).
# Metade é a linguagem do plantão (sinônimos, siglas, termos populares), metade são termos técnicos e códigos.
SINONIMO, EXATO = "sinônimo ou sigla", "termo exato ou código"
GABARITO = {
    "infarto": (SINONIMO, {2}), "ataque cardíaco": (SINONIMO, {1, 2}), "isquemia miocárdica": (SINONIMO, {1, 2}),
    "dor no peito": (SINONIMO, {1}), "AAS 100mg": (SINONIMO, {2}), "AVC": (SINONIMO, {4}),
    "derrame cerebral": (SINONIMO, {4}), "RCR": (SINONIMO, {5}), "parada cardíaca": (SINONIMO, {5}),
    "pressão alta": (SINONIMO, {3}),
    "CÓD-ECG-12D": (EXATO, {1, 6}), "ECG-12D": (EXATO, {1, 6}), "eletrocardiograma": (EXATO, {1, 6}),
    "trombolíticos": (EXATO, {4}), "crise hipertensiva": (EXATO, {3}), "telemetria": (EXATO, {6}),
    "antiagregantes plaquetários": (EXATO, {2}), "código azul": (EXATO, {5}),
}


# ----------------------------------------------------------------------------- Fase 1: pré-processamento
def normalizar(texto: str) -> str:
    """Minúsculas e sem acentos (NFD + descarte das marcas combinantes)."""
    decomposto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in decomposto if unicodedata.category(c) != "Mn")


def remover_especiais(texto: str) -> str:
    """Troca por espaço tudo que não é letra, dígito ou hífen dentro de palavra (códigos como cod-ecg-12d)."""
    texto = re.sub(r"[^a-z0-9\s-]", " ", texto)
    texto = re.sub(r"(?<![a-z0-9])-|-(?![a-z0-9])", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def etapas(texto: str) -> dict[str, list[str] | str]:
    """As etapas do pré-processamento, na ordem, para exibir na interface."""
    minusculas = normalizar(texto)
    limpo = remover_especiais(minusculas)
    tokens = limpo.split()
    sem_stop = [t for t in tokens if t not in STOPWORDS]
    finais = []
    for t in sem_stop:
        finais.append(t)
        if "-" in t:  # código composto: o token inteiro e as partes, para "ECG-12D" achar "CÓD-ECG-12D"
            finais += [p for p in t.split("-") if p and p not in STOPWORDS]
    return {
        "1. Minúsculas e sem acentos": minusculas,
        "2. Sem caracteres especiais": limpo,
        "3. Tokens": tokens,
        "4. Sem stopwords (+ partes dos códigos)": finais,
    }


def preprocessar(texto: str) -> list[str]:
    return etapas(texto)["4. Sem stopwords (+ partes dos códigos)"]


TOKENS_DOCS = [preprocessar(d["conteudo"]) for d in DOCUMENTOS]


# ----------------------------------------------------------------------------- Fase 2: BM25
def scores_bm25(consulta: str, k1: float = K1_PADRAO, b: float = B_PADRAO) -> np.ndarray:
    """Score Okapi BM25 de cada documento (na ordem de DOCUMENTOS); o índice é refeito com k1 e b."""
    return contribuicoes_bm25(consulta, k1, b).sum(axis=1).to_numpy(dtype=float)


def contribuicoes_bm25(consulta: str, k1: float = K1_PADRAO, b: float = B_PADRAO) -> pd.DataFrame:
    """Parcela de cada termo da consulta no score BM25 de cada documento (linhas = docs, colunas = termos).

    O score é calculado termo a termo porque, com k1 = 0, o rank_bm25 faz 0/0 nos documentos sem o termo;
    o limite correto dessa parcela é 0 (sem o termo, não há contribuição), então NaN vira 0.
    """
    bm25 = BM25Okapi(TOKENS_DOCS, k1=k1, b=b)
    tokens = preprocessar(consulta)
    with np.errstate(invalid="ignore", divide="ignore"):
        parcelas = {t: np.nan_to_num(bm25.get_scores([t]), nan=0.0) * tokens.count(t) for t in dict.fromkeys(tokens)}
    return pd.DataFrame(parcelas, index=[f"Doc {i}" for i in IDS], dtype=float)


def saturacao(tf: np.ndarray, k1: float, b: float, razao_tamanho: float) -> np.ndarray:
    """Fator de frequência do BM25: tf·(k1+1) / (tf + k1·(1 − b + b·|D|/avgdl))."""
    return tf * (k1 + 1) / (tf + k1 * (1 - b + b * razao_tamanho))


# ----------------------------------------------------------------------------- Fase 3: motor semântico
class VetorizadorSimulado:
    """Simulação vetorial documentada: hash dos trigramas de caracteres em 256 dimensões, normalizado.

    Aproxima textos que compartilham pedaços de palavras, mas não entende sinônimos. Só é usada quando
    HEALTHSEARCH_EMBEDDER=simulado (por exemplo, sem acesso ao Hugging Face).
    """

    dimensao = 256

    def encode(self, textos, normalize_embeddings: bool = True, **_):
        textos = [textos] if isinstance(textos, str) else list(textos)
        matriz = np.zeros((len(textos), self.dimensao), dtype=np.float32)
        for i, texto in enumerate(textos):
            t = f"  {normalizar(texto)}  "
            for k in range(len(t) - 2):
                matriz[i, int(hashlib.md5(t[k:k + 3].encode()).hexdigest(), 16) % self.dimensao] += 1.0
        if normalize_embeddings:
            normas = np.linalg.norm(matriz, axis=1, keepdims=True)
            matriz = matriz / np.where(normas == 0, 1, normas)
        return matriz


@lru_cache(maxsize=4)
def carregar_modelo(nome: str = MODELO_PADRAO):
    """SentenceTransformer do nome dado, ou a simulação vetorial se HEALTHSEARCH_EMBEDDER=simulado."""
    if os.environ.get("HEALTHSEARCH_EMBEDDER") == "simulado":
        return VetorizadorSimulado()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(nome)


def embeddings_documentos(modelo, nome: str = MODELO_PADRAO) -> np.ndarray:
    """Vetores normalizados dos 6 trechos clínicos (com o prefixo de documento do modelo)."""
    prefixo = MODELOS.get(nome, ("", ""))[1]
    return np.asarray(modelo.encode([prefixo + d["conteudo"] for d in DOCUMENTOS], normalize_embeddings=True), dtype=float)


def embedding_consulta(consulta: str, modelo, nome: str = MODELO_PADRAO) -> np.ndarray:
    prefixo = MODELOS.get(nome, ("", ""))[0]
    return np.asarray(modelo.encode([prefixo + consulta], normalize_embeddings=True), dtype=float)[0]


def cossenos(vetor_consulta: np.ndarray, E_docs: np.ndarray) -> np.ndarray:
    """Similaridade de cosseno entre a consulta e cada documento: (q · d) / (‖q‖ ‖d‖)."""
    normas = np.linalg.norm(E_docs, axis=1) * np.linalg.norm(vetor_consulta)
    return (E_docs @ vetor_consulta) / np.where(normas == 0, 1, normas)


# ----------------------------------------------------------------------------- Fase 4: RRF
def posicoes(scores: np.ndarray) -> np.ndarray:
    """Posição de cada documento (1 = melhor). Empates dividem a melhor posição do grupo (1, 2, 2, 4…)."""
    s = np.round(np.asarray(scores, dtype=float), 12)
    return np.array([1 + int((s > v).sum()) for v in s])


def score_rrf(rank_bm25: np.ndarray, rank_semantico: np.ndarray, alfa: float = ALFA_PADRAO, k: int = K_RRF) -> np.ndarray:
    """Score_RRF(D) = α · 1/(k + Rank_BM25) + (1 − α) · 1/(k + Rank_Semântico)."""
    return alfa / (k + np.asarray(rank_bm25)) + (1 - alfa) / (k + np.asarray(rank_semantico))


def ordenar(scores: np.ndarray) -> list[int]:
    """IDs dos documentos do maior para o menor score (empate: menor ID primeiro, só para exibir)."""
    return [IDS[i] for i in np.argsort(-np.round(np.asarray(scores, dtype=float), 12), kind="stable")]


def grupos(scores: np.ndarray, so_positivos: bool = False) -> list[list[int]]:
    """Documentos agrupados por posição, do melhor para o pior; cada grupo reúne os empatados.

    Com so_positivos=True, ficam de fora os documentos com score 0 (o BM25 não os recupera).
    """
    s = np.round(np.asarray(scores, dtype=float), 12)
    valores = sorted({v for v in s if v > 0 or not so_positivos}, reverse=True)
    return [[IDS[i] for i in range(len(s)) if s[i] == v] for v in valores]


def buscar(consulta: str, modelo, E_docs: np.ndarray, nome: str = MODELO_PADRAO, k1: float = K1_PADRAO,
           b: float = B_PADRAO, alfa: float = ALFA_PADRAO) -> dict:
    """Roda os três motores e devolve a tabela por documento e o tempo de cada etapa (ms)."""
    t0 = time.perf_counter()
    s_bm25 = scores_bm25(consulta, k1, b)
    t1 = time.perf_counter()
    s_sem = cossenos(embedding_consulta(consulta, modelo, nome), E_docs)
    t2 = time.perf_counter()
    r_bm25, r_sem = posicoes(s_bm25), posicoes(s_sem)
    s_rrf = score_rrf(r_bm25, r_sem, alfa)
    t3 = time.perf_counter()
    tabela = pd.DataFrame({
        "id": IDS, "titulo": [d["titulo"] for d in DOCUMENTOS], "conteudo": [d["conteudo"] for d in DOCUMENTOS],
        "score_bm25": s_bm25, "rank_bm25": r_bm25, "cosseno": s_sem, "rank_semantico": r_sem,
        "parcela_bm25": alfa / (K_RRF + r_bm25), "parcela_semantica": (1 - alfa) / (K_RRF + r_sem),
        "score_rrf": s_rrf, "rank_rrf": posicoes(s_rrf),
    })
    return {
        "tabela": tabela, "termos": preprocessar(consulta),
        "ordem_bm25": [i for i in ordenar(s_bm25) if s_bm25[IDS.index(i)] > 0],  # o BM25 só recupera quem tem termo
        "ordem_semantica": ordenar(s_sem), "ordem_rrf": ordenar(s_rrf),
        "grupos_bm25": grupos(s_bm25, so_positivos=True), "grupos_semantica": grupos(s_sem), "grupos_rrf": grupos(s_rrf),
        "tempos": {"BM25": (t1 - t0) * 1000, "Semântico": (t2 - t1) * 1000, "RRF": (t3 - t2) * 1000},
    }


# ----------------------------------------------------------------------------- métricas de desempenho
def metricas(ordem: list[int], relevantes: set[int]) -> dict:
    """P@1, reciprocal rank (1/posição do 1º relevante; 0 se nenhum) e Recall@3 de uma lista ordenada."""
    rr = next((1 / p for p, d in enumerate(ordem, 1) if d in relevantes), 0.0)
    return {"P@1": float(bool(ordem) and ordem[0] in relevantes), "RR": rr,
            "Recall@3": len(set(ordem[:3]) & relevantes) / len(relevantes)}


def metricas_com_empates(grupos_ordem: list[list[int]], relevantes: set[int]) -> dict:
    """Métricas esperadas quando há empates: a média sobre todas as ordens possíveis dentro de cada grupo empatado.

    Assim um empate não é desfeito a favor de ninguém (desempatar pelo ID, por exemplo, seria sorte).
    """
    ordens = [list(itertools.chain(*combo)) for combo in itertools.product(*(itertools.permutations(g) for g in grupos_ordem))]
    return pd.DataFrame([metricas(o, relevantes) for o in ordens]).mean().to_dict()


MOTORES = {"BM25": "grupos_bm25", "Semântico": "grupos_semantica", "Híbrido RRF": "grupos_rrf"}


def avaliar_consultas(modelo, E_docs: np.ndarray, nome: str = MODELO_PADRAO, k1: float = K1_PADRAO, b: float = B_PADRAO,
                      alfa: float = ALFA_PADRAO, gabarito: dict = GABARITO) -> pd.DataFrame:
    """Uma linha por consulta e motor, com P@1, RR e Recall@3 (esperados, se houver empate)."""
    linhas = []
    for consulta, (tipo, relevantes) in gabarito.items():
        r = buscar(consulta, modelo, E_docs, nome, k1, b, alfa)
        for motor, chave in MOTORES.items():
            linhas.append({"consulta": consulta, "tipo": tipo, "motor": motor, **metricas_com_empates(r[chave], relevantes)})
    return pd.DataFrame(linhas)


def avaliar(modelo, E_docs: np.ndarray, nome: str = MODELO_PADRAO, k1: float = K1_PADRAO, b: float = B_PADRAO,
            alfa: float = ALFA_PADRAO, gabarito: dict = GABARITO) -> pd.DataFrame:
    """Médias de P@1, MRR e Recall@3 dos três motores: no gabarito inteiro e por tipo de consulta."""
    por_consulta = avaliar_consultas(modelo, E_docs, nome, k1, b, alfa, gabarito)
    colunas = ["P@1", "RR", "Recall@3"]
    total = por_consulta.groupby("motor", sort=False)[colunas].mean().assign(tipo="todas").reset_index()
    por_tipo = por_consulta.groupby(["tipo", "motor"], sort=False)[colunas].mean().reset_index()
    return pd.concat([total, por_tipo]).rename(columns={"RR": "MRR"}).set_index(["tipo", "motor"])


def separacao(scores: np.ndarray, relevantes: set[int]) -> float:
    """Quanto o motor afasta os relevantes dos demais: (menor score relevante − maior não relevante) / maior score.

    Perto de 1: os não relevantes ficam muito atrás (o BM25 dá 0 a quem não tem o termo). Perto de 0: todos com
    notas parecidas (a "diluição" do semântico em códigos). Negativo: algum não relevante passou na frente.
    Sem nenhum score positivo (o BM25 não recuperou nada), devolve NaN.
    """
    s = np.asarray(scores, dtype=float)
    rel = np.array([i in relevantes for i in IDS])
    topo = float(np.abs(s).max())
    return float("nan") if topo == 0 else float((s[rel].min() - s[~rel].max()) / topo)


def primeiro(grupos_ordem: list[list[int]], relevantes: set[int]) -> str:
    """O 1º colocado com ✅/❌; se houver empate no topo, ⚖️ e os empatados."""
    if not grupos_ordem:
        return "❌ nenhum"
    topo = grupos_ordem[0]
    if len(topo) > 1:
        return f"⚖️ empate: {', '.join(f'Doc {d}' for d in topo)}"
    return f"{'✅' if topo[0] in relevantes else '❌'} Doc {topo[0]}"


def diagnostico(modelo, E_docs: np.ndarray, nome: str = MODELO_PADRAO, k1: float = K1_PADRAO, b: float = B_PADRAO,
                alfa: float = ALFA_PADRAO) -> pd.DataFrame:
    """1º colocado de cada motor nas consultas de demonstração (✅ relevante, ❌ não relevante, ⚖️ empate)."""
    linhas = []
    for consulta, mostra in EXEMPLOS.items():
        r, rel = buscar(consulta, modelo, E_docs, nome, k1, b, alfa), GABARITO[consulta][1]
        t = r["tabela"]
        linhas.append({"Consulta": consulta, "O que mostra": mostra, "Relevantes": ", ".join(f"Doc {d}" for d in sorted(rel)),
                       **{f"1º {m}": primeiro(r[chave], rel) for m, chave in MOTORES.items()},
                       "Separação BM25": separacao(t["score_bm25"], rel), "Separação semântica": separacao(t["cosseno"], rel)})
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------------- interface
ABAS = ["🔤 Léxico (BM25)", "🧠 Semântico", "🔀 Híbrido RRF", "📊 Matriz Comparativa"]


def main():
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="HealthSearch", page_icon="🩺", layout="wide")

    @st.cache_resource(show_spinner="Carregando o modelo de embeddings…")
    def modelo_cache(nome):
        return carregar_modelo(nome)

    @st.cache_data(show_spinner="Gerando os embeddings das diretrizes…")
    def docs_cache(nome):
        return embeddings_documentos(modelo_cache(nome), nome)

    @st.cache_data(show_spinner="Avaliando os motores no gabarito…")
    def avaliar_cache(nome, k1, b, alfa):
        return avaliar(modelo_cache(nome), docs_cache(nome), nome, k1, b, alfa)

    @st.cache_data(show_spinner=False)
    def curva_alfa_cache(nome, k1, b):
        return pd.DataFrame([{"α": a, **avaliar(modelo_cache(nome), docs_cache(nome), nome, k1, b, a)
                              .loc[("todas", "Híbrido RRF")].to_dict()} for a in np.round(np.linspace(0, 1, 11), 2)])

    @st.cache_data(show_spinner=False)
    def diagnostico_cache(nome, k1, b, alfa):
        return diagnostico(modelo_cache(nome), docs_cache(nome), nome, k1, b, alfa)

    st.title("🩺 HealthSearch — Busca Híbrida BM25 + Semântica")
    st.caption("Protocolos de triagem, diretrizes clínicas e guias de emergência. O BM25 garante precisão em termos e "
               "códigos exatos, os embeddings cobrem sinônimos médicos e o RRF combina os dois rankings.")

    # ---------------------------------------------------------------- sidebar
    st.sidebar.header("⚙️ Calibração")
    st.sidebar.subheader("Motor léxico (BM25)")
    k1 = st.sidebar.slider("k1 — saturação de frequência", 0.0, 3.0, K1_PADRAO, 0.1, key="k1",
                           help="0 ignora quantas vezes o termo aparece; valores altos deixam a frequência crescer mais.")
    b = st.sidebar.slider("b — normalização pelo comprimento", 0.0, 1.0, B_PADRAO, 0.05, key="b",
                          help="0 ignora o tamanho do documento; 1 penaliza ao máximo documentos longos.")
    st.sidebar.subheader("Fusão RRF")
    alfa = st.sidebar.slider("α — peso do BM25", 0.0, 1.0, ALFA_PADRAO, 0.05, key="alfa",
                             help="α = 1: só o BM25 · α = 0: só o semântico.")
    st.sidebar.caption(f"k_RRF = {K_RRF} (constante de suavização de posição)")
    st.sidebar.subheader("Motor semântico")
    nome_modelo = st.sidebar.selectbox("Modelo de embedding", list(MODELOS), key="modelo", format_func=lambda m: m.split("/")[-1])
    if os.environ.get("HEALTHSEARCH_EMBEDDER") == "simulado":
        st.sidebar.warning("Modo simulado: o motor semântico usa trigramas de caracteres e não entende sinônimos.")

    modelo = modelo_cache(nome_modelo)
    E_docs = docs_cache(nome_modelo)

    # ---------------------------------------------------------------- Fase 1
    with st.expander("📚 Fase 1 — corpus médico e pré-processamento"):
        st.dataframe(pd.DataFrame([{"ID": f"Doc {d['id']}", "Título": d["titulo"], "Conteúdo": d["conteudo"],
                                    "Tokens (BM25)": " · ".join(t)} for d, t in zip(DOCUMENTOS, TOKENS_DOCS)]),
                     hide_index=True, width="stretch")
        st.caption("Pipeline: minúsculas e sem acentos → sem caracteres especiais (o hífen dos códigos fica) → tokens → "
                   "sem stopwords em português. Códigos compostos entram inteiros e também em partes.")

    # ---------------------------------------------------------------- consulta
    def usar_exemplo(texto):
        st.session_state["consulta"] = texto

    st.session_state.setdefault("consulta", "ataque cardíaco")
    consulta = st.text_input("🔎 Consulta", key="consulta", placeholder="Ex.: infarto, CÓD-ECG-12D, AAS 100mg")
    st.caption("Exemplos dos pontos cegos:")
    colunas = st.columns(len(EXEMPLOS))
    for col, (texto, mostra) in zip(colunas, EXEMPLOS.items()):
        col.button(texto, key=f"ex_{texto}", help=mostra, on_click=usar_exemplo, args=(texto,), width="stretch")

    if not consulta.strip():
        st.info("Digite uma consulta para comparar os motores.")
        return

    r = buscar(consulta, modelo, E_docs, nome_modelo, k1, b, alfa)
    t = r["tabela"]
    topo = r["grupos_rrf"][0]
    titulos = {d["id"]: d["titulo"] for d in DOCUMENTOS}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⏱️ BM25", f"{r['tempos']['BM25']:.2f} ms")
    c2.metric("⏱️ Semântico", f"{r['tempos']['Semântico']:.2f} ms")
    c3.metric("⏱️ Fusão RRF", f"{r['tempos']['RRF']:.3f} ms")
    c4.metric("🏆 1º no híbrido", f"⚖️ Docs {'/'.join(map(str, topo))}" if len(topo) > 1 else f"Doc {topo[0]}",
              help=" · ".join(titulos[d] for d in topo))

    lexico, semantico, hibrido, matriz = st.tabs(ABAS)

    # ---------------------------------------------------------------- 🔤 BM25
    with lexico:
        st.markdown(f"**Okapi BM25** com k1 = {k1:.1f} e b = {b:.2f} · termos da consulta: "
                    f"`{' · '.join(r['termos']) or '—'}`")
        if not r["termos"]:
            st.warning("A consulta não tem termos depois do pré-processamento.")
        elif not r["ordem_bm25"]:
            st.error("Nenhuma diretriz contém os termos da consulta: o BM25 não recupera nada. É o ponto cego léxico "
                     "(sinônimos e siglas que não aparecem no texto).")
        tab_bm25 = t.sort_values(["rank_bm25", "id"])
        st.dataframe(pd.DataFrame({
            "Posição": tab_bm25["rank_bm25"], "Documento": [f"Doc {i}" for i in tab_bm25["id"]], "Título": tab_bm25["titulo"],
            "Score BM25": tab_bm25["score_bm25"].round(4),
            "Recuperado": np.where(tab_bm25["score_bm25"] > 0, "✅", "— sem termos em comum"),
        }), hide_index=True, width="stretch")
        if r["termos"]:
            st.markdown("**Parcela de cada termo no score** (IDF × fator de frequência)")
            st.dataframe(contribuicoes_bm25(consulta, k1, b).round(4), width="stretch")
        tf = np.linspace(0, 10, 101)
        tamanhos = [len(x) for x in TOKENS_DOCS]
        media = float(np.mean(tamanhos))
        curvas = pd.concat([pd.DataFrame({"frequência do termo (tf)": tf, "fator de frequência": saturacao(tf, k1, b, L / media),
                                          "documento": f"{rotulo} ({L} tokens)"})
                            for rotulo, L in [("mais curto", min(tamanhos)), ("mais longo", max(tamanhos))]])
        fig = px.line(curvas, x="frequência do termo (tf)", y="fator de frequência", color="documento",
                      title=f"Saturação da frequência com k1 = {k1:.1f} e b = {b:.2f}")
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")
        st.caption("k1 controla o quanto repetir o termo ainda aumenta o score; b controla a diferença entre documentos "
                   "curtos e longos. Com b = 0 as duas curvas coincidem.")

    # ---------------------------------------------------------------- 🧠 semântico
    with semantico:
        st.markdown(f"**Embeddings densos** (`{nome_modelo.split('/')[-1]}`) e similaridade de cosseno entre a consulta e "
                    "cada diretriz. Captura sinônimos e contexto médico.")
        tab_sem = t.sort_values(["rank_semantico", "id"])
        st.dataframe(pd.DataFrame({
            "Posição": tab_sem["rank_semantico"], "Documento": [f"Doc {i}" for i in tab_sem["id"]], "Título": tab_sem["titulo"],
            "Cosseno": tab_sem["cosseno"].round(4),
        }), hide_index=True, width="stretch")
        fig = px.bar(tab_sem.iloc[::-1], x="cosseno", y=[f"Doc {i} · {tt}" for i, tt in zip(tab_sem["id"][::-1], tab_sem["titulo"][::-1])],
                     orientation="h", labels={"y": "", "cosseno": "similaridade de cosseno"}, title="Similaridade com a consulta")
        fig.update_layout(height=320, xaxis_range=[min(0, float(t["cosseno"].min())), 1])
        st.plotly_chart(fig, width="stretch")
        if nome_modelo.startswith("intfloat/"):
            st.caption("O modelo e5 foi treinado com os prefixos “query: ” e “passage: ”: as notas ficam comprimidas "
                       "(≈ 0,8 a 0,9), mas a ordem é o que entra no RRF.")

    # ---------------------------------------------------------------- 🔀 RRF
    with hibrido:
        st.latex(r"\text{Score}_{RRF}(D) = \alpha \cdot \frac{1}{k_{RRF} + \text{Rank}_{BM25}(D)} + (1-\alpha) \cdot \frac{1}{k_{RRF} + \text{Rank}_{Sem}(D)}")
        st.markdown(f"α = **{alfa:.2f}** · k_RRF = **{K_RRF}** · os ranks são posições a partir de 1; documentos empatados "
                    "(por exemplo, todos com BM25 = 0) dividem a mesma posição.")
        tab_rrf = t.sort_values(["rank_rrf", "id"])
        st.dataframe(pd.DataFrame({
            "Posição": tab_rrf["rank_rrf"], "Documento": [f"Doc {i}" for i in tab_rrf["id"]], "Título": tab_rrf["titulo"],
            "Rank BM25": tab_rrf["rank_bm25"], "Rank Semântico": tab_rrf["rank_semantico"],
            "α/(k+Rank BM25)": tab_rrf["parcela_bm25"].round(6), "(1−α)/(k+Rank Sem.)": tab_rrf["parcela_semantica"].round(6),
            "Score RRF": tab_rrf["score_rrf"].round(6),
        }), hide_index=True, width="stretch")
        conteudos = {d["id"]: d["conteudo"] for d in DOCUMENTOS}
        if len(topo) == 1:
            st.success(f"🏆 Doc {topo[0]} — {titulos[topo[0]]}: {conteudos[topo[0]]}")
        else:
            st.warning(f"⚖️ Empate no 1º lugar entre {', '.join(f'Doc {d}' for d in topo)}: as parcelas somadas se "
                       "igualam" + (" (com α = 0,5, dois documentos em posições espelhadas nos dois motores empatam)"
                                    if abs(alfa - 0.5) < 1e-9 else "") + ". Mova α para dar preferência a um motor.")

    # ---------------------------------------------------------------- 📊 matriz comparativa
    with matriz:
        st.markdown("**Posição de cada diretriz em cada motor** (1 = melhor)")
        comp = t.sort_values(["rank_rrf", "id"])
        st.dataframe(pd.DataFrame({
            "Documento": [f"Doc {i}" for i in comp["id"]], "Título": comp["titulo"], "Rank BM25": comp["rank_bm25"],
            "Rank Semântico": comp["rank_semantico"], "Rank Híbrido RRF": comp["rank_rrf"],
            "BM25 recuperou?": np.where(comp["score_bm25"] > 0, "sim", "não"),
        }), hide_index=True, width="stretch")
        longo = comp.melt(id_vars=["id", "titulo"], value_vars=["rank_bm25", "rank_semantico", "rank_rrf"],
                          var_name="motor", value_name="posição")
        longo["motor"] = longo["motor"].map({"rank_bm25": "BM25", "rank_semantico": "Semântico", "rank_rrf": "Híbrido RRF"})
        longo["documento"] = [f"Doc {i}" for i in longo["id"]]
        fig = px.line(longo, x="motor", y="posição", color="documento", markers=True, hover_data={"titulo": True},
                      title=f"Comparação de ranks · “{consulta}”")
        fig.update_yaxes(autorange="reversed", dtick=1)
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### Métricas de desempenho")
        n_sin = sum(tipo == SINONIMO for tipo, _ in GABARITO.values())
        st.markdown(f"Médias sobre um gabarito de **{len(GABARITO)} consultas** com as diretrizes relevantes marcadas à mão: "
                    f"{n_sin} na linguagem do plantão (sinônimos e siglas) e {len(GABARITO) - n_sin} com termos exatos ou "
                    "códigos. P@1: o 1º é relevante · MRR: média de 1/posição do 1º relevante · Recall@3: fração dos "
                    "relevantes entre os 3 primeiros. Em empates, vale a média sobre as ordens possíveis. "
                    "Tudo muda com k1, b, α e o modelo.")
        aval = avaliar_cache(nome_modelo, k1, b, alfa)
        st.dataframe(aval.reset_index().rename(columns={"tipo": "Consultas", "motor": "Motor"}).round(3),
                     hide_index=True, width="stretch")
        curva = curva_alfa_cache(nome_modelo, k1, b)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curva["α"], y=curva["MRR"], mode="lines+markers", name="Híbrido RRF"))
        fig.add_hline(y=aval.loc[("todas", "BM25"), "MRR"], line_dash="dot", annotation_text="BM25")
        fig.add_hline(y=aval.loc[("todas", "Semântico"), "MRR"], line_dash="dash", annotation_text="Semântico",
                      annotation_position="bottom right")
        fig.add_vline(x=alfa, line_color="gray")
        fig.update_layout(title="MRR do híbrido em função de α (gabarito inteiro)", xaxis_title="α (peso do BM25)",
                          yaxis_title="MRR", yaxis_range=[0, 1.05], height=340)
        st.plotly_chart(fig, width="stretch")
        with st.expander("Gabarito de relevância"):
            st.dataframe(pd.DataFrame([{"Consulta": q, "Tipo": tipo, "Relevantes": ", ".join(f"Doc {d}" for d in sorted(rel))}
                                       for q, (tipo, rel) in GABARITO.items()]), hide_index=True)

        st.markdown("#### Diagnóstico dos pontos cegos")
        diag = diagnostico_cache(nome_modelo, k1, b, alfa)
        for coluna in ["Separação BM25", "Separação semântica"]:
            diag[coluna] = diag[coluna].map(lambda v: "— (nada recuperado)" if np.isnan(v) else f"{v:+.1%}")
        st.dataframe(diag, hide_index=True, width="stretch")
        st.caption("Separação = (menor score dos relevantes − maior score dos não relevantes) ÷ maior score. O BM25 falha "
                   "em sinônimos, siglas fora do texto e palavras ambíguas (“alta”), mas quando acerta separa bem: quem não "
                   "tem o termo fica com 0. O semântico acerta sinônimos, mas dá notas parecidas a todos: num código exato, "
                   "os trechos genéricos ficam logo atrás (diluição). O híbrido RRF combina as posições dos dois; ⚖️ indica "
                   "empate, quando os motores discordam em posições espelhadas.")


if __name__ == "__main__":
    main()
