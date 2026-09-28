# HealthSearch — Motor de Busca Híbrido BM25 + Semântico

Laboratório Prático 05 (Desafio Integrador) da disciplina **Tendências em Ciência da Computação** (Prof. Me. Ricardo Roberto de Lima), valendo 1,0 ponto extra na Unidade II.

Protótipo de busca para protocolos de triagem e diretrizes clínicas da HealthTech Solutions. O Okapi BM25 garante precisão em termos e códigos exatos (“CÓD-ECG-12D”), os embeddings cobrem sinônimos médicos (“ataque cardíaco” → infarto agudo do miocárdio), e o Reciprocal Rank Fusion combina os dois rankings num dashboard Streamlit.

| Pasta | Entrega | Conteúdo |
|---|---|---|
| [`a27/`](a27/) | Desafio principal | App Streamlit num único arquivo (`healthsearch_app.py`) com as 4 fases: pré-processamento, BM25 com sliders de k1 e b, motor semântico e fusão RRF com α. Tem abas Léxico, Semântico, Híbrido RRF e Matriz Comparativa, métricas de desempenho e o relatório técnico `relatorio.pdf` (2 páginas). |
| [`a28/`](a28/) | Desafio bônus | O mesmo app com um checkbox que re-ordena os Top-3 do RRF com o Cross-Encoder `ms-marco-MiniLM-L-6-v2` e compara a posição e a nota de relevância antes e depois. |

Cada pasta é independente e tem o próprio `README.md` (enunciado, como rodar e suposições) e `requirements.txt`.

## Como rodar

```bash
cd a28                      # ou a27
pip install -r requirements.txt
streamlit run healthsearch_app.py
```
