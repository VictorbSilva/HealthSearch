# a27 — HealthSearch: Motor de Busca Híbrido BM25 + Semântico com RRF

**Fonte:** `aulas/Laboratório Prático 05 - Desafio Integrador HealthSearch (Motor de Busca Híbrido BM25 e Semântico).docx`, seções 1 a 6.

## Enunciado (resumo)

> A HealthTech Solutions tem um repositório de protocolos de triagem e diretrizes clínicas. Motores léxicos falham em sinônimos (“infarto”, “ataque cardíaco” × “síndrome coronariana aguda”), e motores semânticos diluem a precisão em códigos e doses (“ECG-12D”, “AAS 100mg”). Construa em Streamlit (`healthsearch_app.py`) um motor híbrido com 4 fases: (1) corpus médico fixo e pré-processamento (minúsculas, remoção de caracteres especiais e de stopwords em português); (2) Okapi BM25 com sliders de k1 (0,0 a 3,0, padrão 1,2) e b (0,0 a 1,0, padrão 0,75); (3) busca semântica por embeddings e cosseno; (4) fusão Score_RRF(D) = α·1/(k + Rank_BM25) + (1 − α)·1/(k + Rank_Semântico), com k = 60. Interface em abas (Léxico, Semântico, Híbrido RRF e Matriz Comparativa) e relatório técnico em PDF de até 2 páginas.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `healthsearch_app.py` | O app, num único arquivo: uma função pura por fase (pré-processamento, BM25, semântico, RRF e métricas) e a interface em `main()`. |
| `relatorio.pdf` | Relatório técnico (2 páginas): arquitetura, gráfico de comparação de ranks, métricas e divisão de tarefas. |
| `gerar_relatorio.py`, `relatorio_pdf.py` | Geram `RELATORIO.md`, `comparacao_ranks.png` e `relatorio.pdf` a partir das funções do app. |

## O app

| Onde | O que tem |
|---|---|
| **Barra lateral** | Sliders de k1 (0–3, padrão 1,2), b (0–1, padrão 0,75) e α (0–1, padrão 0,5); k_RRF = 60 fixo; escolha do modelo de embedding. |
| **📚 Fase 1** (expansível) | As 6 diretrizes com os tokens que entram no BM25. |
| **Consulta** | Campo livre e botões com os exemplos dos pontos cegos: “ataque cardíaco”, “CÓD-ECG-12D”, “pressão alta”, “AVC”, “AAS 100mg” e “infarto”. Tempo de cada motor e 1º colocado do híbrido. |
| **🔤 Léxico (BM25)** | Ranking BM25, documentos não recuperados, parcela de cada termo no score e curva de saturação da frequência para os k1 e b escolhidos. |
| **🧠 Semântico** | Ranking por cosseno e gráfico de barras. |
| **🔀 Híbrido RRF** | A fórmula, e para cada diretriz: rank BM25, rank semântico, as duas parcelas e o score RRF. Empates no 1º lugar são sinalizados. |
| **📊 Matriz Comparativa** | Posição de cada diretriz nos três motores e gráfico de comparação de ranks. Métricas de desempenho (P@1, MRR e Recall@3) num gabarito de 18 consultas, no total e por tipo de consulta; curva do MRR do híbrido em função de α; diagnóstico dos pontos cegos, com a “separação” de cada motor. |

## Resultados (modelo `multilingual-e5-small`, α = 0,5)

| Consultas | BM25 (MRR) | Semântico (MRR) | Híbrido RRF (MRR) |
|---|---|---|---|
| sinônimos e siglas (10) | 0,45 | 1,00 | 0,97 |
| termos exatos e códigos (8) | 1,00 | 1,00 | 1,00 |

- **Ponto cego do BM25:** em “ataque cardíaco”, “AVC” e “AAS 100mg” ele não recupera nada; em “pressão alta”, a palavra “alta” o leva à reanimação (“alta qualidade”).
- **Ponto cego do semântico:** em “CÓD-ECG-12D” ele acerta a ordem, mas dá notas parecidas a todas as diretrizes (separação de 7%, contra 97% do BM25).
- **Híbrido:** fica perto do melhor motor nos dois tipos de consulta. Com α ≤ 0,4 chega a MRR 1,0; com α = 0,5, “pressão alta” empata no topo (os motores discordam em posições espelhadas).

## Como rodar

```bash
pip install -r requirements.txt
streamlit run healthsearch_app.py
python gerar_relatorio.py     # opcional: regera relatorio.pdf
```

Na primeira execução, o modelo `intfloat/multilingual-e5-small` (≈ 470 MB) é baixado do Hugging Face. O alternativo, `paraphrase-multilingual-MiniLM-L12-v2`, só é baixado se for escolhido na barra lateral. Sem rede, rode com `HEALTHSEARCH_EMBEDDER=simulado` (PowerShell: `$env:HEALTHSEARCH_EMBEDDER="simulado"`). O motor semântico passa a usar a **simulação vetorial documentada** que o enunciado permite: hash dos trigramas de caracteres em 256 dimensões. Ela aproxima grafias parecidas, mas não entende sinônimos.

## Suposições

- **Modelo semântico:** `multilingual-e5-small`, treinado para recuperação e usado com os prefixos “query: ” e “passage: ”. Com ele, o semântico acerta “ataque cardíaco”, “infarto”, “AVC” e “AAS 100mg”. O `paraphrase-multilingual-MiniLM-L12-v2`, usado nas aulas, fica como alternativa: ele erra “infarto”, “AVC”, “AAS 100mg” e “RCR”, e mostra melhor quando o BM25 socorre o semântico.
- **O que é indexado:** o trecho clínico de cada diretriz. O título só é exibido. Por isso siglas que só aparecem no título (“AVC”, “RCR”) são um ponto cego do BM25.
- **Pré-processamento:** a lista de stopwords em português é própria, para não depender do download do NLTK. Os acentos também são removidos, junto com os caracteres especiais. O hífen entre letras e dígitos é mantido, e códigos compostos geram o token inteiro e as partes: “ECG-12D” encontra “CÓD-ECG-12D”.
- **k1 = 0:** o `rank_bm25` calcula 0/0 nos documentos sem o termo. O app calcula termo a termo e usa o limite correto, que é 0.
- **Ranks do RRF:** são posições a partir de 1 nos 6 documentos. Documentos empatados dividem a melhor posição do grupo (por exemplo, todos com BM25 = 0), para o RRF não favorecer ninguém pelo ID.
- **α:** o enunciado não fixa o padrão; foi usado 0,5, peso igual para os dois motores.
- **“Métricas de desempenho”:** o enunciado não as define. O app mostra o tempo de cada motor e P@1, MRR e Recall@3 num gabarito próprio de 18 consultas (metade sinônimos e siglas, metade termos exatos e códigos). Nos empates, vale a média sobre as ordens possíveis.
- **Equipe:** o laboratório é para grupos de 2 a 3 alunos; esta entrega é individual, e a divisão de tarefas do relatório reflete isso.
- **Bônus:** o re-ranking por Cross-Encoder está na a28.
