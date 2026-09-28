# a28 — HealthSearch · Bônus: Re-ranking com Cross-Encoder

**Fonte:** `aulas/Laboratório Prático 05 - Desafio Integrador HealthSearch (Motor de Busca Híbrido BM25 e Semântico).docx`, seção 7 (Desafio Bônus, +0,3 ponto).

## Enunciado

> Cross-Encoder Re-Ranking: adicionar um checkbox na interface para aplicar uma camada de re-ranking baseada em modelo Cross-Encoder (ex.: `cross-encoder/ms-marco-MiniLM-L-6-v2`) sobre os Top-3 candidatos recuperados pela busca híbrida RRF, comparando a variação da nota final de relevância.

## O que foi feito

`healthsearch_app.py` é o app completo do HealthSearch da a27 (BM25, semântico, RRF, matriz comparativa e métricas), com o bônus. A pasta é autossuficiente, então o app inteiro está aqui.

| Onde | O que o bônus acrescenta |
|---|---|
| **Barra lateral** | Checkbox **🎯 Re-ranking com Cross-Encoder nos Top-3 do RRF** (desligado por padrão). |
| **Métricas do topo** | Tempo do Cross-Encoder e o 1º colocado depois do re-ranking, ao lado do 1º do híbrido. |
| **Aba 🎯 Re-ranking (Cross-Encoder)** | Para os Top-3 do RRF: posição antes e depois, variação da posição (↑/↓), nota RRF, logit e nota do Cross-Encoder e a variação da nota. Mostra também um gráfico das notas antes × depois, se o 1º lugar mudou e o efeito do re-ranking no gabarito de 18 consultas. |

| Função nova | O que faz |
|---|---|
| `carregar_cross_encoder()` | Carrega o `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers). |
| `reranquear(consulta, resultado, cross)` | Envia ao Cross-Encoder os pares (consulta, trecho) dos 3 primeiros do RRF, re-ordena pela nota e calcula as variações. As outras posições do RRF não mudam. |
| `avaliar_reranking(...)` | P@1, MRR e Recall@3 do híbrido com e sem re-ranking, no total e por tipo de consulta. |

**Nota final de relevância.** O score RRF e o logit do Cross-Encoder estão em escalas diferentes. Para compará-los, as duas notas vão para a faixa de 0 a 1:

- **nota RRF** = score RRF × (k + 1), que vale 1 para quem é 1º nos dois motores;
- **nota Cross-Encoder** = sigmoide do logit.

## Resultado (e5-small, α = 0,5)

| Consultas | Híbrido RRF (MRR) | RRF + Cross-Encoder (MRR) |
|---|---|---|
| sinônimos e siglas (10) | 0,97 | 0,80 |
| termos exatos e códigos (8) | 1,00 | 1,00 |

O `ms-marco-MiniLM-L-6-v2` foi treinado em inglês (MS MARCO). Ele confirma coincidências literais com notas altas, como “infarto” → Doc 2 (0,94) e “CÓD-ECG-12D” → Docs 1 e 6 (> 0,99, contra ≈ 0 para o Doc 5). Mas não entende sinônimos em português: em “ataque cardíaco” dá nota ≈ 0 aos três candidatos e tira o infarto do 1º lugar. Por isso ele mantém a precisão em códigos e piora as consultas da linguagem do plantão. Um Cross-Encoder multilíngue seria o próximo passo. O `mmarco-mMiniLMv2-L12-H384-v1`, avaliado à parte, ordenou pior que o do enunciado neste corpus.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run healthsearch_app.py      # marque o checkbox de re-ranking na barra lateral
```

Na primeira execução, são baixados do Hugging Face o `intfloat/multilingual-e5-small` (≈ 470 MB) e o Cross-Encoder (≈ 90 MB). Sem rede, rode com `HEALTHSEARCH_EMBEDDER=simulado`: o semântico e o Cross-Encoder passam a usar a simulação vetorial por trigramas de caracteres, que não entende sinônimos.

## Suposições

- **Top-3:** são os três primeiros da ordem do RRF. Se houver empate na 3ª posição, o de menor ID entra, e isso aparece na tabela.
- **O que muda:** o re-ranking só reordena os 3 primeiros. As demais posições continuam as do RRF, e as métricas usam essa ordem final.
- **Modelo:** o do exemplo do enunciado, `cross-encoder/ms-marco-MiniLM-L-6-v2`, usado sem ajuste para o português, que é justamente o que a comparação expõe.
