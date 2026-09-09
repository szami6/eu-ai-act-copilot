# Retrieval eval — 0016607

Generated 2026-09-07T21:31:25.610316+00:00 · 24 gold queries · `data/eval/retrieval_gold.yaml` · no LLM in the loop (PLAN.md §7.2).

## Headline (retrieve -> rerank, production default: `rerank_enabled=True`)

| Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| 0.875 | 0.938 | 0.791 | 0.816 |

## By category

| Category | n | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| annex_iii | 5 | 0.900 | 0.900 | 0.900 | 0.849 |
| classification | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| definition | 4 | 0.750 | 1.000 | 0.486 | 0.608 |
| deployer_obligation | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| gdpr | 3 | 1.000 | 1.000 | 0.778 | 0.833 |
| gpai | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| penalty | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| prohibited_practice | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| provider_obligation | 2 | 0.250 | 0.500 | 0.100 | 0.207 |
| transparency | 1 | 1.000 | 1.000 | 1.000 | 1.000 |

## Rerank ablation (Δ precision@5, cross-encoder vs. MMR fusion-only)

| Config | Precision@5 | Recall@5 | MRR |
|---|---|---|---|
| rerank on (cross-encoder) | 0.192 | 0.875 | 0.791 |
| rerank off (MMR fusion-only) | 0.100 | 0.458 | 0.429 |

## Misses (gold id absent from top-10, rerank on)

| Category | Query | Gold | Top-5 returned |
|---|---|---|---|
| annex_iii | Why are AI systems used in the administration of justice and democratic processes classified as high-risk? | ai_act:recital:61, ai_act:annex:III:8 | ai_act:recital:61, ai_act:recital:58, ai_act:recital:52, ai_act:recital:57, ai_act:recital:56 |
| provider_obligation | What conformity assessment must a high-risk AI system undergo? | ai_act:art:16:(f) | ai_act:recital:125, ai_act:art:6:1, ai_act:art:6:1b, ai_act:art:43:4, ai_act:recital:50 |
