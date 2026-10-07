# Machine Learning — classificador de encaminhamento

> **PROTÓTIPO ACADÊMICO — NÃO UTILIZAR PARA DECISÕES CLÍNICAS REAIS.**
> Nenhuma métrica aqui é evidência clínica. Detalhes dos dados, do MIMIC-IV-ED e do domain shift em
> [datasets.md](datasets.md).

## Problema

Classificação supervisionada **multiclasse** do nível de encaminhamento. O modelo não diagnostica doenças.

| Classe (`CareLevel`) | Serviço (`ServiceType`) | Ordem de gravidade |
|---|---|---|
| `PRIMARY_CARE` | UBS | 0 |
| `URGENT_CARE` | UPA | 1 |
| `EMERGENCY` | Pronto-socorro + orientação SAMU 192 | 2 |

## Arquitetura do código

```text
app/ml/
├── data/
│   ├── schemas.py        RoutingTrainingExample (esquema canônico), DatasetInfo, TrainingDataset
│   ├── base.py           CSV canônico processado + sidecar .meta.json
│   ├── synthetic.py      gerador sintético-v1 → esquema canônico
│   └── mimic_ed.py       triage do MIMIC-IV-ED → esquema canônico; map_esi_to_care_level
├── features.py           features de sintomas da API (fonte única de verdade para inferência)
├── feature_builders.py   um builder por feature set:
│                           HealthFlowSymptomFeatureBuilder  (healthflow-symptoms-v1)
│                           MimicStructuredFeatureBuilder    (mimic-structured-vitals-v1)
├── splits.py             split treino/teste e folds de CV (por paciente quando há group_id)
├── metrics.py            métricas, incluindo under/over-triage
├── experiments.py        registro: experimento → dataset, feature set, diretório do modelo
├── classifier.py         carregamento verificado (checksum, feature set) + ModelMetadata
├── inference.py          serviço usado pela API
└── training/
    ├── dataset.py        CLI make dataset-synthetic
    ├── prepare_mimic.py  CLI make dataset-mimic
    ├── train.py          CLI make train-* (compara, seleciona, salva, grava benchmarks)
    ├── evaluate.py       CLI make evaluate-* (só no teste reservado)
    └── benchmark.py      registro JSON imutável por execução
```

Não há `if dataset == ...`: cada fonte converte para o esquema canônico e cada feature set tem seu
builder. A API só aceita modelos do feature set `healthflow-symptoms-v1`. Um modelo de sinais vitais é
**recusado** na inicialização, porque a API não tem sinais vitais.

## Experimentos

| | Dados | Entrada do modelo | Estado |
|---|---|---|---|
| **A** `synthetic_baseline` | `synthetic-v1` (4000 linhas) | sintomas, intensidade, duração, faixa etária e fatores de risco (34 features) | ✅ executado |
| **B** `structured_mimic_baseline` | MIMIC-IV-ED triage | FC, FR, SpO₂, PAS, PAD, temperatura (°C) e dor (7 features) | ✅ pipeline pronto; **sem resultados**: requer acesso credenciado aos dados |
| **C** (planejado) | MIMIC-IV-ED `chiefcomplaint` → Qwen3 4B local → `SymptomExtraction` | features do Health-flow | ⏳ não implementado |

A pergunta do artigo: **dados estruturados reais (B) × texto interpretado pelo LLM (C)**, com A como
baseline acadêmico. A e B/C **não são diretamente comparáveis**: têm rótulos de natureza diferente
(gerador artificial × ESI mapeado).

## Modelos comparados (iguais em todos os experimentos)

Todos dentro de um `Pipeline` com `SimpleImputer(strategy="median", add_indicator=True)`, ajustado
**só no treino** de cada *fold*. Ele não altera nada quando não há faltantes, como no sintético.
Todos usam `class_weight="balanced"` e `random_state=42`.

- `LogisticRegression(max_iter=2000)` com `StandardScaler`
- `DecisionTreeClassifier(max_depth=6, min_samples_leaf=10)`
- `RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5)`

XGBoost e busca de hiperparâmetros ficam fora por enquanto.

## Protocolo de avaliação (sem vazamento)

| Etapa | Sem pacientes identificados (A) | Com `subject_id` (B, C) |
|---|---|---|
| Teste reservado (~20%) | `train_test_split` estratificado por linha | 1 de 5 *folds* de `StratifiedGroupKFold`: **nenhum paciente em treino e teste ao mesmo tempo** |
| Validação cruzada (no treino) | `StratifiedKFold(5)` | `StratifiedGroupKFold(5)`: nenhum paciente em dois *folds* |
| Seleção do modelo | **somente CV**: maior `round(recall EMERGENCY, 2)`, desempate por F1 macro | idem |
| Teste | uma avaliação final por modelo, só para relatório; **não** influencia a seleção | idem |

O identificador de *split* aparece em `metadata.json` e em cada benchmark
(`stratified_row_80_20__cv_stratified_kfold_5` ou
`stratified_group_by_patient_80_20__cv_stratified_group_kfold_5`). `make evaluate-*` recalcula o
mesmo *split* de forma determinística e pontua **apenas** o teste reservado. Se o dataset mudou desde o
treino, ele se recusa a rodar.

## Métricas

accuracy · precision/recall/F1 por classe · precision/recall/F1 macro · matriz de confusão ·
recall de EMERGENCY, e mais:

- `under_triage_rate`: fração de **todos** os casos previstos como menos graves que o rótulo;
- `over_triage_rate`: fração de **todos** os casos previstos como mais graves que o rótulo;
- `critical_under_triage_rate`: fração dos casos **EMERGENCY reais** previstos como outra classe
  (= 1 − recall de EMERGENCY). É o erro mais grave para o sistema.

## Artefatos e benchmarks

```text
models/<experimento>/routing_model.joblib
models/<experimento>/metadata.json   experiment, dataset_name/version, feature_set, features,
                                     split_strategy, number_of_rows, number_of_patients,
                                     model_type, model_version, hyperparameters, metrics,
                                     training_date, git_commit, artifact_sha256, random_state
benchmarks/results/<timestamp>_<experimento>_<modelo>_<run_id>.json
```

- Cada `make train-*` grava **um JSON por modelo candidato** com os mesmos campos, além de métricas de CV
  e de teste, `training_time_seconds`, `inference_time_ms_per_row` e `selected_for_deployment`.
  Arquivos são abertos em modo exclusivo: **nunca sobrescrevem** execuções anteriores.
- `git_commit` termina em `-dirty` se havia mudanças não commitadas. Para o artigo, rode com a árvore
  limpa.
- `make benchmark-summary` gera uma tabela Markdown de todos os resultados.
- O artefato `joblib` só é carregado se o SHA-256 bater com `metadata.json` (joblib executa código ao
  desserializar).

## Resultados — Experimento A (`synthetic-v1`)

**Dados sintéticos: os números medem a recuperação do processo gerador, não a realidade clínica.**
Os valores exatos estão nos JSON em `benchmarks/results/`.

Validação cruzada (treino, n=3200):

| Modelo | Accuracy | Precisão macro | Recall macro | F1 macro | Recall EMERGENCY |
|---|---|---|---|---|---|
| Logistic Regression | 0.880 | 0.847 | 0.854 | 0.850 | 0.798 |
| Decision Tree | 0.854 | 0.829 | 0.843 | 0.833 | 0.799 |
| **Random Forest** (selecionado pela CV) | 0.900 | 0.881 | 0.884 | 0.881 | 0.826 |

Teste reservado (n=800), Random Forest: accuracy 0.931 · F1 macro 0.919 · recall EMERGENCY 0.891 ·
under-triage 0.025 · over-triage 0.044 · critical under-triage 0.109.

| real \ previsto | PRIMARY | URGENT | EMERGENCY |
|---|---|---|---|
| **PRIMARY** | 409 | 22 | 5 |
| **URGENT** | 5 | 213 | 8 |
| **EMERGENCY** | 2 | 13 | 123 |

Leitura: 15 de 138 emergências (≈11%) seriam rebaixadas **pelo modelo sozinho**. Por isso o ML nunca é
a autoridade final: o Safety Engine impõe pisos e o Care Routing nunca reduz um piso.

## Resultados — Experimento B (MIMIC estruturado)

**Ainda não há resultados.** O pipeline foi verificado de ponta a ponta apenas com arquivos **falsos**
no formato do MIMIC (testes automatizados). Os números aparecerão em `benchmarks/results/` quando
alguém com acesso credenciado rodar:

```bash
make dataset-mimic MIMIC_SOURCE_VERSION=<versão>   # data/raw/mimic-iv-ed/triage.csv.gz
make train-mimic
make evaluate-mimic
```

## Experimento C — o que falta

1. CLI `app.ml.training.prepare_mimic_llm` com `--limit`/`--offset`, lendo
   `data/processed/routing_mimic_v1.csv` (a queixa já está lá).
2. Cache local chaveado por `sha256(queixa normalizada + modelo + versão do prompt)`, guardando **só** a
   extração (sem IDs). Queixa repetida não chama o LLM de novo.
3. Escrever `symptoms`/`severity`/`duration_minutes` no esquema canônico, mantendo `group_id` e
   `label`, num dataset `mimic-ed-llm-v1`.
4. Registrar o experimento com `HealthFlowSymptomFeatureBuilder`.
5. Comparação justa com B: **mesmo subconjunto de linhas e mesmo split por paciente** (B precisa ser
   re-treinado no mesmo subconjunto usado por C).
6. Atenção: as queixas estão em inglês e são abreviadas ("CP", "SOB"), mas o prompt atual é em
   português. Medir a taxa de extração vazia antes de treinar.
7. Somente LLM local: o DUA do PhysioNet restringe o envio dos dados a serviços de terceiros.

## Comportamento na API

- O modelo é treinado **antes** (`make train`) e apenas carregado na inicialização a partir de
  `HEALTHFLOW_ML_MODEL_DIR` (padrão `models/synthetic-v1`).
- Modelo ausente, incompatível ou de outro feature set → API sobe com `ml_model_loaded: false` e usa o
  fallback conservador `URGENT_CARE`.
- Confiança abaixo de `HEALTHFLOW_ML_LOW_CONFIDENCE_THRESHOLD` (0.55) → escolhe a mais grave entre as
  duas classes mais prováveis.

## Limitações

- Experimento A: dados sintéticos. Experimento B: domain shift EUA × SUS, rótulo derivado do ESI por
  regra do projeto, sem idade (ver [datasets.md](datasets.md)).
- Sem calibração de probabilidades, sem análise de equidade entre grupos, sem intervalos de confiança.
- Um único *split* de teste; variância entre *seeds* não estimada.
- Sem validação clínica, sem aprovação regulatória.
