# Machine Learning — classificador de encaminhamento

> **PROTÓTIPO ACADÊMICO — NÃO UTILIZAR PARA DECISÕES CLÍNICAS REAIS.**
> Todas as métricas abaixo foram obtidas em **dados sintéticos** e **não são evidência clínica**.

## Problema

Classificação supervisionada **multiclasse** do nível de encaminhamento. O modelo não diagnostica doenças.

| Classe (`CareLevel`) | Serviço (`ServiceType`) |
|---|---|
| `PRIMARY_CARE` | UBS |
| `URGENT_CARE` | UPA |
| `EMERGENCY` | Pronto-socorro + orientação SAMU 192 |

## Dataset

- Gerador: `app/ml/training/dataset.py` (`make dataset`), `random_state=42`, 4000 linhas.
- Versão: `synthetic-v1`. Arquivo: `data/processed/routing_synthetic_v1.csv` (+ `.meta.json` com o aviso).
- Marcação: **"ACADEMIC / SYNTHETIC DATA — NOT FOR CLINICAL USE"**.
- Processo gerador: para cada exemplo sorteia-se uma classe (55% / 30% / 15%), depois sintomas de um
  "perfil" da classe, intensidade, duração, faixa etária e fatores de risco. Ruídos propositais:
  30% dos exemplos ganham um sintoma aleatório de outro perfil; casos de urgência em idosos com fator de
  risco viram emergência com 40% de probabilidade; 5% dos rótulos são trocados aleatoriamente.
- Consequência: as métricas medem **o quanto o modelo recupera esse processo gerador**, não a
  realidade clínica.

### Trocando o dataset

Qualquer dataset que siga o esquema CSV abaixo pode substituir o sintético:

```text
symptoms,severity,duration_minutes,age_range,risk_factors,label
chest_pain|shortness_of_breath,severe,20,adult,cardiovascular_disease,EMERGENCY
```

```bash
uv run python -m app.ml.training.train --dataset caminho.csv --dataset-version real-v1
```

## Features (`app/ml/features.py`)

Um único módulo gera as features no treino e na inferência. A lista `FEATURE_NAMES` é gravada em
`models/metadata.json`; se o código mudar, o artefato antigo é **recusado** (evita *train/serve skew*).

| Grupo | Features |
|---|---|
| Sintomas | 22 indicadores binários (`symptom__chest_pain`, ...) |
| Contagem | `symptom_count` |
| Intensidade | `severity_rank` (1–3) + `severity_unknown` |
| Duração | `duration_log_hours` = log(1 + horas) + `duration_unknown` |
| Idade | `age_range_rank` (0–4) + `age_unknown` |
| Contexto | 5 fatores de risco binários (cardiovascular, respiratório, diabetes, imunossupressão, gestação) |

O nome, o histórico livre e os medicamentos do paciente **não** entram no modelo.

## Modelos comparados

Todos com `class_weight="balanced"` (EMERGENCY é minoritária) e `random_state=42`.

- `LogisticRegression` (com `StandardScaler`)
- `DecisionTreeClassifier` (`max_depth=6`, `min_samples_leaf=10`)
- `RandomForestClassifier` (200 árvores, `max_depth=10`, `min_samples_leaf=5`)

### Protocolo

1. *Split* estratificado 80/20 (treino / teste reservado).
2. Validação cruzada estratificada de 5 *folds* **somente no treino**.
3. Seleção: maior *recall* de EMERGENCY (arredondado a 2 casas), desempate por F1 macro.
4. Reajuste do escolhido no treino completo e **uma** avaliação no teste reservado.

### Resultados (execução de 2026-10-07, `synthetic-v1`)

Validação cruzada (treino, n=3200):

| Modelo | Accuracy | Precisão macro | Recall macro | F1 macro | Recall EMERGENCY |
|---|---|---|---|---|---|
| Logistic Regression | 0.880 | 0.847 | 0.854 | 0.850 | 0.798 |
| Decision Tree | 0.854 | 0.829 | 0.843 | 0.833 | 0.799 |
| **Random Forest** (selecionado) | 0.900 | 0.881 | 0.884 | 0.881 | 0.826 |

Teste reservado (n=800), Random Forest:

| Classe | Precisão | Recall | F1 | Suporte |
|---|---|---|---|---|
| PRIMARY_CARE | 0.983 | 0.938 | 0.960 | 436 |
| URGENT_CARE | 0.859 | 0.943 | 0.899 | 226 |
| EMERGENCY | 0.904 | **0.891** | 0.898 | 138 |

Accuracy 0.931 · F1 macro 0.919.

Matriz de confusão (linhas = real, colunas = previsto):

| | PRIMARY | URGENT | EMERGENCY |
|---|---|---|---|
| **PRIMARY** | 409 | 22 | 5 |
| **URGENT** | 5 | 213 | 8 |
| **EMERGENCY** | 2 | 13 | 123 |

Os números exatos são regenerados por `make train` e gravados em `models/metadata.json`.

### Leitura crítica

- 15 de 138 emergências (≈11%) seriam rebaixadas **pelo modelo sozinho**. Por isso o ML nunca é a
  autoridade final: o Safety Engine impõe pisos e o Care Routing nunca reduz um piso.
- O F1 do teste ficou acima da média da validação cruzada; com dados sintéticos e um único *split*,
  isso reflete variância, não "generalização clínica".
- Um modelo que acerta o gerador sintético pode falhar totalmente em dados reais.

## Comportamento em produção do protótipo

- O modelo é treinado **antes** (`make train`) e apenas carregado na subida da API.
- O artefato (`joblib`) tem SHA-256 registrado nos metadados e é verificado antes de carregar
  (`joblib` desserializa código: só carregamos o artefato que nós mesmos produzimos).
- Modelo ausente/incompatível → API sobe, `/health` mostra `ml_model_loaded: false`, e o roteamento usa
  o fallback conservador `URGENT_CARE` (código `ML_UNAVAILABLE_CONSERVATIVE_FALLBACK`).
- Confiança abaixo de `HEALTHFLOW_ML_LOW_CONFIDENCE_THRESHOLD` (0.55) → escolhe a mais grave entre as
  duas classes mais prováveis (`ML_LOW_CONFIDENCE_ESCALATED`).

## Limitações

- Dados sintéticos; nenhum dado real de pacientes.
- Vocabulário de sintomas fechado (22 códigos); o que o LLM não mapear não chega ao modelo.
- Não há calibração de probabilidades nem análise de equidade entre grupos.
- Sem validação clínica, sem aprovação regulatória.
