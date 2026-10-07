# Datasets

> **PROTÓTIPO ACADÊMICO — NÃO UTILIZAR PARA DECISÕES CLÍNICAS REAIS.**
> Nenhum dataset deste projeto valida clinicamente o encaminhamento no SUS.

## Visão geral

Toda fonte de dados é convertida para um **esquema canônico** único,
`RoutingTrainingExample` (`app/ml/data/schemas.py`). Os *feature builders* leem apenas esse esquema,
então novas fontes ou novas features não exigem reescrever os carregadores. `None` = "não disponível".

| Campo | Sintético | MIMIC-IV-ED (triage) | Triagegeist (Kaggle) |
|---|---|---|---|
| `group_id` (chave pseudônima do paciente; só para o *split*) | — | SHA-256 truncado de `subject_id` | SHA-256 truncado do ID de paciente, **se a coluna existir** |
| `label` | gerado | mapeado de `acuity` (ESI) | mapeado de `triage_acuity` |
| `symptoms`, `severity`, `duration_minutes`, `risk_factors` | ✓ | — (experimento C) | — (futuro, via LLM) |
| `age_range` | ✓ | — | ✓ se houver coluna de idade |
| `heart_rate`, `respiratory_rate`, `oxygen_saturation`, `systolic_bp`, `diastolic_bp` | — | ✓ | ✓ se detectadas |
| `temperature_celsius` | — | ✓ (convertido de °F) | ✓ (°C ou °F, ver §11) |
| `pain` (0–10) | — | ✓ | ✓ se detectada |
| `chief_complaint` (texto livre; nunca usado diretamente como feature) | — | ✓ | ✓ se detectada |

| Experimento | Dataset | Feature set | Modelo salvo em |
|---|---|---|---|
| A `synthetic_baseline` | `synthetic-v1` | `healthflow-symptoms-v1` (o mesmo da API) | `models/synthetic-v1/` |
| B `structured_mimic_baseline` | `mimic-ed-v1` | `mimic-structured-vitals-v1` | `models/mimic-structured-v1/` |
| C (planejado) | `mimic-ed-v1` + LLM | `healthflow-symptoms-v1` | — |
| T `structured_triagegeist_baseline` | `triagegeist-v1` | `triagegeist-structured-v1` | `models/triagegeist-structured-v1/` |

---

## 1. Dataset sintético (`synthetic-v1`)

**ACADEMIC / SYNTHETIC DATA — NOT FOR CLINICAL USE.**

- Gerado por `make dataset-synthetic` (`app/ml/data/synthetic.py`), seed 42, 4000 linhas.
- Rótulos vêm de um processo gerador escrito à mão (detalhes em [machine-learning.md](machine-learning.md)).
- Linhas são pacientes fictícios independentes: não há `group_id`, e o *split* é estratificado por linha.
- Serve como **baseline acadêmico** e como o modelo usado pela API (as features são as que a API
  consegue calcular a partir do relato).

## 2. MIMIC-IV-ED

O [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/) é um módulo do MIMIC-IV com atendimentos
de **pronto-socorro de um hospital universitário de Boston (EUA)**, desidentificados. O Health-flow usa
apenas a tabela `triage`:

```text
subject_id, stay_id, temperature, heartrate, resprate, o2sat, sbp, dbp, pain, acuity, chiefcomplaint
```

### 3. Como obter acesso

O MIMIC é **dado de acesso credenciado** no PhysioNet. Cada pessoa que usar os dados precisa:

1. criar uma conta no PhysioNet;
2. solicitar *credentialed access* (identificação e referência);
3. concluir o treinamento exigido sobre pesquisa com seres humanos (indicado na página do dataset);
4. assinar o **Data Use Agreement (DUA)** do dataset;
5. baixar os arquivos manualmente.

Os requisitos exatos estão na página do PhysioNet e podem mudar; siga a versão atual de lá.

### 4. Onde colocar os arquivos

```text
data/raw/mimic-iv-ed/triage.csv.gz     # formato distribuído
data/raw/mimic-iv-ed/triage.csv        # também aceito (descompactado)
```

```bash
make dataset-mimic                                   # procura em data/raw/mimic-iv-ed/
make dataset-mimic MIMIC_TRIAGE=/outro/caminho/triage.csv.gz MIMIC_SOURCE_VERSION=2.2
```

Saída: `data/processed/routing_mimic_v1.csv` + `routing_mimic_v1.meta.json` (relatório agregado da
preparação: linhas lidas/descartadas, faltantes por campo, valores implausíveis, distribuição de classes,
número de pacientes).

### 5. Os dados não são redistribuídos

- O projeto **nunca baixa** o MIMIC automaticamente e não guarda credenciais do PhysioNet.
- `.gitignore` bloqueia `data/raw/mimic*/`, `data/raw/mimic-iv-ed*/`, `*.csv.gz` e todo `data/processed/`.
- Testes usam apenas arquivos **falsos** com o mesmo formato (`tests/ml/fixtures/fake_mimic.py`).
- Logs mostram apenas contagens agregadas: nunca `subject_id`, `stay_id`, queixa ou linha individual.
- O CSV processado troca `subject_id` por uma chave pseudônima (hash) e descarta `stay_id`.
  Isso é **pseudonimização, não anonimização**: o arquivo processado continua sujeito ao DUA.
- Benchmarks publicados contêm apenas métricas agregadas.
- O DUA do PhysioNet restringe o envio dos dados a serviços de terceiros. O experimento C deve usar
  **somente LLM local** (Ollama), nunca uma API externa.

### 6. Acuity / ESI

`acuity` é o nível do **Emergency Severity Index (ESI)** atribuído pela enfermagem na triagem de
pronto-socorros nos EUA:

| ESI | Ideia geral |
|---|---|
| 1 | precisa de intervenção imediata para salvar a vida |
| 2 | alto risco, confusão/letargia ou dor/sofrimento intenso |
| 3 | estável, mas deve precisar de vários recursos (exames, medicação IV, etc.) |
| 4 | deve precisar de um recurso |
| 5 | não deve precisar de recursos |

ESI ordena **prioridade dentro de um pronto-socorro**. Não é um encaminhamento para outro tipo de serviço.

### 7. Mapeamento experimental para três classes

```text
ESI 1–2 → EMERGENCY
ESI 3   → URGENT_CARE
ESI 4–5 → PRIMARY_CARE
```

- Implementado em `map_esi_to_care_level` (`app/ml/data/mimic_ed.py`).
- `acuity` ausente, não inteiro ou fora de 1–5 → **linha descartada** (contada no relatório). Nenhuma
  classe é inventada.
- **É uma simplificação criada pelo Health-flow** apenas para permitir a comparação experimental entre
  as mesmas três classes do sistema. **Não é protocolo oficial** e **não** significa que ESI 4–5
  equivale clinicamente a atendimento em UBS no SUS. Todos esses pacientes foram, de fato, atendidos em
  um pronto-socorro.

### Limpeza (nada silencioso)

| Situação | Tratamento |
|---|---|
| `acuity` inválido/ausente | linha descartada |
| `subject_id` ausente | linha descartada (o *split* por paciente precisa dele) |
| sinal vital vazio | `None` |
| sinal vital fora de limites amplos de plausibilidade física | `None` + contado como implausível |
| `pain` não numérico ("unable", "denies"...) ou fora de 0–10 | `None` + contado |
| `temperature` | °F → °C; sem tentativa de adivinhar unidade |
| coluna opcional ausente | campo `None` em todas as linhas + aviso |

Limites (unidade de origem): FC 20–300, FR 4–80, SpO₂ 50–100, PAS 40–300, PAD 20–200,
temperatura 86–113 °F. São **guardas de qualidade de dado**, não limiares clínicos.

A imputação **não** acontece na preparação: valores ausentes chegam como `NaN` ao `Pipeline` do
scikit-learn, onde `SimpleImputer(strategy="median", add_indicator=True)` é ajustado **só nos dados de
treino** de cada *fold*. Não preenchemos com zero porque zero tem significado clínico (ex.: dor 0).

### Idade

A tabela `triage` **não tem idade**. A idade do MIMIC-IV fica em outro módulo (`hosp/patients`,
`anchor_age`), e usá-la exigiria juntar tabelas pelo identificador do paciente. Isso está fora do
escopo deste experimento. Por isso o baseline estruturado **não usa idade**, e nada é inferido de
`subject_id`.

## 8. Limitações

- Um único hospital, um único país, um período específico.
- O rótulo é a **decisão de triagem da enfermagem**, não o desfecho: o modelo aprende a reproduzir o ESI
  atribuído, com seus próprios erros e vieses.
- Espera-se forte desbalanceamento entre as classes após o mapeamento (ESI 4–5 tende a ser minoria em
  pronto-socorro). Confira a distribuição real em `routing_mimic_v1.meta.json` antes de interpretar
  métricas.
- O baseline estruturado usa só sinais vitais e dor, sem queixa, idade ou histórico. Um desempenho
  limitado é esperado e é informativo.
- Sinais vitais de triagem são medidos **presencialmente**. O Health-flow recebe texto do usuário em
  casa, sem sinais vitais. O experimento B mede o que sinais vitais explicam do ESI, não o que o
  Health-flow conseguiria em produção.

## 9. Domain shift: EUA × SUS

O MIMIC-IV-ED descreve **pronto-socorros nos EUA**. O Health-flow estuda **navegação no SUS**.
Há diferença em:

| Dimensão | MIMIC-IV-ED | Health-flow / SUS |
|---|---|---|
| População | pacientes de um centro em Boston | população brasileira, perfis regionais diversos |
| Sistema de saúde | sistema dos EUA, com seguros privados/públicos | sistema universal, com rede UBS / UPA / hospital / SAMU |
| Fluxo de acesso | o paciente **já chegou** ao pronto-socorro | o paciente **ainda decide para onde ir** |
| Protocolos | ESI | protocolos locais (ex.: classificação de risco adotada por cada serviço) |
| Idioma | inglês (queixa) | português (relato livre) |
| Disponibilidade de serviços | serviços de um hospital terciário | variável por município |
| Definição das classes | prioridade **dentro** do pronto-socorro | **tipo de serviço** a procurar |

## 10. Por que este dataset NÃO valida o encaminhamento no SUS

1. As classes do Health-flow foram **derivadas** do ESI por uma regra criada pelo projeto; não há
   verdade de referência de "deveria ter ido à UBS / UPA / emergência".
2. Todos os pacientes estavam num pronto-socorro: falta a população que resolveria o problema na
   atenção primária.
3. Contexto, protocolos, idioma e população são diferentes (domain shift).
4. Não há avaliação de desfecho, segurança ou impacto.

O uso correto no artigo: **comparação experimental de algoritmos e de representações de entrada**
(sinais vitais reais × relato interpretado por LLM × dados sintéticos), com as limitações acima
declaradas. Validar o encaminhamento exigiria dados brasileiros rotulados por profissionais, protocolo
aprovado e estudo clínico.

---

## 11. Kaggle Triagegeist (`triagegeist-v1`)

> **Triagegeist não representa o SUS.** É uma fonte adicional para comparar algoritmos com dados
> estruturados de triagem. Não valida encaminhamento para UBS / UPA / emergência.

### Como obter

1. Crie uma conta no [Kaggle](https://www.kaggle.com) e abra a página do Triagegeist.
2. Leia e aceite as regras/licença mostradas lá. Se for uma competição, os dados costumam ser
   restritos ao uso definido pelas regras da competição e **não podem ser redistribuídos**. Confira os
   termos atuais na própria página, porque eles prevalecem sobre este documento.
3. Baixe **manualmente** o `train.csv` pela interface do Kaggle.

O Health-flow **não baixa** o dataset, **não usa** a API do Kaggle e **não guarda** credenciais.
`.gitignore` bloqueia `data/raw/triagegeist/`, `kaggle.json`, `.kaggle/` e todo `data/raw/` e
`data/processed/`.

### Onde colocar

```text
data/raw/triagegeist/train.csv
```

```bash
make dataset-triagegeist                                   # procura data/raw/triagegeist/train.csv
make dataset-triagegeist TRIAGEGEIST_RAW=/outro/train.csv TRIAGEGEIST_SOURCE_VERSION=<versão/data>
make train-triagegeist                                     # → models/triagegeist-structured-v1/
make evaluate-triagegeist                                  # só no teste reservado
make benchmark-summary
```

Saída: `data/processed/routing_triagegeist_v1.csv` + `routing_triagegeist_v1.meta.json`, com
`dataset_name`, `dataset_version`, `source`, `number_of_rows`, `number_of_patients`,
`class_distribution`, `discarded_rows` (total e por motivo), `mapping_used`, `feature_set`,
`features`, `disclaimer` e `preparation` (faltantes por campo, valores implausíveis, distribuição
original de `triage_acuity` e o **mapeamento de colunas detectado**).

### Schema: detectado, não assumido

O cabeçalho real do `train.csv` **ainda não foi verificado neste repositório**: nenhum arquivo real
estava disponível quando o loader foi escrito. Por isso `app/ml/data/triagegeist.py` (`detect_columns`)
compara o cabeçalho, normalizado (minúsculas, não alfanuméricos → `_`), com uma lista explícita de nomes
aceitos por campo canônico:

| Campo canônico | Nomes aceitos (após normalização) | Unidade assumida |
|---|---|---|
| alvo | `triage_acuity` (mude com `--target-column`) | 1–5 |
| `group_id` | `patient_id`, `subject_id`, `patient`, `patient_key`, `patient_identifier`, `person_id`, `mrn` | — |
| `heart_rate` | `heart_rate`, `heartrate`, `hr`, `pulse`, `pulse_rate` | bpm |
| `respiratory_rate` | `respiratory_rate`, `resp_rate`, `resprate`, `rr`, `respiration_rate` | irpm |
| `oxygen_saturation` | `oxygen_saturation`, `o2_saturation`, `o2sat`, `o2_sat`, `spo2`, `sao2` | % |
| `systolic_bp` | `systolic_bp`, `sbp`, `systolic_blood_pressure`, `bp_systolic`, `systolic` | mmHg |
| `diastolic_bp` | `diastolic_bp`, `dbp`, `diastolic_blood_pressure`, `bp_diastolic`, `diastolic` | mmHg |
| `temperature_celsius` | `temperature_c`, `temp_c`, `temperature_celsius`, … (°C) · `temperature_f`, `temp_f`, … (°F) · `temperature`, `temp`, `body_temperature` (ambíguo) | ver abaixo |
| `pain` | `pain`, `pain_score`, `pain_scale`, `pain_level`, `nrs_pain` | 0–10 |
| `age_range` | `age`, `age_years`, `patient_age` → faixa (`AgeRange.from_age`) | anos |
| `chief_complaint` | `chief_complaint`, `chiefcomplaint`, `complaint`, `presenting_complaint`, `reason_for_visit` | texto |

- Sem `triage_acuity`, ou sem **nenhuma** coluna estruturada (sinais vitais/dor), a preparação **falha**
  e lista as colunas encontradas.
- Duas colunas para o mesmo campo (ex.: `heart_rate` e `pulse`) → **erro**, nunca escolha silenciosa.
- Colunas de identificação (`id`, `visit_id`, `encounter_id`, `stay_id`, …) e colunas não reconhecidas
  **nunca** viram feature. Só seus nomes aparecem em `column_mapping` no metadata.
- Se o arquivo real usar outros nomes, acrescente os aliases em `triagegeist.py`, documente aqui a
  correspondência e rode de novo. **Antes de citar resultados, confira `column_mapping` no
  `.meta.json`.**
- Colunas que existirem mas não estiverem na tabela (ex.: modo de chegada, sexo, desfecho/disposição)
  ficam **fora** do baseline v1. Variáveis de desfecho (internação, alta, tempo de permanência) seriam
  **vazamento** de informação posterior à triagem e não devem ser adicionadas como features.

### Limpeza (nada silencioso)

| Situação | Tratamento |
|---|---|
| `triage_acuity` ausente, não inteiro ou fora de 1–5 | linha descartada (`invalid_or_missing_triage_acuity`) |
| coluna de paciente existe, mas o valor está vazio | linha descartada (`missing_patient_id`) |
| valor vazio | `None` |
| não numérico ou fora de limites amplos de plausibilidade | `None` + contado em `implausible_set_to_missing` |
| temperatura com unidade explícita no nome | aceita só na faixa da unidade (30–45 °C ou 86–113 °F); °F → °C |
| temperatura com nome ambíguo (`temperature`) | cada valor é classificado pelas faixas **disjuntas** 30–45 (°C) e 86–113 (°F); fora das duas → `None`. Conversões contadas |
| coluna opcional ausente | campo `None` em todas as linhas (aparece como `null` no mapeamento) |

Limites: FC 20–300, FR 4–80, SpO₂ 50–100, PAS 40–300, PAD 20–200, dor 0–10, idade 0–120. São **guardas
de qualidade de dado**, não limiares clínicos. Não preenchemos com zero: ausentes viram `NaN` e são
imputados (mediana + indicador de ausência) dentro do `Pipeline`, ajustado só no treino.

### Mapeamento experimental 1–5 → 3 classes

```text
triage_acuity 1–2 → EMERGENCY
triage_acuity 3   → URGENT_CARE
triage_acuity 4–5 → PRIMARY_CARE
```

Implementado em `map_triage_acuity_to_care_level` (mesma tabela do MIMIC). É uma **simplificação
experimental do Health-flow**, criada só para comparar as mesmas três classes. **Não é protocolo do
SUS**, e acuity 4–5 **não** equivale oficialmente a atendimento em UBS. O código também não afirma que
a escala do Triagegeist seja ESI: confira a definição da escala na página do dataset.

### Split e vazamento por paciente

- **Com coluna de ID de paciente**: `StratifiedGroupKFold` no teste reservado e na CV. O mesmo paciente
  nunca fica em treino e teste (`stratified_group_by_patient_80_20__cv_stratified_group_kfold_5`).
- **Sem coluna de ID de paciente**: o *split* é estratificado por linha
  (`stratified_row_80_20__cv_stratified_kfold_5`), `number_of_patients` fica `null` e o metadata
  registra `patient_level_split_possible: false`. Nesse caso **não há proteção contra vazamento entre
  visitas do mesmo paciente**, e as métricas podem ser otimistas. Isso deve ser declarado ao citar os
  resultados.

### Triagegeist × MIMIC × SUS

| Dimensão | Triagegeist (Kaggle) | MIMIC-IV-ED | Health-flow / SUS |
|---|---|---|---|
| Origem | dataset público do Kaggle; procedência e população conforme a página do dataset | 1 hospital universitário em Boston (EUA) | população brasileira |
| Acesso | conta Kaggle + aceite das regras | credenciado (PhysioNet + DUA) | — |
| Rótulo | `triage_acuity` 1–5 (escala definida pelo dataset) | ESI 1–5 | não há rótulo de referência |
| Momento | triagem (paciente já chegou ao serviço) | triagem no pronto-socorro | **antes** de decidir aonde ir |
| Entrada | sinais vitais estruturados (+ queixa, se houver) | idem | relato livre em português, sem sinais vitais |
| Classes do Health-flow | derivadas por regra do projeto | derivadas por regra do projeto | UBS / UPA / emergência |

### Limitações e domain shift

- Não se sabe (no repositório) se os dados são reais, simulados ou de competição com amostragem própria.
  Verifique na página do Kaggle e declare isso no artigo.
- O rótulo é a acuity atribuída na triagem, não um desfecho clínico.
- Sinais vitais são medidos presencialmente, mas o Health-flow recebe texto em casa. O experimento T
  mede o que dados estruturados explicam da acuity, não o desempenho do Health-flow em produção.
- Diferenças de país, população, protocolo e idioma em relação ao SUS (domain shift): resultados **não**
  se transferem para o SUS.
- Sem ID de paciente, as métricas podem estar infladas por vazamento entre visitas.
- O modelo `models/triagegeist-structured-v1/` **não é usado pela API**: ela só aceita o feature set
  `healthflow-symptoms-v1` (configurado em `HEALTHFLOW_ML_MODEL_DIR`) e recusa este modelo na
  inicialização.

### Futuro: queixa → LLM local

A queixa (`chief_complaint`) é preservada no CSV processado, mas **ainda não é processada**. O caminho
planejado é o mesmo do experimento C: `chief_complaint` → Qwen3 local via Ollama → `SymptomExtraction`
→ `HealthFlowSymptomFeatureBuilder`, com cache local e execução em lotes. Nada disso é executado nesta
etapa.
