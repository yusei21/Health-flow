# Datasets

> **PROTÓTIPO ACADÊMICO — NÃO UTILIZAR PARA DECISÕES CLÍNICAS REAIS.**
> Nenhum dataset deste projeto valida clinicamente o encaminhamento no SUS.

## Visão geral

Toda fonte de dados é convertida para um **esquema canônico** único,
`RoutingTrainingExample` (`app/ml/data/schemas.py`). Os *feature builders* leem apenas esse esquema,
então novas fontes ou novas features não exigem reescrever os carregadores. `None` = "não disponível".

| Campo | Sintético | MIMIC-IV-ED (triage) |
|---|---|---|
| `group_id` (chave pseudônima do paciente; só para o *split*) | — | SHA-256 truncado de `subject_id` |
| `label` | gerado | mapeado de `acuity` (ESI) |
| `symptoms`, `severity`, `duration_minutes`, `age_range`, `risk_factors` | ✓ | — (experimento C) |
| `heart_rate`, `respiratory_rate`, `oxygen_saturation`, `systolic_bp`, `diastolic_bp` | — | ✓ |
| `temperature_celsius` | — | ✓ (convertido de °F) |
| `pain` (0–10) | — | ✓ |
| `chief_complaint` (texto livre; nunca usado diretamente como feature) | — | ✓ |

| Experimento | Dataset | Feature set | Modelo salvo em |
|---|---|---|---|
| A `synthetic_baseline` | `synthetic-v1` | `healthflow-symptoms-v1` (o mesmo da API) | `models/synthetic-v1/` |
| B `structured_mimic_baseline` | `mimic-ed-v1` | `mimic-structured-vitals-v1` | `models/mimic-structured-v1/` |
| C (planejado) | `mimic-ed-v1` + LLM | `healthflow-symptoms-v1` | — |

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
