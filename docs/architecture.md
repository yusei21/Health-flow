# Arquitetura — Health-flow

> **PROTÓTIPO ACADÊMICO — NÃO UTILIZAR PARA DECISÕES CLÍNICAS REAIS.**
> O Health-flow não diagnostica, não prescreve e não substitui profissionais de saúde.

## Estado atual

| Fase | Conteúdo | Estado |
|---|---|---|
| A | FastAPI, config, schemas, `LLMProvider`, Ollama, extração estruturada | ✅ implementado |
| B | Dataset sintético, features, treino/avaliação, persistência, inferência | ✅ implementado |
| B+ | Esquema canônico de dados, loader MIMIC-IV-ED, split por paciente, métricas de triagem, benchmarks | ✅ implementado (sem dados MIMIC no repositório; ver [datasets.md](datasets.md)) |
| C | Safety Engine, regras simuladas, *override* | ✅ implementado |
| D | Agent Harness, Care Routing, Context Builder | ✅ implementado |
| E | `MockFacilityProvider`, distância, seleção de unidade, endpoint | ✅ implementado |
| F | PostgreSQL, persistência do prontuário simplificado, auditoria, autenticação real | ⏳ planejado (Postgres+pgvector já sobe no compose) |
| G | pgvector, pipeline RAG, documentos oficiais | ⏳ planejado |
| Fase 2 | Exames, procedimentos e cobertura SUS/plano | ⏳ planejado |
| Fase 3 | Localização de medicamentos | ⏳ planejado |

Estilo: **monólito modular FastAPI**. Sem filas, microsserviços ou orquestradores externos.

## Onde está cada técnica de IA

| Pergunta | Resposta | Código |
|---|---|---|
| Onde existe IA generativa? | LLM local (Ollama, `qwen3:4b`) converte texto livre em JSON tipado | `app/llm/`, `app/agents/intent_agent.py` |
| Onde existe Machine Learning? | Classificador supervisionado treinado (scikit-learn) sugere o nível de encaminhamento | `app/ml/` — ver [machine-learning.md](machine-learning.md) |
| Onde existe Agent Harness? | Harness autônomo orientado a estado: um *planner* escolhe a próxima ação num conjunto **fechado**, uma *policy* valida, um *executor* executa | `app/harness/` |
| Onde existe RAG? | Ainda não implementado (Fase G). Postgres com pgvector já provisionado | `scripts/db/init-pgvector.sql` |

### Por que LLM e ML são coisas diferentes aqui

- **LLM (generativo, pré-treinado)**: entende linguagem natural ("dor forte no peito há 20 min") e
  devolve **estrutura** (`chest_pain`, `20`, `severe`). Não é treinado por nós, não produz
  probabilidades calibradas por classe e pode alucinar — por isso a saída é restrita por JSON Schema,
  validada por Pydantic e **não decide nada**.
- **ML (discriminativo, treinado por nós)**: recebe **features numéricas** derivadas da estrutura e do
  contexto e devolve `P(PRIMARY_CARE)`, `P(URGENT_CARE)`, `P(EMERGENCY)`. É reproduzível
  (`random_state`), avaliável com métricas e versionado — mas é auxiliar.
- **Regras (determinísticas)**: sinais críticos com identificador, testáveis um a um; só podem
  **subir** o nível.

```text
LLM entende.  ML classifica.  Regras protegem.  Tools consultam.  Harness coordena.
```

## Agent Harness autônomo (Etapa 1)

**Autonomia aqui = escolher dinamicamente o próximo passo entre ações permitidas.** Não significa deixar
um LLM executar operações arbitrárias: nada fora do enum `HarnessAction` pode rodar, seja qual for o
*planner*.

```text
AutonomousHealthFlowHarness            app/harness/autonomous_harness.py
├── Planner        (Protocol)          app/harness/planner.py   next_action(state) -> PlannedAction
│   ├── DeterministicPlanner           padrão; regras sobre o HarnessState
│   └── JevPlanner                     código experimental preservado, fora da composição da API
├── HarnessPolicy  (policy/guard)      app/harness/policy.py    validate(action, state)
├── ActionExecutor (registry)          app/harness/executor.py  um handler tipado por ação
├── HarnessState                       app/harness/state.py     dados + trilha de auditoria
└── HarnessLimits  (termination guards)                         passos, LLM, tools, timeout
```

### Loop controlado

```python
while not state.finished:
    planned = await planner.next_action(state)  # propõe UMA ação + reason_code
    policy.validate(planned, state)  # ordem de segurança, conjunto permitido, orçamentos
    await executor.execute(planned, state)  # handler do registry; registra histórico
```

O loop roda dentro de `asyncio.timeout(HarnessLimits.timeout_seconds)`. Qualquer violação da *policy*
ou estouro de orçamento **interrompe** a execução com erro controlado (`HarnessPolicyError`,
`HarnessLimitError` ou `HarnessTimeoutError`): o harness nunca "tenta outra coisa" por conta própria.

### Ações permitidas (`HarnessAction`)

| Ação | Componente | Tipo (orçamento) |
|---|---|---|
| `SAFETY_PRECHECK` | `SafetyEngine.assess(texto_bruto, None, None)`: só padrões de texto, **antes do LLM** | interna |
| `EXTRACT_SYMPTOMS` | `IntentAgent` → `LLMProvider` | LLM |
| `LOAD_PATIENT_CONTEXT` | `PatientContextAgent` → repositório + minimização | tool |
| `RUN_SAFETY_ASSESSMENT` | `SafetyEngine.assess(texto, extração, contexto)` | interna |
| `RUN_ML` | `RoutingInferenceService` | tool |
| `APPLY_ROUTING` | `CareRoutingAgent.decide(piso efetivo, ML)` | interna |
| `SEARCH_FACILITIES` | `NavigationAgent` → `FacilityProvider` | tool |
| `FINALIZE` | marca `finished` e `finish_reason` | interna |

### Fluxo

```text
                    ┌──────────────────┐
  POST /routing ──▶ │ SAFETY_PRECHECK  │  regras de texto no relato bruto
                    └────────┬─────────┘
             red flag? ──────┼──────────── não
                 │                          │
                 │                 ┌────────▼─────────┐
                 │                 │ EXTRACT_SYMPTOMS │  LLM (1 chamada no máximo)
                 │                 └────────┬─────────┘
                 │          falhou ─────────┼────────── ok
                 │            │             │
                 │            ▼    ┌────────▼──────────────┐
                 │        FINALIZE │ LOAD_PATIENT_CONTEXT  │
                 │   (erro LLM 503)└────────┬──────────────┘
                 │                 ┌────────▼──────────────┐
                 │                 │ RUN_SAFETY_ASSESSMENT │  texto + estrutura + idade
                 │                 └────────┬──────────────┘
                 │                red flag? ┼─── não
                 │                   │      │
                 │                   │  ┌───▼────┐
                 │                   │  │ RUN_ML │  falha → fallback URGENT_CARE
                 │                   │  └───┬────┘
                 ▼                   ▼      ▼
              ┌──────────────────────────────────┐
              │ APPLY_ROUTING  (piso efetivo+ML) │  nível final ≥ piso de segurança
              └────────────────┬─────────────────┘
              ┌────────────────▼─────────────────┐
              │ SEARCH_FACILITIES                │  falha → mantém decisão, facility: null
              └────────────────┬─────────────────┘
                          ┌────▼─────┐
                          │ FINALIZE │ → build_routing_response
                          └──────────┘
```

| Caso | Sequência |
|---|---|
| Não crítico | `SAFETY_PRECHECK → EXTRACT_SYMPTOMS → LOAD_PATIENT_CONTEXT → RUN_SAFETY_ASSESSMENT → RUN_ML → APPLY_ROUTING → SEARCH_FACILITIES → FINALIZE` |
| *Red flag* no texto bruto | `SAFETY_PRECHECK → APPLY_ROUTING → SEARCH_FACILITIES → FINALIZE` (sem LLM, sem ML) |
| *Red flag* só após a extração | `… → RUN_SAFETY_ASSESSMENT → APPLY_ROUTING → SEARCH_FACILITIES → FINALIZE` (sem ML) |
| LLM falha, sem *red flag* | `SAFETY_PRECHECK → EXTRACT_SYMPTOMS → FINALIZE` → HTTP 503 com lembrete do SAMU 192 |

### Planner

`Planner` é um `Protocol` com um método: `next_action(state) -> PlannedAction(action, reason_code)`.
O `DeterministicPlanner` é o padrão: regras curtas e ordenadas sobre o estado, que **pulam etapas
desnecessárias** (LLM e ML numa *red flag* do precheck, ML em qualquer *red flag*). Cada decisão leva um
`reason_code` (`ReasonCode`), por exemplo `PRECHECK_RED_FLAG_SKIP_LLM_AND_ML` ou `RED_FLAG_SKIP_ML`.

### Policy / guard (autoridade acima do planner)

`HarnessPolicy.validate` roda **antes de toda ação**, para qualquer *planner*:

- a ação pertence ao conjunto permitido para o `Intent` da requisição;
- a execução não terminou e a ação ainda não foi executada (cada ação roda no máximo uma vez);
- orçamentos de passos, chamadas de LLM e chamadas de tools;
- pré-condições de segurança:
  - `SAFETY_PRECHECK` é sempre a primeira ação;
  - `EXTRACT_SYMPTOMS` é **proibida** se o precheck achou *red flag* (não esperar o LLM);
  - `RUN_ML` só depois de `RUN_SAFETY_ASSESSMENT`, com extração e contexto, e **nunca** com *red flag*;
  - `APPLY_ROUTING` exige os pisos conhecidos (*red flag* do precheck ou avaliação completa) e, sem
    *red flag*, que o ML tenha sido **tentado** (preserva o fallback conservador em vez de pular o ML);
  - `SEARCH_FACILITIES` exige decisão de *routing*; `FINALIZE` exige decisão ou erro controlado.

O **piso efetivo** (`HarnessState.effective_safety`) combina precheck e avaliação completa: união das
regras e o nível mínimo **mais severo**. Uma etapa posterior só pode adicionar regras ou subir o piso.
O `SafetyEngine` não foi alterado.

### Executor

`ActionExecutor` mantém um *registry* `dict[HarnessAction, handler]`, verificado na construção (toda ação
tem handler). Para cada ação ele incrementa `step_count` e o contador do orçamento correspondente
(tentativas contam, mesmo com falha), executa dentro de `trace_stage` e registra um `ActionRecord`. As
degradações são as mesmas do harness linear: LLM falha → erro registrado; ML falha → fallback
`URGENT_CARE` no `CareRoutingAgent`; busca de unidades falha → decisão mantida, `facility: null`.

### HarnessState

Dados: `request_id`, `user_id`, `user_message` (só em memória), `intent`, `safety_precheck`,
`extracted_symptoms`, `patient_context`, `safety_assessment`, `ml_prediction`, `routing_decision`,
`facilities`, `errors`.

Controle e auditoria: `completed_actions`, `action_history` (`step`, `action`, `reason_code`, `status`,
`duration_ms`), `step_count`, `llm_call_count`, `tool_call_count`, `finished`, `finish_reason`
(`completed`, `llm_unavailable`, `policy_violation`, `step_limit`, `llm_call_limit`, `tool_call_limit`,
`timeout`).

### Termination guards (`HarnessLimits`)

| Guarda | Padrão | Motivo |
|---|---|---|
| `max_steps` | 10 | o caminho legítimo mais longo tem 8 passos |
| `max_llm_calls` | 1 | uma extração por requisição |
| `max_tool_calls` | 3 | contexto + ML + unidades |
| `timeout_seconds` | 90 | acima do timeout × retries do cliente LLM, que deve disparar antes |

Além disso, "cada ação no máximo uma vez" torna impossível um *loop* infinito mesmo sem os limites.

### Auditoria

Cada decisão gera o log `planner_decision` com `request_id`, `current_action`, `next_action`,
`reason_code` e `step_count`. O fim gera `harness_finished` (ou `harness_aborted`) com `finish_reason`,
contadores e a lista de ações. **Nunca** com o relato, o prompt ou o prontuário (há teste para isso).

### Pipeline linear × harness autônomo

| | `HealthFlowHarness` (linear, removido) | `AutonomousHealthFlowHarness` |
|---|---|---|
| Ordem | fixa no código de `run()` | escolhida a cada passo pelo *planner* a partir do estado |
| *Red flag* no texto | o LLM era chamado mesmo assim | decidido no precheck, **sem esperar o LLM** |
| Garantias de ordem | implícitas na sequência | explícitas e testadas na *policy*, válidas para qualquer *planner* |
| Limites | nenhum | passos, LLM, tools e timeout global |
| Auditoria | log por etapa | log por decisão (com `reason_code`) + `action_history` no estado |
| Extensão | editar `run()` | novo *planner* ou novas ações registradas, sem mudar o loop |

O contrato de `POST /api/v1/routing` não mudou. A única diferença observável é que, quando o texto bruto
já dispara uma *red flag*, a resposta de emergência sai sem chamar o LLM.

### Extensão futura: intents

`Intent` já tem `CARE_ROUTING`, `MEDICATION` e `INSURANCE`, mas só `CARE_ROUTING` tem ações permitidas
(`ALLOWED_ACTIONS`). Para as outras, a *policy* recusa qualquer ação. Na Fase 2, o domínio de
`INSURANCE` será ampliado para exames/procedimentos e cobertura SUS/plano. Na Fase 3,
`MEDICATION` será usado para localização de medicamentos. Novas ações devem entrar no mesmo loop,
sempre validadas pela *policy*.

### Jev fora da execução atual

A API instancia apenas o `DeterministicPlanner` em `app/api/dependencies/container.py`.
O código do `JevPlanner` permanece no repositório para estudo, mas as variáveis antigas
`HEALTHFLOW_HARNESS_PLANNER` e `HEALTHFLOW_JEV_ENABLED` não ativam chamadas externas.
A integração não participa do fluxo de atendimento, da seleção do classificador nem
dos resultados experimentais da Fase 1. Uma eventual retomada exigirá avaliação própria.

## Invariantes de segurança

1. **Nível final ≥ piso do Safety Engine** e **≥ predição do ML**. Implementado em
   `CareRoutingAgent.decide` e testado para todas as combinações (`tests/safety/test_care_routing.py`).
   `Safety=EMERGENCY` + `ML=PRIMARY_CARE` → `EMERGENCY`.
2. Com *red flag*, o ML nem é executado; com *red flag* no texto bruto (precheck), nem o LLM. A *policy*
   impõe isso a qualquer *planner*.
3. As regras também procuram padrões no **texto bruto** (sem acento, minúsculo). Se o LLM falhar ou
   não extrair o sinal crítico, a regra ainda dispara. Negações não são interpretadas — de propósito,
   o erro fica do lado seguro.
4. **LLM indisponível**: se o texto bruto disparar *red flag*, o precheck já decide (o LLM nem é
   chamado); caso contrário → HTTP 503 com mensagem segura e lembrete do SAMU 192.
5. **ML indisponível** → fallback conservador `URGENT_CARE`.
6. **Nenhuma unidade encontrada** → resposta mantém o nível e a orientação, com `facility: null`.
7. A resposta não tem campo de diagnóstico; o sistema **não aciona** o SAMU — apenas orienta ligar 192.
8. Nenhum *planner*, LLM, ML ou agente reduz um piso: o piso efetivo só sobe ao longo da execução.
9. Orçamentos e timeout global impedem *loops*; violações abortam com erro controlado.

## Safety Engine

- Regras declarativas (`SafetyRule`) com `id`, `minimum_care_level`, condições de sintomas,
  intensidade mínima, faixas etárias e padrões de texto (`app/safety/rules.py`).
- Todas marcadas `source = "ACADEMIC_SIMULATED — not an official SUS protocol"`.
- Formato declarativo para permitir, no futuro, carregar regras validadas de fonte oficial sem mudar
  o motor.

| ID | Gatilho (simulado) | Piso |
|---|---|---|
| RED_FLAG_001 | dor no peito **e** falta de ar | EMERGENCY |
| RED_FLAG_002 | perda de consciência / "desmai" | EMERGENCY |
| RED_FLAG_003 | convulsão | EMERGENCY |
| RED_FLAG_004 | face/fala/fraqueza unilateral | EMERGENCY |
| RED_FLAG_005 | sangramento intenso | EMERGENCY |
| RED_FLAG_006 | falta de ar intensa / "não consigo respirar" | EMERGENCY |
| RED_FLAG_007 | ideação autolesiva | EMERGENCY |
| CAUTION_001 | dor no peito isolada | URGENT_CARE |
| CAUTION_002 | febre intensa | URGENT_CARE |
| CAUTION_003 | febre em lactente ou idoso | URGENT_CARE |

## LLM

- `LLMProvider` (Protocol) com um método: `generate_structured(messages, output_model) -> T`.
- `OllamaLLMProvider` usa o **OpenAI Python SDK** apontando para `HEALTHFLOW_LLM_BASE_URL`
  (padrão `http://localhost:11434/v1`). Trocar para outro provedor compatível = mudar variáveis;
  provedor incompatível = nova classe que implementa o Protocol. Agentes não mudam.
- *Structured Outputs* via `response_format: json_schema` (suportado pelo Ollama 0.34), `temperature=0`,
  raciocínio desligado (`reasoning_effort="none"`), *timeout*, *retry* de transporte pelo SDK e *retry*
  limitado para JSON inválido. Erros viram `LLMUnavailableError` / `LLMResponseError`.
- O LLM **só recebe o relato**. Prontuário e contexto nunca são enviados a ele.

## Prontuário inicial e Context Builder

O prontuário já faz parte da **Fase 1**, mas de forma deliberadamente simples.

- `PatientRepository` (Protocol). Hoje: `InMemoryPatientRepository` com **um paciente fictício**.
- `PatientRecord` representa um perfil inicial simplificado com idade, alergias, condições,
  medicamentos ativos e encontros anteriores.
- Na decisão atual, o `PatientContextBuilder` usa apenas o subconjunto necessário: faixa etária,
  fatores de risco categóricos, condições relevantes, alergias quando relacionadas ao relato e
  anticoagulantes quando relacionados a sangramento, trauma ou sinais neurológicos.
- Nome e resumos livres de encontros anteriores são removidos do contexto usado na decisão.
- O histórico longitudinal completo ainda **não** participa do roteamento.
- Não existe integração com prontuário nacional, hospitalar ou sistemas reais do SUS.
- A persistência em PostgreSQL é uma evolução posterior; a Fase 1 primeiro valida a arquitetura com
  contexto sintético e mínimo.

A intenção é começar com:

```text
relato atual
  +
perfil clínico inicial mínimo
  ↓
Context Builder
  ↓
Safety / ML / Routing
```

e somente depois ampliar para contexto longitudinal persistido.

## Geolocalização

`care_level → service_type → unidades compatíveis → distância (haversine) → raio → mais próxima`.
Nunca "o hospital mais próximo". `FacilityProvider` (Protocol) com `MockFacilityProvider` e cinco
unidades **fictícias** em São Paulo, todas marcadas `is_simulated: true`. Futuro: `CNESFacilityProvider`.

## Autenticação (provisória)

`DemoTokenAuthenticator`: um *bearer token* estático (`HEALTHFLOW_DEMO_AUTH_TOKEN`) mapeado para um
usuário demo. Desligado se vazio; a configuração **recusa iniciar** com ele em `production`. CPF nunca é
usado como credencial. Substituição por autenticação real é parte da Fase F.

## Privacidade e observabilidade

- Logs JSON com `request_id`, etapa, duração, status, IDs de regras e nível final.
  **Nunca** o texto do relato, o prompt ou o prontuário.
- `X-Request-ID` aceito apenas se for UUID (evita injeção em logs); sempre devolvido na resposta.
- Exceções mapeadas para mensagens públicas fixas; *stack trace* só no log do servidor.
- Segredos via `.env` (ignorado pelo git); `.env.example` documenta as variáveis.

## Estrutura

```text
app/
├── main.py                    # create_app + lifespan (carrega o modelo, não treina)
├── api/                       # rotas, middleware de request_id, mapeamento de erros, DI
│   └── dependencies/container.py   # composition root: única escolha de implementações
├── core/                      # config, exceções, logging estruturado
├── auth/                      # Principal, DemoTokenAuthenticator
├── agents/                    # intent, patient_context, care_routing, navigation
├── harness/                   # AutonomousHealthFlowHarness, planner, policy, executor,
│                              # actions, HarnessState, montagem da resposta
├── llm/                       # LLMProvider, OllamaLLMProvider
├── ml/                        # data/ (fontes → esquema canônico), feature builders, splits,
│                              # métricas, experimentos, classifier, inference, training/
├── safety/                    # SafetyRule, regras acadêmicas, SafetyEngine
├── context/                   # PatientContextBuilder
├── repositories/              # PatientRepository + implementação em memória
├── tools/                     # FacilityProvider, MockFacilityProvider, haversine
└── schemas/                   # CareLevel, ServiceType, Symptom, Patient, Routing, Facility
tests/{unit,safety,ml,integration}
```

Diferenças deliberadas em relação ao rascunho anterior:

- Não existe `safety_agent.py`: o Safety Engine é determinístico e chamado diretamente pelo executor —
  chamá-lo de "agente" esconderia que ele não depende de modelo algum.
- `rag/`, `services/` e `repositories/routing.py` só serão criados nas fases que os usam (YAGNI).
- `FacilityProvider.find_nearby` recebe `service_type` (não `care_level`): o mapeamento
  nível → tipo de serviço é decisão do roteamento, não do provedor de localização.
- Destinos: três classes (`PRIMARY_CARE`, `URGENT_CARE`, `EMERGENCY`). "Hospital" e "SAMU" ficam
  dentro de `EMERGENCY` (pronto-socorro + orientação de ligar 192).

## Persistência (Fase F — planejado)

PostgreSQL + SQLAlchemy 2 + Alembic:

```text
users, patient_profiles, consents, allergies, conditions, medications, encounters,
routing_requests, routing_results, audit_logs
```

`routing_results` guardará nível, códigos de razão, versão do modelo e regras acionadas — **não** o
texto do relato por padrão.

## RAG (Fase G — planejado)

Separado do prontuário. O prontuário **não** vira embedding por padrão.

```text
documents(id, source, title, version, published_at, url, document_type)
document_chunks(id, document_id, content, embedding vector, metadata)
```

Consulta → embedding → pgvector → *chunks* com metadados → agente → resposta **com fonte**.

## Fase 2 — Exames, procedimentos e cobertura (planejado)

A Fase 2 amplia a navegação para responder, com fontes verificáveis:

```text
Exame / procedimento
        ↓
   ┌────┴────┐
   ▼         ▼
  SUS      Plano
   │         │
   ▼         ▼
oferta /   cobertura do
acesso     produto contratado
   └────┬────┘
        ▼
opções e locais compatíveis
```

A cobertura deve ser verificada pelo produto/plano específico, não apenas pelo nome da operadora.
Quando necessário, essa fase poderá usar RAG sobre fontes oficiais. Sem fonte verificável, o sistema
não deve afirmar disponibilidade ou cobertura.

## Fase 3 — Medicamentos (planejado)

```text
Medicamento informado ou já prescrito
        ↓
MedicationAgent
        ↓
fontes verificáveis de disponibilidade
        ↓
SUS / pontos de dispensação / rede privada
        ↓
localização
        ↓
opções próximas
```

A Fase 3 trata de localizar disponibilidade. Ela não decide qual medicamento o usuário deve tomar e
não substitui prescrição profissional.


## Planner Jev opcional

O `AutonomousHealthFlowHarness` suporta dois planners:

- `DeterministicPlanner`: padrão local e fallback obrigatório;
- `JevPlanner`: integração experimental com `POST /v1/systemone` do Jev.

O Jev recebe somente estado abstrato do workflow. Antes da chamada remota, o backend remove qualquer necessidade de enviar mensagem clínica, identificador do paciente, localização ou prontuário. O payload contém apenas sinais como etapas concluídas, existência de red flag e ações atualmente permitidas.

O fluxo é:

```text
HarnessState
  ↓
HarnessPolicy.allowed_actions()
  ↓
JevPlanner
  ↓
proposta + probabilidade
  ↓
HarnessPolicy.validate()
  ├─ aceita → ActionExecutor
  └─ rejeita / baixa probabilidade / erro → DeterministicPlanner
```

O safety pre-check e qualquer caminho com red flag não dependem do Jev. Assim, indisponibilidade ou latência do serviço remoto não bloqueia o caminho crítico. A autoridade permanece `Safety Engine > Policy > Planner > Agents/Models/Tools`.
