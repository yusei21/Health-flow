# Health-flow

Health-flow é um sistema de navegação em saúde que recebe o relato do usuário, considera um contexto clínico inicial simplificado do paciente, interpreta os sintomas com IA local, combina regras de segurança com Machine Learning e indica o tipo de atendimento mais adequado. A interface também pode usar a localização do próprio usuário, com permissão do navegador, para ajudar a encontrar uma unidade compatível próxima.

O sistema não faz diagnóstico, não prescreve medicamentos e não aciona ambulâncias. Em uma possível emergência, a orientação pode incluir contato com o SAMU pelo 192.

## Como o sistema funciona

```text
Usuário descreve o que está sentindo
        ↓
Safety pre-check
        ↓
LLM local (Qwen3 via Ollama)
        ↓
Prontuário inicial simplificado
        ↓
Context Builder
        ↓
Safety Engine + Machine Learning
        ↓
Care Routing
        ↓
Geolocalização do navegador (com permissão)
        ↓
Busca de unidade compatível
        ↓
Resposta para o usuário
```

O Agent Harness escolhe dinamicamente a próxima ação dentro de um conjunto fechado de ações permitidas. As regras de segurança têm prioridade sobre o LLM e sobre o modelo de Machine Learning.

## Fases do projeto

O desenvolvimento será incremental. A ideia é começar simples e ampliar somente depois que a Fase 1 estiver estável.

### Fase 1 — encaminhamento, contexto inicial e localização

É a fase atual. O sistema considera:

- relato atual do usuário;
- um prontuário inicial simplificado;
- idade/faixa etária;
- fatores de risco categóricos;
- alergias somente quando forem relevantes para o relato;
- medicamentos anticoagulantes somente quando forem relevantes;
- regras de segurança;
- classificação experimental;
- localização para procurar uma unidade compatível.

O protótipo atual usa um paciente fictício em memória. Não existe integração com um prontuário nacional, hospitalar ou do SUS. O histórico longitudinal completo também não participa da decisão nesta etapa. O objetivo inicial é validar o fluxo usando apenas o contexto mínimo necessário.

Fluxo conceitual:

```text
Relato atual
   ↓
Safety pre-check
   ↓
Extração estruturada
   ↓
Prontuário inicial simplificado
   ↓
Context Builder
   ↓
Safety + ML
   ↓
Encaminhamento
   ↓
GPS
   ↓
Unidade compatível
```

### Fase 2 — exames, procedimentos e cobertura

Depois da Fase 1, o sistema deverá consultar fontes verificáveis para responder perguntas como:

- o SUS disponibiliza determinado exame ou procedimento?
- onde esse serviço pode ser realizado?
- se o usuário tiver plano de saúde, o produto contratado cobre esse exame ou procedimento?

Essa fase deverá trabalhar com fontes oficiais e não depender apenas da resposta do LLM.

### Fase 3 — medicamentos

A terceira fase será voltada à localização de medicamentos informados ou já prescritos pelo usuário:

- disponibilidade no SUS;
- pontos de dispensação;
- disponibilidade na rede privada quando houver fonte apropriada;
- ordenação por localização.

A Fase 3 não tem como objetivo prescrever medicamentos.

## Principais componentes

- **Frontend:** React + TypeScript + Vite.
- **API:** FastAPI.
- **LLM local:** Qwen3 4B executado pelo Ollama.
- **Machine Learning:** scikit-learn.
- **Rede neural:** MLPClassifier como modelo experimental.
- **Agent Harness:** planner + policy + executor + estado.
- **Safety Engine:** regras determinísticas de segurança.
- **Geolocalização:** Browser Geolocation API.
- **Banco preparado:** PostgreSQL + pgvector.
- **Benchmarks:** experimentos reproduzíveis para modelos de ML.

## Requisitos

Instale antes:

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)
- Node.js + npm
- Git

## 1. Clonar o projeto

```bash
git clone https://github.com/yusei21/Health-flow.git
cd Health-flow
```

## 2. Preparar o Ollama

Baixe o modelo usado pelo projeto:

```bash
ollama pull qwen3:4b
```

Confirme que o Ollama está funcionando:

```bash
ollama list
curl http://127.0.0.1:11434/api/tags
```

O backend usa por padrão:

```text
http://localhost:11434/v1
```

## 3. Configurar o backend

Crie o arquivo de ambiente:

```bash
cp .env.example .env
```

Edite o `.env` e defina um token de demonstração, por exemplo:

```bash
HEALTHFLOW_DEMO_AUTH_TOKEN=local-dev-token-1234567890
```

Para permitir o frontend local:

```bash
HEALTHFLOW_CORS_ALLOWED_ORIGINS=["http://localhost:5173","http://127.0.0.1:5173"]
```

Instale as dependências:

```bash
make install
```

## 4. Treinar o modelo usado pela API

O modelo de Machine Learning não é treinado quando a API inicia. Gere o dataset sintético e treine antes:

```bash
make train
```

O modelo utilizado pela API é salvo em:

```text
models/synthetic-v1/
```

O treinamento compara modelos como:

- Logistic Regression
- Decision Tree
- Random Forest
- MLP Neural Network

Os resultados dos experimentos ficam em:

```text
benchmarks/results/
```

## 5. Rodar o backend

Em um terminal:

```bash
make dev
```

A API ficará disponível em:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

## 6. Rodar o frontend

Abra outro terminal:

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

No `frontend/.env`, use o mesmo token configurado no backend:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_DEMO_TOKEN=local-dev-token-1234567890
```

Abra:

```text
http://localhost:5173
```

## Uso básico

1. Abra o frontend.
2. Escreva o que está sentindo.
3. Clique em **Usar minha localização**.
4. Autorize o navegador a acessar a localização.
5. Clique em **Buscar atendimento**.
6. O frontend envia relato + latitude + longitude para o backend.
7. O Agent Harness executa o fluxo necessário.
8. O resultado mostra o nível de atendimento, tipo de serviço e unidade encontrada quando disponível.

## Geolocalização

A localização é obtida pelo navegador usando:

```javascript
navigator.geolocation.getCurrentPosition(...)
```

O usuário precisa autorizar o acesso. O projeto não usa rastreamento contínuo e não utiliza `watchPosition`.

A localização funciona em contexto seguro:

- `localhost` durante desenvolvimento;
- HTTPS em produção.

As coordenadas são utilizadas para a busca de unidades e não devem ser armazenadas em `localStorage` nem enviadas para serviços de analytics.

## Testar pelo terminal

Exemplo de chamada direta para a API:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/routing \
  -H 'Authorization: Bearer local-dev-token-1234567890' \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "estou com tosse e coriza há 4 dias",
    "latitude": -23.55,
    "longitude": -46.64
  }'
```

## Comandos úteis

```bash
make dev                  # inicia a API
make test                 # testes Python
make test-ollama          # teste real com Ollama
make lint                 # lint
make typecheck            # mypy
make train                # treina o modelo usado pela API
make evaluate             # avalia o modelo

make dataset-synthetic
make train-synthetic
make evaluate-synthetic

make dataset-mimic
make train-mimic
make evaluate-mimic

make dataset-triagegeist
make train-triagegeist
make evaluate-triagegeist

make benchmark-ml
make benchmark-summary
```

Frontend:

```bash
cd frontend
npm install
npm run dev
npm test
npm run build
```

## Estrutura principal

```text
Health-flow/
├── app/
│   ├── agents/
│   ├── api/
│   ├── harness/
│   ├── llm/
│   ├── ml/
│   ├── safety/
│   └── tools/
├── frontend/
├── tests/
├── data/
├── models/
├── benchmarks/
├── docs/
├── Makefile
└── docker-compose.yml
```

## Agent Harness

O Harness atual é autônomo e orientado a estado. Ele é composto por:

```text
AutonomousHealthFlowHarness
├── Planner
├── Policy
├── Executor
├── HarnessState
└── termination guards
```

O planner escolhe o próximo passo permitido de acordo com o estado atual. A policy impede ações inválidas, e o Safety Engine mantém prioridade sobre qualquer decisão de ML ou LLM.

## Jev Planner experimental

O Harness usa o `DeterministicPlanner` por padrão. Opcionalmente, o backend pode usar o Jev como planner remoto para escolher a próxima ação entre as ações que a `HarnessPolicy` permite naquele estado.

A hierarquia continua:

```text
Safety Engine > Policy > Planner > Agents / Models / Tools
```

O Jev não recebe relato clínico bruto, prontuário, identificador do paciente nem coordenadas. Ele recebe somente um estado abstrato do workflow, como ações já concluídas, presença de red flag e ações permitidas. Se houver timeout, erro HTTP, resposta inválida, ação proibida ou probabilidade abaixo do limite, o sistema volta automaticamente para o `DeterministicPlanner`.

Para habilitar:

```bash
HEALTHFLOW_HARNESS_PLANNER=jev
HEALTHFLOW_JEV_ENABLED=true
HEALTHFLOW_JEV_API_KEY=sua-chave
HEALTHFLOW_JEV_MODEL=jev-latest
HEALTHFLOW_JEV_MIN_PROBABILITY=0.70
```

Nunca coloque a chave do Jev no frontend ou no Git. Sem essa configuração, o projeto continua funcionando normalmente com o planner determinístico.

## Comparação de LLMs (benchmark de extração)

O **Qwen não é fixo no Harness**. A interface `LLMProvider` separa o agente de intenção do cliente de inferência. Por padrão, a API usa Ollama com `qwen3:4b`, mas o modelo pode ser alterado por `HEALTHFLOW_LLM_MODEL`, e o endpoint compatível por `HEALTHFLOW_LLM_BASE_URL`.

Para comparar dois modelos locais no **mesmo conjunto de relatos fictícios**:

```bash
ollama pull qwen3:4b
ollama pull llama3.2:3b

make benchmark-llm LLM_MODEL=qwen3:4b
make benchmark-llm LLM_MODEL=llama3.2:3b
```

Casos rotulados: `benchmarks/llm/cases.jsonl`. Implementação: `benchmarks/scripts/benchmark_llm.py`. Cada execução escreve um JSON novo, sem sobrescrever os anteriores, em `benchmarks/results/llm/`. O JSON registra modelo, endpoint, commit Git, data, casos, falhas por tipo, taxa de saídas estruturadas válidas, exact match, precisão/recall/F1 micro de sintomas e latência p50/p95 das chamadas bem-sucedidas. O tempo total inclui falhas.

**Interpretação:** são apenas casos fictícios escritos manualmente, não benchmarks clínicos, nem uma avaliação de encaminhamento. Resultados servem para verificar o pipeline experimental. Para o artigo, amplie e congele um conjunto independente de referência antes de executar a comparação definitiva. Use o mesmo hardware, versão do Ollama, parâmetros, prompt, ordem de casos e condições de aquecimento para todos os modelos. É recomendável executar múltiplas repetições e relatar a variância. O modelo precisa oferecer geração estruturada compatível com o endpoint configurado.

Para trocar apenas o modelo usado pela API, sem benchmark:

```bash
HEALTHFLOW_LLM_MODEL=llama3.2:3b
```

Reinicie a API após mudar o `.env`. Não use relatos reais de pacientes nos benchmarks.

## Machine Learning e datasets

O projeto possui suporte para:

- dataset sintético reproduzível;
- MIMIC-IV-ED;
- Triagegeist;
- Logistic Regression;
- Decision Tree;
- Random Forest;
- MLP Neural Network.

Os datasets reais não são incluídos no repositório.

Mais detalhes:

- [Arquitetura](docs/architecture.md)
- [Machine Learning](docs/machine-learning.md)
- [Datasets](docs/datasets.md)

## Docker

Para subir a API e PostgreSQL/pgvector:

```bash
docker compose up --build
```

O Ollama continua rodando no host.

Antes de iniciar via Docker, rode:

```bash
make train
```

para gerar o modelo usado pela API.

## Observações

- O LLM recebe somente o relato necessário para interpretar os sintomas.
- O prontuário inicial simplificado não é enviado ao LLM; o Context Builder reduz os dados antes do uso por regras e ML.
- O histórico longitudinal completo ainda não participa da decisão.
- O modelo de ML auxilia o encaminhamento, mas não tem autoridade sobre as regras críticas de segurança.
- O sistema orienta sobre atendimento; não substitui avaliação profissional.
- Em uma possível emergência, siga a orientação apresentada e utilize o SAMU 192 quando indicado.
