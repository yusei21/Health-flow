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

O Agent Harness executa o `DeterministicPlanner`, que escolhe a próxima ação a partir do estado. A `HarnessPolicy` valida cada escolha antes da execução. O Jev está fora do fluxo da API nesta fase; seu código foi preservado para experimentos posteriores.

## Situação experimental (outubro de 2026)

A comparação preliminar `synthetic_baseline` foi concluída com quatro modelos e 4.000
exemplos gerados por regras. Na execução `eb0307e7` (commit `f0ac6eb`, árvore limpa),
o Random Forest foi selecionado pela validação cruzada. No teste reservado de 800
exemplos, obteve F1 macro de 0,9189 e recall da classe `EMERGENCY` de 0,8913.
A Regressão Logística obteve 0,8788 e 0,8406; a Árvore de Decisão, 0,8478 e
0,8043; a MLP, 0,8861 e 0,7899, respectivamente.

Esses dados medem a correspondência aos rótulos do gerador sintético. Não são
validação clínica, nem avaliação do encaminhamento final após o Safety Engine.
Os experimentos externos com MIMIC-IV-ED e Triagegeist ainda não têm resultados.
Os testes automatizados e a integração local com Ollama não substituem esses
experimentos. Tabelas e condições de avaliação: [docs/machine-learning.md](docs/machine-learning.md).

Protocolo de auditoria, critérios mínimos de suporte por classe e limites de inferência: [docs/benchmark-protocol.md](docs/benchmark-protocol.md). Para conferir uma execução já existente, use `make benchmark-audit AUDIT_RUN=781fd757` (MIMIC Demo) ou `AUDIT_RUN=eb0307e7` (sintético).

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

## Planner da Fase 1

A composição da API usa exclusivamente o `DeterministicPlanner`. Cada ação proposta
passa pela `HarnessPolicy`, que controla pré-condições, limites e a precedência das
regras de segurança. O código experimental do Jev permanece em `app/harness/jev.py`,
sem integração ativa na aplicação. Variáveis antigas do Jev no `.env` não o habilitam.

## Modelo local de linguagem

O Qwen3 4B é o padrão, mas o agente de extração usa a interface `LLMProvider`.
É possível trocar `HEALTHFLOW_LLM_MODEL` por outro modelo com suporte à API e à
saída estruturada exigidas pelo projeto. Reinicie a aplicação após a mudança.

## Comparação de modelos LLM — preparada, sem resultados inventados

O **Qwen3 4B é somente o modelo padrão**, não uma dependência fixa do Harness. A implementação usa `LLMProvider` e um cliente OpenAI-compatible para o Ollama. Troque `HEALTHFLOW_LLM_MODEL` no `.env` para executar a aplicação com outro modelo que suporte saída estruturada no endpoint configurado. O protocolo também permite implementar outros provedores pela mesma interface.

Para um **benchmark publicável**, não são aceitos os 12 relatos fictícios incluídos como exemplos de desenvolvimento em `benchmarks/llm/cases.jsonl`. Você precisa fornecer um conjunto rotulado obtido legitimamente, com termos de uso conhecidos e anotações verificadas. O comando se recusa a executar sem o arquivo de proveniência. Nenhuma métrica real foi produzida por essa mudança.

Formato do arquivo de casos `/caminho/seguro/cases.jsonl`, um JSON por linha:

```json
{"case_id":"identificador-opaco","report":"texto da fonte autorizada","symptoms":["cough"],"severity":"unknown","duration_minutes":null,"age":null}
```

Os valores são **apenas a descrição do esquema**, não um caso para benchmark. Os sintomas devem pertencer ao enum `Symptom`, e os rótulos devem ser anotados/revisados independentemente das previsões dos modelos testados.

Formato do arquivo `/caminho/seguro/provenance.json`:

```json
{
  "dataset_name": "NOME_DA_FONTE",
  "source_url": "URL_DA_FONTE",
  "source_version": "VERSAO_EXATA",
  "license_or_access_terms": "LICENCA_OU_TERMO_DE_USO",
  "annotation_method": "COMO_OS_ROTULOS_FORAM_OBTIDOS_E_REVISADOS",
  "language": "pt-BR",
  "clinical_validation": false
}
```

Com o Ollama em execução e os modelos já baixados, compare um por vez **no mesmo arquivo**:

```bash
make benchmark-llm LLM_MODEL=qwen3:4b \
  LLM_CASES=/caminho/seguro/cases.jsonl \
  LLM_PROVENANCE=/caminho/seguro/provenance.json

make benchmark-llm LLM_MODEL=llama3.2:3b \
  LLM_CASES=/caminho/seguro/cases.jsonl \
  LLM_PROVENANCE=/caminho/seguro/provenance.json
```

O script gera JSON imutável em `benchmarks/results/llm/`, incluindo `dataset_sha256`, proveniência, modelo, commit, precisão, recall, F1, correspondência exata, validade de esquema, falhas e latência p50/p95. Só compare execuções com o **mesmo hash dos casos**, protocolo de anotação e ambiente documentado. Essa comparação avalia **extração de sintomas**, não qualidade de encaminhamento. É necessário validar também a adequação do prompt ao idioma da fonte (o prompt atual é em português).

Em particular, MIMIC-IV-ED exige acesso autorizado e restrições próprias: não envie relatos restritos a provedores externos e não publique o texto dos pacientes ou identificadores. Dados reais não são incluídos no repositório. O experimento C de MIMIC com relato em texto ainda requer pipeline próprio e validação de rótulos de extração; os rótulos de ESI não são rótulos de sintomas.

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
