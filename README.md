# Health-flow

> **PROTÓTIPO ACADÊMICO — NÃO UTILIZAR PARA DECISÕES CLÍNICAS REAIS.**

Projeto acadêmico de Inteligência Artificial para navegação assistencial no SUS e, em etapas posteriores, consulta de medicamentos e cobertura de planos de saúde.

O Health-flow **não diagnostica, não prescreve e não substitui profissionais de saúde**. Ele recebe o relato do usuário, estrutura as informações, aplica regras de segurança e usa **Agent Harness + Machine Learning** para indicar o **tipo de serviço** adequado e a unidade compatível mais próxima.

Em possível emergência, orienta ligar para o **SAMU 192**. O sistema **não aciona ambulância** — não existe integração com a regulação.

## Como funciona (Etapa 1)

```text
Login → prontuário autorizado → relato em texto livre
  → LLM local (Ollama) extrai sintomas estruturados
  → Context Builder seleciona só o contexto clínico relevante
  → Safety Engine aplica regras determinísticas (pode impor piso de atendimento)
  → Machine Learning sugere o nível (pulado se houver red flag)
  → Care Routing combina: nunca abaixo do piso de segurança
  → tipo de serviço (UBS | UPA | pronto-socorro + SAMU 192)
  → unidade compatível mais próxima (dados simulados)
  → resposta com próximo passo e aviso
```

| Peça | Papel |
|---|---|
| **LLM** (`qwen3:4b` via Ollama) | entende linguagem natural → JSON validado |
| **Machine Learning** (scikit-learn) | classificação probabilística auxiliar |
| **Safety Engine** | regras críticas com ID; sempre prevalece |
| **Agent Harness** | coordena etapas, falhas e quem tem autoridade |
| **Tools** | busca de unidades (provider simulado) |

Detalhes: [docs/architecture.md](docs/architecture.md) · ML (dataset, modelos, métricas, limitações): [docs/machine-learning.md](docs/machine-learning.md)

## Executando

Pré-requisitos: Python 3.12, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com) com o modelo baixado.

```bash
ollama pull qwen3:4b          # uma vez; não é baixado automaticamente
cp .env.example .env          # ajuste HEALTHFLOW_DEMO_AUTH_TOKEN
make install                  # uv sync
make train                    # gera dataset sintético + treina e salva models/
make dev                      # API em http://localhost:8000 (docs em /docs)
```

Exemplo:

```bash
curl -s -X POST localhost:8000/api/v1/routing \
  -H "Authorization: Bearer $HEALTHFLOW_DEMO_AUTH_TOKEN" -H 'Content-Type: application/json' \
  -d '{"message":"estou com dor forte no peito e falta de ar há 20 minutos","latitude":-23.55,"longitude":-46.64}'
```

```json
{
  "request_id": "39642a51-…",
  "care_level": "EMERGENCY",
  "recommended_service_type": "EMERGENCY_ROOM",
  "facility": {"name": "Pronto-Socorro Exemplo (SIMULADA)", "distance_km": 3.06, "is_simulated": true, "...": "..."},
  "next_step": "Ligue 192 (SAMU) ou dirija-se imediatamente a um pronto-socorro.",
  "emergency_guidance": "Possível situação de emergência. Ligue imediatamente para o SAMU 192. Este sistema NÃO aciona ambulância automaticamente.",
  "reason_codes": ["SAFETY_RULE:RED_FLAG_001", "SAFETY_RULE:RED_FLAG_006", "SAFETY_RULE:CAUTION_001", "SAFETY_OVERRIDE"],
  "safety_override": true,
  "disclaimer": "PROTÓTIPO ACADÊMICO — NÃO UTILIZAR PARA DECISÕES CLÍNICAS REAIS. …"
}
```

### Rotas

| Método | Rota | Auth |
|---|---|---|
| GET | `/health` | não |
| POST | `/api/v1/routing` | Bearer |
| GET | `/api/v1/patients/me` | Bearer |

A autenticação atual é um **token de demonstração** ligado a um paciente **fictício**; é recusada quando `HEALTHFLOW_APP_ENV=production`.

### Comandos

```bash
make test         # pytest (não precisa do Ollama)
make test-ollama  # teste de integração real com o Ollama local
make lint         # ruff check + format --check
make format
make typecheck    # mypy --strict
make train        # dataset + treino + metadados
make evaluate     # reavalia o modelo salvo
make up / down    # docker compose (api + postgres/pgvector)
```

### Docker

`docker compose up --build` sobe a API e o PostgreSQL com pgvector habilitado (o banco ainda não é usado pela API — Fase F). O Ollama continua no host e é acessado em `http://host.docker.internal:11434/v1`; `models/` é montado como volume somente leitura (rode `make train` antes).

Se a API no contêiner responder 503 para o relato, o contêiner provavelmente não alcança o Ollama do host: verifique se o Ollama escuta em `0.0.0.0` (`OLLAMA_HOST=0.0.0.0`) e se o firewall do host (ex.: `ufw`) permite a rede do Docker na porta 11434.

## Etapas do projeto

1. **Etapa 1 — Encaminhamento inteligente** (em andamento; fases A–E implementadas): Agent Harness + ML + regras + geolocalização.
2. **Etapa 2 — Medicamentos no SUS** (planejado): `MedicationAgent` + RAG sobre fontes oficiais + localização. Respostas sempre com fonte; o LLM não pode inventar disponibilidade.
3. **Etapa 3 — Planos de saúde** (planejado): `InsuranceAgent` + `ProviderAgent` com dados verificáveis de cobertura e rede do **produto** contratado.

## Dados e privacidade

- O prontuário completo **nunca** é enviado ao LLM; o LLM recebe apenas o relato.
- O Context Builder aplica minimização de dados antes do ML e das regras.
- Logs registram etapa, duração, status e `request_id` — nunca o relato, prompts ou o prontuário.
- Todos os dados de pacientes, unidades e treino são **sintéticos/fictícios**.
- Futuro: PostgreSQL para dados estruturados; pgvector apenas para documentos (protocolos, regras, manuais), não para o prontuário.

## Stack

Python 3.12 · FastAPI · Pydantic v2 · pydantic-settings · OpenAI SDK (contra Ollama) · scikit-learn · joblib · pytest · ruff · mypy · Docker Compose · PostgreSQL + pgvector (provisionado). Planejado: SQLAlchemy 2, Alembic, Redis.

## Princípios

- Não diagnosticar, não prescrever, não alterar tratamento.
- O LLM não decide; o ML não é autoridade em casos críticos; regras de segurança sempre prevalecem.
- Emergências têm prioridade; nenhuma integração é simulada como se fosse real.
- Resultado auditável por `reason_codes` e `request_id`.
