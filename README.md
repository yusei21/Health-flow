# Health-flow

Projeto acadêmico de Inteligência Artificial para navegação assistencial no SUS e, em etapas posteriores, consulta de medicamentos e cobertura de planos de saúde.

O Health-flow **não realiza diagnóstico**. O sistema recebe o relato do usuário, organiza informações relevantes, aplica regras de segurança e usa **Agent Harness + Machine Learning** para auxiliar o encaminhamento para o tipo de atendimento adequado.

## Objetivo

A proposta é funcionar como um GPS da saúde:

```text
Usuário faz login
      ↓
Prontuário/histórico autorizado fica vinculado
      ↓
Usuário escreve o que está sentindo
      ↓
Agent Harness coordena o fluxo
      ↓
LLM estrutura o relato
      ↓
Machine Learning auxilia a classificação do encaminhamento
      ↓
Regras de segurança e regras do SUS validam a rota
      ↓
Sistema encontra o serviço adequado mais próximo
      ↓
Usuário recebe endereço, rota e orientação
```

Em caso de possível emergência, o sistema deve priorizar o fluxo de emergência e orientar o acionamento do SAMU 192. O projeto não deve prometer envio automático de ambulância sem integração oficial com a regulação competente.

---

# Etapas do projeto

## Etapa 1 — Encaminhamento inteligente

Esta é a primeira e principal etapa do projeto acadêmico.

### Objetivo

Receber o relato do usuário e encaminhá-lo para o tipo de serviço mais adequado, sem diagnosticar.

Possíveis destinos iniciais:

- UBS / Atenção Primária;
- UPA / atendimento de urgência;
- Hospital / emergência, quando aplicável;
- fluxo de emergência / SAMU 192.

### Tecnologias obrigatórias da Etapa 1

- **Agent Harness** — coordena agentes, ferramentas, regras e fluxo;
- **Machine Learning** — auxilia a classificar o tipo de encaminhamento;
- **LLM** — transforma o texto livre do usuário em dados estruturados;
- **regras determinísticas** — protegem situações críticas e validam o resultado;
- **geolocalização** — encontra a unidade adequada mais próxima;
- **banco relacional** — guarda dados estruturados do usuário;
- **RAG / banco vetorial** — consulta regras, protocolos e documentos oficiais quando necessário.

### Papel do Machine Learning

O ML não deve tentar descobrir uma doença.

Ele recebe variáveis estruturadas e devolve uma classificação auxiliar de encaminhamento.

Exemplo:

```text
idade
sintomas estruturados
duração
sinais relatados
contexto clínico relevante
        ↓
Modelo de Machine Learning
        ↓
probabilidades / classe de encaminhamento
        ↓
UBS | Urgência | Emergência
```

Exemplo de saída:

```json
{
  "primary_care": 0.12,
  "urgent_care": 0.73,
  "emergency": 0.15
}
```

O resultado do ML não decide sozinho. O Agent Harness combina:

```text
LLM + ML + regras de segurança + regras do SUS + contexto autorizado
                              ↓
                      decisão de roteamento
```

Uma regra crítica pode sobrepor o modelo:

```python
if red_flag_detected:
    route = "emergency_flow"
```

### Fluxo da Etapa 1

```text
LOGIN
  ↓
IDENTIDADE / PRONTUÁRIO AUTORIZADO
  ↓
"O que você está sentindo?"
  ↓
LLM extrai dados estruturados
  ↓
Safety Agent
  ↓
Machine Learning
  ↓
Care Routing Agent
  ↓
Regras do SUS
  ↓
Agent Harness valida a rota
  ↓
UBS / UPA / Hospital / Emergência
  ↓
Geolocalização
  ↓
Unidade adequada mais próxima
  ↓
Resposta ao usuário
```

### Limite do sistema

O sistema deve responder:

> "Com as informações fornecidas, o encaminhamento indicado pelo sistema é procurar atendimento de urgência."

E não:

> "Você tem pneumonia."

---

## Etapa 2 — Medicamentos no SUS

Depois do roteamento assistencial, o Health-flow passa a responder questões como:

- O SUS disponibiliza este medicamento?
- Ele faz parte da relação aplicável?
- Quais são os critérios de acesso?
- Precisa de receita ou documentação específica?
- Onde o usuário pode tentar obter o medicamento?
- Qual unidade ou farmácia vinculada está mais próxima?

Fluxo:

```text
Usuário pergunta pelo medicamento
        ↓
Medication Agent
        ↓
RAG + fontes oficiais + APIs/bases disponíveis
        ↓
verificação de disponibilidade/regras
        ↓
geolocalização
        ↓
local de acesso mais adequado
        ↓
resposta com fonte
```

O LLM não deve inventar cobertura ou disponibilidade de medicamento. A resposta precisa estar ligada a uma fonte verificável.

---

## Etapa 3 — Planos de saúde

A terceira etapa adiciona saúde suplementar.

O usuário poderá informar ou vincular seu plano e perguntar:

- Meu plano cobre este hospital?
- Este hospital faz parte da minha rede?
- Meu plano cobre determinada especialidade?
- Quais especialistas da minha rede existem perto de mim?
- Preciso de autorização?
- Qual unidade da rede é mais próxima?

Fluxo:

```text
Usuário / plano vinculado
        ↓
Insurance Agent
        ↓
produto/plano específico
        ↓
cobertura + rede credenciada
        ↓
especialidade / hospital / serviço
        ↓
geolocalização
        ↓
opções mais adequadas
        ↓
resposta
```

Não basta conhecer a operadora. O sistema deve considerar o produto/plano específico e as informações verificáveis da rede.

---

## Arquitetura resumida

```text
                        HEALTH-FLOW

                            Usuário
                              |
                            Login
                              |
                    Prontuário autorizado
                              |
                              v
                       Agent Harness
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
         LLM                 ML                Regras
          |                   |              de segurança
          +-------------------+-------------------+
                              |
                              v
                    Care Routing Agent
                              |
                regras / conhecimento SUS
                              |
        +---------------------+---------------------+
        |                     |                     |
       UBS                   UPA               Emergência
        |                     |                     |
        +---------- Geolocalização -----------------+
                              |
                              v
                        Resposta final
```

Mais detalhes em [docs/architecture.md](docs/architecture.md).

## Dados e RAG

O prontuário não deve ser armazenado apenas em banco vetorial.

### PostgreSQL

Para dados estruturados:

- usuários;
- consentimentos;
- medicamentos ativos;
- alergias;
- condições registradas;
- atendimentos;
- exames estruturados;
- plano;
- auditoria.

### PostgreSQL + pgvector

Para RAG e busca semântica:

- protocolos;
- regras do SUS;
- documentação de medicamentos;
- documentos administrativos;
- regras de planos;
- conteúdo oficial não estruturado.

### Object Storage

Para arquivos originais:

- PDFs;
- laudos;
- documentos;
- imagens.

## Princípios

- Não diagnosticar.
- Não prescrever.
- Não alterar tratamento.
- ML auxilia o encaminhamento; não faz diagnóstico.
- Emergências têm prioridade.
- O LLM não decide sozinho.
- O resultado deve ser auditável.
- O prontuário completo não deve ser enviado ao LLM por padrão.
- Dados clínicos só podem ser usados quando necessários e autorizados.
- Informações de medicamentos e planos devem vir de fontes verificáveis.

## Stack inicial sugerida

```text
Frontend: React / Next.js
Backend: Python + FastAPI
Agent orchestration: Agent Harness
LLM: OpenAI
Machine Learning: Python + scikit-learn
Banco: PostgreSQL
Vector DB: pgvector
RAG: embeddings + retrieval
Cache: Redis
Arquivos: S3 / Object Storage
```

## Status

O desenvolvimento começa pela **Etapa 1: Agent Harness + Machine Learning + encaminhamento assistencial**.
