# Health-flow

Health-flow é um assistente inteligente de navegação em saúde para orientar cidadãos entre SUS e saúde suplementar.

A proposta é receber a necessidade do usuário em linguagem natural, entender o contexto, aplicar regras de segurança e usar um **Agent Harness** para coordenar consultas a serviços de saúde, medicamentos, planos, prontuário autorizado e localização.

> O Health-flow não substitui atendimento médico, não realiza diagnóstico autônomo e não prescreve medicamentos. O objetivo é orientar o próximo passo do usuário dentro da rede de saúde.

## Problema

Hoje, uma pessoa pode ter dificuldade para responder perguntas simples como:

- Onde devo procurar atendimento para o que estou sentindo?
- Qual UBS, UPA, hospital ou CAPS está mais próximo?
- O SUS disponibiliza determinado medicamento?
- Onde esse medicamento pode ser retirado?
- Meu plano cobre determinado especialista ou procedimento?
- Qual profissional da minha rede atende perto de mim?
- Quais informações do meu histórico são relevantes para o atendimento?

Essas informações existem em sistemas diferentes. O Health-flow propõe uma camada inteligente para conectá-las.

## Como funciona

```text
Usuário
   |
   v
Agent Harness
   |
   +--> Safety Agent
   +--> Intent Agent
   +--> Patient Context Agent
   +--> SUS Agent
   +--> Medication Agent
   +--> Insurance Agent
   +--> Provider Agent
   +--> Navigation Agent
   |
   v
Resposta consolidada + próximo passo
```

O **Agent Harness** é o componente central da arquitetura. Ele controla quais agentes e ferramentas podem ser usados, em qual ordem, quais dados são necessários e quando uma regra de segurança deve interromper o fluxo normal.

## Exemplo de jornada

Usuário:

> Estou com dor no peito, falta de ar e estou suando. Tenho plano de saúde.

O sistema não deve começar pesquisando cobertura do plano.

O fluxo correto é:

1. detectar sinais potencialmente graves;
2. priorizar atendimento de urgência;
3. localizar uma unidade apropriada;
4. só depois considerar informações de cobertura que sejam relevantes.

Outro exemplo:

> O SUS fornece determinado medicamento?

O Health-flow pode consultar a fonte oficial correspondente e responder:

- se o medicamento consta na relação aplicável;
- em quais condições ele é disponibilizado;
- documentos ou critérios necessários;
- onde procurar atendimento ou retirada;
- fonte usada para sustentar a resposta.

## Agent Harness

O Harness será responsável por:

- identificar a intenção do usuário;
- controlar o fluxo entre agentes;
- aplicar guardrails e políticas de segurança;
- decidir quais tools/APIs podem ser chamadas;
- controlar acesso a dados sensíveis;
- registrar auditoria das ações;
- consolidar a resposta final;
- impedir que o LLM invente informações que devem vir de bases oficiais.

Exemplo de estado interno:

```json
{
  "intent": "find_care",
  "urgency": "urgent",
  "possible_service": "UPA",
  "requires_medical_record": false,
  "insurance": "SUS",
  "location_required": true,
  "tools_required": ["health_facility_search"],
  "human_review_required": false
}
```

## Agentes propostos

| Agente | Responsabilidade |
|---|---|
| Safety Agent | Identificar sinais críticos e aplicar regras de segurança |
| Intent Agent | Entender o objetivo da solicitação |
| Patient Context Agent | Obter apenas contexto clínico autorizado e necessário |
| SUS Agent | Consultar serviços, regras e acesso pelo SUS |
| Medication Agent | Consultar medicamentos e regras de dispensação |
| Insurance Agent | Verificar cobertura, rede e requisitos do plano |
| Provider Agent | Encontrar unidades e profissionais |
| Navigation Agent | Converter os resultados em um próximo passo claro |
| Audit Agent | Registrar fontes, acessos e decisões do fluxo |

## Prontuário e identidade

O sistema não deve liberar um prontuário apenas porque alguém digitou um CPF.

O acesso a informações clínicas deverá considerar:

- autenticação forte;
- autorização e consentimento;
- princípio do menor privilégio;
- rastreabilidade dos acessos;
- criptografia;
- LGPD;
- integração autorizada com sistemas oficiais.

A arquitetura deve tratar prontuário como dado altamente sensível.

## Machine Learning

Não é necessário treinar um modelo de Machine Learning próprio para o primeiro MVP.

O projeto pode começar com:

- LLM para compreensão de linguagem natural;
- Agent Harness para orquestração;
- regras determinísticas para segurança;
- RAG quando necessário;
- APIs e bases oficiais como fonte de verdade.

Machine Learning tradicional pode ser incluído posteriormente para problemas específicos e mensuráveis, como previsão de demanda, classificação auxiliar ou otimização operacional.

Em decisões clínicas de alto risco, o resultado de um modelo de ML não deve ser a única fonte de decisão.

## MVP

### MVP 1

- interface conversacional;
- identificação de intenção;
- triagem de segurança;
- busca de UBS, UPA, hospital e CAPS;
- consulta de medicamentos do SUS;
- respostas com fonte e rastreabilidade.

### MVP 2

- autenticação;
- perfil do usuário;
- integração autorizada com dados clínicos;
- histórico relevante para navegação.

### MVP 3

- planos de saúde;
- cobertura de especialidades e procedimentos;
- rede credenciada;
- prestadores próximos.

### MVP 4

- disponibilidade;
- encaminhamentos;
- agendamento quando houver integração;
- acompanhamento da jornada do usuário.

## Arquitetura inicial

```text
Frontend
   |
   v
Backend / API
   |
   v
Agent Harness
   |
   +--> Guardrails / Safety Rules
   |
   +--> LLM
   |
   +--> Agents
   |      +--> SUS
   |      +--> Medicamentos
   |      +--> Planos
   |      +--> Prontuário
   |      +--> Localização
   |
   +--> Tools / APIs / RAG
   |
   v
Banco + Auditoria + Observabilidade
```

Mais detalhes em [docs/architecture.md](docs/architecture.md).

## Princípios de segurança

- Não diagnosticar autonomamente.
- Não prescrever ou alterar medicamentos.
- Priorizar emergências sobre buscas administrativas.
- Não acessar prontuário apenas com CPF.
- Não responder cobertura com base apenas na memória do LLM.
- Informações de medicamentos devem ser sustentadas por fontes oficiais.
- Informações de planos devem considerar produto, contrato e rede aplicáveis.
- Registrar a origem das informações usadas.
- Expor incerteza quando os dados forem insuficientes.
- Minimizar o tratamento de dados pessoais e sensíveis.

## Objetivo do projeto

Construir um **GPS da saúde**: o usuário descreve sua necessidade e o sistema organiza dados, serviços e regras para indicar de forma segura qual é o próximo passo dentro do SUS ou da saúde suplementar.

---

Projeto em desenvolvimento.
