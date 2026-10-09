# Health-flow

Protótipo de navegação na rede SUS: relata sintomas, usa somente o contexto autorizado
e orienta o serviço de atendimento. **Nunca fornece diagnóstico, hipótese de doença,
prescrição ou indicação de exames. Não é um classificador de risco oficial.**

## Jornada

1. Entrada preparada para gov.br; integração real ainda indisponível.
2. Demonstração explícita com paciente fictício; autorização opcional para consultar o perfil.
3. Relato por texto ou áudio (transcrição opcional configurada no servidor, com consentimento).
4. Revisão da transcrição antes de enviar. Harness: precheck → extração → contexto autorizado
   → regras de segurança → ML auxiliar → serviço SUS → localização.
5. Orientação para UBS, urgência/UPA ou SAMU 192. Observação, internação e ambulatório
   especializado são decisões da equipe/regulação, não do aplicativo.
6. Consultas sobre exames, tratamentos e medicamentos com fontes oficiais; sem inventar
   cobertura, elegibilidade ou estoque.

**Estado atual:** autenticação por token de desenvolvimento e prontuário sintético.
Login gov.br não dá acesso automático à RNDS ou a um prontuário nacional completo.
Sem credenciamento, o aplicativo não busca dados reais. Consentimento no protótipo
vale para cada requisição e pode ser retirado; não constitui infraestrutura de produção.

## Como funciona o harness

O harness é o **coordenador do encaminhamento**: mantém o estado do pedido, escolhe
qual etapa executar, verifica se ela é permitida e chama o componente responsável.
É uma implementação própria em Python, na classe `AutonomousHealthFlowHarness`.
O planejador ativo é `DeterministicPlanner`, com decisões definidas em código.

O ciclo se repete até concluir ou atingir um limite:

```python
while not state.finished:
    planned = await planner.next_action(state)
    policy.validate(planned, state)
    await executor.execute(planned, state)
```

| Componente | Responsabilidade |
|---|---|
| `HarnessState` (`app/harness/state.py`) | Mantém etapas concluídas, contexto, resultados e falhas do pedido |
| `DeterministicPlanner` (`app/harness/planner.py`) | Propõe a próxima ação conforme o estado |
| `HarnessPolicy` (`app/harness/policy.py`) | Valida ordem, ações permitidas e limites antes da execução |
| `ActionExecutor` (`app/harness/executor.py`) | Executa a ação aprovada e registra seu resultado |
| `SafetyEngine` (`app/safety/engine.py`) | Aplica regras de segurança e estabelece o nível mínimo de atendimento |
| `build_routing_response` (`app/harness/response.py`) | Produz a resposta usando mensagens controladas |

### Etapas do pedido

1. **Verificação inicial de segurança:** examina o texto antes de chamar o modelo.
   Se detectar sinal de emergência, segue para encaminhamento e busca de unidade,
   sem executar extração pelo LLM nem classificação por ML.
2. **Extração do relato:** o LLM via Ollama organiza sintomas, duração, intensidade
   e idade informada. A saída é validada por schema; texto livre, diagnóstico e
   recomendações produzidos pelo modelo não são apresentados ao usuário.
3. **Contexto autorizado:** consulta o perfil apenas quando `use_patient_record=true`.
   O Context Builder seleciona informações relevantes, como faixa etária, fatores
   de risco, alergias e medicamentos. Sem autorização, usa somente o relato.
   O prontuário completo, a identidade e os resumos de consultas não vão ao LLM.
4. **Segurança com contexto:** reavalia o relato estruturado e o contexto disponível.
   Um sinal de emergência nessa etapa impede a execução do ML.
5. **Classificador auxiliar:** sugere um nível de atendimento. O agente de
   encaminhamento combina a sugestão com o nível mínimo das regras de segurança.
   O ML não pode reduzir esse mínimo; o modelo atual foi treinado em dados sintéticos.
6. **Unidade próxima:** busca candidatos do tipo de serviço escolhido e ordena por
   distância geográfica. Usa o primeiro resultado para mostrar nome, endereço, km
   e rota. O snapshot CNES precisa estar configurado; cadastro não confirma plantão,
   vagas ou capacidade de observação. A busca possui raio configurável, de 25 km por padrão.
7. **Resposta:** informa o serviço e a próxima ação, ou pede mais informações quando
   necessário. Observação, internação e acesso especializado dependem da equipe e
   da regulação; o aplicativo não atribui diagnóstico ou classificação de risco oficial.

### Limites e falhas

A política permite até **10 etapas, 1 ação de extração pelo LLM, 3 ações de ferramentas
e 90 segundos por pedido**. O provedor de LLM pode realizar as tentativas limitadas
configuradas dentro da ação de extração. Uma ação já concluída não pode ser repetida.

Sem perfil encontrado, o fluxo continua com o relato. Sem ML, o encaminhamento usa
uma alternativa conservadora de urgência. Sem fonte de unidades disponível, mantém
uma orientação sem inventar endereço. Se a extração falhar e não houver sinal de
emergência no precheck, retorna erro controlado. Estouro do prazo encerra o pedido.
Os logs de execução registram etapas, códigos e tempos; não registram relato ou prontuário.

O contexto de navegação SUS está em `app/harness/sus_context.py`, entra no prompt de
extração e nos textos de encaminhamento. As fontes e seus limites estão documentados
em [SUS_CONTEXT.md](docs/SUS_CONTEXT.md). Consultas de exames/medicamentos e transcrição
são endpoints separados: não fazem parte do ciclo de encaminhamento acima.
As regras e o classificador continuam acadêmicos e exigem validação clínica independente.

## Por que o gov.br ainda não está habilitado?

O projeto ainda não tem aprovação nem credenciais de integração. Pelo processo oficial,
a solicitação é feita por agente público ou gestor de serviço público e passa por análise
individual. Um projeto pessoal ou acadêmico, sozinho, não obtém integração automaticamente.
Um órgão público responsável pode apresentar o serviço para avaliação; aprovação não é garantida.
Veja a [solicitação oficial de integração gov.br](https://www.gov.br/governodigital/pt-br/estrategias-e-governanca-digital/transformacao-digital/servico-de-integracao-aos-produtos-de-identidade-digital-gov.br).

Depois da aprovação, ainda será necessário implementar a autenticação conforme o
[roteiro técnico do Login Único](https://acesso.gov.br/roteiro-tecnico/). Hoje há somente
um token de demonstração e o botão gov.br permanece indisponível.

**Login gov.br e acesso ao prontuário são integrações diferentes.** Para RNDS,
o estabelecimento precisa de credenciamento, certificado digital e autorização para
os serviços pretendidos, conforme o [guia oficial](https://rnds-guia.saude.gov.br/docs/publico-alvo/gestor/gestor/).
O consentimento do paciente não substitui esse acesso institucional. Login aprovado
também não garante um prontuário completo: a fonte autorizada e os dados disponíveis
precisam ser definidos. Detalhes em [INTEGRATIONS.md](docs/INTEGRATIONS.md).

## Executar

- `uv sync --frozen` e `cd frontend && npm ci`.
- Copie `.env.example` para `.env` e `frontend/.env.example` para `frontend/.env`.
- Configure o mesmo token **de demonstração** nos dois ambientes e o UUID de teste no backend.
- Inicie Ollama e o modelo documentado; `make train-synthetic` cria o modelo auxiliar fictício.
- `make dev-all` inicia API e interface. Sem CNES configurado, nenhuma unidade SUS é confirmada.
- Áudio: configure o endpoint compatível com transcrição no servidor; sem ele, use texto.

## Organização e referências

| Pasta | Conteúdo |
|---|---|
| `app/` | API, harness, contexto, regras e adaptadores |
| `frontend/` | Entrada, consentimento, prontuário, relato e consultas SUS |
| `tests/` | Contratos, segurança, consentimento e experimentos |
| `docs/` | Contexto SUS, integração e documentação técnica |
| `benchmarks/` | Pesquisa e evidências agregadas; novos resultados locais são ignorados |

- [Contexto SUS e limites do harness](docs/SUS_CONTEXT.md)
- [Integração gov.br, prontuário e áudio](docs/INTEGRATIONS.md)
- [Instalação detalhada e experimentos](docs/development.md)
- [Arquitetura](docs/architecture.md) · [ML](docs/machine-learning.md)
- [CNES](docs/CNES_IMPORT.md) · [Gates de segurança](docs/SAFETY_AND_CNES_GATES.md)

## Verificar

`make check`; no frontend: `npm test` e `npm run build`.
As regras e o ML atuais continuam acadêmicos. Referências oficiais orientam a navegação;
a leitura dessas páginas não valida clinicamente os limiares ou o classificador.
