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
