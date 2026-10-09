# Integrações e autorização

## gov.br e prontuário

O botão gov.br está indisponível de forma explícita. A demonstração não simula um login
oficial nem solicita CPF/senha gov.br. `/api/v1/access/capabilities` informa as capacidades.

Referências:
- [Roteiro Login Único](https://acesso.gov.br/roteiro-tecnico/)
- [Solicitação de integração](https://www.gov.br/governodigital/pt-br/estrategias-e-governanca-digital/transformacao-digital/servico-de-integracao-aos-produtos-de-identidade-digital-gov.br)
- [Guia do gestor RNDS](https://rnds-guia.saude.gov.br/docs/publico-alvo/gestor/gestor/)

A autenticação identifica o usuário; não entrega o prontuário. A integração gov.br depende
de aprovação do serviço. O acesso RNDS exige credenciamento do estabelecimento, certificado
e serviços autorizados. Não existe nesta implementação um endpoint genérico que “baixa
o prontuário completo por CPF”. A fonte real pode ser e-SUS/PEC ou sistema institucional,
conforme autorização e capacidade efetivamente fornecidas.

Próximo passo técnico, depois de credenciar: implementar OIDC no servidor com validação
de assinatura/issuer/audience/nonce/state, sessão segura e expiração; mapear identidade para
registro autorizado; substituir `DemoTokenAuthenticator` e `PatientRepository`; registrar
consentimento versionado no servidor, finalidade, escopo, expiração e revogação. Não colocar
segredos em `VITE_*`. A autorização do protótipo é por requisição e não substitui esse registro
persistente. O token de desenvolvimento é compartilhado com um único usuário fictício.

## Contratos disponíveis

| Endpoint | Condição |
|---|---|
| `GET /api/v1/patients/me?consent=true` | Bearer de demonstração e autorização; sem ela, 403 |
| `POST /api/v1/routing` | `use_patient_record` desativado por padrão; sem ele, não lê perfil |
| `POST /api/v1/audio/transcribe?consent=true` | Bearer, consentimento separado e transcritor configurado |
| `POST /api/v1/sus/questions` | Fonte de informação geral; não consulta prontuário |

Não há persistência de perfil, relato ou áudio na interface; a página usa memória.
Respostas de prontuário e áudio usam `Cache-Control: no-store`. Retirar autorização
não desfaz um pedido já processado; interrompe a exibição e o uso em novos pedidos.

## Áudio

Enviar um arquivo de áudio suportado (até 10 MB), autorizar o envio, transcrever,
revisar/corrigir o texto e só então enviar o relato ao harness. Não há reconhecimento
automático oculto nem envio automático para encaminhamento. A versão atual recebe
arquivo já gravado; gravação pelo microfone na página é uma etapa futura.

Configuração **opcional no servidor**:

```dotenv
HEALTHFLOW_TRANSCRIPTION_URL=https://seu-provedor-aprovado/v1/audio/transcriptions
HEALTHFLOW_TRANSCRIPTION_API_KEY=segredo-do-servidor
HEALTHFLOW_TRANSCRIPTION_MODEL=whisper-1
```

O adaptador usa multipart compatível com transcrição OpenAI, idioma `pt`, timeout de 60s,
e valida o texto retornado antes de exibir. O arquivo recebido fica em memória; a política
de retenção do provedor é separada e deve ser informada aos usuários antes de habilitar.
O Ollama de extração de sintomas não é assumido como serviço de transcrição.
Sem esse endpoint, o áudio é desabilitado e o texto permanece disponível.

## Unidades próximas, exames e medicamentos

Configure `HEALTHFLOW_CNES_DATABASE` usando o importador documentado em `CNES_IMPORT.md`.
A busca usa o tipo compatível e ordena candidatos por km. Para confirmar observação,
plantão ou capacidade, integrar o diretório oficial/regulação do município; CNES sozinho
não confirma esses detalhes. O link de rota envia coordenadas da unidade ao provedor
somente quando o usuário o abre.

SIGTAP/RENAME são referências de consulta. Para responder cobertura de um item específico,
importar competência vigente e seus critérios, com proveniência. Para informar uma farmácia
com estoque, integrar a fonte municipal/estadual com horário de atualização. Sem essas
fontes, a interface declara ausência de verificação e não promete disponibilidade.
