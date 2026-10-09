# Contexto SUS do harness

Revisão das referências: 09/10/2026 UTC. Versão executável: `app/harness/sus_context.py`.
O contexto é incluído na extração estruturada e as orientações de saída são templates
controlados; o texto livre do LLM nunca é exibido ao usuário.

## Serviços e limites

| Situação de navegação | Orientação | Limite |
|---|---|---|
| Cuidado não urgente | Atenção Primária/UBS | Coordena eventual acesso especializado |
| Urgência | Avaliação em UPA/serviço compatível | Acolhimento e classificação feitos pela equipe |
| Risco imediato | Ligar SAMU 192 e seguir regulação | Não aguardar login, prontuário ou ML |
| Observação/internação | Decisão da equipe presencial | Nunca prometer leito ou duração |
| Ambulatório especializado | Encaminhamento e regulação local | Não criar acesso direto por hipótese diagnóstica |

Fontes oficiais consultadas:

- [Atenção Primária](https://www.gov.br/saude/pt-br/composicao/saps/atencao-primaria):
  primeiro nível de atenção e porta principal do SUS.
- [UPA 24h](https://www.gov.br/saude/pt-br/assuntos/saude-de-a-a-z/u/upa-24h):
  integra a rede, avalia e estabiliza; encaminha para referência quando necessário.
- [SAMU 192](https://www.gov.br/saude/pt-br/composicao/saes/samu-192):
  serviço regulado; sinais importantes incluem dificuldade respiratória,
  alteração súbita da fala/força, convulsões e sangramento/traumas graves.
- [Rede de Urgências](https://www.gov.br/saude/pt-br/composicao/saes/samu-192/rau/rede-de-atencao-as-urgencias-e-emergencias):
  componentes articulados, acolhimento e regulação.
- [SIGTAP — atributos](https://wiki.saude.gov.br/sigtap/index.php/Gerais):
  cadastro de procedimentos e condições; não é comprovação de vaga local.
- [Rename](https://www.gov.br/saude/pt-br/composicao/sectics/rename):
  consultar medicamento e apresentação na ferramenta atualizada.
- [CEAF](https://www.gov.br/saude/pt-br/composicao/sectics/daf/ceaf):
  acesso por critérios de protocolos e avaliação/autorização estadual.

## Regras do produto

1. Nunca afirmar doença, hipótese, prescrição ou exame recomendado. O schema da extração
   aceita somente sintomas, duração, intensidade e idade. Respostas vêm de templates.
2. Prontuário só é consultado com autorização explícita na requisição. Sem autorização,
   o Context Builder considera somente o relato. Identidade e resumos de consultas não
   vão ao LLM; a exibição de condições já registradas não é um novo diagnóstico.
3. Negação de prontuário não bloqueia navegação. Retirar consentimento limpa o perfil
   da interface e as próximas requisições não consultam o repositório.
4. Precheck precede extração e ML. Sinal de risco não espera modelos. As regras existentes
   são ilustrativas, não equivalem a protocolo clínico SUS/Manchester validado.
5. Procurar serviços compatíveis **antes** de ordenar por distância geográfica. Nunca
   escolher um hospital só porque é mais próximo. CNES confirma cadastro, não plantão,
   capacidade específica, atendimento naquele momento ou leito de observação. A distância
   Haversine não é tempo de viagem. Mostrar nome, endereço, km e rota quando disponíveis.
6. Sem snapshot CNES configurado, não apresentar pontos OSM como unidades SUS confirmadas.
   A decisão e o aviso de indisponibilidade continuam; não inventar endereço.
7. Consultas SUS são independentes de consulta a prontuário. Sem catálogo oficial atual e
   integração municipal, o status é `not_verified`, com caminho de verificação nas fontes.
   Não responder “não cobre” só porque a integração ou o registro estão ausentes.

## Pendências clínicas e de dados

Antes de qualquer uso assistencial: protocolo aprovado pela equipe responsável, validação
independente dos limiares e modelos, atualização de referências, unidade de referência
local verificada e monitoramento de encaminhamentos. As fontes de navegação acima não
validam os limiares do ML, não fornecem um protocolo completo de triagem e não comprovam
que o modelo acadêmico é seguro para uso real.
