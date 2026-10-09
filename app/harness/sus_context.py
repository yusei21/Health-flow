"""Reviewed navigation context; not a clinical triage protocol. See docs/SUS_CONTEXT.md."""

from app.schemas.care import CareLevel

SUS_CONTEXT_VERSION = "2026-10-09"
SUS_NAVIGATION_CONTEXT = """
Contexto de navegação SUS:
- Nunca indicar diagnóstico, hipótese de doença, prescrição ou exame para o usuário.
- Atenção Primária/UBS é a porta preferencial para cuidado não urgente e coordenação.
- Urgências precisam de avaliação presencial; UPA integra a rede de urgências.
- Risco imediato: orientar SAMU 192; não aguardar prontuário, ML ou busca de unidade.
- Observação, internação e ambulatório especializado dependem da equipe e regulação.
- O aplicativo não atribui classificação de risco oficial nem substitui acolhimento.
- CNES não comprova plantão, vagas, capacidade de emergência ou estoque de medicamentos.
- SIGTAP/RENAME não garantem acesso individual: confirmar critérios e rede local.
"""

NEXT_STEP = {
    CareLevel.PRIMARY_CARE: "Procure uma Unidade Básica de Saúde (UBS) para avaliação. "
    "A equipe coordena o cuidado e eventual encaminhamento ao ambulatório especializado.",
    CareLevel.URGENT_CARE: "Procure atendimento de urgência (UPA) o quanto antes. "
    "A equipe fará o acolhimento e decidirá sobre observação ou encaminhamento hospitalar.",
    CareLevel.EMERGENCY: "Ligue imediatamente 192 (SAMU) e siga as orientações da regulação.",
}
