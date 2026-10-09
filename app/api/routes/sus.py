"""Source-based navigation only; never fabricate coverage or pharmacy stock."""

from enum import StrEnum

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(prefix="/sus", tags=["sus"])


class Topic(StrEnum):
    PROCEDURE = "procedure"
    MEDICINE = "medicine"


class SUSQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    topic: Topic
    question: str = Field(min_length=3, max_length=500)


@router.post("/questions")
async def answer_question(body: SUSQuestion) -> dict[str, object]:
    if body.topic is Topic.MEDICINE:
        return {
            "status": "not_verified",
            "answer": "Não foi possível confirmar esse medicamento ou seu estoque. "
            "Consulte o nome do princípio ativo e a apresentação na Rename. "
            "Confirme os critérios e a farmácia de referência com a assistência farmacêutica "
            "municipal ou estadual. Para o componente especializado, o acesso depende "
            "dos protocolos e da autorização. Não altere seu tratamento por esta consulta.",
            "sources": [
                {
                    "title": "Rename",
                    "url": "https://www.gov.br/saude/pt-br/composicao/sectics/rename",
                },
                {
                    "title": "CEAF",
                    "url": "https://www.gov.br/saude/pt-br/composicao/sectics/daf/ceaf",
                },
            ],
        }
    return {
        "status": "not_verified",
        "answer": "Não foi possível confirmar a oferta desse exame ou tratamento. "
        "Consulte o procedimento e a competência vigente no SIGTAP. "
        "A presença na tabela não garante acesso individual ou disponibilidade local. "
        "A UBS e a regulação municipal orientam os critérios, a solicitação profissional "
        "e a unidade de referência. Esta consulta não indica qual exame você deve fazer.",
        "sources": [
            {
                "title": "SIGTAP",
                "url": "http://sigtap.datasus.gov.br/tabela-unificada/app/sec/inicio.jsp",
            },
            {
                "title": "Atenção Primária",
                "url": "https://www.gov.br/saude/pt-br/composicao/saps/atencao-primaria",
            },
        ],
    }
