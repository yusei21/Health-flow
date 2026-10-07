"""Application error hierarchy.

`public_message` is the only text that may reach API clients; internal details stay
in the exception chain and logs.
"""


class HealthFlowError(Exception):
    public_message = "Não foi possível processar a solicitação."


class LLMError(HealthFlowError):
    public_message = "O serviço de interpretação do relato está indisponível."


class LLMUnavailableError(LLMError):
    """The provider could not be reached, timed out or returned an HTTP error."""


class LLMResponseError(LLMError):
    """The provider answered, but not with a valid structured payload."""


class MLInferenceError(HealthFlowError):
    public_message = "O classificador auxiliar está indisponível."


class MLModelUnavailableError(MLInferenceError):
    """The trained artifact is missing or incompatible with the current code."""


class PatientNotFoundError(HealthFlowError):
    public_message = "Perfil do paciente não encontrado."


class FacilityProviderError(HealthFlowError):
    public_message = "A busca de unidades de saúde está indisponível."


class SafetyEngineError(HealthFlowError):
    public_message = "Falha na verificação de segurança."


class AuthenticationError(HealthFlowError):
    public_message = "Credenciais ausentes ou inválidas."


class HarnessError(HealthFlowError):
    """The harness stopped a run before it could produce a routing decision."""

    public_message = "Não foi possível concluir a orientação."

    def __init__(self, message: str, finish_reason: str) -> None:
        super().__init__(message)
        self.finish_reason = finish_reason


class HarnessPolicyError(HarnessError):
    """A planner proposed an action the policy does not allow in the current state."""


class HarnessLimitError(HarnessError):
    """A step, LLM-call or tool-call budget was exhausted."""


class HarnessTimeoutError(HarnessError):
    """The run exceeded its global time budget."""
