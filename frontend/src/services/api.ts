import type { ApiErrorBody, RoutingRequest, RoutingResponse } from "../types/routing";
import type { ApiConfig } from "./config";

export class ApiError extends Error {
  readonly status: number | null;
  readonly emergencyHint: string | null;

  constructor(message: string, status: number | null, emergencyHint: string | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.emergencyHint = emergencyHint;
  }
}

const DEFAULT_EMERGENCY_HINT = "Em caso de emergência, ligue 192 (SAMU).";

function isApiErrorBody(value: unknown): value is ApiErrorBody {
  return typeof value === "object" && value !== null && typeof (value as ApiErrorBody).error === "string";
}

function messageForStatus(status: number): string {
  if (status === 401) return "Acesso não autorizado. Verifique o token de demonstração (VITE_DEMO_TOKEN).";
  if (status === 422) return "Não foi possível enviar: verifique o texto e a localização.";
  if (status === 503) return "O serviço está temporariamente indisponível. Tente novamente em instantes.";
  return "Não foi possível obter a orientação agora. Tente novamente.";
}

export async function requestRouting(
  request: RoutingRequest,
  config: ApiConfig,
  fetchFn: typeof fetch = fetch,
): Promise<RoutingResponse> {
  let response: Response;
  try {
    response = await fetchFn(`${config.baseUrl}/api/v1/routing`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(config.token ? { Authorization: `Bearer ${config.token}` } : {}),
      },
      body: JSON.stringify(request),
    });
  } catch {
    throw new ApiError(
      "Não foi possível conectar ao servidor. Verifique se o backend está em execução.",
      null,
      DEFAULT_EMERGENCY_HINT,
    );
  }

  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const hint = isApiErrorBody(body) ? body.emergency_hint : DEFAULT_EMERGENCY_HINT;
    throw new ApiError(messageForStatus(response.status), response.status, hint);
  }
  if (body === null) {
    throw new ApiError("Resposta inválida do servidor.", response.status, DEFAULT_EMERGENCY_HINT);
  }
  return body as RoutingResponse;
}
