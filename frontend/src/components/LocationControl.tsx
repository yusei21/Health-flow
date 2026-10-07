import type { UserLocation } from "../types/routing";
import { formatAccuracy } from "../services/presentation";

export type LocationState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; location: UserLocation }
  | { status: "error"; message: string };

interface Props {
  state: LocationState;
  onRequest: () => void;
  disabled: boolean;
}

export function LocationControl({ state, onRequest, disabled }: Props) {
  return (
    <div className="location">
      <button
        type="button"
        className="button secondary"
        onClick={onRequest}
        disabled={disabled || state.status === "loading"}
      >
        {state.status === "loading" ? "Obtendo localização…" : "Usar minha localização"}
      </button>
      <p className={`location-status ${state.status}`} role="status" aria-live="polite">
        {state.status === "idle" && "Localização não informada."}
        {state.status === "loading" && "Aguardando o navegador…"}
        {state.status === "ready" && `Localização obtida (${formatAccuracy(state.location.accuracy)}).`}
        {state.status === "error" && `Localização não disponível. ${state.message}`}
      </p>
    </div>
  );
}
