import { useState, type FormEvent } from "react";
import { LocationControl, type LocationState } from "./components/LocationControl";
import { ResultCard } from "./components/ResultCard";
import { ApiError, requestRouting } from "./services/api";
import { readApiConfig } from "./services/config";
import { getCurrentLocation, LocationError, LOCATION_ERROR_MESSAGES } from "./services/geolocation";
import { MESSAGE_MAX_LENGTH, MESSAGE_MIN_LENGTH, type RoutingResponse } from "./types/routing";

const DISCLAIMER =
  "Health-flow fornece orientação de navegação em saúde e não substitui avaliação profissional.";

type SubmitState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "done"; result: RoutingResponse }
  | { status: "error"; message: string; emergencyHint: string | null };

export function App() {
  const [message, setMessage] = useState("");
  // Coordinates live only in this component's memory: never logged or persisted.
  const [location, setLocation] = useState<LocationState>({ status: "idle" });
  const [submit, setSubmit] = useState<SubmitState>({ status: "idle" });

  const loading = submit.status === "loading";
  const messageOk = message.trim().length >= MESSAGE_MIN_LENGTH;
  const canSubmit = messageOk && location.status === "ready" && !loading;

  async function handleLocation() {
    setLocation({ status: "loading" });
    try {
      setLocation({ status: "ready", location: await getCurrentLocation() });
    } catch (error) {
      const text =
        error instanceof LocationError ? error.message : LOCATION_ERROR_MESSAGES.unavailable;
      setLocation({ status: "error", message: text });
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!canSubmit || location.status !== "ready") return;
    setSubmit({ status: "loading" });
    try {
      const result = await requestRouting(
        {
          message: message.trim(),
          latitude: location.location.latitude,
          longitude: location.location.longitude,
        },
        readApiConfig(),
      );
      setSubmit({ status: "done", result });
    } catch (error) {
      setSubmit({
        status: "error",
        message: error instanceof ApiError ? error.message : "Erro inesperado. Tente novamente.",
        emergencyHint: error instanceof ApiError ? error.emergencyHint : null,
      });
    }
  }

  return (
    <main className="page">
      <div className="card">
        <h1>Health-flow</h1>
        <form onSubmit={handleSubmit}>
          <label htmlFor="message" className="label">
            O que você está sentindo?
          </label>
          <textarea
            id="message"
            rows={6}
            value={message}
            maxLength={MESSAGE_MAX_LENGTH}
            onChange={(event) => setMessage(event.target.value)}
            placeholder="Descreva seus sintomas, há quanto tempo começaram e a intensidade."
            disabled={loading}
          />

          <LocationControl state={location} onRequest={handleLocation} disabled={loading} />

          <button type="submit" className="button primary" disabled={!canSubmit}>
            {loading ? "Buscando…" : "Buscar atendimento"}
          </button>
          {location.status !== "ready" && (
            <p className="muted small">
              A localização é necessária para encontrar uma unidade compatível próxima.
            </p>
          )}
        </form>

        {submit.status === "error" && (
          <div className="error" role="alert">
            <p>{submit.message}</p>
            {submit.emergencyHint && <p className="small">{submit.emergencyHint}</p>}
          </div>
        )}
        {submit.status === "done" && <ResultCard result={submit.result} />}

        <p className="disclaimer">{DISCLAIMER}</p>
      </div>
    </main>
  );
}
