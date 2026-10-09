import { useRef, useState, type FormEvent } from "react";
import { PatientAccess } from "./components/PatientAccess";
import { AudioInput } from "./components/AudioInput";
import { SUSQuestions } from "./components/SUSQuestions";
import {
  LocationControl,
  type LocationState,
} from "./components/LocationControl";
import { ResultCard } from "./components/ResultCard";
import { ApiError, requestRouting } from "./services/api";
import { readApiConfig } from "./services/config";
import {
  getCurrentLocation,
  LocationError,
  LOCATION_ERROR_MESSAGES,
} from "./services/geolocation";
import {
  MESSAGE_MAX_LENGTH,
  MESSAGE_MIN_LENGTH,
  type RoutingResponse,
} from "./types/routing";

const DISCLAIMER =
  "Health-flow fornece orientação de navegação em saúde e não substitui avaliação profissional.";

type SubmitState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "done"; result: RoutingResponse }
  | { status: "error"; message: string; emergencyHint: string | null };

export function App() {
  const requestGeneration = useRef(0);
  const [access, setAccess] = useState({ ready: false, consent: false });
  const [reviewAudio, setReviewAudio] = useState(false);
  const [message, setMessage] = useState("");
  // Coordinates live only in this component's memory: never logged or persisted.
  const [location, setLocation] = useState<LocationState>({ status: "idle" });
  const [submit, setSubmit] = useState<SubmitState>({ status: "idle" });

  const loading = submit.status === "loading";
  const messageOk = message.trim().length >= MESSAGE_MIN_LENGTH;
  const canSubmit =
    access.ready &&
    !reviewAudio &&
    messageOk &&
    location.status === "ready" &&
    !loading;

  async function handleLocation() {
    setLocation({ status: "loading" });
    try {
      setLocation({ status: "ready", location: await getCurrentLocation() });
    } catch (error) {
      const text =
        error instanceof LocationError
          ? error.message
          : LOCATION_ERROR_MESSAGES.unavailable;
      setLocation({ status: "error", message: text });
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!canSubmit || location.status !== "ready") return;
    const generation = ++requestGeneration.current;
    setSubmit({ status: "loading" });
    try {
      const result = await requestRouting(
        {
          message: message.trim(),
          latitude: location.location.latitude,
          longitude: location.location.longitude,
          use_patient_record: access.consent,
        },
        readApiConfig(),
      );
      if (generation !== requestGeneration.current) return;
      setSubmit({ status: "done", result });
    } catch (error) {
      if (generation !== requestGeneration.current) return;
      setSubmit({
        status: "error",
        message:
          error instanceof ApiError
            ? error.message
            : "Erro inesperado. Tente novamente.",
        emergencyHint: error instanceof ApiError ? error.emergencyHint : null,
      });
    }
  }

  return (
    <main className="page">
      <div className="card">
        <h1>Health-flow</h1>
        <p className="emergency-notice">
          Em risco imediato, ligue <a href="tel:192">192 (SAMU)</a>. Não espere
          login, prontuário ou localização.
        </p>
        <PatientAccess
          onAccess={(ready, consent) => {
            requestGeneration.current++;
            setAccess({ ready, consent });
            setSubmit({ status: "idle" });
          }}
        />
        {access.ready && (
          <form onSubmit={handleSubmit}>
            <h2>2. Relato e atendimento mais próximo</h2>
            <p className="muted small">
              {access.consent
                ? "Consulta do perfil fictício autorizada."
                : "Sem consulta ao prontuário: o relato continua disponível."}
            </p>
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

            <AudioInput
              disabled={loading}
              onTranscript={(text) => {
                setMessage(text);
                setReviewAudio(true);
                setSubmit({ status: "idle" });
              }}
            />
            {reviewAudio && (
              <label>
                <input
                  type="checkbox"
                  onChange={(e) => setReviewAudio(!e.target.checked)}
                />{" "}
                Revisei e corrigi a transcrição acima.
              </label>
            )}

            <LocationControl
              state={location}
              onRequest={handleLocation}
              disabled={loading}
            />

            <button
              type="submit"
              className="button primary"
              disabled={!canSubmit}
            >
              {loading ? "Buscando…" : "Buscar atendimento"}
            </button>
            {location.status !== "ready" && (
              <p className="muted small">
                A localização é necessária para encontrar uma unidade compatível
                próxima.
              </p>
            )}
          </form>
        )}

        {submit.status === "error" && (
          <div className="error" role="alert">
            <p>{submit.message}</p>
            {submit.emergencyHint && (
              <p className="small">{submit.emergencyHint}</p>
            )}
          </div>
        )}
        {submit.status === "done" && <ResultCard result={submit.result} />}

        <SUSQuestions />
        <p className="disclaimer">{DISCLAIMER}</p>
      </div>
    </main>
  );
}
