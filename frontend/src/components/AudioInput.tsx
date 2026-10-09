import { useEffect, useState } from "react";
import { getCapabilities, transcribeAudio } from "../services/api";
import { readApiConfig } from "../services/config";

export function AudioInput({
  onTranscript,
  disabled,
}: {
  onTranscript: (text: string) => void;
  disabled: boolean;
}) {
  const [available, setAvailable] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    getCapabilities(readApiConfig())
      .then((x) => {
        if (active) setAvailable(x.audio_available);
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, []);
  async function transcribe() {
    if (!file || !consent || busy || disabled) return;
    setBusy(true);
    setError("");
    try {
      if (file.size > 10 * 1024 * 1024)
        throw new Error("Áudio deve ter no máximo 10 MB.");
      const result = await transcribeAudio(file, readApiConfig());
      onTranscript(result.text);
      setFile(null);
      setConsent(false);
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "Transcrição indisponível.",
      );
    } finally {
      setBusy(false);
    }
  }
  return (
    <fieldset disabled={disabled || busy || !available} className="audio">
      <legend>Ou envie um áudio</legend>
      {!available && (
        <p className="muted small">
          Transcrição ainda não configurada. Use o campo de texto.
        </p>
      )}
      <input
        aria-label="Arquivo de áudio"
        type="file"
        accept="audio/webm,audio/ogg,audio/wav,audio/mpeg,audio/mp4"
        onChange={(e) => {
          setFile(e.target.files?.[0] ?? null);
          setConsent(false);
          setError("");
        }}
      />
      <label>
        <input
          type="checkbox"
          checked={consent}
          onChange={(e) => setConsent(e.target.checked)}
        />
        Autorizo enviar este áudio ao serviço de transcrição configurado. Vou
        revisar o texto antes de encaminhar.
      </label>
      <button
        type="button"
        className="button secondary"
        disabled={!file || !consent || busy}
        onClick={transcribe}
      >
        {busy ? "Transcrevendo…" : "Transcrever áudio"}
      </button>
      {error && <p role="alert">{error}</p>}
    </fieldset>
  );
}
