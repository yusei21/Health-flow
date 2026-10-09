import { useState, type FormEvent } from "react";
import { askSUS } from "../services/api";
import { readApiConfig } from "../services/config";
import type { SUSAnswer } from "../types/patient";

export function SUSQuestions() {
  const [topic, setTopic] = useState<"medicine" | "procedure">("procedure");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<SUSAnswer | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (busy || question.trim().length < 3) return;
    setBusy(true);
    setAnswer(null);
    setError("");
    try {
      setAnswer(await askSUS(topic, question.trim(), readApiConfig()));
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "Consulta indisponível.",
      );
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="sus-questions">
      <h2>Consultar serviços e medicamentos no SUS</h2>
      <p>
        Esta versão orienta onde consultar. Ainda não confirma itens específicos
        ou estoque local.
      </p>
      <form onSubmit={submit}>
        <label htmlFor="topic">Tipo de consulta</label>
        <select
          id="topic"
          value={topic}
          disabled={busy}
          onChange={(e) => {
            setTopic(e.target.value as typeof topic);
            setAnswer(null);
          }}
        >
          <option value="procedure">Exame ou tratamento</option>
          <option value="medicine">Medicamento e local de retirada</option>
        </select>
        <label htmlFor="sus-question">Sua pergunta</label>
        <input
          id="sus-question"
          maxLength={500}
          value={question}
          disabled={busy}
          onChange={(e) => {
            setQuestion(e.target.value);
            setAnswer(null);
          }}
          placeholder="Ex.: como consultar a oferta deste exame no SUS?"
        />
        <button
          className="button secondary"
          disabled={busy || question.trim().length < 3}
        >
          {busy ? "Consultando…" : "Ver orientação e fontes"}
        </button>
      </form>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      {answer && (
        <div className="result" aria-live="polite">
          <strong>Disponibilidade não verificada</strong>
          <p>{answer.answer}</p>
          <ul>
            {answer.sources.map((source) => (
              <li key={source.url}>
                <a href={source.url} target="_blank" rel="noreferrer">
                  {source.title}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
