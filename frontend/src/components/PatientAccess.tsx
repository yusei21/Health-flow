import { useEffect, useRef, useState } from "react";
import { getCapabilities, getPatientRecord } from "../services/api";
import { readApiConfig } from "../services/config";
import type { Capabilities, PatientRecord } from "../types/patient";

export function PatientAccess({
  onAccess,
}: {
  onAccess: (ready: boolean, consent: boolean) => void;
}) {
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [entered, setEntered] = useState(false);
  const [record, setRecord] = useState<PatientRecord | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const generation = useRef(0);
  useEffect(() => {
    let active = true;
    getCapabilities(readApiConfig())
      .then((value) => {
        if (active) setCapabilities(value);
      })
      .catch(() => {
        if (active)
          setError(
            "Não foi possível verificar o acesso. Confira o servidor e recarregue.",
          );
      });
    return () => {
      active = false;
      generation.current++;
    };
  }, []);
  async function authorize() {
    const current = ++generation.current;
    setBusy(true);
    setError("");
    try {
      const value = await getPatientRecord(readApiConfig());
      if (current !== generation.current) return;
      setRecord(value);
      onAccess(true, true);
    } catch (error) {
      if (current === generation.current)
        setError(
          error instanceof Error ? error.message : "Consulta indisponível.",
        );
    } finally {
      if (current === generation.current) setBusy(false);
    }
  }
  function revoke() {
    generation.current++;
    setBusy(false);
    setRecord(null);
    setError("");
    onAccess(true, false);
  }
  return (
    <section className="access" aria-labelledby="access-title">
      <h2 id="access-title">1. Acesso e prontuário</h2>
      {!entered ? (
        <>
          <p>
            Entre para começar e escolha se autoriza a consulta ao seu
            histórico.
          </p>
          <button className="button primary" disabled>
            Entrar com gov.br — em preparação
          </button>
          <p className="muted">
            A integração gov.br e o prontuário real ainda não estão disponíveis.
          </p>
          <button
            className="button secondary"
            disabled={!capabilities?.demo_available || !readApiConfig().token}
            onClick={() => {
              setEntered(true);
              onAccess(true, false);
            }}
          >
            Entrar na demonstração fictícia
          </button>
        </>
      ) : (
        <>
          <p className="tag">
            Demonstração: paciente fictício, sem acesso ao SUS real
          </p>
          <p>
            Você autoriza consultar idade, alergias, condições e medicamentos do
            perfil fictício para contextualizar somente o encaminhamento? O
            histórico não será enviado ao modelo de linguagem. Você pode
            continuar sem ele e retirar a autorização.
          </p>
          <div className="actions">
            <button
              className="button primary"
              disabled={busy || !!record}
              onClick={authorize}
            >
              {busy
                ? "Consultando…"
                : "Autorizar e consultar prontuário fictício"}
            </button>
            <button className="button secondary" onClick={revoke}>
              {record || busy
                ? "Retirar autorização"
                : "Continuar sem prontuário"}
            </button>
          </div>
          {record && (
            <div className="record" aria-live="polite">
              <h3>Perfil fictício carregado</h3>
              <p>
                {record.display_name} · {record.age ?? "Idade não informada"}{" "}
                anos
              </p>
              <p>
                Alergias: {record.allergies.join(", ") || "Não registradas"}
              </p>
              <p>
                Condições registradas:{" "}
                {record.conditions.map((x) => x.name).join(", ") ||
                  "Não registradas"}
              </p>
              <p>
                Medicamentos:{" "}
                {record.active_medications.map((x) => x.name).join(", ") ||
                  "Não registrados"}
              </p>
            </div>
          )}
        </>
      )}
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
    </section>
  );
}
