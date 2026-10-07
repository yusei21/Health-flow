import type { RoutingResponse } from "../types/routing";
import { careLevelLabel, formatDistance, isEmergency, serviceTypeLabel } from "../services/presentation";

export function ResultCard({ result }: { result: RoutingResponse }) {
  const emergency = !result.needs_more_information && isEmergency(result.care_level);
  const facility = result.facility;

  return (
    <section className={`result ${emergency ? "emergency" : ""}`} aria-live="polite">
      {emergency && (
        <div className="emergency-box" role="alert">
          <strong>Possível situação de emergência.</strong>
          {result.emergency_guidance && <p>{result.emergency_guidance}</p>}
        </div>
      )}

      {result.needs_more_information && (\n        <div role="status">\n          <strong>Informações insuficientes para determinar a gravidade.</strong>\n          <p>Responda às perguntas abaixo no relato e faça uma nova busca.</p>\n          <ul>{(result.follow_up_questions ?? []).map((question) => <li key={question}>{question}</li>)}</ul>\n        </div>\n      )}\n      <dl className="result-grid">
        <dt>Nível de atendimento</dt>
        <dd>{result.needs_more_information ? "Avaliação adicional necessária" : careLevelLabel(result.care_level)}</dd>
        <dt>Tipo de serviço</dt>
        <dd>{serviceTypeLabel(result.recommended_service_type)}</dd>
        <dt>Orientação</dt>
        <dd>{result.next_step}</dd>
      </dl>

      {facility && (
        <div className="facility">
          <h3>Unidade encontrada</h3>
          <p className="facility-name">
            {facility.name}
            {facility.is_simulated && <span className="tag">dados simulados</span>}
          </p>
          <p>{facility.address}</p>
          {formatDistance(facility.distance_km) && (
            <p className="muted">Distância aproximada: {formatDistance(facility.distance_km)}</p>
          )}
        </div>
      )}
    </section>
  );
}
