import type { RoutingResponse } from "../types/routing";
import { careLevelLabel, formatDistance, isEmergency, serviceTypeLabel } from "../services/presentation";

export function ResultCard({ result }: { result: RoutingResponse }) {
  const emergency = isEmergency(result.care_level);
  const facility = result.facility;

  return (
    <section className={`result ${emergency ? "emergency" : ""}`} aria-live="polite">
      {emergency && (
        <div className="emergency-box" role="alert">
          <strong>Possível situação de emergência.</strong>
          {result.emergency_guidance && <p>{result.emergency_guidance}</p>}
        </div>
      )}

      <dl className="result-grid">
        <dt>Nível de atendimento</dt>
        <dd>{careLevelLabel(result.care_level)}</dd>
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
