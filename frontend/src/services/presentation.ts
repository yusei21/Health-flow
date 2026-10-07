import type { CareLevel, ServiceType } from "../types/routing";

// Display-only labels; the API values themselves are never changed.
const CARE_LEVEL_LABELS: Record<CareLevel, string> = {
  PRIMARY_CARE: "Atenção primária / UBS",
  URGENT_CARE: "Atendimento de urgência / UPA",
  EMERGENCY: "Emergência",
};

const SERVICE_TYPE_LABELS: Record<ServiceType, string> = {
  UBS: "Unidade Básica de Saúde (UBS)",
  UPA: "Unidade de Pronto Atendimento (UPA)",
  EMERGENCY_ROOM: "Pronto-socorro",
};

export function careLevelLabel(level: CareLevel): string {
  return CARE_LEVEL_LABELS[level] ?? level;
}

export function serviceTypeLabel(type: ServiceType): string {
  return SERVICE_TYPE_LABELS[type] ?? type;
}

export function isEmergency(level: CareLevel): boolean {
  return level === "EMERGENCY";
}

export function formatDistance(km: number): string {
  if (!Number.isFinite(km) || km < 0) return "";
  if (km < 1) return `${Math.round(km * 1000)} m`;
  return `${km.toLocaleString("pt-BR", { maximumFractionDigits: 1 })} km`;
}

export function formatAccuracy(meters: number): string {
  return `precisão aproximada de ${Math.round(meters).toLocaleString("pt-BR")} m`;
}
