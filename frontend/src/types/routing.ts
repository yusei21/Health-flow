// Mirrors app/schemas/routing.py and app/schemas/care.py. Keep in sync with the API.

export type CareLevel = "PRIMARY_CARE" | "URGENT_CARE" | "EMERGENCY";
export type ServiceType = "UBS" | "UPA" | "EMERGENCY_ROOM";

export interface RoutingRequest {
  message: string;
  latitude: number;
  longitude: number;
}

export interface FacilityResponse {
  name: string;
  service_type: ServiceType;
  address: string;
  latitude: number;
  longitude: number;
  distance_km: number;
  is_simulated: boolean;
}

export interface RoutingResponse {
  request_id: string;
  care_level: CareLevel;
  recommended_service_type: ServiceType;
  facility: FacilityResponse | null;
  next_step: string;
  emergency_guidance: string | null;
  reason_codes: string[];
  safety_override: boolean;
  disclaimer: string;\n  needs_more_information?: boolean;\n  follow_up_questions?: string[];
}

/** Body of 401/404/503/500 responses (app/api/errors.py). */
export interface ApiErrorBody {
  error: string;
  emergency_hint: string;
  request_id: string | null;
}

export interface UserLocation {
  latitude: number;
  longitude: number;
  accuracy: number;
}

/** Limits enforced by RoutingRequest on the backend. */
export const MESSAGE_MIN_LENGTH = 3;
export const MESSAGE_MAX_LENGTH = 2000;
