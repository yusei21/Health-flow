import type { UserLocation } from "../types/routing";

export type LocationErrorCode = "unsupported" | "denied" | "unavailable" | "timeout";

export class LocationError extends Error {
  readonly code: LocationErrorCode;

  constructor(code: LocationErrorCode) {
    super(LOCATION_ERROR_MESSAGES[code]);
    this.name = "LocationError";
    this.code = code;
  }
}

export const LOCATION_ERROR_MESSAGES: Record<LocationErrorCode, string> = {
  unsupported: "Este navegador não oferece localização. Use um navegador atualizado.",
  denied:
    "Permissão de localização negada. Para buscar uma unidade próxima, permita o acesso à localização nas configurações do navegador e tente novamente.",
  unavailable: "Não foi possível obter sua localização. Você pode tentar novamente.",
  timeout: "A localização demorou demais para responder. Você pode tentar novamente.",
};

// GeolocationPositionError codes (W3C Geolocation API).
const PERMISSION_DENIED = 1;
const TIMEOUT = 3;

function codeFor(error: GeolocationPositionError): LocationErrorCode {
  if (error.code === PERMISSION_DENIED) return "denied";
  if (error.code === TIMEOUT) return "timeout";
  return "unavailable";
}

/**
 * One-shot location read, only ever called from an explicit user click.
 * No watchPosition, no caching (maximumAge: 0), nothing persisted.
 */
export function getCurrentLocation(
  geolocation: Pick<Geolocation, "getCurrentPosition"> | undefined = globalThis.navigator?.geolocation,
): Promise<UserLocation> {
  return new Promise((resolve, reject) => {
    if (!geolocation) {
      reject(new LocationError("unsupported"));
      return;
    }
    geolocation.getCurrentPosition(
      (position) =>
        resolve({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          accuracy: position.coords.accuracy,
        }),
      (error) => reject(new LocationError(codeFor(error))),
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 },
    );
  });
}
