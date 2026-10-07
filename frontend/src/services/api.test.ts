import { describe, expect, it, vi } from "vitest";
import type { RoutingResponse } from "../types/routing";
import { ApiError, requestRouting } from "./api";

const CONFIG = { baseUrl: "http://api.test", token: "demo-token" };
const REQUEST = { message: "dor de cabeça", latitude: -23.55, longitude: -46.64 };
const RESPONSE: RoutingResponse = {
  request_id: "r1",
  care_level: "PRIMARY_CARE",
  recommended_service_type: "UBS",
  facility: null,
  next_step: "Procure uma UBS.",
  emergency_guidance: null,
  reason_codes: [],
  safety_override: false,
  disclaimer: "aviso",
};

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("requestRouting", () => {
  it("posts the exact API contract with the bearer token", async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse(200, RESPONSE));

    await expect(requestRouting(REQUEST, CONFIG, fetchFn)).resolves.toEqual(RESPONSE);

    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe("http://api.test/api/v1/routing");
    expect(init.method).toBe("POST");
    expect(init.headers.Authorization).toBe("Bearer demo-token");
    expect(JSON.parse(init.body)).toEqual(REQUEST);
  });

  it("omits Authorization when no token is configured", async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse(200, RESPONSE));
    await requestRouting(REQUEST, { ...CONFIG, token: "" }, fetchFn);
    expect(fetchFn.mock.calls[0][1].headers).not.toHaveProperty("Authorization");
  });

  it("maps a 401 to a friendly error and keeps the backend emergency hint", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse(401, { error: "x", emergency_hint: "Ligue 192 (SAMU).", request_id: "r" }),
    );
    const error = await requestRouting(REQUEST, CONFIG, fetchFn).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(401);
    expect((error as ApiError).message).toMatch(/token/);
    expect((error as ApiError).emergencyHint).toBe("Ligue 192 (SAMU).");
  });

  it("maps 422 validation errors without exposing raw details", async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse(422, { detail: [{ loc: ["body"] }] }));
    const error = (await requestRouting(REQUEST, CONFIG, fetchFn).catch((e: unknown) => e)) as ApiError;
    expect(error.status).toBe(422);
    expect(error.emergencyHint).toMatch(/192/);
  });

  it("reports network failures as connection errors", async () => {
    const fetchFn = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));
    const error = (await requestRouting(REQUEST, CONFIG, fetchFn).catch((e: unknown) => e)) as ApiError;
    expect(error.status).toBeNull();
    expect(error.message).toMatch(/conectar/);
  });
});
