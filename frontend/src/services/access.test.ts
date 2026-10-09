import { afterEach, describe, expect, it, vi } from "vitest";
import { askSUS, getPatientRecord, transcribeAudio } from "./api";

const config = { baseUrl: "https://api.test", token: "demo-token" };
afterEach(() => vi.unstubAllGlobals());

function mockFetch(body: unknown = {}) {
  const fn = vi.fn().mockResolvedValue(
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }),
  );
  vi.stubGlobal("fetch", fn);
  return fn;
}

describe("authorized access", () => {
  it("requests only the current user's record with explicit consent and no cache", async () => {
    const fetchFn = mockFetch({ display_name: "fictício" });
    await getPatientRecord(config);
    expect(fetchFn.mock.calls[0][0]).toBe(
      "https://api.test/api/v1/patients/me?consent=true",
    );
    expect(fetchFn.mock.calls[0][1].headers.Authorization).toBe(
      "Bearer demo-token",
    );
    expect(fetchFn.mock.calls[0][1].cache).toBe("no-store");
  });

  it("sends audio separately, without automatically requesting a care decision", async () => {
    const fetchFn = mockFetch({ text: "Estou com dor" });
    const audio = new File(["audio"], "report.webm", { type: "audio/webm" });
    await expect(transcribeAudio(audio, config)).resolves.toEqual({
      text: "Estou com dor",
    });
    expect(fetchFn).toHaveBeenCalledTimes(1);
    expect(fetchFn.mock.calls[0][0]).toContain(
      "/audio/transcribe?consent=true",
    );
    expect(fetchFn.mock.calls[0][1].body).toBe(audio);
  });

  it("does not attach the patient record to general SUS questions", async () => {
    const fetchFn = mockFetch({ status: "not_verified", sources: [] });
    await askSUS("medicine", "Onde consultar este medicamento?", config);
    expect(JSON.parse(fetchFn.mock.calls[0][1].body)).toEqual({
      topic: "medicine",
      question: "Onde consultar este medicamento?",
    });
  });
});
