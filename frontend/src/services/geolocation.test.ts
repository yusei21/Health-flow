import { describe, expect, it } from "vitest";
import { getCurrentLocation, LocationError } from "./geolocation";

type Success = PositionCallback;
type Failure = PositionErrorCallback;

function fakeGeolocation(run: (ok: Success, fail: Failure) => void) {
  return {
    getCurrentPosition: (ok: Success, fail?: Failure | null) => run(ok, fail!),
  };
}

function positionError(code: number): GeolocationPositionError {
  return { code, message: "", PERMISSION_DENIED: 1, POSITION_UNAVAILABLE: 2, TIMEOUT: 3 };
}

describe("getCurrentLocation", () => {
  it("returns latitude, longitude and accuracy", async () => {
    const geo = fakeGeolocation((ok) =>
      ok({ coords: { latitude: -23.5, longitude: -46.6, accuracy: 12 } } as GeolocationPosition),
    );
    await expect(getCurrentLocation(geo)).resolves.toEqual({
      latitude: -23.5,
      longitude: -46.6,
      accuracy: 12,
    });
  });

  it("rejects when the browser has no geolocation", async () => {
    await expect(getCurrentLocation(undefined)).rejects.toMatchObject({ code: "unsupported" });
  });

  it.each([
    [1, "denied"],
    [2, "unavailable"],
    [3, "timeout"],
  ])("maps error code %i to %s", async (code, expected) => {
    const geo = fakeGeolocation((_, fail) => fail(positionError(code)));
    const error = await getCurrentLocation(geo).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(LocationError);
    expect((error as LocationError).code).toBe(expected);
    expect((error as LocationError).message).not.toBe("");
  });
});
