import { describe, expect, it } from "vitest";
import { careLevelLabel, formatDistance, isEmergency, serviceTypeLabel } from "./presentation";

describe("presentation", () => {
  it("translates care levels for display", () => {
    expect(careLevelLabel("PRIMARY_CARE")).toBe("Atenção primária / UBS");
    expect(careLevelLabel("URGENT_CARE")).toBe("Atendimento de urgência / UPA");
    expect(careLevelLabel("EMERGENCY")).toBe("Emergência");
  });

  it("translates service types for display", () => {
    expect(serviceTypeLabel("UBS")).toMatch(/UBS/);
    expect(serviceTypeLabel("UPA")).toMatch(/UPA/);
    expect(serviceTypeLabel("EMERGENCY_ROOM")).toBe("Pronto-socorro");
  });

  it("flags only EMERGENCY as emergency", () => {
    expect(isEmergency("EMERGENCY")).toBe(true);
    expect(isEmergency("URGENT_CARE")).toBe(false);
    expect(isEmergency("PRIMARY_CARE")).toBe(false);
  });

  it("formats distances", () => {
    expect(formatDistance(0.45)).toBe("450 m");
    expect(formatDistance(3.06)).toBe("3,1 km");
    expect(formatDistance(Number.NaN)).toBe("");
  });
});
