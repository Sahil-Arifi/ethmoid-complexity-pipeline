import { describe, expect, it } from "vitest";
import { annotationLabels, validateProtocol, type ProtocolDefinition } from "./protocol";

const protocol: ProtocolDefinition = {
  schema_version: 1,
  protocol_version: "ethmoid-provisional-v1",
  training_eligibility: "PROVISIONAL_NOT_FOR_TRAINING",
  labels: [
    { value: 0, name: "Background", color: "#000000", laterality: null, region: null },
    { value: 1, name: "Left anterior ethmoid", color: "#2F80ED", laterality: "left", region: "anterior_ethmoid" },
    { value: 2, name: "Right anterior ethmoid", color: "#EB5757", laterality: "right", region: "anterior_ethmoid" },
    { value: 3, name: "Left posterior ethmoid", color: "#56CCF2", laterality: "left", region: "posterior_ethmoid" },
    { value: 4, name: "Right posterior ethmoid", color: "#F2994A", laterality: "right", region: "posterior_ethmoid" },
  ],
};

describe("server-provided annotation protocol", () => {
  it("accepts the versioned four-label provisional protocol", () => {
    expect(validateProtocol(protocol)).toBe(protocol);
    expect(annotationLabels(protocol).map((label) => label.value)).toEqual([1, 2, 3, 4]);
    expect(annotationLabels(protocol).map((label) => label.color)).toEqual([
      "#2F80ED",
      "#EB5757",
      "#56CCF2",
      "#F2994A",
    ]);
  });

  it("rejects a protocol that could produce invalid mask values", () => {
    expect(() =>
      validateProtocol({
        ...protocol,
        labels: protocol.labels.map((label, index) => ({ ...label, value: index + 1 })),
      }),
    ).toThrow("unsupported annotation protocol");
  });
});
