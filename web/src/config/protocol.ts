export type ProtocolLabel = {
  value: number;
  name: string;
  color: string;
  laterality: "left" | "right" | null;
  region: string | null;
};

export type ProtocolDefinition = {
  schema_version: number;
  protocol_version: string;
  training_eligibility: string;
  labels: ProtocolLabel[];
};

const HEX_COLOR = /^#[0-9A-Fa-f]{6}$/;

export function validateProtocol(value: ProtocolDefinition): ProtocolDefinition {
  const expectedValues = [0, 1, 2, 3, 4];
  if (
    value.schema_version !== 1 ||
    !value.protocol_version ||
    value.training_eligibility !== "PROVISIONAL_NOT_FOR_TRAINING" ||
    value.labels.length !== expectedValues.length ||
    value.labels.some(
      (label, index) =>
        label.value !== expectedValues[index] ||
        !label.name ||
        !HEX_COLOR.test(label.color),
    )
  ) {
    throw new Error("The server returned an unsupported annotation protocol.");
  }
  return value;
}

export function annotationLabels(protocol: ProtocolDefinition): ProtocolLabel[] {
  return protocol.labels.filter((label) => label.value !== 0);
}
