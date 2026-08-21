import type { CSSProperties } from "react";

type Props = {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  valueLabel: string;
  lowLabel: string;
  highLabel: string;
  onChange: (value: number) => void;
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function DisplaySlider({
  id,
  label,
  value,
  min,
  max,
  step,
  valueLabel,
  lowLabel,
  highLabel,
  onChange,
}: Props) {
  const progress = ((value - min) / (max - min)) * 100;
  const rangeStyle = { "--range-progress": `${progress}%` } as CSSProperties;
  const update = (raw: string) => {
    const parsed = Number(raw);
    if (Number.isFinite(parsed)) onChange(clamp(parsed, min, max));
  };

  return (
    <div className="display-control">
      <div className="display-control-heading">
        <label htmlFor={id}>{label}</label>
        <output htmlFor={id}>{valueLabel}</output>
      </div>
      <div className="display-control-row">
        <input
          id={id}
          className="display-range"
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          style={rangeStyle}
          aria-valuetext={valueLabel}
          onChange={(event) => update(event.target.value)}
        />
        <input
          className="display-number"
          type="number"
          min={min}
          max={max}
          step={step}
          value={value}
          aria-label={`${label} exact value`}
          onChange={(event) => update(event.target.value)}
        />
      </div>
      <div className="range-scale" aria-hidden="true">
        <span>{lowLabel}</span>
        <span>{highLabel}</span>
      </div>
    </div>
  );
}
