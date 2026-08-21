type Props = {
  compact?: boolean;
};

export function BrandMark({ compact = false }: Props) {
  return (
    <span className={compact ? "brand-mark compact" : "brand-mark"} aria-hidden="true">
      <span />
      <span />
      <span />
      <span />
    </span>
  );
}
