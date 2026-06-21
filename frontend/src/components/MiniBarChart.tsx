interface MiniBarChartProps {
  label: string;
  values: number[];
  ticks: string[];
  color: string;
}

export default function MiniBarChart({ label, values, ticks, color }: MiniBarChartProps) {
  const max = Math.max(1, ...values);

  return (
    <div className="bar-chart">
      <span className="eyebrow">{label}</span>
      <div className="bar-chart__plot">
        {values.map((v, i) => (
          <div className="bar-chart__col" key={i} title={`${ticks[i]} · ${v}`}>
            <div
              className="bar-chart__bar"
              style={{ height: `${(v / max) * 100}%`, background: color }}
            />
          </div>
        ))}
      </div>
      <div className="bar-chart__ticks">
        <span>{ticks[0] ?? "—"}</span>
        <span>{ticks[ticks.length - 1] ?? "—"}</span>
      </div>
    </div>
  );
}
