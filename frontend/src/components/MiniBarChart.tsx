interface MiniBarChartProps {
  label: string;
  values: number[];
  ticks: string[];
  color: string;
}

export default function MiniBarChart({ label, values, ticks, color }: MiniBarChartProps) {
  const peak = Math.max(0, ...values);
  const scale = Math.max(1, peak);
  const total = values.reduce((a, b) => a + b, 0);

  return (
    <div className="bar-chart">
      <div className="bar-chart__head">
        <span className="eyebrow">{label}</span>
        <span className="bar-chart__peak" style={{ color }}>
          {total.toLocaleString()}
        </span>
      </div>
      <div className="bar-chart__plot" style={{ "--bar-color": color } as React.CSSProperties}>
        {values.map((v, i) => (
          <div className="bar-chart__col" key={i} title={`${ticks[i]} · ${v}`}>
            <div className="bar-chart__bar" style={{ height: `${(v / scale) * 100}%` }} />
          </div>
        ))}
      </div>
      <div className="bar-chart__ticks">
        <span>{ticks[0] ?? "—"}</span>
        <span>peak {peak.toLocaleString()}</span>
        <span>{ticks[ticks.length - 1] ?? "—"}</span>
      </div>
    </div>
  );
}
