import { useEffect, useState } from "react";
import { fetchStatusCounts } from "../api/client";
import type { StatusCountsResponse } from "../types";

interface StatusCountsStripProps {
  start: Date;
  end: Date;
  reloadKey: number;
}

const STAGES: { key: keyof StatusCountsResponse; label: string; color: string }[] = [
  { key: "received", label: "Received", color: "var(--ink-dim)" },
  { key: "flagged", label: "Flagged", color: "var(--brand)" },
  { key: "embedding", label: "Embedding", color: "var(--warn)" },
  { key: "embedded", label: "Embedded", color: "var(--normal)" },
];

export default function StatusCountsStrip({ start, end, reloadKey }: StatusCountsStripProps) {
  const [counts, setCounts] = useState<StatusCountsResponse | null>(null);

  useEffect(() => {
    if (start >= end) return;
    let active = true;
    fetchStatusCounts(start, end)
      .then((r) => {
        if (active) setCounts(r);
      })
      .catch(() => {
        if (active) setCounts(null);
      });
    return () => {
      active = false;
    };
  }, [start, end, reloadKey]);

  return (
    <div className="ledger-strip">
      {STAGES.map((stage) => (
        <div className="ledger-stat" key={stage.key}>
          <span className="ledger-stat__value data" style={{ color: stage.color }}>
            {(counts?.[stage.key] ?? 0).toLocaleString()}
          </span>
          <span className="eyebrow">{stage.label}</span>
        </div>
      ))}
    </div>
  );
}
