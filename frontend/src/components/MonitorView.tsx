import { useEffect, useState } from "react";
import { fetchTimeseries, injectConfigured, injectTransactions } from "../api/client";
import type { TimeseriesResponse } from "../types";
import MiniBarChart from "./MiniBarChart";
import "./MonitorView.css";

function floor5(d: Date): Date {
  const dt = new Date(d);
  dt.setSeconds(0, 0);
  dt.setMinutes(Math.floor(dt.getMinutes() / 5) * 5);
  return dt;
}

function ceil5(d: Date): Date {
  const dt = new Date(d);
  dt.setSeconds(0, 0);
  const remainder = dt.getMinutes() % 5;
  dt.setMinutes(dt.getMinutes() + (remainder !== 0 ? 5 - remainder : 5));
  return dt;
}

function toInputValue(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export default function MonitorView() {
  const [start, setStart] = useState(() => {
    const n = floor5(new Date());
    n.setMinutes(n.getMinutes() - 30);
    return n;
  });
  const [end, setEnd] = useState(() => ceil5(new Date()));
  const [endLocked, setEndLocked] = useState(false);
  const [result, setResult] = useState<TimeseriesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);

  const [duration, setDuration] = useState(1);
  const [injecting, setInjecting] = useState(false);
  const [injectStatus, setInjectStatus] = useState<{ ok: boolean; message: string } | null>(null);

  useEffect(() => {
    function tick() {
      if (!endLocked) setEnd(ceil5(new Date()));
    }
    const id = setInterval(tick, 30_000);
    return () => clearInterval(id);
  }, [endLocked]);

  useEffect(() => {
    if (start >= end) {
      setError("Start must be before end.");
      setResult(null);
      return;
    }
    let active = true;
    fetchTimeseries(start, end)
      .then((r) => {
        if (!active) return;
        setResult(r);
        setError(null);
        setLastRefreshed(new Date());
      })
      .catch((e) => {
        if (!active) return;
        setError(e instanceof Error ? e.message : "Failed to fetch.");
      });
    return () => {
      active = false;
    };
  }, [start, end]);

  async function handleInject(e: React.FormEvent) {
    e.preventDefault();
    setInjecting(true);
    setInjectStatus(null);
    try {
      await injectTransactions(duration);
      setInjectStatus({ ok: true, message: `Injected ${duration}s of transactions — chart updates on next refresh.` });
    } catch (err) {
      setInjectStatus({ ok: false, message: err instanceof Error ? err.message : "Inject failed." });
    } finally {
      setInjecting(false);
    }
  }

  const totalTx = result?.total_transactions ?? 0;
  const totalFraud = result?.total_fraud ?? 0;
  const totalNormal = result?.total_normal ?? 0;
  const fraudRate = totalTx > 0 ? ((totalFraud / totalTx) * 100).toFixed(1) : "0.0";
  const buckets = result?.data ?? [];
  const ticks = buckets.map((b) => b.bucket);

  return (
    <section className="monitor-view">
      <div className="monitor-row monitor-row--range">
        <label className="monitor-field">
          <span className="eyebrow">Start</span>
          <input
            type="datetime-local"
            step={300}
            value={toInputValue(start)}
            onChange={(e) => e.target.value && setStart(new Date(e.target.value))}
          />
        </label>
        <span className="monitor-arrow" aria-hidden="true">
          →
        </span>
        <label className="monitor-field">
          <span className="eyebrow">End</span>
          <input
            type="datetime-local"
            step={300}
            value={toInputValue(end)}
            onChange={(e) => {
              if (!e.target.value) return;
              setEnd(new Date(e.target.value));
              setEndLocked(true);
            }}
          />
        </label>
        <button
          type="button"
          className="monitor-now"
          onClick={() => {
            setEnd(ceil5(new Date()));
            setEndLocked(false);
          }}
        >
          Now
        </button>
      </div>

      {error ? (
        <p className="monitor-error">{error}</p>
      ) : (
        <p className="monitor-caption">
          Last refreshed {lastRefreshed ? lastRefreshed.toLocaleTimeString() : "—"} · auto-refreshes every 30s
          {endLocked ? " · end time locked" : ""}
        </p>
      )}

      <div className="monitor-stats">
        <div className="monitor-stat">
          <span className="monitor-stat__value mono">{totalTx.toLocaleString()}</span>
          <span className="eyebrow">Transactions</span>
        </div>
        <div className="monitor-stat">
          <span className="monitor-stat__value mono" style={{ color: "var(--alert)" }}>
            {totalFraud.toLocaleString()}
          </span>
          <span className="eyebrow">Fraud</span>
        </div>
        <div className="monitor-stat">
          <span className="monitor-stat__value mono" style={{ color: "var(--verified)" }}>
            {totalNormal.toLocaleString()}
          </span>
          <span className="eyebrow">Normal</span>
        </div>
        <div className="monitor-stat">
          <span className="monitor-stat__value mono">{fraudRate}%</span>
          <span className="eyebrow">Fraud rate</span>
        </div>
      </div>

      <div className="monitor-charts">
        <MiniBarChart
          label="Transactions / min"
          values={buckets.map((b) => b.transactions)}
          ticks={ticks}
          color="var(--ink-soft)"
        />
        <MiniBarChart label="Fraud" values={buckets.map((b) => b.fraud)} ticks={ticks} color="var(--alert)" />
        <MiniBarChart label="Normal" values={buckets.map((b) => b.normal)} ticks={ticks} color="var(--verified)" />
      </div>

      <form className="monitor-inject" onSubmit={handleInject}>
        <span className="eyebrow">Inject messages</span>
        <div className="monitor-inject__row">
          <label className="monitor-field monitor-field--inline">
            <span className="eyebrow">Duration (s)</span>
            <input
              type="number"
              min={1}
              max={300}
              value={duration}
              onChange={(e) => setDuration(Math.min(300, Math.max(1, Number(e.target.value) || 1)))}
            />
          </label>
          <button type="submit" disabled={injecting || !injectConfigured()} title={!injectConfigured() ? "No producer configured (VITE_INJECT_URL is unset)." : undefined}>
            {injecting ? "Injecting…" : "Inject"}
          </button>
          {injectStatus && (
            <span className={injectStatus.ok ? "monitor-inject__success" : "monitor-inject__failure"}>
              {injectStatus.message}
            </span>
          )}
        </div>
      </form>
    </section>
  );
}
