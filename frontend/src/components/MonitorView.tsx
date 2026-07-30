import { useCallback, useEffect, useState } from "react";
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

const RANGES: { label: string; minutes: number }[] = [
  { label: "15m", minutes: 15 },
  { label: "1h", minutes: 60 },
  { label: "6h", minutes: 360 },
  { label: "24h", minutes: 1440 },
];

export default function MonitorView() {
  const [rangeMinutes, setRangeMinutes] = useState(30);
  const [start, setStart] = useState(() => {
    const n = floor5(new Date());
    n.setMinutes(n.getMinutes() - 30);
    return n;
  });
  const [end, setEnd] = useState(() => ceil5(new Date()));
  const [endLocked, setEndLocked] = useState(false);
  const [result, setResult] = useState<TimeseriesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  const [duration, setDuration] = useState(1);
  const [injecting, setInjecting] = useState(false);
  const [injectStatus, setInjectStatus] = useState<{ ok: boolean; message: string } | null>(null);

  function applyRange(minutes: number) {
    const e = ceil5(new Date());
    const s = new Date(e);
    s.setMinutes(s.getMinutes() - minutes);
    setRangeMinutes(minutes);
    setStart(floor5(s));
    setEnd(e);
    setEndLocked(false);
  }

  const refresh = useCallback(() => {
    if (!endLocked) {
      const e = ceil5(new Date());
      const s = new Date(e);
      s.setMinutes(s.getMinutes() - rangeMinutes);
      setStart(floor5(s));
      setEnd(e);
    }
    setReloadKey((k) => k + 1);
  }, [endLocked, rangeMinutes]);

  // Auto-refresh a live (unlocked) range every 30s.
  useEffect(() => {
    if (endLocked) return;
    const id = setInterval(refresh, 30_000);
    return () => clearInterval(id);
  }, [endLocked, refresh]);

  useEffect(() => {
    if (start >= end) {
      setError("Start must be before end.");
      setResult(null);
      return;
    }
    let active = true;
    setLoading(true);
    fetchTimeseries(start, end)
      .then((r) => {
        if (!active) return;
        setResult(r);
        setError(null);
        setLastRefreshed(new Date());
      })
      .catch((e) => {
        if (!active) return;
        setError(e instanceof Error ? e.message : "Couldn't load transactions. Check the API is running.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [start, end, reloadKey]);

  async function handleInject(e: React.FormEvent) {
    e.preventDefault();
    setInjecting(true);
    setInjectStatus(null);
    try {
      await injectTransactions(duration);
      setInjectStatus({ ok: true, message: `Injected ${duration}s of traffic — refresh to see it land.` });
    } catch (err) {
      setInjectStatus({ ok: false, message: err instanceof Error ? err.message : "Inject failed." });
    } finally {
      setInjecting(false);
    }
  }

  const totalTx = result?.total_transactions ?? 0;
  const totalFraud = result?.total_fraud ?? 0;
  const totalNormal = result?.total_normal ?? 0;
  const fraudRateNum = totalTx > 0 ? (totalFraud / totalTx) * 100 : 0;
  const fraudRate = fraudRateNum.toFixed(1);
  const fraudColor = fraudRateNum >= 5 ? "var(--red)" : fraudRateNum >= 1 ? "var(--amber)" : "var(--green)";
  const buckets = result?.data ?? [];
  const ticks = buckets.map((b) => b.bucket);
  const hasData = buckets.length > 0;

  return (
    <section className="monitor-view">
      <div className="monitor-toolbar">
        <div className="monitor-presets" role="group" aria-label="Quick ranges">
          {RANGES.map((r) => (
            <button
              key={r.label}
              type="button"
              className={`monitor-preset ${!endLocked && rangeMinutes === r.minutes ? "is-active" : ""}`}
              onClick={() => applyRange(r.minutes)}
            >
              {r.label}
            </button>
          ))}
        </div>
        <button type="button" className="monitor-refresh" onClick={refresh} disabled={loading}>
          {loading ? "⟳ Loading…" : "⟳ Refresh"}
        </button>
      </div>

      <div className="monitor-row monitor-row--range">
        <label className="monitor-field">
          <span className="eyebrow">Start</span>
          <input
            type="datetime-local"
            step={300}
            value={toInputValue(start)}
            onChange={(e) => e.target.value && (setStart(new Date(e.target.value)), setEndLocked(true))}
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
      </div>

      {error ? (
        <p className="monitor-error">✕ {error}</p>
      ) : (
        <p className="monitor-caption">
          {lastRefreshed ? `Updated ${lastRefreshed.toLocaleTimeString()}` : "Loading…"}
          {endLocked ? " · fixed range" : " · live, auto-refreshing every 30s"}
        </p>
      )}

      <div className="monitor-stats">
        <div className="monitor-stat">
          <span className="monitor-stat__value">{totalTx.toLocaleString()}</span>
          <span className="eyebrow">Transactions</span>
        </div>
        <div className="monitor-stat">
          <span className="monitor-stat__value" style={{ color: "var(--red)" }}>
            {totalFraud.toLocaleString()}
          </span>
          <span className="eyebrow">Fraud</span>
        </div>
        <div className="monitor-stat">
          <span className="monitor-stat__value" style={{ color: "var(--green)" }}>
            {totalNormal.toLocaleString()}
          </span>
          <span className="eyebrow">Normal</span>
        </div>
        <div className="monitor-stat">
          <span className="monitor-stat__value" style={{ color: fraudColor }}>
            {fraudRate}%
          </span>
          <span className="eyebrow">Fraud rate</span>
        </div>
      </div>

      {hasData ? (
        <div className="monitor-charts">
          <MiniBarChart label="Transactions / min" values={buckets.map((b) => b.transactions)} ticks={ticks} color="var(--cyan)" />
          <MiniBarChart label="Fraud" values={buckets.map((b) => b.fraud)} ticks={ticks} color="var(--red)" />
          <MiniBarChart label="Normal" values={buckets.map((b) => b.normal)} ticks={ticks} color="var(--green)" />
        </div>
      ) : (
        <div className="monitor-empty">
          <span className="eyebrow">▚ NO SIGNAL</span>
          <p>No transactions in this range. Widen the window, or inject some traffic below to see the console light up.</p>
        </div>
      )}

      <form className="monitor-inject" onSubmit={handleInject}>
        <span className="eyebrow">Inject test traffic</span>
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
          <button
            type="submit"
            className="monitor-inject__button"
            disabled={injecting || !injectConfigured()}
            title={!injectConfigured() ? "No producer configured (VITE_INJECT_URL is unset)." : undefined}
          >
            {injecting ? "Injecting…" : "Inject ▶"}
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
