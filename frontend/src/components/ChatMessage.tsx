import { useState } from "react";
import type { ChatEntry } from "../types";

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export default function ChatMessage({ entry }: { entry: ChatEntry }) {
  const [traceOpen, setTraceOpen] = useState(false);

  if (entry.role === "user") {
    return (
      <div className="entry entry--user">
        <span className="eyebrow">You · {formatTime(entry.timestamp)}</span>
        <p className="entry__text">{entry.content}</p>
      </div>
    );
  }

  if (entry.role === "error") {
    return (
      <div className="entry entry--error">
        <span className="eyebrow">Request failed · {formatTime(entry.timestamp)}</span>
        <p className="entry__text">{entry.content}</p>
      </div>
    );
  }

  return (
    <div className="report-slip">
      <div className="report-slip__perforation" aria-hidden="true" />
      <div className="report-slip__body">
        <span className="eyebrow">
          Agent reply{entry.topK ? ` · top_k ${entry.topK}` : ""} · {formatTime(entry.timestamp)}
        </span>
        <p className="entry__text">{entry.content}</p>
        {entry.raw !== undefined && (
          <>
            <button
              type="button"
              className="report-slip__trace-toggle"
              aria-expanded={traceOpen}
              onClick={() => setTraceOpen((open) => !open)}
            >
              {traceOpen ? "▾ Hide trace" : "▸ View trace"}
            </button>
            {traceOpen && <pre className="report-slip__trace mono">{JSON.stringify(entry.raw, null, 2)}</pre>}
          </>
        )}
      </div>
    </div>
  );
}
