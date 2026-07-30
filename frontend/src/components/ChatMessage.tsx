import { useState } from "react";
import type { ChatEntry } from "../types";
import Markdown from "./Markdown";

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export default function ChatMessage({ entry }: { entry: ChatEntry }) {
  const [traceOpen, setTraceOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  async function copyAnswer() {
    try {
      await navigator.clipboard.writeText(entry.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked — nothing to do */
    }
  }

  if (entry.role === "user") {
    return (
      <div className="entry entry--user">
        <span className="eyebrow">YOU · {formatTime(entry.timestamp)}</span>
        <p className="entry__text">&gt; {entry.content}</p>
      </div>
    );
  }

  if (entry.role === "error") {
    return (
      <div className="entry entry--error">
        <span className="eyebrow">✕ REQUEST FAILED · {formatTime(entry.timestamp)}</span>
        <p className="entry__text">{entry.content}</p>
      </div>
    );
  }

  return (
    <div className="report-slip">
      <div className="report-slip__tab" aria-hidden="true">
        AGENT
      </div>
      <div className="report-slip__body">
        <div className="report-slip__head">
          <span className="eyebrow">
            REPLY{entry.topK ? ` · ${entry.topK} cases` : ""} · {formatTime(entry.timestamp)}
          </span>
          <button type="button" className="report-slip__copy" onClick={copyAnswer}>
            {copied ? "✓ Copied" : "Copy"}
          </button>
        </div>
        <Markdown content={entry.content} />
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
            {traceOpen && <pre className="report-slip__trace term">{JSON.stringify(entry.raw, null, 2)}</pre>}
          </>
        )}
      </div>
    </div>
  );
}
