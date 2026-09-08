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
        <span className="eyebrow">{formatTime(entry.timestamp)}</span>
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
    <div className="finding">
      <div className="finding__head">
        <span className="eyebrow">
          {entry.topK ? `${entry.topK} similar cases · ` : ""}
          {formatTime(entry.timestamp)}
          {/* Short prefix is enough to find the trace and keeps the header
              readable; the full id is in the trace panel below. */}
          {entry.traceId ? ` · trace ${entry.traceId.slice(0, 8)}` : ""}
        </span>
        <button type="button" className="finding__copy" onClick={copyAnswer}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <Markdown content={entry.content} />
      {entry.raw !== undefined && (
        <>
          <button
            type="button"
            className="finding__trace-toggle"
            aria-expanded={traceOpen}
            onClick={() => setTraceOpen((open) => !open)}
          >
            {traceOpen ? "Hide trace" : "View trace"}
          </button>
          {traceOpen && <pre className="finding__trace data">{JSON.stringify(entry.raw, null, 2)}</pre>}
        </>
      )}
    </div>
  );
}
