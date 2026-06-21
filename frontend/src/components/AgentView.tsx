import { useEffect, useRef, useState } from "react";
import { askAgent } from "../api/client";
import type { ChatEntry } from "../types";
import "./AgentView.css";
import ChatMessage from "./ChatMessage";

function makeId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

const EXAMPLE_PROMPT = "Any anomaly transactions over 1000 EUR in the last hour?";

export default function AgentView() {
  const [history, setHistory] = useState<ChatEntry[]>([]);
  const [prompt, setPrompt] = useState("");
  const [topK, setTopK] = useState(3);
  const [pending, setPending] = useState(false);
  const transcriptRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    transcriptRef.current?.scrollTo({ top: transcriptRef.current.scrollHeight, behavior: "smooth" });
  }, [history, pending]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = prompt.trim();
    if (!trimmed || pending) return;

    setHistory((h) => [...h, { id: makeId(), role: "user", content: trimmed, timestamp: new Date().toISOString() }]);
    setPrompt("");
    setPending(true);

    try {
      const { answer, raw } = await askAgent(trimmed, topK);
      setHistory((h) => [
        ...h,
        { id: makeId(), role: "assistant", content: answer, timestamp: new Date().toISOString(), topK, raw },
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error.";
      setHistory((h) => [...h, { id: makeId(), role: "error", content: message, timestamp: new Date().toISOString() }]);
    } finally {
      setPending(false);
    }
  }

  return (
    <section className="agent-view">
      <div className="agent-view__transcript" ref={transcriptRef}>
        {history.length === 0 && (
          <div className="agent-view__empty">
            <span className="eyebrow">Case file empty</span>
            <p>
              Ask the agent about flagged activity — it pulls similar past transactions and scores their features
              before answering.
            </p>
            <button type="button" className="agent-view__example" onClick={() => setPrompt(EXAMPLE_PROMPT)}>
              “{EXAMPLE_PROMPT}”
            </button>
          </div>
        )}
        {history.map((entry) => (
          <ChatMessage key={entry.id} entry={entry} />
        ))}
        {pending && (
          <div className="report-slip report-slip--pending">
            <div className="report-slip__perforation" aria-hidden="true" />
            <div className="report-slip__body">
              <span className="eyebrow">Agent reply · querying…</span>
              <div className="agent-view__typing" aria-label="Agent is thinking">
                <span />
                <span />
                <span />
              </div>
            </div>
          </div>
        )}
      </div>

      <form className="composer" onSubmit={handleSubmit}>
        <div className="composer__controls">
          <label className="composer__topk">
            <span className="eyebrow">Context chunks</span>
            <input
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(Math.min(20, Math.max(1, Number(e.target.value) || 1)))}
            />
          </label>
          <button
            type="button"
            className="composer__clear"
            onClick={() => setHistory([])}
            disabled={history.length === 0}
          >
            Clear history
          </button>
        </div>
        <div className="composer__row">
          <input
            type="text"
            className="composer__input"
            placeholder="Ask about transactions…"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={pending}
          />
          <button type="submit" className="composer__send" disabled={pending || !prompt.trim()}>
            Ask
          </button>
        </div>
      </form>
    </section>
  );
}
