import { useEffect, useRef, useState } from "react";
import { askAgent } from "../api/client";
import type { ChatEntry } from "../types";
import "./AgentView.css";
import ChatMessage from "./ChatMessage";

function makeId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

const HISTORY_KEY = "fraud-ledger.agent.history";

const EXAMPLE_PROMPTS = [
  "Any anomaly transactions over 1000 EUR in the last hour?",
  "Show me the most suspicious transactions right now.",
  "Find transactions similar to known fraud patterns.",
];

function loadHistory(): ChatEntry[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? (JSON.parse(raw) as ChatEntry[]) : [];
  } catch {
    return [];
  }
}

export default function AgentView() {
  const [history, setHistory] = useState<ChatEntry[]>(loadHistory);
  const [prompt, setPrompt] = useState("");
  const [topK, setTopK] = useState(3);
  const [pending, setPending] = useState(false);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    transcriptRef.current?.scrollTo({ top: transcriptRef.current.scrollHeight, behavior: "smooth" });
  }, [history, pending]);

  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch {
      /* storage full or unavailable — history simply won't persist */
    }
  }, [history]);

  async function submit(text: string) {
    const trimmed = text.trim();
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
      const message = err instanceof Error ? err.message : "Something went wrong. Try again.";
      setHistory((h) => [...h, { id: makeId(), role: "error", content: message, timestamp: new Date().toISOString() }]);
    } finally {
      setPending(false);
      inputRef.current?.focus();
    }
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    void submit(prompt);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends; Shift+Enter inserts a newline.
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void submit(prompt);
    }
  }

  return (
    <section className="agent-view">
      <div className="agent-view__transcript" ref={transcriptRef}>
        {history.length === 0 && (
          <div className="agent-view__empty">
            <span className="eyebrow">▚ NEW CASE FILE</span>
            <p className="agent-view__empty-lead">
              Ask about flagged activity. The agent pulls similar past transactions and scores their features before
              answering.
            </p>
            <span className="eyebrow">Try one</span>
            <div className="agent-view__examples">
              {EXAMPLE_PROMPTS.map((ex) => (
                <button key={ex} type="button" className="agent-view__example" onClick={() => void submit(ex)}>
                  &gt; {ex}
                </button>
              ))}
            </div>
          </div>
        )}
        {history.map((entry) => (
          <ChatMessage key={entry.id} entry={entry} />
        ))}
        {pending && (
          <div className="report-slip report-slip--pending">
            <div className="report-slip__tab" aria-hidden="true">
              AGENT
            </div>
            <div className="report-slip__body">
              <span className="eyebrow">Scanning ledger…</span>
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
            <span className="eyebrow">Similar cases</span>
            <input
              type="number"
              min={1}
              max={20}
              value={topK}
              aria-label="Number of similar past transactions to retrieve"
              onChange={(e) => setTopK(Math.min(20, Math.max(1, Number(e.target.value) || 1)))}
            />
          </label>
          <button
            type="button"
            className="composer__clear"
            onClick={() => setHistory([])}
            disabled={history.length === 0 || pending}
          >
            Clear history
          </button>
        </div>
        <div className="composer__row">
          <textarea
            ref={inputRef}
            className="composer__input"
            placeholder="Ask about transactions…  (Enter to send · Shift+Enter for a new line)"
            value={prompt}
            rows={1}
            onKeyDown={handleKeyDown}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={pending}
          />
          <button type="submit" className="composer__send" disabled={pending || !prompt.trim()}>
            {pending ? "…" : "Ask ▶"}
          </button>
        </div>
      </form>
    </section>
  );
}
