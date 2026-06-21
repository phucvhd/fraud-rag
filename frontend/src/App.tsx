import { useState } from "react";
import "./App.css";
import AgentView from "./components/AgentView";
import MonitorView from "./components/MonitorView";

type View = "agent" | "monitor";

export default function App() {
  const [view, setView] = useState<View>("agent");

  return (
    <>
      <header className="topbar">
        <div className="topbar__brand">
          <span className="topbar__dot" aria-hidden="true" />
          <span className="topbar__wordmark">Fraud Ledger</span>
        </div>
        <nav className="topbar__nav" aria-label="Views">
          <button
            type="button"
            className={`topbar__tab ${view === "agent" ? "is-active" : ""}`}
            aria-current={view === "agent"}
            onClick={() => setView("agent")}
          >
            Agent
          </button>
          <button
            type="button"
            className={`topbar__tab ${view === "monitor" ? "is-active" : ""}`}
            aria-current={view === "monitor"}
            onClick={() => setView("monitor")}
          >
            Monitor
          </button>
        </nav>
      </header>
      <main className="app-main">{view === "agent" ? <AgentView /> : <MonitorView />}</main>
    </>
  );
}
