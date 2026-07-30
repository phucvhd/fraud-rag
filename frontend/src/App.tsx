import { useState } from "react";
import "./App.css";
import AgentView from "./components/AgentView";
import MonitorView from "./components/MonitorView";

type View = "agent" | "monitor";

const TABS: { id: View; label: string }[] = [
  { id: "agent", label: "Agent" },
  { id: "monitor", label: "Monitor" },
];

export default function App() {
  const [view, setView] = useState<View>("agent");

  return (
    <>
      <header className="topbar">
        <div className="topbar__brand">
          <span className="topbar__cursor" aria-hidden="true" />
          <span className="topbar__wordmark">FRAUD LEDGER</span>
          <span className="topbar__live" aria-label="Live console">
            <span className="topbar__live-dot" aria-hidden="true" />
            LIVE
          </span>
        </div>
        <nav className="topbar__nav" aria-label="Views">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={`topbar__tab ${view === tab.id ? "is-active" : ""}`}
              aria-current={view === tab.id ? "page" : undefined}
              onClick={() => setView(tab.id)}
            >
              <span className="topbar__tab-arrow" aria-hidden="true">
                &#9654;
              </span>
              {tab.label}
            </button>
          ))}
        </nav>
      </header>
      <main className="app-main">{view === "agent" ? <AgentView /> : <MonitorView />}</main>
    </>
  );
}
