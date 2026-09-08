import { useEffect, useState } from "react";
import "./App.css";
import AgentView from "./components/AgentView";
import MonitorView from "./components/MonitorView";

export default function App() {
  const [clock, setClock] = useState(() => new Date());

  useEffect(() => {
    const id = setInterval(() => setClock(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="shell">
      <aside className="rail">
        <div className="rail__brand">
          <span className="rail__mark" aria-hidden="true" />
          <span className="rail__name">Fraud Ledger</span>
        </div>
        <div className="rail__status">
          <span className="rail__status-dot" aria-hidden="true" />
          <span>Live</span>
        </div>
      </aside>
      <div className="shell__main">
        <header className="topbar">
          <h1 className="topbar__title">Dashboard</h1>
          <span className="topbar__clock data">
            {clock.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
          </span>
        </header>
        <main className="app-main">
          <MonitorView />
          <AgentView />
        </main>
      </div>
    </div>
  );
}
