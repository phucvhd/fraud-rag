import { useEffect, useState } from "react";
import { fetchTransactions } from "../api/client";
import type { TransactionRecord } from "../types";
import "./TransactionTable.css";

type StatusFilter = "all" | "fraud" | "clear";
type SortColumn = "time" | "amount" | "status" | "risk";
type SortDir = "asc" | "desc";

const PAGE_SIZE = 50;
const STATUS_FILTERS: { id: StatusFilter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "fraud", label: "Fraud" },
  { id: "clear", label: "Clear" },
];

interface TransactionTableProps {
  start: Date;
  end: Date;
  reloadKey: number;
}

function formatAmount(amount: number): string {
  return amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function shortId(id: string): string {
  return id.split("-")[0];
}

function formatRisk(probability: number | null): string {
  return probability == null ? "—" : `${Math.round(probability * 100)}%`;
}

export default function TransactionTable({ start, end, reloadKey }: TransactionTableProps) {
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [sortBy, setSortBy] = useState<SortColumn>("time");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [page, setPage] = useState(0);

  const [records, setRecords] = useState<TransactionRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const id = setTimeout(() => setDebouncedSearch(search.trim()), 300);
    return () => clearTimeout(id);
  }, [search]);

  // Any change to the query resets pagination back to the first page.
  useEffect(() => {
    setPage(0);
  }, [debouncedSearch, statusFilter, sortBy, sortDir, start, end, reloadKey]);

  useEffect(() => {
    if (start >= end) return;
    let active = true;
    setLoading(true);
    fetchTransactions(start, end, {
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
      isFraud: statusFilter === "all" ? undefined : statusFilter === "fraud",
      search: debouncedSearch || undefined,
      sortBy,
      sortDir,
    })
      .then((r) => {
        if (!active) return;
        setRecords(r.data);
        setTotal(r.total);
        setError(null);
      })
      .catch((e) => {
        if (!active) return;
        setError(e instanceof Error ? e.message : "Couldn't load transactions.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [start, end, reloadKey, page, debouncedSearch, statusFilter, sortBy, sortDir]);

  function toggleSort(column: SortColumn) {
    if (sortBy === column) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortBy(column);
      setSortDir("desc");
    }
  }

  function sortIndicator(column: SortColumn) {
    if (sortBy !== column) return null;
    return <span className="tx-table__sort-arrow">{sortDir === "asc" ? "↑" : "↓"}</span>;
  }

  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const rangeStart = total === 0 ? 0 : page * PAGE_SIZE + 1;
  const rangeEnd = Math.min(total, (page + 1) * PAGE_SIZE);

  return (
    <div className="tx-table">
      <div className="tx-table__toolbar">
        <span className="eyebrow tx-table__title">Transactions</span>
        <input
          type="search"
          className="tx-table__search"
          placeholder="Search by transaction ID or source…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <div className="segmented" role="group" aria-label="Filter by status">
          {STATUS_FILTERS.map((f) => (
            <button
              key={f.id}
              type="button"
              className={`segmented__item ${statusFilter === f.id ? "is-active" : ""}`}
              onClick={() => setStatusFilter(f.id)}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {error ? (
        <p className="tx-table__error">{error}</p>
      ) : total === 0 && !loading ? (
        <p className="tx-table__empty">No transactions match these filters.</p>
      ) : (
        <>
          <div className="tx-table__scroll">
            <table>
              <thead>
                <tr>
                  <th className="tx-table__sortable" onClick={() => toggleSort("time")}>
                    Time {sortIndicator("time")}
                  </th>
                  <th>Transaction</th>
                  <th className="tx-table__num tx-table__sortable" onClick={() => toggleSort("amount")}>
                    Amount {sortIndicator("amount")}
                  </th>
                  <th className="tx-table__sortable" onClick={() => toggleSort("status")}>
                    Status {sortIndicator("status")}
                  </th>
                  <th className="tx-table__num tx-table__sortable" onClick={() => toggleSort("risk")}>
                    Risk {sortIndicator("risk")}
                  </th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                {records.map((r) => (
                  <tr key={r.transaction_id}>
                    <td className="data">
                      {new Date(r.event_timestamp).toLocaleString([], { dateStyle: "short", timeStyle: "medium" })}
                    </td>
                    <td className="data" title={r.transaction_id}>
                      {shortId(r.transaction_id)}
                    </td>
                    <td className="data tx-table__num">{formatAmount(r.amount)}</td>
                    <td>
                      <span className={`tx-badge ${r.is_fraud ? "tx-badge--fraud" : "tx-badge--normal"}`}>
                        {r.is_fraud ? "Fraud" : "Clear"}
                      </span>
                    </td>
                    <td className="data tx-table__num">{formatRisk(r.fraud_probability)}</td>
                    <td className="tx-table__source">{r.data_source}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="tx-table__pagination">
            <span className="tx-table__caption">
              {loading
                ? "Loading…"
                : `${rangeStart.toLocaleString()}–${rangeEnd.toLocaleString()} of ${total.toLocaleString()}`}
            </span>
            <div className="tx-table__pager">
              <button
                type="button"
                className="btn btn--ghost"
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
              >
                Prev
              </button>
              <span className="tx-table__page-indicator">
                Page {page + 1} of {pageCount}
              </span>
              <button
                type="button"
                className="btn btn--ghost"
                onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
                disabled={page >= pageCount - 1}
              >
                Next
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
