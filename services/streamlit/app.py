import os
import sys
from datetime import datetime, timedelta

import altair as alt
import pandas as pd
import requests
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from shared.config_loader import config_loader

st.set_page_config(page_title="Fraud RAG Monitor", layout="wide", page_icon="🔍")

cfg = config_loader.load()
RAG_URL          = cfg.dashboard.rag_url
INJECT_BASE_URL  = cfg.dashboard.inject_url.split("?")[0]
TRANSACTIONS_URL = cfg.dashboard.transactions_url

# --- Session state init ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --- Helpers ---

def _floor5(dt: datetime) -> datetime:
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


def _ceil5(dt: datetime) -> datetime:
    dt = dt.replace(second=0, microsecond=0)
    remainder = dt.minute % 5
    return dt + timedelta(minutes=(5 - remainder) if remainder != 0 else 5)


def fetch_timeseries(start_dt: datetime, end_dt: datetime) -> dict | None:
    try:
        resp = requests.get(
            TRANSACTIONS_URL,
            params={"start": start_dt.isoformat(), "end": end_dt.isoformat()},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def timeseries_to_df(data: list[dict]) -> pd.DataFrame:
    if not data:
        return pd.DataFrame([{"time": "—", "transactions": 0, "fraud": 0, "normal": 0}])
    return pd.DataFrame(data).rename(columns={"bucket": "time"})


# --- App ---

st.title("🔍 Fraud RAG Monitor")

tab_dashboard, tab_agent = st.tabs(["📊 Transaction Dashboard", "🤖 Agent Query"])


# ── Tab 1: Transaction Dashboard ─────────────────────────────────────────────

with tab_dashboard:
    _now0 = _floor5(datetime.now())
    _end0 = _ceil5(datetime.now())
    _defaults = {
        "bs_start_date": (_now0 - timedelta(minutes=30)).date(),
        "bs_start_time": (_now0 - timedelta(minutes=30)).time(),
        "bs_end_date":   _end0.date(),
        "bs_end_time":   _end0.time(),
        "bs_end_locked": False,
    }
    for k, v in _defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    @st.fragment(run_every=30)
    def render_dashboard():
        st.subheader("Transaction Activity")

        # Auto-advance end time unless user locked it
        if not st.session_state.bs_end_locked:
            n = _ceil5(datetime.now())
            st.session_state.bs_end_date = n.date()
            st.session_state.bs_end_time = n.time()

        # ── Time range ───────────────────────────────────────────────────────
        st.markdown("**Time Range**")
        col_sd, col_st, col_arrow, col_ed, col_et, col_now = st.columns([2, 1.5, 0.2, 2, 1.5, 0.8])
        with col_sd:
            new_sd = st.date_input("Start date", value=st.session_state.bs_start_date)
        with col_st:
            new_st = st.time_input("Start time", value=st.session_state.bs_start_time, step=300)
        with col_arrow:
            st.write("")
            st.markdown("→")
        with col_ed:
            new_ed = st.date_input("End date", value=st.session_state.bs_end_date)
        with col_et:
            new_et = st.time_input("End time", value=st.session_state.bs_end_time, step=300)
        with col_now:
            st.write("")
            if st.button("Now", width="stretch", help="Reset end to next 5-min mark and resume auto-advance"):
                n = _ceil5(datetime.now())
                st.session_state.bs_end_date   = n.date()
                st.session_state.bs_end_time   = n.time()
                st.session_state.bs_end_locked = False
                st.rerun()

        # Detect manual end edit → lock auto-advance
        if new_ed != st.session_state.bs_end_date or new_et != st.session_state.bs_end_time:
            st.session_state.bs_end_locked = True

        st.session_state.bs_start_date = new_sd
        st.session_state.bs_start_time = new_st
        st.session_state.bs_end_date   = new_ed
        st.session_state.bs_end_time   = new_et

        start_dt = datetime.combine(new_sd, new_st)
        end_dt   = datetime.combine(new_ed, new_et)

        if start_dt >= end_dt:
            st.warning("Start must be before end.")
            return

        # ── Fetch data ───────────────────────────────────────────────────────
        result = fetch_timeseries(start_dt, end_dt)

        if result is None or "error" in result:
            st.error(f"Failed to fetch data: {result.get('error') if result else 'unknown error'}")
            return

        total_tx     = result["total_transactions"]
        total_fraud  = result["total_fraud"]
        total_normal = result["total_normal"]
        fraud_rate   = (total_fraud / total_tx * 100) if total_tx > 0 else 0.0

        # ── Metrics ──────────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Transactions", total_tx)
        m2.metric("Fraud", total_fraud)
        m3.metric("Normal", total_normal)
        m4.metric("Fraud Rate", f"{fraud_rate:.1f}%")

        st.divider()

        # ── Charts ───────────────────────────────────────────────────────────
        df = timeseries_to_df(result["data"])
        col_chart1, col_chart2, col_chart3 = st.columns(3)
        with col_chart1:
            st.markdown("**Transactions per Minute**")
            chart1 = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("time:N", sort=None, title="Time"),
                    y=alt.Y("transactions:Q", title="Transactions"),
                    tooltip=["time", "transactions"],
                )
            )
            st.altair_chart(chart1, width="stretch")
        with col_chart2:
            st.markdown("**Fraud Decisions**")
            chart2 = (
                alt.Chart(df)
                .mark_bar(color="#e45756")
                .encode(
                    x=alt.X("time:N", sort=None, title="Time"),
                    y=alt.Y("fraud:Q", title="Fraud"),
                    tooltip=["time", "fraud"],
                )
            )
            st.altair_chart(chart2, width="stretch")
        with col_chart3:
            st.markdown("**Normal Decisions**")
            chart3 = (
                alt.Chart(df)
                .mark_bar(color="#4c78a8")
                .encode(
                    x=alt.X("time:N", sort=None, title="Time"),
                    y=alt.Y("normal:Q", title="Normal"),
                    tooltip=["time", "normal"],
                )
            )
            st.altair_chart(chart3, width="stretch")

        st.caption(
            f"Last refreshed: {datetime.now().strftime('%H:%M:%S')} · "
            f"auto-refreshes every 30s"
            + (" · end time locked" if st.session_state.bs_end_locked else "")
        )

        st.divider()

        # ── Inject ───────────────────────────────────────────────────────────
        st.markdown("**Inject Messages**")
        col_dur, col_btn, col_msg = st.columns([2, 1, 3])
        with col_dur:
            duration = st.number_input(
                "Duration (seconds)", min_value=1, max_value=300, value=1, step=1,
                help="How many seconds of transactions to inject into Kafka",
            )
        with col_btn:
            st.write("")
            if st.button("💉 Inject", width="stretch"):
                try:
                    url = f"{INJECT_BASE_URL}?duration_seconds={int(duration)}"
                    resp = requests.post(url, timeout=30)
                    if resp.status_code == 200:
                        col_msg.success(f"Injected {int(duration)}s of transactions — chart updates on next refresh.")
                    else:
                        col_msg.error(f"Failed: {resp.status_code} — {resp.text}")
                except Exception as e:
                    col_msg.error(f"Error: {e}")

    render_dashboard()


# ── Tab 2: Agent Query UI ─────────────────────────────────────────────────────

with tab_agent:
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry["role"] == "assistant" and entry.get("raw"):
                with st.expander("Raw response"):
                    st.json(entry["raw"])

    col_topk, col_clear = st.columns([4, 1])
    with col_topk:
        top_k = st.number_input("Context chunks (top_k)", min_value=1, max_value=20, value=3, step=1,
                                label_visibility="collapsed")
    with col_clear:
        if st.button("🗑 Clear history", width="stretch"):
            st.session_state.chat_history = []
            st.rerun()

    prompt = st.chat_input("Ask about transactions, e.g. 'Any anomaly transactions over 1000 EUR?'")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Querying RAG agent..."):
                try:
                    resp = requests.post(RAG_URL, json={"prompt": prompt, "top_k": int(top_k)}, timeout=None)
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data.get("answer", "_No answer returned._")
                        st.markdown(answer)
                        with st.expander("Raw response"):
                            st.json(data)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "raw": data,
                        })
                    else:
                        err = f"API error {resp.status_code}: {resp.text}"
                        st.error(err)
                        st.session_state.chat_history.append({"role": "assistant", "content": err})
                except Exception as e:
                    err = f"Request failed: {e}"
                    st.error(err)
                    st.session_state.chat_history.append({"role": "assistant", "content": err})
