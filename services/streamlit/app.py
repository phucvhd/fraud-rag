import json
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st
from confluent_kafka import Consumer
import threading
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from shared.config_loader import config_loader

st.set_page_config(page_title="Transaction Dashboard", layout="wide")

cfg = config_loader.load()
RAG_URL = cfg.dashboard.rag_url
INJECT_URL = cfg.dashboard.inject_url

TOPIC_TRANSACTIONS = "transactions"
TOPIC_DECISIONS = "transaction-decisions"

KAFKA_CONFIG = {
    'bootstrap.servers': cfg.kafka.bootstrap_servers,
    'group.id': 'fraud-detection-client-consumer-group',
    'auto.offset.reset': 'earliest'
}

if "kafka_started" not in st.session_state:
    st.session_state.kafka_started = False
    st.session_state.counts = {
        TOPIC_TRANSACTIONS: {},
        TOPIC_DECISIONS: {},
        "decisions_fraud": {},
        "decisions_normal": {},
    }

counts = st.session_state.counts
lock = threading.Lock()


def parse_ts(ts_str: str) -> datetime:
    return datetime.fromisoformat(ts_str)


def bucket_minute(dt: datetime) -> str:
    return dt.strftime("%H:%M %d-%m-%Y")


def parse_bucket(bucket_str: str) -> datetime:
    return datetime.strptime(bucket_str, "%H:%M %d-%m-%Y")


def update_counts(topic: str, ts_str: str, is_fraud=None):
    try:
        dt = parse_ts(ts_str)
        bucket = bucket_minute(dt)
        with lock:
            counts[topic][bucket] = counts[topic].get(bucket, 0) + 1

            if is_fraud is not None and topic == TOPIC_DECISIONS:
                fraud_key = "decisions_fraud" if is_fraud else "decisions_normal"
                counts[fraud_key][bucket] = counts[fraud_key].get(bucket, 0) + 1
    except Exception:
        pass


def kafka_consumer_thread(topic: str, ts_field: str):
    consumer = Consumer(KAFKA_CONFIG)
    consumer.subscribe([topic])
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                continue
            try:
                value = msg.value().decode("utf-8")
                data = json.loads(value)
            except Exception:
                continue
            if ts_field in data:
                if topic == TOPIC_DECISIONS and "is_fraud" in data:
                    update_counts(topic, data[ts_field], data["is_fraud"])
                else:
                    update_counts(topic, data[ts_field])
    finally:
        consumer.close()


def ensure_kafka():
    if not st.session_state.kafka_started:
        st.session_state.kafka_started = True
        t1 = threading.Thread(
            target=kafka_consumer_thread,
            args=(TOPIC_TRANSACTIONS, "timestamp"),
            daemon=True,
        )
        t2 = threading.Thread(
            target=kafka_consumer_thread,
            args=(TOPIC_DECISIONS, "event_timestamp"),
            daemon=True,
        )
        t1.start()
        t2.start()


def build_counts_df():
    now = datetime.now()
    cutoff = now - timedelta(minutes=30)

    with lock:
        buckets = set(counts[TOPIC_TRANSACTIONS].keys()) | set(counts[TOPIC_DECISIONS].keys())
        data = []
        for b in sorted(buckets):
            try:
                bucket_dt = parse_bucket(b)
                if bucket_dt >= cutoff:
                    data.append({
                        "bucket": b,
                        "transactions": counts[TOPIC_TRANSACTIONS].get(b, 0),
                        "fraud": counts["decisions_fraud"].get(b, 0),
                        "normal": counts["decisions_normal"].get(b, 0),
                    })
            except:
                continue
        if not data:
            data.append({
                "bucket": "No data",
                "transactions": 0,
                "fraud": 0,
                "normal": 0,
            })
    return pd.DataFrame(data)


st.set_page_config(page_title="Transaction Monitor", layout="wide")
st.title("Transaction Monitor")
st.subheader("Message Activity")

prompt = st.text_area(
    "RAG Prompt",
    value="Are there any anomaly transactions over 1000 EUR?",
    height=120,
)
top_k = st.number_input("top_k", min_value=1, max_value=20, value=3, step=1)

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("Ask RAG"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            try:
                payload = {"prompt": prompt, "top_k": int(top_k)}
                resp = requests.post(RAG_URL, json=payload, timeout=None)
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "")
                    st.markdown("**Answer**")
                    st.markdown(answer)
                    with st.expander("Raw response"):
                        st.json(data)
                else:
                    st.error(f"API error: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

with col_btn2:
    if st.button("Inject Messages"):
        try:
            resp = requests.post(INJECT_URL, timeout=10)
            if resp.status_code == 200:
                st.success("Messages injected!")
            else:
                st.error(f"Inject failed: {resp.status_code}")
        except Exception as e:
            st.error(f"Inject request failed: {e}")

ensure_kafka()

placeholder_charts = st.empty()

while True:
    df = build_counts_df()
    with placeholder_charts.container():
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.subheader("Transactions")
            tx_df = df.set_index("bucket")["transactions"]
            st.bar_chart(tx_df, x_label=None)

        with col_chart2:
            st.subheader("Transaction Decisions")
            decisions_df = df.set_index("bucket")[["fraud", "normal"]]
            st.bar_chart(decisions_df, x_label=None)

    time.sleep(2)