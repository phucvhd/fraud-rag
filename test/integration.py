import json
from confluent_kafka import Producer
from uuid import uuid4
from datetime import datetime

from services.consumer.consumer import FraudTransactionConsumer
from services.embedder.worker import EmbeddingWorker


class KafkaMockProducer:
    def __init__(self, bootstrap_servers: str):
        self.producer = Producer({'bootstrap.servers': bootstrap_servers})

    def send_mock_transaction(self, topic: str):
        mock_data = {
            "transaction_id": str(uuid4()),
            "Time": 12345,
            "Amount": 99.99,
            "is_fraud": False,
            "features": {f"V{i}": 0.123 for i in range(1, 29)},
            "event_timestamp": datetime.now().isoformat(),
            "data_source": "test_manual",
            "created_at": datetime.now().isoformat()
        }

        self.producer.produce(
            topic,
            key=mock_data["transaction_id"],
            value=json.dumps(mock_data)
        )
        self.producer.flush()
        print(f"Sent mock transaction: {mock_data['transaction_id']}")


if __name__ == "__main__":
    # consumer = FraudTransactionConsumer()
    # consumer.start()
    #
    # embedder = EmbeddingWorker()
    # embedder.start()

    tester = KafkaMockProducer('localhost:9092')
    tester.send_mock_transaction('transaction-decisions')