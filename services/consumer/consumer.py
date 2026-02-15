import json
import logging
from confluent_kafka import Consumer, KafkaError

from services.repository.repository import TransactionRepository
from shared.config_loader import config_loader
from schemas.transaction import TransactionCanonical

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FraudConsumer")

class FraudTransactionConsumer:
    def __init__(self):
        self.cfg = config_loader.load()
        self.repo = TransactionRepository()
        self.consumer = Consumer({
            'bootstrap.servers': self.cfg.kafka.bootstrap_servers,
            'group.id': self.cfg.kafka.group_id,
            'auto.offset.reset': 'earliest'
        })

    def _handle_message(self, msg):
        try:
            raw_data = json.loads(msg.value().decode('utf-8'))
            transaction = TransactionCanonical(**raw_data)
            self.repo.insert_if_not_exists(transaction)
            logger.info(f"Ingested: {transaction.transaction_id}")
        except Exception as e:
            logger.error(f"Error: {e}")

    def start(self):
        self.consumer.subscribe([self.cfg.kafka.topic])
        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Kafka error: {msg.error()}")
                        break
                self._handle_message(msg)
        finally:
            self.stop()

    def stop(self):
        self.consumer.close()

if __name__ == "__main__":
    consumer = FraudTransactionConsumer()
    consumer.start()