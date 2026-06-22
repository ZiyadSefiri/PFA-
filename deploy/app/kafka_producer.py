import json
import logging
import os

from aiokafka import AIOKafkaProducer

logger = logging.getLogger(__name__)

_producer: AIOKafkaProducer | None = None
_TOPIC = os.getenv("KAFKA_TOPIC", "inference-events")


async def init_producer(bootstrap_servers: str = "redpanda:9092"):
    global _producer
    _producer = AIOKafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode(),
    )
    await _producer.start()
    logger.info("Kafka producer connected to %s", bootstrap_servers)


async def close_producer():
    global _producer
    if _producer is not None:
        await _producer.stop()
        _producer = None


async def send_inference(input_records: list[dict], results: list[dict]):
    if _producer is None:
        logger.warning("Kafka producer not available, dropping %d records", len(input_records))
        return
    for inp, out in zip(input_records, results):
        msg = {
            "input": inp,
            "prediction": out["prediction"],
            "probability_readmitted": out["probability_readmitted"],
            "probability_not_readmitted": out["probability_not_readmitted"],
        }
        await _producer.send(_TOPIC, msg)
