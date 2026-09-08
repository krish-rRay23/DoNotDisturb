import json

from adaptive_plasticity.logging_utils import configure_logging


def test_logging_initialization(tmp_path) -> None:
    target = tmp_path / "logs" / "run.jsonl"
    logger = configure_logging(target)
    logger.info("initialized", extra={"event": "test"})
    for handler in logger.handlers:
        handler.flush()
    record = json.loads(target.read_text(encoding="utf-8"))
    assert record["message"] == "initialized"
    assert record["event"] == "test"

