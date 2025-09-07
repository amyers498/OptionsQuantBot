import hashlib
import uuid
from typing import Any


def deterministic_key(*parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def random_key() -> str:
    return uuid.uuid4().hex

