from __future__ import annotations

import datetime as dt
from typing import Optional

import pytz


NY_TZ = pytz.timezone("America/New_York")


def set_local_tz(tz_name: str) -> None:
    global NY_TZ
    NY_TZ = pytz.timezone(tz_name)


def now_et() -> dt.datetime:
    return dt.datetime.now(tz=NY_TZ)


def today_et() -> dt.date:
    return now_et().date()


def to_et(dt_obj: dt.datetime) -> dt.datetime:
    if dt_obj.tzinfo is None:
        return NY_TZ.localize(dt_obj)
    return dt_obj.astimezone(NY_TZ)

