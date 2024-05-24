import datetime
import logging
import sys
import time


class DeltaTimeFormatter(logging.Formatter):
    def format(self, record):
        delta = datetime.timedelta(
            seconds=int(record.relativeCreated / 1000)
        )  # no milliseconds
        record.delta = delta
        return super().format(record)


def set_logger(level: int = logging.INFO):
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    tz, *_ = time.tzname

    LOGFORMAT = "%(asctime)s - %(delta)s - %(name)s - %(levelname)s - %(message)s"
    TIMEFORMAT = f"%Y-%m-%d %H:%M:%S ({tz})"
    formatter = DeltaTimeFormatter(LOGFORMAT, TIMEFORMAT)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(formatter)
    root.addHandler(handler)
