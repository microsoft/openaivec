import functools
import json
import time
import uuid
from logging import Logger
from typing import Callable

__all__ = ["observe"]


def observe(logger: Logger):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def decorated(self: object, *args, **kwargs):
            child_logger: Logger = logger.getChild(self.__class__.__name__).getChild(func.__name__)
            transaction_id: str = str(uuid.uuid4())
            child_logger.info(
                json.dumps(
                    {
                        "transaction_id": transaction_id,
                        "type": "start",
                        "class": self.__class__.__name__,
                        "method": func.__name__,
                        "logged_at": time.time_ns(),
                    }
                )
            )
            try:
                res = func(self, *args, **kwargs)
                return res
            finally:
                child_logger.info(
                    json.dumps(
                        {
                            "transaction_id": transaction_id,
                            "type": "end",
                            "class": self.__class__.__name__,
                            "method": func.__name__,
                            "logged_at": time.time_ns(),
                        }
                    )
                )

        return decorated

    return decorator
