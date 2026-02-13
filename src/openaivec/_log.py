import functools
import inspect
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from logging import Logger
from typing import ParamSpec, TypeVar, cast

__all__ = []

P = ParamSpec("P")
R = TypeVar("R")


def observe(logger: Logger) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def decorated_async(*args: P.args, **kwargs: P.kwargs) -> R:
                if not args:
                    return await cast(Callable[P, Awaitable[R]], func)(*args, **kwargs)
                self = args[0]
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
                    return await cast(Callable[P, Awaitable[R]], func)(*args, **kwargs)
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

            return cast(Callable[P, R], decorated_async)

        @functools.wraps(func)
        def decorated(*args: P.args, **kwargs: P.kwargs) -> R:
            if not args:
                return func(*args, **kwargs)
            self = args[0]
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
                res = func(*args, **kwargs)
                return cast(R, res)
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
