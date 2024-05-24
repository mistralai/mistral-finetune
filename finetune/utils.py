import contextlib
import dataclasses
import datetime
import logging
import time
from typing import Optional, Protocol

import torch

logger = logging.getLogger("utils")


@dataclasses.dataclass
class TrainState:
    max_steps: int
    step: int = 0
    elapsed_time: float = 0.0
    n_seen_tokens: int = 0
    this_step_time: float = 0.0
    begin_step_time: float = 0.0
    this_eval_perplexity: Optional[float] = None
    this_eval_loss: Optional[float] = None

    def start_step(self):
        self.step += 1
        self.begin_step_time = time.time()

    def end_step(self, n_batch_tokens: int):
        self.this_step_time = time.time() - self.begin_step_time
        self.this_step_tokens = n_batch_tokens

        self.elapsed_time += self.this_step_time
        self.n_seen_tokens += self.this_step_tokens

        self.begin_step_time = time.time()

    @property
    def wps(self):
        return self.this_step_tokens / self.this_step_time

    @property
    def avg_wps(self):
        return self.n_seen_tokens / self.elapsed_time

    @property
    def eta(self):
        steps_left = self.max_steps - self.step
        avg_time_per_step = self.elapsed_time / self.step

        return steps_left * avg_time_per_step


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Closable(Protocol):
    def close(self):
        pass


@contextlib.contextmanager
def logged_closing(thing: Closable, name: str):
    """
    Logging the closing to be sure something is not hanging at exit time
    """
    try:
        setattr(thing, "wrapped_by_closing", True)
        yield
    finally:
        logger.info(f"Closing: {name}")
        try:
            thing.close()
        except Exception:
            logger.error(f"Error while closing {name}!")
            raise
        logger.info(f"Closed: {name}")


def now_as_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
