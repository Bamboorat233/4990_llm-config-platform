"""Fine-tune progress collector (issue #31).

``TrainingProgress`` is fed every line of the log by the detector's single
streaming pass and keeps only a handful of numbers, never the step lines
themselves (a full run logs 9,936 of them):

- ``last_loss``   the loss of the last ``{'loss': '...'}`` line, as a float;
- ``epoch``       the ``epoch`` value of that same line, as a float;
- ``eval_losses`` every ``eval_loss`` in order, as floats;
- ``step`` / ``total_steps``  the last position of the training progress bar.

The training bar is the unlabelled bar with the largest total, so the short
evaluation bar (for example 386/386) and the labelled ``Loading weights`` and
``Writing model shards`` bars are never mistaken for it. A bar at ``0/N``
with no loss lines means training was reached but made no step: ``step`` is
0 and ``last_loss`` stays ``None``.

The class imports only :mod:`app.parsing.errors`; it knows nothing about the
detector, so it can be tested on its own.
"""

from __future__ import annotations

import re
from typing import Any, Final

from app.parsing.errors import PATTERNS

__all__ = ["TrainingProgress"]

_LOSS: Final = re.compile(r"\{'loss': '(?P<loss>[^']*)'")
_EPOCH: Final = re.compile(r"'epoch': '(?P<epoch>[^']*)'")


def _to_float(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:
        return None


class TrainingProgress:
    """Bounded state for the ``progress`` dict of a fine-tune run."""

    def __init__(self) -> None:
        self.step: int | None = None
        self.total_steps: int | None = None
        self.epoch: float | None = None
        self.last_loss: float | None = None
        self.eval_losses: list[float] = []

    def feed(self, text: str, line: int | None = None) -> None:
        """Consume one line (``[rankN]: `` prefix already stripped)."""

        if "'loss': '" in text:
            match = _LOSS.search(text)
            loss = _to_float(match.group("loss")) if match else None
            if loss is not None:
                self.last_loss = loss
                epoch = _EPOCH.search(text)
                value = _to_float(epoch.group("epoch")) if epoch else None
                if value is not None:
                    self.epoch = value
        elif "'eval_loss': '" in text:
            match = PATTERNS["eval_loss"].search(text)
            value = _to_float(match.group(1)) if match else None
            if value is not None:
                self.eval_losses.append(value)
        elif "|" in text and "/" in text and "[" in text:
            self._bar(text)

    def _bar(self, text: str) -> None:
        match = PATTERNS["progress_bar"].search(text)
        if not match:
            return
        done, total = int(match.group(1)), int(match.group(2))
        if total <= 0 or done > total:
            return
        if any(char.isalpha() for char in text.split("|", 1)[0]):
            return  # a labelled bar (Loading weights, Writing model shards)
        if self.total_steps is None or total > self.total_steps:
            self.total_steps, self.step = total, done
        elif total == self.total_steps:
            self.step = done

    def result(self) -> dict[str, Any]:
        """Return the five fine-tune ``progress`` keys."""

        return {
            "step": self.step,
            "total_steps": self.total_steps,
            "epoch": self.epoch,
            "last_loss": self.last_loss,
            "eval_losses": list(self.eval_losses),
        }
