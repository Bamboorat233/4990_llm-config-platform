"""Inference progress collector (issue #31).

``InferenceProgress`` is fed every line by the detector's single streaming
pass and keeps a few counters, never the records themselves.

Two record layouts exist in the real logs and a log never mixes them:

- **banner** (``inferenceMistral.py``): each record is an ``INPUTS`` banner,
  an optional ``LENGTH`` banner, an ``OUTPUT`` banner and then the decoded
  text, which normally ends with a ``### Response:`` line. A record is *done*
  when its ``OUTPUT`` block is closed by the next ``INPUTS`` banner (the
  script moved on, even when the model's reply was empty and no
  ``### Response:`` line was printed: 7 such records in log 02), or, for the
  last block of a log, when it contains a ``### Response:`` line. A block is
  counted once: log 03 ends with a bare list of ``### Response:`` lines after
  the last block. A record that only reached its ``INPUTS`` banner (log 15,
  cut off by the time limit) is not counted.
- **timestamped** (``think`` and ``table_only``): each record prints
  ``Tokenizing Inputs`` ... ``Done Decoding`` timestamp lines. A record is
  *done* when its ``Done Decoding`` line appears.

``records_done`` is ``None`` when the log shows no record at all (the run
failed before the first record, logs 01 and 14) and 0 when a record was
started but none finished (log 06). ``records_expected`` is always ``None``:
no log prints how many records the run will produce. The scoring step
compares ``records_done`` with the gold-set size instead
(``ScoreReport.complete``).

The class imports only :mod:`app.parsing.errors`.
"""

from __future__ import annotations

from typing import Any, Final

from app.parsing.errors import PATTERNS

__all__ = ["InferenceProgress"]

_RESPONSE: Final = "### Response:"
_DONE_DECODING: Final = "Done Decoding"
_TOKENIZING: Final = "Tokenizing Inputs"


class InferenceProgress:
    """Bounded state for the ``progress`` dict of an inference run."""

    def __init__(self) -> None:
        self.saw_banner_record = False
        self.saw_timestamp_record = False
        self.in_output = False
        self.block_counted = False
        self.banner_done = 0
        self.decoded = 0

    def feed(self, text: str, line: int | None = None) -> None:
        """Consume one line (``[rankN]: `` prefix already stripped)."""

        first = text[:1]
        if first in ("*", "#"):
            match = PATTERNS["banner"].match(text)
            if match:
                section = match.group(1)
                if section in ("INPUTS", "OUTPUT"):
                    self.saw_banner_record = True
                    self._close_block()
                    self.in_output = section == "OUTPUT"
                    self.block_counted = False
                return
            if self.in_output and not self.block_counted and text.startswith(_RESPONSE):
                self.block_counted = True
                self.banner_done += 1
        elif first == "T" and text.startswith(_TOKENIZING):
            self.saw_timestamp_record = True
        elif first == "D" and text.startswith(_DONE_DECODING):
            if PATTERNS["phase_timestamp"].match(text):
                self.decoded += 1

    def _close_block(self) -> None:
        """A block closed by the next banner counts, even without a verdict."""

        if self.in_output and not self.block_counted:
            self.banner_done += 1

    def result(self) -> dict[str, Any]:
        """Return the two inference ``progress`` keys."""

        if self.saw_banner_record:
            done: int | None = self.banner_done
        elif self.saw_timestamp_record:
            done = self.decoded
        else:
            done = None
        return {"records_done": done, "records_expected": None}
