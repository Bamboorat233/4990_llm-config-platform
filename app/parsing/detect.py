"""Detect family, phase outcomes, status and output layout of a SLURM log.

``detect(path)`` makes a **single forward-only pass** over the file.  The
file is opened once with ``encoding="utf-8", errors="replace",
newline=None`` and iterated line by line; nothing but a small, bounded set
of first-seen markers and terminal events is retained.  A 21 MB / 200k-line
log therefore uses the same memory as a 200-line one.

Every non-obvious decision (family, each phase outcome, status and output
layout) is written to ``evidence`` as ``{"rule", "line", "text"}`` so a
reviewer can reproduce the verdict from the log.  ``errors`` and
``warnings`` carry ``{"kind", "message", "line", "rank"}``.

The detection rules are those of ``design/M1.5-2_Spec_and_roadmap.md``
section 3.3 and the contract in ``specs/interfaces.md`` section 4.  The
pattern table, script-to-family map, phase order and marker constants are
imported from :mod:`app.parsing.errors`; this module adds only the marker
literals that the fixtures require (see the constants below) and the
decision logic.

Nothing here imports NiceGUI: parsing must stay usable from a worker
thread and from the command line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from app.parsing.errors import (
    DECODING_OUTPUT,
    DONE_LOADING_EVAL_DATA,
    DONE_LOADING_MODEL,
    DONE_LOADING_TOKENIZER,
    DONE_LOADING_TRAIN_DATA,
    DONE_PREPARING_INPUT,
    FAMILY_CHECKPOINT_EXPORT,
    FAMILY_FINE_TUNE,
    FAMILY_INFERENCE,
    FAMILY_UNKNOWN,
    PATTERNS,
    PHASE_ORDER,
    SHARD_WRITING_COMPLETE,
    TOKENIZING_INPUTS,
    VERDICTS_BANNER,
    family_for_script,
)
from app.registry import MODELS

__all__ = ["detect"]


# ---------------------------------------------------------------------------
# Local literals forced by the fixtures
# ---------------------------------------------------------------------------
# These are marker literals, not regular expressions: the pattern table stays
# the single source of matching truth.  ``Done Loading the Data`` is
# deliberately not a phase marker (s14 prints it and then fails while it is
# still building the prompt), and the banner itself marks the end of data
# preparation in the banner layout.

_END_LOADING_MODEL: Final = "End - Loading the model -"
_THINK_MARKER: Final = "<think>"
_RESPONSE_PREFIX: Final = "### Response:"
_LOADING_WEIGHTS: Final = "Loading weights"
_TRACEBACK_START: Final = "Traceback (most recent call last):"

_RANK_PREFIX: Final = re.compile(r"^\[rank(?P<rank>\d+)\]: ?")
_TS_FORMAT: Final = "%Y-%m-%d %H:%M:%S.%f"

# Full registry ids only.  The logs mention the short display name
# ("Mistral-Small-24B-Instruct-2501") and an unrelated discussion URL, neither
# of which is the id of the model that produced the run.
_MODEL_IDS: Final = tuple(MODELS)

#: The phase that decides the run: a failure before (or at) it is an error,
#: a failure after it is a success with a warning.
_CORE_PHASE: Final = {
    FAMILY_FINE_TUNE: "fine_tuning",
    FAMILY_INFERENCE: "generation",
    FAMILY_CHECKPOINT_EXPORT: "model_saving",
}

_FAILED: Final = "failed"
_TERMINATED: Final = "terminated"
_SUCCEEDED: Final = "succeeded"
_NOT_REACHED: Final = "not_reached"


@dataclass(frozen=True)
class _Hit:
    """One matched line: its 1-based number, text and rank."""

    line: int
    text: str
    rank: int | None


def _trim(text: str, limit: int = 160) -> str:
    """Strip a line and cap it at ``limit`` characters for evidence."""

    cleaned = text.strip()
    return cleaned if len(cleaned) <= limit else cleaned[:limit]


def _exception_kind(text: str) -> str:
    """Return a stable lowercase token for an exception line."""

    if "FileNotFoundError" in text:
        return "file_not_found"
    if "IndexError" in text:
        return "index_error"
    if "TypeError" in text:
        return "type_error"
    if "RuntimeError" in text:
        return "runtime_error"
    if "ValueError" in text:
        return "value_error"
    if "ImportError" in text:
        return "import_error"
    return "exception"


def _duration(started: str | None, finished: str | None) -> float | None:
    """Return ``finished - started`` in seconds, or None when unknown."""

    if started is None or finished is None:
        return None
    try:
        start = datetime.strptime(started, _TS_FORMAT)
        end = datetime.strptime(finished, _TS_FORMAT)
    except ValueError:
        return None
    return round((end - start).total_seconds(), 6)


class _Tracker:
    """Bounded, first-seen state accumulated during the single pass."""

    def __init__(self, source_path: str) -> None:
        self.source_path = source_path
        self.hits: dict[str, _Hit] = {}
        self.script: str | None = None
        self.script_hit: tuple[str, _Hit] | None = None
        self.phase_hint: tuple[str, _Hit] | None = None
        self.model_id: str | None = None

        self.fatal: _Hit | None = None
        self.fatal_kind: str | None = None
        self.missing: _Hit | None = None
        self.time_limit: _Hit | None = None
        self.traceback: _Hit | None = None

        self.first_ts: str | None = None
        self.last_ts: str | None = None
        self.last_ts_line: int | None = None

        self.max_training_total = 0
        self.training_complete: _Hit | None = None
        self.last_nonempty: _Hit | None = None

    # -- recording helpers ------------------------------------------------

    def _first(self, key: str, hit: _Hit) -> None:
        if key not in self.hits:
            self.hits[key] = hit

    def _first_of(self, *keys: str) -> _Hit | None:
        found = [self.hits[key] for key in keys if key in self.hits]
        return min(found, key=lambda hit: hit.line) if found else None

    # -- the single pass --------------------------------------------------

    def feed(self, text: str, line: int, rank: int | None) -> None:
        """Consume one line whose ``[rankN]: `` prefix has been stripped."""

        stripped = text.strip()
        if stripped:
            self.last_nonempty = _Hit(line, stripped, rank)

        if self.model_id is None:
            for model_id in _MODEL_IDS:
                if model_id in text:
                    self.model_id = model_id
                    break

        # Script name -> family (precedence 1).
        if '.py"' in text:
            match = PATTERNS["script_name"].search(text)
            if match:
                family = family_for_script(match.group("script"))
                if family != FAMILY_UNKNOWN and self.script_hit is None:
                    self.script = match.group("script")
                    self.script_hit = (family, _Hit(line, text, rank))

        # Phase timestamps: start/end times and the precedence-2 family hint.
        if "-" in text and ":" in text:
            match = PATTERNS["phase_timestamp"].match(text)
            if match:
                timestamp = match.group("ts")
                if self.first_ts is None:
                    self.first_ts = timestamp
                self.last_ts = timestamp
                self.last_ts_line = line
                if self.phase_hint is None:
                    if DONE_LOADING_TRAIN_DATA in text:
                        self.phase_hint = (FAMILY_FINE_TUNE, _Hit(line, text, rank))
                    elif TOKENIZING_INPUTS in text or DECODING_OUTPUT in text:
                        self.phase_hint = (FAMILY_INFERENCE, _Hit(line, text, rank))

        # Content markers (precedence 3) and layout markers.
        if text[:1] in ("*", "#"):
            match = PATTERNS["banner"].match(text)
            if match:
                section = match.group(1)
                self._first(f"banner_{section.lower()}", _Hit(line, text, rank))
        if " | " in text and PATTERNS["verdict_row"].match(text):
            self._first("table_row", _Hit(line, text, rank))
        if text.startswith("{'loss': '"):
            self._first("loss_dict", _Hit(line, text, rank))
        if SHARD_WRITING_COMPLETE in text:
            self._first("shard_writing", _Hit(line, text, rank))
        if _THINK_MARKER in text:
            self._first("think", _Hit(line, text, rank))
        if text.startswith(_RESPONSE_PREFIX):
            self._first("response", _Hit(line, text, rank))

        # Phase-completion markers from the shared constant table.
        if DONE_LOADING_MODEL in text:
            self._first("done_loading_model", _Hit(line, text, rank))
        if _END_LOADING_MODEL in text:
            self._first("end_loading_model", _Hit(line, text, rank))
        if DONE_LOADING_TOKENIZER in text:
            self._first("done_loading_tokenizer", _Hit(line, text, rank))
        if DONE_PREPARING_INPUT in text:
            self._first("done_preparing_input", _Hit(line, text, rank))
        if DONE_LOADING_EVAL_DATA in text:
            self._first("done_loading_eval_data", _Hit(line, text, rank))

        # Progress bars: a "Loading weights" completion closes model loading;
        # an unlabelled bar carries the training loop.  The training total is
        # the largest unlabelled total in the file, so the small eval bar
        # (e.g. 386/386) is never mistaken for a finished run.
        if "|" in text and "/" in text and "[" in text:
            match = PATTERNS["progress_bar"].search(text)
            if match:
                done = int(match.group(1))
                total = int(match.group(2))
                if total > 0 and _LOADING_WEIGHTS in text and done == total:
                    self._first("loading_weights_complete", _Hit(line, text, rank))
                prefix = text.split("|", 1)[0]
                if total > 0 and not any(char.isalpha() for char in prefix):
                    if total > self.max_training_total:
                        self.max_training_total = total
                        self.training_complete = None
                    if (
                        done == total
                        and total == self.max_training_total
                        and self.training_complete is None
                    ):
                        self.training_complete = _Hit(line, text, rank)

        # Fatal lines: keep the first one only (earliest line wins).
        if "Configured CUDA binary not found" in text or "ModuleNotFoundError" in text:
            hit = _Hit(line, text, rank)
            if self.missing is None:
                self.missing = hit
            if self.fatal is None:
                self.fatal = hit
                self.fatal_kind = "missing_library"
        elif "OutOfMemoryError" in text or "out of memory" in text:
            if self.fatal is None:
                self.fatal = _Hit(line, text, rank)
                self.fatal_kind = "oom"
        elif ("Error: " in text or "Exception: " in text) and PATTERNS[
            "exception_line"
        ].match(text):
            if self.fatal is None:
                self.fatal = _Hit(line, text, rank)
                self.fatal_kind = _exception_kind(text)
        elif "exitcode" in text and "-11" in text and PATTERNS["exitcode"].search(text):
            if self.fatal is None:
                self.fatal = _Hit(line, text, rank)
                self.fatal_kind = "exit_code"
        elif "ChildFailedError" in text and PATTERNS["child_failed"].search(text):
            if self.fatal is None:
                self.fatal = _Hit(line, text, rank)
                self.fatal_kind = "child_failed"
        elif text.startswith("srun: error:") and PATTERNS["srun_error"].match(text):
            if self.fatal is None:
                self.fatal = _Hit(line, text, rank)
                self.fatal_kind = "srun_error"

        # SLURM time limit and a traceback fallback for truncated tracebacks.
        if "DUE TO TIME LIMIT" in text and PATTERNS["time_limit"].search(text):
            if self.time_limit is None:
                self.time_limit = _Hit(line, text, rank)
        if text.startswith(_TRACEBACK_START) and self.traceback is None:
            self.traceback = _Hit(line, text, rank)

    # -- end-of-file decisions -------------------------------------------

    def _family(self) -> tuple[str, _Hit | None, str]:
        if self.script_hit is not None:
            family, hit = self.script_hit
            return family, hit, "family_script"
        if self.phase_hint is not None:
            family, hit = self.phase_hint
            return family, hit, "family_phase_timestamp"

        inference_hits = [
            self.hits[key]
            for key in ("banner_inputs", "banner_output", "banner_verdicts", "table_row")
            if key in self.hits
        ]
        if inference_hits:
            return FAMILY_INFERENCE, min(inference_hits, key=lambda hit: hit.line), "family_content"
        if "loss_dict" in self.hits:
            return FAMILY_FINE_TUNE, self.hits["loss_dict"], "family_content"
        if "shard_writing" in self.hits:
            return FAMILY_CHECKPOINT_EXPORT, self.hits["shard_writing"], "family_content"
        return FAMILY_UNKNOWN, None, "family_none"

    def _terminal(self) -> tuple[_Hit | None, bool]:
        """Return the terminal event and whether it forces a setup failure."""

        candidates = [hit for hit in (self.fatal, self.time_limit) if hit is not None]
        if self.missing is not None and (
            not candidates or self.missing.line <= min(hit.line for hit in candidates)
        ):
            return self.missing, True
        if candidates:
            return min(candidates, key=lambda hit: hit.line), False
        if self.traceback is not None:
            return self.traceback, False
        return None, False

    def _layout(self, family: str) -> tuple[str | None, _Hit | None]:
        if family != FAMILY_INFERENCE:
            return None, None
        banner_hits = [
            self.hits[key]
            for key in ("banner_inputs", "banner_output", "banner_verdicts")
            if key in self.hits
        ]
        if banner_hits:
            return "banner", min(banner_hits, key=lambda hit: hit.line)
        if "think" in self.hits:
            return "think", self.hits["think"]
        if "table_row" in self.hits:
            return "table_only", self.hits["table_row"]
        return None, None

    def _clean_tail(self) -> bool:
        """True when an inference log ends on a complete output element."""

        last = self.last_nonempty
        if last is None:
            return False
        text = last.text
        if text.startswith(_RESPONSE_PREFIX):
            return True
        if PATTERNS["verdict_row"].match(text):
            return True
        match = PATTERNS["banner"].match(text)
        return bool(match and match.group(1) == VERDICTS_BANNER)

    def _done_lines(self, family: str, terminal: _Hit | None) -> tuple[dict[str, _Hit | None], set[str]]:
        """Return each phase's completion line and the set of implied phases."""

        phases = PHASE_ORDER[family]
        done: dict[str, _Hit | None] = {name: None for name in phases}
        implied: set[str] = set()

        def usable(hit: _Hit | None) -> bool:
            return hit is not None and (terminal is None or hit.line <= terminal.line)

        if family == FAMILY_INFERENCE:
            model = self._first_of("done_loading_model", "end_loading_model")
            tokenizer = self.hits.get("done_loading_tokenizer")
            if model is not None and tokenizer is not None:
                done["setup"] = model if model.line >= tokenizer.line else tokenizer
            done["data_preparation"] = self._first_of(
                "done_preparing_input", "banner_inputs"
            )
            verdict = self._first_of("verdicts_banner", "table_row")
            if usable(verdict):
                done["verdict_output"] = verdict
                done["generation"] = verdict
                implied.add("generation")
        elif family == FAMILY_FINE_TUNE:
            done["setup"] = self.hits.get("done_loading_eval_data")
            done["fine_tuning"] = self.training_complete
            done["checkpoint_creation"] = self.hits.get("shard_writing")
        elif family == FAMILY_CHECKPOINT_EXPORT:
            done["setup"] = self._first_of(
                "done_loading_eval_data", "loading_weights_complete", "done_loading_model"
            )
            done["model_saving"] = self.hits.get("shard_writing")

        # A completed later phase implies that every earlier phase finished.
        for index, name in enumerate(phases):
            if done[name] is None:
                later = [
                    done[other]
                    for other in phases[index + 1 :]
                    if usable(done[other])
                ]
                if later:
                    done[name] = min(later, key=lambda hit: hit.line)
                    implied.add(name)
        return done, implied

    def result(self) -> dict[str, Any]:
        family, family_hit, family_rule = self._family()
        layout, layout_hit = self._layout(family)
        result: dict[str, Any] = {
            "source_path": self.source_path,
            "script": self.script,
            "model_id": self.model_id,
            "family": family,
            "status": "unknown",
            "scenario": f"{family}_unknown",
            "phases": [],
            "failed_phase": None,
            "output_format": layout,
            "started_at": self.first_ts,
            "finished_at": None,
            "duration_seconds": None,
            "progress": {},
            "errors": [],
            "warnings": [],
            "evidence": [],
            "notes": [],
        }

        evidence: list[dict[str, Any]] = result["evidence"]
        evidence.append(
            {
                "rule": family_rule,
                "line": family_hit.line if family_hit is not None else None,
                "text": _trim(family_hit.text) if family_hit is not None else "no family marker found",
            }
        )

        if family == FAMILY_UNKNOWN:
            result["notes"].append(
                "Could not classify the log: no script name, phase timestamp or content marker was found."
            )
            result["finished_at"] = self.last_ts
            result["duration_seconds"] = _duration(self.first_ts, self.last_ts)
            return result

        phases = list(PHASE_ORDER[family])
        terminal, force_setup = self._terminal()
        event: str | None = None
        if terminal is not None:
            event = _TERMINATED if terminal is self.time_limit else _FAILED

        done, implied = self._done_lines(family, terminal)
        outcomes: dict[str, str] = {}
        clean_exit = False

        if terminal is not None:
            current: int | None = None
            if force_setup:
                current = 0
            else:
                for index, name in enumerate(phases):
                    hit = done[name]
                    if hit is None or hit.line > terminal.line:
                        current = index
                        break
            if current is None:
                for name in phases:
                    outcomes[name] = _SUCCEEDED
            else:
                for index, name in enumerate(phases):
                    if index < current:
                        outcomes[name] = _SUCCEEDED
                    elif index == current:
                        outcomes[name] = event or _FAILED
                    else:
                        outcomes[name] = _NOT_REACHED
        else:
            if family == FAMILY_INFERENCE:
                clean_exit = self._clean_tail()
            elif family == FAMILY_FINE_TUNE:
                clean_exit = (
                    done["fine_tuning"] is not None
                    and done["checkpoint_creation"] is not None
                )
            elif family == FAMILY_CHECKPOINT_EXPORT:
                clean_exit = done["model_saving"] is not None

            if clean_exit:
                for name in phases:
                    outcomes[name] = _SUCCEEDED
            else:
                current = next(
                    (index for index, name in enumerate(phases) if done[name] is None),
                    None,
                )
                for index, name in enumerate(phases):
                    if current is None or index < current:
                        outcomes[name] = _SUCCEEDED
                    elif index == current:
                        outcomes[name] = _TERMINATED
                    else:
                        outcomes[name] = _NOT_REACHED

        failed_phase = next(
            (name for name in phases if outcomes[name] in (_FAILED, _TERMINATED)),
            None,
        )
        if all(outcome == _SUCCEEDED for outcome in outcomes.values()):
            status = "success"
        elif failed_phase is None:
            status = "unknown"
        else:
            failed_index = phases.index(failed_phase)
            core_index = phases.index(_CORE_PHASE[family])
            if failed_index <= core_index:
                status = "error" if outcomes[failed_phase] == _FAILED else "terminated"
            else:
                status = "success"

        terminal_hit = terminal if terminal is not None else self.last_nonempty
        result["phases"] = [
            {
                "name": name,
                "outcome": outcomes[name],
                "evidence_line": (
                    done[name].line
                    if outcomes[name] == _SUCCEEDED and done[name] is not None
                    else terminal_hit.line
                    if outcomes[name] in (_FAILED, _TERMINATED) and terminal_hit is not None
                    else None
                ),
            }
            for name in phases
        ]
        result["failed_phase"] = failed_phase
        result["status"] = status
        result["scenario"] = f"{family}_{status}"

        # Every resolved phase gets one non-empty evidence entry.
        for name in phases:
            outcome = outcomes[name]
            if outcome == _NOT_REACHED:
                continue
            if outcome == _SUCCEEDED:
                hit = done[name]
                if hit is None:
                    text = "clean end of log"
                elif name in implied:
                    text = _trim(f"implied by a later phase: {hit.text}")
                else:
                    text = _trim(hit.text)
                evidence.append(
                    {
                        "rule": f"phase_{name}_{outcome}",
                        "line": hit.line if hit is not None else None,
                        "text": text,
                    }
                )
            else:
                hit = terminal if terminal is not None else self.last_nonempty
                evidence.append(
                    {
                        "rule": f"phase_{name}_{outcome}",
                        "line": hit.line if hit is not None else None,
                        "text": _trim(hit.text) if hit is not None else "log ended",
                    }
                )

        if layout is not None and layout_hit is not None:
            evidence.append(
                {
                    "rule": f"layout_{layout}",
                    "line": layout_hit.line,
                    "text": _trim(layout_hit.text),
                }
            )

        evidence.append(
            {
                "rule": "status_rule",
                "line": terminal_hit.line if terminal_hit is not None else None,
                "text": f"{status}: core phase {_CORE_PHASE[family]}, failed_phase {failed_phase}",
            }
        )

        # Errors: the terminal event, or a truncated end of file.
        if terminal is not None:
            if terminal is self.time_limit:
                kind = "slurm_time_limit"
            elif terminal is self.missing:
                kind = "missing_library"
            else:
                kind = self.fatal_kind or "traceback"
            result["errors"].append(
                {
                    "kind": kind,
                    "message": terminal.text.strip(),
                    "line": terminal.line,
                    "rank": terminal.rank,
                }
            )
        elif not clean_exit and self.last_nonempty is not None:
            result["errors"].append(
                {
                    "kind": "truncated",
                    "message": f"log ends before phase {failed_phase} completes",
                    "line": self.last_nonempty.line,
                    "rank": self.last_nonempty.rank,
                }
            )

        # Warnings: the success-with-warning cases of the contract.
        warning_hit = terminal if terminal is not None else self.last_nonempty
        if status == "success" and failed_phase is not None:
            kind = (
                "phase_failed_after_core"
                if outcomes[failed_phase] == _FAILED
                else "phase_terminated_after_core"
            )
            result["warnings"].append(
                {
                    "kind": kind,
                    "message": (
                        f"{failed_phase} {outcomes[failed_phase]} after the core "
                        f"phase {_CORE_PHASE[family]} succeeded"
                    ),
                    "line": warning_hit.line if warning_hit is not None else None,
                    "rank": warning_hit.rank if warning_hit is not None else None,
                }
            )
            result["notes"].append(
                f"The core phase {_CORE_PHASE[family]} succeeded, so the run is "
                "reported as success with a warning."
            )
        elif status == "success" and terminal is not None:
            # Every phase completed, but the run was still cut short (e.g. a
            # time limit after the last output).  Mirror the failure rule.
            result["warnings"].append(
                {
                    "kind": "ended_after_core",
                    "message": (
                        f"the run ended after every phase, including the core "
                        f"phase {_CORE_PHASE[family]}, had succeeded"
                    ),
                    "line": warning_hit.line if warning_hit is not None else None,
                    "rank": warning_hit.rank if warning_hit is not None else None,
                }
            )
            result["notes"].append(
                "Every phase succeeded, but the run then stopped early; it is "
                "reported as success with a warning."
            )
        elif status == "terminated":
            result["notes"].append(
                "The run stopped before completion: SLURM time limit or a truncated log."
            )

        # Timestamps: a fatal line without a timestamp leaves finished_at None.
        if terminal is not None:
            finished_at = self.last_ts if self.last_ts_line == terminal.line else None
        else:
            finished_at = self.last_ts
        result["finished_at"] = finished_at
        result["duration_seconds"] = _duration(self.first_ts, finished_at)

        return result


def detect(path: str | Path) -> dict[str, Any]:
    """Detect the family, phases, status and layout of ``path``.

    The function makes one streaming pass over the file.  It never loads the
    whole log and keeps only first-seen markers, so memory is flat.  It
    raises ``OSError`` for an unreadable path (a caller error); malformed,
    undecodable or truncated log content is reported through the returned
    dictionary instead of raising.

    The returned dictionary has exactly the seventeen keys of
    :class:`app.parsing.ParseResult`.
    """

    source_path = str(path)
    tracker = _Tracker(source_path)
    with open(path, encoding="utf-8", errors="replace", newline=None) as handle:
        for number, raw_line in enumerate(handle, start=1):
            text = raw_line.rstrip("\r\n")
            rank: int | None = None
            if text.startswith("[rank"):
                match = _RANK_PREFIX.match(text)
                if match:
                    rank = int(match.group("rank"))
                    text = text[match.end() :]
            tracker.feed(text, number, rank)
    return tracker.result()
