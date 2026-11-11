"""Declarative detection data shared by the log parser (issue #28).

This module is data, not logic: the pattern table, the script-name-to-family
map, the per-family phase order and the phase-completion markers. Detection
and the family parsers consume these tables in issue #30.

Matching conventions
--------------------
- Every regular expression is compiled once, at import time.
- Lines are read with ``newline=None`` (universal newlines), so a line has
  no trailing ``\\r`` when it reaches a pattern.
- A leading ``[rankN]: `` prefix is stripped before matching; the caller
  keeps ``N`` as the rank. The two-space indent of a traceback ``File`` line
  therefore remains and is not part of any pattern.
- Where two rules can match, the first match wins, in declaration order.
  This is why :data:`SCRIPT_FAMILY` is an insertion-ordered dict and why
  consumers must test :data:`PATTERNS` keys explicitly rather than assuming
  that iteration order implies priority.

Every pattern here occurs in at least one log fixture under
``tests/fixtures/logs``. The contract that consumes this module is
``specs/interfaces.md`` section 4 and the design note is
``design/M1.5-2_Spec_and_roadmap.md`` section 3.3.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = [
    "FAMILY_CHECKPOINT_EXPORT",
    "FAMILY_FINE_TUNE",
    "FAMILY_INFERENCE",
    "FAMILY_UNKNOWN",
    "PATTERNS",
    "PHASE_ORDER",
    "SCRIPT_FAMILY",
    "SCRIPT_FAMILY_PATTERNS",
    "DECODING_OUTPUT",
    "DONE_LOADING_EVAL_DATA",
    "DONE_LOADING_MODEL",
    "DONE_LOADING_TOKENIZER",
    "DONE_LOADING_TRAIN_DATA",
    "DONE_PREPARING_INPUT",
    "SHARD_WRITING_COMPLETE",
    "TOKENIZING_INPUTS",
    "VERDICTS_BANNER",
    "family_for_script",
]


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

FAMILY_FINE_TUNE: Final = "fine_tune"
FAMILY_INFERENCE: Final = "inference"
FAMILY_CHECKPOINT_EXPORT: Final = "checkpoint_export"
FAMILY_UNKNOWN: Final = "unknown"


# ---------------------------------------------------------------------------
# Pattern table
# ---------------------------------------------------------------------------
# Keys are stable rule names. They are stored in ``ParseResult["evidence"]``
# as ``{"rule": <key>, ...}``, so renaming a key is a contract change.
#
# Grouped by the job they do:
#   identity/layout - phase_timestamp, script_name, banner, verdict_row
#   progress        - training_loss, eval_loss, progress_bar, shard_writing
#   failure         - traceback_start, exception_line, out_of_memory,
#                     missing_library, exitcode, child_failed, srun_error
#   termination     - time_limit

PATTERNS: dict[str, re.Pattern[str]] = {
    # "Done Loading the Eval Data :  2026-02-07 20:23:07.639352"
    "phase_timestamp": re.compile(
        r"^(?P<label>[A-Za-z][\w ()'-]+?)\s+[:-]\s+(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)"
    ),
    # '  File "/project/0000000/user/fineTuneMistral.py", line 214, in <module>'
    "script_name": re.compile(
        r'File ".*/(?P<script>[\w-]+\.py)", line \d+, in <module>'
    ),
    # "*************************************INPUTS****************************************"
    "banner": re.compile(r"^[*#]{5,}\s*(INPUTS|OUTPUT|VERDICTS|LENGTH)\s*[*#]{5,}$"),
    # "12 | Verdict: True"
    "verdict_row": re.compile(r"^(\d+) \| (.*)$"),
    # "{'loss': '0.3951', 'grad_norm': ...}"
    "training_loss": re.compile(r"^\{'loss': '"),
    # "{'eval_loss': '0.6387', 'eval_runtime': ...}"
    "eval_loss": re.compile(r"\{'eval_loss': '([\d.e-]+)'"),
    # " 50%|.....| 4969/9936 [3:22:58<195:46:34, 141.90s/it]"
    "progress_bar": re.compile(r"\|\s*(\d+)/(\d+) \["),
    # "Writing model shards: 100%|...| 1/1 [00:01<00:00,  1.17s/it]"
    "shard_writing": re.compile(r"Writing model shards: 100%"),
    # "Traceback (most recent call last):"
    "traceback_start": re.compile(r"^Traceback \(most recent call last\):"),
    # "RuntimeError: Configured CUDA binary not found at ..."
    "exception_line": re.compile(r"^([\w.]+(Error|Exception)): (.*)$"),
    # "torch.OutOfMemoryError: CUDA out of memory. ..."
    "out_of_memory": re.compile(r"OutOfMemoryError|CUDA error: out of memory"),
    # "ModuleNotFoundError: No module named 'pandas'"
    # "RuntimeError: Configured CUDA binary not found at ..."
    "missing_library": re.compile(r"ModuleNotFoundError|Configured CUDA binary not found"),
    # "failed (exitcode: -11) local_rank: 0 (pid: 257015) of binary: ..."
    # "  exitcode  : -11 (pid: 257015)"
    "exitcode": re.compile(r"exitcode\s*:\s*-11"),
    # "torch.distributed.elastic.multiprocessing.errors.ChildFailedError: "
    "child_failed": re.compile(r"ChildFailedError"),
    # "srun: error: fc00000: tasks 0-3: Exited with exit code 1"
    "srun_error": re.compile(r"^srun: error:"),
    # "slurmstepd: error: *** JOB 29242690 ON fc00000 CANCELLED AT 2026-03-27T18:48:59 DUE TO TIME LIMIT ***"
    "time_limit": re.compile(
        r"slurmstepd: error: \*\*\* JOB \d+ .* CANCELLED AT .* DUE TO TIME LIMIT \*\*\*"
    ),
}


# ---------------------------------------------------------------------------
# Script name -> family
# ---------------------------------------------------------------------------
# Keys are regular expressions anchored at the start of the script basename;
# values are family names. Declaration order is the priority order: the first
# matching entry wins. ``family_for_script`` does the matching; do not iterate
# the dict directly, because its keys are unanchored pattern sources.
#
# Examples from the fixtures:
#   fineTuneMistral.py                        -> fine_tune
#   inferenceFactCheckerThanh.py              -> inference
#   inferenceFactCheckerThanhWithoutEvidence.py -> inference
#   distZeroShot.py                           -> inference
#   saveMistralCheckpoint.py (not in fixtures) -> checkpoint_export

SCRIPT_FAMILY: dict[str, str] = {
    r"^fineTune": FAMILY_FINE_TUNE,
    r"^inference": FAMILY_INFERENCE,
    r"^distZeroShot": FAMILY_INFERENCE,
    r"^save.*Checkpoint": FAMILY_CHECKPOINT_EXPORT,
}

# Compiled once at import time; ordered in lockstep with SCRIPT_FAMILY.
SCRIPT_FAMILY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), family) for pattern, family in SCRIPT_FAMILY.items()
)


def family_for_script(name: str | None) -> str:
    """Return the family for a script basename, or ``"unknown"``.

    ``name`` is the ``script`` group of ``PATTERNS["script_name"]``, e.g.
    ``"fineTuneMistral.py"``. A path prefix, if present, is ignored. The
    match is case-sensitive and anchored at the start, and the first entry
    of :data:`SCRIPT_FAMILY_PATTERNS` wins.
    """
    if not name:
        return FAMILY_UNKNOWN
    base = re.split(r"[\\/]", name)[-1]
    for pattern, family in SCRIPT_FAMILY_PATTERNS:
        if pattern.match(base):
            return family
    return FAMILY_UNKNOWN


# ---------------------------------------------------------------------------
# Phase order
# ---------------------------------------------------------------------------
# The phase lists match tests/fixtures/logs/expected.json, which is the
# oracle for issue #30. Phase names are contract, not implementation detail.

PHASE_ORDER: dict[str, tuple[str, ...]] = {
    FAMILY_FINE_TUNE: ("setup", "fine_tuning", "checkpoint_creation", "model_saving"),
    FAMILY_INFERENCE: ("setup", "data_preparation", "generation", "verdict_output"),
    FAMILY_CHECKPOINT_EXPORT: ("setup", "model_saving"),
}


# ---------------------------------------------------------------------------
# Phase-completion markers
# ---------------------------------------------------------------------------
# Literal substrings, not regexes. The phase timestamp prefix is not part of
# a marker: compare against a line after the "[rankN]: " prefix is stripped.
#
# Phase map (design section 3.3):
#   fine_tune.setup                 DONE_LOADING_EVAL_DATA
#   fine_tune.fine_tuning           progress_bar where done == total
#   fine_tune.checkpoint_creation   SHARD_WRITING_COMPLETE
#   fine_tune.model_saving          clean exit (no literal marker)
#   inference.setup                 DONE_LOADING_MODEL and DONE_LOADING_TOKENIZER
#   inference.data_preparation      DONE_PREPARING_INPUT
#   inference.generation            progress bar where done == total
#   inference.verdict_output        VERDICTS_BANNER (or both verdict tables)
#   checkpoint_export.setup         model-load markers, as for fine_tune
#   checkpoint_export.model_saving  SHARD_WRITING_COMPLETE

DONE_LOADING_TRAIN_DATA: Final = "Done Loading the Train Data"
DONE_LOADING_EVAL_DATA: Final = "Done Loading the Eval Data"
TOKENIZING_INPUTS: Final = "Tokenizing Inputs"
DECODING_OUTPUT: Final = "Decoding the Output"
DONE_LOADING_MODEL: Final = "Done Loading the Model"
DONE_LOADING_TOKENIZER: Final = "Done Loading the Tokenizer"
DONE_PREPARING_INPUT: Final = "Done Preparing the Input"
SHARD_WRITING_COMPLETE: Final = "Writing model shards: 100%"
# The section tag inside the banner line, e.g. "*****VERDICTS*****". Match the
# whole line with PATTERNS["banner"]; this constant is the value to compare
# against the banner's first group.
VERDICTS_BANNER: Final = "VERDICTS"
