"""Score extracted verdicts against the gold ``label`` field.

Pipeline position (``design/M1.5-2_Spec_and_roadmap.md`` section 1)::

    app.parsing.verdicts        (#34, not merged yet)
        -> VerdictRecord        (defined here until #34 lands)
        -> app.scoring.score_verdicts        (this module, issue #35)
        -> ScoreReport          (specs/interfaces.md section 5)
        -> app.store.upsert_score, app.ui.results

Three guarantees hold for every report:

1. The denominator is always ``records``, the size of the gold set. A
   partial run (log 15) is never rescored on a shorter denominator: the
   missing records are ``unfinished`` and count as wrong predictions.
2. Every prediction that is not ``exact`` or ``recovered`` is scored as
   ``__INVALID__``, including ``empty``, ``out_of_set``, ``unparsed`` and
   ``unfinished``.
3. Every division by zero yields ``0.0``. A report never contains ``NaN``
   and never raises because a class has no gold support or no prediction.

The module is pure: no file I/O, no database and no NiceGUI import. Only
the optional :func:`load_gold_labels` helper reads a file.

``VerdictRecord`` is defined here because issue #34
(``app/parsing/verdicts.py``) is not merged yet. When it lands it should
produce exactly this shape: ``index`` is the 0-based position in the gold
set, ``raw`` is the decoded text as extracted, ``parsed`` is the normalised
verdict (``None`` when unusable) and ``kind`` is one of the five verdict
kinds of design section 3.5. Scoring never imports from :mod:`app.parsing`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Literal, Sequence, TypedDict

__all__ = [
    "VerdictRecord",
    "ScoreReport",
    "score_verdicts",
    "render_report_markdown",
    "load_gold_labels",
]


INVALID_LABEL: Final[str] = "__INVALID__"

VerdictKind = Literal["exact", "recovered", "empty", "out_of_set", "unparsed"]

#: Kinds that are scored as a correct prediction; every other kind is invalid.
CORRECT_KINDS: Final[frozenset[str]] = frozenset({"exact", "recovered"})

#: The five kinds a verdict extractor may emit (design section 3.5).
VALID_KINDS: Final[frozenset[str]] = frozenset(
    {"exact", "recovered", "empty", "out_of_set", "unparsed"}
)

KIND_ORDER: Final[tuple[str, ...]] = (
    "exact",
    "recovered",
    "empty",
    "out_of_set",
    "unparsed",
)

METRIC_KEYS: Final[tuple[str, ...]] = (
    "accuracy",
    "macro_f1",
    "micro_f1",
    "weighted_f1",
    "precision",
    "recall",
    "f1",
    "confusion",
)


class VerdictRecord(TypedDict):
    """One extracted verdict before scoring.

    Issue #34 (``app/parsing/verdicts.py``) will produce this shape; it is
    declared here because #34 is not merged yet. ``index`` is the 0-based
    position in the gold set, ``raw`` is the decoded text as extracted,
    ``parsed`` is the normalised verdict (``None`` when unusable) and
    ``kind`` is the classification from design section 3.5.
    """

    index: int
    raw: str
    parsed: str | None
    kind: VerdictKind


class MetricBlock(TypedDict):
    """The shared metric block used by exact and inclusive scoring.

    The keys mirror the top-level metric fields of ``ScoreReport``.
    """

    accuracy: float
    macro_f1: float
    micro_f1: float
    weighted_f1: float
    precision: dict[str, float]
    recall: dict[str, float]
    f1: dict[str, float]
    confusion: dict[str, dict[str, int]]


class ScoreReport(TypedDict):
    """The frozen result shape of ``specs/interfaces.md`` section 5.

    Every key is always present. ``records`` is the gold size and the only
    denominator; ``complete`` is ``False`` exactly when ``unfinished > 0``.
    The top-level metric fields mirror ``inclusive_metrics``.
    """

    marker: str
    labels: list[str]
    source: Literal["parsed", "script_table"]
    records: int
    complete: bool
    exact: int
    recovered: int
    empty: int
    out_of_set: int
    unparsed: int
    unfinished: int
    empty_rate: float
    invalid_rate: float
    accuracy: float
    macro_f1: float
    micro_f1: float
    weighted_f1: float
    precision: dict[str, float]
    recall: dict[str, float]
    f1: dict[str, float]
    confusion: dict[str, dict[str, int]]
    exact_metrics: MetricBlock
    inclusive_metrics: MetricBlock


def _rounded(value: float) -> float:
    """Round a metric to four decimals, as the contract example shows."""

    return round(value, 4)


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Divide, yielding ``0.0`` instead of raising or producing ``NaN``."""

    if denominator == 0:
        return 0.0
    return _rounded(numerator / denominator)


def _metric_block(
    gold: Sequence[str],
    predicted: Sequence[str],
    labels: Sequence[str],
) -> MetricBlock:
    """Compute one metric block from aligned gold and predicted sequences.

    ``predicted`` contains either a configured label or ``__INVALID__``.
    Precision, recall, F1 and the macro/weighted means are computed from
    the confusion counts; every zero denominator becomes ``0.0``.
    """

    records = len(gold)
    support = {label: 0 for label in labels}
    predicted_count = {label: 0 for label in labels}
    true_positive = {label: 0 for label in labels}
    confusion = {
        gold_label: {column: 0 for column in (*labels, INVALID_LABEL)}
        for gold_label in labels
    }

    correct = 0
    for gold_label, prediction in zip(gold, predicted):
        support[gold_label] += 1
        confusion[gold_label][prediction] += 1
        if prediction in predicted_count:
            predicted_count[prediction] += 1
        if prediction == gold_label:
            true_positive[gold_label] += 1
            correct += 1

    raw_precision: dict[str, float] = {}
    raw_recall: dict[str, float] = {}
    raw_f1: dict[str, float] = {}
    for label in labels:
        tp = true_positive[label]
        raw_precision[label] = (
            tp / predicted_count[label] if predicted_count[label] else 0.0
        )
        raw_recall[label] = tp / support[label] if support[label] else 0.0
        # F1 from counts: 2TP / (2TP + FP + FN); the denominator is zero
        # exactly when the class has neither gold nor predicted records.
        denominator = (
            2 * tp
            + (predicted_count[label] - tp)
            + (support[label] - tp)
        )
        raw_f1[label] = 2 * tp / denominator if denominator else 0.0

    macro_f1 = (
        _rounded(sum(raw_f1.values()) / len(labels)) if labels else 0.0
    )
    weighted_f1 = (
        _rounded(
            sum(raw_f1[label] * support[label] for label in labels) / records
        )
        if records
        else 0.0
    )
    accuracy = _safe_ratio(correct, records)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": accuracy,
        "weighted_f1": weighted_f1,
        "precision": {
            label: _rounded(raw_precision[label]) for label in labels
        },
        "recall": {label: _rounded(raw_recall[label]) for label in labels},
        "f1": {label: _rounded(raw_f1[label]) for label in labels},
        "confusion": confusion,
    }


def _copy_block(block: MetricBlock) -> MetricBlock:
    """Return an independent copy so callers cannot alias the two blocks."""

    return {
        "accuracy": block["accuracy"],
        "macro_f1": block["macro_f1"],
        "micro_f1": block["micro_f1"],
        "weighted_f1": block["weighted_f1"],
        "precision": dict(block["precision"]),
        "recall": dict(block["recall"]),
        "f1": dict(block["f1"]),
        "confusion": {
            gold_label: dict(row) for gold_label, row in block["confusion"].items()
        },
    }


def score_verdicts(
    verdicts: Sequence[VerdictRecord],
    gold_labels: Sequence[str],
    *,
    marker: str,
    labels: Sequence[str],
    source: Literal["parsed", "script_table"] = "parsed",
) -> ScoreReport:
    """Align verdicts with gold labels and score them.

    ``gold_labels`` is the test file's ``label`` field in file order and
    defines ``records``. ``verdicts`` are aligned by their ``index`` rather
    than by position in the sequence, so a caller may pass them in any
    order. Records without a verdict are ``unfinished`` and are scored as
    ``__INVALID__``; ``complete`` is ``True`` only when none are missing.

    Raises ``ValueError`` before building a report when there are more
    verdicts than gold labels, when an index is out of range or duplicated,
    when a ``kind`` is unknown, or when an ``exact``/``recovered`` verdict
    carries a ``parsed`` value outside ``labels``. Those are programming
    errors; no record is silently dropped or reordered.
    """

    gold = list(gold_labels)
    label_list = list(labels)
    label_set = set(label_list)
    ordered = list(verdicts)
    records = len(gold)

    # Rule 2: more verdicts than gold labels is a caller error, and it is
    # detected before any report exists.
    if len(ordered) > records:
        raise ValueError(
            f"{len(ordered)} verdicts were supplied for {records} gold "
            "labels; a partial run may have fewer, never more"
        )

    for position, gold_label in enumerate(gold):
        if not isinstance(gold_label, str) or gold_label not in label_set:
            raise ValueError(
                f"gold label at index {position} is {gold_label!r}, which is "
                f"not one of the configured labels {label_list!r}"
            )

    aligned: list[VerdictRecord | None] = [None] * records
    seen: set[int] = set()
    counts: dict[str, int] = {kind: 0 for kind in KIND_ORDER}

    for position, verdict in enumerate(ordered):
        if not isinstance(verdict, dict):
            raise ValueError(
                f"verdict at position {position} is not a mapping: {verdict!r}"
            )

        index = verdict.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError(
                f"verdict at position {position} has a non-integer index "
                f"{index!r}"
            )
        if index < 0 or index >= records:
            raise ValueError(
                f"verdict index {index} is out of range for {records} gold "
                "labels"
            )
        if index in seen:
            raise ValueError(f"duplicate verdict index {index}")

        kind = verdict.get("kind")
        if kind not in VALID_KINDS:
            raise ValueError(
                f"verdict index {index} has unknown kind {kind!r}; expected "
                f"one of {sorted(VALID_KINDS)!r}"
            )

        parsed = verdict.get("parsed")
        if kind in CORRECT_KINDS and (
            not isinstance(parsed, str) or parsed not in label_set
        ):
            raise ValueError(
                f"verdict index {index} has kind {kind!r} but parsed value "
                f"{parsed!r} is not one of the configured labels "
                f"{label_list!r}"
            )

        seen.add(index)
        aligned[index] = verdict
        counts[kind] += 1

    # Rule 6: a prediction is usable only when the verdict is correct. All
    # four other kinds, plus a missing verdict, become __INVALID__.
    inclusive_predicted: list[str] = []
    exact_predicted: list[str] = []
    for index in range(records):
        verdict = aligned[index]
        if verdict is None:
            inclusive_predicted.append(INVALID_LABEL)
            exact_predicted.append(INVALID_LABEL)
            continue

        kind = verdict["kind"]
        parsed = verdict["parsed"]

        if kind in CORRECT_KINDS and parsed is not None:
            inclusive_predicted.append(parsed)
        else:
            inclusive_predicted.append(INVALID_LABEL)

        if kind == "exact" and parsed is not None:
            exact_predicted.append(parsed)
        else:
            exact_predicted.append(INVALID_LABEL)

    inclusive_metrics = _metric_block(gold, inclusive_predicted, label_list)
    exact_metrics = _metric_block(gold, exact_predicted, label_list)

    unfinished = records - len(ordered)
    invalid = (
        counts["empty"]
        + counts["out_of_set"]
        + counts["unparsed"]
        + unfinished
    )

    # The six counts partition the gold set by construction: the verdicts
    # are unique and in range, and every missing index is unfinished.
    assert sum(counts.values()) + unfinished == records

    report: ScoreReport = {
        "marker": marker,
        "labels": label_list,
        "source": source,
        "records": records,
        "complete": unfinished == 0,
        "exact": counts["exact"],
        "recovered": counts["recovered"],
        "empty": counts["empty"],
        "out_of_set": counts["out_of_set"],
        "unparsed": counts["unparsed"],
        "unfinished": unfinished,
        "empty_rate": _safe_ratio(counts["empty"], records),
        "invalid_rate": _safe_ratio(invalid, records),
        "accuracy": inclusive_metrics["accuracy"],
        "macro_f1": inclusive_metrics["macro_f1"],
        "micro_f1": inclusive_metrics["micro_f1"],
        "weighted_f1": inclusive_metrics["weighted_f1"],
        "precision": dict(inclusive_metrics["precision"]),
        "recall": dict(inclusive_metrics["recall"]),
        "f1": dict(inclusive_metrics["f1"]),
        "confusion": {
            gold_label: dict(row)
            for gold_label, row in inclusive_metrics["confusion"].items()
        },
        "exact_metrics": exact_metrics,
        "inclusive_metrics": inclusive_metrics,
    }
    return report


def render_report_markdown(report: ScoreReport) -> str:
    """Render a report as a compact Markdown block.

    The block has a one-line summary, a one-line kind-count breakdown, a
    per-class precision/recall/F1 table (with support) and the confusion
    matrix with one row per gold label and the ``__INVALID__`` column.
    Every line is free of trailing whitespace and the result ends with a
    newline.
    """

    complete = "yes" if report["complete"] else "no"
    lines = [
        (
            f"**Summary:** records={report['records']}, complete={complete}, "
            f"unfinished={report['unfinished']}, "
            f"accuracy={report['accuracy']:.4f}, "
            f"macro_f1={report['macro_f1']:.4f}, "
            f"empty_rate={report['empty_rate']:.4f}, "
            f"invalid_rate={report['invalid_rate']:.4f}"
        ),
        (
            f"**Counts:** exact={report['exact']}, "
            f"recovered={report['recovered']}, empty={report['empty']}, "
            f"out_of_set={report['out_of_set']}, "
            f"unparsed={report['unparsed']}, "
            f"unfinished={report['unfinished']}"
        ),
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "| --- | --- | --- | --- | --- |",
    ]

    for label in report["labels"]:
        support = sum(report["confusion"][label].values())
        lines.append(
            f"| {label} | {report['precision'][label]:.4f} | "
            f"{report['recall'][label]:.4f} | {report['f1'][label]:.4f} | "
            f"{support} |"
        )

    lines.append("")
    columns = [*report["labels"], INVALID_LABEL]
    lines.append("| Gold | " + " | ".join(columns) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in columns) + " |")
    for gold_label in report["labels"]:
        row = report["confusion"][gold_label]
        cells = " | ".join(str(row[column]) for column in columns)
        lines.append(f"| {gold_label} | {cells} |")

    return "\n".join(lines).rstrip() + "\n"


def load_gold_labels(path: str | Path) -> list[str]:
    """Return the ``label`` field of every record in a JSON test file.

    The format is the one documented for ``c.test_data_file``: a JSON list
    of objects with a ``string`` ``label`` field, aligned by index. The
    labels are returned in file order, which is the order the run produces.
    Raises ``ValueError`` for a file that is not a JSON list or a record
    without a string ``label``.
    """

    with open(path, encoding="utf-8") as handle:
        data: Any = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(
            f"gold data in {path!s} must be a JSON list, got "
            f"{type(data).__name__}"
        )

    labels: list[str] = []
    for position, record in enumerate(data):
        if not isinstance(record, dict) or "label" not in record:
            raise ValueError(
                f"gold record at index {position} has no 'label' field"
            )
        label = record["label"]
        if not isinstance(label, str):
            raise ValueError(
                f"gold label at index {position} is not a string: {label!r}"
            )
        labels.append(label)
    return labels
