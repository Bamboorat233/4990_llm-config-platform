"""Tests for issue #35: gold-label alignment and scoring.

All inputs are synthesised: the repository has no gold test-data file (it
lives outside the repository, in ``data/``), so every test builds its own
gold list and verdict records. The oracle is ``specs/interfaces.md``
section 5 and design section 3.6.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from app.scoring import (
    INVALID_LABEL,
    METRIC_KEYS,
    ScoreReport,
    VerdictRecord,
    load_gold_labels,
    render_report_markdown,
    score_verdicts,
)

LABELS = ["True", "False", "Conflicting"]
MARKER = "### Response: Verdict: "
KINDS = ("exact", "recovered", "empty", "out_of_set", "unparsed")


def verdict(
    index: int,
    kind: str,
    parsed: str | None = None,
    raw: str | None = None,
) -> VerdictRecord:
    """Build one verdict record; ``raw`` defaults to the parsed text."""

    return {
        "index": index,
        "raw": raw if raw is not None else (parsed or ""),
        "parsed": parsed,
        "kind": kind,  # type: ignore[typeddict-item]
    }


def floats_in(value: Any) -> list[float]:
    """Yield every float nested in a report, for NaN/Infinity checks."""

    if isinstance(value, dict):
        found: list[float] = []
        for item in value.values():
            found.extend(floats_in(item))
        return found
    if isinstance(value, (list, tuple)):
        found = []
        for item in value:
            found.extend(floats_in(item))
        return found
    if isinstance(value, float):
        return [value]
    return []


def assert_finite(report: ScoreReport) -> None:
    """No metric may be NaN or infinite, at any nesting depth."""

    for value in floats_in(report):
        assert not math.isnan(value)
        assert math.isfinite(value)


def assert_partition(report: ScoreReport) -> None:
    """The six kind counts partition the gold set exactly."""

    total = sum(report[kind] for kind in KINDS) + report["unfinished"]
    assert total == report["records"]
    assert report["complete"] is (report["unfinished"] == 0)


def support_for(report: ScoreReport, label: str) -> int:
    """The gold support of a class is its confusion row sum."""

    return sum(report["confusion"][label].values())


# ---------------------------------------------------------------------------
# 1. Log 15: a partial run keeps the gold denominator
# ---------------------------------------------------------------------------


def test_log15_partial_run_uses_gold_denominator():
    """2,495 gold records, 2,480 verdicts, 15 unfinished."""

    gold = [LABELS[position % 3] for position in range(2495)]
    verdicts = [verdict(position, "exact", gold[position]) for position in range(2480)]

    report = score_verdicts(verdicts, gold, marker=MARKER, labels=LABELS)

    assert report["records"] == 2495
    assert report["unfinished"] == 15
    assert report["complete"] is False
    assert report["exact"] == 2480
    assert report["recovered"] == 0
    # The denominator is the gold size, not the 2,480 records that finished.
    assert report["accuracy"] == round(2480 / 2495, 4)
    assert report["accuracy"] != 1.0
    assert report["micro_f1"] == report["accuracy"]
    # All 15 missing records are invalid predictions.
    assert report["invalid_rate"] == round(15 / 2495, 4)
    assert report["empty_rate"] == 0.0
    assert_partition(report)
    assert_finite(report)


# ---------------------------------------------------------------------------
# 2. More verdicts than gold labels
# ---------------------------------------------------------------------------


def test_more_verdicts_than_gold_raises_value_error():
    """2,496 verdicts for 2,495 gold labels names both counts."""

    gold = [LABELS[position % 3] for position in range(2495)]
    verdicts = [
        verdict(position, "exact", LABELS[position % 3])
        for position in range(2496)
    ]

    with pytest.raises(ValueError) as excinfo:
        score_verdicts(verdicts, gold, marker=MARKER, labels=LABELS)

    message = str(excinfo.value)
    assert "2496" in message
    assert "2495" in message


def test_one_extra_verdict_for_small_gold_raises():
    """The length check is not special-cased for log 15."""

    gold = ["True", "False"]
    verdicts = [
        verdict(0, "exact", "True"),
        verdict(1, "exact", "False"),
        verdict(2, "exact", "True"),
    ]

    with pytest.raises(ValueError):
        score_verdicts(verdicts, gold, marker=MARKER, labels=LABELS)


# ---------------------------------------------------------------------------
# 3. Zero-support classes
# ---------------------------------------------------------------------------


def test_zero_support_class_yields_zero_not_nan():
    """A class with no gold records scores 0.0 without an exception."""

    gold = ["True", "True", "False"]
    verdicts = [
        verdict(0, "exact", "True"),
        verdict(1, "exact", "True"),
        verdict(2, "exact", "False"),
    ]

    report = score_verdicts(verdicts, gold, marker=MARKER, labels=LABELS)

    assert "Conflicting" not in gold
    assert report["precision"]["Conflicting"] == 0.0
    assert report["recall"]["Conflicting"] == 0.0
    assert report["f1"]["Conflicting"] == 0.0
    assert report["exact_metrics"]["precision"]["Conflicting"] == 0.0
    assert report["confusion"]["Conflicting"] == {
        "True": 0,
        "False": 0,
        "Conflicting": 0,
        INVALID_LABEL: 0,
    }
    # Macro-F1 divides by three classes, one of which scores zero.
    assert report["macro_f1"] == round((1.0 + 1.0 + 0.0) / 3, 4)
    assert_finite(report)


def test_zero_support_class_with_predictions_still_zero():
    """A class can be predicted while having no gold support."""

    gold = ["True", "False"]
    verdicts = [
        verdict(0, "exact", "True"),
        verdict(1, "exact", "False"),
    ]

    report = score_verdicts(verdicts, gold, marker=MARKER, labels=LABELS)

    assert report["precision"]["Conflicting"] == 0.0
    assert report["recall"]["Conflicting"] == 0.0
    assert report["f1"]["Conflicting"] == 0.0
    assert_finite(report)


# ---------------------------------------------------------------------------
# 4. Partition invariant across mixed scenarios
# ---------------------------------------------------------------------------


MIXED_GOLD = ["True", "False", "Conflicting", "True", "False", "Conflicting"]
MIXED_VERDICTS = [
    verdict(0, "exact", "True"),
    verdict(1, "recovered", "False", raw="False</s>"),
    verdict(2, "empty"),
    verdict(3, "out_of_set", "Mostly True", raw="Mostly True"),
    verdict(4, "unparsed", None, raw="???"),
    # index 5 is missing -> unfinished
]


@pytest.mark.parametrize(
    "gold, verdicts, expected_counts",
    [
        (
            ["True", "False", "Conflicting"],
            [
                verdict(0, "exact", "True"),
                verdict(1, "exact", "False"),
                verdict(2, "exact", "Conflicting"),
            ],
            (3, 0, 0, 0, 0, 0),
        ),
        (
            MIXED_GOLD,
            MIXED_VERDICTS,
            (1, 1, 1, 1, 1, 1),
        ),
        (
            ["True", "False", "Conflicting", "True"],
            [
                verdict(0, "exact", "True"),
                verdict(1, "empty"),
                verdict(2, "out_of_set", "Half true"),
                verdict(3, "unparsed", None),
            ],
            (1, 0, 1, 1, 1, 0),
        ),
    ],
)
def test_partition_invariant(gold, verdicts, expected_counts):
    """The six kinds plus unfinished always sum to records."""

    report = score_verdicts(verdicts, gold, marker=MARKER, labels=LABELS)
    actual = tuple(
        report[kind] for kind in (*KINDS, "unfinished")
    )

    assert actual == expected_counts
    assert_partition(report)
    assert_finite(report)


def test_partition_with_verdicts_in_reverse_order():
    """Alignment uses the index, not the sequence position."""

    gold = ["True", "False", "Conflicting"]
    verdicts = [
        verdict(2, "exact", "Conflicting"),
        verdict(0, "exact", "True"),
        verdict(1, "recovered", "False", raw="False"),
    ]

    report = score_verdicts(verdicts, gold, marker=MARKER, labels=LABELS)

    assert report["exact"] == 2
    assert report["recovered"] == 1
    assert report["accuracy"] == 1.0
    assert report["confusion"]["True"]["True"] == 1
    assert report["confusion"]["Conflicting"]["Conflicting"] == 1
    assert_partition(report)


# ---------------------------------------------------------------------------
# 5. Exact vs inclusive metrics
# ---------------------------------------------------------------------------


def test_recovered_is_invalid_in_exact_and_correct_in_inclusive():
    """A recovered verdict is bucketed per metric block."""

    gold = ["True", "False", "Conflicting"]
    verdicts = [
        verdict(0, "exact", "True"),
        verdict(1, "recovered", "False", raw="False</s>"),
        verdict(2, "empty"),
    ]

    report = score_verdicts(verdicts, gold, marker=MARKER, labels=LABELS)

    assert report["inclusive_metrics"]["accuracy"] == round(2 / 3, 4)
    assert report["exact_metrics"]["accuracy"] == round(1 / 3, 4)
    assert report["exact_metrics"]["macro_f1"] < report["inclusive_metrics"]["macro_f1"]

    # Inclusive confusion: the recovered record lands in the False column.
    assert report["inclusive_metrics"]["confusion"]["False"]["False"] == 1
    assert report["inclusive_metrics"]["confusion"]["False"][INVALID_LABEL] == 0
    # Exact confusion: the same record is an invalid prediction.
    assert report["exact_metrics"]["confusion"]["False"]["False"] == 0
    assert report["exact_metrics"]["confusion"]["False"][INVALID_LABEL] == 1

    # Top-level metric fields mirror the inclusive block exactly.
    for key in METRIC_KEYS:
        assert report[key] == report["inclusive_metrics"][key]


def test_exact_kind_that_disagrees_with_gold_is_misclassified():
    """``exact`` describes normalisation, not correctness.

    A well-formed verdict can still name the wrong class; it must count as
    a misclassification in the confusion matrix and in every metric.
    """

    gold = ["True", "False"]
    verdicts = [
        verdict(0, "exact", "False"),
        verdict(1, "exact", "False"),
    ]

    report = score_verdicts(verdicts, gold, marker=MARKER, labels=LABELS)

    assert report["exact"] == 2
    assert report["accuracy"] == 0.5
    assert report["confusion"]["True"]["False"] == 1
    assert report["confusion"]["True"]["True"] == 0
    assert report["precision"]["False"] == 0.5
    assert report["recall"]["False"] == 1.0
    assert report["recall"]["True"] == 0.0
    assert report["f1"]["False"] == round(2 / 3, 4)
    assert report["weighted_f1"] == round(1 / 3, 4)
    assert_finite(report)


def test_top_level_metrics_are_independent_copies():
    """Mutating top-level metrics cannot corrupt inclusive_metrics."""

    gold = ["True"]
    report = score_verdicts(
        [verdict(0, "exact", "True")], gold, marker=MARKER, labels=LABELS
    )

    report["precision"]["True"] = 0.123
    assert report["inclusive_metrics"]["precision"]["True"] == 1.0


# ---------------------------------------------------------------------------
# 6. Confusion matrix
# ---------------------------------------------------------------------------


def test_confusion_rows_columns_and_invalid_column():
    """Every row sums to its support; the invalid column collects all kinds."""

    report = score_verdicts(
        MIXED_VERDICTS, MIXED_GOLD, marker=MARKER, labels=LABELS
    )

    assert list(report["confusion"]) == LABELS
    assert list(report["confusion"]["True"]) == [*LABELS, INVALID_LABEL]

    # Two records per class in this fixture.
    for label in LABELS:
        assert support_for(report, label) == 2

    assert report["confusion"]["True"] == {
        "True": 1,
        "False": 0,
        "Conflicting": 0,
        INVALID_LABEL: 1,
    }
    assert report["confusion"]["False"] == {
        "True": 0,
        "False": 1,
        "Conflicting": 0,
        INVALID_LABEL: 1,
    }
    assert report["confusion"]["Conflicting"] == {
        "True": 0,
        "False": 0,
        "Conflicting": 0,
        INVALID_LABEL: 2,
    }

    invalid_column = sum(
        report["confusion"][label][INVALID_LABEL] for label in LABELS
    )
    assert invalid_column == (
        report["empty"]
        + report["out_of_set"]
        + report["unparsed"]
        + report["unfinished"]
    )
    assert invalid_column == 4
    assert_partition(report)
    assert_finite(report)


# ---------------------------------------------------------------------------
# 7. Invalid programmatic input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "verdicts, offending",
    [
        # Duplicate index.
        (
            [
                verdict(0, "exact", "True"),
                verdict(0, "exact", "True"),
            ],
            "0",
        ),
        # Index equal to the gold size.
        ([verdict(2, "exact", "True")], "2"),
        # Negative index.
        ([verdict(-1, "exact", "True")], "-1"),
        # Unknown kind.
        ([verdict(0, "maybe", "True")], "0"),
        # exact with a parsed value outside labels.
        ([verdict(0, "exact", "Maybe")], "0"),
        # recovered without a usable parsed value.
        ([verdict(0, "recovered")], "0"),
        # recovered with a parsed value outside labels.
        ([verdict(0, "recovered", "Half true")], "0"),
    ],
)
def test_invalid_programmatic_input_raises_value_error(verdicts, offending):
    """Bad alignment or kind data is a ValueError naming the index."""

    gold = ["True", "False"]

    with pytest.raises(ValueError) as excinfo:
        score_verdicts(verdicts, gold, marker=MARKER, labels=LABELS)

    assert offending in str(excinfo.value)


def test_gold_label_outside_configured_labels_raises():
    """A gold label that is not in the label domain cannot be scored."""

    with pytest.raises(ValueError) as excinfo:
        score_verdicts(
            [verdict(0, "exact", "True")],
            ["True", "Maybe"],
            marker=MARKER,
            labels=LABELS,
        )

    assert "Maybe" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 8. Rates
# ---------------------------------------------------------------------------


def test_rates_match_hand_computed_values():
    """One record of each kind plus one unfinished, over six records."""

    gold = ["True", "False", "Conflicting", "True", "False", "Conflicting"]
    verdicts = [
        verdict(0, "exact", "True"),
        verdict(1, "recovered", "False"),
        verdict(2, "empty"),
        verdict(3, "out_of_set", "Mostly True"),
        verdict(4, "unparsed", None, raw="???"),
    ]

    report = score_verdicts(verdicts, gold, marker=MARKER, labels=LABELS)

    assert report["empty"] == 1
    assert report["empty_rate"] == round(1 / 6, 4)
    # empty + out_of_set + unparsed + unfinished = 1 + 1 + 1 + 1
    assert report["invalid_rate"] == round(4 / 6, 4)
    assert report["accuracy"] == round(2 / 6, 4)
    assert report["unfinished"] == 1
    assert report["complete"] is False


# ---------------------------------------------------------------------------
# 9. Markdown rendering
# ---------------------------------------------------------------------------


def test_render_report_markdown_contents():
    """The block carries counts, both tables and the invalid column."""

    gold = ["True", "False", "Conflicting", "True", "False", "Conflicting"]
    report = score_verdicts(
        MIXED_VERDICTS, gold, marker=MARKER, labels=LABELS
    )

    rendered = render_report_markdown(report)

    assert "records=6" in rendered
    assert "complete=no" in rendered
    assert "unfinished=1" in rendered
    assert "exact=1" in rendered
    assert "recovered=1" in rendered
    assert "empty=1" in rendered
    assert "out_of_set=1" in rendered
    assert "unparsed=1" in rendered
    assert "accuracy=" in rendered
    assert "macro_f1=" in rendered
    assert "empty_rate=" in rendered
    assert "invalid_rate=" in rendered

    assert "| Class | Precision | Recall | F1 | Support |" in rendered
    assert "| True |" in rendered
    assert "| False |" in rendered
    assert "| Conflicting |" in rendered

    assert "| Gold | True | False | Conflicting | __INVALID__ |" in rendered
    assert "__INVALID__" in rendered

    assert rendered.endswith("\n")
    for line in rendered.splitlines():
        assert line == line.rstrip()


def test_render_report_markdown_marks_complete_run():
    """A complete run says so in the summary."""

    report = score_verdicts(
        [verdict(0, "exact", "True")],
        ["True"],
        marker=MARKER,
        labels=LABELS,
    )

    rendered = render_report_markdown(report)

    assert "complete=yes" in rendered
    assert "unfinished=0" in rendered
    assert rendered.endswith("\n")


def test_render_report_markdown_empty_run():
    """An empty report renders tables with no rows and no division error."""

    report = score_verdicts([], [], marker=MARKER, labels=LABELS)

    rendered = render_report_markdown(report)

    assert "records=0" in rendered
    assert "| Class | Precision | Recall | F1 | Support |" in rendered
    assert "| Gold | True | False | Conflicting | __INVALID__ |" in rendered
    assert rendered.endswith("\n")


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------


def test_empty_gold_and_empty_verdicts():
    """A zero-record report is valid and every metric is 0.0."""

    report = score_verdicts([], [], marker=MARKER, labels=LABELS)

    assert report["records"] == 0
    assert report["complete"] is True
    assert report["unfinished"] == 0
    assert report["empty_rate"] == 0.0
    assert report["invalid_rate"] == 0.0
    assert report["accuracy"] == 0.0
    assert report["macro_f1"] == 0.0
    assert report["micro_f1"] == 0.0
    assert report["weighted_f1"] == 0.0
    assert report["precision"] == {"True": 0.0, "False": 0.0, "Conflicting": 0.0}
    assert report["recall"] == {"True": 0.0, "False": 0.0, "Conflicting": 0.0}
    assert report["f1"] == {"True": 0.0, "False": 0.0, "Conflicting": 0.0}
    for label in LABELS:
        assert report["confusion"][label] == {
            "True": 0,
            "False": 0,
            "Conflicting": 0,
            INVALID_LABEL: 0,
        }
    assert report["exact_metrics"] == report["inclusive_metrics"]
    assert_partition(report)
    assert_finite(report)


def test_single_record_exact():
    """A one-record run scores 1.0."""

    report = score_verdicts(
        [verdict(0, "exact", "False")],
        ["False"],
        marker=MARKER,
        labels=LABELS,
    )

    assert report["records"] == 1
    assert report["complete"] is True
    assert report["accuracy"] == 1.0
    assert report["macro_f1"] == round(1 / 3, 4)
    assert report["weighted_f1"] == 1.0
    assert report["precision"]["False"] == 1.0
    assert report["recall"]["False"] == 1.0
    assert report["f1"]["False"] == 1.0
    assert_finite(report)


def test_single_record_unfinished():
    """One gold record with no verdict is unfinished and invalid."""

    report = score_verdicts([], ["True"], marker=MARKER, labels=LABELS)

    assert report["records"] == 1
    assert report["unfinished"] == 1
    assert report["complete"] is False
    assert report["accuracy"] == 0.0
    assert report["invalid_rate"] == 1.0
    assert report["confusion"]["True"][INVALID_LABEL] == 1
    assert_partition(report)


def test_marker_and_source_are_carried_through():
    """The configuration fields are stored unchanged."""

    report = score_verdicts(
        [verdict(0, "exact", "True")],
        ["True"],
        marker=MARKER,
        labels=LABELS,
        source="script_table",
    )

    assert report["marker"] == MARKER
    assert report["labels"] == LABELS
    assert report["source"] == "script_table"


def test_labels_order_defines_key_order():
    """Per-class dictionaries and confusion rows follow ``labels``."""

    reversed_labels = ["Conflicting", "False", "True"]
    report = score_verdicts(
        MIXED_VERDICTS, MIXED_GOLD, marker=MARKER, labels=reversed_labels
    )

    assert list(report["precision"]) == reversed_labels
    assert list(report["recall"]) == reversed_labels
    assert list(report["f1"]) == reversed_labels
    assert list(report["confusion"]) == reversed_labels
    assert list(report["confusion"]["Conflicting"]) == [*reversed_labels, INVALID_LABEL]


# ---------------------------------------------------------------------------
# Optional gold loader
# ---------------------------------------------------------------------------


def test_load_gold_labels_preserves_file_order(tmp_path):
    """The loader returns the ``label`` field in record order."""

    path = tmp_path / "testData.json"
    payload = [
        {"claim": "a", "evidences": [], "label": "True"},
        {
            "claim": "b",
            "evidences": [{"questions": "q", "top_k_doc": []}],
            "label": "False",
        },
        {"claim": "c", "evidences": [], "label": "Conflicting"},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_gold_labels(path) == ["True", "False", "Conflicting"]


def test_load_gold_labels_rejects_non_list(tmp_path):
    """A JSON object is not the documented test-file format."""

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"label": "True"}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_gold_labels(path)


def test_load_gold_labels_rejects_missing_or_non_string_label(tmp_path):
    """Each record must carry a string ``label`` field."""

    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps([{"claim": "a"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_gold_labels(missing)

    non_string = tmp_path / "non_string.json"
    non_string.write_text(json.dumps([{"label": 3}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_gold_labels(non_string)


# ---------------------------------------------------------------------------
# ScoreReport key contract
# ---------------------------------------------------------------------------


def test_score_report_has_every_contract_key():
    """The report has exactly the keys and count of section 5."""

    report = score_verdicts(
        [verdict(0, "exact", "True")],
        ["True"],
        marker=MARKER,
        labels=LABELS,
    )

    expected = {
        "marker",
        "labels",
        "source",
        "records",
        "complete",
        "exact",
        "recovered",
        "empty",
        "out_of_set",
        "unparsed",
        "unfinished",
        "empty_rate",
        "invalid_rate",
        "accuracy",
        "macro_f1",
        "micro_f1",
        "weighted_f1",
        "precision",
        "recall",
        "f1",
        "confusion",
        "exact_metrics",
        "inclusive_metrics",
    }
    assert set(report) == expected
    assert set(report) == set(ScoreReport.__annotations__)
    assert set(report["exact_metrics"]) == set(METRIC_KEYS)
    assert set(report["inclusive_metrics"]) == set(METRIC_KEYS)
