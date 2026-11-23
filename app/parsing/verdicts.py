"""Verdict extraction and normalisation for inference logs (issue #34).

Extracts one 0-based ``VerdictRecord`` per *completed* output.  The result
shape is compatible with ``app.scoring.score_verdicts``; this module does not
import the scoring engine or a UI.

Layouts:
* banner: ``INPUTS`` / ``OUTPUT`` blocks, last response marker per block;
* think: decoded outputs followed by two script verdict tables;
* table_only: the second (final) script verdict table.

A script-table fallback from a placeholder to a supported label is marked
``recovered`` rather than treating the placeholder as an empty answer.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final, Literal, Sequence, TypedDict

__all__ = ["VerdictRecord", "VerdictKind", "normalize_verdict", "extract_verdicts"]

VerdictKind = Literal["exact", "recovered", "empty", "out_of_set", "unparsed"]
OutputFormat = Literal["banner", "think", "table_only"]


class VerdictRecord(TypedDict):
    index: int
    raw: str
    parsed: str | None
    kind: VerdictKind


DEFAULT_MARKER: Final = "### Response: Verdict: "
DEFAULT_LABELS: Final = ("True", "False", "Conflicting")
EOS_TOKENS: Final = ("</s>", "<｜end▁of▁sentence｜>")
# The distinction between out_of_set and unparsed is underspecified by the
# roadmap. Keep this set explicit: never guess a supported label from prose.
OUT_OF_SET: Final = frozenset({
    "mostly true", "mostly false", "half true", "half false",
    "not enough information", "partially true","partly true",
    "supports", "refutes", "incorrect",})


_EMPTY: Final = frozenset({"", "no response", "empty response", "<verdict>"})
_ROW: Final = re.compile(r"^(\d+)\s*\|\s*(.*)$")
_BANNER: Final = re.compile(r"^[*#]{5,}\s*(INPUTS|OUTPUT|VERDICTS|LENGTH)\s*[*#]{5,}$")
_PREFIX: Final = re.compile(r"^Verdict\s*:\s*", re.IGNORECASE)
_GENERATED_VERDICT: Final = re.compile(r"\bVerdict\s*:[ \t]*([^\r\n]*)", re.IGNORECASE)
_TIMESTAMP: Final = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+")
_RANK: Final = re.compile(r"^\[rank\d+\]: ?")
_RESPONSE: Final = "### Response:"


def normalize_verdict(
    raw: str, labels: Sequence[str] = DEFAULT_LABELS
) -> tuple[str | None, VerdictKind]:
    """Classify a *located* verdict, preserving exact-versus-cleaned status.

    ``exact`` describes the *format*, not correctness against the gold set.
    Never coerce an unsupported verdict into one of the configured labels.
    """
    if not isinstance(raw, str):
        raise TypeError("raw verdict must be a string")
    allowed = tuple(labels)
    if raw in allowed:
        return raw, "exact"

    # First generated line only (not an explanation after a newline).
    first = raw.splitlines()[0] if raw.splitlines() else ""
    first = first.strip()
    if first.casefold() in _EMPTY:
        return None, "empty"

    first = _PREFIX.sub("", first, count=1).strip()
    if first.casefold() in _EMPTY:
        return None, "empty"
    for token in EOS_TOKENS:
        if first.endswith(token):
            first = first[:-len(token)].rstrip()
    cleaned = first.rstrip("\t \r\n.!?:;,\"").strip()
    if cleaned.casefold() in _EMPTY:
        return None, "empty"

    for label in allowed:
        if cleaned.casefold() == label.casefold():
            return label, "recovered"
    if cleaned.casefold() in OUT_OF_SET:
        return None, "out_of_set"
    return None, "unparsed"


def _record(index: int, raw: str, labels: Sequence[str]) -> VerdictRecord:
    parsed, kind = normalize_verdict(raw, labels)
    return {"index": index, "raw": raw, "parsed": parsed, "kind": kind}


def _banner_text(block: list[str], marker: str) -> str:
    """Extract from a single OUTPUT block, excluding trailing bare lists.

    Some logs append 2,495 bare ``### Response`` rows after the *last*
    decoded output with no separating banner. Two consecutive response rows
    mark that list, not two additional records. Keep the first row and then
    apply rfind to that OUTPUT block (the script's rule).
    """
    response_marker = marker if marker else DEFAULT_MARKER
    cutoff = len(block)
    for i in range(len(block) - 1):
        if block[i].startswith(response_marker) and block[i + 1].startswith(response_marker):
            cutoff = i + 1
            break
    text = "\n".join(block[:cutoff])
    pos = text.rfind(response_marker)
    return "No response" if pos < 0 else text[pos + len(response_marker):].splitlines()[0].strip()


def _banner_verdicts(
    lines, marker: str, labels: Sequence[str]
) -> list[VerdictRecord]:
    result: list[VerdictRecord] = []
    output: list[str] = []
    inside = False
    saw_response = False

    def close(complete: bool) -> None:
        nonlocal output, inside, saw_response
        if inside and complete:
            result.append(_record(len(result), _banner_text(output, marker), labels))
        output = []
        inside = False
        saw_response = False

    for line in lines:
        line = _RANK.sub("", line.rstrip("\r\n"))
        section_match = _BANNER.match(line)
        if section_match:
            section = section_match.group(1)
            if section == "INPUTS":
                # The next INPUTS proves the previous OUTPUT was completed.
                close(True)
            elif section == "OUTPUT":
                # A preceding LENGTH banner is not a completed record.
                close(True)
                inside = True
            elif section == "VERDICTS":
                close(True)
            continue
        if inside:
            output.append(line)
            if marker and marker in line:
                saw_response = True
    # A truncated last OUTPUT without a response is NOT a complete record.
    close(saw_response)
    return result


def _think_answer(block: list[str]) -> str:
    """Extract the last generated ``Verdict: ...`` after the prompt marker.

    Some real outputs never close ``<think>`` (and a few print a second
    opening tag instead).  Requiring ``</think>`` loses genuine verdicts.
    Search only *after the last* ``### Response:`` so the prompt's example
    ``Verdict: <verdict>`` cannot be mistaken for an answer.

    If no verdict is printed, use the same placeholder as the script's
    first-pass table; its final table may still recover a prediction.
    """
    content = "\n".join(block)
    pos = content.rfind(_RESPONSE)
    if pos < 0:
        return "No response"
    generated = content[pos + len(_RESPONSE):]
    matches = list(_GENERATED_VERDICT.finditer(generated))
    if matches:
        return matches[-1].group(0).strip()
    return "Verdict: <verdict>"


def _timestamped_data(
    lines, *, include_decoded: bool
) -> tuple[list[dict[int, str]], list[str]]:
    """One streaming pass; only two small tables and per-record answers remain.

    Blocks can contain enormous echoed prompts, so at most one current block
    is stored, never every line of the log. A table index reset begins the
    second (final) script table. Only completed ``Done Decoding`` records
    contribute decoded answers, and rows do not count as new records.
    """
    tables: list[dict[int, str]] = []
    answers: list[str] = []
    previous_index: int | None = None
    block: list[str] = []
    started = False
    decoded = False

    def close() -> None:
        nonlocal block, started, decoded
        if include_decoded and started and decoded:
            answers.append(_think_answer(block))
        block = []
        started = False
        decoded = False

    for item in lines:
        line = _RANK.sub("", item.rstrip("\r\n"))
        row = _ROW.fullmatch(line.strip())
        if row:
            if started:
                close()
            index = int(row.group(1))
            if not tables or (previous_index is not None and index <= previous_index):
                tables.append({})
            tables[-1][index] = row.group(2).strip()
            previous_index = index
            continue

        if line.startswith("Tokenizing Inputs") and _TIMESTAMP.match(line.split(":", 1)[-1].strip()):
            close()
            started = True
        elif started and line.startswith("Done Decoding") and _TIMESTAMP.match(line.split(":", 1)[-1].strip()):
            decoded = True
        if include_decoded and started:
            block.append(line)
    close()
    return tables, answers


def _final_table_verdict(
    index: int,
    first_pass: str | None,
    final_value: str,
    labels: Sequence[str],
    *,
    original: str | None,
) -> VerdictRecord:
    """Use the script's final verdict, preserving any fallback recovery."""
    raw = original if original is not None else (first_pass or final_value)
    final_parsed, final_kind = normalize_verdict(final_value, labels)
    if final_parsed is not None:
        # The script recovered a placeholder or otherwise unusable first
        # pass. Preserve that fact for exact-versus-inclusive scoring.
        previous_parsed, previous_kind = normalize_verdict(raw, labels)
        if previous_parsed != final_parsed or previous_kind in ("empty", "unparsed", "out_of_set"):
            return {"index": index, "raw": raw, "parsed": final_parsed, "kind": "recovered"}
        # A table_only script reports its final, already normalised label.
        if original is None and first_pass is not None:
            old, old_kind = normalize_verdict(first_pass, labels)
            if old != final_parsed or old_kind in ("empty", "out_of_set", "unparsed"):
                return {"index": index, "raw": first_pass, "parsed": final_parsed, "kind": "recovered"}
            if old_kind == "recovered" and first_pass.strip() not in (f"Verdict: {final_parsed}", final_parsed):
                return {"index": index, "raw": first_pass, "parsed": final_parsed, "kind": "recovered"}
            return {"index": index, "raw": final_value, "parsed": final_parsed, "kind": final_kind}
        return {"index": index, "raw": raw, "parsed": final_parsed, "kind": previous_kind}
    # The final table is authoritative, including Empty Response.
    return {"index": index, "raw": raw, "parsed": None, "kind": final_kind}


def extract_verdicts(
    path: str | Path,
    *,
    output_format: OutputFormat | None = None,
    marker: str = DEFAULT_MARKER,
    labels: Sequence[str] = DEFAULT_LABELS,
) -> list[VerdictRecord]:
    """Extract scored-ready verdict records from one inference log.

    When ``output_format`` is omitted it is read from ``parse_log_file``.
    The caller supplies configured ``p.verdict_parse_marker`` and
    ``c.label_values`` to override defaults. Unfinished records are *not*
    fabricated here; ``score_verdicts`` counts absent indices as unfinished.

    For full logs the final script tables are authoritative; excerpt files
    with omitted spans may yield only the surviving indexed records.
    """
    if output_format is None:
        from app.parsing import parse_log_file
        output_format = parse_log_file(path)["output_format"]
    if output_format is None:
        return []
    if output_format not in ("banner", "think", "table_only"):
        raise ValueError(f"unsupported output format: {output_format!r}")

    with open(path, encoding="utf-8", errors="replace", newline=None) as f:
        if output_format == "banner":
            return _banner_verdicts(f, marker, labels)
        tables, decoded = _timestamped_data(
            f, include_decoded=output_format == "think"
        )

    first = tables[0] if tables else {}
    final = tables[-1] if len(tables) >= 2 else {}
    if output_format == "table_only":
        # Without both tables, no 'final' source can be asserted.
        return [
            _final_table_verdict(i, first.get(i), val, labels, original=None)
            for i, val in sorted(final.items())
        ]

    if not final:
        return [_record(i, raw, labels) for i, raw in enumerate(decoded)]

    # The full log has one decoded record per table entry. In a *trimmed*
    # fixture those records are noncontiguous: prefer the table's first pass
    # for the missing positions rather than assigning wrong indices.
    full_alignment = len(decoded) == len(final) and all(i in final for i in range(len(decoded)))
    return [
        _final_table_verdict(
            i, first.get(i), val, labels,
            original=decoded[i] if full_alignment else None,
        )
        for i, val in sorted(final.items())
    ]
