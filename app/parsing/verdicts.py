"""Extract and normalise the verdict printed by a fact-check inference run.

One log yields one :class:`app.scoring.VerdictRecord` per gold index:
``{"index", "raw", "parsed", "kind"}``. ``index`` is the 0-based position in
the gold set, ``raw`` is the decoded text (first line, capped at 500
characters) so the UI can show it, ``parsed`` is the normalised value (or
``None`` for ``empty``) and ``kind`` is one of the five kinds below. The
module never emits ``unfinished``: scoring adds that for missing indices.

Layouts
-------
``banner`` (logs 02, 03, 05 and 15, produced by ``inferenceMistral.py``)
    Each record is an ``OUTPUT`` block that echoes the prompt and then the
    generation. The prompt echo means the marker can occur more than once, so
    the value is the text after the *last* marker in the block (``rfind``, as
    ``inferenceMistral.py`` does). A block closed by a later banner that
    holds no marker is the literal ``No response`` and is ``empty``. The
    block still open at end of file is dropped unless it holds a
    ``### Response:`` line, so a record the run never finished stays missing.
    Some runs append a bare list of every response after the last ``OUTPUT``
    block; a pair of consecutive marker lines starts that list, so only the
    first line of the pair belongs to the record.
``think`` (log 04)
    The decoded record contains a prompt echo and then the generation after
    ``### Response:``. The verdict is the text after the last ``Verdict: `` in
    the generation, so the prompt's ``Verdict: <verdict>`` cannot be mistaken
    for an answer. Two ``N | ...`` tables follow the records: the first is the
    script's first pass and the second is its final output. The decoded record
    is the primary source; the final table resolves the indices where the
    decoded record carries no verdict (the script's fallback). The table is
    the *final* table only when two tables were printed; with a single table
    the decoded records are the source.
``table_only`` (log 07)
    There is no prompt echo and no decoded record in the log. The second
    (final) script table is the source, hence ``source="script_table"``. A
    lone table is not treated as final, so a log that printed only one table
    yields no records: nothing can be asserted as the final output.

The script tables are **0-based**: log 04 has rows ``0`` to ``2494`` (2,495
rows) for a gold set of 2,495 records, and log 07 has exactly the same range.
A 1-based table would have ended at ``2495``; it does not. The index of a
record is therefore ``row - 0``, with no offset.

Five kinds (design section 3.5)
-------------------------------
``empty``
    No usable text (``Empty Response``, ``No response``, the ``<verdict>``
    instruction placeholder or nothing at all). ``parsed`` is ``None``.
``exact``
    The located text is one of ``labels`` before Step 1 trims it.
``recovered``
    The located text matches a label only after trimming (``False</s>``,
    ``Conflicting.``) or after the script's fallback. ``parsed`` is the label.
``out_of_set``
    A plausible verdict outside ``labels`` (``Mostly True``, ``Half true``,
    ``Not enough information``, ``Supports``, ``Refutes``). ``parsed`` keeps
    the trimmed text for review.
``unparsed``
    Anything else; never coerced into a label. ``parsed`` is ``None`` and
    ``raw`` keeps the text, because a filled Parsed cell would look like a
    successful parse.

Step 0, locate
--------------
``banner`` uses the caller's ``marker`` (``p.verdict_parse_marker``, whose
default is ``### Response: Verdict: ``). The rule was verified against
``inferenceMistral.py``. ``think`` and ``table_only`` follow the documented
``Verdict: <verdict>`` rule of design section 3.5: the generated text that
starts with the literal instruction placeholder ``Verdict: <verdict>`` is
empty; otherwise the value is the text after the last ``Verdict: `` in the
generation. That script (``inferenceFactCheckerThanh.py``) is **not in this
repository**, so the rule is implemented exactly as documented in the design
note and nothing that the logs contradict is invented.

Step 1, trim
------------
First line only; strip trailing whitespace, the model EOS tokens (``</s>``,
the DeepSeek ``<|end of sentence|>`` token), trailing punctuation and a
leading ``Verdict: ``.

Completeness
------------
Whether a record is present is decided from evidence inside the stream, not
from the run's termination status (design section 3.4). The rules are the
same whether or not the caller supplies ``layout``.

``banner``
    A block closed by a later banner (``INPUTS``, ``OUTPUT``, ``VERDICTS``)
    is **always** a record, even when it holds no marker; the marker-less
    block is the literal ``No response`` and ``empty``. The block still open
    at end of file is a record only when it contains a ``### Response:``
    line. A marker test on every block would lose the marker-less records the
    script completed (log 02 and log 15 each have such records).

``think``
    A decoded block is a record only when ``Done Decoding`` was seen for it
    and the block contains a ``### Response:`` line. In the real logs the
    phase line precedes ``Decoded output:``, so the flag is carried into the
    block. Requiring only ``Done Decoding`` would invent an ``empty`` record
    in the short window between that phase line and the decoded text. A block
    still open at end of file is tested the same way, because the kill may
    have cut it short.

A record the run never finished must stay *missing*, because scoring counts a
missing index as ``unfinished`` whereas an ``empty`` record is an answer the
model never gave. The absence of a response line in the final open banner
block, or of ``Done Decoding`` plus a response in the final open think block,
is that evidence: the record is dropped.

Scoring
-------
:func:`extract_verdicts_with_source` returns the ``source`` that
:func:`app.scoring.score_verdicts` requires (``"parsed"`` for ``banner`` and
``think``, ``"script_table"`` for ``table_only``) together with the
records; callers must pass it to ``score_verdicts(source=...)`` or a
``table_only`` extraction would be scored as parsed output. The
records-only :func:`extract_verdicts` delegates to it, so the two can never
disagree.

Memory
------
:func:`extract_verdicts_with_source` makes one forward-only pass over the
file (plus the ``detect()`` pass only when ``layout`` is ``None``), opened
with ``encoding="utf-8", errors="replace", newline=None``. Only the current
``OUTPUT``/decoded block, the two table dictionaries and one short string
per record are retained, never the file's lines: a 200k-line log uses the
same memory as a 200-line one plus the records that are returned anyway.

Nothing here imports NiceGUI: parsing runs in a worker thread and from the
command line.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final, Literal, Sequence

from app.parsing.detect import detect
from app.parsing.errors import TOKENIZING_INPUTS
from app.scoring import VerdictKind, VerdictRecord

__all__ = [
    "OutputLayout",
    "VerdictSource",
    "extract_verdicts",
    "extract_verdicts_with_source",
    "verdict_source",
]

OutputLayout = Literal["banner", "think", "table_only"]
VerdictSource = Literal["parsed", "script_table"]

DEFAULT_LABELS: Final[tuple[str, ...]] = ("True", "False", "Conflicting")
OUTPUT_FORMATS: Final[tuple[str, ...]] = ("banner", "think", "table_only")

#: ``source`` per layout. The table of design section 3.5 and the contract in
#: ``specs/interfaces.md`` section 5 rule 1.
SOURCES: Final[dict[str, VerdictSource]] = {
    "banner": "parsed",
    "think": "parsed",
    "table_only": "script_table",
}

#: Texts that carry no verdict. ``<verdict>`` is the literal placeholder of
#: the instruction template (``Give the response in the format - Verdict:
#: <verdict>``); a model that echoes it has not answered.
EMPTY_TEXTS: Final[frozenset[str]] = frozenset(
    {"", "empty response", "no response", "<verdict>"}
)

#: Model end-of-sequence tokens removed by Step 1. The second token is the
#: DeepSeek sentence token; it is spelled with escapes so this source file
#: stays ASCII while the CJK policy allows the fullwidth bar.
EOS_TOKENS: Final[tuple[str, ...]] = (
    "</s>",
    "<\uff5cend\u2581of\u2581sentence\uff5c>",
)

#: Trailing punctuation removed by Step 1 (``Conflicting.``, ``True,``).
TRAILING_PUNCTUATION: Final = ".,;:!?\"'"

#: Verdict phrasings that are plausible but outside the label set. The list
#: is bounded on purpose: an unrecognised sentence stays ``unparsed`` rather
#: than being guessed into a verdict.
OUT_OF_SET_PHRASES: Final[frozenset[str]] = frozenset(
    {
        "mostly true",
        "mostly false",
        "half true",
        "half false",
        "partly true",
        "partly false",
        "partially true",
        "partially false",
        "not enough information",
        "not enough info",
        "insufficient information",
        "supports",
        "supported",
        "refutes",
        "refuted",
        "incorrect",
        "mixed",
        "true and false",
    }
)

#: A phrase with one of these prefixes is out-of-set even when it continues
#: (``Supports the claim``, ``Not enough evidence``).
OUT_OF_SET_PREFIXES: Final[tuple[str, ...]] = (
    "mostly ",
    "half ",
    "partly ",
    "partially ",
    "not enough ",
    "supports ",
    "supported ",
    "refutes ",
    "refuted ",
)

_RAW_LIMIT: Final = 500

_BANNER: Final = re.compile(
    r"^[*#]{5,}\s*(INPUTS|OUTPUT|VERDICTS|LENGTH)\s*[*#]{5,}$"
)
_ROW: Final = re.compile(r"^(\d+) \| (.*)$")
_RANK_PREFIX: Final = re.compile(r"^\[rank\d+\]: ?")
_LEADING_VERDICT: Final = re.compile(r"^Verdict:\s*", re.IGNORECASE)

_DECODED_HEADER: Final = "Decoded output:"
_DONE_DECODING: Final = "Done Decoding"
_RESPONSE: Final = "### Response:"
_VERDICT_PREFIX: Final = "Verdict: "
_NO_RESPONSE: Final = "No response"

#: The rule line the script prints between two records, just before the next
#: ``Tokenizing Inputs``. No pattern for it exists in
#: :mod:`app.parsing.errors`, so it is compiled here.
_SEPARATOR: Final = re.compile(r"^\*{20,}$")


def _starts_next_record(line: str) -> bool:
    """True for a line that proves the record before it was completed.

    In the ``think`` layout the next record begins with the ``Tokenizing
    Inputs`` phase line of :mod:`app.parsing.errors`, printed after the
    ``***`` separator. Either marker closes the decoded block; the block is a
    record only if ``Done Decoding`` was seen and it holds a response line.
    """

    if line.startswith(TOKENIZING_INPUTS):
        return True
    return _SEPARATOR.match(line) is not None


# ---------------------------------------------------------------------------
# Step 1 and Step 2
# ---------------------------------------------------------------------------


def _first_line(text: str) -> str:
    """Return the first line of ``text`` or an empty string."""

    lines = text.splitlines()
    return lines[0] if lines else ""


def _cap(text: str, limit: int = _RAW_LIMIT) -> str:
    """Cap a raw line so one pathological record cannot dominate memory."""

    return text if len(text) <= limit else text[:limit]


def _trim(value: str) -> str:
    """Apply Step 1: first line, EOS tokens, punctuation, leading marker."""

    line = _first_line(value).strip()
    line = _LEADING_VERDICT.sub("", line, count=1).strip()
    changed = True
    while changed:
        changed = False
        for token in EOS_TOKENS:
            if line.endswith(token):
                line = line[: -len(token)].rstrip()
                changed = True
        stripped = line.rstrip(TRAILING_PUNCTUATION + " \t")
        if stripped != line:
            line = stripped
            changed = True
    return line


def _is_out_of_set(value: str) -> bool:
    """True for a plausible verdict phrasing outside the label set."""

    folded = " ".join(value.casefold().split())
    if folded in OUT_OF_SET_PHRASES:
        return True
    return any(folded.startswith(prefix) for prefix in OUT_OF_SET_PREFIXES)


def _classify(value: str, labels: Sequence[str]) -> tuple[str | None, VerdictKind]:
    """Classify a *located* value into ``(parsed, kind)``.

    ``value`` is the text after the layout's marker (or a table cell). Step 1
    is :func:`_trim`; Step 2 applies the five kinds in the documented order.
    ``parsed`` is the label for ``exact``/``recovered``, ``None`` for
    ``empty`` and ``unparsed``, and the trimmed text for ``out_of_set``
    (out-of-set values are counted by value for review).
    """

    line = _first_line(value).strip()

    # Explicit rule of issue #34: a generated text that starts with the
    # literal instruction placeholder is the prompt echoed back, not a
    # verdict, and is empty. This is deliberately narrower than "starts with
    # Verdict: ": in the think layout a real answer is "Verdict: False".
    if line.casefold().startswith("verdict: <verdict>"):
        return None, "empty"
    if line.casefold() in EMPTY_TEXTS:
        return None, "empty"

    # Rule 2: exact is decided before Step 1 removes EOS tokens or
    # punctuation. Surrounding whitespace is line handling, not trimming.
    if line in labels:
        return line, "exact"

    trimmed = _trim(line)
    if trimmed.casefold() in EMPTY_TEXTS:
        return None, "empty"
    for label in labels:
        if trimmed.casefold() == label.casefold():
            return label, "recovered"
    if _is_out_of_set(trimmed):
        return trimmed, "out_of_set"
    return None, "unparsed"


def _record(
    index: int, raw: str, parsed: str | None, kind: VerdictKind
) -> VerdictRecord:
    """Build one record with exactly the keys of ``VerdictRecord``.

    ``raw`` is the untrimmed first line of the located text, capped: Step 1
    has not removed tokens or punctuation from it, so the UI can show what
    the model actually printed.
    """

    return {
        "index": index,
        "raw": _cap(_first_line(raw)),
        "parsed": parsed,
        "kind": kind,
    }


def _table_cell(cell: str | None) -> str:
    """Strip the first-pass table's own ``Verdict: `` formatting.

    The first script table prints ``12 | Verdict: Conflicting``; the prefix is
    the table's layout, not part of the model's answer. The second table has
    bare values already, so this is a no-op there.
    """

    if cell is None:
        return ""
    return _LEADING_VERDICT.sub("", cell.strip(), count=1).strip()


# ---------------------------------------------------------------------------
# Layout resolution
# ---------------------------------------------------------------------------


def verdict_source(layout: str) -> VerdictSource:
    """Return the ``ScoreReport`` source for ``layout``.

    ``banner`` and ``think`` produce verdicts from the decoded records
    (``"parsed"``); ``table_only`` has no decoded record in the log, so the
    script's second table is the source (``"script_table"``). An unknown
    layout is a caller error.
    """

    try:
        return SOURCES[layout]
    except KeyError:
        raise ValueError(
            f"unknown output layout {layout!r}; expected one of "
            f"{list(OUTPUT_FORMATS)!r}"
        ) from None


def _resolve_layout(path: str | Path, layout: OutputLayout | None) -> OutputLayout:
    """Use the caller's layout or ask :func:`detect` for one.

    :func:`detect` is consulted only when ``layout`` is ``None`` and exists
    only to resolve the layout, not to decide completeness. The records are
    therefore identical whether or not the caller passes ``layout``; an
    explicit layout merely skips the extra pass.
    """

    if layout is not None:
        if layout not in OUTPUT_FORMATS:
            raise ValueError(
                f"unknown output layout {layout!r}; expected one of "
                f"{list(OUTPUT_FORMATS)!r}"
            )
        return layout

    result = detect(path)
    detected = result.get("output_format")
    if detected is None:
        raise ValueError(
            f"{path!s} has no inference output format; the log never reached "
            "the generation stage. Pass layout=... only when the layout is "
            "known independently."
        )
    if detected not in OUTPUT_FORMATS:
        raise ValueError(f"detect() reported an unsupported layout: {detected!r}")
    return detected  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# banner
# ---------------------------------------------------------------------------


def _banner_value(block: Sequence[str], marker: str) -> str:
    """Return the located value of one ``OUTPUT`` block (Step 0).

    The value is the text after the *last* marker (``rfind``). A trailing
    bare list of responses (some runs print every verdict again with no
    banner) starts at the first pair of consecutive marker lines; the list is
    not part of the record, so the search is cut just after the first line of
    the pair.
    """

    cutoff = len(block)
    for position in range(len(block) - 1):
        if marker in block[position] and marker in block[position + 1]:
            cutoff = position + 1
            break

    found: str | None = None
    for line in block[:cutoff]:
        at = line.rfind(marker)
        if at >= 0:
            found = line[at + len(marker) :]
    return _NO_RESPONSE if found is None else found


def _banner_records(
    path: str | Path,
    marker: str,
    labels: Sequence[str],
) -> list[VerdictRecord]:
    """One record per ``OUTPUT`` block, in file order.

    A block closed by a later banner is always a record, even without a
    marker. The block still open at end of file is a record only when it
    holds a ``### Response:`` line; otherwise the record is missing rather
    than invented as ``empty``.
    """

    records: list[VerdictRecord] = []
    block: list[str] = []
    inside = False

    def close() -> None:
        nonlocal block, inside
        if not inside:
            return
        value = _banner_value(block, marker)
        parsed, kind = _classify(value, labels)
        records.append(_record(len(records), value, parsed, kind))
        block = []
        inside = False

    with open(path, encoding="utf-8", errors="replace", newline=None) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            match = _RANK_PREFIX.match(line)
            if match:
                line = line[match.end() :]

            banner = _BANNER.match(line)
            if banner is not None:
                section = banner.group(1)
                if section == "OUTPUT":
                    close()
                    inside = True
                    block = []
                elif inside and section in ("INPUTS", "VERDICTS"):
                    # The next INPUTS (or the VERDICTS section) proves this
                    # OUTPUT block was completed.
                    close()
                continue

            if inside:
                block.append(line)

    # The final open block is a record only when the model printed a response
    # line. Without one the run never finished the record, so it stays
    # missing and scoring reports it as ``unfinished``.
    if inside and any(line.startswith(_RESPONSE) for line in block):
        close()
    return records


# ---------------------------------------------------------------------------
# think / table_only
# ---------------------------------------------------------------------------


def _think_value(block: Sequence[str]) -> str:
    """Return the located value of one decoded record (Step 0 for think).

    Only the text after the last ``### Response:`` is searched, so the
    ``Verdict: <verdict>`` in the echoed prompt is never an answer. The value
    is the text after the last ``Verdict: `` in the generation; when there is
    none, the first non-empty generated line is returned, stripped, as raw
    evidence for the UI while Step 2 reports ``empty``.
    """

    content = "\n".join(block)
    at = content.rfind(_RESPONSE)
    if at < 0:
        return _NO_RESPONSE
    generated = content[at + len(_RESPONSE) :]

    marker_at = generated.rfind(_VERDICT_PREFIX)
    if marker_at >= 0:
        return generated[marker_at + len(_VERDICT_PREFIX) :]

    for line in generated.splitlines():
        if line.strip():
            # The space that follows ``### Response:`` is layout, not text;
            # stripping it cannot change a classification, which strips the
            # first line too.
            return line.strip()
    return _NO_RESPONSE


def _scan_timestamped(
    path: str | Path,
    *,
    collect_decoded: bool,
) -> tuple[list[dict[int, str]], list[str]]:
    """One streaming pass over a ``think``/``table_only`` log.

    Returns the tables in file order (a new table starts when the row index
    stops increasing) and, when requested, one located value per decoded
    record. Only the current record block is buffered.

    A decoded record is a record only when ``Done Decoding`` was seen for it
    (the line precedes the ``Decoded output:`` header in the real logs) and
    its block holds a ``### Response:`` line. The next record's start (the
    ``Tokenizing Inputs`` phase line or the ``***`` separator) closes the
    block, and a block still open at end of file is tested the same way, so a
    record the run never finished stays missing.
    """

    tables: list[dict[int, str]] = []
    decoded: list[str] = []
    block: list[str] = []
    inside = False
    has_response = False
    has_done_decoding = False
    # ``Done Decoding`` is printed *before* the ``Decoded output:`` header in
    # the real logs, so the phase flag is carried into the block the header
    # opens. A synthetic record that prints it inside the block also counts.
    pending_done_decoding = False
    previous_index: int | None = None

    def close() -> None:
        nonlocal block, inside, has_response, has_done_decoding
        if collect_decoded and inside and has_response and has_done_decoding:
            decoded.append(_think_value(block))
        block = []
        inside = False
        has_response = False
        has_done_decoding = False

    with open(path, encoding="utf-8", errors="replace", newline=None) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            match = _RANK_PREFIX.match(line)
            if match:
                line = line[match.end() :]

            row = _ROW.match(line)
            if row is not None:
                # A table row always ends the current decoded record; the
                # fallback bodies the script prints between rows are not
                # records and are ignored because no record is open.
                close()
                index = int(row.group(1))
                value = row.group(2).strip()
                if not tables or (
                    previous_index is not None and index <= previous_index
                ):
                    tables.append({})
                tables[-1][index] = value
                previous_index = index
                continue

            if line.startswith(_DECODED_HEADER):
                close()
                inside = True
                block = []
                has_done_decoding = pending_done_decoding
                pending_done_decoding = False
                continue

            if inside and _starts_next_record(line):
                # The next record has begun, so this one is complete even
                # though its closing ``Decoded output:`` has not arrived.
                close()
                continue

            if inside:
                block.append(line)
                if line.startswith(_RESPONSE):
                    has_response = True
                elif line.startswith(_DONE_DECODING):
                    has_done_decoding = True
            elif line.startswith(_DONE_DECODING):
                # Belongs to the record whose ``Decoded output:`` follows.
                pending_done_decoding = True

    close()
    return tables, decoded


def _table_row_record(
    index: int,
    first_cell: str | None,
    final_value: str,
    labels: Sequence[str],
) -> VerdictRecord:
    """Build a record when the script tables are the (only) source.

    Used for ``table_only`` and for a trimmed ``think`` excerpt whose decoded
    records do not cover every index. The final table supplies the value; the
    first table says whether the script's fallback had to recover it, and
    supplies the raw first-pass text for the UI.
    """

    final_parsed, final_kind = _classify(final_value, labels)

    # A trimmed excerpt may keep only some first-pass rows. Without the
    # counterpart there is no evidence of a fallback, so the final value is
    # classified on its own (a clean label stays exact).
    if first_cell is None:
        return _record(index, final_value, final_parsed, final_kind)

    if final_kind == "empty":
        return _record(index, first_cell, None, "empty")

    if final_kind in ("exact", "recovered"):
        # ``cell_kind`` is exact when the first pass already held the final
        # label; anything else (a placeholder, punctuation or a different
        # value) is a fallback recovery.
        cell_parsed, cell_kind = _classify(_table_cell(first_cell), labels)
        if cell_parsed == final_parsed and cell_kind == "exact":
            kind: VerdictKind = "exact"
        else:
            kind = "recovered"
        return _record(index, first_cell, final_parsed, kind)

    # Out-of-set or unparsed in the script's own final table.
    return _record(index, first_cell, final_parsed, final_kind)


def _think_records(
    path: str | Path, labels: Sequence[str]
) -> list[VerdictRecord]:
    """Records for a ``think`` log, decoded record first, final table next.

    The script tables are the fallback only when two of them were printed:
    a lone table is not the final output, so the decoded records are used
    instead.
    """

    tables, decoded = _scan_timestamped(path, collect_decoded=True)

    if len(tables) < 2:
        # A lone table is not a final table. The decoded records are the
        # source; a block is a record only when ``Done Decoding`` was seen
        # and it holds a response line.
        return [
            _record(index, value, *_classify(value, labels))
            for index, value in enumerate(decoded)
        ]

    first = tables[0]
    final = tables[-1]
    indices = sorted(final)

    # Decoded records align by position only when the excerpt kept all of
    # them, which is the case for a full log and never for a trimmed fixture.
    aligned = len(decoded) == len(indices) and indices == list(range(len(indices)))

    records: list[VerdictRecord] = []
    for index in indices:
        final_value = final[index]

        if not aligned:
            # A trimmed excerpt: the script tables are the only complete
            # source, exactly as for table_only.
            records.append(
                _table_row_record(index, first.get(index), final_value, labels)
            )
            continue

        final_parsed, final_kind = _classify(final_value, labels)
        primary_raw = decoded[index]
        primary_parsed, primary_kind = _classify(primary_raw, labels)

        if final_kind == "empty":
            records.append(_record(index, primary_raw, None, "empty"))
            continue

        if final_kind in ("exact", "recovered"):
            if primary_parsed == final_parsed and primary_kind in (
                "exact",
                "recovered",
            ):
                # The decoded record already carries the final value; keep
                # its exact-versus-recovered distinction.
                records.append(
                    _record(index, primary_raw, primary_parsed, primary_kind)
                )
            else:
                # The decoded record had no usable verdict and the final
                # table does: this is the script's fallback (log 04: 128,
                # 925 and 1064), so the value is recovered, not exact.
                raw = primary_raw or first.get(index, final_value)
                records.append(_record(index, raw, final_parsed, "recovered"))
            continue

        # Out-of-set or unparsed in the final table: keep the raw text.
        records.append(_record(index, primary_raw, final_parsed, final_kind))

    return records


def _table_only_records(
    path: str | Path, labels: Sequence[str]
) -> list[VerdictRecord]:
    """Records for a ``table_only`` log: the second table is the source.

    A lone table is not treated as final, so it yields no records: there is
    no decoded record to fall back on and nothing can be asserted as the
    script's final output.
    """

    tables, _ = _scan_timestamped(path, collect_decoded=False)
    if len(tables) < 2:
        return []

    first = tables[0]
    final = tables[-1]
    return [
        _table_row_record(index, first.get(index), final[index], labels)
        for index in sorted(final)
    ]


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def extract_verdicts_with_source(
    path: str | Path,
    *,
    marker: str,
    labels: Sequence[str] = DEFAULT_LABELS,
    layout: OutputLayout | None = None,
    source: VerdictSource | None = None,
) -> tuple[VerdictSource, list[VerdictRecord]]:
    """Extract records together with the ``score_verdicts`` source.

    ``marker`` is the configured ``p.verdict_parse_marker`` used by the
    ``banner`` layout. ``labels`` is ``c.label_values`` and defines the label
    domain; a value outside it is never coerced.

    When ``layout`` is ``None`` the layout is read from
    :func:`app.parsing.detect.detect`, and ``ValueError`` is raised for an
    inference log that never reached the output stage. ``detect()`` is used
    only to resolve the layout: the records are identical whether or not the
    caller passes ``layout``; an explicit layout merely skips that extra pass.

    The returned ``source`` is the one the resolved layout requires:
    ``"parsed"`` for ``banner``/``think`` and ``"script_table"`` for
    ``table_only``. Callers must pass it to
    :func:`app.scoring.score_verdicts`, which needs it to report where the
    verdicts came from.

    ``source`` is optional. When supplied it must equal
    :func:`verdict_source` of the resolved layout; a mismatch is a caller
    error, so a caller cannot label a table extraction as parsed output.

    The function makes one streaming pass over the file (plus the layout
    pass of ``detect`` when ``layout`` is ``None``). It raises ``ValueError``
    for an empty marker, an unknown layout or a source mismatch, and
    ``OSError`` for an unreadable path.
    """

    if not marker:
        raise ValueError("marker must be a non-empty string")
    if not labels:
        raise ValueError("labels must not be empty")

    resolved = _resolve_layout(path, layout)
    expected = verdict_source(resolved)
    if source is not None and source != expected:
        raise ValueError(
            f"source {source!r} does not match layout {resolved!r}, whose "
            f"source is {expected!r}"
        )

    if resolved == "banner":
        records = _banner_records(path, marker, labels)
    elif resolved == "think":
        records = _think_records(path, labels)
    else:
        records = _table_only_records(path, labels)
    return expected, records


def extract_verdicts(
    path: str | Path,
    *,
    marker: str,
    labels: Sequence[str] = DEFAULT_LABELS,
    layout: OutputLayout | None = None,
    source: VerdictSource | None = None,
) -> list[VerdictRecord]:
    """Extract one verdict record per index present in ``path``.

    A thin wrapper around :func:`extract_verdicts_with_source` that keeps
    only the records. Use the latter when the records are going to
    :func:`app.scoring.score_verdicts`, because that function also needs the
    ``source``.
    """

    return extract_verdicts_with_source(
        path, marker=marker, labels=labels, layout=layout, source=source
    )[1]
