"""Cut trimmed fixture excerpts from the full sample logs (issue #29).

Usage:
    python make_fixtures.py <logs_dir> <out_dir>

<logs_dir> holds sampleOutput01.txt ... sampleOutput15.txt (never committed).
<out_dir> receives the excerpts named s01_inference_error.log etc. plus the
synthetic terminated fine-tune log.

Rules (spec 3.9): first 150 lines, last 150 lines, 10 lines around every
evidence line, and for inference logs a handful of complete records with the
two result tables trimmed to the same indices. Everything is done on raw
bytes so CRLF line endings survive. Omitted stretches are replaced by one
ASCII marker line so nobody mistakes an excerpt for a whole log.
"""
import re
import sys
from pathlib import Path

HEAD = 150
TAIL = 150
CTX = 10
MAX_BYTES = 200_000
FIRST_RECORDS = 5

RANK = re.compile(r"^\[rank\d+\]: ")
P = {
    "phase_ts": re.compile(r"^(?P<label>[A-Za-z][\w ()'-]+?)\s+[:-]\s+(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)"),
    "script": re.compile(r'File ".*/(?P<script>[\w-]+\.py)", line \d+, in <module>'),
    "banner": re.compile(r"^[*#]{5,}\s*(INPUTS|OUTPUT|VERDICTS|LENGTH)\s*[*#]{5,}$"),
    "table_row": re.compile(r"^(\d+) \| (.*)$"),
    "loss": re.compile(r"^\{'loss': '"),
    "eval_loss": re.compile(r"\{'eval_loss': '([\d.e-]+)'"),
    "progress": re.compile(r"\|\s*(\d+)/(\d+) \["),
    "shards": re.compile(r"Writing model shards: 100%"),
    "traceback": re.compile(r"^Traceback \(most recent call last\):"),
    "exception": re.compile(r"^([\w.]+(Error|Exception)): (.*)$"),
    "oom": re.compile(r"OutOfMemoryError|CUDA error: out of memory"),
    "missing_lib": re.compile(r"ModuleNotFoundError|Configured CUDA binary not found"),
    "exitcode": re.compile(r"exitcode\s*:\s*-11"),
    "childfailed": re.compile(r"ChildFailedError"),
    "srun_error": re.compile(r"^srun: error:"),
    "time_limit": re.compile(r"slurmstepd: error: \*\*\* JOB \d+ .* CANCELLED AT .* DUE TO TIME LIMIT \*\*\*"),
}
ALWAYS = ("traceback", "exception", "oom", "missing_lib", "exitcode", "childfailed",
          "srun_error", "time_limit", "script", "shards", "eval_loss")

# (sample number) -> (fixture name, layout, labels for expected.json handled elsewhere)
NAMES = {
    1: "s01_inference_error.log", 2: "s02_inference_success.log",
    3: "s03_inference_success.log", 4: "s04_inference_success.log",
    5: "s05_inference_error.log", 6: "s06_inference_error.log",
    7: "s07_inference_success.log", 8: "s08_fine_tune_error.log",
    9: "s09_fine_tune_error.log", 10: "s10_fine_tune_error.log",
    11: "s11_fine_tune_error.log", 12: "s12_fine_tune_success_warning.log",
    13: "s13_checkpoint_export_success.log", 14: "s14_inference_error.log",
    15: "s15_inference_terminated.log",
}
RECORD_LAYOUT = {2: "banner", 3: "banner", 5: "banner", 15: "banner", 4: "tables", 7: "tables"}


def decode(line: bytes) -> str:
    return RANK.sub("", line.decode("utf-8", "replace").rstrip("\r"))


def read_lines(path: Path):
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    return lines


def classify(lines):
    hits = {k: [] for k in P}
    prog_by_total = {}
    for i, line in enumerate(lines):
        s = decode(line)
        for k, pat in P.items():
            m = pat.search(s)
            if m:
                hits[k].append(i)
                if k == "progress":
                    prog_by_total.setdefault(int(m.group(2)), []).append(i)
    return hits, prog_by_total


def add_range(keep, a, b, n):
    for i in range(max(0, a), min(n, b + 1)):
        keep.add(i)


def banner_records(lines, hits):
    """Record = from an INPUTS banner to the line before the next INPUTS banner."""
    starts = [i for i in hits["banner"] if "INPUTS" in decode(lines[i])]
    return starts


def timestamp_records(lines):
    return [i for i, l in enumerate(lines) if l.startswith(b"Tokenizing Inputs")]


def table_rows(lines, hits):
    rows = [(i, int(P["table_row"].match(decode(lines[i])).group(1)), P["table_row"].match(decode(lines[i])).group(2))
            for i in hits["table_row"]]
    half = len(rows) // 2
    return rows[:half], rows[half:]


def pick_indices(t1, t2):
    """First 5, every fallback recovery, first two empties, first two punctuation trims, last."""
    idx = set(range(FIRST_RECORDS)) | {t1[-1][1]}
    empties = trims = 0
    for a, b in zip(t1, t2):
        x = re.sub(r"^Verdict: ", "", a[2].rstrip("\r"))
        y = b[2].rstrip("\r")
        if a[2].strip() == "Verdict: <verdict>" and y in ("True", "False", "Conflicting"):
            idx.add(a[1])  # recovered by the script's fallback
        elif a[2].strip() == "Verdict: <verdict>" and y == "Empty Response" and empties < 2:
            idx.add(a[1]); empties += 1
        elif x != y and x.rstrip(".") == y and trims < 2:
            idx.add(a[1]); trims += 1
    return sorted(idx)


def build_keep(lines, n_sample, ctx=CTX):
    n = len(lines)
    hits, prog = classify(lines)
    keep = set()
    add_range(keep, 0, HEAD - 1, n)
    add_range(keep, n - TAIL, n - 1, n)
    for k in ALWAYS:
        pos = hits[k]
        if len(pos) > 40:
            pos = pos[:20] + pos[-20:]
        for i in pos:
            add_range(keep, i - ctx, i + ctx, n)
    # repetitive patterns: first and last few only
    pos = hits["phase_ts"]
    for i in pos[:20] + pos[-8:]:
        add_range(keep, i - 2, i + 2, n)
    for i in hits["loss"][:3] + hits["loss"][-3:]:
        add_range(keep, i - 2, i + 2, n)
    for total, where in prog.items():
        for i in where[:2] + where[-2:]:
            add_range(keep, i - 1, i + 1, n)

    layout = RECORD_LAYOUT.get(n_sample)
    note = ""
    if layout == "banner":
        starts = banner_records(lines, hits)
        chosen = list(range(min(FIRST_RECORDS, len(starts)))) + [len(starts) - 2, len(starts) - 1]
        for r in sorted(set(c for c in chosen if 0 <= c < len(starts))):
            end = starts[r + 1] - 1 if r + 1 < len(starts) else min(n - 1, starts[r] + 120)
            add_range(keep, starts[r], end, n)
        note = f"{len(starts)} INPUTS banners"
    elif layout == "tables":
        starts = timestamp_records(lines)
        t1, t2 = table_rows(lines, hits)
        idx = pick_indices(t1, t2)
        first_row = t1[0][0]
        for r in idx:
            end = starts[r + 1] - 1 if r + 1 < len(starts) else first_row - 4
            end = min(end, starts[r] + 80)
            add_range(keep, starts[r], end, n)
        for table in (t1, t2):
            add_range(keep, table[0][0] - 3, table[0][0] - 1, n)
            by_idx = {row[1]: row[0] for row in table}
            for r in idx:
                keep.add(by_idx[r])
        note = f"{len(starts)} records, kept indices {idx}"
    return keep, note


def render(lines, keep):
    out = []
    prev = -1
    for i in sorted(keep):
        if i != prev + 1:
            omitted = i - prev - 1
            out.append(f"[... {omitted} lines omitted from the original log ...]".encode("ascii") + b"\r")
        out.append(lines[i])
        prev = i
    if prev != len(lines) - 1:
        out.append(f"[... {len(lines) - 1 - prev} lines omitted from the original log ...]".encode("ascii") + b"\r")
    return b"\n".join(out) + b"\n"


SCRUB = [
    (re.compile(rb"gaganm"), b"user"),
    (re.compile(rb"/project/\d+"), b"/project/0000000"),
    (re.compile(rb"\bfc\d{5}\b"), b"fc00000"),
]


def scrub(data: bytes) -> bytes:
    """Replace the researcher's cluster username, project id and host names."""
    for pat, rep in SCRUB:
        data = pat.sub(rep, data)
    return data


def cut(path: Path, n_sample: int):
    lines = read_lines(path)
    for ctx in (CTX, 6, 4, 2):
        keep, note = build_keep(lines, n_sample, ctx)
        data = scrub(render(lines, keep))
        if len(data) <= MAX_BYTES:
            return data, note, len(lines), ctx
    return data, note + " (STILL TOO BIG)", len(lines), ctx


def synthetic(logs: Path) -> bytes:
    """Fine-tune run cut off by the time limit, built from sample 10 (real text)."""
    lines = read_lines(logs / "sampleOutput10.txt")
    hits, _ = classify(lines)
    cutoff = hits["loss"][20]  # keep the run up to its 21st logged loss
    sub = lines[: cutoff + 6]
    keep, _ = build_keep(sub, 0)
    keep |= set(range(len(sub) - 6, len(sub)))
    body = render(sub, keep)
    header = (b"# SYNTHETIC FIXTURE - not a real log. Fine-tune run cut off by the SLURM time limit.\r\n"
              b"# Built from sampleOutput10 up to a mid-training step, plus a time-limit line.\r\n")
    last = (b"slurmstepd: error: *** JOB 99999999 ON fc00000 CANCELLED AT 2026-03-27T18:48:59 "
            b"DUE TO TIME LIMIT ***\r\n")
    return scrub(header + body + last)


def main():
    logs, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    print(f"{'fixture':<38}{'orig lines':>11}{'orig KB':>9}{'out KB':>8}{'ctx':>5}  note")
    for n, name in NAMES.items():
        src = logs / f"sampleOutput{n:02d}.txt"
        data, note, nlines, ctx = cut(src, n)
        (out / name).write_bytes(data)
        print(f"{name:<38}{nlines:>11,}{src.stat().st_size / 1000:>9.0f}{len(data) / 1000:>8.1f}{ctx:>5}  {note}")
    syn = synthetic(logs)
    (out / "synthetic_fine_tune_terminated.log").write_bytes(syn)
    print(f"{'synthetic_fine_tune_terminated.log':<38}{'':>11}{'':>9}{len(syn) / 1000:>8.1f}")


if __name__ == "__main__":
    main()
