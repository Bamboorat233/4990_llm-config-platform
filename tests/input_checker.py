"""Input checker: simulate a user entering choices, then validate them.

There is no finished UI yet, so this stands in for the person using the
form. The user's choices live in one dictionary, ``user_input``:

    user_input = {
        "model": "mistral",         # also accepted: "nameModel"
        "run_mode": "inference",    # fine_tune / inference, also: "runMode"
        "pad_token": "<myPadToken>",
        "temperature": 0,
        "do_sample": True,
    }

Names come from the CHOICES table below (``pad_token`` is ``b.pad_token``);
a full field path from specs/interfaces.md works for anything else.
Whatever the user does not choose falls back to the form's defaults.

The dictionary is built at run time, in any of three ways:
  * typed at the prompt (guided, one choice at a time, or as edit lines),
  * loaded from the cases file (one dictionary per case),
  * generated in code (tests/input_sweeps.py: many combinations).

Each dictionary then goes through the real pipeline:

    defaults_for() -> user choices -> ConfigRecord -> validate_cross_fields()
        -> console.evaluate() / apply_issue_props() -> (optional) save

The report separates two things:
  * OUTCOME - what the system did (OK / BLOCKED / SCHEMA_ERROR, red fields,
              warnings, whether it was saved)
  * VERDICT - PASS or FAIL against the expectation; "none" if no expectation.

Usage (repo root, venv active):

    python tests/input_checker.py                 # interactive prompt
    python tests/input_checker.py --run           # run every saved case
    python tests/input_checker.py --run --sweeps  # ... plus generated inputs
    python tests/input_checker.py --db my.db      # keep saved configs here
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import ValidationError  # noqa: E402

from app import store  # noqa: E402
from app.registry import DEEPSEEK_ID, MISTRAL_ID, defaults_for  # noqa: E402
from app.schemas import ConfigRecord, validate_cross_fields  # noqa: E402
from app.ui import console  # noqa: E402

CASES_FILE = Path(__file__).resolve().parent / "cases" / "input_cases.json"
MODEL_ALIASES = {"mistral": MISTRAL_ID, "deepseek": DEEPSEEK_ID}
SHORT_NAMES = {MISTRAL_ID: "mistral", DEEPSEEK_ID: "deepseek"}
STATUSES = ("OK", "BLOCKED", "SCHEMA_ERROR")

# (name, field path, which run mode shows it). Order = order of the guided prompt.
CHOICES = [
    ("pad_token", "b.pad_token", "both"),
    ("padding_side", "b.padding_side", "both"),
    ("fix_mistral_regex", "b.fix_mistral_regex", "both"),
    ("tp_plan", "a.tp_plan", "both"),
    ("device_map", "a.device_map", "both"),
    ("train_data_file", "c.train_data_file", "fine_tune"),
    ("eval_data_file", "c.eval_data_file", "fine_tune"),
    ("test_data_file", "c.test_data_file", "inference"),
    ("eval_strategy", "d.sft.eval_strategy", "fine_tune"),
    ("save_strategy", "d.sft.save_strategy", "fine_tune"),
    ("load_best_model_at_end", "d.sft.load_best_model_at_end", "fine_tune"),
    ("max_length", "e.max_length", "both"),
    ("max_new_tokens", "f.max_new_tokens", "both"),
    ("batch_size", "f.batch_size", "both"),
    ("temperature", "f.temperature", "both"),
    ("top_p", "f.top_p", "both"),
    ("top_k", "f.top_k", "both"),
    ("do_sample", "f.do_sample", "both"),
    ("system_prompt", "p.system_prompt", "both"),
    ("user_prompt_template", "p.user_prompt_template", "both"),
    ("assistant_prompt", "p.assistant_prompt", "both"),
    ("verdict_parse_marker", "p.verdict_parse_marker", "both"),
]
PATH_BY_NAME = {name: path for name, path, _ in CHOICES}
PATH_BY_NAME.update({"created_by": "created_by", "notes": "notes"})  # typed-only
NAME_BY_PATH = {path: name for name, path, _ in CHOICES}
KEY_ALIASES = {"nameModel": "model", "model_id": "model",
               "runMode": "run_mode", "task": "run_mode", "task_type": "run_mode"}


# ------------------------------------------------------------ input parsing

def resolve_model(name: str) -> str:
    """Accept 'mistral', 'deepseek' or a full model id."""
    key = str(name).strip()
    return MODEL_ALIASES.get(key.lower(), key)


def parse_value(text: str):
    """Read a typed value: JSON when it parses, plain text otherwise."""
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        return text


def parse_edit_line(line: str) -> tuple:
    """Split 'temperature = 0' into ('temperature', 0)."""
    if "=" not in line:
        raise ValueError(f"expected 'name = value', got: {line!r}")
    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"missing name in: {line!r}")
    return key, parse_value(value)


def _edit_pairs(edits) -> list:
    if not edits:
        return []
    if isinstance(edits, dict):
        return list(edits.items())
    return [parse_edit_line(e) if isinstance(e, str) else tuple(e) for e in edits]


def case_to_user_input(case: dict) -> dict:
    """The user's choices for a case (older cases used model / task / edits)."""
    if "user_input" in case:
        return dict(case["user_input"])
    user_input = {"model": case.get("model", "mistral"),
                  "run_mode": case.get("task", "inference")}
    for key, value in _edit_pairs(case.get("edits")):
        user_input[key] = value
    return user_input


def expand_user_input(user_input: dict) -> tuple:
    """user_input -> (model id, run mode, [(field path, value), ...])."""
    choices = {KEY_ALIASES.get(k, k): v for k, v in user_input.items()}
    model = resolve_model(choices.pop("model", "mistral"))
    run_mode = choices.pop("run_mode", "inference")
    pairs = []
    for key, value in choices.items():
        path = PATH_BY_NAME.get(key) or (key if "." in key else None)
        if path is None:
            raise ValueError(
                f"unknown choice {key!r}; use one of {', '.join(sorted(PATH_BY_NAME))} "
                "or a full field path such as b.pad_token")
        pairs.append((path, value))
    return model, run_mode, pairs


def deep_merge(base: dict, overlay: dict) -> dict:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def apply_edits(payload: dict, pairs: list) -> dict:
    """Set each (field path, value); missing parents are created."""
    for path, value in pairs:
        *parents, leaf = path.split(".")
        node = payload
        for key in parents:
            if not isinstance(node.get(key), dict):
                node[key] = {}
            node = node[key]
        node[leaf] = value
    return payload


def get_path(payload: dict, path: str):
    node = payload
    for key in path.split("."):
        node = node.get(key) if isinstance(node, dict) else None
    return node


def parse_expect_text(text: str) -> dict:
    """'BLOCKED red=b.pad_token,f.do_sample warn=p.system_prompt'."""
    parts = text.split()
    if not parts:
        raise ValueError("empty expectation")
    status = parts[0].upper()
    if status not in STATUSES:
        raise ValueError(f"status must be one of {', '.join(STATUSES)}")
    expect: dict = {"status": status}
    for token in parts[1:]:
        key, _, rest = token.partition("=")
        items = [p for p in rest.split(",") if p]
        if key == "red":
            expect["red"] = items
        elif key in ("warn", "warnings"):
            expect["warnings"] = items
        elif key == "saved":
            expect["saved"] = rest.lower() in ("1", "true", "yes", "y")
        else:
            raise ValueError(f"unknown expectation part: {token!r}")
    return expect


# ------------------------------------------------------------ the pipeline

class _Widget:
    """Stand-in for a NiceGUI element: remembers whether it is red."""

    def __init__(self) -> None:
        self.error_message = None

    def props(self, add: str = None, remove: str = None) -> None:
        if remove and "error" in remove:
            self.error_message = None
        if add and add.startswith("error"):
            self.error_message = add[add.index('"') + 1:add.rindex('"')]


def leaf_paths(node: dict, prefix: str = "") -> list:
    paths = []
    for key, value in node.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            paths += leaf_paths(value, path + ".")
        else:
            paths.append(path)
    return paths


@dataclass
class Result:
    status: str
    save_enabled: bool
    red: list
    errors: list = field(default_factory=list)      # (field, msg)
    warnings: list = field(default_factory=list)    # (field, msg)
    unmapped: list = field(default_factory=list)    # error fields with no widget
    json_text: str = ""
    saved_id: str = None
    save_note: str = ""


def run_case(case: dict, db_path=None) -> Result:
    """Take the user's choices, run the real pipeline, return the outcome."""
    model_id, run_mode, pairs = expand_user_input(case_to_user_input(case))
    payload = defaults_for(model_id, run_mode)  # ValueError on a bad model / mode
    if case.get("config"):
        deep_merge(payload, copy.deepcopy(case["config"]))
    apply_edits(payload, pairs)

    widgets = {p: _Widget() for p in leaf_paths(defaults_for(MISTRAL_ID, "fine_tune"))}
    try:
        cfg = ConfigRecord(**copy.deepcopy(payload))
    except ValidationError as exc:
        details = [(".".join(str(x) for x in e["loc"]), e["msg"]) for e in exc.errors()]
        return Result(
            status="SCHEMA_ERROR", save_enabled=False,
            red=sorted({p for p, _ in details}), errors=details,
            json_text=json.dumps(payload, indent=2, ensure_ascii=False),
        )

    issues = list(validate_cross_fields(cfg))
    state = console.evaluate(cfg, issues)
    red = sorted(console.apply_issue_props(widgets, issues))
    result = Result(
        status="BLOCKED" if state.errors else "OK",
        save_enabled=state.save_enabled,
        red=red,
        errors=[(i["field"], i["msg"]) for i in state.errors],
        warnings=[(i["field"], i["msg"]) for i in state.warnings],
        unmapped=sorted({i["field"] for i in state.errors if i["field"] not in widgets}),
        json_text=state.json_text,
    )
    if case.get("save"):
        if not state.save_enabled:
            result.save_note = "not saved: Save is disabled while errors exist"
        else:
            try:
                result.saved_id = store.insert_config(cfg, db_path)
                result.save_note = f"saved as {result.saved_id}"
            except ValueError as exc:  # duplicate fingerprint
                result.save_note = f"not saved: {exc}"
    return result


def judge(result: Result, expect) -> tuple:
    """Compare an outcome with an expectation -> (verdict, mismatches)."""
    if not expect:
        return "NONE", []
    if "status" not in expect:
        raise ValueError("an expectation needs a 'status'")
    problems = []
    if result.status != expect["status"]:
        problems.append(f"status: expected {expect['status']}, got {result.status}")
    if "red" in expect and set(result.red) != set(expect["red"]):
        problems.append(f"red fields: expected {sorted(expect['red'])}, got {result.red}")
    if "warnings" in expect:
        got = sorted({f for f, _ in result.warnings})
        if set(got) != set(expect["warnings"]):
            problems.append(f"warnings: expected {sorted(expect['warnings'])}, got {got}")
    if "saved" in expect and (result.saved_id is not None) != bool(expect["saved"]):
        problems.append(
            f"saved: expected {bool(expect['saved'])}, got {result.saved_id is not None}")
    return ("FAIL" if problems else "PASS"), problems


# ------------------------------------------------------------ report text

def describe_input(case: dict) -> str:
    user_input = case_to_user_input(case)
    if case.get("config"):
        user_input = {**user_input, "config (merged)": case["config"]}
    return json.dumps(user_input, ensure_ascii=False)


def describe_expect(expect) -> str:
    if not expect:
        return "(none given)"
    text = expect["status"]
    if expect.get("red"):
        text += ", red: " + ", ".join(expect["red"])
    if expect.get("warnings"):
        text += ", warning: " + ", ".join(expect["warnings"])
    if "saved" in expect:
        text += ", saved" if expect["saved"] else ", not saved"
    return text


def describe_outcome(result: Result) -> str:
    text = result.status
    if result.status == "OK":
        text += " (Save enabled)"
    elif result.status == "BLOCKED":
        text += " (Save disabled)"
    if result.red:
        label = "invalid fields" if result.status == "SCHEMA_ERROR" else "red"
        text += f", {label}: " + ", ".join(result.red)
    if result.warnings:
        text += ", warning: " + ", ".join(sorted({f for f, _ in result.warnings}))
    if not (result.red or result.warnings) and result.status == "OK":
        text += ", no issues"
    return text


def format_report(index: int, name: str, case: dict, result: Result,
                  verdict: str, problems: list, show_json: bool = False) -> str:
    verdict_text = {"NONE": "none (no expectation given)"}.get(verdict, verdict)
    lines = [
        f"CASE {index}  {name}",
        f"  user input: {describe_input(case)}",
        f"  expected:   {describe_expect(case.get('expect'))}",
        f"  outcome:    {describe_outcome(result)}",
    ]
    for field_path, message in result.errors:
        lines.append(f"      ERROR    {field_path}: {message}")
    for field_path, message in result.warnings:
        lines.append(f"      WARNING  {field_path}: {message}")
    if result.unmapped:
        lines.append("      NOTE     error on a field the form has no widget for: "
                     + ", ".join(result.unmapped))
    if result.save_note:
        lines.append(f"      SAVE     {result.save_note}")
    suffix = ("   <- " + "; ".join(problems)) if problems else ""
    lines.append(f"  verdict:    {verdict_text}{suffix}")
    if show_json:
        lines.append("  json preview:")
        lines += ["    " + ln for ln in result.json_text.splitlines()]
    return "\n".join(lines)


# ------------------------------------------------------------ cases file

def load_cases(path=CASES_FILE) -> list:
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def append_case(path, case: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"cases": []}
    if any(c["name"] == case["name"] for c in data["cases"]):
        raise ValueError(f"a case named {case['name']!r} already exists")
    data["cases"].append(case)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_all(cases: list, db_path=None, show_json=False, out=print,
            quiet_passes=False) -> int:
    """Run every case, print the reports and a summary. Returns the exit code."""
    counts = {"PASS": 0, "FAIL": 0, "SKIPPED": 0, "NONE": 0}
    for index, case in enumerate(cases, start=1):
        name = case.get("name", f"case-{index}")
        if case.get("skip"):
            counts["SKIPPED"] += 1
            out(f"CASE {index}  {name}\n  user input: {describe_input(case)}\n"
                f"  verdict:    SKIPPED ({case['skip']})\n")
            continue
        try:
            result = run_case(case, db_path)
            verdict, problems = judge(result, case.get("expect"))
        except ValueError as exc:
            out(f"CASE {index}  {name}\n  verdict:    FAIL   <- bad case definition: {exc}\n")
            counts["FAIL"] += 1
            continue
        counts[verdict] += 1
        if quiet_passes and verdict == "PASS":
            continue
        out(format_report(index, name, case, result, verdict, problems, show_json) + "\n")
    out(f"SUMMARY: {counts['PASS']} passed, {counts['FAIL']} failed, "
        f"{counts['SKIPPED']} skipped, {counts['NONE']} without expectation "
        f"({len(cases)} cases)")
    return 1 if counts["FAIL"] else 0


# ------------------------------------------------------------ the prompt

def _ask(input_fn, text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input_fn(f"{text}{suffix}: ").strip()
    return answer or default


def _yes(input_fn, text: str, default: bool = False) -> bool:
    answer = input_fn(f"{text} ({'Y/n' if default else 'y/N'}): ").strip().lower()
    return default if not answer else answer.startswith("y")


def _shown(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = text.replace("\n", "\\n")
    return text if len(text) <= 40 else text[:37] + "..."


def _guided_choices(model_id: str, run_mode: str, input_fn, out) -> dict:
    """Ask for each choice in turn; Enter keeps the default shown."""
    defaults = defaults_for(model_id, run_mode)
    choices: dict = {}
    out("Enter a value, or press Enter to keep the default shown. "
        "Type . to skip the rest.\n"
        "Values: numbers, true/false and null are read as such; \"\" is an empty text.")
    for name, path, applies in CHOICES:
        if applies not in ("both", run_mode):
            continue
        answer = input_fn(f"  {name} [{_shown(get_path(defaults, path))}]: ").strip()
        if answer == ".":
            break
        if answer:
            choices[name] = parse_value(answer)
    return choices


def _edit_choices(input_fn, out) -> tuple:
    """Edit lines: 'name = value'. A JSON object line is merged over defaults."""
    out("Enter edits, one per line: name = value  (e.g. temperature = 0).\n"
        "A JSON object on one line is merged over the defaults. Blank line to run.")
    choices: dict = {}
    config: dict = {}
    while True:
        line = input_fn("edit> ").strip()
        if not line:
            return choices, config
        if line.startswith("{"):
            try:
                deep_merge(config, json.loads(line))
            except ValueError as exc:
                out(f"  not valid JSON: {exc}")
            continue
        try:
            key, value = parse_edit_line(line)
        except ValueError as exc:
            out(f"  {exc}")
            continue
        choices[key] = value


def interactive(db_path, cases_file=CASES_FILE, input_fn=input, out=print) -> None:
    """Build one user_input dictionary at a time, show the report, optionally keep it."""
    out("Input checker. You enter choices; it validates them like the form will.\n"
        "Ctrl+C or Ctrl+D to quit.\n")
    number = 0
    try:
        while True:
            number += 1
            model = _ask(input_fn, "Model (mistral / deepseek)", "mistral")
            run_mode = _ask(input_fn, "Run mode (fine_tune / inference)", "inference")
            try:
                defaults_for(resolve_model(model), run_mode)
            except ValueError as exc:
                out(f"  {exc}\n")
                number -= 1
                continue

            style = _ask(input_fn, "Enter choices: guided (one at a time) or edits",
                         "guided").lower()
            config: dict = {}
            if style.startswith("e"):
                choices, config = _edit_choices(input_fn, out)
            else:
                choices = _guided_choices(resolve_model(model), run_mode, input_fn, out)
            user_input = {"model": model, "run_mode": run_mode, **choices}

            expect = None
            while True:
                text = input_fn("expect (e.g. BLOCKED red=b.pad_token; blank = none)> ").strip()
                if not text:
                    break
                try:
                    expect = parse_expect_text(text)
                    break
                except ValueError as exc:
                    out(f"  {exc}")

            case = {"name": f"typed-{number}", "user_input": user_input,
                    "save": _yes(input_fn, "Save to DB if allowed?")}
            if config:
                case["config"] = config
            if expect:
                case["expect"] = expect
            try:
                result = run_case(case, db_path)
                verdict, problems = judge(result, expect)
            except ValueError as exc:
                out(f"  could not run: {exc}\n")
                continue
            show = _yes(input_fn, "Show the JSON preview?")
            out("\n" + format_report(number, case["name"], case, result, verdict,
                                     problems, show) + "\n")

            if _yes(input_fn, "Keep this as a saved case?"):
                name = _ask(input_fn, "Case name", f"case-{number}")
                kept_expect, note = expect, None
                while kept_expect is None:
                    out("A kept case needs an expectation. Write what the SPEC says\n"
                        "should happen, not just what the output above shows. Type `same`\n"
                        "only after you have checked the outcome against the spec.")
                    text = input_fn("expected> ").strip()
                    if text.lower() == "same":
                        kept_expect = {"status": result.status, "red": list(result.red),
                                       "warnings": sorted({f for f, _ in result.warnings})}
                        note = "expectation accepted from the observed outcome"
                    else:
                        try:
                            kept_expect = parse_expect_text(text)
                        except ValueError as exc:
                            out(f"  {exc}")
                saved_case = {"name": name, "user_input": user_input}
                if config:
                    saved_case["config"] = config
                if case["save"]:
                    saved_case["save"] = True
                saved_case["expect"] = kept_expect
                if note:
                    saved_case["note"] = note
                try:
                    append_case(cases_file, saved_case)
                    out(f"  kept as {name!r} in {cases_file}\n")
                except ValueError as exc:
                    out(f"  {exc}\n")
            if not _yes(input_fn, "Another input?", default=True):
                break
    except (EOFError, KeyboardInterrupt):
        out("\nbye")


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # tokens like <｜end▁of▁sentence｜>
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="Simulate user input and validate it.")
    parser.add_argument("--run", action="store_true", help="run every saved case")
    parser.add_argument("--sweeps", action="store_true",
                        help="with --run: also run the generated inputs")
    parser.add_argument("--file", default=str(CASES_FILE), help="cases file")
    parser.add_argument("--db", default=None, help="database file for saves")
    parser.add_argument("--json", action="store_true", help="print the JSON preview")
    args = parser.parse_args(argv)

    db_path = args.db or str(Path(tempfile.mkdtemp(prefix="input_checker_")) / "checker.db")
    if args.run:
        cases = load_cases(args.file)
        if args.sweeps:
            import input_sweeps
            cases += input_sweeps.all_sweeps()
        return run_all(cases, db_path, args.json, quiet_passes=args.sweeps)
    interactive(db_path, args.file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
