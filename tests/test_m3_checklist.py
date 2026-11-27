"""M2.5 / M3.0 acceptance checklist tests (issue #66), first half: M2.5.

Turns every Milestone 2.5 acceptance item in ``planning/TaskList.md`` into an
executable assertion on the public generator contract:

    ConfigRecord -> app.codegen.generate_bundle() -> GenerationResult
        (script_code, requirements_txt, syntax_ok, parity_ok, pitfall_issues,
         marker_issues)

Expected values are written here from the TaskList wording and the model
facts, not read back from the generator. Where a check could pass trivially
(the verifier saying "fine"), the test also plants the defect in a copy of the
script and requires the verifier to catch it.

TaskList M2.5 item                                   -> tests in this module
----------------------------------------------------------------------------
Finalise the generator design (templates + injection) -> test_scripts_are_templates_with_config_literal_and_cli_overrides
Generate a script automatically from the config       -> test_each_model_task_script_compiles_and_has_zero_parity_diff
Script includes imports, hyperparameters, data
loading, LoRA/SFT (training), generation + verdict
extraction (inference)                                -> test_training_script_*, test_inference_script_*
Additionally generate requirements.txt                -> test_requirements_*
Validate against the reference section by section     -> test_each_model_task_script_compiles_and_has_zero_parity_diff
Fix the known pitfalls (6 sub-items)                  -> test_pitfall_1 ... test_pitfall_6, test_planted_*
Nov 28 demo checklist                                 -> test_demo_checklist_same_config_both_scripts
Also required by the issue:
  all 7 evidence_config modes work                    -> test_evidence_mode_*
  unknown **kwargs rejected                           -> test_unknown_kwargs_*
  requirements.txt conditional bitsandbytes /
  mistral-common                                      -> test_requirements_*

The M3.0 half (pure modules without NiceGUI, closed-loop end-to-end test) is
added in a second pull request once the comparison modules are merged.
"""

from __future__ import annotations

import __future__
import ast
import hashlib
import random
import re
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.codegen import ConfigValidationError, generate_bundle  # noqa: E402
from app.codegen.verify import check_known_pitfalls  # noqa: E402
from app.registry import DEEPSEEK_ID, MISTRAL_ID, defaults_for  # noqa: E402
from app.schemas import ConfigRecord  # noqa: E402
from app.validation import KWARGS_WHITELISTS  # noqa: E402

MODELS = (MISTRAL_ID, DEEPSEEK_ID)
TASKS = ("fine_tune", "inference")
COMBOS = [(m, t) for m in MODELS for t in TASKS]
COMBO_IDS = [f"{m.split('/')[-1][:8]}-{t}" for m, t in COMBOS]

# The DeepSeek end-of-sentence token, written with escapes so this file stays ASCII.
DEEPSEEK_EOS = "<" + chr(0xFF5C) + "end" + chr(0x2581) + "of" + chr(0x2581) + "sentence" + chr(0xFF5C) + ">"
MISTRAL_PAD_TOKENS = {"<pad>", "</s>"}

TASKLIST = ROOT / "planning" / "TaskList.md"


# ----------------------------------------------------------------- helpers


def _merge(dst: dict, src: dict) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge(dst[key], value)
        else:
            dst[key] = value


def make_cfg(model_id: str, task: str, **overrides: Any) -> ConfigRecord:
    payload = defaults_for(model_id, task)
    _merge(payload, overrides)
    return ConfigRecord(**payload)


def bundle(model_id: str, task: str, *, adapter: str | None = None, **overrides: Any) -> dict[str, Any]:
    cfg = make_cfg(model_id, task, **overrides)
    if adapter is not None:
        # ModelSection forbids an `adapter_path` field, but the generator reads it
        # with getattr(cfg.a, "adapter_path", None); set it past the schema.
        cfg.a.__dict__["adapter_path"] = adapter
    return generate_bundle(cfg)


def parse(code: str) -> ast.Module:
    return ast.parse(code)


def config_node(tree: ast.Module) -> ast.Assign:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "CONFIG" for t in node.targets
        ):
            return node
    raise AssertionError("generated script has no top-level CONFIG dictionary")


def config_of(code: str) -> dict[str, Any]:
    return ast.literal_eval(config_node(parse(code)).value)


def function_names(tree: ast.Module) -> set[str]:
    return {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def calls_to(tree: ast.AST, dotted: str) -> list[ast.Call]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith(dotted):
            found.append(node)
    return found


def string_constants_outside_config(tree: ast.Module) -> list[str]:
    config = config_node(tree)
    skip = {id(n) for n in ast.walk(config)}
    return [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip
    ]


def load_pure_functions(code: str, names: set[str]) -> dict[str, Any]:
    """Execute only the named top-level functions of a generated script."""
    tree = parse(code)
    picked = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in picked} == names, "generated script is missing helper functions"
    module = ast.Module(body=picked, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {"random": random}
    exec(  # noqa: S102 - generated, pure helper functions only
        compile(module, "<generated-helpers>", "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    return namespace


@pytest.fixture(scope="module")
def bundles() -> dict[tuple[str, str], dict[str, Any]]:
    return {(m, t): bundle(m, t) for m, t in COMBOS}


# ======================================================================
# M2.5 - generate a script automatically and validate it section by section
# ======================================================================


@pytest.mark.parametrize(("model_id", "task"), COMBOS, ids=COMBO_IDS)
def test_each_model_task_script_compiles_and_has_zero_parity_diff(model_id, task, bundles):
    # TaskList: "Generate a Python training or inference script automatically from the config"
    # TaskList: "Validate against the reference script ... parameter parity + syntax check"
    b = bundles[(model_id, task)]
    code = b["script_code"]
    assert b["task_type"] == task and b["model_id"] == model_id
    assert b["syntax_ok"] is True and b["syntax_error"] is None
    compile(code, b["script_filename"], "exec")  # independent of the verifier
    assert b["parity_diffs"] == [] and b["parity_ok"] is True
    assert b["pitfall_issues"] == []
    assert b["marker_issues"] == []
    assert b["script_sha256"] == hashlib.sha256(code.encode("utf-8")).hexdigest()
    prefix = "fineTune" if task == "fine_tune" else "inference"
    tag = "Mistral24B" if model_id == MISTRAL_ID else "DeepSeek32B"
    assert re.fullmatch(rf"{prefix}_{tag}_[0-9A-Za-z]{{8}}\.py", b["script_filename"])


@pytest.mark.parametrize(("model_id", "task"), COMBOS, ids=COMBO_IDS)
def test_scripts_are_templates_with_config_literal_and_cli_overrides(model_id, task, bundles):
    # TaskList: "Finalise the generator design ... (templates + parameter injection,
    # not string concatenation)": one injected CONFIG literal plus a CLI override layer.
    code = bundles[(model_id, task)]["script_code"]
    tree = parse(code)
    cfg = config_of(code)
    assert cfg["model_id"] == model_id and cfg["config_id"] and cfg["config_hash"].startswith("sha256:")
    assert "parse_args" in function_names(tree)
    assert calls_to(tree, "argparse.ArgumentParser"), "CLI overrides use argparse"


def test_hyperparameters_from_the_config_are_injected_into_the_scripts():
    # TaskList: "Generated script includes ... injected hyperparameters"
    ft = config_of(
        bundle(
            MISTRAL_ID,
            "fine_tune",
            d={"sft": {"learning_rate": 3.3e-05, "num_train_epochs": 5}, "lora": {"r": 17}},
        )["script_code"]
    )
    assert ft["d"]["sft"]["learning_rate"] == 3.3e-05
    assert ft["d"]["sft"]["num_train_epochs"] == 5
    assert ft["d"]["lora"]["r"] == 17
    inf = config_of(
        bundle(DEEPSEEK_ID, "inference", f={"max_new_tokens": 321, "temperature": 0.0, "do_sample": False})[
            "script_code"
        ]
    )
    assert inf["f"]["max_new_tokens"] == 321 and inf["f"]["do_sample"] is False


@pytest.mark.parametrize("model_id", MODELS, ids=["mistral", "deepseek"])
def test_training_script_has_data_loading_and_lora_sft_setup(model_id, bundles):
    # TaskList: "data loading, LoRA/SFT setup (training)"
    code = bundles[(model_id, "fine_tune")]["script_code"]
    tree = parse(code)
    assert calls_to(tree, "LoraConfig"), "LoRA setup"
    assert calls_to(tree, "SFTConfig"), "SFT setup"
    assert calls_to(tree, "SFTTrainer"), "trainer is built"
    assert calls_to(tree, "trainer.train"), "training is started"
    assert "build_dataset" in function_names(tree), "data loading helper"
    assert len(calls_to(tree, "build_dataset")) >= 2, "train and eval sets are both built"


@pytest.mark.parametrize("model_id", MODELS, ids=["mistral", "deepseek"])
def test_inference_script_has_generation_and_verdict_extraction(model_id, bundles):
    # TaskList: "generation + verdict extraction (inference)"
    code = bundles[(model_id, "inference")]["script_code"]
    tree = parse(code)
    assert calls_to(tree, ".generate"), "generation call"
    assert "extract_verdict" in function_names(tree), "verdict extraction helper"
    assert calls_to(tree, "extract_verdict"), "verdict extraction is used"
    assert any(imp for imp in ast.walk(tree) if isinstance(imp, (ast.Import, ast.ImportFrom)))


def test_demo_checklist_same_config_both_scripts():
    # TaskList: "Nov 28 demo checklist: training script and inference script generated
    # from the same config + syntax check passes + parameter parity diff is empty."
    shared = {"config_id": "demo-cfg-0001-aaaa-bbbb"}
    ft = bundle(MISTRAL_ID, "fine_tune", **shared)
    adapter = config_of(ft["script_code"])["save_model_dir"]
    inf = bundle(MISTRAL_ID, "inference", adapter=adapter, **shared)
    for b in (ft, inf):
        assert b["syntax_ok"] and b["parity_diffs"] == [] and b["pitfall_issues"] == []
    id8 = re.sub(r"[^0-9A-Za-z]", "", shared["config_id"])[:8]
    assert ft["script_filename"].endswith(f"_{id8}.py") and inf["script_filename"].endswith(f"_{id8}.py")
    assert ft["config_id"] == inf["config_id"]
    # the inference script loads the adapter the training script saves, and parses the
    # same verdict prefix the training script teaches
    ft_cfg, inf_cfg = config_of(ft["script_code"]), config_of(inf["script_code"])
    assert inf_cfg["adapter_path"] == adapter == ft_cfg["save_model_dir"]
    assert inf_cfg["verdict_parse_marker"] == ft_cfg["p"]["assistant_prompt"]
    assert "peft" in inf["requirements_txt"], "an adapter needs peft at inference time"


# ======================================================================
# M2.5 - all 7 evidence_config modes
# ======================================================================

EVIDENCE_MODES = (
    "with_evidence",
    "without_evidence",
    "random_evidence",
    "question_only",
    "answer_only",
    "all_questions_last_answer",
    "all_questions_all_but_last_answer",
)
PLACEHOLDER = "No answer could be found !"
RECORD = {
    "claim": "A claim.",
    "evidences": [
        {"questions": "Q1?", "top_k_doc": []},  # empty on purpose (pitfall 1)
        {"questions": "Q2?", "top_k_doc": ["A2a", "A2b"]},
        {"questions": "Q3?", "top_k_doc": ["A3a"]},
    ],
}
DONOR = [
    {"questions": "D1?", "top_k_doc": ["DA1"]},
    {"questions": "D2?", "top_k_doc": ["DA2"]},
]


def pair(q, a):
    return {"question": q, "answer": a}


# Written from the mode names (TaskList section 1.1), one entry per mode.
EXPECTED_ITEMS = {
    "with_evidence": [pair("Q1?", PLACEHOLDER), pair("Q2?", "A2a"), pair("Q3?", "A3a")],
    "without_evidence": [],
    "question_only": [pair("Q1?", None), pair("Q2?", None), pair("Q3?", None)],
    "answer_only": [pair(None, PLACEHOLDER), pair(None, "A2a"), pair(None, "A3a")],
    "all_questions_last_answer": [pair("Q1?", None), pair("Q2?", None), pair("Q3?", "A3a")],
    "all_questions_all_but_last_answer": [pair("Q1?", PLACEHOLDER), pair("Q2?", "A2a"), pair("Q3?", None)],
}
EXPECTED_TEXT = {
    "with_evidence": "### Question: Q1?\n### Answer: " + PLACEHOLDER
    + "\n### Question: Q2?\n### Answer: A2a\n### Question: Q3?\n### Answer: A3a",
    "without_evidence": "",
    "question_only": "### Question: Q1?\n### Question: Q2?\n### Question: Q3?",
    "answer_only": "### Answer: " + PLACEHOLDER + "\n### Answer: A2a\n### Answer: A3a",
    "all_questions_last_answer": "### Question: Q1?\n### Question: Q2?\n### Question: Q3?\n### Answer: A3a",
    "all_questions_all_but_last_answer": "### Question: Q1?\n### Answer: " + PLACEHOLDER
    + "\n### Question: Q2?\n### Answer: A2a\n### Question: Q3?",
}


@pytest.mark.parametrize("mode", EVIDENCE_MODES)
@pytest.mark.parametrize("task", TASKS)
def test_evidence_mode_generates_a_valid_script(mode, task):
    b = bundle(MISTRAL_ID, task, c={"evidence_config": mode})
    assert b["syntax_ok"] and b["parity_diffs"] == [] and b["pitfall_issues"] == []
    assert config_of(b["script_code"])["c"]["evidence_config"] == mode


@pytest.mark.parametrize("model_id", MODELS, ids=["mistral", "deepseek"])
@pytest.mark.parametrize("mode", [m for m in EVIDENCE_MODES if m != "random_evidence"])
def test_evidence_mode_selects_and_formats_the_right_items(mode, model_id, bundles):
    ns = load_pure_functions(
        bundles[(model_id, "inference")]["script_code"],
        {"_resolve_answer", "select_evidence_items", "format_evidence_items"},
    )
    items = ns["select_evidence_items"](RECORD, mode, 1, PLACEHOLDER, random.Random(1))
    assert items == EXPECTED_ITEMS[mode]
    assert ns["format_evidence_items"](items) == EXPECTED_TEXT[mode]


def test_evidence_mode_random_uses_the_donor_evidence_and_is_seeded(bundles):
    ns = load_pure_functions(
        bundles[(MISTRAL_ID, "inference")]["script_code"],
        {"_resolve_answer", "select_evidence_items", "format_evidence_items"},
    )
    pick = ns["select_evidence_items"]
    one = pick(RECORD, "random_evidence", 1, PLACEHOLDER, random.Random(7), donor_evidences=DONOR)
    again = pick(RECORD, "random_evidence", 1, PLACEHOLDER, random.Random(7), donor_evidences=DONOR)
    assert one == again, "the same seed gives the same random evidence"
    key = lambda item: item["question"]  # noqa: E731
    assert sorted(one, key=key) == [pair("D1?", "DA1"), pair("D2?", "DA2")]
    assert not any(item["question"] in {"Q1?", "Q2?", "Q3?"} for item in one)


def test_evidence_answers_used_joins_the_first_n_documents(bundles):
    ns = load_pure_functions(
        bundles[(MISTRAL_ID, "inference")]["script_code"],
        {"_resolve_answer", "select_evidence_items", "format_evidence_items"},
    )
    items = ns["select_evidence_items"](RECORD, "with_evidence", 2, PLACEHOLDER, random.Random(1))
    assert items[1] == pair("Q2?", "A2a\nA2b") and items[2] == pair("Q3?", "A3a")


# ======================================================================
# M2.5 - the six known pitfalls (one test each, plus planted-bug checks)
# ======================================================================


@pytest.mark.parametrize(("model_id", "task"), COMBOS, ids=COMBO_IDS)
def test_pitfall_1_empty_top_k_doc_keeps_the_question(model_id, task, bundles):
    # `"...Answer: " + doc[0] if doc else ...` dropped the Question when top_k_doc was empty.
    tree = parse(bundles[(model_id, task)]["script_code"])
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            assert "### Question:" not in ast.unparse(node.body), "unparenthesised ternary around the Question"
    # runtime check: see test_evidence_mode_selects_and_formats_the_right_items (Q1 has no documents)


@pytest.mark.parametrize(("model_id", "task"), COMBOS, ids=COMBO_IDS)
def test_pitfall_2_pad_token_is_a_real_token(model_id, task, bundles):
    code = bundles[(model_id, task)]["script_code"]
    assert "<myPadToken>" not in code
    valid = MISTRAL_PAD_TOKENS if model_id == MISTRAL_ID else {DEEPSEEK_EOS}
    assert config_of(code)["b"]["pad_token"] in valid
    assigned = [
        n.value
        for n in ast.walk(parse(code))
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "tokenizer.pad_token" for t in n.targets)
        and isinstance(n.value, ast.Constant)
    ]
    assert all(a.value in valid for a in assigned)


@pytest.mark.parametrize(("model_id", "task"), COMBOS, ids=COMBO_IDS)
def test_pitfall_3_fix_mistral_regex_is_passed_to_from_pretrained(model_id, task, bundles):
    code = bundles[(model_id, task)]["script_code"]
    tree = parse(code)
    assert not re.search(r"\btokenizer\.fix_mistral_regex\s*=", code), "assigned after loading has no effect"
    keywords = {k.arg for call in calls_to(tree, "AutoTokenizer.from_pretrained") for k in call.keywords}
    if model_id == MISTRAL_ID:
        assert "fix_mistral_regex" in keywords
    else:
        assert "fix_mistral_regex" not in keywords


def test_pitfall_4_train_and_eval_use_the_same_assistant_prefix(bundles):
    ft_code = bundles[(MISTRAL_ID, "fine_tune")]["script_code"]
    inf_code = bundles[(MISTRAL_ID, "inference")]["script_code"]
    prefix = config_of(ft_code)["p"]["assistant_prompt"]
    assert prefix.endswith("Verdict: ")
    # one shared builder produces both datasets, so the prefixes cannot drift apart
    tree = parse(ft_code)
    assert {"build_messages", "build_dataset"} <= function_names(tree)
    assert len(calls_to(tree, "build_dataset")) >= 2
    assert "### Response:" not in set(string_constants_outside_config(tree)), "a second, different prefix"
    # and inference parses the same prefix the training taught
    assert config_of(inf_code)["verdict_parse_marker"] == prefix
    assert "NOTE TO SELF" not in ft_code + inf_code


@pytest.mark.parametrize(("model_id", "task"), COMBOS, ids=COMBO_IDS)
def test_pitfall_5_paths_come_from_the_config(model_id, task):
    unique = {
        "fine_tune": {"c": {"train_data_file": "data/uniqueTrain123.json", "eval_data_file": "data/uniqueEval123.json"},
                      "d": {"sft": {"output_dir": "uniqueOut123"}}},
        "inference": {"c": {"test_data_file": "data/uniqueTest123.json"}},
    }[task]
    b = bundle(model_id, task, **unique)
    tree = parse(b["script_code"])
    in_config = ast.dump(config_node(tree))
    outside = string_constants_outside_config(tree)
    for value in ("uniqueTrain123", "uniqueEval123", "uniqueOut123", "uniqueTest123"):
        if value in str(unique):
            assert value in in_config, f"{value} never reaches the CONFIG literal"
            assert not any(value in s for s in outside), f"{value} is hard-coded outside CONFIG"
    assert not any(re.match(r"^(/|[A-Za-z]:\\)|/home/|/project/|/scratch", s) for s in outside)
    for call in calls_to(tree, "open"):
        if call.args:
            assert not isinstance(call.args[0], ast.Constant), "open() with a literal path"
    first = calls_to(tree, "AutoModelForCausalLM.from_pretrained")[0].args[0]
    assert not isinstance(first, ast.Constant), "the model id is not hard-coded"
    assert b["pitfall_issues"] == []


@pytest.mark.parametrize(("model_id", "task"), COMBOS, ids=COMBO_IDS)
def test_pitfall_6_seed_is_part_of_the_config_and_is_used(model_id, task):
    b = bundle(model_id, task, d={"sft": {"seed": 1234}})
    code = b["script_code"]
    tree = parse(code)
    cfg = config_of(code)
    assert (cfg["d"]["sft"]["seed"] if task == "fine_tune" else cfg["seed"]) == 1234
    assert "set_reproducibility_seed" in function_names(tree)
    assert calls_to(tree, "set_reproducibility_seed"), "the seed helper is called"
    if task == "fine_tune":
        assert "seed" in {k.arg for call in calls_to(tree, "SFTConfig") for k in call.keywords}
    assert "rank=0" not in code, "no rank=0 in distributed setup"


# Planted-bug checks: the verifier must catch each defect when it is put back.
PLANTED = {
    "pitfall 1 (ternary drops the Question)": ('\n_q = "### Question: x" if True else ""\n', "Defect 1"),
    "pitfall 2 (invalid pad token)": ('\n_pad = "<myPadToken>"\n', "Defect 2"),
    "pitfall 3 (regex assigned after loading)": ("\ntokenizer.fix_mistral_regex = True\n", "Defect 3"),
    "pitfall 4 (leftover note)": ("\n# NOTE TO SELF: fix the prefix\n", "Defect 4"),
    "pitfall 5 (hard-coded temperature)": ("\nmodel.generation_config.temperature = 0.1\n", "Defect 5"),
    "rank=0 in distributed init": ("\n_x = 'rank=0'\n", "rank=0"),
}


@pytest.mark.parametrize(("model_id", "task"), COMBOS, ids=COMBO_IDS)
@pytest.mark.parametrize("name", sorted(PLANTED))
def test_planted_pitfall_is_caught_by_the_verifier(name, model_id, task):
    cfg = make_cfg(model_id, task)
    code = generate_bundle(cfg)["script_code"]
    suffix, needle = PLANTED[name]
    issues = check_known_pitfalls(cfg, code + suffix)
    assert any(needle in issue for issue in issues), (name, issues)


@pytest.mark.parametrize(("model_id", "task"), COMBOS, ids=COMBO_IDS)
def test_planted_pitfall_6_missing_seed_helper_is_caught(model_id, task):
    cfg = make_cfg(model_id, task)
    code = generate_bundle(cfg)["script_code"]
    broken = code.replace("set_reproducibility_seed", "renamed_helper")
    assert broken != code
    assert any("Defect 6" in i for i in check_known_pitfalls(cfg, broken))


# ======================================================================
# M2.5 - unknown **kwargs are rejected at generation time
# ======================================================================

BOGUS = "definitely_not_a_real_kwarg"


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("stage", sorted(KWARGS_WHITELISTS))
def test_unknown_kwargs_are_rejected_not_ignored(stage, task):
    # TaskList 1.1 h: "unknown keys must fail at generation time rather than being silently ignored"
    cfg = make_cfg(MISTRAL_ID, task, h={stage: {BOGUS: 1}})
    with pytest.raises(ConfigValidationError) as excinfo:
        generate_bundle(cfg)
    assert any(i["field"] == f"h.{stage}.{BOGUS}" and i["level"] == "error" for i in excinfo.value.issues)
    assert BOGUS in str(excinfo.value)


def test_unknown_kwargs_stage_name_is_rejected_too():
    with pytest.raises(ConfigValidationError):
        generate_bundle(make_cfg(MISTRAL_ID, "inference", h={"made_up_stage": {"x": 1}}))


def test_known_kwargs_pass_through_into_the_generated_script():
    b = bundle(MISTRAL_ID, "inference", h={"load_tokenizer": {"use_fast": False}})
    assert b["syntax_ok"] and b["parity_diffs"] == []
    assert config_of(b["script_code"])["h"]["load_tokenizer"] == {"use_fast": False}


# ======================================================================
# M2.5 - requirements.txt
# ======================================================================


def packages(text: str) -> set[str]:
    names = set()
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.add(re.split(r"[<>=!~ ]", line, maxsplit=1)[0].lower())
    return names


@pytest.mark.parametrize(("model_id", "task"), COMBOS, ids=COMBO_IDS)
def test_requirements_txt_is_generated_with_the_core_libraries(model_id, task, bundles):
    # TaskList: "Additionally generate requirements.txt"
    text = bundles[(model_id, task)]["requirements_txt"]
    assert text.endswith("\n")
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+(>=|==)[0-9][0-9.]*", ln) for ln in text.splitlines() if ln and not ln.startswith("#"))
    assert {"torch", "transformers", "accelerate"} <= packages(text)
    assert ("trl" in packages(text) and "peft" in packages(text)) == (task == "fine_tune")


@pytest.mark.parametrize("task", TASKS)
def test_requirements_mistral_common_only_for_mistral(task):
    assert "mistral-common" in packages(bundle(MISTRAL_ID, task)["requirements_txt"])
    assert "mistral-common" not in packages(bundle(DEEPSEEK_ID, task)["requirements_txt"])


@pytest.mark.parametrize("model_id", MODELS, ids=["mistral", "deepseek"])
def test_requirements_bitsandbytes_only_when_quantised_or_8bit_optimiser(model_id):
    assert "bitsandbytes" not in packages(bundle(model_id, "inference")["requirements_txt"])
    plain = bundle(model_id, "fine_tune", d={"sft": {"optim": "adamw_torch"}})["requirements_txt"]
    assert "bitsandbytes" not in packages(plain)
    # the registry default optimiser is an 8-bit one, so the default fine-tune needs it
    assert "bitsandbytes" in packages(bundle(model_id, "fine_tune")["requirements_txt"])
    for quant in ("4bit", "8bit"):
        text = bundle(model_id, "inference", a={"quantization": quant})["requirements_txt"]
        assert "bitsandbytes" in packages(text), quant
    optim = bundle(model_id, "fine_tune", d={"sft": {"optim": "paged_adamw_8bit"}})["requirements_txt"]
    assert "bitsandbytes" in packages(optim)


def test_requirements_peft_at_inference_only_with_an_adapter():
    assert "peft" not in packages(bundle(MISTRAL_ID, "inference")["requirements_txt"])
    with_adapter = bundle(MISTRAL_ID, "inference", adapter="fineTuningOutput/final_model")
    assert "peft" in packages(with_adapter["requirements_txt"])


# ======================================================================
# Coverage: every M2.5 TaskList item is still in the list and has a test
# ======================================================================

M2_5_ITEMS = {
    "Finalise the generator design": "test_scripts_are_templates_with_config_literal_and_cli_overrides",
    "Generate a Python training or inference script": "test_each_model_task_script_compiles_and_has_zero_parity_diff",
    "Generated script includes": "test_training_script_has_data_loading_and_lora_sft_setup",
    "Additionally generate `requirements.txt`": "test_requirements_txt_is_generated_with_the_core_libraries",
    "Validate against the reference script section by section": "test_each_model_task_script_compiles_and_has_zero_parity_diff",
    "Operator-precedence bug": "test_pitfall_1_empty_top_k_doc_keeps_the_question",
    "is not a real token": "test_pitfall_2_pad_token_is_a_real_token",
    "fix_mistral_regex": "test_pitfall_3_fix_mistral_regex_is_passed_to_from_pretrained",
    "Assistant prefix differs": "test_pitfall_4_train_and_eval_use_the_same_assistant_prefix",
    "All paths injected from config": "test_pitfall_5_paths_come_from_the_config",
    "`seed` must be part of the config": "test_pitfall_6_seed_is_part_of_the_config_and_is_used",
    "Fix the known pitfalls": "test_pitfall_6_seed_is_part_of_the_config_and_is_used",
    "Nov 28 demo checklist": "test_demo_checklist_same_config_both_scripts",
}


def m2_5_section() -> str:
    text = TASKLIST.read_text(encoding="utf-8")
    start = text.index("## M2.5")
    end = text.index("## M3.0")
    return text[start:end]


@pytest.mark.parametrize("snippet", sorted(M2_5_ITEMS))
def test_every_m2_5_tasklist_item_has_a_matching_test(snippet):
    assert snippet in m2_5_section(), f"TaskList wording changed: {snippet!r}; update this checklist"
    assert M2_5_ITEMS[snippet] in globals(), f"missing test {M2_5_ITEMS[snippet]}"


def test_no_new_m2_5_checkbox_item_without_a_test():
    items = [ln for ln in m2_5_section().splitlines() if ln.lstrip().startswith("- [ ]") or ln.lstrip().startswith("- [x]")]
    top_level = [ln for ln in items if ln.startswith("- [")]
    covered = [ln for ln in top_level if any(snippet.strip("`") in ln.replace("`", "") or snippet in ln for snippet in M2_5_ITEMS)]
    assert len(covered) == len(top_level), "a new M2.5 TaskList item needs an assertion here"


def test_this_module_has_no_cjk_characters():
    text = Path(__file__).read_text(encoding="utf-8")
    assert not re.search("[" + chr(0x3040) + "-" + chr(0x30FF) + chr(0x3400) + "-" + chr(0x4DBF) + chr(0x4E00) + "-" + chr(0x9FFF) + chr(0xAC00) + "-" + chr(0xD7AF) + "]", text)
