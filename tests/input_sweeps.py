"""Generated inputs: user_input dictionaries built at run time.

Instead of writing each input by hand, these functions build many
``user_input`` dictionaries from lists of values (every model, every run
mode, a range of temperatures, and so on). Each generated case gets an
expectation worked out here from the rules in the spec - a small
independent oracle, not a call to the validator - so a sweep can fail if
the validator is wrong.

Limits of the oracles (by design):
  * Length: only Mistral is checked across the whole range. DeepSeek is
    checked only well below 16384 and well above 32768, because the spec
    gives its limit as "32768 / 16384" and the in-between band is not
    settled. The length sweep checks the status only, not which field is red.
  * The other sweeps also check the red / warning field, because the
    contract (specs/interfaces.md) names it.
"""
from itertools import product

MODELS = ("mistral", "deepseek")
RUN_MODES = ("fine_tune", "inference")
DS_EOS = "<｜end▁of▁sentence｜>"
SEQUENCE_LIMIT = 32768  # spec rule 1


def _case(name, user_input, status, red=None, warnings=None):
    expect = {"status": status}
    if red is not None:
        expect["red"] = list(red)
    if warnings is not None:
        expect["warnings"] = list(warnings)
    return {"name": name, "user_input": user_input, "expect": expect}


def temperature_sweep() -> list:
    """Rule 5: temperature 0 <-> do_sample false; temperature > 0 <-> true."""
    cases = []
    for model, mode, temp, sample in product(
            MODELS, RUN_MODES, (0, 0.1, 0.6, 1.0), (True, False)):
        consistent = (temp == 0) != sample  # 0 needs False, >0 needs True
        user_input = {"model": model, "run_mode": mode,
                      "temperature": temp, "do_sample": sample}
        cases.append(_case(
            f"sweep-temp-{model}-{mode}-t{temp}-sample-{sample}", user_input,
            "OK" if consistent else "BLOCKED",
            red=[] if consistent else ["f.do_sample"], warnings=[]))
    return cases


def pad_token_sweep() -> list:
    """Rule 3: each model accepts only its own pad tokens."""
    allowed = {"mistral": {"<pad>", "</s>"}, "deepseek": {DS_EOS}}
    cases = []
    for model, mode, token in product(
            MODELS, RUN_MODES, ("<pad>", "</s>", DS_EOS, "<myPadToken>", "")):
        ok = token in allowed[model]
        cases.append(_case(
            f"sweep-pad-{model}-{mode}-{token or 'empty'}",
            {"model": model, "run_mode": mode, "pad_token": token},
            "OK" if ok else "BLOCKED", red=[] if ok else ["b.pad_token"],
            warnings=[]))
    return cases


def length_sweep() -> list:
    """Rule 1: (max_length or 0) + max_new_tokens must not exceed 32768."""
    cases = []
    for max_length, new in product((None, 4096, 30000, 32768), (1, 512, 2768, 5000)):
        total = (max_length or 0) + new
        cases.append(_case(
            f"sweep-length-mistral-{max_length}-plus-{new}",
            {"model": "mistral", "run_mode": "inference",
             "max_length": max_length, "max_new_tokens": new},
            "OK" if total <= SEQUENCE_LIMIT else "BLOCKED"))
    # DeepSeek: only totals clearly inside 16384 or clearly beyond 32768
    for max_length, new in ((4096, 1024), (10000, 1024), (32000, 2000), (30000, 5000)):
        total = max_length + new
        cases.append(_case(
            f"sweep-length-deepseek-{max_length}-plus-{new}",
            {"model": "deepseek", "run_mode": "inference",
             "max_length": max_length, "max_new_tokens": new},
            "OK" if total <= 16384 else "BLOCKED"))
    return cases


def system_prompt_sweep() -> list:
    """Rules 9 and 10: Mistral needs a system prompt; DeepSeek warns on one."""
    cases = []
    for model, mode, prompt in product(
            MODELS, RUN_MODES, ("", "   ", "You are a fact-checker.")):
        blank = not prompt.strip()
        user_input = {"model": model, "run_mode": mode, "system_prompt": prompt}
        name = f"sweep-prompt-{model}-{mode}-{'blank' if blank else 'text'}-{len(prompt)}"
        if model == "mistral":
            cases.append(_case(name, user_input,
                               "BLOCKED" if blank else "OK",
                               red=["p.system_prompt"] if blank else [], warnings=[]))
        else:
            cases.append(_case(name, user_input, "OK", red=[],
                               warnings=[] if blank else ["p.system_prompt"]))
    return cases


def all_sweeps() -> list:
    return (temperature_sweep() + pad_token_sweep()
            + length_sweep() + system_prompt_sweep())
