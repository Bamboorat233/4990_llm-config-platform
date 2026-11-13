"""Tests for the fine-tune progress collector (issue #31).

Expected numbers are read off the real logs (values worked out by hand and
cross-checked with tools/progress_oracle.py, which shares no code with the
parser). The slow test compares the parser with the oracle on the full logs
when LOGS_DIR is set.
"""
import importlib.util
import os
from pathlib import Path

import pytest

from app.parsing import parse_log_file
from app.parsing.training_log import TrainingProgress

FIX = Path(__file__).parent / "fixtures" / "logs"
ROOT = Path(__file__).parent.parent

NO_PROGRESS = {"step": None, "total_steps": None, "epoch": None, "last_loss": None, "eval_losses": []}

# fixture -> expected progress (identical on the excerpt and the full log)
EXPECTED = {
    "s08_fine_tune_error.log": {**NO_PROGRESS, "step": 0, "total_steps": 9936},
    "s09_fine_tune_error.log": NO_PROGRESS,
    "s10_fine_tune_error.log": {"step": 62, "total_steps": 9936, "epoch": 0.01248,
                                "last_loss": 0.6572, "eval_losses": []},
    "s11_fine_tune_error.log": {**NO_PROGRESS, "step": 0, "total_steps": 9936},
    "s12_fine_tune_success_warning.log": {"step": 9936, "total_steps": 9936, "epoch": 2.0,
                                          "last_loss": 0.491, "eval_losses": [0.6387, 0.6405]},
    "synthetic_fine_tune_terminated.log": {"step": 22, "total_steps": 9936, "epoch": 0.004428,
                                           "last_loss": 0.7695, "eval_losses": []},
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fixture_progress(name):
    assert parse_log_file(FIX / name)["progress"] == EXPECTED[name]


def test_other_families_keep_their_own_keys():
    assert parse_log_file(FIX / "s13_checkpoint_export_success.log")["progress"] == {}
    inf = parse_log_file(FIX / "s02_inference_success.log")["progress"]
    assert set(inf) == {"records_done", "records_expected"}


def feed(*lines):
    t = TrainingProgress()
    for n, line in enumerate(lines, 1):
        t.feed(line, n)
    return t.result()


def test_loss_values_are_floats_and_last_one_wins():
    r = feed("{'loss': '2.013', 'epoch': '0.5'}", "{'loss': '1.5', 'epoch': '1.25'}")
    assert r["last_loss"] == 1.5 and r["epoch"] == 1.25


def test_eval_losses_keep_order_and_do_not_touch_last_loss():
    r = feed("{'loss': '0.9', 'epoch': '1'}",
             "\x1b[A{'eval_loss': '0.6387', 'epoch': '1'}",
             "{'eval_loss': '0.6405', 'epoch': '2'}")
    assert r["eval_losses"] == [0.6387, 0.6405] and r["last_loss"] == 0.9


def test_small_evaluation_bar_is_not_the_training_bar():
    r = feed(" 50%|#####     | 4969/9936 [03:00<03:00,  1.0s/it]",
             " 90%|#########| 348/386 [00:40<00:04,  9.0it/s]",
             " 51%|#####     | 4970/9936 [03:01<03:00,  1.0s/it]")
    assert (r["step"], r["total_steps"]) == (4970, 9936)


def test_labelled_bars_are_ignored():
    r = feed("Loading weights: 100%|##########| 363/363 [00:03<00:00, 100it/s]",
             "Writing model shards: 100%|##########| 1/1 [00:01<00:00,  1.2s/it]")
    assert (r["step"], r["total_steps"]) == (None, None)


def test_bigger_total_replaces_a_smaller_one_seen_first():
    r = feed(" 100%|##########| 386/386 [00:40<00:00,  9.0it/s]",
             "  0%|          | 0/9936 [00:00<?, ?it/s]")
    assert (r["step"], r["total_steps"]) == (0, 9936)


def test_bar_followed_by_other_text_on_the_same_line():
    r = feed("  1%|          | 62/9936 [02:44<7:17:59,  2.66s/it][rank0]: Traceback (most recent call last):")
    assert (r["step"], r["total_steps"]) == (62, 9936)


@pytest.mark.parametrize("bad", ["{'loss': 'nan-ish', 'epoch': '1'}", "{'loss': '', 'epoch': '1'}",
                                 "{'eval_loss': 'oops'}", "  1%|  | 5/0 [00:00]", "  1%|  | 12/9 [00:00]",
                                 "{'loss': '", ""])
def test_malformed_lines_are_ignored_not_raised(bad):
    assert feed(bad)["last_loss"] is None and feed(bad)["eval_losses"] == [] and feed(bad)["step"] is None


def test_state_stays_bounded_over_many_lines():
    t = TrainingProgress()
    for i in range(100_000):
        t.feed("{'loss': '0.5', 'grad_norm': '0.1', 'epoch': '1'}", i)
        t.feed(f"  1%|  | {i % 9936}/9936 [00:01<00:01, 1.0s/it]", i)
    assert t.eval_losses == [] and len(vars(t)) == 5


@pytest.mark.skipif(not os.environ.get("LOGS_DIR"), reason="set LOGS_DIR to the folder with the full sample logs")
def test_parser_matches_independent_oracle_on_full_logs():
    spec = importlib.util.spec_from_file_location("progress_oracle", ROOT / "tools" / "progress_oracle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for n in (8, 9, 10, 11, 12):
        full = Path(os.environ["LOGS_DIR"]) / f"sampleOutput{n:02d}.txt"
        assert parse_log_file(full)["progress"] == mod.oracle(str(full)), full.name
