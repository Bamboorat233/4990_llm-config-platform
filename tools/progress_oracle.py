"""Independent oracle for fine-tune progress (issue #31).

Deliberately NOT sharing code with app/parsing/training_log.py: it reads the
whole file at once, evaluates the loss dictionaries with ast.literal_eval and
finds the training bar with a two-pass rule. Usage: progress_oracle.py FILE...
"""
import ast
import json
import re
import sys


def oracle(path):
    text = open(path, encoding="utf-8", errors="replace", newline=None).read()
    lines = text.split("\n")
    losses, evals, bars = [], [], []
    for ln in lines:
        s = ln.lstrip("\x1b[A")
        i = s.find("{'loss'")
        j = s.find("{'eval_loss'")
        for start, bucket in ((i, losses), (j, evals)):
            if start >= 0:
                end = s.find("}", start)
                if end > 0:
                    try:
                        bucket.append(ast.literal_eval(s[start:end + 1]))
                    except (ValueError, SyntaxError):
                        pass
        m = re.search(r"\|\s*(\d+)/(\d+) \[", ln)
        if m and not re.search(r"[A-Za-z]", ln.split("|")[0]):
            bars.append((int(m.group(1)), int(m.group(2))))
    biggest = max((t for _, t in bars), default=None)
    done = [d for d, t in bars if t == biggest]
    return {
        "step": done[-1] if done else None,
        "total_steps": biggest,
        "epoch": float(losses[-1]["epoch"]) if losses else None,
        "last_loss": float(losses[-1]["loss"]) if losses else None,
        "eval_losses": [float(e["eval_loss"]) for e in evals],
    }


if __name__ == "__main__":
    print(json.dumps({p.split("/")[-1]: oracle(p) for p in sys.argv[1:]}, indent=1))
