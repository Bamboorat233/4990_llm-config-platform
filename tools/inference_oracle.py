"""Independent oracle for inference progress (issue #31).

Shares no code with app/parsing/inference_log.py. It reads the whole file and
splits it into record segments at each INPUTS banner, instead of streaming.
Usage: inference_oracle.py FILE...
"""
import json
import re
import sys

BANNER = re.compile(r"^[*#]{5,}\s*(INPUTS|OUTPUT|VERDICTS|LENGTH)\s*[*#]{5,}$", re.M)


def oracle(path):
    text = open(path, encoding="utf-8", errors="replace", newline=None).read()
    marks = [(m.start(), m.group(1)) for m in BANNER.finditer(text)]
    tokenizing = len(re.findall(r"^Tokenizing Inputs\s+:", text, re.M))
    decoded = len(re.findall(r"^Done Decoding\s+:", text, re.M))
    if marks:
        ends = [p for p, _ in marks[1:]] + [len(text)]
        done = 0
        for (start, kind), end in zip(marks, ends):
            if kind != "OUTPUT":
                continue
            # the OUTPUT block runs to the next INPUTS banner, whatever lies between
            nxt = [p for p, k in marks if p > start and k == "INPUTS"]
            block = text[start:(nxt[0] if nxt else len(text))]
            closed = bool(nxt)  # the script moved on to a later record
            if closed or re.search(r"^### Response:", block, re.M):
                done += 1
        return {"records_done": done, "records_expected": None}
    if tokenizing:
        return {"records_done": decoded, "records_expected": None}
    return {"records_done": None, "records_expected": None}


if __name__ == "__main__":
    print(json.dumps({p.split("/")[-1]: oracle(p) for p in sys.argv[1:]}, indent=1))
