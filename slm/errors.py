"""Where models disagree with the labels, from evaluate.py's .rows.json files.

    python slm/errors.py runs/colab/results/fg270m/valid.rows.json runs/colab/results/q17/valid.rows.json
"""

import json
import sys
from collections import Counter


def main():
    for path in sys.argv[1:]:
        details = next(
            value for key, value in json.load(open(path)).items() if key.endswith("_all")
        )
        rows = details["commands"]
        wrong = [row for row in rows if not row["raw"]["correct"]]
        print(f"\n{path}: {len(wrong)}/{len(rows)} commands wrong (raw)")
        for stratum, count in Counter(row["stratum"] for row in wrong).most_common(12):
            print(f"  {count:4} {stratum}")
        invalid = Counter((row["raw"]["error"] or "")[:70] for row in wrong if row["raw"]["error"])
        for error, count in invalid.most_common(5):
            print(f"  invalid {count:3}: {error}")
        sessions = details["sessions_raw"]
        firsts = {}
        for row in sessions:
            if not row["correct"] and row["state_matched_before"]:
                firsts.setdefault(row["session_id"], row)
        print(f"  first session errors ({len(firsts)} sessions):")
        for row in firsts.values():
            print(f"    turn {row['turn']}: {row['command'][:90]!r}")
            print(f"      expected {json.dumps(row['expected'])[:160]}")
            print(f"      got      {row['raw_output'][:160]}")


if __name__ == "__main__":
    main()
