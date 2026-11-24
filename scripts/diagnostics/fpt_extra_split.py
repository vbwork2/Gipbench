"""Split the extra FPT pool into a new heldout and a diagnostic part, once, before any diagnosis on it.

`TODO_rea.md` item 1.2. The old heldout was used once (hotwords) and can no longer test anything new.
This script seals a new one out of `manifests_extra/extra_scored.jsonl` (written by `fpt_extra_pool.py`).
Run it once, commit the lock it writes under `reports/`, and only then run items 1.3 and 1.4. It refuses
to overwrite its outputs.

Registered rules, fixed before the run:
* text key: `" ".join(tokens(reference, case))` with the vocabulary case of `configs/model_map.json`,
  the same key `fpt_extra_pool.py` used to exclude heldout texts;
* rows whose text key equals a text key of the sealed development pool (`full_scored.jsonl`) are FORCED
  to the diagnostic part. The development pool is already diagnostic, so these rows only have to stay
  out of the new heldout. They are not dropped;
* the unit of assignment is the text group, all rows with one text key, so one text never sits on both
  sides. Rows that repeat a text are not removed;
* stratum of a group: (Set of the group's row with the lowest `audio_idx`, taken from the audio file name
  `..._SetNNN_...`; whether any row of the group has `cs_tokens`). Strata use the reference label only,
  never the baseline errors. 53 groups have rows in both Sets; they follow the lowest-index row;
* inside each stratum, groups are sorted by `sha256(SALT + "\\n" + text_key)` and taken in that order
  until the stratum's heldout rows reach at least HELDOUT_FRACTION of its rows. The realized fractions
  are reported as they come; the salt is fixed here and the split is not redrawn;
* no row from the old heldout can occur: `fpt_extra_pool.py` already dropped heldout audio and texts.

Registered uses of the new heldout, each tested once and only after its rule is registered:
* the best variant of the top-10 gated blank penalty (item 2.1), if one qualifies on the diagnostic part;
* the best learned Joiner trigger (item 2.2), if one qualifies on the diagnostic part;
* the reviewed-name preset (ANDY MURRAY, CHAPLIN, CROWELL, SMITH);
* optionally a replicate of the X2 prediction P1/P2 of `fpt_span_lock.py`, under its own registration.
Items 1.3 and 1.4 and every design or selection step read `extra_diagnostic.jsonl` only. The heldout's
stored baseline hypotheses are not read except by those registered tests; this script reads them only to
count CS error spans per side.

Limits: the pool has no speaker labels, so a speaker can occur on both sides. Set is only a proxy for a
recording batch.
"""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gipformer_mech.text_norm import tokens

SALT = "fpt-extra-heldout-v1"
HELDOUT_FRACTION = 0.30
SET = re.compile(r"_(Set\d+)_")


def file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_of(row: dict) -> str:
    match = SET.search(row["audio_file"])
    if match is None:
        raise ValueError(f"No Set in audio file name: {row['audio_file']}")
    return match.group(1)


def text_key(row: dict, case: str) -> str:
    return " ".join(tokens(row["reference"], case))


def cs_spans(row: dict) -> int:
    return len(row["score_counts"]["cs_word_errors"])


def split(rows: list[dict], development_texts: set[str], case: str, salt: str = SALT,
          fraction: float = HELDOUT_FRACTION) -> tuple[list[dict], list[dict], dict]:
    """Return (heldout, diagnostic, report). Both keep the input row order."""
    keys = [text_key(row, case) for row in rows]
    groups: dict[str, list[dict]] = defaultdict(list)
    for key, row in zip(keys, rows):
        groups[key].append(row)
    strata: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for key, members in groups.items():
        if key in development_texts:
            continue
        first = min(members, key=lambda row: row["audio_idx"])
        strata[(set_of(first), any(row["cs_tokens"] for row in members))].append(key)
    heldout_keys: set[str] = set()
    report = {}
    for (set_name, has_cs), members in sorted(strata.items()):
        members.sort(key=lambda key: sha256(f"{salt}\n{key}".encode("utf-8")).hexdigest())
        total = sum(len(groups[key]) for key in members)
        taken = 0
        for key in members:
            if taken >= fraction * total:
                break
            heldout_keys.add(key)
            taken += len(groups[key])
        report[f"{set_name}|{'cs' if has_cs else 'no_cs'}"] = {
            "groups": len(members), "rows": total, "heldout_rows": taken,
            "heldout_fraction": taken / total if total else 0.0,
        }
    heldout = [row for key, row in zip(keys, rows) if key in heldout_keys]
    diagnostic = [row for key, row in zip(keys, rows) if key not in heldout_keys]
    forced = [row for key, row in zip(keys, rows) if key in development_texts]
    report = {"strata": report, "forced_development_text_rows": len(forced),
              "forced_development_text_cs_spans": sum(map(cs_spans, forced))}
    return heldout, diagnostic, report


def side(rows: list[dict]) -> dict:
    sets = defaultdict(int)
    for row in rows:
        sets[set_of(row)] += 1
    return {"rows": len(rows), "sets": dict(sorted(sets.items())),
            "rows_with_cs_tokens": sum(bool(row["cs_tokens"]) for row in rows),
            "rows_with_cs_errors": sum(bool(cs_spans(row)) for row in rows),
            "cs_error_spans": sum(map(cs_spans, rows))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--extra-dir", type=Path, default=ROOT / "data/manifests_extra")
    parser.add_argument("--lock", type=Path, default=ROOT / "reports/phase1_extra_split_lock.json")
    args = parser.parse_args()

    outputs = [args.extra_dir / name for name in ("extra_heldout.jsonl", "extra_diagnostic.jsonl",
                                                    "extra_split_lock.json")] + [args.lock]
    for path in outputs:
        if path.exists():
            parser.error(f"{path} exists. The split is written once.")

    case = json.loads((ROOT / "configs/model_map.json").read_text(encoding="utf-8"))["vocabulary_case"]
    old_lock = json.loads((args.manifests / "heldout_lock.json").read_text(encoding="utf-8"))
    pool_path = args.manifests / "full_scored.jsonl"
    if file_sha(pool_path) != old_lock["development_pool_sha256"]:
        raise ValueError("The development pool differs from the sealed copy.")
    summary = json.loads((args.extra_dir / "extra_summary.json").read_text(encoding="utf-8"))
    extra_path = args.extra_dir / "extra_scored.jsonl"
    if file_sha(extra_path) != summary["extra_scored_sha256"]:
        raise ValueError("extra_scored.jsonl differs from extra_summary.json.")

    development_texts = {text_key(row, case) for row in read_rows(pool_path)}
    rows = read_rows(extra_path)
    heldout, diagnostic, report = split(rows, development_texts, case)

    write_rows(outputs[0], heldout)
    write_rows(outputs[1], diagnostic)
    lock = {
        "rules": {"salt": SALT, "heldout_fraction": HELDOUT_FRACTION, "case": case,
                  "unit": "normalized reference text group",
                  "stratum": "Set of lowest audio_idx row x any cs_tokens",
                  "development_text_rows": "forced to diagnostic"},
        "input_sha256": {"extra_scored": summary["extra_scored_sha256"],
                         "development_pool": old_lock["development_pool_sha256"]},
        "heldout": {**side(heldout), "sha256": file_sha(outputs[0])},
        "diagnostic": {**side(diagnostic), "sha256": file_sha(outputs[1])},
        **report,
    }
    text = json.dumps(lock, indent=2) + "\n"
    outputs[2].write_text(text, encoding="utf-8")
    args.lock.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
