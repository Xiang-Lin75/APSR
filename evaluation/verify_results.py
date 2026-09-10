"""Verify published result artifacts using only the Python standard library.

This checks CSV integrity and score aggregation; it does not run the model.
"""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    for name in ("metrics", "trials"):
        digest = hashlib.sha256((directory / (name + ".csv")).read_bytes()).hexdigest()
        require(digest == summary[name + "_csv_sha256"], name + " checksum mismatch")
    with (directory / "metrics.csv").open(encoding="utf-8", newline="") as f:
        metrics = list(csv.DictReader(f))
    with (directory / "trials.csv").open(encoding="utf-8", newline="") as f:
        trials = list(csv.DictReader(f))
    keys = {row["key"] for row in metrics}
    require(len(metrics) == len(trials) == len(keys) == summary["num_queries"] == 6000, "Query count/uniqueness mismatch")
    mapping = {row["key"]: row for row in trials}
    require(set(mapping) == keys and len(mapping) == len(trials), "Trial key mismatch")
    pairs = defaultdict(list)
    numeric_columns = set(metrics[0]) - {"idx", "case_idx", "key"}
    for raw in metrics:
        values = {k: float(raw[k]) for k in numeric_columns}
        require(all(math.isfinite(x) for x in values.values()), "Nonfinite metric")
        require(values["seconds"] > 0, "Invalid duration")
        row = mapping[raw["key"]]
        require((row["idx"], row["case_idx"]) == (raw["idx"], raw["case_idx"]), "Pair index mismatch")
        for name in ("mixture", "target", "enrollment"):
            p = PurePosixPath(row[name + "_relpath"])
            require(not p.is_absolute() and ".." not in p.parts and ":" not in str(p) and "\\" not in str(p), "Nonportable path")
            require(p.suffix == ".wav", "Unexpected recording extension")
        mix = PurePosixPath(row["mixture_relpath"])
        target = PurePosixPath(row["target_relpath"])
        enrollment = PurePosixPath(row["enrollment_relpath"])
        require(mix.parent.as_posix() == "tt/mix" and mix.stem == row["mixture_id"], "Mixture mapping mismatch")
        require(target.parent.as_posix() == "tt/" + row["target_source"] and mix.name == target.name, "Target mapping mismatch")
        require(raw["key"] == mix.stem + "_" + enrollment.stem, "Enrollment mapping mismatch")
        for metric in ("si_sdr", "sdr", "proj_sdr"):
            require(math.isclose(values[metric] - values[metric + "_mix"], values[metric + "_i"], abs_tol=1e-9, rel_tol=0), "Mixture-baseline subtraction mismatch")
        margin = values["si_sdr"] - values["interferer_si_sdr"]
        require(math.isclose(margin, values["target_margin_db"], abs_tol=1e-9, rel_tol=0), "Margin mismatch")
        require(values["target_confusion_rate"] == float(margin <= 0), "Confusion decision mismatch")
        pairs[int(raw["idx"])].append((row, values))
    require(set(pairs) == set(range(summary["num_mixtures"])), "Mixture count/index mismatch")
    mixture_ids = set()
    both_positive = 0
    for pair in pairs.values():
        require(len(pair) == 2 and {r["case_idx"] for r, _ in pair} == {"0", "1"}, "Incomplete query pair")
        require({r["target_source"] for r, _ in pair} == {"s1", "s2"}, "Sources are not paired")
        require(len({r["mixture_relpath"] for r, _ in pair}) == 1, "Different mixtures in pair")
        require(len({v["seconds"] for _, v in pair}) == 1, "Pair duration mismatch")
        mixture_ids.add(pair[0][0]["mixture_id"])
        both_positive += all(v["target_margin_db"] > 0 for _, v in pair)
    require(len(mixture_ids) == summary["num_mixtures"], "Duplicated mixture")
    recomputed = {name: statistics.fmean(float(r[name]) for r in metrics) for name in summary["metrics"]}
    for name, value in recomputed.items():
        require(math.isclose(value, summary["metrics"][name], abs_tol=1e-9, rel_tol=0), "Mean mismatch: " + name)
    confusion_count = sum(float(r["target_confusion_rate"]) == 1 for r in metrics)
    require(confusion_count == summary["confusion_count"], "Confusion count mismatch")
    require(both_positive == summary["both_queries_positive_margin_count"], "Paired success count mismatch")
    return {"status": "PASSED", "queries": len(metrics), "mixtures": len(pairs),
            "confusion_count": confusion_count, "both_queries_positive_margin_count": both_positive,
            "metrics_recomputed": recomputed,
            "scope": "Published artifact verification; no audio inference performed."}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent / "wsj0-2mix"), indent=2))
