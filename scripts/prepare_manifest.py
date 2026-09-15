"""Resolve portable trial identities against the user's licensed audio roots."""

import argparse
import csv
from pathlib import Path

import soundfile as sf
from apsr.utils.runtime import sha256, write_json


def contained(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Path escapes audio root: {relative}")
    return path


def prepare(trials, mixture_root, enrollment_root, output, wham=False):
    rows = list(csv.DictReader(open(trials, encoding="utf-8")))
    if not rows or len({r["key"] for r in rows}) != len(rows):
        raise ValueError("Trial list must be nonempty with unique keys")
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty manifest directory")
    maps = {name: [] for name in ["mix.scp", "ref.scp", "aux.scp"]}
    groups = {}
    for row in rows:
        mixrel = Path(row["mixture_relpath"])
        refrel = Path(row["target_relpath"])
        if wham:
            mixrel = Path(mixrel.parts[0]) / "mix_both" / mixrel.name
        mix = contained(mixture_root, mixrel)
        target = contained(mixture_root, refrel)
        enrollment = contained(enrollment_root, row["enrollment_relpath"])
        infos = [sf.info(p) for p in (mix, target, enrollment)]
        if any(
            i.samplerate != 8000 or i.channels != 1 or i.frames <= 128 for i in infos
        ):
            raise ValueError(f"Expected usable mono 8-kHz audio: {row['key']}")
        if infos[0].frames != infos[1].frames:
            raise ValueError(f"Mixture/target length mismatch: {row['key']}")
        groups.setdefault(str(mix), []).append(row["target_source"])
        for name, path in zip(maps, (mix, target, enrollment)):
            maps[name].append(f"{row['key']} {path.as_posix()}\n")
    if any(sorted(v) != ["s1", "s2"] for v in groups.values()):
        raise ValueError("Every mixture must have one query for each target")
    out.mkdir(parents=True, exist_ok=True)
    for name, lines in maps.items():
        (out / name).write_text("".join(lines), encoding="utf-8")
    write_json(
        out / "manifest_info.json",
        dict(
            num_queries=len(rows),
            num_mixtures=len(groups),
            trials_sha256=sha256(trials),
            dataset="WHAM! mix_both" if wham else "WSJ0-2mix",
            enrollment_selection="Original trial identities retained; no new enrollment selection",
            files={n: sha256(out / n) for n in maps},
        ),
    )
    print(f"Prepared {len(rows)} queries / {len(groups)} mixtures in {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", required=True)
    p.add_argument(
        "--mixture-root", required=True, help="wav8k/min directory containing tr/cv/tt"
    )
    p.add_argument(
        "--enrollment-root",
        required=True,
        help="Directory containing wsj0/si_tr_s_8k_all etc.",
    )
    p.add_argument("--output", required=True)
    p.add_argument(
        "--wham",
        action="store_true",
        help="Use already synthesized WHAM! mix_both; does not synthesize noise",
    )
    a = p.parse_args()
    prepare(a.trials, a.mixture_root, a.enrollment_root, a.output, a.wham)


if __name__ == "__main__":
    main()
