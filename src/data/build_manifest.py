"""Build a deduplicated manifest and train/val/test CSV splits for Dusha Crowd."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from random import Random

from sklearn.model_selection import train_test_split

from src.config import DUSHA_ROOT, PROCESSED_DIR, RANDOM_SEED
from src.data.manifest import LABEL_TO_ID, load_split_rows, write_csv


def _apply_debug_limits(
    rows: list[dict[str, object]],
    *,
    max_total: int | None,
    max_per_class: int | None,
    seed: int,
) -> list[dict[str, object]]:
    if max_total is None and max_per_class is None:
        return rows

    rng = Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)

    kept: list[dict[str, object]] = []
    per_class: Counter[int] = Counter()
    for row in shuffled:
        label = int(row["label"])
        if max_per_class is not None and per_class[label] >= max_per_class:
            continue
        kept.append(row)
        per_class[label] += 1
        if max_total is not None and len(kept) >= max_total:
            break
    return kept


def _counts(rows: list[dict[str, object]]) -> Counter[str]:
    return Counter(str(row["label_name"]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DUSHA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-total", type=int, default=None)
    parser.add_argument("--max-per-class", type=int, default=None)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    train_jsonl = dataset_root / "crowd_train" / "raw_crowd_train.jsonl"
    test_jsonl = dataset_root / "crowd_test" / "raw_crowd_test.jsonl"

    for path in (train_jsonl, test_jsonl):
        if not path.exists():
            raise SystemExit(f"Markup file not found: {path}")

    train_rows = load_split_rows(
        train_jsonl,
        dataset_root / "crowd_train",
        source_split="train",
    )
    test_rows = load_split_rows(
        test_jsonl,
        dataset_root / "crowd_test",
        source_split="test",
    )

    train_rows = _apply_debug_limits(
        train_rows,
        max_total=args.max_total,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )

    labels = [int(row["label"]) for row in train_rows]
    train_split, val_split = train_test_split(
        train_rows,
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True,
        stratify=labels,
    )

    manifest_rows = sorted(
        [*train_split, *val_split, *test_rows],
        key=lambda row: (str(row["source_split"]), int(row["label"]), str(row["utt_id"])),
    )

    write_csv(manifest_rows, output_dir / "manifest.csv")
    write_csv(train_split, output_dir / "splits" / "train.csv")
    write_csv(val_split, output_dir / "splits" / "val.csv")
    write_csv(test_rows, output_dir / "splits" / "test.csv")

    print(f"Dataset root: {dataset_root}")
    print(f"Classes: {list(LABEL_TO_ID)}")
    print(f"Train: {len(train_split)} {_counts(train_split)}")
    print(f"Val:   {len(val_split)} {_counts(val_split)}")
    print(f"Test:  {len(test_rows)} {_counts(test_rows)}")
    print(f"Wrote manifest: {output_dir / 'manifest.csv'}")
    print(f"Wrote splits dir: {output_dir / 'splits'}")


if __name__ == "__main__":
    main()
