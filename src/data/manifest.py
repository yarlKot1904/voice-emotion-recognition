"""Helpers for turning Dusha jsonl markup into training-ready rows."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.config import CLASS_NAMES

LABEL_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
GOLDEN_TO_LABEL: dict[int, str] = {
    1: "positive",
    2: "neutral",
    3: "sad",
    4: "angry",
    5: "other",
}

MANIFEST_COLUMNS: tuple[str, ...] = (
    "path",
    "label",
    "label_name",
    "duration",
    "source_split",
    "utt_id",
    "text",
    "num_votes",
    "label_source",
)


def _mapped_golden(value: Any) -> str | None:
    if not isinstance(value, (int, float)) or math.isnan(value):
        return None
    return GOLDEN_TO_LABEL.get(int(value))


def resolve_emotion(records: list[dict[str, Any]]) -> tuple[str | None, str]:
    golden_labels = [
        label
        for item in records
        if (label := _mapped_golden(item.get("golden_emo"))) in LABEL_TO_ID
    ]
    if golden_labels:
        counts = Counter(golden_labels)
        return counts.most_common(1)[0][0], "golden"

    votes = Counter(
        item.get("annotator_emo")
        for item in records
        if item.get("annotator_emo") in LABEL_TO_ID
    )
    if not votes:
        speaker = next(
            (item.get("speaker_emo") for item in records if item.get("speaker_emo") in LABEL_TO_ID),
            None,
        )
        if speaker is not None:
            return speaker, "speaker"
        return None, "missing"

    top_count = max(votes.values())
    tied = [label for label, count in votes.items() if count == top_count]
    if len(tied) == 1:
        return tied[0], "vote"

    speaker = next(
        (item.get("speaker_emo") for item in records if item.get("speaker_emo") in tied),
        None,
    )
    if speaker is not None:
        return speaker, "vote_tie_speaker"

    ordered = sorted(tied, key=lambda label: LABEL_TO_ID[label])
    return ordered[0], "vote_tie_order"


def load_split_rows(
    jsonl_path: Path,
    wav_root: Path,
    *,
    source_split: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            grouped[row["audio_path"]].append(row)

    rows: list[dict[str, Any]] = []
    for rel_audio_path, records in grouped.items():
        label_name, label_source = resolve_emotion(records)
        if label_name is None:
            continue

        wav_path = (wav_root / rel_audio_path).resolve()
        if not wav_path.exists():
            continue

        first = records[0]
        rows.append(
            {
                "path": str(wav_path),
                "label": LABEL_TO_ID[label_name],
                "label_name": label_name,
                "duration": float(first.get("duration") or 0.0),
                "source_split": source_split,
                "utt_id": first.get("hash_id") or wav_path.stem,
                "text": "" if first.get("speaker_text") is None else str(first.get("speaker_text")),
                "num_votes": len(records),
                "label_source": label_source,
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

