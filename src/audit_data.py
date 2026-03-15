from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.common import (
    ITEM_COLUMNS,
    USER_BEHAVIOR_COLUMNS,
    ensure_dir,
    iter_behavior_chunks,
    load_target_item_ids,
    resolve_default_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Tianchi dataset (chunk-based).")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("outputs/audit/data_profile.md"))
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Debug mode: read at most N rows from each behavior file.",
    )
    parser.add_argument(
        "--sample-chunks",
        type=int,
        default=None,
        help="Debug mode: only process first N chunks in total.",
    )
    parser.add_argument(
        "--compute-unique",
        action="store_true",
        help="Compute exact unique user/item/category counts (may consume memory).",
    )
    return parser.parse_args()


def audit_items(item_file: Path) -> Dict[str, float]:
    total_rows = 0
    geohash_missing = 0
    unique_item_ids: set[int] = set()
    unique_categories: set[int] = set()

    for chunk in pd.read_csv(
        item_file,
        sep="\t",
        names=ITEM_COLUMNS,
        header=None,
        dtype={"item_id": "int64", "item_geohash": "string", "item_category": "int32"},
        chunksize=1_000_000,
        keep_default_na=False,
    ):
        total_rows += len(chunk)
        geohash_missing += (chunk["item_geohash"] == "").sum()
        unique_item_ids.update(chunk["item_id"].tolist())
        unique_categories.update(chunk["item_category"].tolist())

    missing_ratio = geohash_missing / total_rows if total_rows else 0.0
    return {
        "item_rows": total_rows,
        "item_unique_item_ids": len(unique_item_ids),
        "item_unique_categories": len(unique_categories),
        "item_geohash_missing_ratio": missing_ratio,
    }


def audit_behaviors(
    user_files: tuple[Path, ...],
    item_ids: set[int],
    chunksize: int,
    max_rows_per_file: Optional[int],
    sample_chunks: Optional[int],
    compute_unique: bool,
) -> Dict[str, object]:
    total_rows = 0
    target_item_rows = 0
    geohash_missing = 0
    behavior_counts = Counter()
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    unique_users: set[int] = set()
    unique_items: set[int] = set()
    unique_categories: set[int] = set()

    chunks_seen = 0
    for chunk in iter_behavior_chunks(
        user_files=user_files,
        chunksize=chunksize,
        max_rows_per_file=max_rows_per_file,
    ):
        chunks_seen += 1
        if sample_chunks is not None and chunks_seen > sample_chunks:
            break

        total_rows += len(chunk)
        geohash_missing += (chunk["user_geohash"] == "").sum()
        behavior_counts.update(chunk["behavior_type"].tolist())

        date_str = chunk["time"].str.slice(0, 10)
        current_min = date_str.min()
        current_max = date_str.max()
        min_date = current_min if min_date is None else min(min_date, current_min)
        max_date = current_max if max_date is None else max(max_date, current_max)

        target_item_rows += chunk["item_id"].isin(item_ids).sum()

        if compute_unique:
            unique_users.update(chunk["user_id"].tolist())
            unique_items.update(chunk["item_id"].tolist())
            unique_categories.update(chunk["item_category"].tolist())

    geohash_missing_ratio = geohash_missing / total_rows if total_rows else 0.0
    target_item_ratio = target_item_rows / total_rows if total_rows else 0.0

    metrics: Dict[str, object] = {
        "behavior_rows": total_rows,
        "target_item_rows": target_item_rows,
        "target_item_ratio": target_item_ratio,
        "user_geohash_missing_ratio": geohash_missing_ratio,
        "behavior_type_counts": dict(sorted(behavior_counts.items())),
        "min_date": min_date,
        "max_date": max_date,
    }
    if compute_unique:
        metrics["unique_users"] = len(unique_users)
        metrics["unique_items"] = len(unique_items)
        metrics["unique_categories"] = len(unique_categories)

    return metrics


def format_markdown(
    files_info: Dict[str, str],
    item_metrics: Dict[str, float],
    behavior_metrics: Dict[str, object],
    args: argparse.Namespace,
) -> str:
    lines = [
        "# Data Profile Report",
        "",
        f"Generated at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## Inputs",
        f"- item_file: `{files_info['item_file']}`",
        f"- user_file_A: `{files_info['user_file_A']}`",
        f"- user_file_B: `{files_info['user_file_B']}`",
        "",
        "## Run Config",
        f"- chunksize: `{args.chunksize}`",
        f"- max_rows_per_file: `{args.max_rows_per_file}`",
        f"- sample_chunks: `{args.sample_chunks}`",
        f"- compute_unique: `{args.compute_unique}`",
        "",
        "## Item Table Stats",
        f"- item_rows: `{item_metrics['item_rows']}`",
        f"- unique_item_ids: `{item_metrics['item_unique_item_ids']}`",
        f"- unique_categories: `{item_metrics['item_unique_categories']}`",
        f"- item_geohash_missing_ratio: `{item_metrics['item_geohash_missing_ratio']:.6f}`",
        "",
        "## Behavior Table Stats",
        f"- behavior_rows: `{behavior_metrics['behavior_rows']}`",
        f"- target_item_rows: `{behavior_metrics['target_item_rows']}`",
        f"- target_item_ratio: `{behavior_metrics['target_item_ratio']:.6f}`",
        f"- user_geohash_missing_ratio: `{behavior_metrics['user_geohash_missing_ratio']:.6f}`",
        f"- min_date: `{behavior_metrics['min_date']}`",
        f"- max_date: `{behavior_metrics['max_date']}`",
        "",
        "### behavior_type_counts",
        "```json",
        json.dumps(behavior_metrics["behavior_type_counts"], indent=2, ensure_ascii=False),
        "```",
    ]
    if "unique_users" in behavior_metrics:
        lines.extend(
            [
                "",
                "### Optional Unique Counts",
                f"- unique_users: `{behavior_metrics['unique_users']}`",
                f"- unique_items: `{behavior_metrics['unique_items']}`",
                f"- unique_categories: `{behavior_metrics['unique_categories']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    data_files = resolve_default_files(args.data_dir)

    missing = [
        str(path)
        for path in [data_files.item_file, *data_files.user_files]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    item_ids = load_target_item_ids(data_files.item_file)
    item_metrics = audit_items(data_files.item_file)
    behavior_metrics = audit_behaviors(
        user_files=data_files.user_files,
        item_ids=item_ids,
        chunksize=args.chunksize,
        max_rows_per_file=args.max_rows_per_file,
        sample_chunks=args.sample_chunks,
        compute_unique=args.compute_unique,
    )

    ensure_dir(args.output.parent)
    report = format_markdown(
        files_info={
            "item_file": str(data_files.item_file),
            "user_file_A": str(data_files.user_files[0]),
            "user_file_B": str(data_files.user_files[1]),
        },
        item_metrics=item_metrics,
        behavior_metrics=behavior_metrics,
        args=args,
    )
    args.output.write_text(report, encoding="utf-8")
    print(f"[OK] Data profile saved to: {args.output}")


if __name__ == "__main__":
    main()
