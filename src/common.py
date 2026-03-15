from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import pandas as pd


USER_BEHAVIOR_COLUMNS = [
    "user_id",
    "item_id",
    "behavior_type",
    "user_geohash",
    "item_category",
    "time",
]

ITEM_COLUMNS = ["item_id", "item_geohash", "item_category"]

BEHAVIOR_WEIGHTS: Dict[int, float] = {
    1: 1.0,   # click
    2: 3.0,   # favorite
    3: 6.0,   # cart
    4: 10.0,  # purchase
}

FEATURE_WINDOWS: tuple[int, ...] = (1, 3, 7, 14)
FEATURE_BEHAVIORS: tuple[int, ...] = (1, 2, 3, 4)
BEHAVIOR_FEATURE_COLS: tuple[str, ...] = tuple(
    f"b{behavior}_{window}d"
    for window in FEATURE_WINDOWS
    for behavior in FEATURE_BEHAVIORS
)


@dataclass(frozen=True)
class DataFiles:
    item_file: Path
    user_files: tuple[Path, ...]


def resolve_default_files(data_dir: Path) -> DataFiles:
    return DataFiles(
        item_file=data_dir / "tianchi_fresh_comp_train_item_online.txt",
        user_files=(
            data_dir / "tianchi_fresh_comp_train_user_online_partA.txt",
            data_dir / "tianchi_fresh_comp_train_user_online_partB.txt",
        ),
    )


def load_target_item_ids(item_file: Path) -> set[int]:
    item_ids = set()
    for chunk in pd.read_csv(
        item_file,
        sep="\t",
        names=ITEM_COLUMNS,
        header=None,
        usecols=[0],
        dtype={"item_id": "int64"},
        chunksize=1_000_000,
    ):
        item_ids.update(chunk["item_id"].tolist())
    return item_ids


def iter_behavior_chunks(
    user_files: Iterable[Path],
    chunksize: int = 1_000_000,
    max_rows_per_file: Optional[int] = None,
) -> Iterator[pd.DataFrame]:
    dtype_map = {
        "user_id": "int64",
        "item_id": "int64",
        "behavior_type": "int8",
        "user_geohash": "string",
        "item_category": "int32",
        "time": "string",
    }
    for user_file in user_files:
        row_budget = max_rows_per_file
        for chunk in pd.read_csv(
            user_file,
            sep="\t",
            names=USER_BEHAVIOR_COLUMNS,
            header=None,
            dtype=dtype_map,
            chunksize=chunksize,
            keep_default_na=False,
        ):
            if row_budget is not None:
                if row_budget <= 0:
                    break
                if len(chunk) > row_budget:
                    chunk = chunk.iloc[:row_budget].copy()
                row_budget -= len(chunk)
            yield chunk


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
