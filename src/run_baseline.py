from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.common import (
    BEHAVIOR_WEIGHTS,
    BEHAVIOR_FEATURE_COLS,
    FEATURE_BEHAVIORS,
    FEATURE_WINDOWS,
    ensure_dir,
    iter_behavior_chunks,
    load_target_item_ids,
    resolve_default_files,
)


@dataclass
class BuildStats:
    history_rows: int = 0
    label_rows: int = 0
    target_item_rows: int = 0
    chunks: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rule-based baseline for Tianchi mobile recommendation.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--mode",
        choices=["offline", "submit", "both"],
        default="offline",
    )
    parser.add_argument("--eval-date", type=str, default="2014-12-18")
    parser.add_argument("--predict-date", type=str, default="2014-12-19")
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--decay", type=float, default=0.85)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--topk-grid", type=str, default="10,20,30")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--db-name", type=str, default="baseline_work.db")
    parser.add_argument("--weight-click", type=float, default=BEHAVIOR_WEIGHTS[1])
    parser.add_argument("--weight-favorite", type=float, default=BEHAVIOR_WEIGHTS[2])
    parser.add_argument("--weight-cart", type=float, default=BEHAVIOR_WEIGHTS[3])
    parser.add_argument("--weight-buy", type=float, default=BEHAVIOR_WEIGHTS[4])
    parser.add_argument(
        "--log-every-chunks",
        type=int,
        default=50,
        help="Print progress every N chunks while scanning behavior logs.",
    )
    return parser.parse_args()


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def to_date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def parse_topk_grid(value: str) -> List[int]:
    result = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        result.append(int(part))
    return sorted(set(result))


def _feature_columns_sql() -> str:
    return ",\n            ".join(f"{col} INTEGER NOT NULL DEFAULT 0" for col in BEHAVIOR_FEATURE_COLS)


def _feature_insert_columns(key_cols: List[str]) -> List[str]:
    return key_cols + ["last_hours_gap", *BEHAVIOR_FEATURE_COLS]


def _feature_upsert_sql(table: str, key_cols: List[str]) -> str:
    insert_cols = _feature_insert_columns(key_cols)
    key_expr = ", ".join(key_cols)
    update_parts = ["last_hours_gap = MIN(last_hours_gap, excluded.last_hours_gap)"]
    update_parts.extend([f"{col} = {col} + excluded.{col}" for col in BEHAVIOR_FEATURE_COLS])
    update_expr = ",\n            ".join(update_parts)
    placeholders = ", ".join(["?"] * len(insert_cols))
    cols_expr = ", ".join(insert_cols)
    return f"""
        INSERT INTO {table} ({cols_expr})
        VALUES ({placeholders})
        ON CONFLICT({key_expr})
        DO UPDATE SET
            {update_expr}
    """


def init_sqlite(db_path: Path) -> sqlite3.Connection:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(
        """
        CREATE TABLE scores (
            user_id INTEGER NOT NULL,
            item_id INTEGER NOT NULL,
            score REAL NOT NULL,
            PRIMARY KEY (user_id, item_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE labels (
            user_id INTEGER NOT NULL,
            item_id INTEGER NOT NULL,
            PRIMARY KEY (user_id, item_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE active_users (
            user_id INTEGER PRIMARY KEY
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE pair_features (
            user_id INTEGER NOT NULL,
            item_id INTEGER NOT NULL,
            last_hours_gap INTEGER NOT NULL DEFAULT 99999,
            {_feature_columns_sql()},
            PRIMARY KEY (user_id, item_id)
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE user_features (
            user_id INTEGER PRIMARY KEY,
            last_hours_gap INTEGER NOT NULL DEFAULT 99999,
            {_feature_columns_sql()}
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE item_features (
            item_id INTEGER PRIMARY KEY,
            last_hours_gap INTEGER NOT NULL DEFAULT 99999,
            {_feature_columns_sql()}
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE category_features (
            item_category INTEGER PRIMARY KEY,
            last_hours_gap INTEGER NOT NULL DEFAULT 99999,
            {_feature_columns_sql()}
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE user_category_features (
            user_id INTEGER NOT NULL,
            item_category INTEGER NOT NULL,
            last_hours_gap INTEGER NOT NULL DEFAULT 99999,
            {_feature_columns_sql()},
            PRIMARY KEY (user_id, item_category)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE item_category_map (
            item_id INTEGER PRIMARY KEY,
            item_category INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def upsert_scores(conn: sqlite3.Connection, grouped: pd.DataFrame) -> None:
    if grouped.empty:
        return
    payload = list(grouped.itertuples(index=False, name=None))
    conn.executemany(
        """
        INSERT INTO scores (user_id, item_id, score)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, item_id)
        DO UPDATE SET score = score + excluded.score
        """,
        payload,
    )


def upsert_feature_table(
    conn: sqlite3.Connection,
    table: str,
    key_cols: List[str],
    grouped: pd.DataFrame,
) -> None:
    if grouped.empty:
        return
    insert_cols = _feature_insert_columns(key_cols)
    payload = list(grouped[insert_cols].itertuples(index=False, name=None))
    conn.executemany(_feature_upsert_sql(table=table, key_cols=key_cols), payload)


def insert_labels(conn: sqlite3.Connection, labels: pd.DataFrame) -> None:
    if labels.empty:
        return
    payload = list(labels.itertuples(index=False, name=None))
    conn.executemany(
        """
        INSERT OR IGNORE INTO labels (user_id, item_id)
        VALUES (?, ?)
        """,
        payload,
    )


def insert_active_users(conn: sqlite3.Connection, users: Iterable[int]) -> None:
    payload = [(int(u),) for u in users]
    if not payload:
        return
    conn.executemany(
        """
        INSERT OR IGNORE INTO active_users (user_id)
        VALUES (?)
        """,
        payload,
    )


def insert_item_category_map(conn: sqlite3.Connection, item_cats: pd.DataFrame) -> None:
    if item_cats.empty:
        return
    payload = list(item_cats.itertuples(index=False, name=None))
    conn.executemany(
        """
        INSERT OR IGNORE INTO item_category_map (item_id, item_category)
        VALUES (?, ?)
        """,
        payload,
    )


def build_tables_for_target_date(
    conn: sqlite3.Connection,
    user_files: tuple[Path, ...],
    item_ids: set[int],
    target_date: datetime,
    lookback_days: int,
    decay: float,
    weights: Dict[int, float],
    chunksize: int,
    max_rows_per_file: Optional[int],
    log_every_chunks: int,
) -> BuildStats:
    stats = BuildStats()
    target_date_str = to_date_str(target_date)
    hist_start = target_date - timedelta(days=lookback_days)
    hist_start_str = to_date_str(hist_start)
    hist_end = target_date - timedelta(days=1)

    date_diff_map = {
        to_date_str(hist_start + timedelta(days=i)): (target_date - (hist_start + timedelta(days=i))).days
        for i in range((hist_end - hist_start).days + 1)
    }

    for chunk in iter_behavior_chunks(
        user_files=user_files,
        chunksize=chunksize,
        max_rows_per_file=max_rows_per_file,
    ):
        stats.chunks += 1
        chunk = chunk[chunk["item_id"].isin(item_ids)]
        if chunk.empty:
            continue

        stats.target_item_rows += len(chunk)
        insert_item_category_map(
            conn,
            chunk[["item_id", "item_category"]]
            .drop_duplicates()
            .astype({"item_id": "int64", "item_category": "int32"}),
        )
        date_str = chunk["time"].str.slice(0, 10)
        chunk = chunk.assign(date=date_str)

        label_mask = (chunk["date"] == target_date_str) & (chunk["behavior_type"] == 4)
        label_pairs = chunk.loc[label_mask, ["user_id", "item_id"]].drop_duplicates()
        stats.label_rows += len(label_pairs)
        insert_labels(conn, label_pairs)

        history_mask = (chunk["date"] >= hist_start_str) & (chunk["date"] < target_date_str)
        history = chunk.loc[
            history_mask,
            ["user_id", "item_id", "behavior_type", "item_category", "date", "time"],
        ]
        if history.empty:
            conn.commit()
            continue

        insert_active_users(conn, history["user_id"].drop_duplicates().tolist())
        days_diff = history["date"].map(date_diff_map).astype("int16")
        hour_vals = history["time"].str[-2:].astype("int16")
        hours_gap = ((days_diff.astype("int32") - 1) * 24 + (24 - hour_vals.astype("int32"))).astype(
            "int32"
        )

        behavior_w = history["behavior_type"].map(weights).astype("float32")
        recency = np.power(decay, days_diff.to_numpy(dtype=np.float32))
        history_scores = pd.DataFrame(
            {
                "user_id": history["user_id"].to_numpy(dtype=np.int64),
                "item_id": history["item_id"].to_numpy(dtype=np.int64),
                "score": behavior_w.to_numpy(dtype=np.float32) * recency,
            }
        )
        grouped = history_scores.groupby(["user_id", "item_id"], as_index=False)["score"].sum()
        stats.history_rows += len(history_scores)
        upsert_scores(conn, grouped)

        feature_frame = pd.DataFrame(
            {
                "user_id": history["user_id"].to_numpy(dtype=np.int64),
                "item_id": history["item_id"].to_numpy(dtype=np.int64),
                "item_category": history["item_category"].to_numpy(dtype=np.int32),
                "last_hours_gap": hours_gap.to_numpy(dtype=np.int32),
            }
        )
        for window in FEATURE_WINDOWS:
            in_window = (days_diff <= window).to_numpy()
            for behavior in FEATURE_BEHAVIORS:
                col = f"b{behavior}_{window}d"
                feature_frame[col] = (
                    (history["behavior_type"].to_numpy() == behavior) & in_window
                ).astype(np.int16)

        agg_map = {"last_hours_gap": "min", **{col: "sum" for col in BEHAVIOR_FEATURE_COLS}}
        pair_group = feature_frame.groupby(["user_id", "item_id"], as_index=False).agg(agg_map)
        user_group = feature_frame.groupby(["user_id"], as_index=False).agg(agg_map)
        item_group = feature_frame.groupby(["item_id"], as_index=False).agg(agg_map)
        category_group = feature_frame.groupby(["item_category"], as_index=False).agg(agg_map)
        user_category_group = feature_frame.groupby(["user_id", "item_category"], as_index=False).agg(agg_map)

        upsert_feature_table(conn, "pair_features", ["user_id", "item_id"], pair_group)
        upsert_feature_table(conn, "user_features", ["user_id"], user_group)
        upsert_feature_table(conn, "item_features", ["item_id"], item_group)
        upsert_feature_table(conn, "category_features", ["item_category"], category_group)
        upsert_feature_table(
            conn,
            "user_category_features",
            ["user_id", "item_category"],
            user_category_group,
        )
        conn.commit()

        if log_every_chunks > 0 and stats.chunks % log_every_chunks == 0:
            print(
                "[PROGRESS]",
                f"target_date={target_date_str}",
                f"chunks={stats.chunks}",
                f"target_item_rows={stats.target_item_rows}",
                f"history_rows={stats.history_rows}",
                f"label_rows={stats.label_rows}",
                flush=True,
            )

    conn.execute("CREATE INDEX IF NOT EXISTS idx_scores_user_score ON scores(user_id, score DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_labels_pair ON labels(user_id, item_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_item_category_map_cat ON item_category_map(item_category)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_category_features_uc ON user_category_features(user_id, item_category)"
    )
    conn.commit()
    return stats


def fetch_ranked_predictions(
    conn: sqlite3.Connection,
    max_topk: int,
    min_score: float,
) -> pd.DataFrame:
    query = """
    WITH ranked AS (
        SELECT
            user_id,
            item_id,
            score,
            ROW_NUMBER() OVER (
                PARTITION BY user_id
                ORDER BY score DESC, item_id ASC
            ) AS rn
        FROM scores
        WHERE score >= ?
    )
    SELECT user_id, item_id, score, rn
    FROM ranked
    WHERE rn <= ?
    ORDER BY user_id ASC, rn ASC
    """
    df = pd.read_sql_query(query, conn, params=[float(min_score), int(max_topk)])
    return df


def compute_metrics(
    ranked_predictions: pd.DataFrame,
    labels: pd.DataFrame,
    topk_values: List[int],
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    if labels.empty:
        for topk in topk_values:
            metrics[str(topk)] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "predictions": float((ranked_predictions["rn"] <= topk).sum()),
                "labels": 0.0,
                "hits": 0.0,
            }
        return metrics

    labels_key = labels[["user_id", "item_id"]].drop_duplicates()
    labels_count = len(labels_key)

    for topk in topk_values:
        pred = ranked_predictions.loc[
            ranked_predictions["rn"] <= topk, ["user_id", "item_id"]
        ]
        pred = pred.drop_duplicates()
        hits = pred.merge(labels_key, on=["user_id", "item_id"], how="inner")
        hit_count = len(hits)
        pred_count = len(pred)
        precision = hit_count / pred_count if pred_count else 0.0
        recall = hit_count / labels_count if labels_count else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        metrics[str(topk)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": float(pred_count),
            "labels": float(labels_count),
            "hits": float(hit_count),
        }
    return metrics


def export_predictions(
    ranked_predictions: pd.DataFrame,
    topk: int,
    output_path: Path,
) -> int:
    out = ranked_predictions.loc[ranked_predictions["rn"] <= topk, ["user_id", "item_id"]]
    out = out.drop_duplicates()
    out.to_csv(output_path, sep="\t", index=False, header=False)
    return len(out)


def run_for_target(
    conn: sqlite3.Connection,
    user_files: tuple[Path, ...],
    item_ids: set[int],
    target_date_str: str,
    lookback_days: int,
    decay: float,
    weights: Dict[int, float],
    chunksize: int,
    max_rows_per_file: Optional[int],
    min_score: float,
    topk: int,
    topk_grid: List[int],
    output_pred_path: Path,
    output_metrics_path: Optional[Path],
    log_every_chunks: int,
) -> Dict[str, object]:
    target_date = parse_date(target_date_str)
    build_stats = build_tables_for_target_date(
        conn=conn,
        user_files=user_files,
        item_ids=item_ids,
        target_date=target_date,
        lookback_days=lookback_days,
        decay=decay,
        weights=weights,
        chunksize=chunksize,
        max_rows_per_file=max_rows_per_file,
        log_every_chunks=log_every_chunks,
    )

    max_topk = max(topk_grid + [topk])
    ranked = fetch_ranked_predictions(conn, max_topk=max_topk, min_score=min_score)
    pred_count = export_predictions(ranked, topk=topk, output_path=output_pred_path)

    result: Dict[str, object] = {
        "target_date": target_date_str,
        "build_stats": build_stats.__dict__,
        "predictions_exported": pred_count,
        "prediction_file": str(output_pred_path),
        "topk": topk,
        "min_score": min_score,
    }

    if output_metrics_path is not None:
        labels = pd.read_sql_query("SELECT user_id, item_id FROM labels", conn)
        metrics = compute_metrics(ranked, labels, topk_values=topk_grid)
        result["metrics_by_topk"] = metrics
        best_topk = max(metrics.items(), key=lambda kv: kv[1]["f1"])[0]
        result["best_topk_by_f1"] = int(best_topk)
        output_metrics_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        result["metrics_file"] = str(output_metrics_path)
    return result


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    data_files = resolve_default_files(args.data_dir)

    missing = [
        str(path)
        for path in [data_files.item_file, *data_files.user_files]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    weights = {
        1: float(args.weight_click),
        2: float(args.weight_favorite),
        3: float(args.weight_cart),
        4: float(args.weight_buy),
    }
    topk_grid = parse_topk_grid(args.topk_grid)
    item_ids = load_target_item_ids(data_files.item_file)

    summary: Dict[str, object] = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "weights": weights,
        "lookback_days": args.lookback_days,
        "decay": args.decay,
        "chunksize": args.chunksize,
        "max_rows_per_file": args.max_rows_per_file,
    }

    if args.mode in {"offline", "both"}:
        offline_db = args.output_dir / f"offline_{args.db_name}"
        conn = init_sqlite(offline_db)
        try:
            offline_pred = args.output_dir / "offline_predictions.tsv"
            offline_metrics = args.output_dir / "offline_metrics.json"
            offline_result = run_for_target(
                conn=conn,
                user_files=data_files.user_files,
                item_ids=item_ids,
                target_date_str=args.eval_date,
                lookback_days=args.lookback_days,
                decay=args.decay,
                weights=weights,
                chunksize=args.chunksize,
                max_rows_per_file=args.max_rows_per_file,
                min_score=args.min_score,
                topk=args.topk,
                topk_grid=topk_grid,
                output_pred_path=offline_pred,
                output_metrics_path=offline_metrics,
                log_every_chunks=args.log_every_chunks,
            )
            summary["offline"] = offline_result
        finally:
            conn.close()

    if args.mode in {"submit", "both"}:
        submit_db = args.output_dir / f"submit_{args.db_name}"
        conn = init_sqlite(submit_db)
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            submit_file = args.output_dir / f"submit_{ts}.tsv"
            submit_result = run_for_target(
                conn=conn,
                user_files=data_files.user_files,
                item_ids=item_ids,
                target_date_str=args.predict_date,
                lookback_days=args.lookback_days,
                decay=args.decay,
                weights=weights,
                chunksize=args.chunksize,
                max_rows_per_file=args.max_rows_per_file,
                min_score=args.min_score,
                topk=args.topk,
                topk_grid=topk_grid,
                output_pred_path=submit_file,
                output_metrics_path=None,
                log_every_chunks=args.log_every_chunks,
            )
            summary["submit"] = submit_result
        finally:
            conn.close()

    summary_path = args.output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Baseline run finished. Summary: {summary_path}")


if __name__ == "__main__":
    main()
