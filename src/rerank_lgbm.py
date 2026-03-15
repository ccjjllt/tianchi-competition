from __future__ import annotations

import argparse
import gc
import heapq
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.common import BEHAVIOR_FEATURE_COLS, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Second-stage reranking with LightGBM (GPU/CPU) on baseline candidates."
    )
    parser.add_argument(
        "--offline-db",
        type=Path,
        default=Path("outputs/baseline/offline_baseline_work.db"),
    )
    parser.add_argument(
        "--submit-db",
        type=Path,
        default=Path("outputs/baseline/submit_baseline_work.db"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rerank"))
    parser.add_argument("--candidate-topk", type=int, default=50)
    parser.add_argument(
        "--offline-eval-mode",
        choices=["global", "per_user"],
        default="global",
        help="offline metric protocol: global topN or per-user topK",
    )
    parser.add_argument(
        "--topk-grid",
        type=str,
        default="30000,40000,50000,60000,70000,80000,90000,100000",
    )
    parser.add_argument("--submit-topk", type=int, default=None)
    parser.add_argument("--neg-pos-ratio", type=int, default=30)
    parser.add_argument(
        "--neg-sample-mode",
        choices=["random", "hard", "mixed"],
        default="mixed",
    )
    parser.add_argument("--num-boost-round", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-data-in-leaf", type=int, default=50)
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.10,
        help="Validation split ratio used by LightGBM early stopping.",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
        help="Early stopping rounds for LightGBM; <=0 disables early stopping.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument(
        "--train-streaming",
        action="store_true",
        help="stream candidates and build a bounded sampled train set to avoid peak memory",
    )
    parser.add_argument(
        "--train-max-positives",
        type=int,
        default=200_000,
        help="max positive rows kept by streaming sampler (all if <=0)",
    )
    parser.add_argument(
        "--train-max-negatives",
        type=int,
        default=600_000,
        help="max negative rows kept by streaming sampler (all if <=0)",
    )
    parser.add_argument(
        "--train-hard-neg-ratio",
        type=float,
        default=0.85,
        help="ratio of hard negatives in streaming negative sample",
    )
    parser.add_argument(
        "--train-read-chunksize",
        type=int,
        default=1_000_000,
        help="rows per SQL chunk while streaming offline candidates for training",
    )
    parser.add_argument("--predict-batch-size", type=int, default=2_000_000)
    parser.add_argument(
        "--submit-read-chunksize",
        type=int,
        default=1_000_000,
        help="Rows per SQL chunk when streaming submit candidates.",
    )
    parser.add_argument("--log-eval", type=int, default=50)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Debug mode: cap candidate rows after loading from DB.",
    )
    parser.add_argument(
        "--disable-enhanced-features",
        action="store_true",
        help="Fallback switch: disable newly added engineered features in rerank stage.",
    )
    return parser.parse_args()

def parse_topk_grid(value: str) -> List[int]:
    result: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        result.append(int(part))
    if not result:
        raise ValueError("topk grid cannot be empty.")
    return sorted(set(result))




def _count_non_finite(frame: pd.DataFrame, cols: List[str]) -> int:
    non_finite = 0
    for col in cols:
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        non_finite += int(np.count_nonzero(~np.isfinite(values)))
    return int(non_finite)


def _sample_stratified_negatives(
    neg_df: pd.DataFrame,
    take: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    neg_pos = neg_df.reset_index(drop=True)
    if take <= 0 or neg_pos.empty:
        return neg_pos.iloc[:0].copy()
    if len(neg_pos) <= take:
        return neg_pos.copy()

    work = neg_pos[["rn", "score"]].copy()
    rn = pd.to_numeric(work["rn"], errors="coerce").fillna(999).astype("int32")
    score = pd.to_numeric(work["score"], errors="coerce").fillna(0.0).astype("float32")

    work["rn_bin"] = np.select(
        [rn <= 5, rn <= 10, rn <= 20, rn <= 50],
        ["r1", "r2", "r3", "r4"],
        default="r5",
    )
    score_pct = score.rank(method="first", pct=True)
    work["score_bin"] = pd.cut(
        score_pct,
        bins=[0.0, 0.25, 0.5, 0.75, 1.0],
        labels=["q1", "q2", "q3", "q4"],
        include_lowest=True,
    ).astype(str)
    work["stratum"] = work["rn_bin"].astype(str) + "_" + work["score_bin"].astype(str)

    grp = work.groupby("stratum", observed=True)
    stratum_counts = grp.size().astype("int64")
    quotas_raw = stratum_counts / float(stratum_counts.sum()) * float(take)
    quotas = np.floor(quotas_raw).astype("int64")
    remainder = int(take - int(quotas.sum()))

    if remainder > 0:
        frac = (quotas_raw - quotas).sort_values(ascending=False)
        for key in frac.index[:remainder]:
            quotas.loc[key] += 1

    chosen_idx: List[int] = []
    for stratum, count in stratum_counts.items():
        quota = int(min(int(quotas.get(stratum, 0)), int(count)))
        if quota <= 0:
            continue
        indices = grp.groups[stratum]
        if quota >= len(indices):
            chosen_idx.extend(indices)
        else:
            chosen_idx.extend(rng.choice(np.asarray(indices), size=quota, replace=False).tolist())

    if len(chosen_idx) < take:
        remain = take - len(chosen_idx)
        all_idx = np.arange(len(neg_pos), dtype=np.int64)
        picked = np.zeros(len(neg_pos), dtype=bool)
        if chosen_idx:
            picked[np.asarray(chosen_idx, dtype=np.int64)] = True
        rest = all_idx[~picked]
        if len(rest) > 0:
            add_n = min(remain, len(rest))
            chosen_idx.extend(rng.choice(rest, size=add_n, replace=False).tolist())

    if len(chosen_idx) > take:
        chosen_idx = rng.choice(np.asarray(chosen_idx, dtype=np.int64), size=take, replace=False).tolist()

    return neg_pos.iloc[np.asarray(chosen_idx, dtype=np.int64)].copy()

def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _build_candidate_query(has_feature_tables: bool, has_category_tables: bool) -> str:
    base_cte = """
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
    ),
    cands AS (
        SELECT
            user_id,
            item_id,
            score,
            rn,
            MAX(score) OVER (PARTITION BY user_id) AS user_score_max,
            SUM(score) OVER (PARTITION BY user_id) AS user_score_sum,
            AVG(score) OVER (PARTITION BY user_id) AS user_score_mean,
            SUM(score * score) OVER (PARTITION BY user_id) AS user_score_sq_sum,
            COUNT(*) OVER (PARTITION BY user_id) AS user_candidate_count,
            MAX(score) OVER (PARTITION BY item_id) AS item_score_max,
            AVG(score) OVER (PARTITION BY item_id) AS item_score_mean,
            SUM(score * score) OVER (PARTITION BY item_id) AS item_score_sq_sum,
            COUNT(*) OVER (PARTITION BY item_id) AS item_candidate_user_count
        FROM ranked
        WHERE rn <= ?
    )
    """
    if not has_feature_tables and not has_category_tables:
        return (
            base_cte
            + """
        SELECT
            c.user_id,
            c.item_id,
            c.score,
            c.rn,
            c.user_score_max,
            c.user_score_sum,
            c.user_score_mean,
            c.user_score_sq_sum,
            c.user_candidate_count,
            c.item_score_max,
            c.item_score_mean,
            c.item_score_sq_sum,
            c.item_candidate_user_count
        FROM cands c
        ORDER BY c.user_id ASC, c.rn ASC
        """
        )

    select_cols = [
        "c.user_id",
        "c.item_id",
        "c.score",
        "c.rn",
        "c.user_score_max",
        "c.user_score_sum",
        "c.user_score_mean",
        "c.user_score_sq_sum",
        "c.user_candidate_count",
        "c.item_score_max",
        "c.item_score_mean",
        "c.item_score_sq_sum",
        "c.item_candidate_user_count",
    ]
    joins = []

    if has_feature_tables:
        pair_cols = [f"COALESCE(pf.{col}, 0) AS p_{col}" for col in BEHAVIOR_FEATURE_COLS]
        user_cols = [f"COALESCE(uf.{col}, 0) AS u_{col}" for col in BEHAVIOR_FEATURE_COLS]
        item_cols = [f"COALESCE(itf.{col}, 0) AS i_{col}" for col in BEHAVIOR_FEATURE_COLS]
        select_cols.extend(
            [
                "COALESCE(pf.last_hours_gap, 99999) AS p_last_hours_gap",
                *pair_cols,
                "COALESCE(uf.last_hours_gap, 99999) AS u_last_hours_gap",
                *user_cols,
                "COALESCE(itf.last_hours_gap, 99999) AS i_last_hours_gap",
                *item_cols,
            ]
        )
        joins.extend(
            [
                """
        LEFT JOIN pair_features pf
            ON c.user_id = pf.user_id
           AND c.item_id = pf.item_id
                """,
                """
        LEFT JOIN user_features uf
            ON c.user_id = uf.user_id
                """,
                """
        LEFT JOIN item_features itf
            ON c.item_id = itf.item_id
                """,
            ]
        )

    if has_category_tables:
        cat_cols = [f"COALESCE(cf.{col}, 0) AS c_{col}" for col in BEHAVIOR_FEATURE_COLS]
        uc_cols = [f"COALESCE(ucf.{col}, 0) AS uc_{col}" for col in BEHAVIOR_FEATURE_COLS]
        select_cols.extend(
            [
                "COALESCE(icm.item_category, -1) AS item_category",
                "COALESCE(cf.last_hours_gap, 99999) AS c_last_hours_gap",
                *cat_cols,
                "COALESCE(ucf.last_hours_gap, 99999) AS uc_last_hours_gap",
                *uc_cols,
            ]
        )
        joins.extend(
            [
                """
        LEFT JOIN item_category_map icm
            ON c.item_id = icm.item_id
                """,
                """
        LEFT JOIN category_features cf
            ON icm.item_category = cf.item_category
                """,
                """
        LEFT JOIN user_category_features ucf
            ON c.user_id = ucf.user_id
           AND icm.item_category = ucf.item_category
                """,
            ]
        )

    select_expr = ",\n            ".join(select_cols)
    join_expr = "\n".join(joins)
    return (
        base_cte
        + f"""
        SELECT
            {select_expr}
        FROM cands c
        {join_expr}
        ORDER BY c.user_id ASC, c.rn ASC
        """
    )


def _downcast_candidate_df(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = [col for col in df.columns if col not in ("user_id", "item_id", "rn")]
    int_cols = [col for col in df.columns if col in ("user_id", "item_id", "rn")]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")
    return df


def read_candidates(
    db_path: Path,
    candidate_topk: int,
    max_candidates: Optional[int] = None,
) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        has_feature_tables = all(
            _table_exists(conn, table)
            for table in ("pair_features", "user_features", "item_features")
        )
        has_category_tables = all(
            _table_exists(conn, table)
            for table in ("item_category_map", "category_features", "user_category_features")
        )
        query = _build_candidate_query(
            has_feature_tables=has_feature_tables,
            has_category_tables=has_category_tables,
        )
        parts: List[pd.DataFrame] = []
        remain = int(max_candidates) if max_candidates is not None else None
        for chunk in pd.read_sql_query(
            query,
            conn,
            params=[int(candidate_topk)],
            chunksize=1_000_000,
        ):
            if remain is not None:
                if remain <= 0:
                    break
                if len(chunk) > remain:
                    chunk = chunk.iloc[:remain].copy()
                remain -= len(chunk)
            parts.append(_downcast_candidate_df(chunk))
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    finally:
        conn.close()
    return df


def read_labels(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        labels = pd.read_sql_query(
            "SELECT DISTINCT user_id, item_id FROM labels",
            conn,
        )
    finally:
        conn.close()
    return labels


def iter_candidates(
    db_path: Path,
    candidate_topk: int,
    chunksize: int,
) -> Iterator[pd.DataFrame]:
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        has_feature_tables = all(
            _table_exists(conn, table)
            for table in ("pair_features", "user_features", "item_features")
        )
        has_category_tables = all(
            _table_exists(conn, table)
            for table in ("item_category_map", "category_features", "user_category_features")
        )
        query = _build_candidate_query(
            has_feature_tables=has_feature_tables,
            has_category_tables=has_category_tables,
        )
        for chunk in pd.read_sql_query(
            query,
            conn,
            params=[int(candidate_topk)],
            chunksize=int(chunksize),
        ):
            yield _downcast_candidate_df(chunk)
    finally:
        conn.close()


def build_features(candidates: pd.DataFrame, enable_enhanced: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    df = candidates.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0).astype("float32")
    df["rn"] = pd.to_numeric(df["rn"], errors="coerce").fillna(999).astype("int32")
    df["log_score"] = np.log1p(df["score"])

    if "user_score_max" not in df.columns:
        user_stats = (
            df.groupby("user_id")["score"]
            .agg(["max", "sum", "mean", "std", "count"])
            .rename(
                columns={
                    "max": "user_score_max",
                    "sum": "user_score_sum",
                    "mean": "user_score_mean",
                    "std": "user_score_std",
                    "count": "user_candidate_count",
                }
            )
            .reset_index()
        )
        df = df.merge(user_stats, on="user_id", how="left")

    if "item_score_max" not in df.columns:
        item_stats = (
            df.groupby("item_id")["score"]
            .agg(["max", "mean", "std", "count"])
            .rename(
                columns={
                    "max": "item_score_max",
                    "mean": "item_score_mean",
                    "std": "item_score_std",
                    "count": "item_candidate_user_count",
                }
            )
            .reset_index()
        )
        df = df.merge(item_stats, on="item_id", how="left")

    eps = 1e-6
    if {"user_score_sq_sum", "user_candidate_count", "user_score_mean"}.issubset(df.columns):
        df["user_score_std"] = np.sqrt(
            np.maximum(
                df["user_score_sq_sum"] / (df["user_candidate_count"] + eps)
                - np.square(df["user_score_mean"]),
                0.0,
            )
        )
    if {"item_score_sq_sum", "item_candidate_user_count", "item_score_mean"}.issubset(df.columns):
        df["item_score_std"] = np.sqrt(
            np.maximum(
                df["item_score_sq_sum"] / (df["item_candidate_user_count"] + eps)
                - np.square(df["item_score_mean"]),
                0.0,
            )
        )

    df["rank_inv"] = 1.0 / df["rn"].astype("float32")
    df["rank_pct"] = df["rn"].astype("float32") / (
        df["user_candidate_count"].astype("float32") + eps
    )
    df["score_div_user_max"] = df["score"] / (df["user_score_max"] + eps)
    df["score_div_user_sum"] = df["score"] / (df["user_score_sum"] + eps)
    df["score_minus_user_mean"] = df["score"] - df["user_score_mean"]
    df["score_minus_item_mean"] = df["score"] - df["item_score_mean"]
    df["user_candidate_count_log"] = np.log1p(df["user_candidate_count"])
    df["item_candidate_user_count_log"] = np.log1p(df["item_candidate_user_count"])

    feature_cols: List[str] = [
        "score",
        "log_score",
        "rn",
        "rank_inv",
        "rank_pct",
        "score_div_user_max",
        "score_div_user_sum",
        "score_minus_user_mean",
        "score_minus_item_mean",
        "user_score_mean",
        "user_score_std",
        "user_candidate_count",
        "item_score_mean",
        "item_score_std",
        "item_candidate_user_count",
        "user_candidate_count_log",
        "item_candidate_user_count_log",
    ]

    if "p_last_hours_gap" in df.columns or "c_last_hours_gap" in df.columns or "uc_last_hours_gap" in df.columns:
        prefixed_raw_cols = [
            col
            for col in df.columns
            if col.startswith("p_b")
            or col.startswith("u_b")
            or col.startswith("i_b")
            or col.startswith("c_b")
            or col.startswith("uc_b")
            or col in (
                "p_last_hours_gap",
                "u_last_hours_gap",
                "i_last_hours_gap",
                "c_last_hours_gap",
                "uc_last_hours_gap",
            )
        ]

        for col in (
            "p_last_hours_gap",
            "u_last_hours_gap",
            "i_last_hours_gap",
            "c_last_hours_gap",
            "uc_last_hours_gap",
        ):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(99999).clip(lower=0, upper=99999)
                df[f"{col}_inv"] = 1.0 / (df[col] + 1.0)
                feature_cols.append(f"{col}_inv")

        for scope in ("p", "u", "i", "c", "uc"):
            b1_14 = f"{scope}_b1_14d"
            b2_14 = f"{scope}_b2_14d"
            b3_14 = f"{scope}_b3_14d"
            b4_14 = f"{scope}_b4_14d"
            b1_1 = f"{scope}_b1_1d"
            b2_1 = f"{scope}_b2_1d"
            b3_1 = f"{scope}_b3_1d"
            b4_1 = f"{scope}_b4_1d"

            if all(col in df.columns for col in (b1_14, b2_14, b3_14, b4_14)):
                df[f"{scope}_buy_click_ratio_14d"] = df[b4_14] / (df[b1_14] + 1.0)
                df[f"{scope}_buy_fav_ratio_14d"] = df[b4_14] / (df[b2_14] + 1.0)
                df[f"{scope}_buy_cart_ratio_14d"] = df[b4_14] / (df[b3_14] + 1.0)
                df[f"{scope}_engage_14d"] = df[b1_14] + 3.0 * df[b2_14] + 6.0 * df[b3_14] + 10.0 * df[b4_14]
                feature_cols.extend(
                    [
                        f"{scope}_buy_click_ratio_14d",
                        f"{scope}_buy_fav_ratio_14d",
                        f"{scope}_buy_cart_ratio_14d",
                        f"{scope}_engage_14d",
                    ]
                )

                if enable_enhanced and scope in ("p", "c", "uc"):
                    df[f"{scope}_buy_click_ratio_14d_s"] = (df[b4_14] + 1.0) / (df[b1_14] + 2.0)
                    df[f"{scope}_buy_cart_ratio_14d_s"] = (df[b4_14] + 1.0) / (df[b3_14] + 2.0)
                    feature_cols.extend(
                        [
                            f"{scope}_buy_click_ratio_14d_s",
                            f"{scope}_buy_cart_ratio_14d_s",
                        ]
                    )

            if all(col in df.columns for col in (b1_1, b2_1, b3_1, b4_1)):
                df[f"{scope}_engage_1d"] = df[b1_1] + 3.0 * df[b2_1] + 6.0 * df[b3_1] + 10.0 * df[b4_1]
                feature_cols.append(f"{scope}_engage_1d")

        if all(col in df.columns for col in ("p_b4_14d", "u_b4_14d", "i_b4_14d")):
            df["p_buy_share_user_14d"] = df["p_b4_14d"] / (df["u_b4_14d"] + 1.0)
            df["p_buy_share_item_14d"] = df["p_b4_14d"] / (df["i_b4_14d"] + 1.0)
            feature_cols.extend(["p_buy_share_user_14d", "p_buy_share_item_14d"])

        if all(col in df.columns for col in ("p_b3_14d", "u_b3_14d", "i_b3_14d")):
            df["p_cart_share_user_14d"] = df["p_b3_14d"] / (df["u_b3_14d"] + 1.0)
            df["p_cart_share_item_14d"] = df["p_b3_14d"] / (df["i_b3_14d"] + 1.0)
            feature_cols.extend(["p_cart_share_user_14d", "p_cart_share_item_14d"])

        if all(col in df.columns for col in ("p_b4_14d", "c_b4_14d", "uc_b4_14d")):
            df["p_buy_share_cat_14d"] = df["p_b4_14d"] / (df["c_b4_14d"] + 1.0)
            df["p_buy_share_uc_14d"] = df["p_b4_14d"] / (df["uc_b4_14d"] + 1.0)
            feature_cols.extend(["p_buy_share_cat_14d", "p_buy_share_uc_14d"])

        if all(col in df.columns for col in ("p_b3_14d", "c_b3_14d", "uc_b3_14d")):
            df["p_cart_share_cat_14d"] = df["p_b3_14d"] / (df["c_b3_14d"] + 1.0)
            df["p_cart_share_uc_14d"] = df["p_b3_14d"] / (df["uc_b3_14d"] + 1.0)
            feature_cols.extend(["p_cart_share_cat_14d", "p_cart_share_uc_14d"])

        if enable_enhanced:
            if all(col in df.columns for col in ("p_last_hours_gap_inv", "p_engage_1d")):
                recency_signal = np.clip(df["p_last_hours_gap_inv"].to_numpy(dtype=np.float32) * 24.0, 0.0, 1.0)
                intent_strength = np.log1p(np.clip(df["p_engage_1d"].to_numpy(dtype=np.float32), 0.0, None))
                df["recency_intent_cross"] = recency_signal * intent_strength
                feature_cols.append("recency_intent_cross")

            if all(col in df.columns for col in ("p_engage_1d", "u_engage_1d")):
                df["pair_share_in_user_1d"] = df["p_engage_1d"] / (df["u_engage_1d"] + 1.0)
                feature_cols.append("pair_share_in_user_1d")

            if all(col in df.columns for col in ("p_engage_1d", "uc_engage_1d")):
                df["pair_share_in_uc_1d"] = df["p_engage_1d"] / (df["uc_engage_1d"] + 1.0)
                feature_cols.append("pair_share_in_uc_1d")

            if all(col in df.columns for col in ("p_b2_1d", "p_b3_1d", "p_b4_1d")):
                strong_intent = (
                    (df["p_b2_1d"].to_numpy(dtype=np.float32) > 0)
                    | (df["p_b3_1d"].to_numpy(dtype=np.float32) > 0)
                ) & (df["p_b4_1d"].to_numpy(dtype=np.float32) <= 0)
                df["strong_intent_flag"] = strong_intent.astype("float32")
                feature_cols.append("strong_intent_flag")

            if all(col in df.columns for col in ("i_b1_14d", "i_b2_14d", "i_b3_14d", "i_b4_14d")):
                item_non_buy_interact = (
                    df["i_b1_14d"].to_numpy(dtype=np.float32)
                    + df["i_b2_14d"].to_numpy(dtype=np.float32)
                    + df["i_b3_14d"].to_numpy(dtype=np.float32)
                )
                zombie = (item_non_buy_interact >= 60.0) & (df["i_b4_14d"].to_numpy(dtype=np.float32) <= 0)
                df["zombie_item_flag"] = zombie.astype("float32")
                feature_cols.append("zombie_item_flag")

        feature_cols.extend(prefixed_raw_cols)

    if "item_category" in df.columns:
        df["item_category"] = pd.to_numeric(df["item_category"], errors="coerce").fillna(-1).astype("float32")
        feature_cols.append("item_category")

    feature_cols = sorted(set(feature_cols))
    feature_frame = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = feature_frame.fillna(0.0).astype("float32")
    return df, feature_cols


def _pair_keys_from_frame(df: pd.DataFrame) -> np.ndarray:
    uid = df["user_id"].to_numpy(dtype=np.uint64, copy=False)
    iid = df["item_id"].to_numpy(dtype=np.uint64, copy=False)
    return (uid << np.uint64(32)) | iid


def build_label_key_array(labels: pd.DataFrame) -> np.ndarray:
    lbl_uid = labels["user_id"].to_numpy(dtype=np.uint64, copy=False)
    lbl_iid = labels["item_id"].to_numpy(dtype=np.uint64, copy=False)
    return (lbl_uid << np.uint64(32)) | lbl_iid


def attach_labels_from_keys(features_df: pd.DataFrame, label_keys: np.ndarray) -> pd.DataFrame:
    feature_keys = _pair_keys_from_frame(features_df)
    features_df["label"] = np.isin(feature_keys, label_keys, assume_unique=False).astype("int8")
    return features_df


def attach_labels(features_df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    # Avoid a full DataFrame merge here, which can double peak memory on large candidate sets.
    label_keys = build_label_key_array(labels)
    return attach_labels_from_keys(features_df, label_keys)


def _normalize_cap(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    val = int(value)
    if val <= 0:
        return None
    return val


def _align_feature_columns(
    frame: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    for col in feature_cols:
        if col not in frame.columns:
            frame[col] = 0.0
    return frame


def sample_training_data_streaming(
    db_path: Path,
    candidate_topk: int,
    label_keys: np.ndarray,
    seed: int,
    read_chunksize: int,
    max_candidates: Optional[int],
    max_positives: Optional[int],
    max_negatives: Optional[int],
    hard_neg_ratio: float,
    enable_enhanced: bool = True,
) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    max_positives = _normalize_cap(max_positives)
    max_negatives = _normalize_cap(max_negatives)
    if max_negatives is None:
        raise ValueError("train-streaming requires --train-max-negatives > 0 to bound memory.")

    hard_neg_ratio = float(np.clip(hard_neg_ratio, 0.0, 1.0))
    hard_budget = int(max_negatives * hard_neg_ratio)
    rand_budget = int(max_negatives - hard_budget)

    hard_chunk_cap = max(1_000, hard_budget // 30) if hard_budget > 0 else 0
    rand_chunk_cap = max(1_000, rand_budget // 30) if rand_budget > 0 else 0

    rng = np.random.default_rng(seed)
    processed = 0
    chunk_no = 0

    positives_seen = 0
    negatives_seen = 0

    feature_cols: List[str] = []
    pos_parts: List[pd.DataFrame] = []
    hard_parts: List[pd.DataFrame] = []
    rand_parts: List[pd.DataFrame] = []

    for chunk in iter_candidates(
        db_path=db_path,
        candidate_topk=candidate_topk,
        chunksize=read_chunksize,
    ):
        chunk_no += 1
        if max_candidates is not None:
            if processed >= int(max_candidates):
                break
            remain = int(max_candidates) - processed
            if len(chunk) > remain:
                chunk = chunk.iloc[:remain].copy()
        if chunk.empty:
            continue

        feat_chunk, chunk_feature_cols = build_features(chunk, enable_enhanced=enable_enhanced)
        if not feature_cols:
            feature_cols = list(chunk_feature_cols)
        else:
            new_cols = [c for c in chunk_feature_cols if c not in feature_cols]
            if new_cols:
                feature_cols.extend(new_cols)
                for df_part in [*pos_parts, *hard_parts, *rand_parts]:
                    for col in new_cols:
                        df_part[col] = 0.0

        feat_chunk = _align_feature_columns(feat_chunk, feature_cols)
        feat_chunk = feat_chunk[["user_id", "item_id", *feature_cols]].copy()
        feat_chunk = attach_labels_from_keys(feat_chunk, label_keys)

        pos_chunk = feat_chunk.loc[feat_chunk["label"] == 1]
        neg_chunk = feat_chunk.loc[feat_chunk["label"] == 0]
        positives_seen += int(len(pos_chunk))
        negatives_seen += int(len(neg_chunk))

        if not pos_chunk.empty:
            pos_parts.append(pos_chunk.copy())

        if not neg_chunk.empty and hard_chunk_cap > 0:
            neg_rn = neg_chunk["rn"].to_numpy(dtype=np.int32, copy=False)
            neg_score = neg_chunk["score"].to_numpy(dtype=np.float32, copy=False)
            order = np.lexsort((-neg_score, neg_rn))
            take = min(len(order), hard_chunk_cap)
            hard_idx = order[:take]
            hard_parts.append(neg_chunk.iloc[hard_idx].copy())

        if not neg_chunk.empty and rand_chunk_cap > 0:
            take = min(len(neg_chunk), rand_chunk_cap)
            rand_parts.append(_sample_stratified_negatives(neg_chunk, take, rng))

        processed += len(chunk)
        if chunk_no % 10 == 0:
            print(
                f"[PROGRESS] stream-train chunks={chunk_no} processed={processed} "
                f"pos_seen={positives_seen} neg_seen={negatives_seen}",
                flush=True,
            )

    pos_df = pd.concat(pos_parts, ignore_index=True) if pos_parts else pd.DataFrame(columns=["label", *feature_cols])
    if max_positives is not None and len(pos_df) > max_positives:
        pos_df = pos_df.sample(n=max_positives, replace=False, random_state=seed).reset_index(drop=True)

    hard_df = pd.concat(hard_parts, ignore_index=True) if hard_parts else pd.DataFrame(columns=["label", *feature_cols])
    if hard_budget > 0 and len(hard_df) > hard_budget:
        rn_arr = hard_df["rn"].to_numpy(dtype=np.int32, copy=False)
        score_arr = hard_df["score"].to_numpy(dtype=np.float32, copy=False)
        order = np.lexsort((-score_arr, rn_arr))
        hard_df = hard_df.iloc[order[:hard_budget]].reset_index(drop=True)

    rand_df = pd.concat(rand_parts, ignore_index=True) if rand_parts else pd.DataFrame(columns=["label", *feature_cols])
    if rand_budget > 0 and len(rand_df) > rand_budget:
        rand_df = _sample_stratified_negatives(rand_df, rand_budget, rng).reset_index(drop=True)

    neg_parts: List[pd.DataFrame] = []
    if not hard_df.empty:
        neg_parts.append(hard_df)
    if not rand_df.empty:
        neg_parts.append(rand_df)
    if neg_parts:
        neg_df = pd.concat(neg_parts, ignore_index=True)
    else:
        neg_df = pd.DataFrame(columns=["label", *feature_cols])
    if len(neg_df) > max_negatives:
        neg_df = neg_df.sample(n=max_negatives, replace=False, random_state=seed + 2).reset_index(drop=True)

    train_df = pd.concat([pos_df, neg_df], ignore_index=True)
    if train_df.empty:
        raise ValueError("No training rows were sampled in streaming mode.")

    train_df["label"] = pd.to_numeric(train_df["label"], errors="coerce").fillna(0).astype("int8")
    train_df = _align_feature_columns(train_df, feature_cols)
    train_df[feature_cols] = (
        train_df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
    )

    shuffled_idx = np.arange(len(train_df), dtype=np.int64)
    rng.shuffle(shuffled_idx)
    train_df = train_df.iloc[shuffled_idx].reset_index(drop=True)

    stats = {
        "stream_processed_rows": int(processed),
        "stream_chunks": int(chunk_no),
        "stream_positives_seen": int(positives_seen),
        "stream_negatives_seen": int(negatives_seen),
        "stream_positives_kept": int(len(pos_df)),
        "stream_negatives_kept": int(len(neg_df)),
    }
    return train_df, feature_cols, stats

def sample_training_data(
    df: pd.DataFrame,
    neg_pos_ratio: int,
    seed: int,
    mode: str = "random",
) -> pd.DataFrame:
    labels = df["label"].to_numpy(dtype=np.int8, copy=False)
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    if len(pos_idx) == 0:
        raise ValueError("No positive samples found in candidate set.")

    sample_neg_n = min(len(neg_idx), len(pos_idx) * int(neg_pos_ratio))
    rng = np.random.default_rng(seed)

    if sample_neg_n <= 0:
        neg_sample_idx = np.empty(0, dtype=np.int64)
    elif mode == "random":
        neg_sample_idx = rng.choice(neg_idx, size=sample_neg_n, replace=False)
    elif mode == "hard" and {"rn", "score"}.issubset(df.columns):
        rn_arr = df["rn"].to_numpy(dtype=np.int32, copy=False)
        score_arr = df["score"].to_numpy(dtype=np.float32, copy=False)
        order = np.lexsort((-score_arr[neg_idx], rn_arr[neg_idx]))
        neg_sample_idx = neg_idx[order[:sample_neg_n]]
    elif mode == "mixed" and {"rn", "score"}.issubset(df.columns):
        hard_n = int(sample_neg_n * 0.7)
        rand_n = sample_neg_n - hard_n
        rn_arr = df["rn"].to_numpy(dtype=np.int32, copy=False)
        score_arr = df["score"].to_numpy(dtype=np.float32, copy=False)
        order = np.lexsort((-score_arr[neg_idx], rn_arr[neg_idx]))
        hard_idx = neg_idx[order[:hard_n]] if hard_n > 0 else np.empty(0, dtype=np.int64)
        remain_idx = neg_idx[order[hard_n:]] if hard_n < len(neg_idx) else np.empty(0, dtype=np.int64)
        if rand_n > 0 and len(remain_idx) > 0:
            rand_pick = rng.choice(remain_idx, size=min(rand_n, len(remain_idx)), replace=False)
            neg_sample_idx = np.concatenate([hard_idx, rand_pick])
        else:
            neg_sample_idx = hard_idx
    else:
        neg_sample_idx = rng.choice(neg_idx, size=sample_neg_n, replace=False)

    train_idx = np.concatenate([pos_idx, neg_sample_idx]).astype(np.int64, copy=False)
    rng.shuffle(train_idx)

    sampled = df.iloc[train_idx].copy()
    sampled.reset_index(drop=True, inplace=True)
    return sampled

def train_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    args: argparse.Namespace,
) -> Tuple[lgb.Booster, str]:
    X = train_df[feature_cols]
    y = train_df["label"].astype("int8")

    rng = np.random.default_rng(args.seed)
    valid_ratio = float(np.clip(args.valid_ratio, 0.01, 0.40))
    train_ratio = float(1.0 - valid_ratio)
    mask = rng.random(len(train_df)) < train_ratio
    train_split = train_df.loc[mask]
    valid_split = train_df.loc[~mask]

    train_set = lgb.Dataset(
        train_split[feature_cols],
        label=train_split["label"],
        feature_name=feature_cols,
        free_raw_data=False,
    )
    use_valid = (
        len(valid_split) > 0
        and train_split["label"].nunique() > 1
        and valid_split["label"].nunique() > 1
    )
    use_early_stopping = bool(args.early_stopping_rounds and int(args.early_stopping_rounds) > 0)

    params_base = {
        "objective": "binary",
        "metric": ["auc"],
        "verbosity": -1,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": args.seed,
        "is_unbalance": True,
    }

    device_plan = []
    if args.device == "auto":
        device_plan = ["gpu", "cpu"]
    elif args.device == "gpu":
        device_plan = ["gpu"]
    else:
        device_plan = ["cpu"]

    last_err: Optional[Exception] = None
    for device in device_plan:
        params = dict(params_base)
        params["device_type"] = device
        try:
            if use_valid:
                valid_set = lgb.Dataset(
                    valid_split[feature_cols],
                    label=valid_split["label"],
                    feature_name=feature_cols,
                    free_raw_data=False,
                )
                booster = lgb.train(
                    params,
                    train_set,
                    num_boost_round=args.num_boost_round,
                    valid_sets=[valid_set],
                    callbacks=(
                        [
                            lgb.early_stopping(
                                stopping_rounds=max(1, int(args.early_stopping_rounds)),
                                verbose=False,
                            ),
                            lgb.log_evaluation(period=max(1, args.log_eval)),
                        ]
                        if use_early_stopping
                        else [lgb.log_evaluation(period=max(1, args.log_eval))]
                    ),
                )
            else:
                booster = lgb.train(
                    params,
                    train_set,
                    num_boost_round=args.num_boost_round,
                    callbacks=[lgb.log_evaluation(period=max(1, args.log_eval))],
                )
            return booster, device
        except Exception as exc:  # pylint: disable=broad-except
            last_err = exc
            if device == device_plan[-1]:
                break
            print(
                f"[WARN] Training with device='{device}' failed, fallback to next device.",
                flush=True,
            )

    raise RuntimeError(f"LightGBM training failed: {last_err}")


def predict_in_batches(
    booster: lgb.Booster,
    features_df: pd.DataFrame,
    feature_cols: List[str],
    batch_size: int,
) -> np.ndarray:
    n = len(features_df)
    out = np.empty(n, dtype=np.float32)
    best_iter = booster.best_iteration if booster.best_iteration else booster.current_iteration()
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        out[start:end] = booster.predict(
            features_df.iloc[start:end][feature_cols],
            num_iteration=best_iter,
        ).astype(np.float32)
    return out


def rank_predictions(features_df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    pred = features_df[["user_id", "item_id"]].copy()
    pred["prob"] = probs
    pred.sort_values(
        ["user_id", "prob", "item_id"],
        ascending=[True, False, True],
        inplace=True,
        kind="mergesort",
    )
    pred["rn"] = pred.groupby("user_id").cumcount() + 1
    return pred


def compute_metrics(
    ranked_pred: pd.DataFrame,
    labels: pd.DataFrame,
    topk_grid: List[int],
) -> Dict[str, Dict[str, float]]:
    labels_key = labels.drop_duplicates().copy()
    labels_key["user_id"] = labels_key["user_id"].astype("int64")
    labels_key["item_id"] = labels_key["item_id"].astype("int64")
    ranked_pred = ranked_pred.copy()
    ranked_pred["user_id"] = ranked_pred["user_id"].astype("int64")
    ranked_pred["item_id"] = ranked_pred["item_id"].astype("int64")
    label_count = len(labels_key)
    metrics: Dict[str, Dict[str, float]] = {}

    for k in topk_grid:
        top = ranked_pred.loc[ranked_pred["rn"] <= k, ["user_id", "item_id"]].drop_duplicates()
        hits = top.merge(labels_key, on=["user_id", "item_id"], how="inner")
        hit_count = len(hits)
        pred_count = len(top)
        precision = hit_count / pred_count if pred_count else 0.0
        recall = hit_count / label_count if label_count else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        metrics[str(k)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": float(pred_count),
            "labels": float(label_count),
            "hits": float(hit_count),
        }
    return metrics


def compute_global_metrics_from_ranked(
    ranked_pairs: List[Tuple[int, int]],
    label_set: set[Tuple[int, int]],
    label_count: int,
    topn_grid: List[int],
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for topn in topn_grid:
        n = min(int(topn), len(ranked_pairs))
        top_slice = ranked_pairs[:n]
        hits = 0
        for uid, iid in top_slice:
            if (uid, iid) in label_set:
                hits += 1
        precision = hits / n if n else 0.0
        recall = hits / label_count if label_count else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        metrics[str(topn)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": float(n),
            "labels": float(label_count),
            "hits": float(hits),
        }
    return metrics


def export_topk(ranked_pred: pd.DataFrame, topk: int, out_file: Path) -> int:
    out = ranked_pred.loc[ranked_pred["rn"] <= topk, ["user_id", "item_id"]].drop_duplicates()
    out.to_csv(out_file, sep="\t", index=False, header=False)
    return len(out)


def export_global_topn_pairs(ranked_pairs: List[Tuple[int, int]], topn: int, out_file: Path) -> int:
    out = pd.DataFrame(ranked_pairs[: int(topn)], columns=["user_id", "item_id"])
    out.to_csv(out_file, sep="\t", index=False, header=False)
    return len(out)


def score_submit_streaming(
    booster: lgb.Booster,
    submit_db: Path,
    candidate_topk: int,
    submit_topk: int,
    feature_cols: List[str],
    predict_batch_size: int,
    read_chunksize: int,
    out_file: Path,
    max_candidates: Optional[int] = None,
    enable_enhanced: bool = True,
) -> int:
    per_user_heap: Dict[int, List[Tuple[float, int, int]]] = {}
    processed = 0
    chunk_no = 0

    for chunk in iter_candidates(
        db_path=submit_db,
        candidate_topk=candidate_topk,
        chunksize=read_chunksize,
    ):
        chunk_no += 1
        if max_candidates is not None:
            if processed >= max_candidates:
                break
            remain = int(max_candidates) - processed
            if len(chunk) > remain:
                chunk = chunk.iloc[:remain].copy()
        if chunk.empty:
            continue

        feat_chunk, _ = build_features(chunk, enable_enhanced=enable_enhanced)
        feat_chunk = _align_feature_columns(feat_chunk, feature_cols)
        probs = predict_in_batches(
            booster=booster,
            features_df=feat_chunk,
            feature_cols=feature_cols,
            batch_size=predict_batch_size,
        )
        users = feat_chunk["user_id"].to_numpy(dtype=np.int64)
        items = feat_chunk["item_id"].to_numpy(dtype=np.int64)

        for uid, iid, prob in zip(users, items, probs, strict=False):
            uid_i = int(uid)
            iid_i = int(iid)
            entry = (float(prob), -iid_i, iid_i)
            heap = per_user_heap.setdefault(uid_i, [])
            if len(heap) < submit_topk:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)

        processed += len(chunk)
        if chunk_no % 10 == 0:
            print(
                f"[PROGRESS] submit-per-user chunks={chunk_no} processed={processed}",
                flush=True,
            )

    rows: List[Tuple[int, int]] = []
    for uid in sorted(per_user_heap.keys()):
        picked = sorted(per_user_heap[uid], key=lambda x: (-x[0], x[2]))
        rows.extend((uid, item_id) for _, _, item_id in picked)

    out_df = pd.DataFrame(rows, columns=["user_id", "item_id"])
    out_df.to_csv(out_file, sep="\t", index=False, header=False)
    return len(out_df)


def score_submit_global_streaming(
    booster: lgb.Booster,
    submit_db: Path,
    candidate_topk: int,
    submit_topn: int,
    feature_cols: List[str],
    predict_batch_size: int,
    read_chunksize: int,
    out_file: Path,
    max_candidates: Optional[int] = None,
    enable_enhanced: bool = True,
) -> Tuple[int, int]:
    heap: List[Tuple[float, int, int, int, int]] = []
    processed = 0
    chunk_no = 0

    for chunk in iter_candidates(
        db_path=submit_db,
        candidate_topk=candidate_topk,
        chunksize=read_chunksize,
    ):
        chunk_no += 1
        if max_candidates is not None:
            if processed >= int(max_candidates):
                break
            remain = int(max_candidates) - processed
            if len(chunk) > remain:
                chunk = chunk.iloc[:remain].copy()
        if chunk.empty:
            continue

        feat_chunk, _ = build_features(chunk, enable_enhanced=enable_enhanced)
        feat_chunk = _align_feature_columns(feat_chunk, feature_cols)
        probs = predict_in_batches(
            booster=booster,
            features_df=feat_chunk,
            feature_cols=feature_cols,
            batch_size=predict_batch_size,
        )
        users = feat_chunk["user_id"].to_numpy(dtype=np.int64)
        items = feat_chunk["item_id"].to_numpy(dtype=np.int64)

        for uid, iid, prob in zip(users, items, probs, strict=False):
            uid_i = int(uid)
            iid_i = int(iid)
            entry = (float(prob), -uid_i, -iid_i, uid_i, iid_i)
            if len(heap) < submit_topn:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)

        processed += len(chunk)
        if chunk_no % 10 == 0:
            print(
                f"[PROGRESS] submit-global chunks={chunk_no} processed={processed}",
                flush=True,
            )

    ranked = sorted(heap, key=lambda x: (-x[0], x[3], x[4]))
    ranked_pairs = [(u, i) for _, _, _, u, i in ranked]
    rows = export_global_topn_pairs(ranked_pairs, submit_topn, out_file)
    return rows, int(processed)


def evaluate_offline_global_streaming(
    booster: lgb.Booster,
    offline_db: Path,
    candidate_topk: int,
    feature_cols: List[str],
    topn_grid: List[int],
    label_set: set[Tuple[int, int]],
    label_count: int,
    predict_batch_size: int,
    read_chunksize: int,
    max_candidates: Optional[int],
    out_file: Path,
    enable_enhanced: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], int, int, int]:
    max_topn = int(max(topn_grid))
    heap: List[Tuple[float, int, int, int, int]] = []
    processed = 0
    chunk_no = 0

    for chunk in iter_candidates(
        db_path=offline_db,
        candidate_topk=candidate_topk,
        chunksize=read_chunksize,
    ):
        chunk_no += 1
        if max_candidates is not None:
            if processed >= int(max_candidates):
                break
            remain = int(max_candidates) - processed
            if len(chunk) > remain:
                chunk = chunk.iloc[:remain].copy()
        if chunk.empty:
            continue

        feat_chunk, _ = build_features(chunk, enable_enhanced=enable_enhanced)
        feat_chunk = _align_feature_columns(feat_chunk, feature_cols)
        probs = predict_in_batches(
            booster=booster,
            features_df=feat_chunk,
            feature_cols=feature_cols,
            batch_size=predict_batch_size,
        )
        users = feat_chunk["user_id"].to_numpy(dtype=np.int64)
        items = feat_chunk["item_id"].to_numpy(dtype=np.int64)

        for uid, iid, prob in zip(users, items, probs, strict=False):
            uid_i = int(uid)
            iid_i = int(iid)
            entry = (float(prob), -uid_i, -iid_i, uid_i, iid_i)
            if len(heap) < max_topn:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)

        processed += len(chunk)
        if chunk_no % 10 == 0:
            print(
                f"[PROGRESS] offline-global chunks={chunk_no} processed={processed}",
                flush=True,
            )

    ranked = sorted(heap, key=lambda x: (-x[0], x[3], x[4]))
    ranked_pairs = [(u, i) for _, _, _, u, i in ranked]
    metrics = compute_global_metrics_from_ranked(ranked_pairs, label_set, label_count, topn_grid)
    best_topn = int(max(metrics.items(), key=lambda kv: kv[1]["f1"])[0])
    offline_rows = export_global_topn_pairs(ranked_pairs, best_topn, out_file)
    return metrics, best_topn, int(offline_rows), int(processed)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    topk_grid = parse_topk_grid(args.topk_grid)
    use_enhanced_features = not bool(args.disable_enhanced_features)

    labels = read_labels(args.offline_db)
    labels = labels.drop_duplicates().copy()
    labels["user_id"] = labels["user_id"].astype("int64")
    labels["item_id"] = labels["item_id"].astype("int64")
    label_keys = build_label_key_array(labels)
    label_set = set((int(u), int(i)) for u, i in labels[["user_id", "item_id"]].itertuples(index=False, name=None))

    training_stats: Dict[str, int] = {}
    offline_candidate_count: Optional[int] = None
    pos_seen: Optional[int] = None
    feature_cols: List[str]

    if args.train_streaming:
        print("[STEP] Streaming sample for training...", flush=True)
        train_df, feature_cols, training_stats = sample_training_data_streaming(
            db_path=args.offline_db,
            candidate_topk=args.candidate_topk,
            label_keys=label_keys,
            seed=args.seed,
            read_chunksize=args.train_read_chunksize,
            max_candidates=args.max_candidates,
            max_positives=args.train_max_positives,
            max_negatives=args.train_max_negatives,
            hard_neg_ratio=args.train_hard_neg_ratio,
            enable_enhanced=use_enhanced_features,
        )
        offline_candidate_count = int(training_stats["stream_processed_rows"])
        pos_seen = int(training_stats["stream_positives_seen"])
    else:
        print("[STEP] Loading offline candidates...", flush=True)
        offline_candidates = read_candidates(
            db_path=args.offline_db,
            candidate_topk=args.candidate_topk,
            max_candidates=args.max_candidates,
        )
        offline_candidate_count = int(len(offline_candidates))
        print(f"[INFO] offline candidate rows: {offline_candidate_count}", flush=True)

        print("[STEP] Building offline features...", flush=True)
        offline_feat, feature_cols = build_features(
            offline_candidates,
            enable_enhanced=use_enhanced_features,
        )
        del offline_candidates
        gc.collect()
        offline_feat = attach_labels_from_keys(offline_feat, label_keys)
        pos_seen = int(offline_feat["label"].sum())
        print(f"[INFO] offline positives in candidates: {pos_seen}", flush=True)

        print("[STEP] Sampling training data...", flush=True)
        train_df = sample_training_data(
            df=offline_feat,
            neg_pos_ratio=args.neg_pos_ratio,
            seed=args.seed,
            mode=args.neg_sample_mode,
        )
        del offline_feat
        gc.collect()

    train_rows_count = int(len(train_df))
    feature_non_finite_count = _count_non_finite(train_df, feature_cols)
    if feature_non_finite_count > 0:
        raise ValueError(
            f"Detected non-finite feature values before training: {feature_non_finite_count}"
        )
    print(
        f"[INFO] training rows: {train_rows_count} | positives: {int(train_df['label'].sum())}",
        flush=True,
    )

    print("[STEP] Training LightGBM...", flush=True)
    booster, used_device = train_model(train_df, feature_cols, args)
    del train_df
    gc.collect()

    model_file = args.output_dir / "lgbm_rerank_model.txt"
    booster.save_model(str(model_file))
    print(f"[INFO] model saved: {model_file} | device: {used_device}", flush=True)

    model_feature_cols = booster.feature_name()
    if model_feature_cols:
        feature_cols = list(model_feature_cols)

    metric_protocol = "global_topn" if args.offline_eval_mode == "global" else "per_user_topk"
    offline_pred_file = args.output_dir / "offline_rerank_predictions.tsv"
    submit_file = args.output_dir / f"submit_rerank_{datetime.now().strftime('%Y%m%d_%H%M')}.tsv"

    if args.offline_eval_mode == "global":
        print("[STEP] Scoring offline candidates (global-topN streaming)...", flush=True)
        metrics, best_topn, offline_rows, offline_eval_processed = evaluate_offline_global_streaming(
            booster=booster,
            offline_db=args.offline_db,
            candidate_topk=args.candidate_topk,
            feature_cols=feature_cols,
            topn_grid=topk_grid,
            label_set=label_set,
            label_count=len(label_set),
            predict_batch_size=args.predict_batch_size,
            read_chunksize=args.train_read_chunksize,
            max_candidates=args.max_candidates,
            out_file=offline_pred_file,
            enable_enhanced=use_enhanced_features,
        )

        submit_topn = int(args.submit_topk) if args.submit_topk is not None else int(best_topn)
        print("[STEP] Scoring submit candidates (global-topN streaming)...", flush=True)
        submit_rows, submit_processed = score_submit_global_streaming(
            booster=booster,
            submit_db=args.submit_db,
            candidate_topk=args.candidate_topk,
            submit_topn=submit_topn,
            feature_cols=feature_cols,
            predict_batch_size=args.predict_batch_size,
            read_chunksize=args.submit_read_chunksize,
            out_file=submit_file,
            max_candidates=args.max_candidates,
            enable_enhanced=use_enhanced_features,
        )
    else:
        print("[STEP] Loading offline candidates for per-user evaluation...", flush=True)
        offline_candidates = read_candidates(
            db_path=args.offline_db,
            candidate_topk=args.candidate_topk,
            max_candidates=args.max_candidates,
        )
        offline_feat, _ = build_features(
            offline_candidates,
            enable_enhanced=use_enhanced_features,
        )
        offline_feat = _align_feature_columns(offline_feat, feature_cols)
        offline_probs = predict_in_batches(
            booster=booster,
            features_df=offline_feat,
            feature_cols=feature_cols,
            batch_size=args.predict_batch_size,
        )
        offline_ranked = rank_predictions(offline_feat, offline_probs)
        metrics = compute_metrics(offline_ranked, labels, topk_grid=topk_grid)
        best_topn = int(max(metrics.items(), key=lambda kv: kv[1]["f1"])[0])
        offline_rows = export_topk(offline_ranked, topk=best_topn, out_file=offline_pred_file)
        submit_topn = int(args.submit_topk) if args.submit_topk is not None else int(best_topn)

        print("[STEP] Scoring submit candidates (per-user streaming)...", flush=True)
        submit_rows = score_submit_streaming(
            booster=booster,
            submit_db=args.submit_db,
            candidate_topk=args.candidate_topk,
            submit_topk=submit_topn,
            feature_cols=feature_cols,
            predict_batch_size=args.predict_batch_size,
            read_chunksize=args.submit_read_chunksize,
            out_file=submit_file,
            max_candidates=args.max_candidates,
            enable_enhanced=use_enhanced_features,
        )
        submit_processed = None
        offline_eval_processed = int(len(offline_candidates))

    summary = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metric_protocol": metric_protocol,
        "offline_db": str(args.offline_db),
        "submit_db": str(args.submit_db),
        "candidate_topk": args.candidate_topk,
        "topk_grid": topk_grid,
        "submit_topk": int(submit_topn),
        "device_requested": args.device,
        "device_used": used_device,
        "valid_ratio": float(args.valid_ratio),
        "early_stopping_rounds": int(args.early_stopping_rounds),
        "best_iteration": int(booster.best_iteration) if booster.best_iteration else None,
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "enhanced_features_enabled": bool(use_enhanced_features),
        "feature_non_finite_count": int(feature_non_finite_count),
        "neg_sample_mode": args.neg_sample_mode,
        "train_streaming": bool(args.train_streaming),
        "train_max_positives": args.train_max_positives,
        "train_max_negatives": args.train_max_negatives,
        "train_hard_neg_ratio": args.train_hard_neg_ratio,
        "offline_candidates": int(offline_candidate_count) if offline_candidate_count is not None else None,
        "offline_positives_in_candidates": int(pos_seen) if pos_seen is not None else None,
        "train_rows": int(train_rows_count),
        "streaming_sampler": training_stats,
        "offline_best_topk": int(best_topn),
        "offline_metrics_by_topk": metrics,
        "offline_eval_processed_rows": int(offline_eval_processed),
        "offline_prediction_rows": int(offline_rows),
        "offline_prediction_file": str(offline_pred_file),
        "submit_processed_rows": int(submit_processed) if submit_processed is not None else None,
        "submit_prediction_rows": int(submit_rows),
        "submit_prediction_file": str(submit_file),
        "model_file": str(model_file),
    }

    if args.offline_eval_mode == "global":
        summary["offline_best_topn"] = int(best_topn)
        summary["submit_topn"] = int(submit_topn)

    summary_file = args.output_dir / "rerank_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] rerank done. summary: {summary_file}", flush=True)


if __name__ == "__main__":
    main()






