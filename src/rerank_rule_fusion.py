from __future__ import annotations

import argparse
import heapq
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.common import ensure_dir
from src.rerank_lgbm import (
    build_features,
    compute_metrics,
    iter_candidates,
    parse_topk_grid,
    predict_in_batches,
    rank_predictions,
    read_candidates,
    read_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rule + model fusion rerank for Tianchi mobile recommendation."
    )
    parser.add_argument(
        "--offline-db",
        type=Path,
        default=Path("outputs/baseline_v2/offline_baseline_work.db"),
    )
    parser.add_argument(
        "--submit-db",
        type=Path,
        default=Path("outputs/baseline_v2/submit_baseline_work.db"),
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=Path("outputs/rerank_v2/lgbm_rerank_model.txt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rerank_fusion"))
    parser.add_argument("--candidate-topk", type=int, default=50)
    parser.add_argument("--per-user-topk-grid", type=str, default="5,10,20,30")
    parser.add_argument(
        "--global-topn-grid",
        type=str,
        default="900,2000,5000,10000,20000,50000,100000,200000,500000",
    )
    parser.add_argument("--submit-strategy", choices=["auto", "per_user", "global"], default="auto")
    parser.add_argument("--submit-per-user-topk", type=int, default=None)
    parser.add_argument("--submit-global-topn", type=int, default=None)
    parser.add_argument("--predict-batch-size", type=int, default=2_000_000)
    parser.add_argument("--submit-read-chunksize", type=int, default=1_000_000)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--zombie-click-threshold", type=float, default=60.0)
    parser.add_argument("--zombie-penalty", type=float, default=0.05)
    parser.add_argument("--strong-bonus", type=float, default=0.25)
    parser.add_argument("--loyalty-weight", type=float, default=0.50)
    parser.add_argument("--cvr-weight", type=float, default=0.25)
    parser.add_argument("--recency-weight", type=float, default=0.10)
    return parser.parse_args()


def _parse_int_grid(value: str) -> List[int]:
    arr = []
    for p in value.split(","):
        p = p.strip()
        if not p:
            continue
        arr.append(int(p))
    arr = sorted(set(arr))
    if not arr:
        raise ValueError("grid cannot be empty")
    return arr


def compute_hybrid_score(df: pd.DataFrame, model_prob: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    score = model_prob.astype(np.float32).copy()
    eps = 1e-6

    if all(c in df.columns for c in ("p_b2_1d", "p_b3_1d", "p_b4_1d")):
        strong = (
            (df["p_b2_1d"].to_numpy() > 0)
            | (df["p_b3_1d"].to_numpy() > 0)
            | (df["p_b4_1d"].to_numpy() > 0)
        )
        score *= (1.0 + float(args.strong_bonus) * strong.astype(np.float32))

    if all(c in df.columns for c in ("p_engage_1d", "u_engage_1d")):
        loyalty = df["p_engage_1d"].to_numpy(dtype=np.float32) / (
            df["u_engage_1d"].to_numpy(dtype=np.float32) + 1.0
        )
        loyalty = np.clip(loyalty, 0.0, 1.5)
        score *= (1.0 + float(args.loyalty_weight) * loyalty)

    if all(c in df.columns for c in ("i_b4_14d", "i_b1_14d")):
        item_cvr = df["i_b4_14d"].to_numpy(dtype=np.float32) / (
            df["i_b1_14d"].to_numpy(dtype=np.float32) + 1.0
        )
        item_cvr = np.clip(item_cvr, 0.0, 1.0)
        score *= (1.0 + float(args.cvr_weight) * item_cvr)

    if "p_last_hours_gap_inv" in df.columns:
        recency = np.clip(df["p_last_hours_gap_inv"].to_numpy(dtype=np.float32) * 24.0, 0.0, 1.0)
        score *= (1.0 + float(args.recency_weight) * recency)

    if all(c in df.columns for c in ("i_b1_14d", "i_b4_14d")):
        zombie = (df["i_b1_14d"].to_numpy(dtype=np.float32) >= float(args.zombie_click_threshold)) & (
            df["i_b4_14d"].to_numpy(dtype=np.float32) <= eps
        )
        score *= np.where(zombie, float(args.zombie_penalty), 1.0).astype(np.float32)

    return score


def compute_global_metrics(
    scored_df: pd.DataFrame,
    labels: pd.DataFrame,
    topn_grid: List[int],
    score_col: str = "score_fusion",
) -> Dict[str, Dict[str, float]]:
    labels_key = labels[["user_id", "item_id"]].drop_duplicates().copy()
    labels_key["user_id"] = labels_key["user_id"].astype("int64")
    labels_key["item_id"] = labels_key["item_id"].astype("int64")
    labels_count = len(labels_key)

    sort_df = scored_df[["user_id", "item_id", score_col]].copy()
    sort_df.sort_values(
        [score_col, "user_id", "item_id"],
        ascending=[False, True, True],
        inplace=True,
        kind="mergesort",
    )
    sort_df = sort_df[["user_id", "item_id"]]

    metrics: Dict[str, Dict[str, float]] = {}
    for n in topn_grid:
        top = sort_df.head(int(n)).drop_duplicates()
        hits = top.merge(labels_key, on=["user_id", "item_id"], how="inner")
        hit_count = len(hits)
        pred_count = len(top)
        precision = hit_count / pred_count if pred_count else 0.0
        recall = hit_count / labels_count if labels_count else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        metrics[str(n)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": float(pred_count),
            "labels": float(labels_count),
            "hits": float(hit_count),
        }
    return metrics


def export_per_user_topk(scored_df: pd.DataFrame, topk: int, output: Path, score_col: str = "score_fusion") -> int:
    rank_df = scored_df[["user_id", "item_id", score_col]].copy()
    rank_df.rename(columns={score_col: "prob"}, inplace=True)
    rank_df.sort_values(
        ["user_id", "prob", "item_id"],
        ascending=[True, False, True],
        inplace=True,
        kind="mergesort",
    )
    rank_df["rn"] = rank_df.groupby("user_id").cumcount() + 1
    out = rank_df.loc[rank_df["rn"] <= int(topk), ["user_id", "item_id"]].drop_duplicates()
    out.to_csv(output, sep="\t", index=False, header=False)
    return len(out)


def export_global_topn(scored_df: pd.DataFrame, topn: int, output: Path, score_col: str = "score_fusion") -> int:
    out = scored_df[["user_id", "item_id", score_col]].copy()
    out.sort_values(
        [score_col, "user_id", "item_id"],
        ascending=[False, True, True],
        inplace=True,
        kind="mergesort",
    )
    out = out.head(int(topn))[["user_id", "item_id"]].drop_duplicates()
    out.to_csv(output, sep="\t", index=False, header=False)
    return len(out)


def stream_submit_per_user(
    booster: lgb.Booster,
    submit_db: Path,
    candidate_topk: int,
    submit_topk: int,
    feature_cols: List[str],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, int]:
    heaps: Dict[int, List[Tuple[float, int, int]]] = {}
    processed = 0
    chunk_no = 0

    for chunk in iter_candidates(submit_db, candidate_topk, args.submit_read_chunksize):
        chunk_no += 1
        if args.max_candidates is not None:
            if processed >= int(args.max_candidates):
                break
            remain = int(args.max_candidates) - processed
            if len(chunk) > remain:
                chunk = chunk.iloc[:remain].copy()

        if chunk.empty:
            continue

        feat_chunk, _ = build_features(chunk)
        probs = predict_in_batches(booster, feat_chunk, feature_cols, args.predict_batch_size)
        fusion = compute_hybrid_score(feat_chunk, probs, args)
        users = feat_chunk["user_id"].to_numpy(dtype=np.int64)
        items = feat_chunk["item_id"].to_numpy(dtype=np.int64)
        for uid, iid, s in zip(users, items, fusion, strict=False):
            uid_i = int(uid)
            iid_i = int(iid)
            entry = (float(s), -iid_i, iid_i)
            heap = heaps.setdefault(uid_i, [])
            if len(heap) < submit_topk:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)

        processed += len(chunk)
        if chunk_no % 10 == 0:
            print(f"[PROGRESS] submit-per-user chunks={chunk_no} processed={processed}", flush=True)

    rows: List[Tuple[int, int, float]] = []
    for uid in sorted(heaps.keys()):
        picked = sorted(heaps[uid], key=lambda x: (-x[0], x[2]))
        rows.extend((uid, item_id, score) for score, _, item_id in picked)
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "score_fusion"])
    return df, len(df)


def stream_submit_global(
    booster: lgb.Booster,
    submit_db: Path,
    candidate_topk: int,
    submit_topn: int,
    feature_cols: List[str],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, int]:
    heap: List[Tuple[float, int, int, int, int]] = []
    processed = 0
    chunk_no = 0

    for chunk in iter_candidates(submit_db, candidate_topk, args.submit_read_chunksize):
        chunk_no += 1
        if args.max_candidates is not None:
            if processed >= int(args.max_candidates):
                break
            remain = int(args.max_candidates) - processed
            if len(chunk) > remain:
                chunk = chunk.iloc[:remain].copy()

        if chunk.empty:
            continue

        feat_chunk, _ = build_features(chunk)
        probs = predict_in_batches(booster, feat_chunk, feature_cols, args.predict_batch_size)
        fusion = compute_hybrid_score(feat_chunk, probs, args)
        users = feat_chunk["user_id"].to_numpy(dtype=np.int64)
        items = feat_chunk["item_id"].to_numpy(dtype=np.int64)

        for uid, iid, s in zip(users, items, fusion, strict=False):
            uid_i = int(uid)
            iid_i = int(iid)
            entry = (float(s), -uid_i, -iid_i, uid_i, iid_i)
            if len(heap) < submit_topn:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)

        processed += len(chunk)
        if chunk_no % 10 == 0:
            print(f"[PROGRESS] submit-global chunks={chunk_no} processed={processed}", flush=True)

    heap_sorted = sorted(heap, key=lambda x: (-x[0], x[3], x[4]))
    rows = [(u, i, s) for s, _, _, u, i in heap_sorted]
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "score_fusion"])
    return df, len(df)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    if not args.model_file.exists():
        raise FileNotFoundError(f"model file not found: {args.model_file}")

    per_user_grid = parse_topk_grid(args.per_user_topk_grid)
    global_grid = _parse_int_grid(args.global_topn_grid)

    print("[STEP] Loading offline candidates...", flush=True)
    offline_candidates = read_candidates(
        db_path=args.offline_db,
        candidate_topk=args.candidate_topk,
        max_candidates=args.max_candidates,
    )
    print(f"[INFO] offline candidate rows: {len(offline_candidates)}", flush=True)

    print("[STEP] Building offline features...", flush=True)
    offline_feat, feature_cols = build_features(offline_candidates)
    booster = lgb.Booster(model_file=str(args.model_file))
    model_feature_cols = booster.feature_name()
    if model_feature_cols:
        for col in model_feature_cols:
            if col not in offline_feat.columns:
                offline_feat[col] = 0.0
        feature_cols = model_feature_cols

    print("[STEP] Predicting offline scores...", flush=True)
    model_probs = predict_in_batches(
        booster=booster,
        features_df=offline_feat,
        feature_cols=feature_cols,
        batch_size=args.predict_batch_size,
    )
    fusion_scores = compute_hybrid_score(offline_feat, model_probs, args)
    scored_df = offline_feat[["user_id", "item_id"]].copy()
    scored_df["score_model"] = model_probs.astype(np.float32)
    scored_df["score_fusion"] = fusion_scores.astype(np.float32)

    labels = read_labels(args.offline_db)

    print("[STEP] Evaluating per-user topk...", flush=True)
    rank_df = rank_predictions(
        features_df=scored_df.rename(columns={"score_fusion": "prob"}),
        probs=scored_df["score_fusion"].to_numpy(dtype=np.float32),
    )
    per_user_metrics = compute_metrics(rank_df, labels, topk_grid=per_user_grid)
    best_per_user_topk, best_per_user = max(per_user_metrics.items(), key=lambda kv: kv[1]["f1"])

    print("[STEP] Evaluating global topN...", flush=True)
    global_metrics = compute_global_metrics(scored_df, labels, global_grid, score_col="score_fusion")
    best_global_topn, best_global = max(global_metrics.items(), key=lambda kv: kv[1]["f1"])

    if args.submit_strategy == "per_user":
        chosen_strategy = "per_user"
    elif args.submit_strategy == "global":
        chosen_strategy = "global"
    else:
        chosen_strategy = "per_user" if best_per_user["f1"] >= best_global["f1"] else "global"

    offline_best_file = args.output_dir / "offline_best_predictions.tsv"
    if chosen_strategy == "per_user":
        chosen_per_user_topk = int(args.submit_per_user_topk) if args.submit_per_user_topk else int(best_per_user_topk)
        offline_rows = export_per_user_topk(scored_df, chosen_per_user_topk, offline_best_file, "score_fusion")
        chosen_value = chosen_per_user_topk
    else:
        chosen_global_topn = int(args.submit_global_topn) if args.submit_global_topn else int(best_global_topn)
        offline_rows = export_global_topn(scored_df, chosen_global_topn, offline_best_file, "score_fusion")
        chosen_value = chosen_global_topn

    submit_file = args.output_dir / f"submit_fusion_{datetime.now().strftime('%Y%m%d_%H%M')}.tsv"
    print(f"[STEP] Scoring submit in strategy={chosen_strategy}...", flush=True)
    if chosen_strategy == "per_user":
        submit_topk = int(args.submit_per_user_topk) if args.submit_per_user_topk else int(best_per_user_topk)
        submit_df, _ = stream_submit_per_user(
            booster=booster,
            submit_db=args.submit_db,
            candidate_topk=args.candidate_topk,
            submit_topk=submit_topk,
            feature_cols=feature_cols,
            args=args,
        )
        submit_df[["user_id", "item_id"]].to_csv(submit_file, sep="\t", index=False, header=False)
        submit_rows = len(submit_df)
    else:
        submit_topn = int(args.submit_global_topn) if args.submit_global_topn else int(best_global_topn)
        submit_df, _ = stream_submit_global(
            booster=booster,
            submit_db=args.submit_db,
            candidate_topk=args.candidate_topk,
            submit_topn=submit_topn,
            feature_cols=feature_cols,
            args=args,
        )
        submit_df[["user_id", "item_id"]].to_csv(submit_file, sep="\t", index=False, header=False)
        submit_rows = len(submit_df)

    summary = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "offline_db": str(args.offline_db),
        "submit_db": str(args.submit_db),
        "model_file": str(args.model_file),
        "candidate_topk": args.candidate_topk,
        "feature_count": len(feature_cols),
        "per_user_topk_grid": per_user_grid,
        "global_topn_grid": global_grid,
        "best_per_user_topk": int(best_per_user_topk),
        "best_per_user_metrics": best_per_user,
        "best_global_topn": int(best_global_topn),
        "best_global_metrics": best_global,
        "per_user_metrics": per_user_metrics,
        "global_metrics": global_metrics,
        "chosen_strategy": chosen_strategy,
        "chosen_value": int(chosen_value),
        "offline_best_rows": int(offline_rows),
        "offline_best_file": str(offline_best_file),
        "submit_rows": int(submit_rows),
        "submit_file": str(submit_file),
        "rule_params": {
            "zombie_click_threshold": args.zombie_click_threshold,
            "zombie_penalty": args.zombie_penalty,
            "strong_bonus": args.strong_bonus,
            "loyalty_weight": args.loyalty_weight,
            "cvr_weight": args.cvr_weight,
            "recency_weight": args.recency_weight,
        },
    }
    summary_file = args.output_dir / "fusion_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] fusion done. summary: {summary_file}", flush=True)


if __name__ == "__main__":
    main()
