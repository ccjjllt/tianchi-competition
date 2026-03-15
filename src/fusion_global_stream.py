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
from src.rerank_lgbm import build_features, iter_candidates, parse_topk_grid, predict_in_batches, read_labels
from src.rerank_rule_fusion import compute_hybrid_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming global TopN search for Tianchi mobile recommendation."
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
    parser.add_argument(
        "--model-file-2",
        type=Path,
        default=None,
        help="Optional second model for linear score ensembling.",
    )
    parser.add_argument(
        "--model2-weight",
        type=float,
        default=0.0,
        help="Weight for model-file-2 in blended prediction: pred=(1-w)*m1 + w*m2.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fusion_stream"))
    parser.add_argument("--candidate-topk-grid", type=str, default="50,80,120")
    parser.add_argument(
        "--global-topn-grid",
        type=str,
        default="50000,80000,100000,120000,150000,200000,300000,500000",
    )
    parser.add_argument("--predict-batch-size", type=int, default=2_000_000)
    parser.add_argument("--read-chunksize", type=int, default=1_000_000)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--skip-submit", action="store_true")
    parser.add_argument(
        "--global-user-cap",
        type=int,
        default=0,
        help="Max items per user in global TopN selection. 0 means unlimited.",
    )
    parser.add_argument("--zombie-click-threshold", type=float, default=60.0)
    parser.add_argument("--zombie-penalty", type=float, default=0.05)
    parser.add_argument("--strong-bonus", type=float, default=0.25)
    parser.add_argument("--loyalty-weight", type=float, default=0.50)
    parser.add_argument("--cvr-weight", type=float, default=0.25)
    parser.add_argument("--recency-weight", type=float, default=0.10)
    parser.add_argument("--metric-protocol", type=str, default="global_topn")
    parser.add_argument("--lookback-days", type=int, default=None)
    return parser.parse_args()


def parse_int_grid(value: str) -> List[int]:
    vals = []
    for p in value.split(","):
        p = p.strip()
        if not p:
            continue
        vals.append(int(p))
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("grid cannot be empty")
    return vals


def stream_top_heap(
    db_path: Path,
    candidate_topk: int,
    booster: lgb.Booster,
    feature_cols: List[str],
    booster2: Optional[lgb.Booster],
    feature_cols2: List[str],
    model2_weight: float,
    max_topn: int,
    args: argparse.Namespace,
) -> Tuple[List[Tuple[float, int, int, int, int]], int]:
    heap: List[Tuple[float, int, int, int, int]] = []
    processed = 0
    chunk_no = 0
    model2_weight = float(np.clip(model2_weight, 0.0, 1.0))
    for chunk in iter_candidates(db_path, candidate_topk, args.read_chunksize):
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
        for col in feature_cols:
            if col not in feat_chunk.columns:
                feat_chunk[col] = 0.0
        probs1 = predict_in_batches(
            booster=booster,
            features_df=feat_chunk,
            feature_cols=feature_cols,
            batch_size=args.predict_batch_size,
        )
        probs = probs1
        if booster2 is not None and model2_weight > 0.0:
            for col in feature_cols2:
                if col not in feat_chunk.columns:
                    feat_chunk[col] = 0.0
            probs2 = predict_in_batches(
                booster=booster2,
                features_df=feat_chunk,
                feature_cols=feature_cols2,
                batch_size=args.predict_batch_size,
            )
            probs = ((1.0 - model2_weight) * probs1 + model2_weight * probs2).astype("float32", copy=False)
        fusion_scores = compute_hybrid_score(feat_chunk, probs, args)
        users = feat_chunk["user_id"].to_numpy(dtype="int64")
        items = feat_chunk["item_id"].to_numpy(dtype="int64")
        for uid, iid, s in zip(users, items, fusion_scores, strict=False):
            uid_i = int(uid)
            iid_i = int(iid)
            entry = (float(s), -uid_i, -iid_i, uid_i, iid_i)
            if len(heap) < max_topn:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
        processed += len(chunk)
        if chunk_no % 10 == 0:
            print(
                f"[PROGRESS] topk={candidate_topk} chunks={chunk_no} processed={processed}",
                flush=True,
            )
    return heap, processed


def eval_topn_from_heap(
    heap: List[Tuple[float, int, int, int, int]],
    labels_set: set[Tuple[int, int]],
    label_count: int,
    topn_grid: List[int],
    user_cap: int = 0,
) -> Dict[str, Dict[str, float]]:
    ranked = sorted(heap, key=lambda x: (-x[0], x[3], x[4]))

    metrics: Dict[str, Dict[str, float]] = {}
    for n in topn_grid:
        picked = 0
        hits = 0
        user_cnt: Dict[int, int] = {}
        for _, _, _, uid, iid in ranked:
            if user_cap > 0:
                c = user_cnt.get(uid, 0)
                if c >= user_cap:
                    continue
                user_cnt[uid] = c + 1
            picked += 1
            if (uid, iid) in labels_set:
                hits += 1
            if picked >= int(n):
                break
        n2 = picked
        precision = hits / n2 if n2 else 0.0
        recall = hits / label_count if label_count else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        metrics[str(n)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": float(n2),
            "labels": float(label_count),
            "hits": float(hits),
        }
    return metrics


def export_heap_topn(
    heap: List[Tuple[float, int, int, int, int]],
    topn: int,
    output: Path,
    user_cap: int = 0,
) -> int:
    ranked_all = sorted(heap, key=lambda x: (-x[0], x[3], x[4]))
    picked: List[Tuple[int, int]] = []
    user_cnt: Dict[int, int] = {}
    for _, _, _, u, i in ranked_all:
        if user_cap > 0:
            c = user_cnt.get(u, 0)
            if c >= user_cap:
                continue
            user_cnt[u] = c + 1
        picked.append((u, i))
        if len(picked) >= int(topn):
            break
    out = pd.DataFrame(picked, columns=["user_id", "item_id"])
    out.to_csv(output, sep="\t", index=False, header=False)
    return len(out)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    if not args.model_file.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_file}")
    model2_weight = float(np.clip(args.model2_weight, 0.0, 1.0))
    if args.model_file_2 is not None and model2_weight > 0.0 and not args.model_file_2.exists():
        raise FileNotFoundError(f"Model file 2 not found: {args.model_file_2}")

    candidate_topk_grid = parse_int_grid(args.candidate_topk_grid)
    topn_grid = parse_int_grid(args.global_topn_grid)
    max_topn = max(topn_grid)

    booster = lgb.Booster(model_file=str(args.model_file))
    feature_cols = booster.feature_name()
    booster2: Optional[lgb.Booster] = None
    feature_cols2: List[str] = []
    if args.model_file_2 is not None and model2_weight > 0.0:
        booster2 = lgb.Booster(model_file=str(args.model_file_2))
        feature_cols2 = booster2.feature_name()
        print(
            f"[INFO] model ensemble enabled: m1={args.model_file} m2={args.model_file_2} w2={model2_weight:.3f}",
            flush=True,
        )

    labels = read_labels(args.offline_db)
    labels["user_id"] = labels["user_id"].astype("int64")
    labels["item_id"] = labels["item_id"].astype("int64")
    labels_set = set((int(u), int(i)) for u, i in labels[["user_id", "item_id"]].itertuples(index=False, name=None))
    label_count = len(labels_set)

    all_results: Dict[str, Dict[str, object]] = {}
    best_choice: Optional[Tuple[int, int, Dict[str, float]]] = None

    print("[STEP] Offline streaming search...", flush=True)
    for c_topk in candidate_topk_grid:
        heap, processed = stream_top_heap(
            db_path=args.offline_db,
            candidate_topk=c_topk,
            booster=booster,
            feature_cols=feature_cols,
            booster2=booster2,
            feature_cols2=feature_cols2,
            model2_weight=model2_weight,
            max_topn=max_topn,
            args=args,
        )
        metrics = eval_topn_from_heap(
            heap,
            labels_set,
            label_count,
            topn_grid,
            user_cap=int(args.global_user_cap),
        )
        best_topn_for_k, best_m = max(metrics.items(), key=lambda kv: kv[1]["f1"])
        all_results[str(c_topk)] = {
            "processed_rows": int(processed),
            "best_topn": int(best_topn_for_k),
            "best_metrics": best_m,
            "metrics_by_topn": metrics,
        }
        print(
            f"[OFFLINE] candidate_topk={c_topk} best_topn={best_topn_for_k} f1={best_m['f1']:.6f}",
            flush=True,
        )
        if best_choice is None or best_m["f1"] > best_choice[2]["f1"]:
            best_choice = (c_topk, int(best_topn_for_k), best_m)

    if best_choice is None:
        raise RuntimeError("No valid choice from offline search.")

    best_c_topk, best_topn, best_metrics = best_choice
    print(
        f"[BEST] candidate_topk={best_c_topk} topn={best_topn} "
        f"f1={best_metrics['f1']:.6f} p={best_metrics['precision']:.6f} r={best_metrics['recall']:.6f}",
        flush=True,
    )

    submit_processed = 0
    submit_rows = 0
    submit_file = None
    if not args.skip_submit:
        print("[STEP] Submit streaming export...", flush=True)
        submit_heap, submit_processed = stream_top_heap(
            db_path=args.submit_db,
            candidate_topk=best_c_topk,
            booster=booster,
            feature_cols=feature_cols,
            booster2=booster2,
            feature_cols2=feature_cols2,
            model2_weight=model2_weight,
            max_topn=best_topn,
            args=args,
        )
        submit_file = args.output_dir / f"submit_stream_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        submit_rows = export_heap_topn(
            submit_heap,
            best_topn,
            submit_file,
            user_cap=int(args.global_user_cap),
        )

    summary = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metric_protocol": args.metric_protocol,
        "lookback_days": args.lookback_days,
        "offline_db": str(args.offline_db),
        "submit_db": str(args.submit_db),
        "model_file": str(args.model_file),
        "model_file_2": str(args.model_file_2) if args.model_file_2 is not None else None,
        "model2_weight": model2_weight,
        "candidate_topk_grid": candidate_topk_grid,
        "global_topn_grid": topn_grid,
        "search_results": all_results,
        "best_candidate_topk": int(best_c_topk),
        "best_topn": int(best_topn),
        "best_metrics": best_metrics,
        "submit_processed_rows": int(submit_processed),
        "submit_rows": int(submit_rows),
        "submit_file": str(submit_file) if submit_file is not None else None,
        "rule_params": {
            "zombie_click_threshold": args.zombie_click_threshold,
            "zombie_penalty": args.zombie_penalty,
            "strong_bonus": args.strong_bonus,
            "loyalty_weight": args.loyalty_weight,
            "cvr_weight": args.cvr_weight,
            "recency_weight": args.recency_weight,
            "global_user_cap": int(args.global_user_cap),
        },
    }
    summary_file = args.output_dir / "fusion_stream_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] stream search done. summary: {summary_file}", flush=True)


if __name__ == "__main__":
    main()


