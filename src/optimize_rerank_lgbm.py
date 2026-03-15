from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from src.common import ensure_dir
from src.rerank_lgbm import (
    attach_labels,
    build_features,
    compute_metrics,
    export_topk,
    parse_topk_grid,
    predict_in_batches,
    rank_predictions,
    read_candidates,
    read_labels,
    sample_training_data,
    score_submit_streaming,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize and train LightGBM reranker with Optuna."
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rerank_opt"))
    parser.add_argument("--candidate-topk", type=int, default=50)
    parser.add_argument("--topk-grid", type=str, default="5,10,20,30")
    parser.add_argument("--submit-topk", type=int, default=None)
    parser.add_argument("--optuna-trials", type=int, default=25)
    parser.add_argument("--optuna-timeout-sec", type=int, default=0)
    parser.add_argument("--valid-user-mod", type=int, default=5)
    parser.add_argument("--valid-user-rem", type=int, default=0)
    parser.add_argument(
        "--optimize-max-candidates",
        type=int,
        default=4_000_000,
        help="Cap candidate rows for Optuna stage to speed up trials.",
    )
    parser.add_argument(
        "--max-candidates-final",
        type=int,
        default=None,
        help="Debug mode: cap final-stage candidates.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--predict-batch-size", type=int, default=2_000_000)
    parser.add_argument("--submit-read-chunksize", type=int, default=1_000_000)
    parser.add_argument("--log-eval", type=int, default=100)
    return parser.parse_args()


def _train_lgbm_with_device_fallback(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    params: Dict[str, float],
    num_boost_round: int,
    log_eval: int,
    device: str,
    seed: int,
) -> Tuple[lgb.Booster, str]:
    params_base = {
        "objective": "binary",
        "metric": ["auc"],
        "verbosity": -1,
        "learning_rate": float(params["learning_rate"]),
        "num_leaves": int(params["num_leaves"]),
        "min_data_in_leaf": int(params["min_data_in_leaf"]),
        "feature_fraction": float(params["feature_fraction"]),
        "bagging_fraction": float(params["bagging_fraction"]),
        "bagging_freq": int(params["bagging_freq"]),
        "lambda_l1": float(params["lambda_l1"]),
        "lambda_l2": float(params["lambda_l2"]),
        "seed": int(seed),
        "is_unbalance": True,
    }

    device_plan: List[str]
    if device == "auto":
        device_plan = ["gpu", "cpu"]
    elif device == "gpu":
        device_plan = ["gpu"]
    else:
        device_plan = ["cpu"]

    train_set = lgb.Dataset(
        train_df[feature_cols],
        label=train_df["label"],
        feature_name=feature_cols,
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        valid_df[feature_cols],
        label=valid_df["label"],
        feature_name=feature_cols,
        free_raw_data=False,
    )

    last_err: Optional[Exception] = None
    for dev in device_plan:
        params_try = dict(params_base)
        params_try["device_type"] = dev
        try:
            model = lgb.train(
                params_try,
                train_set,
                num_boost_round=int(num_boost_round),
                valid_sets=[valid_set],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=80, verbose=False),
                    lgb.log_evaluation(period=max(1, int(log_eval))),
                ],
            )
            return model, dev
        except Exception as exc:  # pylint: disable=broad-except
            last_err = exc
            if dev != device_plan[-1]:
                print(f"[WARN] device={dev} failed, fallback...", flush=True)
                continue
            break

    raise RuntimeError(f"Train failed on all devices. last_err={last_err}")


def _make_param_space(trial: optuna.Trial) -> Dict[str, float]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_boost_round": trial.suggest_int("num_boost_round", 200, 800),
        "neg_pos_ratio": trial.suggest_int("neg_pos_ratio", 10, 80),
    }


def _split_train_valid_by_user(
    df: pd.DataFrame,
    mod: int,
    rem: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask_valid = (df["user_id"] % int(mod)) == int(rem)
    valid_df = df.loc[mask_valid].copy()
    train_df = df.loc[~mask_valid].copy()
    if valid_df.empty or train_df.empty:
        raise ValueError("train/valid split is empty. adjust valid-user-mod/rem.")
    return train_df, valid_df


def _run_optuna(
    optimize_df: pd.DataFrame,
    feature_cols: List[str],
    topk_grid: List[int],
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    train_pool, valid_pool = _split_train_valid_by_user(
        optimize_df,
        mod=args.valid_user_mod,
        rem=args.valid_user_rem,
    )
    valid_labels = valid_pool.loc[valid_pool["label"] == 1, ["user_id", "item_id"]]

    records: List[Dict[str, float]] = []

    def objective(trial: optuna.Trial) -> float:
        p = _make_param_space(trial)
        train_sampled = sample_training_data(
            train_pool,
            neg_pos_ratio=int(p["neg_pos_ratio"]),
            seed=args.seed + trial.number,
        )
        model, used_device = _train_lgbm_with_device_fallback(
            train_df=train_sampled,
            valid_df=valid_pool,
            feature_cols=feature_cols,
            params=p,
            num_boost_round=int(p["num_boost_round"]),
            log_eval=args.log_eval,
            device=args.device,
            seed=args.seed + trial.number,
        )
        valid_probs = predict_in_batches(
            booster=model,
            features_df=valid_pool,
            feature_cols=feature_cols,
            batch_size=args.predict_batch_size,
        )
        ranked = rank_predictions(valid_pool, valid_probs)
        metrics = compute_metrics(ranked, valid_labels, topk_grid=topk_grid)
        best_topk, best_metrics = max(
            metrics.items(),
            key=lambda kv: kv[1]["f1"],
        )
        f1 = float(best_metrics["f1"])
        precision = float(best_metrics["precision"])
        recall = float(best_metrics["recall"])

        trial.set_user_attr("best_topk", int(best_topk))
        trial.set_user_attr("precision", precision)
        trial.set_user_attr("recall", recall)
        trial.set_user_attr("device", used_device)
        records.append(
            {
                "trial": trial.number,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "best_topk": int(best_topk),
                "device": used_device,
                **{k: float(v) for k, v in p.items()},
            }
        )
        print(
            f"[TRIAL {trial.number}] f1={f1:.6f} p={precision:.6f} r={recall:.6f} topk={best_topk} device={used_device}",
            flush=True,
        )
        return f1

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    timeout = int(args.optuna_timeout_sec) if int(args.optuna_timeout_sec) > 0 else None
    study.optimize(
        objective,
        n_trials=int(args.optuna_trials),
        timeout=timeout,
        show_progress_bar=False,
    )

    best = dict(study.best_params)
    best["neg_pos_ratio"] = int(best["neg_pos_ratio"])
    best["num_boost_round"] = int(best["num_boost_round"])
    trials_df = pd.DataFrame.from_records(records).sort_values("f1", ascending=False)
    return best, trials_df


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    topk_grid = parse_topk_grid(args.topk_grid)

    print("[STEP] Loading offline candidates (optimization subset)...", flush=True)
    offline_opt_candidates = read_candidates(
        db_path=args.offline_db,
        candidate_topk=args.candidate_topk,
        max_candidates=args.optimize_max_candidates,
    )
    offline_opt_feat, feature_cols = build_features(offline_opt_candidates)
    offline_labels = read_labels(args.offline_db)
    offline_opt_feat = attach_labels(offline_opt_feat, offline_labels)
    print(
        f"[INFO] optimize candidates={len(offline_opt_feat)} positives={int(offline_opt_feat['label'].sum())}",
        flush=True,
    )

    print("[STEP] Optuna parameter search...", flush=True)
    best_params, trials_df = _run_optuna(
        optimize_df=offline_opt_feat,
        feature_cols=feature_cols,
        topk_grid=topk_grid,
        args=args,
    )
    trials_file = args.output_dir / "optuna_trials.csv"
    trials_df.to_csv(trials_file, index=False)

    print("[STEP] Loading full offline candidates for final train/eval...", flush=True)
    offline_full_candidates = read_candidates(
        db_path=args.offline_db,
        candidate_topk=args.candidate_topk,
        max_candidates=args.max_candidates_final,
    )
    offline_full_feat, _ = build_features(offline_full_candidates)
    offline_full_feat = attach_labels(offline_full_feat, offline_labels)
    print(
        f"[INFO] full offline candidates={len(offline_full_feat)} positives={int(offline_full_feat['label'].sum())}",
        flush=True,
    )

    train_pool, valid_pool = _split_train_valid_by_user(
        offline_full_feat,
        mod=args.valid_user_mod,
        rem=args.valid_user_rem,
    )
    final_train_df = sample_training_data(
        train_pool,
        neg_pos_ratio=int(best_params["neg_pos_ratio"]),
        seed=args.seed,
    )
    final_model, used_device = _train_lgbm_with_device_fallback(
        train_df=final_train_df,
        valid_df=valid_pool,
        feature_cols=feature_cols,
        params=best_params,
        num_boost_round=int(best_params["num_boost_round"]),
        log_eval=args.log_eval,
        device=args.device,
        seed=args.seed,
    )
    model_file = args.output_dir / "lgbm_rerank_opt_model.txt"
    final_model.save_model(str(model_file))

    print("[STEP] Scoring full offline...", flush=True)
    offline_probs = predict_in_batches(
        booster=final_model,
        features_df=offline_full_feat,
        feature_cols=feature_cols,
        batch_size=args.predict_batch_size,
    )
    offline_ranked = rank_predictions(offline_full_feat, offline_probs)
    full_metrics = compute_metrics(offline_ranked, offline_labels, topk_grid=topk_grid)
    best_topk = int(max(full_metrics.items(), key=lambda kv: kv[1]["f1"])[0])
    offline_out = args.output_dir / "offline_rerank_opt_predictions.tsv"
    offline_rows = export_topk(offline_ranked, topk=best_topk, out_file=offline_out)

    print("[STEP] Scoring full submit candidates (streaming)...", flush=True)
    submit_topk = int(args.submit_topk) if args.submit_topk is not None else best_topk
    submit_file = args.output_dir / f"submit_rerank_opt_{datetime.now().strftime('%Y%m%d_%H%M')}.tsv"
    submit_rows = score_submit_streaming(
        booster=final_model,
        submit_db=args.submit_db,
        candidate_topk=args.candidate_topk,
        submit_topk=submit_topk,
        feature_cols=feature_cols,
        predict_batch_size=args.predict_batch_size,
        read_chunksize=args.submit_read_chunksize,
        out_file=submit_file,
        max_candidates=args.max_candidates_final,
    )

    summary = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device_requested": args.device,
        "device_used": used_device,
        "offline_db": str(args.offline_db),
        "submit_db": str(args.submit_db),
        "candidate_topk": args.candidate_topk,
        "optuna_trials": int(args.optuna_trials),
        "optimize_max_candidates": args.optimize_max_candidates,
        "max_candidates_final": args.max_candidates_final,
        "topk_grid": topk_grid,
        "best_params": best_params,
        "best_topk_offline": best_topk,
        "submit_topk": submit_topk,
        "offline_metrics_by_topk": full_metrics,
        "offline_prediction_rows": int(offline_rows),
        "offline_prediction_file": str(offline_out),
        "submit_prediction_rows": int(submit_rows),
        "submit_prediction_file": str(submit_file),
        "model_file": str(model_file),
        "trials_file": str(trials_file),
    }
    summary_file = args.output_dir / "rerank_opt_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] optimize rerank done. summary: {summary_file}", flush=True)


if __name__ == "__main__":
    main()
