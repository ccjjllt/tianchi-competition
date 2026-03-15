# 天池经典打榜赛-赛道八：移动推荐算法赛（公开复现版）

本仓库用于复现与迭代阿里天池比赛 **「移动推荐算法赛」** 的完整流程：数据审计、候选召回、二阶段重排（LightGBM）、规则融合、离线评测与提交文件生成。  
比赛入口：[天池官方页面](https://tianchi.aliyun.com/competition/entrance/532423/information)

## 1. 项目目标

- 基于历史行为数据，预测 **2014-12-19** 用户对商品子集 `P` 的购买对 `(user_id, item_id)`。
- 核心评测指标为 `F1`（同时关注 `Precision`、`Recall`）。
- 提交文件遵循比赛格式：两列、tab 分隔、无表头、去重。

## 2. 方法概览

- Baseline：按时间窗口构建候选对与统计特征，生成离线/提交数据库。
- Rerank：使用 LightGBM 对候选对进行二分类排序（支持流式训练与硬负样本策略）。
- Fusion：模型分数 + 规则项（CVR/忠诚度/时效性等）进行全局 TopN 搜索与导出。
- 评测口径统一使用 `global_topn`（避免与 per-user 口径混用）。

## 3. 仓库结构

```text
.
├─ src/                     # 核心代码
│  ├─ run_baseline.py       # Baseline 构建
│  ├─ rerank_lgbm.py        # LightGBM 重排训练/验证
│  ├─ fusion_global_stream.py # 流式融合+全局TopN
│  └─ ...
├─ scripts/                 # 一键实验脚本（PowerShell）
│  ├─ run_14d_push.ps1
│  ├─ run_14d_sprint_09.ps1
│  └─ ...
├─ configs/                 # 配置样例
└─ requirements.txt
```

## 4. 环境依赖

- Python 3.10+
- 依赖安装：

```bash
pip install -r requirements.txt
```

`requirements.txt` 当前包含：`pandas`、`numpy`、`lightgbm`、`optuna`。

## 5. 数据准备

将比赛原始文件放在仓库根目录（不上传到 GitHub）：

- `tianchi_fresh_comp_train_item_online.txt`
- `tianchi_fresh_comp_train_user_online_partA.txt`
- `tianchi_fresh_comp_train_user_online_partB.txt`

## 6. 快速开始（14 天主链路）

1. 构建 baseline（离线 + 提交）：

```bash
python -m src.run_baseline --mode both --eval-date 2014-12-18 --predict-date 2014-12-19 --lookback-days 14 --output-dir outputs/baseline_v2
```

2. 训练 rerank（global_topN 离线评估）：

```bash
python -m src.rerank_lgbm --offline-db outputs/baseline_v2/offline_baseline_work.db --submit-db outputs/baseline_v2/submit_baseline_work.db --output-dir outputs/rerank_v2 --candidate-topk 50 --offline-eval-mode global --topk-grid 30000,40000,50000,60000,70000,80000 --train-streaming --device auto
```

3. 融合并导出提交文件（`.txt`）：

```bash
python -m src.fusion_global_stream --offline-db outputs/baseline_v2/offline_baseline_work.db --submit-db outputs/baseline_v2/submit_baseline_work.db --model-file outputs/rerank_v2/lgbm_rerank_model.txt --output-dir outputs/fusion_submit --candidate-topk-grid 50 --global-topn-grid 40000 --metric-protocol global_topn --lookback-days 14
```

## 7. 离线评测结果（已记录）

以下结果均为 `eval-date=2014-12-18`、`metric_protocol=global_topn`：

| 类型 | F1 | Precision | Recall | best_topn | candidate_topk | 说明 |
|---|---:|---:|---:|---:|---:|---|
| 稳定链路最佳（推荐参考） | **0.0836895** | 0.0771 | 0.0915 | 40000 | 50 | 线上表现更接近该区间 |
| 离线峰值（高召回配置） | **0.1218987** | - | - | 80000 | 50 | 离线很高，但线上存在明显偏差风险 |

对应产物：

- 稳定链路最佳：`outputs/fusion_recover_j_cvr0p46_submit/fusion_stream_summary.json`
- 离线峰值：`outputs/14d_sprint09_round_20260314_180115/summary.json`

说明：线上排行榜与离线评测存在分布偏移，建议以“多提交小步验证”作为主策略，而不是仅追离线峰值。

## 8. 提交文件规范

- 文件后缀：建议 `.txt`
- 两列：`user_id`、`item_id`
- 分隔符：`tab`
- 无表头
- 无重复 `(user_id, item_id)`
