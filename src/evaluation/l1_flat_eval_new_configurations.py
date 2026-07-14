#!/usr/bin/env python
"""新臂(SLAug_official / CCSDG_port_pvt)的 SUN-SEG 形态分层评估。

## 为什么单独写
`l1_flat_eval_orchestrator.formal_runs()` 从主表读 run 且**硬断言 len==60**（预注册的 formal 矩阵）。
新臂不在那 60 里。为不动预注册矩阵，这里直接从新臂的 `summary.json` 构造 run 列表，
**复用** orchestrator 的 `evaluate_one_run` / `summarize_sunseg`（同一套 SUN-SEG 数据、同一套形态分组、
同一套 cluster-bootstrap CI）—— 复用而非另写，才谈得上与旧 60 run 同口径。

## 为什么这一步是必需的
H1/H2/H3（见 `L1_rebuild_prereg_2026-07-12.md`）全都要看 **hard_flat_IIa** 上的表现。
只有 PolypGen 多中心数字判不了机制假设。

## 排除 CCSDG_official
它是官方 ResNet-UNet（**忠实度锚**），backbone 与 polyp_pvt 臂不同，
放进同一张形态比较表会混淆 backbone 与 DG 机制。故默认不评它。

## 用法
  python l1_flat_eval_new_arms.py --device cuda --batch-size 8
  # summarize 会把新臂与既有 60 run 汇总进同一张 sunseg_flat_summary.csv（同口径可直接对比）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import l1_flat_eval_orchestrator as flat
from s1_loco_common import SMOKE_ROOT

# 只评与 polyp_pvt 臂同 backbone 的新臂（可公平进形态比较表）
DEFAULT_METHODS = ("SLAug_official", "CCSDG_port_pvt")


def runs_from_summaries(run_suffix: str, methods: tuple[str, ...]) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for run_dir in sorted(SMOKE_ROOT.glob(f"*{run_suffix}")):
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            continue
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        if s.get("method") not in methods:
            continue
        ckpt = s.get("checkpoint_best")
        if not ckpt or not Path(ckpt).is_file():
            print(f"[WARN] {s.get('run_id')}: checkpoint_best 缺失，跳过 -> {ckpt}", file=sys.stderr)
            continue
        runs.append(
            {
                "run_id": s["run_id"],
                "source_center": s["source_center"],
                "model": s["model"],
                "method": s["method"],
                "seed": int(s["seed"]),
                "checkpoint_best": ckpt,
            }
        )
    return sorted(runs, key=lambda r: (r["source_center"], r["method"], r["seed"]))


def main() -> int:
    ap = argparse.ArgumentParser(description="新臂的 SUN-SEG 形态分层评估（复用 flat_eval_orchestrator 核心）")
    ap.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    ap.add_argument("--run-suffix", default="L1official")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--skip-summarize", action="store_true")
    args = ap.parse_args()

    if "CCSDG_official" in args.methods:
        raise SystemExit(
            "CCSDG_official 是官方 ResNet-UNet（忠实度锚），backbone 与 polyp_pvt 臂不同，"
            "不得进同一张形态比较表（会混淆 backbone 与 DG 机制）。拒绝评估。"
        )

    flat.ensure_dirs()
    runs = runs_from_summaries(args.run_suffix, tuple(args.methods))
    if not runs:
        raise SystemExit(f"没找到任何新臂 run（suffix={args.run_suffix}, methods={args.methods}）。先跑 driver。")

    print(f"[info] 待评 {len(runs)} 个新臂 run:")
    for r in runs:
        print(f"    {r['run_id']}")

    rows = flat.sunseg_rows()
    print(f"[info] SUN-SEG frames: {len(rows)}")

    # ⚠ flat eval 内部的 `_load_method_model` 是**按 method 名猜 backbone** 的
    #   （`ARM_MODEL_KEY.get(method, "polyp_pvt")`）。猜错 → load_state_dict 直接失败。
    #   EAT 那 6 个 run 在重评里被跳过就是栽在这；跨 backbone 批（pranet）会一模一样地栽。
    #   故这里按每个 run 的**真实 model 字段**显式覆盖，不让它猜。
    import method_dev_batch2 as mdb  # noqa: E402

    done, failed = [], []
    for i, run in enumerate(runs, start=1):
        try:
            mdb.ARM_MODEL_KEY[run["method"]] = run["model"]  # 显式指定该 run 的 backbone
            flat.evaluate_one_run(run, rows, args.device, args.batch_size, args.num_workers, False)
            done.append(run["run_id"])
            print(f"[{i}/{len(runs)}] {run['run_id']} (model={run['model']}): OK", flush=True)
        except Exception as exc:  # 不静默吞
            failed.append({"run_id": run["run_id"], "error": repr(exc)})
            print(f"[{i}/{len(runs)}] {run['run_id']}: FAILED {exc!r}", file=sys.stderr, flush=True)

    if not args.skip_summarize:
        # allow_partial=True：新臂 + 既有 60 run 一起汇总。
        # ⚠ allow_partial 会放过"某些 run 没评到"——**必须**回头核 summary 里实际出现的 run_id 数，
        #   不能因为脚本没报错就当作全评了。
        flat.summarize_sunseg(SimpleNamespace(allow_partial=True))
        print(f"[info] 汇总已写入 {flat.FLAT_ROOT / 'sunseg_flat_summary.csv'}")

    print(json.dumps({"done": len(done), "failed": failed}, ensure_ascii=False, indent=2))
    if failed:
        print(f"\n[WARN] {len(failed)} 个 run 失败，未进汇总——不可当作全量评估。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
