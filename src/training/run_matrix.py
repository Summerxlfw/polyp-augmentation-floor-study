#!/usr/bin/env python
"""L1 官方忠实复现臂 driver（不动预注册的 60-run FORMAL_ARMS 矩阵）。

## 背景
主矩阵中的 "SLAug / CCSDG / CSDG / MixStyle / DSU" 是共享协议下的轻量研究实现，
其中历史 key "SLAug" 实际对应空间仿射 warp，而已发表 SLAug 是强度域
location-scale + SBF。本 driver 用于作者代码适配和协议对齐实验。

## 跑什么
默认 `SLAug_official`（官方 GLA/LLA 双视图 + SBF 显著性融合 two-pass）× {C3,C1} × seed{0,1,2} = 6 run。
recipe 完全同 FAIR_BUDGET（与既有 60-run 对称），split 同一 manifest。

## 不做的事
- 不改 FORMAL_ARMS / 不重跑 60-run（保预注册语义）。
- 不覆盖既有主表；append 行单独落盘，由人核对后再合表。

## 用法
  python l1_official_arms_driver.py --methods SLAug_official --out-dir <outputs>/official_arms_<date>
  # GPU 由 --device / CUDA_VISIBLE_DEVICES 控制；启动权在用户，本脚本不自行选卡。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from s1_loco_common import SMOKE_ROOT, write_json

S1_ROOT = Path(__file__).resolve().parent

FAIR_BUDGET = {
    "epochs": 20,
    "patience": 5,
    "min_epochs": 10,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "batch_size": 4,
    "eval_batch_size": 8,
    "grad_clip": 1.0,
    "consistency_weight": 0.2,
}
SOURCE_CENTERS = (3, 1)
SEEDS = (0, 1, 2)
RUN_SUFFIX = "L1official"

# method → backbone。
# polyp_pvt 的三个臂彼此可比（同 backbone，只差 DG 机制）。
# CCSDG_official 用官方 ResNet-UNet ⇒ **只作忠实度锚，不与 polyp_pvt 臂同表做公平比较**。
METHOD_MODEL = {
    "SLAug_official": "polyp_pvt",
    "CCSDG_port_pvt": "polyp_pvt",
    "CCSDG_official": "unet_ccsdg",
    # 跨 backbone 验证用臂（默认 polyp_pvt；用 --model 覆盖成 pranet 即可换骨架）
    "strong_aug_floor": "polyp_pvt",
    "spatial_warp_aug": "polyp_pvt",
    "spectral_consistency": "polyp_pvt",
    # 组合臂（真 SLAug 底座 + 我们的几何 warp）
    "slaug_official_plus_warp": "polyp_pvt",
    # A2 隔离对照（TMI 的 A1>A2 门）
    "warp_alpha100": "polyp_pvt",
    "warp_alpha015": "polyp_pvt",
    "warp_shift_only": "polyp_pvt",
    "joint_affine_floor": "polyp_pvt",  # P01 隔离臂：image+mask 同步仿射，无 alpha 混合
    "paired_affine_softmix": "polyp_pvt",  # factorial 补格 U1B1：paired target + alpha blend（soft target，post-hoc control）
}


def build_cmd(
    method: str,
    source_center: int,
    seed: int,
    device: str,
    model_override: str | None = None,
    run_suffix: str = RUN_SUFFIX,
) -> list[str]:
    model = model_override or METHOD_MODEL.get(method)
    if model is None:
        raise SystemExit(f"未知 method {method}；已知: {sorted(METHOD_MODEL)}")
    cmd = [
        sys.executable,
        str(S1_ROOT / "run_smoke.py"),
        "--single",
        "--source-center", str(source_center),
        "--model", model,
        "--method", method,
        "--seed", str(seed),
        "--run-suffix", run_suffix,
        "--device", device,
    ]
    for key, val in FAIR_BUDGET.items():
        cmd += [f"--{key.replace('_', '-')}", str(val)]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description="跑官方忠实复现臂（默认 SLAug_official）")
    ap.add_argument("--methods", nargs="+", default=["SLAug_official"])
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--model",
        default=None,
        help="覆盖 backbone（跨 backbone 验证用，如 --model pranet）。不传则用 METHOD_MODEL 默认。",
    )
    ap.add_argument("--run-suffix", default=RUN_SUFFIX, help="区分批次（跨 backbone 批建议 L1backbone）")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    planned = [
        (m, sc, sd) for m in args.methods for sc in SOURCE_CENTERS for sd in SEEDS
    ]
    print(f"[info] 计划 {len(planned)} run: methods={args.methods} sources={SOURCE_CENTERS} seeds={SEEDS}")
    print(f"[info] SBF_GRID_SIZE={os.environ.get('S1_SBF_GRID_SIZE', '8')} (官方无 polyp 配置，此为我们自定)")

    completed: list[str] = []
    failed: list[dict[str, str]] = []
    start = time.time()

    for i, (method, sc, sd) in enumerate(planned, start=1):
        model = args.model or METHOD_MODEL[method]
        run_id = f"srcC{sc}_{model}_{method}_seed{sd}_{args.run_suffix}"
        cmd = build_cmd(method, sc, sd, args.device, args.model, args.run_suffix)
        print(f"\n[{i}/{len(planned)}] {run_id}\n  $ {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or "")[-1500:]
            failed.append({"run_id": run_id, "returncode": str(proc.returncode), "stderr_tail": tail})
            print(f"  FAILED rc={proc.returncode}\n{tail}", file=sys.stderr, flush=True)
            continue
        completed.append(run_id)
        summary_path = SMOKE_ROOT / run_id / "summary.json"
        if summary_path.exists():
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            centers = {c["center"]: round(c["dice_mean"], 4) for c in s.get("per_center", [])}
            print(
                f"  OK best_epoch={s.get('best_epoch')} val={s.get('best_val_dice'):.4f} "
                f"eval_ckpt={s.get('per_center_eval_checkpoint')} per_center={centers}",
                flush=True,
            )

    done = {
        "state": "done" if not failed else "done_with_failures",
        "methods": args.methods,
        "model_override": args.model,  # 跨 backbone 批必须留痕
        "models_used": sorted({args.model or METHOD_MODEL[m] for m in args.methods}),
        "planned": len(planned),
        "completed": len(completed),
        "failed": failed,  # 非空必须在汇报里暴露，不得当作全量成功
        "fair_budget": FAIR_BUDGET,
        "sbf_grid_size": int(os.environ.get("S1_SBF_GRID_SIZE", "8")),
        "run_suffix": args.run_suffix,  # provenance 修正：记录实际批次后缀，而非模块常量
        "seconds": round(time.time() - start, 2),
    }
    write_json(args.out_dir / "official_arms_DONE.json", done)
    print("\n" + json.dumps(done, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
