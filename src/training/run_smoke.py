#!/usr/bin/env python
"""S1 LOCO gate smoke runner。

默认 `--run-all`：C3 source、seed0、6 方法 × 2 backbone，每个 job 1 epoch。
正式训练必须另等用户明确 go。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from s1_loco_common import (
    CHECKPOINT_ROOT,
    EXP_ROOT,
    IMAGE_SIZE,
    LOG_ROOT,
    PREP_ROOT,
    SBF_GRID_SIZE,
    SEED,
    SMOKE_ROOT,
    CCSDGProjector,
    PolypGenDataset,
    batch_dice_from_logits,
    ccsdg_contrastive_loss,
    ccsdg_fda_batch,
    boundary_weighted_consistency_loss,
    build_method_model,
    build_model,
    consistency_loss,
    deep_supervision_loss,
    ensure_dirs,
    fuse_outputs,
    method_manifest,
    multiscale_consistency_loss,
    official_sbf_saliency,
    read_csv_dicts,
    register_feature_hook,
    set_seed,
    spatial_tensor_perturb,
    spectral_tensor_perturb,
    summarize_counts,
    uses_ccsdg_triview,
    uses_dual_view_sbf,
    write_csv_dicts,
    write_json,
)

METHODS = tuple(row["method"] for row in method_manifest())
MODELS = ("pranet", "polyp_pvt", "sam2_unet")


def uses_spectral_consistency(method: str) -> bool:
    """需要谱扰动一致性正则的 method。"""
    return method in {"spectral_consistency", "spectral_ibn_combo"}


def uses_spatial_consistency(method: str) -> bool:
    """需要等变边界一致性正则的 method（build-on spatial_warp_aug twist；slaug_* 为 deprecated 别名）。"""
    return method in {"spatialwarp_boundary_consistency", "slaug_boundary_consistency"}


def uses_multiscale_consistency(method: str) -> bool:
    """需要多尺度一致性正则的 method（build-on spatial_warp_aug twist；slaug_* 为 deprecated 别名）。"""
    return method in {"spatialwarp_multiscale_consistency", "slaug_multiscale_consistency"}


def _rows_for(source_center: int, split: str, center: int | None = None) -> list[dict[str, str]]:
    manifest = PREP_ROOT / "loco_splits" / "s1_single_source_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"缺 split manifest: {manifest}，先跑 prep_loco_splits.py")
    rows = [
        row
        for row in read_csv_dicts(manifest)
        if row["source_center"] == f"C{source_center}" and row["split"] == split
    ]
    if center is not None:
        rows = [row for row in rows if row["center"] == f"C{center}"]
    return rows


@torch.no_grad()
def evaluate(raw_model: torch.nn.Module, model_key: str, rows: list[dict[str, str]], device: str, batch_size: int, method: str) -> dict[str, float]:
    raw_model.eval()
    loader = DataLoader(PolypGenDataset(rows, train=False, method=method), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=device.startswith("cuda"))
    dice_values: list[torch.Tensor] = []
    for image, mask in loader:
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = fuse_outputs(model_key, raw_model(image))
        dice_values.append(batch_dice_from_logits(logits, mask))
    if not dice_values:
        return {"n": 0.0, "dice_mean": 0.0}
    values = torch.cat(dice_values).cpu()
    return {"n": float(values.numel()), "dice_mean": float(values.mean())}


def train_one(args: argparse.Namespace) -> None:
    ensure_dirs()
    set_seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；smoke 已指定 cuda。")

    run_suffix = args.run_suffix.strip()
    if run_suffix and not run_suffix.startswith("_"):
        run_suffix = f"_{run_suffix}"
    run_id = f"srcC{args.source_center}_{args.model}_{args.method}_seed{args.seed}{run_suffix}"
    run_dir = SMOKE_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = CHECKPOINT_ROOT / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{run_id}.log"

    train_rows = _rows_for(args.source_center, "train")
    val_rows = _rows_for(args.source_center, "val")
    test_rows_by_center = {
        center: _rows_for(args.source_center, "test", center=center)
        for center in (1, 2, 3, 4, 5, 6)
        if center != args.source_center
    }
    if not train_rows or not val_rows:
        raise RuntimeError(f"{run_id} train/val 为空：train={len(train_rows)} val={len(val_rows)}")

    raw_model, _adapter = build_model(args.model, device)
    raw_model = build_method_model(raw_model, args.method).to(device)
    raw_model.train()
    handles = register_feature_hook(raw_model, args.method)
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # CCSDG：官方用**第二个优化器**单独更新 channel_prompt + projector（train_unet_ccsdg.py:66-68）。
    # 官方用 SGD(momentum .99)，这里沿用 FAIR_BUDGET 的 AdamW/lr 以与其它臂对称 —— optimizer 类型属已知 GAP，
    # 已记入 manifest；若 CCSDG 表现异常，这是嫌疑点之一。
    projector = None
    optimizer_prompt = None
    if uses_ccsdg_triview(args.method):
        # Projector 的 fc 输入维度由 first-layer 特征尺寸决定（官方 131072 是其 256×256 的结果，非超参）。
        # 两种 backbone 的 first-layer 下采样倍数不同（PVT stage1 /4=88；ResNet conv1 /2=176），故实测探测。
        with torch.no_grad():
            probe = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
            probe_content, _probe_style = raw_model.forward_first_layer(probe)
        projector = CCSDGProjector(in_ch=probe_content.shape[1], feat_hw=probe_content.shape[-1]).to(device)
        optimizer_prompt = torch.optim.AdamW(
            list(projector.parameters()) + [raw_model.channel_prompt],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    loader = DataLoader(
        PolypGenDataset(train_rows, train=True, method=args.method),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )

    start = time.time()
    losses: list[float] = []
    val_curve: list[dict[str, float | int]] = []
    best_val = -1.0
    best_epoch = 0
    best_path = ckpt_dir / "best.pth"
    last_path = ckpt_dir / "last.pth"
    for stale_path in (best_path, last_path):
        if stale_path.exists() or stale_path.is_symlink():
            stale_path.unlink()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {
                    "event": "start",
                    "run_id": run_id,
                    "train_n": len(train_rows),
                    "val_n": len(val_rows),
                    "max_epochs": args.epochs,
                    "patience": args.patience,
                    "min_epochs": args.min_epochs,
                    "min_delta": args.min_delta,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        stopped_early = False
        stop_epoch = args.epochs
        epochs_without_improvement = 0
        for epoch in range(1, args.epochs + 1):
            raw_model.train()
            for step, batch in enumerate(loader, start=1):
                stepped_internally = False
                if uses_ccsdg_triview(args.method):
                    # ===== 官方 CCSDG：三视图 + 对比阶段 + 三次分割更新 =====
                    # 逐步对齐官方 train_unet_ccsdg.py:95-158。
                    data_img, gla_img, mask = batch
                    data_img = data_img.to(device, non_blocking=True)
                    gla_img = gla_img.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    fda_img = ccsdg_fda_batch(data_img)  # batch 级：target = batch 反转
                    views = [data_img, fda_img, gla_img]

                    # ① 对比阶段：只更新 channel_prompt + projector（官方 :101-128）
                    optimizer_prompt.zero_grad(set_to_none=True)
                    content_loss, style_loss = ccsdg_contrastive_loss(raw_model, projector, views)
                    (content_loss + style_loss).backward()
                    optimizer_prompt.step()

                    # ② 分割阶段：三视图**各一次独立** zero_grad/backward/step（官方 :130-158）
                    #    ⚠ 这使 CCSDG 每 batch 的分割更新次数 = 其它臂的 3 倍（官方设计，已记 GAP）。
                    seg_vals: list[float] = []
                    for view in views:
                        optimizer.zero_grad(set_to_none=True)
                        out = raw_model(view)
                        seg_loss = deep_supervision_loss(args.model, out, mask, args.method)
                        seg_loss.backward()
                        if args.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
                        optimizer.step()
                        seg_vals.append(float(seg_loss.detach().cpu()))
                    loss = torch.tensor(sum(seg_vals) / len(seg_vals))
                    stepped_internally = True
                elif uses_dual_view_sbf(args.method):
                    # ===== 官方 SLAug：Saliency-Balancing Fusion two-pass =====
                    # 逐步对齐官方 engine.train_one_epoch_SBF：
                    #   pass1 用 GLA 图前向+反向（retain_graph）拿**对输入的梯度** → 梯度 RMS → SBF 显著图
                    #   → 按显著图融合 GLA/LLA → pass2 学融合图；两次 backward 梯度累加后一次 step。
                    gla_image, lla_image, mask = batch
                    gla_image = gla_image.to(device, non_blocking=True)
                    lla_image = lla_image.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)

                    input_var = gla_image.clone().requires_grad_(True)
                    outputs = raw_model(input_var)
                    loss_gla = deep_supervision_loss(args.model, outputs, mask, args.method)
                    loss_gla.backward(retain_graph=True)

                    gradient = torch.sqrt(torch.mean(input_var.grad**2, dim=1, keepdim=True)).detach()
                    saliency = official_sbf_saliency(gradient, SBF_GRID_SIZE)
                    mixed_image = gla_image.detach() * saliency + lla_image * (1.0 - saliency)

                    aug_outputs = raw_model(mixed_image)
                    loss_sbf = deep_supervision_loss(args.model, aug_outputs, mask, args.method)
                    loss_sbf.backward()
                    loss = (loss_gla + loss_sbf).detach()
                else:
                    image, mask = batch
                    image = image.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    outputs = raw_model(image)
                    loss = deep_supervision_loss(args.model, outputs, mask, args.method)
                    if uses_spectral_consistency(args.method):
                        aug_image = spectral_tensor_perturb(image)
                        aug_outputs = raw_model(aug_image)
                        loss = loss + args.consistency_weight * consistency_loss(outputs, aug_outputs, args.model, mask)
                    elif uses_spatial_consistency(args.method):
                        aug_image, grid = spatial_tensor_perturb(image)
                        aug_outputs = raw_model(aug_image)
                        loss = loss + args.consistency_weight * boundary_weighted_consistency_loss(
                            outputs, aug_outputs, args.model, mask, grid
                        )
                    elif uses_multiscale_consistency(args.method):
                        loss = loss + args.consistency_weight * multiscale_consistency_loss(
                            raw_model, image, args.model, mask
                        )
                    loss.backward()
                if not stepped_internally:  # CCSDG 分支已在内部完成 clip+step
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
                    optimizer.step()
                losses.append(float(loss.detach().cpu()))
                if step % args.log_every == 0 or step == 1 or step == len(loader):
                    rec = {"event": "train_step", "epoch": epoch, "step": step, "steps": len(loader), "loss": losses[-1]}
                    log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    log.flush()
                if args.max_train_batches and step >= args.max_train_batches:
                    break
            val_metrics = evaluate(raw_model, args.model, val_rows, device, args.eval_batch_size, args.method)
            val_dice = float(val_metrics["dice_mean"])
            val_curve.append({"epoch": epoch, "val_dice": val_dice})
            if val_dice > best_val + args.min_delta:
                best_val = val_dice
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model": raw_model.state_dict(),
                        "epoch": epoch,
                        "seed": args.seed,
                        "source_center": args.source_center,
                        "method": args.method,
                        "model_key": args.model,
                        "val_dice": val_dice,
                        "run_id": run_id,
                    },
                    best_path,
                )
            else:
                epochs_without_improvement += 1
            log.write(json.dumps({"event": "val", "epoch": epoch, **val_metrics}, ensure_ascii=False) + "\n")
            log.flush()
            if args.patience > 0 and epoch >= args.min_epochs and epochs_without_improvement >= args.patience:
                stopped_early = True
                stop_epoch = epoch
                log.write(
                    json.dumps(
                        {
                            "event": "early_stop",
                            "epoch": epoch,
                            "best_epoch": best_epoch,
                            "best_val_dice": best_val,
                            "patience": args.patience,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                log.flush()
                break

    for handle in handles:
        handle.remove()

    # last.pth 必须先于下方重载 best.pth 落盘，否则 last 会被 best 权重覆盖污染。
    torch.save(
        {
            "model": raw_model.state_dict(),
            "epoch": args.epochs,
            "seed": args.seed,
            "source_center": args.source_center,
            "method": args.method,
            "model_key": args.model,
            "val_dice": val_metrics["dice_mean"],
            "run_id": run_id,
        },
        last_path,
    )
    checkpoint_best_is_hardlink = False
    if best_epoch == args.epochs:
        best_path.unlink(missing_ok=True)
        os.link(last_path, best_path)
        checkpoint_best_is_hardlink = True

    # per-center eval 统一改用 best.pth，与 SUN-SEG flat eval（l1_flat_eval_orchestrator 载 checkpoint_best）口径对齐。
    # 此前这里直接用训练末轮的内存模型 → 主表(末轮) 与 flat 表(best) 不是同一份权重，两表不可比。
    per_center_eval_checkpoint = "best"
    if best_path.exists():
        try:
            state = torch.load(best_path, map_location=device, weights_only=False)
        except TypeError:  # 旧版 torch 无 weights_only 参数
            state = torch.load(best_path, map_location=device)
        raw_model.load_state_dict(state["model"])
    else:
        # 不静默降级：best.pth 缺失必须在 summary 与 stderr 同时暴露
        per_center_eval_checkpoint = "last_fallback_best_missing"
        print(f"[WARN] {run_id}: best.pth 缺失，per-center eval 退回末轮权重", file=sys.stderr)

    center_metrics = []
    for center, rows in sorted(test_rows_by_center.items()):
        metric = evaluate(raw_model, args.model, rows, device, args.eval_batch_size, args.method)
        center_metrics.append({"center": f"C{center}", **metric})

    summary = {
        "run_id": run_id,
        "source_center": f"C{args.source_center}",
        "model": args.model,
        "method": args.method,
        "seed": args.seed,
        "epochs": stop_epoch,
        "max_epochs": args.epochs,
        "stopped_early": stopped_early,
        "patience": args.patience,
        "min_epochs": args.min_epochs,
        "min_delta": args.min_delta,
        "train_n": len(train_rows),
        "val_n": len(val_rows),
        "loss_last": losses[-1] if losses else None,
        "loss_mean": sum(losses) / len(losses) if losses else None,
        "val_dice": val_metrics["dice_mean"],
        "best_val_dice": best_val,
        "best_epoch": best_epoch,
        "val_curve": val_curve,
        "per_center": center_metrics,
        "checkpoint_last": str(last_path),
        "checkpoint_best": str(best_path),
        "checkpoint_best_is_hardlink": checkpoint_best_is_hardlink,
        "per_center_eval_checkpoint": per_center_eval_checkpoint,
        "log_path": str(log_path),
        "seconds": round(time.time() - start, 2),
    }
    write_json(run_dir / "summary.json", summary)
    write_csv_dicts(run_dir / "per_center_dice.csv", center_metrics, ["center", "n", "dice_mean"])
    print(json.dumps(summary, ensure_ascii=False))


def run_all(args: argparse.Namespace) -> None:
    ensure_dirs()
    write_csv_dicts(SMOKE_ROOT / "method_manifest.csv", method_manifest(), ["method", "status", "note"])
    jobs = [(model, method) for model in MODELS for method in METHODS]
    rows = []
    failed = []
    for model, method in jobs:
        run_id = f"srcC{args.source_center}_{model}_{method}_seed{args.seed}"
        log_path = LOG_ROOT / f"{run_id}.outer.log"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single",
            "--source-center",
            str(args.source_center),
            "--model",
            model,
            "--method",
            method,
            "--seed",
            str(args.seed),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--device",
            args.device,
            "--num-workers",
            str(args.num_workers),
        ]
        if args.max_train_batches:
            cmd += ["--max-train-batches", str(args.max_train_batches)]
        print(f"[run_all] start {run_id}")
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(EXP_ROOT), text=True, stdout=log, stderr=subprocess.STDOUT, check=False)
        summary_path = SMOKE_ROOT / run_id / "summary.json"
        if proc.returncode != 0 or not summary_path.exists():
            failed.append({"run_id": run_id, "returncode": proc.returncode, "outer_log": str(log_path)})
            print(f"[run_all] FAILED {run_id} rc={proc.returncode}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for center_metric in summary["per_center"]:
            rows.append(
                {
                    "run_id": run_id,
                    "source_center": summary["source_center"],
                    "model": model,
                    "method": method,
                    "seed": args.seed,
                    "center": center_metric["center"],
                    "n": center_metric["n"],
                    "dice_mean": center_metric["dice_mean"],
                    "val_dice": summary["val_dice"],
                    "loss_last": summary["loss_last"],
                    "seconds": summary["seconds"],
                    "log_path": summary["log_path"],
                    "summary_path": str(summary_path),
                    "outer_log": str(log_path),
                }
            )
        print(f"[run_all] done {run_id}")
    write_csv_dicts(
        SMOKE_ROOT / "smoke_per_center_summary.csv",
        rows,
        [
            "run_id",
            "source_center",
            "model",
            "method",
            "seed",
            "center",
            "n",
            "dice_mean",
            "val_dice",
            "loss_last",
            "seconds",
            "log_path",
            "summary_path",
            "outer_log",
        ],
    )
    write_csv_dicts(SMOKE_ROOT / "smoke_run_failures.csv", failed, ["run_id", "returncode", "outer_log"])
    write_json(
        SMOKE_ROOT / "summary.json",
        {
            "source_center": f"C{args.source_center}",
            "seed": args.seed,
            "epochs": args.epochs,
            "n_jobs": len(jobs),
            "n_success": len(jobs) - len(failed),
            "n_failed": len(failed),
            "failures": failed,
            "counts_by_model_method": summarize_counts(rows, ["model", "method"]),
        },
    )
    if failed:
        raise SystemExit(f"{len(failed)} smoke jobs failed; see {SMOKE_ROOT / 'smoke_run_failures.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--source-center", type=int, default=3)
    parser.add_argument("--model", choices=MODELS, default="pranet")
    parser.add_argument("--method", choices=METHODS, default="strong_aug_floor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--consistency-weight", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--run-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.single:
        train_one(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
