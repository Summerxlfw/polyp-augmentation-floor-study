#!/usr/bin/env python
"""S1 LOCO gate 公共工具。

本目录只承载 prep/smoke 的可复现入口；正式训练需用户再次明确 go。

公开版只把服务器绝对路径改成环境变量；算法、采样范围、损失与训练预算未改。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import Dataset


PROJECT_ROOT = Path(
    os.environ.get("POLYP_PROJECT_ROOT", Path(__file__).resolve().parents[3])
).expanduser()


def _env_path(name: str, default: Path) -> Path:
    """允许正式实验覆盖输出路径，同时保持默认 smoke 路径不变。"""
    return Path(os.environ.get(name, str(default))).expanduser()


EXP_ROOT = _env_path("S1_EXP_ROOT", PROJECT_ROOT / "03_experiments" / "S1_loco_gate_smoke_20260710")
OUTPUT_ROOT = _env_path("S1_OUTPUT_ROOT", EXP_ROOT / "outputs")
LOG_ROOT = _env_path("S1_LOG_ROOT", EXP_ROOT / "logs")
PREP_ROOT = _env_path("S1_PREP_ROOT", OUTPUT_ROOT / "prep")
SMOKE_ROOT = _env_path("S1_SMOKE_ROOT", OUTPUT_ROOT / "smoke")
CHECKPOINT_ROOT = _env_path(
    "S1_CHECKPOINT_ROOT",
    PROJECT_ROOT / "checkpoints" / "S1_loco_gate_smoke_20260710",
)
POLYPGEN_ROOT = _env_path("POLYPGEN_ROOT", PROJECT_ROOT / "data" / "PolypGen")
POLYP_SIZE_ROOT = _env_path("POLYP_SIZE_ROOT", PROJECT_ROOT / "data" / "Polyp_Size_Dataset")
SUNSEG_ROOT = _env_path("SUNSEG_ROOT", PROJECT_ROOT / "data" / "SUN-SEG")
REPRO_ROOT = _env_path("POLYP_REPRO_ROOT", PROJECT_ROOT / "third_party" / "repro")
THIRD_PARTY_ROOT = _env_path("POLYP_THIRD_PARTY_ROOT", PROJECT_ROOT / "third_party")
# 官方 SOTA repo（忠实复现用；此前主表的 SLAug/CCSDG 等只是轻量近似，未接官方实现）
S1_REPRO_REPOS = _env_path("S1_REPRO_REPOS", PROJECT_ROOT / "03_experiments" / "repro_repos_s1_20260710")

CENTER_IDS = (1, 2, 3, 4, 5, 6)
SOURCE_CENTERS = (3, 1)
SEED = 20260710
IMAGE_SIZE = 352
NEG_MASK_MAX = 30
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 官方 SLAug 的 SBF 显著图网格粒度。
# ⚠ 官方按任务调过此超参且**没有 polyp 配置**（腹部 CHAOS/SABSCT=3，心脏 bSSFP/LEG=18）。
# 这里的 8 是**我们为 polyp(352×352) 自定的**，非官方值。若 SLAug_official 表现异常低于地板，
# 第一嫌疑就是它——先试 {3, 18} 再谈"真 SLAug 对扁平无效"。
SBF_GRID_SIZE = int(os.environ.get("S1_SBF_GRID_SIZE", "8"))

_SLAUG_OFFICIAL_AUG = None  # 官方 LocationScaleAugmentation 惰性单例


def add_official_slaug_to_path() -> None:
    """把官方 SLAug repo 加进 sys.path，用于 import 其原版实现（而非手抄）。"""
    slaug_root = S1_REPRO_REPOS / "SLAug"
    if not slaug_root.exists():
        raise FileNotFoundError(
            f"官方 SLAug repo 不存在: {slaug_root}。忠实复现必须接官方实现；"
            f"可用环境变量 S1_REPRO_REPOS 指定 repro_repos_s1_20260710 的位置。"
        )
    if str(slaug_root) not in sys.path:
        sys.path.insert(0, str(slaug_root))

METHOD_DEV_ARMS = (
    "strong_aug_floor",
    "m2_fad_far",
    "m1_wavelet_boundary",
    "topo_cldice",
    "ibn_whitening",
    "fourier_amp_aug",
    "isw_full",
    "sam2_adapter_dg",
    "sam2_plain_frozen",
    "spectral_consistency",
    "spectral_ibn_combo",
    "joint_affine_floor",  # P01 隔离臂：image+mask 同步仿射，无 alpha 混合（A1>A2 门）
    "paired_affine_softmix",  # factorial 补格 U1B1：paired target + alpha blend（soft target，post-hoc control）
)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SUNSEG_SPLITS = (
    "TrainDataset",
    "TestEasyDataset/Seen",
    "TestEasyDataset/Unseen",
    "TestHardDataset/Seen",
    "TestHardDataset/Unseen",
)


def ensure_dirs() -> None:
    """创建本实验需要的小文件目录。"""
    for path in (OUTPUT_ROOT, LOG_ROOT, PREP_ROOT, SMOKE_ROOT, CHECKPOINT_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    """固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_bucket(*parts: str, seed: int = SEED) -> float:
    """把 key 稳定映射到 [0, 1)，用于 split hash。"""
    payload = "|".join([str(seed), *parts]).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return int(digest, 16) / float(16**16)


def classify_paris(value: Any) -> dict[str, Any]:
    """按用户确认口径标注 Paris 形态。

    主承重 hard-flat = Paris 0-II 系，即 IIa/IIb/IIc；Is/Isp/Ip 保留为非 flat 子组。
    """
    raw = "" if value is None else str(value).strip()
    norm = raw.replace("0-", "").replace("0", "").strip().upper()
    norm = norm.replace(" ", "")
    if norm in {"IIA", "IIB", "IIC"}:
        return {"paris_raw": raw, "paris_norm": norm, "hard_flat": True, "paris_group": f"hard_flat_{norm}"}
    if norm == "IS":
        return {"paris_raw": raw, "paris_norm": norm, "hard_flat": False, "paris_group": "polypoid_Is"}
    if norm == "ISP":
        return {"paris_raw": raw, "paris_norm": norm, "hard_flat": False, "paris_group": "polypoid_Isp"}
    if norm == "IP":
        return {"paris_raw": raw, "paris_norm": norm, "hard_flat": False, "paris_group": "polypoid_Ip"}
    return {"paris_raw": raw, "paris_norm": norm, "hard_flat": False, "paris_group": "other_or_unknown"}


def parse_flat_polyps_list(path: Path = SUNSEG_ROOT / "flat_polyps_list.txt") -> list[dict[str, Any]]:
    """解析 SUN-SEG flat_polyps_list.txt。

    原文件按中文小节列出 Is 与 IIa case，IIa 才作为 hard-flat 主子集。
    """
    if not path.exists():
        raise FileNotFoundError(f"缺少 SUN-SEG flat list: {path}")
    rows: list[dict[str, Any]] = []
    current_shape = ""
    line_re = re.compile(r"^(case[\w_]+),\s*位置:\s*([A-Z]+)\(([^)]+)\),\s*病理:\s*(.+)$")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "Is" in line and "类型" in line:
            current_shape = "Is"
            continue
        if "IIa" in line and "类型" in line:
            current_shape = "IIa"
            continue
        match = line_re.match(line)
        if not match:
            continue
        case_id, location_code, location_text, pathology = match.groups()
        paris = classify_paris(current_shape)
        rows.append(
            {
                "case_id": case_id,
                "shape": current_shape,
                "paris_group": paris["paris_group"],
                "hard_flat": bool(paris["hard_flat"]),
                "location_code": location_code,
                "location_text": location_text,
                "pathology_codes": pathology.replace("，", ",").replace(" ", ""),
                "source": "flat_polyps_list",
            }
        )
    return rows


def _count_image_files(path: Path | None) -> int:
    if path is None or not path.is_dir():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMG_EXTS)


def _find_sunseg_case_dirs(case_id: str, root: Path) -> tuple[str, Path | None, Path | None]:
    for split in SUNSEG_SPLITS:
        frame_dir = root / split / "Frame" / case_id
        gt_dir = root / split / "GT" / case_id
        if frame_dir.is_dir() or gt_dir.is_dir():
            return split, frame_dir if frame_dir.is_dir() else None, gt_dir if gt_dir.is_dir() else None
    return "", None, None


def resolve_sunseg_case_paths(row: dict[str, Any], sunseg_root: Path = SUNSEG_ROOT) -> dict[str, Any]:
    """把 SUN-SEG case 映射到 full_res/cache_256x448 的 Frame/GT 目录。"""
    case_id = str(row.get("case_id") or row.get("No") or row.get("case") or "").strip()
    if not case_id:
        raise ValueError(f"SUN-SEG row 缺少 case_id: {row}")
    split, full_frame_dir, full_gt_dir = _find_sunseg_case_dirs(case_id, sunseg_root / "full_res")
    cache_split, cache_frame_dir, cache_gt_dir = _find_sunseg_case_dirs(case_id, sunseg_root / "cache_256x448")
    if split and cache_split and split != cache_split:
        raise RuntimeError(f"{case_id} full_res/cache split 不一致: {split} vs {cache_split}")
    resolved_split = split or cache_split
    return {
        "case_id": case_id,
        "split": resolved_split,
        "full_res_frame_dir": str(full_frame_dir or ""),
        "full_res_gt_dir": str(full_gt_dir or ""),
        "cache_frame_dir": str(cache_frame_dir or ""),
        "cache_gt_dir": str(cache_gt_dir or ""),
        "n_full_res_frames": _count_image_files(full_frame_dir),
        "n_full_res_masks": _count_image_files(full_gt_dir),
        "n_cache_frames": _count_image_files(cache_frame_dir),
        "n_cache_masks": _count_image_files(cache_gt_dir),
    }


def edge_from_binary_mask(mask: np.ndarray, dilation: int = 3) -> np.ndarray:
    """从二值 mask 派生 ESPNet edge-GT，返回 float32 0/1。"""
    if mask.ndim == 3:
        mask = mask.squeeze()
    binary = (mask > 0).astype(np.uint8) * 255
    edge = cv2.Canny(binary, 100, 200)
    if dilation > 1:
        kernel = np.ones((dilation, dilation), dtype=np.uint8)
        edge = cv2.dilate(edge, kernel, iterations=1)
    return (edge > 0).astype(np.float32)


def uniform_frame_indices(total_frames: int, target_count: int = 16) -> list[int]:
    """固定每视频最多 target_count 帧的均匀采样索引。"""
    if total_frames <= 0:
        return []
    if total_frames <= target_count:
        return list(range(total_frames))
    values = np.linspace(0, total_frames - 1, num=target_count)
    return sorted({int(round(v)) for v in values})


def phash_from_bgr(frame: np.ndarray) -> np.uint64:
    """计算 64-bit pHash。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    block = dct[:8, :8]
    median = np.median(block[1:, 1:])
    bits = (block > median).astype(np.uint8).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return np.uint64(value)


def phash_hamming(a: np.uint64, b: np.uint64) -> int:
    return int(int(a ^ b).bit_count())


def is_near_duplicate_phash(value: np.uint64, seen: Iterable[np.uint64], threshold: int = 6) -> bool:
    return any(phash_hamming(value, old) <= threshold for old in seen)


@dataclass(frozen=True)
class PolypPair:
    center: int
    stem: str
    image_path: Path
    mask_path: Path
    image_sha256: str
    mask_sha256: str


def scan_center_pairs(center: int, root: Path = POLYPGEN_ROOT) -> tuple[list[PolypPair], dict[str, int]]:
    """扫描 PolypGen 单中心正帧配对。"""
    img_dir = root / f"data_C{center}" / f"images_C{center}"
    mask_dir = root / f"data_C{center}" / f"masks_C{center}"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"C{center} 缺少 image/mask 目录: {img_dir}, {mask_dir}")

    mask_by_stem: dict[str, Path] = {}
    for name in sorted(os.listdir(mask_dir)):
        if not name.lower().endswith(IMG_EXTS):
            continue
        stem = Path(name).stem
        if stem.endswith("_mask"):
            stem = stem[: -len("_mask")]
        mask_by_stem[stem] = mask_dir / name

    pairs: list[PolypPair] = []
    stats = {"n_images": 0, "n_pos": 0, "n_neg": 0, "n_unpaired": 0}
    for name in sorted(os.listdir(img_dir)):
        if not name.lower().endswith(IMG_EXTS):
            continue
        stats["n_images"] += 1
        stem = Path(name).stem
        img_path = img_dir / name
        mask_path = mask_by_stem.get(stem)
        if mask_path is None:
            stats["n_unpaired"] += 1
            continue
        mask = np.asarray(Image.open(mask_path).convert("L"))
        if int(mask.max()) <= NEG_MASK_MAX:
            stats["n_neg"] += 1
            continue
        pairs.append(
            PolypPair(
                center=center,
                stem=stem,
                image_path=img_path,
                mask_path=mask_path,
                image_sha256=sha256_file(img_path),
                mask_sha256=sha256_file(mask_path),
            )
        )
        stats["n_pos"] += 1
    return pairs, stats


def build_single_source_rows(source_center: int, val_frac: float = 0.2, seed: int = SEED) -> list[dict[str, Any]]:
    """S1 单源 split：source 正帧 80/20 train/val，其他中心正帧 frozen test。"""
    rows: list[dict[str, Any]] = []
    for center in CENTER_IDS:
        pairs, _stats = scan_center_pairs(center)
        for pair in pairs:
            if center == source_center:
                bucket = hash_bucket(f"C{center}", pair.stem, seed=seed)
                split = "val" if bucket < val_frac else "train"
            else:
                bucket = ""
                split = "test"
            rows.append(
                {
                    "source_center": f"C{source_center}",
                    "center": f"C{center}",
                    "split": split,
                    "stem": pair.stem,
                    "image_path": str(pair.image_path),
                    "mask_path": str(pair.mask_path),
                    "image_sha256": pair.image_sha256,
                    "mask_sha256": pair.mask_sha256,
                    "split_hash_seed": seed,
                    "split_hash_bucket": bucket,
                }
            )
    return rows


class PolypGenDataset(Dataset):
    """PolypGen manifest 数据集，训练返回 ImageNet norm 图像 + binary mask。"""

    def __init__(self, rows: list[dict[str, str]], image_size: int = IMAGE_SIZE, train: bool = False, method: str = "strong_aug_floor") -> None:
        self.rows = rows
        self.image_size = image_size
        self.train = train
        self.method = method

    def __len__(self) -> int:
        return len(self.rows)

    def _img_tensor(self, img: Image.Image) -> torch.Tensor:
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_t = TF.to_tensor(img)
        return TF.normalize(img_t, IMAGENET_MEAN, IMAGENET_STD)

    def _mask_tensor(self, mask: Image.Image) -> torch.Tensor:
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        arr = np.asarray(mask, dtype=np.float32)
        if mask.mode == "F":
            # soft target：PIL "F" 模式 float32 [0,1]（仅 _apply_paired_affine_softmix train 产出）。
            # 保留 soft，**不做 >0.5 二值化**——否则破坏凸组合 soft target 语义（= silent threshold）。
            # val/test 的 GT 与所有其他 method 的 mask 均为 "L" 0-255 → 走下方原二值化路径，bit-wise 不变。
            # mode-gated（非 method-gated）：避免 val/test 时对原始 GT 误走 soft 通道导致 mask 值域 0-255。
            mask_arr = arr
        else:
            mask_arr = (arr / 255.0 > 0.5).astype(np.float32)
        return torch.from_numpy(mask_arr).unsqueeze(0)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")
        # SLAug_official 训练时返回 (GLA, LLA, mask) 三元组；其余臂与所有 eval 保持 (img, mask) 二元组不变。
        # 单视图路径的算子顺序全为确定性操作（不消耗随机数），重构后与改动前 bit-wise 等价。
        if self.train and uses_dual_view_sbf(self.method):
            gla_img, lla_img, mask = apply_train_aug_dual(img, mask, self.method)
            return self._img_tensor(gla_img), self._img_tensor(lla_img), self._mask_tensor(mask)
        if self.train and uses_ccsdg_triview(self.method):
            # (原图, GLA, mask)；第三视图 FDA 是 batch 级，在训练循环里算
            data_img, gla_img, mask = apply_train_aug_ccsdg(img, mask, self.method)
            return self._img_tensor(data_img), self._img_tensor(gla_img), self._mask_tensor(mask)
        if self.train:
            img, mask = apply_train_aug(img, mask, self.method)
        return self._img_tensor(img), self._mask_tensor(mask)


def _joint_geometric_aug(img: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.2:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    angle = random.uniform(-15.0, 15.0)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)
    return img, mask


def _color_jitter(img: Image.Image, strength: float = 0.18) -> Image.Image:
    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = 1.0 + random.uniform(-strength, strength)
        img = enhancer_cls(img).enhance(factor)
    return img


def _sample_spatial_warp_params(h: int, w: int) -> tuple[float, float, float, float]:
    """采样一组仿射 warp 参数（scale/dx/dy/alpha）。

    抽出来是为了让**多个视图能共用同一组参数** —— 组合臂 `SLAug_official + warp` 里，
    GLA/LLA 两个视图必须保持**空间对齐**（SBF 要按显著图融合二者），
    各自独立采样 warp 会破坏对齐、让 SBF 融合出错。
    """
    scale = random.uniform(0.84, 1.16)
    dx = random.uniform(-0.08, 0.08) * w
    dy = random.uniform(-0.08, 0.08) * h
    alpha = random.uniform(0.35, 0.65)
    return scale, dx, dy, alpha


def _apply_spatial_warp_with_params(img: Image.Image, params: tuple[float, float, float, float]) -> Image.Image:
    scale, dx, dy, alpha = params
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    mat = np.array([[scale, 0, (1 - scale) * w / 2 + dx], [0, scale, (1 - scale) * h / 2 + dy]], dtype=np.float32)
    warped = cv2.warpAffine(arr, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    out = alpha * warped + (1.0 - alpha) * arr
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _apply_joint_affine(
    img: Image.Image,
    mask: Image.Image,
    _params: tuple[float, float, float] | None = None,
) -> tuple[Image.Image, Image.Image]:
    """joint_affine_floor：对 image 和 mask 施加**同一个**仿射矩阵，**不做 alpha 混合**。

    相对 A1 (spatial_warp_aug / _apply_spatial_affine_warp) 有两个结构差异：
      ① mask 跟着同步变换（A1 是 image-only，不改 mask）；
      ② 不做 alpha 混合（A1 用 alpha∈[0.35,0.65] 把 warp 图与原图混合）。
    scale / shift 的采样范围与 A1 的 _sample_spatial_warp_params **逐字一致**
    （scale∈[0.84,1.16]、shift±0.08×(W,H)），保证隔离公平：几何幅度对齐，
    只拿掉“图-标签错位”这一个变量。注意 A1 与本 control 同时在 mask 同步与 alpha blend
    两处不同，不得把任一差异单独命名为“唯一差异”或“标签错位机制”。
    """
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]

    if _params is None:
        scale = random.uniform(0.84, 1.16)           # 与 A1 一致
        dx = random.uniform(-0.08, 0.08) * w          # 与 A1 一致
        dy = random.uniform(-0.08, 0.08) * h          # 与 A1 一致
    else:
        scale, dx, dy = _params

    # image 与 mask 共用同一个 2×3 仿射矩阵（本臂存在的理由）
    mat = np.array(
        [[scale, 0, (1 - scale) * w / 2 + dx],
         [0, scale, (1 - scale) * h / 2 + dy]],
        dtype=np.float32,
    )
    warped_img = cv2.warpAffine(
        arr, mat, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )
    # mask 用**同一个 mat** + 最近邻插值（保持二值），边界 constant-zero
    m = np.asarray(mask).astype(np.float32)
    warped_mask = cv2.warpAffine(
        m, mat, (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    # 无 alpha 混合：输出即纯 warp（与 mask 同步一起构成相对 A1 的两个结构差异）
    return (
        Image.fromarray(np.clip(warped_img, 0, 255).astype(np.uint8)),
        Image.fromarray(warped_mask.astype(np.uint8)),
    )


def _apply_paired_affine_softmix(
    img: Image.Image,
    mask: Image.Image,
    _params: tuple[float, float, float, float] | None = None,
) -> tuple[Image.Image, Image.Image]:
    """paired_affine_softmix：image 与 mask 用**同一个仿射矩阵 + 同一个 alpha**做凸组合。

    factorial 补格（2×2 唯一缺格 U1B1 = paired target + alpha blend）：
      - image 分支：``alpha * warped_img + (1-alpha) * arr`` —— 与 A1
        (``_apply_spatial_warp_with_params``) 的 image 处理**逐字相同**
        （同 mat、同 alpha、同 INTER_LINEAR + BORDER_REFLECT_101）；
      - mask 分支：``alpha * warped_mask + (1-alpha) * m`` —— target 跟随仿射
        （同 joint_affine_floor 的 nearest + constant-zero），**且**做凸组合，
        输出为 [0,1] **soft target**（PIL "F" float32），不 threshold/不 round/不取 OR/AND。

    采样范围与 A1 ``_sample_spatial_warp_params`` 逐字一致
    （scale∈[0.84,1.16]、shift±0.08×(W,H)、alpha∈[0.35,0.65]）。
    ⚠ image 与 mask 共用同一 mat 与同一 alpha —— 本臂存在的理由，fixed-param 测试须证明。
    ⚠ soft mask 经 ``PolypGenDataset._mask_tensor`` 的 method-gated soft 通道保留 [0,1]，
      现有 ``structure_loss``(wBCE+wIoU) 原生接受 soft target，无需改 loss。
    """
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]

    if _params is None:
        scale = random.uniform(0.84, 1.16)           # 与 A1 一致
        dx = random.uniform(-0.08, 0.08) * w          # 与 A1 一致
        dy = random.uniform(-0.08, 0.08) * h          # 与 A1 一致
        alpha = random.uniform(0.35, 0.65)           # 与 A1 一致
    else:
        scale, dx, dy, alpha = _params

    # image 与 mask 共用同一个 2×3 仿射矩阵（本臂存在的理由）
    mat = np.array(
        [[scale, 0, (1 - scale) * w / 2 + dx],
         [0, scale, (1 - scale) * h / 2 + dy]],
        dtype=np.float32,
    )
    # image: bilinear + BORDER_REFLECT_101（与 A1 image 分支逐字一致）
    warped_img = cv2.warpAffine(
        arr, mat, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )
    img_out = alpha * warped_img + (1.0 - alpha) * arr   # 与 A1 image 处理逐字相同

    # mask: 先转 float [0,1]，同一 mat + 最近邻 + constant-zero border（同 joint_affine_floor）
    m = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    warped_mask = cv2.warpAffine(
        m, mat, (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    # soft target：original/warped mask 的凸组合，保留 [0,1]，不 threshold
    mask_out = alpha * warped_mask + (1.0 - alpha) * m

    # image 回 uint8；mask 用 PIL "F" 模式保留 float32 [0,1]（_mask_tensor soft 通道原样读取）
    return (
        Image.fromarray(np.clip(img_out, 0, 255).astype(np.uint8)),
        Image.fromarray(mask_out.astype(np.float32), mode="F"),
    )


def _apply_spatial_affine_warp(img: Image.Image) -> Image.Image:
    """空间仿射 warp 增广（scale + 平移，再与原图 alpha 混合）——**我们自己的空间几何扰动**。

    ⚠ 命名史：此函数曾名 `_apply_slaug_like`，并被 method "SLAug" 使用，但它与官方 SLAug
    **机制上毫无关系**。官方 SLAug 的 "location-scale" 是统计学的位置-尺度参数（即强度偏移与增益，
    见 `_apply_slaug_official_views`），是**强度/外观域**增广；而本函数做的是 `cv2.warpAffine`
    **空间几何**变换。旧名属张冠李戴，已正名为 spatial_warp_aug。

    ⚠ image-only：不改 mask，靠 alpha 混合让 GT 近似有效（继承自原实现的取舍，非本次引入）。
    """
    arr = np.asarray(img)
    return _apply_spatial_warp_with_params(img, _sample_spatial_warp_params(arr.shape[0], arr.shape[1]))


# 向后兼容旧调用点/旧 run 复现（旧名不再用于新臂）
_apply_slaug_like = _apply_spatial_affine_warp


def _get_official_slaug_augmenter():
    """惰性构造官方 `LocationScaleAugmentation`（Bezier 多项式表构造较贵，进程内复用一份）。

    直接 import 官方原版实现，不手抄——忠实复现的关键是逐行同源。
    """
    global _SLAUG_OFFICIAL_AUG
    if _SLAUG_OFFICIAL_AUG is None:
        add_official_slaug_to_path()
        from dataloaders.location_scale_augmentation import LocationScaleAugmentation  # 官方原版

        _SLAUG_OFFICIAL_AUG = LocationScaleAugmentation(vrange=(0.0, 1.0), background_threshold=0.01)
    return _SLAUG_OFFICIAL_AUG


def _apply_slaug_official_views(img: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    """官方 SLAug 的 GLA / LLA 双视图（**强度域**增广，非空间几何）。

    - GLA = Global_Location_Scale_Augmentation：全图 Bezier 非线性强度映射 + 强度 location-scale。
    - LLA = Local_Location_Scale_Augmentation：按 mask 分区域（背景 / 各前景类）分别施加上述强度变换。
    官方 vrange=(0,1)，故这里按 /255 归一后再调用，回程还原到 uint8。
    """
    lsa = _get_official_slaug_augmenter()
    arr = np.asarray(img).astype(np.float32) / 255.0
    m = (np.asarray(mask.convert("L"), dtype=np.uint8) > 127).astype(np.int32)
    gla = lsa.Global_Location_Scale_Augmentation(arr.copy())
    lla = lsa.Local_Location_Scale_Augmentation(arr.copy(), m)

    def _to_pil(a: np.ndarray) -> Image.Image:
        return Image.fromarray(np.clip(np.asarray(a, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8))

    return _to_pil(gla), _to_pil(lla)


# ---- 隔离对照臂（TMI 要求的 A1>A2）：拆解 spatial_warp_aug 的成分，证明"是机制在起作用，不是白加一个扰动" ----
# A1 = spatial_warp_aug：scale∈[0.84,1.16] 随机 + shift±0.08 随机 + alpha∈[0.35,0.65] 随机混合。
# 下列 A2 变体各消解掉 A1 的**一个**成分。若某个 A2 与 A1 打平 ⇒ 该成分不是机制来源，claim 必须改写。
_WARP_ISOLATION = {
    # alpha 扫描（同时给出 TMI 要的"非单调最优"证据）
    "warp_alpha100": {"alpha": 1.00},  # 纯几何 warp，**不与原图混合**
    "warp_alpha015": {"alpha": 0.15},  # 几乎全是原图 ⇒ warp 影响极小；若它也一样好，warp 根本没用
    # 成分消解
    "warp_shift_only": {"no_scale": True},  # 去掉尺度扰动，只平移；若打平，"扁平=尺度问题"的叙事崩
}


def _apply_warp_isolation_variant(img: Image.Image, variant: str) -> Image.Image:
    """隔离对照：与 `_apply_spatial_affine_warp` 同一套采样，只消解指定成分。"""
    cfg = _WARP_ISOLATION[variant]
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    scale = 1.0 if cfg.get("no_scale") else random.uniform(0.84, 1.16)
    dx = random.uniform(-0.08, 0.08) * w
    dy = random.uniform(-0.08, 0.08) * h
    alpha = cfg["alpha"] if "alpha" in cfg else random.uniform(0.35, 0.65)
    return _apply_spatial_warp_with_params(img, (scale, dx, dy, alpha))


def _apply_spatial_warp_scale_adaptive(img: Image.Image, mask: Image.Image) -> Image.Image:
    """病灶自适应的空间 warp：对小/扁平病灶用更强的 scale/平移。

    与 `_apply_spatial_affine_warp` 同族（我们自己的空间几何扰动，非官方 SLAug），
    仅把扰动幅度调大：scale∈[0.7,1.3]、shift±0.12，前景面积越小幅度越大。
    image-only（不改 mask，靠 alpha 与原图混合保持 GT 近似有效）。
    """
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    mask_arr = np.asarray(mask.convert("L"), dtype=np.float32) if not isinstance(mask, np.ndarray) else mask
    fg_frac = float((mask_arr > 127).mean()) if mask_arr.size else 0.0
    # 前景占比小（小/扁平病灶）→ boost 大：fg_frac>=0.10 时 boost=0，fg_frac→0 时 boost=1。
    boost = float(np.clip(1.0 - fg_frac / 0.10, 0.0, 1.0))
    scale_dev = 0.30 + 0.15 * boost  # [0.30, 0.45] → scale∈[0.55,1.45]，基础 [0.7,1.3]
    shift_max = 0.12 + 0.06 * boost  # [0.12, 0.18]
    scale = random.uniform(1.0 - scale_dev, 1.0 + scale_dev)
    dx = random.uniform(-shift_max, shift_max) * w
    dy = random.uniform(-shift_max, shift_max) * h
    mat = np.array([[scale, 0, (1 - scale) * w / 2 + dx], [0, scale, (1 - scale) * h / 2 + dy]], dtype=np.float32)
    warped = cv2.warpAffine(arr, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    alpha = random.uniform(0.35, 0.65)
    out = alpha * warped + (1.0 - alpha) * arr
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# 向后兼容旧调用点/旧 run 复现（旧名不再用于新臂）
_apply_slaug_scale_adaptive = _apply_spatial_warp_scale_adaptive


def official_sbf_saliency(gradient: torch.Tensor, grid_size: int = SBF_GRID_SIZE) -> torch.Tensor:
    """官方 SLAug 的 Saliency-Balancing Fusion 显著图。

    逐行同官方 `saliency_balancing_fusion.get_SBF_map`（b-spline 平滑 + rescale 到 [0,1]），
    **唯一改动**：官方 `bspline_kernel_2d` 把 device 硬编码为 'cuda'，这里把 kernel 对齐到
    gradient 实际所在 device（否则 CPU smoke 直接挂）。数值行为不变。
    """
    add_official_slaug_to_path()
    from dataloaders.saliency_balancing_fusion import bspline_kernel_2d, rescale_intensity  # 官方原版

    _b, _c, h, w = gradient.size()
    spacing = [h // grid_size, h // grid_size]
    # 官方 get_bspline_kernel 调 bspline_kernel_2d 时未传 device，落到默认 device='cuda'（CPU 下直接 assert 崩）。
    # 故绕开 get_bspline_kernel 这层壳，直接调官方 bspline_kernel_2d 并显式传 device；
    # padding 计算逐行照抄官方 get_bspline_kernel（saliency_balancing_fusion.py:36-37）。kernel 数值与官方一致。
    bs_kernel = bspline_kernel_2d(spacing, order=2, asTensor=True, device=gradient.device)
    bs_pad = ((np.array(bs_kernel.size()[2:]) - 1) / 2).astype(dtype=int).tolist()
    saliency = F.adaptive_avg_pool2d(gradient, grid_size)
    saliency = F.conv_transpose2d(saliency, bs_kernel, padding=bs_pad, stride=h // grid_size)
    saliency = F.interpolate(saliency, size=(h, w), mode="bilinear", align_corners=True)
    return rescale_intensity(saliency)


def uses_dual_view_sbf(method: str) -> bool:
    """需要 GLA/LLA 双视图 + SBF two-pass 训练的 method（官方 SLAug 机制及其组合臂）。"""
    return method in {"SLAug_official", "slaug_official_plus_warp"}


def apply_train_aug_dual(img: Image.Image, mask: Image.Image, method: str) -> tuple[Image.Image, Image.Image, Image.Image]:
    """SLAug_official 专用：共享同一 strong-aug floor 底座，产出 GLA / LLA 双视图。

    floor 的几何+颜色在双视图分叉**之前**施加（两视图共享同一 GT）；blur 在分叉**之后**各自采样。
    与单视图臂保持同一 floor，满足 FAIR_BUDGET 对称（其余臂 = floor + 自身机制）。
    """
    img, mask = _joint_geometric_aug(img, mask)
    img = _color_jitter(img, 0.16)
    gla_img, lla_img = _apply_slaug_official_views(img, mask)  # 强度域（官方 SLAug）

    if method == "slaug_official_plus_warp":
        # 几何域（我们的 spatial_warp，与强度域**正交**）叠在官方强度双视图之上。
        # 两条硬约束，都不能改：
        #  1) warp 必须**后置**于 GLA/LLA —— LLA 要按 mask 分区做强度变换，先 warp 图像会让 mask 错位。
        #  2) 两个视图必须用**同一组 warp 参数** —— SBF 按显著图融合 GLA/LLA，
        #     视图间空间不对齐会让 `gla*saliency + lla*(1-saliency)` 融出鬼影。
        params = _sample_spatial_warp_params(gla_img.size[1], gla_img.size[0])
        gla_img = _apply_spatial_warp_with_params(gla_img, params)
        lla_img = _apply_spatial_warp_with_params(lla_img, params)

    if random.random() < 0.15:
        gla_img = gla_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
    if random.random() < 0.15:
        lla_img = lla_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
    return gla_img, lla_img, mask


# ============================ 官方 CCSDG 忠实复现 ============================
# 官方 = channel_prompt 通道解耦(content/style) + 三视图(原图/FDA/GLA) + Projector 对比一致性。
# 主表原 "CCSDG" 臂只有 RGB 随机 gain/offset，与此无关（已标 † 不承重）。
_CCSDG_GLA_AUG = None


def add_official_ccsdg_to_path() -> None:
    """把官方 CCSDG repo 加进 sys.path（import 其原版 FDA，不手抄）。"""
    ccsdg_root = S1_REPRO_REPOS / "CCSDG"
    if not ccsdg_root.exists():
        raise FileNotFoundError(f"官方 CCSDG repo 不存在: {ccsdg_root}（可用 S1_REPRO_REPOS 指定）")
    if str(ccsdg_root) not in sys.path:
        sys.path.insert(0, str(ccsdg_root))


def uses_ccsdg_triview(method: str) -> bool:
    """需要 CCSDG 三视图 + channel_prompt 对比训练的 method。

    - `CCSDG_port_pvt`   : 机制移植到 polyp_pvt（进主表，与其它臂 backbone 对称、可归因于机制）
    - `CCSDG_official`   : 官方 UNetCCSDG(ResNet-UNet) 原架构（**忠实度锚**，不与 polyp_pvt 臂同表比较）
    """
    return method in {"CCSDG_port_pvt", "CCSDG_official"}


def _get_ccsdg_gla_augmenter():
    """官方 CCSDG 的 GLA 直接复用 SLAug 的 LocationScaleAugmentation，但 vrange=(0,255)。

    见官方 CCSDG `datasets/utils/transform.py:19`（它自己 import 了 slaug）。
    """
    global _CCSDG_GLA_AUG
    if _CCSDG_GLA_AUG is None:
        add_official_slaug_to_path()
        from dataloaders.location_scale_augmentation import LocationScaleAugmentation  # 官方原版

        _CCSDG_GLA_AUG = LocationScaleAugmentation(vrange=(0.0, 255.0), background_threshold=0.01)
    return _CCSDG_GLA_AUG


def _apply_ccsdg_gla(img: Image.Image) -> Image.Image:
    """CCSDG 的 GLA 视图（官方对比训练只取 GLA，不取 LLA）。"""
    lsa = _get_ccsdg_gla_augmenter()
    arr = np.asarray(img).astype(np.float32)  # vrange=(0,255)，不归一化
    gla = lsa.Global_Location_Scale_Augmentation(arr.copy())
    return Image.fromarray(np.clip(np.asarray(gla, dtype=np.float32), 0, 255).astype(np.uint8))


def ccsdg_fda_batch(batch_t: torch.Tensor, fda_beta: float = 0.1) -> torch.Tensor:
    """官方 CCSDG 的 FDA 视图：用 **batch 反转** 当 target 做低频振幅交换。

    逐行同官方 `transform.fourier_augmentation_reverse`（beta 采样 + `data[::-1]` 作 target）
    并调官方 `fourier.FDA_source_to_target_np`。
    ⚠ FDA 是 **batch 级**操作（target 来自同 batch 的其它样本），因此**只能在训练循环里做**，
    放不进单样本 Dataset —— 这是它与 GLA 的结构性差异。
    """
    add_official_ccsdg_to_path()
    from ccsdg.utils.fourier import FDA_source_to_target_np  # 官方原版

    data = batch_t.detach().cpu().numpy()
    this_beta = round(0.05 + np.random.random() * fda_beta, 2)  # 官方 transform.py:12
    lowf_batch = data[::-1]  # 官方 transform.py:13
    fda = FDA_source_to_target_np(data, lowf_batch, L=this_beta)
    return torch.from_numpy(np.ascontiguousarray(fda)).to(device=batch_t.device, dtype=batch_t.dtype)


class CCSDGProjector(nn.Module):
    """官方 CCSDG Projector（unet_ccsdg.py:9）。

    官方 `fc` 输入维度 131072 = 8×128×128，由其 first_layer 输出 256×256 决定——**是尺寸的结果，不是超参**。
    本移植按实际 stage1 特征尺寸计算（352/4=88 → 8×44×44），机制不变。
    """

    def __init__(self, in_ch: int = 64, feat_hw: int = IMAGE_SIZE // 4, output_size: int = 1024) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 8, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(8 * (feat_hw // 2) * (feat_hw // 2), output_size)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.conv(x_in)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, dim=1)


class CCSDGPolypPVT(nn.Module):
    """CCSDG 机制移植到 Polyp-PVT（统一 backbone，保证与其它臂可比）。

    对应关系（官方 UNetCCSDG → 本移植）：
      官方  first_layer(ResNet conv1, 64ch) → channel_prompt softmax 解耦 → rn(f_content) → UNet decoder
      移植  PVT stage1(64ch；pvt_v2_b2 embed_dims[0]=64，与 ResNet conv1 同宽) → **同一** channel_prompt 机制
            → stage2-4 只走 f_content → PolypPVT 原 decoder(CIM/CFM/SAM)
    **分割只用 content 通道**，与官方一致。
    ⚠ 这是"机制移植到统一 backbone"，非官方架构原样；官方 ResNet 版另跑作忠实度锚。
    """

    def __init__(self, base: nn.Module, tau: float = 0.1) -> None:
        super().__init__()
        bb = getattr(base, "backbone", None)
        if bb is None or not hasattr(bb, "patch_embed1"):
            raise TypeError(f"CCSDGPolypPVT 需要 PolypPVT(含 .backbone=pvt_v2)，实际拿到 {type(base).__name__}")
        self.base = base
        self.tau = tau
        self.channel_prompt = nn.Parameter(torch.randn(2, 64, 1, 1))  # 同官方 unet_ccsdg.py:54

    def _stage1(self, x: torch.Tensor) -> torch.Tensor:
        bb = self.base.backbone
        b = x.shape[0]
        x, h, w = bb.patch_embed1(x)
        for blk in bb.block1:
            x = blk(x, h, w)
        x = bb.norm1(x)
        return x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

    def _split(self, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        onehot = torch.softmax(self.channel_prompt / self.tau, dim=0)  # 官方 unet_ccsdg.py:68
        f_content = x1 * onehot[0].view(1, *onehot[0].shape)
        f_style = x1 * onehot[1].view(1, *onehot[1].shape)
        return f_content, f_style

    def forward_first_layer(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """对应官方 UNetCCSDG.forward_first_layer：只到解耦，供对比 loss 用。"""
        return self._split(self._stage1(x))

    def _stages_2to4(self, x: torch.Tensor) -> list[torch.Tensor]:
        bb = self.base.backbone
        b = x.shape[0]
        outs: list[torch.Tensor] = []
        for pe, blocks, norm in (
            (bb.patch_embed2, bb.block2, bb.norm2),
            (bb.patch_embed3, bb.block3, bb.norm3),
            (bb.patch_embed4, bb.block4, bb.norm4),
        ):
            x, h, w = pe(x)
            for blk in blocks:
                x = blk(x, h, w)
            x = norm(x)
            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs

    def forward(self, x: torch.Tensor):
        f_content, _f_style = self._split(self._stage1(x))
        x2, x3, x4 = self._stages_2to4(f_content)  # 分割只走 content 分支（同官方）
        b = self.base
        x1 = b.ca(f_content) * f_content
        cim_feature = b.sa(x1) * x1
        cfm_feature = b.CFM(b.Translayer4_1(x4), b.Translayer3_1(x3), b.Translayer2_1(x2))
        t2 = b.down05(b.Translayer2_0(cim_feature))
        sam_feature = b.SAM(cfm_feature, t2)
        p1 = F.interpolate(b.out_CFM(cfm_feature), scale_factor=8, mode="bilinear")
        p2 = F.interpolate(b.out_SAM(sam_feature), scale_factor=8, mode="bilinear")
        return p1, p2


def ccsdg_contrastive_loss(model: nn.Module, projector: nn.Module, views: list[torch.Tensor]):
    """官方 CCSDG 对比 loss（train_unet_ccsdg.py:103-124）。

    content 跨视图两两 L1 **拉近**（最小化）；style 跨视图两两 L1 **推远**（取负号）。
    官方对 3 视图做对称成对共 6 项，这里逐项复现。
    """
    contents: list[torch.Tensor] = []
    styles: list[torch.Tensor] = []
    for v in views:
        f_content, f_style = model.forward_first_layer(v)
        contents.append(projector(f_content))
        styles.append(projector(f_style))

    def _pairwise_sym_l1(feats: list[torch.Tensor]) -> torch.Tensor:
        total = feats[0].new_zeros(())
        for i in range(len(feats)):
            for j in range(len(feats)):
                if i != j:
                    total = total + F.l1_loss(feats[i], feats[j], reduction="mean")
        return total

    content_loss = _pairwise_sym_l1(contents)
    style_loss = -_pairwise_sym_l1(styles)  # 官方 train_unet_ccsdg.py:124
    return content_loss, style_loss


def apply_train_aug_ccsdg(img: Image.Image, mask: Image.Image, method: str) -> tuple[Image.Image, Image.Image, Image.Image]:
    """CCSDG 专用：共享同一 strong-aug floor 底座，产 (原图, GLA) 两图。

    第三个视图 FDA 是 batch 级操作（见 `ccsdg_fda_batch`），在训练循环里算，不在这里。
    """
    img, mask = _joint_geometric_aug(img, mask)
    img = _color_jitter(img, 0.16)
    gla_img = _apply_ccsdg_gla(img)
    if random.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
    if random.random() < 0.15:
        gla_img = gla_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
    return img, gla_img, mask


def _apply_bias_field(img: Image.Image) -> Image.Image:
    arr = np.asarray(img).astype(np.float32) / 255.0
    h, w = arr.shape[:2]
    low = np.random.uniform(0.75, 1.25, size=(4, 4)).astype(np.float32)
    field = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(h, w) / 10.0)
    field = field / max(float(field.mean()), 1e-6)
    out = np.clip(arr * field[..., None], 0.0, 1.0)
    return Image.fromarray((out * 255.0).astype(np.uint8))


def _apply_channel_style(img: Image.Image) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    gains = np.random.uniform(0.82, 1.18, size=(1, 1, 3)).astype(np.float32)
    offsets = np.random.uniform(-8, 8, size=(1, 1, 3)).astype(np.float32)
    return Image.fromarray(np.clip(arr * gains + offsets, 0, 255).astype(np.uint8))


def _apply_fourier_amplitude_exchange(img: Image.Image) -> Image.Image:
    """Fourier 振幅交换 cheap-control：低频相位保留，高频振幅扰动。"""
    arr = np.asarray(img).astype(np.float32) / 255.0
    freq = np.fft.fft2(arr, axes=(0, 1))
    amp = np.abs(freq)
    phase = np.angle(freq)
    h, w = arr.shape[:2]
    donor = np.roll(arr, shift=random.randint(3, max(4, w // 12)), axis=1)
    donor_freq = np.fft.fft2(donor, axes=(0, 1))
    donor_amp = np.abs(donor_freq)
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    high_mask = np.fft.ifftshift((radius > min(h, w) * 0.18).astype(np.float32))[..., None]
    lam = random.uniform(0.25, 0.55)
    mixed_amp = amp * (1.0 - lam * high_mask) + donor_amp * (lam * high_mask)
    mixed = np.fft.ifft2(mixed_amp * np.exp(1j * phase), axes=(0, 1)).real
    return Image.fromarray(np.clip(mixed * 255.0, 0, 255).astype(np.uint8))


def spectral_tensor_perturb(x: torch.Tensor, high_dropout: float = 0.18) -> torch.Tensor:
    """频谱扰动 consistency 分支：保相位，随机扰动振幅并 dropout 高频。"""
    freq = torch.fft.rfft2(x, norm="ortho")
    amp = torch.abs(freq)
    phase = torch.angle(freq)
    b, c, h, w_freq = freq.shape
    h_full = x.shape[-2]
    w_full = x.shape[-1]
    fy = torch.fft.fftfreq(h_full, device=x.device).view(1, 1, h_full, 1)
    fx = torch.fft.rfftfreq(w_full, device=x.device).view(1, 1, 1, w_freq)
    radius = (fy.square() + fx.square()).sqrt()
    high = radius > 0.18
    amp_noise = torch.empty((b, c, 1, 1), device=x.device, dtype=x.dtype).uniform_(0.85, 1.15)
    dropout = (torch.rand((b, c, h, w_freq), device=x.device) > high_dropout).to(x.dtype)
    mixed_amp = amp * amp_noise * torch.where(high, dropout, torch.ones_like(dropout))
    perturbed = torch.fft.irfft2(mixed_amp * torch.exp(1j * phase), s=x.shape[-2:], norm="ortho")
    return perturbed.clamp(float(x.min().detach()), float(x.max().detach()))


def _affine_perturb_grid(
    x: torch.Tensor, scale: torch.Tensor, tx: torch.Tensor, ty: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 per-sample scale / 平移构造仿射 theta，采样 x 并返回 (perturbed, grid)。

    theta 把输出归一化坐标映射到输入归一化坐标：out=p 对应输入 scale*p+t。
    identity（scale=1, t=0）时 grid 即恒等网格，perturbed≈x。
    """
    b = x.shape[0]
    theta = torch.zeros(b, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    perturbed = F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")
    return perturbed, grid


def spatial_tensor_perturb(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """空间等变一致性分支：每样本随机 scale∈[0.85,1.15]、平移∈[-0.08,0.08] 的仿射扰动。

    返回 (perturbed, grid)；grid 供一致性把原图 outputs warp 到扰动坐标系，做等变对齐。
    """
    b = x.shape[0]
    scale = torch.empty(b, device=x.device, dtype=x.dtype).uniform_(0.85, 1.15)
    tx = torch.empty(b, device=x.device, dtype=x.dtype).uniform_(-0.08, 0.08)
    ty = torch.empty(b, device=x.device, dtype=x.dtype).uniform_(-0.08, 0.08)
    return _affine_perturb_grid(x, scale, tx, ty)


def warp_logits_with_grid(logits: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """用同一 grid 把原图 logits warp 到扰动坐标系（等变对齐）。

    grid 编码的是归一化坐标下的几何变换，与 logits 空间分辨率无关：输出空间尺寸 = grid 的 H×W，
    在 logits 上按归一化坐标采样。调用前需保证 grid 的 H×W 与目标输出尺寸一致（见 _resize_grid_to）。
    """
    return F.grid_sample(logits, grid, align_corners=False)


def _resize_grid_to(grid: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """把 [B,H,W,2] 采样 grid 重采样到目标空间尺寸。

    affine grid 的归一化坐标是输出像素位置的线性函数，bilinear 重采样在内部近乎精确；
    formal run 中各张量尺寸一致（无 resize）时直接返回原 grid。
    """
    if tuple(grid.shape[1:3]) == tuple(size):
        return grid
    g = grid.permute(0, 3, 1, 2)  # [B,2,H,W]
    g = F.interpolate(g, size=size, mode="bilinear", align_corners=False)
    return g.permute(0, 2, 3, 1).contiguous()  # [B,size_h,size_w,2]


def apply_train_aug(img: Image.Image, mask: Image.Image, method: str) -> tuple[Image.Image, Image.Image]:
    """训练增强（单视图路径）。共享 floor = 几何 + 颜色抖动 + blur，各臂在其上叠自身机制。

    ⚠ 标 `faithful_light_reimpl` 的臂只是原论文思想的**轻量近似**，不得用于"打赢具名 SOTA"的 claim。
    官方忠实复现走 `*_official` 臂（SLAug_official 需双视图，见 `apply_train_aug_dual`）。
    """
    img, mask = _joint_geometric_aug(img, mask)
    img = _color_jitter(img, 0.16)
    if method in {"SLAug", "spatial_warp_aug"}:
        # "SLAug" 为历史误名（实为空间仿射 warp，与官方 SLAug 机制无关）；spatial_warp_aug 为其正名。
        # 二者指向同一实现，保证历史 run 可复现。官方 SLAug = SLAug_official（强度域 + SBF）。
        img = _apply_spatial_affine_warp(img)
    elif method == "CSDG":
        img = _apply_bias_field(img)
    elif method == "CCSDG":
        img = _apply_channel_style(img)
    elif method in {"MixStyle", "DSU"}:
        # MixStyle/DSU 主要在 feature hook 中注入，这里只保留强增强底座。
        pass
    elif method == "fourier_amp_aug":
        img = _apply_fourier_amplitude_exchange(img)
    elif method in {
        "slaug_boundary_consistency",
        "slaug_multiscale_consistency",
        "spatialwarp_boundary_consistency",
        "spatialwarp_multiscale_consistency",
    }:
        # twist：数据增广同 spatial_warp_aug 底座，一致性分支在训练循环里加。
        img = _apply_spatial_affine_warp(img)
    elif method in {"slaug_scale_adaptive", "spatialwarp_scale_adaptive"}:
        # twist：只改数据增广，对小/扁平病灶更强的空间扰动，不加一致性 loss。
        img = _apply_spatial_warp_scale_adaptive(img, mask)
    elif method in _WARP_ISOLATION:
        # A2 隔离对照：消解 spatial_warp_aug 的单个成分（TMI 的 A1>A2 门）
        img = _apply_warp_isolation_variant(img, method)
    elif method == "joint_affine_floor":
        # P01 隔离臂：image+mask 同步仿射（scale+shift，与 A1 同采样范围），无 alpha 混合。
        # 与上面 _WARP_ISOLATION（image-only）并列；本分支改 mask，故走 _apply_joint_affine。
        img, mask = _apply_joint_affine(img, mask)
    elif method == "paired_affine_softmix":
        # factorial 补格 U1B1：image+mask 同步仿射 + 同一 alpha 凸组合（soft target）。
        # image 处理与 A1 逐字相同，mask 为 [0,1] soft（_mask_tensor soft 通道保留，不二值化）。
        img, mask = _apply_paired_affine_softmix(img, mask)
    if random.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
    return img, mask


class FrequencyDecouplingWrapper(nn.Module):
    """M2 faithful-reimpl smoke：FFT 低/高频分离后做轻量 FAR refinement。"""

    def __init__(self, model: nn.Module, cutoff: float = 0.18) -> None:
        super().__init__()
        self.model = model
        self.cutoff = cutoff
        self.mix_logit = nn.Parameter(torch.tensor(0.0))
        self.alpha = nn.Parameter(torch.tensor(0.08))

    def forward(self, x: torch.Tensor) -> Any:
        h, w = x.shape[-2:]
        freq = torch.fft.rfft2(x, norm="ortho")
        fy = torch.fft.fftfreq(h, device=x.device).view(h, 1)
        fx = torch.fft.rfftfreq(w, device=x.device).view(1, -1)
        mask = ((fy.square() + fx.square()).sqrt() <= self.cutoff).to(freq.dtype)
        low = torch.fft.irfft2(freq * mask.view(1, 1, h, -1), s=(h, w), norm="ortho")
        high = x - low
        gate = torch.sigmoid(self.mix_logit)
        refined = x + self.alpha.tanh() * (gate * high + (1.0 - gate) * low)
        return self.model(refined)


class WaveletBoundaryWrapper(nn.Module):
    """M1 faithful-reimpl smoke：Haar-like 子带高频边界注入。"""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.alpha = nn.Parameter(torch.tensor(0.08))

    def forward(self, x: torch.Tensor) -> Any:
        low = F.avg_pool2d(x, kernel_size=2, stride=2)
        low_up = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)
        high = x - low_up
        refined = x + self.alpha.tanh() * high
        return self.model(refined)


def build_method_model(model: nn.Module, method: str) -> nn.Module:
    """按 method-dev arm 给 raw segmentation model 加机制壳。"""
    if method == "m2_fad_far":
        return FrequencyDecouplingWrapper(model)
    if method == "m1_wavelet_boundary":
        return WaveletBoundaryWrapper(model)
    if method == "sam2_plain_frozen":
        zero_and_freeze_sam2_adapters(model)
    if method == "CCSDG_port_pvt":
        return CCSDGPolypPVT(model)  # channel_prompt 解耦壳（挂 polyp_pvt stage1）
    # CCSDG_official：官方 UNetCCSDG 自带 channel_prompt + forward_first_layer，无需包壳。
    return model


class MixStyleHook(nn.Module):
    """轻量 MixStyle feature perturbation。"""

    def __init__(self, p: float = 0.5, alpha: float = 0.1) -> None:
        super().__init__()
        self.p = p
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or random.random() > self.p or x.size(0) < 2:
            return x
        mu = x.mean(dim=[2, 3], keepdim=True)
        sig = (x.var(dim=[2, 3], keepdim=True) + 1e-6).sqrt()
        x_norm = (x - mu) / sig
        perm = torch.randperm(x.size(0), device=x.device)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((x.size(0), 1, 1, 1)).to(x.device)
        mu_mix = mu * lam + mu[perm] * (1.0 - lam)
        sig_mix = sig * lam + sig[perm] * (1.0 - lam)
        return x_norm * sig_mix + mu_mix


class DSUHook(nn.Module):
    """Distribution Uncertainty feature perturbation。"""

    def __init__(self, p: float = 0.5, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or random.random() > self.p or x.size(0) < 2:
            return x
        mu = x.mean(dim=[2, 3], keepdim=True)
        sig = (x.var(dim=[2, 3], keepdim=True) + self.eps).sqrt()
        mu_std = mu.std(dim=0, keepdim=True)
        sig_std = sig.std(dim=0, keepdim=True)
        beta = torch.randn_like(mu) * mu_std + mu
        gamma = torch.randn_like(sig) * sig_std + sig
        return (x - mu) / sig * gamma + beta


class InstanceWhiteningHook(nn.Module):
    """实例白化 smoke hook：去除样本内 feature style 统计，保留部分原特征。"""

    def __init__(self, blend: float = 0.35, eps: float = 1e-6) -> None:
        super().__init__()
        self.blend = blend
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        mu = x.mean(dim=[2, 3], keepdim=True)
        sig = (x.var(dim=[2, 3], keepdim=True, unbiased=False) + self.eps).sqrt()
        whitened = (x - mu) / sig
        return x * (1.0 - self.blend) + whitened * self.blend


class ISWFullHook(nn.Module):
    """RobustNet ISW smoke：选择高方差通道做实例白化，并弱化通道协方差。"""

    def __init__(self, blend: float = 0.55, select_ratio: float = 0.5, eps: float = 1e-6) -> None:
        super().__init__()
        self.blend = blend
        self.select_ratio = select_ratio
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or x.dim() != 4:
            return x
        b, c, h, w = x.shape
        flat = x.flatten(2)
        mu = flat.mean(dim=2, keepdim=True)
        centered = flat - mu
        sig = (centered.var(dim=2, keepdim=True, unbiased=False) + self.eps).sqrt()
        normalized = centered / sig
        k = max(1, int(c * self.select_ratio))
        style_var = sig.squeeze(-1).mean(dim=0)
        selected = torch.topk(style_var, k=k, largest=True).indices
        mixed = flat.clone()
        mixed[:, selected, :] = (
            flat[:, selected, :] * (1.0 - self.blend) + normalized[:, selected, :] * self.blend
        )
        return mixed.view(b, c, h, w)


def register_feature_hook(model: nn.Module, method: str) -> list[torch.utils.hooks.RemovableHandle]:
    """给 feature-space 方法注册第一个 4D feature hook。"""
    if method not in {"MixStyle", "DSU", "ibn_whitening", "isw_full", "spectral_ibn_combo"}:
        return []
    if method == "MixStyle":
        injector: nn.Module = MixStyleHook()
    elif method == "DSU":
        injector = DSUHook()
    elif method == "isw_full":
        injector = ISWFullHook()
    else:
        injector = InstanceWhiteningHook()
    injector.train()
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def hook(_module: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> torch.Tensor:
        if isinstance(out, torch.Tensor) and out.dim() == 4:
            return injector(out)
        return out

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            handles.append(module.register_forward_hook(hook))
            break
    return handles


def add_repro_to_path() -> None:
    for path in (REPRO_ROOT,):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault("POLYP_THIRD_PARTY", str(THIRD_PARTY_ROOT))


def _add_sam2_unet_to_path() -> Path:
    sam_root = THIRD_PARTY_ROOT / "SAM2-UNet"
    if str(sam_root) not in sys.path:
        sys.path.insert(0, str(sam_root))
    return sam_root


def zero_and_freeze_sam2_adapters(model: nn.Module) -> None:
    """把 SAM2-UNet Adapter prompt 置零冻结，作为 plain frozen encoder 对照。"""
    for module in model.modules():
        prompt = getattr(module, "prompt_learn", None)
        if prompt is None:
            continue
        for sub in prompt.modules():
            if isinstance(sub, nn.Linear):
                nn.init.zeros_(sub.weight)
                if sub.bias is not None:
                    nn.init.zeros_(sub.bias)
        for param in prompt.parameters():
            param.requires_grad = False


def build_model(model_key: str, device: str) -> tuple[nn.Module, Any]:
    """构建原仓 raw model 和对应 adapter。"""
    if model_key == "sam2_unet":
        sam_root = _add_sam2_unet_to_path()
        from SAM2UNet import SAM2UNet

        ckpt = sam_root / "sam2_hiera_large.pt"
        raw_model = SAM2UNet(str(ckpt)).to(device)
        return raw_model, None

    if model_key == "unet_ccsdg":
        # 官方 CCSDG 原架构（ResNet-UNet，自带 channel_prompt）——作**忠实度锚**，不与 polyp_pvt 臂同表比较。
        add_official_ccsdg_to_path()
        from ccsdg.models.unet_ccsdg import UNetCCSDG  # 官方原版

        raw_model = UNetCCSDG(resnet="resnet34", num_classes=1, pretrained=True).to(device)
        return raw_model, None

    add_repro_to_path()
    from adapters.sota_adapters import REGISTRY

    if model_key not in {"pranet", "polyp_pvt"}:
        raise ValueError(f"smoke 只启用 PraNet/Polyp-PVT/SAM2-UNet/UNetCCSDG，got {model_key}")
    adapter = REGISTRY[model_key](device=device)
    raw_model = adapter._model.model.to(device)
    return raw_model, adapter


def fuse_outputs(model_key: str, outputs: Any) -> torch.Tensor:
    if model_key == "pranet":
        return outputs[-1]
    if model_key == "polyp_pvt":
        return outputs[0] + outputs[1]
    if isinstance(outputs, torch.Tensor):
        return outputs
    return outputs[0]


def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """PraNet wBCE + wIoU。"""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred_sigmoid = torch.sigmoid(pred)
    inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def _soft_skeleton(prob: torch.Tensor, iterations: int = 8) -> torch.Tensor:
    """soft skeletonization，用于 clDice smoke loss。"""
    img = prob
    skeleton = torch.zeros_like(img)
    for _ in range(iterations):
        eroded = -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)
        opened = F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)
        delta = F.relu(img - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
        img = eroded
    return skeleton


def soft_cldice_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """soft-clDice 损失，约束预测与 GT 的连通 skeleton 重合。"""
    if pred.shape[-2:] != mask.shape[-2:]:
        pred = F.interpolate(pred, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    prob = torch.sigmoid(pred)
    pred_skel = _soft_skeleton(prob)
    mask_skel = _soft_skeleton(mask)
    tprec = (pred_skel * mask).sum(dim=(2, 3)) / (pred_skel.sum(dim=(2, 3)) + 1e-6)
    tsens = (mask_skel * prob).sum(dim=(2, 3)) / (mask_skel.sum(dim=(2, 3)) + 1e-6)
    cldice = (2.0 * tprec * tsens) / (tprec + tsens + 1e-6)
    return (1.0 - cldice).mean()


def deep_supervision_loss(model_key: str, outputs: Any, mask: torch.Tensor, method: str = "") -> torch.Tensor:
    preds = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
    if model_key == "polyp_pvt":
        preds = preds[:2]
    loss = sum(structure_loss(pred, mask) for pred in preds)
    if method == "topo_cldice":
        loss = loss + 0.2 * sum(soft_cldice_loss(pred, mask) for pred in preds)
    return loss


def consistency_loss(outputs_a: Any, outputs_b: Any, model_key: str, mask: torch.Tensor) -> torch.Tensor:
    """预测一致性正则，用于频谱外变换 consistency arm。"""
    logits_a = fuse_outputs(model_key, outputs_a)
    logits_b = fuse_outputs(model_key, outputs_b)
    if logits_a.shape[-2:] != mask.shape[-2:]:
        logits_a = F.interpolate(logits_a, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    if logits_b.shape[-2:] != mask.shape[-2:]:
        logits_b = F.interpolate(logits_b, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    return F.mse_loss(torch.sigmoid(logits_a), torch.sigmoid(logits_b))


def boundary_weighted_consistency_loss(
    outputs_a: Any, outputs_b: Any, model_key: str, mask: torch.Tensor, grid: torch.Tensor
) -> torch.Tensor:
    """等变边界一致性正则（twist build-on-SLAug）。

    outputs_a = 原图预测，outputs_b = 空间扰动图预测，grid = 生成扰动图所用采样网格。
    等变对齐：把原图 logits `la` 用同一 grid warp 到扰动坐标系（la_w = W[la]），再与扰动图
    预测 lb 比较——两者此时都在扰动坐标系下。边界带权重 band 也 warp 到同坐标系后加权。
    模型若对该空间变换等变，则 model(W[img])≈W[model(img)]，即 lb≈la_w。
    """
    la = fuse_outputs(model_key, outputs_a)
    lb = fuse_outputs(model_key, outputs_b)
    size = tuple(mask.shape[-2:])
    if tuple(la.shape[-2:]) != size:
        la = F.interpolate(la, size=size, mode="bilinear", align_corners=False)
    if tuple(lb.shape[-2:]) != size:
        lb = F.interpolate(lb, size=size, mode="bilinear", align_corners=False)
    # grid 按原图尺寸构建；warp 目标是 mask 尺寸的 la，故把 grid 重采样到 mask 尺寸再采样。
    grid_m = _resize_grid_to(grid, size)
    la_w = warp_logits_with_grid(la, grid_m)  # 原图预测 → 扰动坐标系（等变对齐）
    # 边界带：膨胀减腐蚀得 1px 环，再外扩几像素；同样 warp 到扰动坐标系与 la_w/lb 对齐。
    dil = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    ero = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    band = dil - ero
    band = F.max_pool2d(band, kernel_size=5, stride=1, padding=2)
    band_w = warp_logits_with_grid(band, grid_m)
    diff = (torch.sigmoid(la_w) - torch.sigmoid(lb)) ** 2
    return (band_w * diff).sum() / (band_w.sum() + 1e-6)


def multiscale_consistency_loss(
    model: nn.Module,
    image: torch.Tensor,
    model_key: str,
    mask: torch.Tensor,
    scales: tuple[float, ...] = (0.75, 1.25),
    base_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """多尺度一致性正则（twist build-on-SLAug）。

    对每个 scale：把原图 resize 后过模型，预测再插值回 mask 尺寸，与原图预测（base_logits，
    未传则重算一次）算 sigmoid-MSE；多个 scale 求平均。约束分割对全局缩放近似不变。
    """
    size = tuple(mask.shape[-2:])
    if base_logits is None:
        base_logits = fuse_outputs(model_key, model(image))
    if tuple(base_logits.shape[-2:]) != size:
        base_logits = F.interpolate(base_logits, size=size, mode="bilinear", align_corners=False)
    base_prob = torch.sigmoid(base_logits)
    losses: list[torch.Tensor] = []
    for s in scales:
        img_s = F.interpolate(
            image, scale_factor=float(s), mode="bilinear", align_corners=False, recompute_scale_factor=False
        )
        log_s = fuse_outputs(model_key, model(img_s))
        if tuple(log_s.shape[-2:]) != size:
            log_s = F.interpolate(log_s, size=size, mode="bilinear", align_corners=False)
        losses.append(F.mse_loss(torch.sigmoid(log_s), base_prob))
    return torch.stack(losses).mean()


@torch.no_grad()
def batch_dice_from_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != mask.shape[-2:]:
        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = 2.0 * (pred * mask).sum(dim=(2, 3))
    den = pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3)) + 1e-7
    return (inter / den).flatten()


def summarize_counts(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    out: dict[tuple[Any, ...], int] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        out[key] = out.get(key, 0) + 1
    return [{**{k: key[i] for i, k in enumerate(keys)}, "n": n} for key, n in sorted(out.items())]


def method_manifest() -> list[dict[str, str]]:
    return [
        {"method": "strong_aug_floor", "status": "baseline", "note": "强增强底座；不对应单篇 DG 方法。"},
        {"method": "m2_fad_far", "status": "faithful_reimpl", "note": "M2 频率解耦：FFT FAD + 轻量 FAR refinement，叠加强增强地板。"},
        {"method": "m1_wavelet_boundary", "status": "faithful_reimpl", "note": "M1 小波边界：Haar-like 高频子带边界注入，叠加强增强地板。"},
        {"method": "topo_cldice", "status": "faithful_reimpl", "note": "Topo 连通：结构损失 + soft-clDice topology loss，PH 正则未启用。"},
        {"method": "ibn_whitening", "status": "faithful_reimpl", "note": "实例白化：encoder feature instance whitening hook，叠加强增强地板。"},
        {"method": "fourier_amp_aug", "status": "cheap_control", "note": "Fourier 振幅交换增强 cheap-control；不是机制 arm。"},
        {"method": "isw_full", "status": "faithful_reimpl_degraded", "note": "RobustNet ISW 家族：本地无官方 RobustNet repo，按 instance-selective whitening 机制重实现。"},
        {"method": "sam2_adapter_dg", "status": "faithful_reimpl", "note": "SAM2-UNet：冻结 SAM2-L encoder + adapter/decoder；配 sam2_plain_frozen 对照。"},
        {"method": "sam2_plain_frozen", "status": "aux_control", "note": "SAM2-UNet plain 控制：prompt adapter 置零冻结，仅训 decoder/RFB/head。"},
        {"method": "spectral_consistency", "status": "faithful_reimpl", "note": "频谱扰动一致性：原图/谱扰动图预测一致性正则，叠加强增强地板。"},
        {"method": "spectral_ibn_combo", "status": "faithful_reimpl_combo", "note": "谱扰动一致性 + IBN feature instance whitening 组合；两个单臂超参保持 batch2 值，不为组合重调。"},
        # ===== 忠实复现的官方 SOTA：可承重，可进"打赢具名 SOTA"主表 =====
        {
            "method": "SLAug_official",
            "status": "faithful_official",
            "note": (
                "官方 SLAug 忠实复现（import 官方源码，非手抄）：GLA/LLA 双视图（**强度域** location-scale"
                "＝强度偏移+增益，配 Bezier 非线性强度映射，按 mask 分区施加）+ 官方 SBF 显著性融合 two-pass 训练。"
                "⚠ SBF grid_size 官方无 polyp 配置（腹部=3/心脏=18），本仓默认 8 为**我们自定**（S1_SBF_GRID_SIZE 可覆盖）；"
                "若本臂异常低于地板，先查该超参再下结论。"
            ),
        },
        {
            "method": "CCSDG_official",
            "status": "faithful_official_anchor",
            "note": (
                "官方 CCSDG 原架构 UNetCCSDG(ResNet34-UNet, num_classes=1)，import 官方源码原样跑。"
                "**用途 = 忠实度锚**:证明我们没把 CCSDG 复现坏。"
                "⚠ backbone 与其它臂(polyp_pvt)不同 ⇒ **不得与 polyp_pvt 臂同表做公平比较**"
                "（否则混淆 backbone 与 DG 机制）。公平比较用 CCSDG_port_pvt。"
                "跑法:--model unet_ccsdg --method CCSDG_official。"
            ),
        },
        {
            "method": "CCSDG_port_pvt",
            "status": "faithful_mechanism_port",
            "note": (
                "官方 CCSDG 机制移植到统一 backbone(polyp_pvt)：channel_prompt(2,64,1,1) softmax 解耦 content/style"
                "（挂 PVT stage1，embed_dims[0]=64 与官方 ResNet conv1 同宽）+ 三视图(原图/官方FDA/官方GLA)"
                " + 官方 Projector 对比一致性(content 拉近、style 推远)；分割只走 content 分支。"
                "⚠ 非官方架构原样（官方是 ResNet-UNet）——移植是为了与其它臂 backbone 对称、使比较可归因于机制而非 backbone；"
                "官方 ResNet 版另跑作忠实度锚。"
                "⚠⚠ **训练预算 GAP**：官方每 batch 做 1 次 prompt 更新 + **3 次分割更新**(三视图各一次，"
                "train_unet_ccsdg.py:130-158)，即有效更新次数 ≈ 其它臂的 3 倍。此为官方设计，本移植照搬；"
                "**比较时必须计入，不得当作同预算对比**。"
            ),
        },
        # ===== 我们自己的方法（正名：此前误挂 "SLAug" 之名）=====
        {
            "method": "spatial_warp_aug",
            "status": "ours_spatial",
            "note": (
                "**我们自己的**空间仿射 warp 增广（cv2.warpAffine scale+平移，再与原图 alpha 混合）。"
                "与官方 SLAug 机制无关（那是强度域）。旧名 'SLAug' 系张冠李戴，此为正名；实现与旧臂同一份，历史 run 可复现。"
            ),
        },
        {
            "method": "slaug_official_plus_warp",
            "status": "ours_combo",
            "note": (
                "组合臂：官方 SLAug（强度域 GLA/LLA + SBF）为底座，叠我们的 spatial_warp（几何域）。"
                "施加顺序有两条硬约束：① warp 必须**后置**于 GLA/LLA（LLA 按 mask 分区做强度变换，先 warp 会让 mask 错位）；"
                "② 两视图共用**同一组 warp 参数**（SBF 按显著图融合 GLA/LLA，空间不对齐会融出鬼影）。"
                "⚠ 用途=回答'几何能否在最强底座上再加分'，**不是**已证明的'机制正交'——两机制只在实现上作用于不同域，"
                "效果可加性未验证，禁止写成 'orthogonal'。"
            ),
        },
        # ===== A2 隔离对照（TMI 的 A1>A2 门；不是方法，是**用来杀死自己**的对照）=====
        {"method": "warp_alpha100", "status": "isolation_control", "note": "A2：alpha=1.0，纯几何 warp 不与原图混合。若优于 A1 ⇒ 几何是真机制、混合在削弱它；若崩 ⇒ 起作用的可能是混合产生的重影/模糊。"},
        {"method": "warp_alpha015", "status": "isolation_control", "note": "A2：alpha=0.15，几乎全是原图（warp 影响极小）。**若它与 A1 打平 ⇒ warp 根本没起作用，几何机制 claim 当场崩。**"},
        {"method": "warp_shift_only", "status": "isolation_control", "note": "A2：去掉 scale，只随机平移。**若它与 A1 打平 ⇒ '尺度'不是关键成分，'扁平=尺度/形态问题'的叙事崩。**"},
        {
            "method": "joint_affine_floor",
            "status": "joint_geometry_control",
            "note": (
                "P01 隔离臂：对 image 和 mask 施加**同一个**仿射（scale∈[0.84,1.16]、shift±0.08×(W,H)，"
                "与 A1 采样范围逐字一致），**不做 alpha 混合**，mask 最近邻同步。"
                "回答‘A1 相对 floor 的增益里，有多少只是补回地板缺失的标准 joint scale/shift’。"
                "⚠ 与 A1 同时在 mask 同步与 alpha blend 两处不同，不得把剩余成分单独命名为‘标签错位机制’。"
            ),
        },
        {
            "method": "paired_affine_softmix",
            "status": "posthoc_factorial_control",
            "note": (
                "factorial 补格 U1B1（paired target + alpha blend）：image 与 mask 用同一仿射 + 同一 alpha 凸组合，"
                "image 分支与 A1 逐字相同，mask 为 [0,1] soft target（不阈值化）。"
                "描述性 post-hoc control，**非营销方法名**；旧 test endpoint 已看过，论文中不得冒充 original preregistered control。"
                "⚠ 与 A1 同时在 mask 同步与 alpha blend 两处不同，不得命名为‘标签错位机制’；若与 A1 打平只能按 CI 报告未区分。"
            ),
        },
        {"method": "spatialwarp_boundary_consistency", "status": "ours_twist", "note": "build-on spatial_warp_aug：等变边界一致性——原图预测经同一空间扰动 grid warp 到扰动坐标系，与扰动图预测在边界带加权比较。"},
        {"method": "spatialwarp_multiscale_consistency", "status": "ours_twist", "note": "build-on spatial_warp_aug：多尺度一致性——原图缩放(0.75/1.25)预测插值回原尺寸与原图预测 MSE，约束尺度不变。"},
        {"method": "spatialwarp_scale_adaptive", "status": "ours_twist", "note": "build-on spatial_warp_aug：只改数据增广，对小/扁平病灶用更强空间扰动(scale∈[0.7,1.3]、shift±0.12，前景越小越强)，无一致性 loss。"},
        # ===== 轻量近似：† 不承重，只能作覆盖 caveat，禁止挂主 claim =====
        {
            "method": "SLAug",
            "status": "MISNAMED_light_reimpl",
            "note": (
                "†⚠ **历史误名**：本臂实现是空间仿射 warp（= spatial_warp_aug），**不是官方 SLAug**"
                "（官方为强度域 location-scale + SBF，见 SLAug_official）。保留仅为历史 run 留痕，"
                "**禁止用于任何与 SLAug 相关的对比或 claim**。"
            ),
        },
        {"method": "CCSDG", "status": "faithful_light_reimpl", "note": "†单味轻量近似（仅 RGB 随机 gain/offset），非官方 CCSDG（channel-prompt 解耦+FDA+对比一致性）。不承重。"},
        {"method": "CSDG", "status": "faithful_light_reimpl", "note": "†bias-field perturbation 思想的轻量重实现。不承重。"},
        {"method": "MixStyle", "status": "faithful_light_reimpl", "note": "†feature MixStyle hook。不承重。"},
        {"method": "DSU", "status": "faithful_light_reimpl", "note": "†feature Distribution Uncertainty hook。不承重。"},
        # ===== 旧 twist 名：deprecated 别名，指向同一实现（服务器 _twist_reference/ 仍用旧名）=====
        {"method": "slaug_boundary_consistency", "status": "deprecated_alias", "note": "⚠ 已正名为 spatialwarp_boundary_consistency（build 的是我们自己的空间 warp，不是官方 SLAug）。别名保留，勿用于新实验。"},
        {"method": "slaug_multiscale_consistency", "status": "deprecated_alias", "note": "⚠ 已正名为 spatialwarp_multiscale_consistency。别名保留，勿用于新实验。"},
        {"method": "slaug_scale_adaptive", "status": "deprecated_alias", "note": "⚠ 已正名为 spatialwarp_scale_adaptive。别名保留，勿用于新实验。"},
    ]
