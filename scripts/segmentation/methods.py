"""Mask generation methods for RGB + thermal data."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def normalize_thermal(thermal_img: np.ndarray) -> np.ndarray:
    tmin = float(thermal_img.min())
    tmax = float(thermal_img.max())
    return (thermal_img - tmin) / (tmax - tmin + 1e-6)


def thermal_mask(thermal_img: np.ndarray, low: float = 0.6) -> np.ndarray:
    norm = normalize_thermal(thermal_img)
    return (norm >= low).astype(np.uint8)


def kmeans_1d(values: np.ndarray, k: int = 3, iters: int = 10) -> tuple[np.ndarray, np.ndarray]:
    centers = np.linspace(values.min(), values.max(), k)
    labels = np.zeros(len(values), dtype=np.int32)
    for _ in range(iters):
        dists = np.abs(values[:, None] - centers[None, :])
        labels = dists.argmin(axis=1)
        for idx in range(k):
            mask = labels == idx
            if np.any(mask):
                centers[idx] = values[mask].mean()
    return centers, labels


def thermal_cluster_mask(
    thermal_img: np.ndarray,
    k: int = 3,
    iters: int = 10,
    min_ratio: float = 0.002,
    low_fallback: float = 0.6,
) -> np.ndarray:
    norm = normalize_thermal(thermal_img)
    values = norm.flatten()
    centers, labels = kmeans_1d(values, k=k, iters=iters)
    hottest = np.argsort(centers)[-1]
    mask = labels.reshape(norm.shape) == hottest
    if mask.mean() < min_ratio:
        return thermal_mask(thermal_img, low=low_fallback)
    return mask.astype(np.uint8)


def mask_centroid(mask_img: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask_img > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def filter_masks_center(
    masks: list[np.ndarray], image_shape: tuple[int, int], center_frac: float
) -> list[np.ndarray]:
    if not masks:
        return []
    h, w = image_shape
    cx, cy = w / 2.0, h / 2.0
    half_w = w * center_frac / 2.0
    half_h = h * center_frac / 2.0
    x1, x2 = cx - half_w, cx + half_w
    y1, y2 = cy - half_h, cy + half_h
    filtered: list[np.ndarray] = []
    for mask in masks:
        centroid = mask_centroid(mask)
        if centroid is None:
            continue
        x, y = centroid
        if x1 <= x <= x2 and y1 <= y <= y2:
            filtered.append(mask)
    return filtered


def select_largest_mask(masks: list[np.ndarray]) -> np.ndarray | None:
    if not masks:
        return None
    return max(masks, key=lambda m: float(m.sum()))


_SAM_PREDICTOR = None
_SAM_AUTO_GENERATORS: dict[tuple[str, str, int | None, float | None, float | None], object] = {}
_DINO_MODEL = None
_DINO_MODEL_KEY: tuple[str, str] | None = None


def get_sam_predictor(root: Path, model_type: str, checkpoint: Path):
    global _SAM_PREDICTOR
    if _SAM_PREDICTOR is not None:
        return _SAM_PREDICTOR

    try:
        import torch
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError as exc:
        raise ImportError(
            "segment-anything is required. Install from https://github.com/facebookresearch/segment-anything"
        ) from exc

    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing SAM checkpoint: {checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device)
    _SAM_PREDICTOR = SamPredictor(sam)
    return _SAM_PREDICTOR


def get_sam_auto_generator(
    root: Path,
    model_type: str,
    checkpoint: Path,
    min_area: int | None = None,
    pred_iou: float | None = None,
    stability: float | None = None,
):
    key = (model_type, str(checkpoint), min_area, pred_iou, stability)
    if key in _SAM_AUTO_GENERATORS:
        return _SAM_AUTO_GENERATORS[key]

    try:
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            "segment-anything is required. Install from https://github.com/facebookresearch/segment-anything"
        ) from exc

    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing SAM checkpoint: {checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device)
    kwargs: dict[str, object] = {}
    if min_area is not None:
        kwargs["min_mask_region_area"] = min_area
    if pred_iou is not None:
        kwargs["pred_iou_thresh"] = pred_iou
    if stability is not None:
        kwargs["stability_score_thresh"] = stability
    generator = SamAutomaticMaskGenerator(sam, **kwargs)
    _SAM_AUTO_GENERATORS[key] = generator
    return generator


def get_dino_model(config_path: Path | None, checkpoint_path: Path | None):
    global _DINO_MODEL, _DINO_MODEL_KEY
    if config_path is None or checkpoint_path is None:
        raise ValueError("Grounding DINO config and checkpoint are required for sam_v4")
    key = (str(config_path), str(checkpoint_path))
    if _DINO_MODEL is not None and _DINO_MODEL_KEY == key:
        return _DINO_MODEL

    if not config_path.exists():
        raise FileNotFoundError(f"Missing Grounding DINO config: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing Grounding DINO checkpoint: {checkpoint_path}")

    try:
        from groundingdino.util.inference import load_model
    except ImportError as exc:
        raise ImportError(
            "GroundingDINO is required for sam_v4. Install from https://github.com/IDEA-Research/GroundingDINO"
        ) from exc

    _DINO_MODEL = load_model(str(config_path), str(checkpoint_path))
    _DINO_MODEL_KEY = key
    return _DINO_MODEL


def thermal_prompts(
    thermal_img: np.ndarray, low: float = 0.6, topk: int = 20
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    norm = normalize_thermal(thermal_img)
    mask = norm >= low
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, None, None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    coords = np.column_stack([ys, xs])
    vals = norm[mask]
    top_idx = np.argsort(vals)[-min(topk, len(vals)) :]
    points = coords[top_idx][:, ::-1]
    labels = np.ones(len(points), dtype=np.int32)
    box = np.array([x1, y1, x2, y2])
    return points, labels, box


def sam_prompt_mask(
    rgb_img: np.ndarray,
    thermal_img: np.ndarray,
    root: Path,
    low: float = 0.6,
    topk: int = 20,
    model_type: str = "vit_b",
    checkpoint: Path | None = None,
) -> np.ndarray:
    points, _, box = thermal_prompts(thermal_img, low=low, topk=topk)
    if points is None or box is None:
        return np.zeros(thermal_img.shape, dtype=np.uint8)
    centroid = points.mean(axis=0, keepdims=True)
    predictor = get_sam_predictor(root, model_type=model_type, checkpoint=checkpoint)
    predictor.set_image(rgb_img)
    masks, scores, _ = predictor.predict(
        point_coords=centroid,
        point_labels=np.array([1], dtype=np.int32),
        box=box[None, :],
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores)) if len(scores) else 0
    return masks[best_idx].astype(np.uint8)


def sam_auto_masks(
    rgb_img: np.ndarray,
    root: Path,
    model_type: str = "vit_b",
    checkpoint: Path | None = None,
    min_area: int | None = None,
    pred_iou: float | None = None,
    stability: float | None = None,
) -> list[np.ndarray]:
    generator = get_sam_auto_generator(
        root,
        model_type=model_type,
        checkpoint=checkpoint,
        min_area=min_area,
        pred_iou=pred_iou,
        stability=stability,
    )
    masks = generator.generate(rgb_img)
    return [mask["segmentation"].astype(np.uint8) for mask in masks]


def sam_auto_union_mask(
    rgb_img: np.ndarray,
    root: Path,
    model_type: str = "vit_b",
    checkpoint: Path | None = None,
    min_area: int | None = None,
    pred_iou: float | None = None,
    stability: float | None = None,
) -> np.ndarray:
    masks = sam_auto_masks(
        rgb_img,
        root=root,
        model_type=model_type,
        checkpoint=checkpoint,
        min_area=min_area,
        pred_iou=pred_iou,
        stability=stability,
    )
    if not masks:
        return np.zeros(rgb_img.shape[:2], dtype=np.uint8)
    combined = np.zeros(rgb_img.shape[:2], dtype=bool)
    for mask in masks:
        combined |= mask.astype(bool)
    return combined.astype(np.uint8)


def sam_auto_filtered_mask(
    rgb_img: np.ndarray,
    root: Path,
    model_type: str = "vit_b",
    checkpoint: Path | None = None,
    min_area: int | None = None,
    pred_iou: float | None = None,
    stability: float | None = None,
    center_frac: float = 0.5,
) -> np.ndarray:
    masks = sam_auto_masks(
        rgb_img,
        root=root,
        model_type=model_type,
        checkpoint=checkpoint,
        min_area=min_area,
        pred_iou=pred_iou,
        stability=stability,
    )
    if not masks:
        return np.zeros(rgb_img.shape[:2], dtype=np.uint8)
    filtered = filter_masks_center(masks, rgb_img.shape[:2], center_frac=center_frac)
    filtered = filtered or masks
    selected = select_largest_mask(filtered)
    if selected is None:
        return np.zeros(rgb_img.shape[:2], dtype=np.uint8)
    return selected.astype(np.uint8)


def grounding_dino_box(
    rgb_img: np.ndarray,
    config_path: Path | None,
    checkpoint_path: Path | None,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> np.ndarray | None:
    model = get_dino_model(config_path, checkpoint_path)

    try:
        from groundingdino.util.inference import load_image, predict
    except ImportError as exc:
        raise ImportError(
            "GroundingDINO is required for sam_v4. Install from https://github.com/IDEA-Research/GroundingDINO"
        ) from exc

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        Image.fromarray(rgb_img).save(tmp_path)
        image_source, image = load_image(tmp_path)
        boxes, logits, _ = predict(
            model=model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)

    if boxes is None or len(boxes) == 0:
        return None

    boxes_np = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
    if boxes_np.ndim == 1:
        boxes_np = boxes_np[None, :]

    if hasattr(logits, "detach"):
        scores = logits.max(dim=1).values.detach().cpu().numpy()
    else:
        scores = np.max(logits, axis=1)
    best_idx = int(np.argmax(scores))
    box = boxes_np[best_idx]

    box = box.astype(np.float32)
    if box.max() <= 1.5:
        try:
            import torch
            from groundingdino.util import box_ops

            box_xyxy = box_ops.box_cxcywh_to_xyxy(torch.as_tensor(box[None, :]))[0].cpu().numpy()
        except Exception:
            cx, cy, bw, bh = box
            box_xyxy = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        h, w = image_source.shape[:2]
        box_xyxy[0::2] *= w
        box_xyxy[1::2] *= h
    else:
        box_xyxy = box

    h, w = rgb_img.shape[:2]
    box_xyxy[0::2] = np.clip(box_xyxy[0::2], 0, w - 1)
    box_xyxy[1::2] = np.clip(box_xyxy[1::2], 0, h - 1)
    return box_xyxy


def sam_grounded_mask(
    rgb_img: np.ndarray,
    root: Path,
    model_type: str,
    checkpoint: Path,
    dino_config: Path | None,
    dino_checkpoint: Path | None,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> np.ndarray:
    box = grounding_dino_box(
        rgb_img,
        config_path=dino_config,
        checkpoint_path=dino_checkpoint,
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    if box is None:
        return np.zeros(rgb_img.shape[:2], dtype=np.uint8)
    predictor = get_sam_predictor(root, model_type=model_type, checkpoint=checkpoint)
    predictor.set_image(rgb_img)
    masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
    best_idx = int(np.argmax(scores)) if len(scores) else 0
    return masks[best_idx].astype(np.uint8)


def generate_mask_by_method(
    rgb_img: np.ndarray,
    thermal_img: np.ndarray,
    method: str,
    args: argparse.Namespace,
) -> np.ndarray:
    if method == "thermal":
        return thermal_mask(thermal_img, low=args.thermal_low)
    if method == "thermal_cluster":
        return thermal_cluster_mask(
            thermal_img,
            k=args.cluster_k,
            iters=args.cluster_iters,
            min_ratio=args.cluster_min_ratio,
            low_fallback=args.thermal_low,
        )
    if method == "sam":
        return sam_auto_union_mask(
            rgb_img,
            root=args.root,
            model_type=args.sam_model_type,
            checkpoint=args.sam_checkpoint,
            min_area=args.sam_auto_min_area,
            pred_iou=args.sam_auto_pred_iou,
            stability=args.sam_auto_stability,
        )
    if method == "sam_v2":
        sam_low = args.sam_low if args.sam_low is not None else args.thermal_low
        return sam_prompt_mask(
            rgb_img,
            thermal_img,
            root=args.root,
            low=sam_low,
            topk=args.sam_topk,
            model_type=args.sam_model_type,
            checkpoint=args.sam_checkpoint,
        )
    if method == "sam_v3":
        return sam_auto_filtered_mask(
            rgb_img,
            root=args.root,
            model_type=args.sam_model_type,
            checkpoint=args.sam_checkpoint,
            min_area=args.sam_v3_min_area,
            pred_iou=args.sam_v3_pred_iou,
            stability=args.sam_v3_stability,
            center_frac=args.sam_v3_center_frac,
        )
    if method == "sam_v4":
        return sam_grounded_mask(
            rgb_img,
            root=args.root,
            model_type=args.sam_model_type,
            checkpoint=args.sam_checkpoint,
            dino_config=args.dino_config,
            dino_checkpoint=args.dino_checkpoint,
            text_prompt=args.dino_text_prompt,
            box_threshold=args.dino_box_threshold,
            text_threshold=args.dino_text_threshold,
        )
    raise ValueError(f"Unknown method: {method}")
