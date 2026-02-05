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


def keep_significant_components(mask: np.ndarray, area_ratio_threshold: float = 0.15) -> np.ndarray:
    """
    保留 Mask 中顯著的連通區塊。
    邏輯：找出最大的區塊面積 max_area，
    只保留面積 > (max_area * area_ratio_threshold) 的區塊。
    這允許畫面中同時存在多個鍋子，但會過濾掉細碎雜訊。
    """
    import cv2

    mask = mask.astype(np.uint8)

    # 尋找輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask

    # 1. 找出最大面積
    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas)

    # 2. 篩選：只留面積夠大的
    keep_contours = []
    for cnt, area in zip(contours, areas):
        if area >= max_area * area_ratio_threshold:
            keep_contours.append(cnt)

    # 3. 繪製新的乾淨 Mask
    new_mask = np.zeros_like(mask)
    cv2.drawContours(new_mask, keep_contours, -1, 1, thickness=cv2.FILLED)

    return new_mask


def select_largest_mask(masks: list[np.ndarray]) -> np.ndarray | None:
    if not masks:
        return None
    return max(masks, key=lambda m: float(m.sum()))


_SAM_PREDICTOR = None
_SAM_AUTO_GENERATORS: dict[tuple[str, str, int | None, float | None, float | None], object] = {}
_DINO_MODEL = None
_DINO_MODEL_KEY: tuple[str, str] | None = None

_FLORENCE2_MODEL = None
_FLORENCE2_PROCESSOR = None
_FLORENCE2_MODEL_KEY: str | None = None

_YOLO_WORLD_MODEL = None
_SAM_EFFICIENT_MODEL = None
_YOLO_WORLD_KEY: tuple[str, str] | None = None


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
        raise ValueError("Grounding DINO config and checkpoint are required for groundingdino")
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
            "GroundingDINO is required for groundingdino. Install from https://github.com/IDEA-Research/GroundingDINO"
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
            "GroundingDINO is required for groundingdino. Install from https://github.com/IDEA-Research/GroundingDINO"
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
        logits_tensor = logits
        if logits_tensor.ndim == 1:
            scores = logits_tensor.detach().cpu().numpy()
        else:
            scores = logits_tensor.max(dim=1).values.detach().cpu().numpy()
    else:
        logits_np = np.asarray(logits)
        if logits_np.ndim == 1:
            scores = logits_np
        else:
            scores = np.max(logits_np, axis=1)
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
    raw_mask = masks[best_idx].astype(np.uint8)
    # Apply significant component filtering with relative area threshold
    return keep_significant_components(raw_mask, area_ratio_threshold=0.15)


def get_florence2_model(model_id: str = "microsoft/Florence-2-large", device: str | None = None):
    """Lazy load Florence-2 model with robust patching."""
    global _FLORENCE2_MODEL, _FLORENCE2_PROCESSOR, _FLORENCE2_MODEL_KEY
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if _FLORENCE2_MODEL is not None and _FLORENCE2_MODEL_KEY == model_id:
        return _FLORENCE2_MODEL, _FLORENCE2_PROCESSOR, device

    try:
        from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
    except ImportError as exc:
        raise ImportError(
            "transformers is required for florence2. Install: pip install transformers torch"
        ) from exc

    # Patch Florence-2 remote code cache before loading config
    try:
        # 嘗試找到 Hugging Face cache 路徑
        module_root = Path.home() / ".cache/huggingface/modules/transformers_modules"
        # 搜尋所有可能的 Florence-2 配置檔 (包含使用者路徑中的 microsoft 目錄)
        candidates = list(module_root.glob("**/microsoft/Florence-2-large/*/configuration_florence2.py")) + \
                     list(module_root.glob("**/Florence-2-large/*/configuration_florence2.py"))
        
        # 如果找不到，嘗試觸發一次下載 (預期會失敗，但會下載檔案)
        if not candidates:
            try:
                AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            except Exception:
                pass # 忽略錯誤，我們只是要確保檔案被下載
            # 重新搜尋
            candidates = list(module_root.glob("**/microsoft/Florence-2-large/*/configuration_florence2.py"))

        for config_path in candidates:
            content = config_path.read_text()
            modified = False

            # Fix 1: Add forced_bos_token_id to __init__ signature (既有的修正)
            if "forced_bos_token_id=" not in content:
                content = content.replace(
                    "decoder_start_token_id=2,\n        forced_eos_token_id=2,",
                    "decoder_start_token_id=2,\n        forced_bos_token_id=None,\n        forced_eos_token_id=2,"
                )
                modified = True

            # Fix 2: Add forced_bos_token_id to super().__init__ (既有的修正)
            if "forced_bos_token_id=forced_bos_token_id," not in content:
                content = content.replace(
                    "decoder_start_token_id=decoder_start_token_id,\n            forced_eos_token_id=forced_eos_token_id,",
                    "decoder_start_token_id=decoder_start_token_id,\n            forced_bos_token_id=forced_bos_token_id,\n            forced_eos_token_id=forced_eos_token_id,"
                )
                modified = True

            # Fix 3 (關鍵修正): 在初始化 Florence2LanguageConfig 之前移除字典中的 forced_bos_token_id
            # 這是造成您截圖中 TypeError 的主因
            target_line = "self.text_config = Florence2LanguageConfig(**text_config)"
            if target_line in content:
                # 我們插入一行 pop 指令來移除該參數
                content = content.replace(
                    target_line,
                    "text_config.pop('forced_bos_token_id', None)\n        self.text_config = Florence2LanguageConfig(**text_config)"
                )
                modified = True

            if modified:
                config_path.write_text(content)
                print(f"Patched Florence-2 config at: {config_path}")

    except Exception as e:
        print(f"Warning: Failed to patch Florence-2 config: {e}")

    # Load config and model
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    # 額外保險：如果 Patch 失敗，手動在記憶體中修正 Config
    if hasattr(config, "text_config") and isinstance(config.text_config, dict):
         config.text_config.pop("forced_bos_token_id", None)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, trust_remote_code=True
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    _FLORENCE2_MODEL = model
    _FLORENCE2_PROCESSOR = processor
    _FLORENCE2_MODEL_KEY = model_id
    return model, processor, device


def florence2_mask(
    rgb_img: np.ndarray,
    text_prompt: str = "black wok",
    model_id: str = "microsoft/Florence-2-large",
    device: str | None = None,
) -> np.ndarray:
    """
    Florence-2 Referring Expression Segmentation.
    """
    import cv2
    from PIL import Image

    model, processor, device = get_florence2_model(model_id, device)

    image_pil = Image.fromarray(rgb_img)
    task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>"
    prompt = task_prompt + text_prompt

    inputs = processor(text=prompt, images=image_pil, return_tensors="pt").to(device)

    import torch
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    prediction = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image_pil.width, image_pil.height)
    )

    mask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)
    segmentation_result = prediction.get(task_prompt, {})
    polygons = segmentation_result.get('polygons', [])
    for poly in polygons:
        poly = np.array(poly).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [poly], 1)

    return mask


def get_yolo_world_sam(yolo_model: str = "yolov8s-world.pt", sam_model: str = "mobile_sam.pt", device: str = "cuda"):
    """Lazy load YOLO-World and SAM models."""
    global _YOLO_WORLD_MODEL, _SAM_EFFICIENT_MODEL, _YOLO_WORLD_KEY
    key = (yolo_model, sam_model, device)
    if _YOLO_WORLD_MODEL is not None and _YOLO_WORLD_KEY == key:
        return _YOLO_WORLD_MODEL, _SAM_EFFICIENT_MODEL

    try:
        from ultralytics import YOLOWorld, SAM
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for yolo_world_sam. Install: pip install ultralytics"
        ) from exc

    yolo = YOLOWorld(yolo_model)
    sam = SAM(sam_model)

    _YOLO_WORLD_MODEL = yolo
    _SAM_EFFICIENT_MODEL = sam
    _YOLO_WORLD_KEY = key
    return yolo, sam


def yolo_world_sam_mask(
    rgb_img: np.ndarray,
    classes: list[str] | None = None,
    yolo_model: str = "yolov8s-world.pt",
    sam_model: str = "mobile_sam.pt",
    conf: float = 0.15,
    device: str = "cuda",
) -> np.ndarray:
    """
    YOLO-World for open-vocabulary detection + SAM for segmentation.
    """
    yolo, sam = get_yolo_world_sam(yolo_model, sam_model, device)

    if classes:
        classes = [c.strip() for c in classes if c.strip()]
    if classes:
        try:
            vocab_size = None
            model_obj = getattr(yolo, "model", None)
            candidates = [model_obj, getattr(model_obj, "clip_model", None)]
            for candidate in candidates:
                if candidate is None:
                    continue
                token_embedding = getattr(candidate, "token_embedding", None)
                if token_embedding is not None and hasattr(token_embedding, "weight"):
                    vocab_size = int(token_embedding.weight.shape[0])
                    break

            # Detect model device from a parameter tensor (needed for both patches)
            model_device = None
            if model_obj is not None:
                try:
                    for p in model_obj.parameters():
                        model_device = p.device
                        break
                except Exception:
                    pass

            if model_obj is not None and hasattr(model_obj, "text_tokenizer"):
                base_tokenizer = model_obj.text_tokenizer

                def _safe_tokenize(texts, context_length=77, truncate=False):
                    tokens = base_tokenizer(texts, context_length=context_length, truncate=truncate)
                    if vocab_size is not None:
                        tokens = tokens.clamp_(0, vocab_size - 1)
                    # Move to same device as model to avoid embedding index_select device mismatch
                    if model_device is not None and hasattr(tokens, "to"):
                        tokens = tokens.to(model_device)
                    return tokens

                model_obj.text_tokenizer = _safe_tokenize

            # Additional patch: wrap encode_text to ensure input is on correct device
            clip_text_model = getattr(model_obj, "clip_model", None)
            if clip_text_model is not None and hasattr(clip_text_model, "encode_text"):
                # Get the actual unbound method to avoid recursion
                orig_encode_text = clip_text_model.encode_text
                _md = model_device  # local binding for closure

                def _safe_encode_text(texts):
                    # texts are token indices tensor, ensure on correct device
                    if hasattr(texts, "to") and _md is not None:
                        texts = texts.to(_md)
                    # Call the original method directly, not through attribute lookup
                    return type(clip_text_model).encode_text(clip_text_model, texts)

                clip_text_model.encode_text = _safe_encode_text
        except Exception:
            pass

        yolo.set_classes(classes)

    yolo_results = yolo.predict(rgb_img, conf=conf, device=device, verbose=False)
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        return np.zeros(rgb_img.shape[:2], dtype=np.uint8)

    sam_results = sam(rgb_img, bboxes=boxes, verbose=False)
    final_mask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)

    if sam_results[0].masks is not None:
        masks = sam_results[0].masks.data.cpu().numpy()
        for m in masks:
            final_mask = np.maximum(final_mask, m.astype(np.uint8))

    return final_mask


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
    if method == "groundingdino":
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
    if method == "florence2":
        return florence2_mask(
            rgb_img,
            text_prompt=args.florence2_text_prompt,
            model_id=args.florence2_model_id,
            device=getattr(args, 'device', None),
        )
    if method == "yolo_world_sam":
        classes = args.yolo_world_classes.split(",") if args.yolo_world_classes else ["black wok", "cooking pot"]
        return yolo_world_sam_mask(
            rgb_img,
            classes=classes,
            yolo_model=args.yolo_world_model,
            sam_model=args.yolo_world_sam_model,
            conf=args.yolo_world_conf,
            device=getattr(args, 'device', 'cuda'),
        )
    raise ValueError(f"Unknown method: {method}")
