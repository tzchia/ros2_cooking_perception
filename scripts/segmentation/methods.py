"""Mask generation methods for RGB + thermal data with Thermal Verification."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import warnings

import numpy as np
from PIL import Image
import cv2


# --- Helper Functions: Morphology & Geometry ---

def normalize_thermal(thermal_img: np.ndarray) -> np.ndarray:
    tmin = float(thermal_img.min())
    tmax = float(thermal_img.max())
    if tmax == tmin:
        return np.zeros_like(thermal_img, dtype=np.float32)
    return (thermal_img - tmin) / (tmax - tmin + 1e-6)


def thermal_mask(thermal_img: np.ndarray, low: float = 0.6) -> np.ndarray:
    norm = normalize_thermal(thermal_img)
    return (norm >= low).astype(np.uint8)


def smooth_mask_contours(mask: np.ndarray, epsilon_factor: float = 0.002) -> np.ndarray:
    """
    Smoothes the jagged edges of a binary mask using polygon approximation.
    Higher epsilon_factor = smoother but less detailed.
    """
    if mask.sum() == 0:
        return mask
        
    mask = mask.astype(np.uint8)
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    smoothed_mask = np.zeros_like(mask)
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 50: # Skip tiny noise
            continue
            
        # Epsilon is the accuracy parameter. 
        # 0.002 * arcLength is a good balance for round objects like woks.
        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Draw the smoothed polygon
        cv2.drawContours(smoothed_mask, [approx], -1, 1, thickness=cv2.FILLED)
        
    return smoothed_mask


def refine_mask_morphology(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    
    mask = mask.astype(np.uint8)

    # 1. Keep Largest Component (removes fragmented noise outliers)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        # stats[1:, 4] are areas (skipping background 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8)
    
    # 2. Fill Holes (Closing)
    # Using a slightly larger kernel to bridge gaps caused by reflections
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 3. Fill Internal Contours (Hole Filling)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)
    
    # 4. Smooth Edges (Deburring)
    mask = smooth_mask_contours(mask, epsilon_factor=0.002)
    
    return mask


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


# --- Thermal Processing ---

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
    
    # If the cluster is too small (noise), fallback to simple threshold
    if mask.mean() < min_ratio:
        return thermal_mask(thermal_img, low=low_fallback)
        
    return refine_mask_morphology(mask.astype(np.uint8))


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
    points = coords[top_idx][:, ::-1] # XY
    labels = np.ones(len(points), dtype=np.int32)
    box = np.array([x1, y1, x2, y2]) # XYXY
    
    return points, labels, box


# --- Selection Logic ---

def select_best_mask_by_thermal(
    masks: np.ndarray, scores: np.ndarray, thermal_ref_mask: np.ndarray
) -> np.ndarray:
    """
    Selects best mask by balancing IoU with Thermal and Area Consistency.
    """
    if masks.shape[0] == 0:
        return np.zeros_like(thermal_ref_mask)
    if masks.shape[0] == 1:
        return refine_mask_morphology(masks[0].astype(np.uint8))

    best_score = -float('inf')
    best_idx = 0
    
    thermal_area = thermal_ref_mask.sum()
    
    for i in range(masks.shape[0]):
        current_mask = masks[i].astype(np.uint8)
        mask_area = current_mask.sum()
        
        iou = calculate_iou(current_mask, thermal_ref_mask)
        
        # Area Penalty: 
        # 1. Too big (> 3.5x thermal): Likely capturing the whole stove (Red Sea).
        # 2. Too small (< 0.1x thermal): Likely capturing just a reflection spot.
        area_ratio = mask_area / (thermal_area + 1e-6)
        penalty = 0.0
        
        if area_ratio > 3.5: 
            penalty = 1.0 
        elif area_ratio < 0.1:
            penalty = 0.5
            
        score = iou - penalty + (0.05 * scores[i])
        
        if score > best_score:
            best_score = score
            best_idx = i
            
    final_mask = masks[best_idx].astype(np.uint8)
    return refine_mask_morphology(final_mask)


# --- Global Model Caches ---
_SAM_PREDICTOR = None
_HQ_SAM_PREDICTOR = None
_SAM2_PREDICTOR = None
_DINO_PROCESSOR = None
_DINO_MODEL = None
_DINO_MODEL_KEY = None
_FLORENCE2_MODEL = None
_FLORENCE2_PROCESSOR = None
_FLORENCE2_MODEL_KEY = None
_YOLO_WORLD_MODEL = None
_SAM_EFFICIENT_MODEL = None
_YOLO_WORLD_KEY = None


# --- Model Loaders ---

def _normalize_device(device: str | int | None) -> str:
    if device is None: return "cpu"
    d = str(device)
    if d in {"-1", "cpu"}: return "cpu"
    if d.startswith("cuda") or d.startswith("mps"): return d
    if d.isdigit(): return f"cuda:{d}"
    return d

def get_sam_predictor(root, model_type, checkpoint):
    global _SAM_PREDICTOR
    if _SAM_PREDICTOR: return _SAM_PREDICTOR
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    if not checkpoint.exists(): raise FileNotFoundError(f"Missing: {checkpoint}")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint)).to("cuda" if torch.cuda.is_available() else "cpu")
    _SAM_PREDICTOR = SamPredictor(sam)
    return _SAM_PREDICTOR

def get_hq_sam_predictor(root, model_type, checkpoint):
    global _HQ_SAM_PREDICTOR
    if _HQ_SAM_PREDICTOR: return _HQ_SAM_PREDICTOR
    import torch
    from segment_anything_hq import sam_model_registry, SamPredictor
    if not checkpoint.exists(): raise FileNotFoundError(f"Missing: {checkpoint}")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint)).to("cuda" if torch.cuda.is_available() else "cpu")
    _HQ_SAM_PREDICTOR = SamPredictor(sam)
    return _HQ_SAM_PREDICTOR

def get_sam2_predictor(config_file, checkpoint):
    global _SAM2_PREDICTOR
    if _SAM2_PREDICTOR: return _SAM2_PREDICTOR
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if not checkpoint.exists(): raise FileNotFoundError(f"Missing: {checkpoint}")
    sam2_model = build_sam2(config_file, str(checkpoint), device="cuda" if torch.cuda.is_available() else "cpu")
    _SAM2_PREDICTOR = SAM2ImagePredictor(sam2_model)
    return _SAM2_PREDICTOR

def get_hf_grounding_dino(checkpoint_str, device="cuda"):
    global _DINO_MODEL, _DINO_PROCESSOR, _DINO_MODEL_KEY
    if _DINO_MODEL and _DINO_MODEL_KEY == checkpoint_str:
        return _DINO_MODEL, _DINO_PROCESSOR
    
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    
    path_obj = Path(checkpoint_str)
    model_id = checkpoint_str
    
    if path_obj.suffix == ".pth":
        print(f"Warning: {checkpoint_str} is a raw weight file, switching to Hugging Face model 'IDEA-Research/grounding-dino-base'.")
        model_id = "IDEA-Research/grounding-dino-base"
        
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    _DINO_MODEL = model
    _DINO_PROCESSOR = processor
    _DINO_MODEL_KEY = checkpoint_str
    return model, processor

def get_florence2_model(model_id, device=None):
    global _FLORENCE2_MODEL, _FLORENCE2_PROCESSOR, _FLORENCE2_MODEL_KEY
    if not device: 
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if _FLORENCE2_MODEL and _FLORENCE2_MODEL_KEY == model_id:
        return _FLORENCE2_MODEL, _FLORENCE2_PROCESSOR, device
    
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
    
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    # Fix for Florence-2 config crash
    for cfg in [config, getattr(config, 'text_config', None), getattr(config, 'vision_config', None)]:
        if cfg and isinstance(cfg, dict): cfg.pop("forced_bos_token_id", None)
        elif cfg and hasattr(cfg, "pop"): cfg.pop("forced_bos_token_id", None)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, trust_remote_code=True
    ).to(device).eval()
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    _FLORENCE2_MODEL = model
    _FLORENCE2_PROCESSOR = processor
    _FLORENCE2_MODEL_KEY = model_id
    return model, processor, device

def get_yolo_world_sam(yolo_model, sam_model, device="cuda"):
    """
    Fixes the multi-device error by enforcing device placement BEFORE setting classes.
    """
    global _YOLO_WORLD_MODEL, _SAM_EFFICIENT_MODEL, _YOLO_WORLD_KEY
    normalized_device = _normalize_device(device)
    key = (yolo_model, sam_model, normalized_device)
    
    if _YOLO_WORLD_MODEL and _YOLO_WORLD_KEY == key:
        return _YOLO_WORLD_MODEL, _SAM_EFFICIENT_MODEL
        
    from ultralytics import YOLOWorld, SAM
    
    yolo = YOLOWorld(yolo_model)
    sam = SAM(sam_model)
    
    # --- Critical Fix for YOLO-World Device Mismatch ---
    # We must move the model to the device immediately.
    # Otherwise set_classes() might run CLIP text encoding on CPU while model is confused.
    if normalized_device != "cpu":
        yolo.to(normalized_device)
        # sam.to(normalized_device) # SAM handles device in predict usually, but can be explicit
    
    _YOLO_WORLD_MODEL = yolo
    _SAM_EFFICIENT_MODEL = sam
    _YOLO_WORLD_KEY = key
    return yolo, sam


# --- Optimized Methods ---

def sam_prompt_mask(rgb_img, thermal_img, root, low=0.6, topk=20, model_type="vit_b", checkpoint=None):
    points, labels, box = thermal_prompts(thermal_img, low=low, topk=topk)
    ref_mask = thermal_mask(thermal_img, low=low)
    
    if box is None: return np.zeros_like(ref_mask)
    
    predictor = get_sam_predictor(root, model_type, checkpoint)
    predictor.set_image(rgb_img)
    masks, scores, _ = predictor.predict(
        point_coords=points, point_labels=labels, box=box[None, :], multimask_output=True
    )
    return select_best_mask_by_thermal(masks, scores, ref_mask)


def hq_sam_prompt_mask(rgb_img, thermal_img, root, low=0.6, topk=20, model_type="vit_b", checkpoint=None):
    points, labels, box = thermal_prompts(thermal_img, low=low, topk=topk)
    ref_mask = thermal_mask(thermal_img, low=low)
    if box is None: return np.zeros_like(ref_mask)

    predictor = get_hq_sam_predictor(root, model_type, checkpoint)
    predictor.set_image(rgb_img)
    masks, scores, _ = predictor.predict(
        point_coords=points, point_labels=labels, box=box[None, :], multimask_output=True
    )
    return select_best_mask_by_thermal(masks, scores, ref_mask)


def sam2_prompt_mask(rgb_img, thermal_img, config_file, checkpoint, low=0.6, topk=20):
    points, labels, box = thermal_prompts(thermal_img, low=low, topk=topk)
    ref_mask = thermal_mask(thermal_img, low=low)
    if box is None: return np.zeros_like(ref_mask)

    points = np.ascontiguousarray(points)
    box = np.ascontiguousarray(box)
    
    predictor = get_sam2_predictor(config_file, checkpoint)
    predictor.set_image(rgb_img)
    masks, scores, _ = predictor.predict(
        point_coords=points, point_labels=labels, box=box[None, :], multimask_output=True
    )
    return select_best_mask_by_thermal(masks, scores, ref_mask)


def grounding_dino_optimized(
    rgb_img, thermal_img, root, model_type, checkpoint,
    dino_config, dino_checkpoint, text_prompt, box_threshold, text_threshold, low=0.6
):
    import torch
    
    _, _, thermal_box = thermal_prompts(thermal_img, low=low)
    ref_mask = thermal_mask(thermal_img, low=low)
    if thermal_box is None: return np.zeros(rgb_img.shape[:2], dtype=np.uint8)
    
    t_x1, t_y1, t_x2, t_y2 = thermal_box
    thermal_area = (t_x2 - t_x1) * (t_y2 - t_y1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = get_hf_grounding_dino(str(dino_checkpoint), device=device)
    
    image_pil = Image.fromarray(rgb_img)
    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    try:
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=box_threshold, text_threshold=text_threshold, target_sizes=[image_pil.size[::-1]]
        )[0]
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_threshold, target_sizes=[image_pil.size[::-1]]
        )[0]
    
    boxes = results["boxes"].cpu().numpy()
    
    if len(boxes) == 0: return np.zeros(rgb_img.shape[:2], dtype=np.uint8)

    best_iou = 0.0
    best_box = None
    boxes_xyxy = boxes * torch.tensor([image_pil.width, image_pil.height, image_pil.width, image_pil.height])
    
    for i in range(len(boxes_xyxy)):
        cx, cy, bw, bh = boxes_xyxy[i].numpy()
        b_x1, b_y1, b_x2, b_y2 = cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2
        
        ix1, iy1 = max(t_x1, b_x1), max(t_y1, b_y1)
        ix2, iy2 = min(t_x2, b_x2), min(t_y2, b_y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        
        if inter > 0:
            box_area = (b_x2 - b_x1) * (b_y2 - b_y1)
            union = thermal_area + box_area - inter
            iou = inter / union
            # Filter: Box overlap with thermal, and not > 3.5x larger
            if iou > best_iou and (box_area < 3.5 * thermal_area):
                best_iou = iou
                best_box = np.array([b_x1, b_y1, b_x2, b_y2])

    if best_box is None: return np.zeros(rgb_img.shape[:2], dtype=np.uint8)

    predictor = get_sam_predictor(root, model_type, checkpoint)
    predictor.set_image(rgb_img)
    masks, scores, _ = predictor.predict(box=best_box[None, :], multimask_output=True)
    return select_best_mask_by_thermal(masks, scores, ref_mask)


def florence2_optimized(
    rgb_img, thermal_img, text_prompt, model_id, device, low=0.6
):
    import cv2
    model, processor, device = get_florence2_model(model_id, device)
    image_pil = Image.fromarray(rgb_img)
    task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>"
    
    inputs = processor(text=task_prompt + text_prompt, images=image_pil, return_tensors="pt").to(device)
    import torch
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            max_new_tokens=1024, do_sample=False, num_beams=3
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    prediction = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image_pil.width, image_pil.height)
    )
    
    rgb_mask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)
    polygons = prediction.get(task_prompt, {}).get('polygons', [])
    for poly in polygons:
        poly = np.array(poly).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(rgb_mask, [poly], 1)
        
    ref_mask = thermal_mask(thermal_img, low=low)
    iou = calculate_iou(rgb_mask, ref_mask)
    
    if iou < 0.1 or (rgb_mask.sum() > 4.0 * ref_mask.sum()):
        return np.zeros_like(rgb_mask)
            
    return refine_mask_morphology(rgb_mask)


def yolo_world_optimized(
    rgb_img, thermal_img, classes, yolo_model, sam_model, conf, device, low=0.6
):
    normalized_device = _normalize_device(device)
    yolo, sam = get_yolo_world_sam(yolo_model, sam_model, normalized_device)
    
    # Set classes with device safety
    try:
        yolo.set_classes(classes)
    except Exception as e:
        print(f"Warning: YOLO set_classes failed: {e}. Trying to run without class filter.")
        
    yolo_results = yolo.predict(rgb_img, conf=conf, device=normalized_device, verbose=False)
    
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0: return np.zeros(rgb_img.shape[:2], dtype=np.uint8)
    
    _, _, thermal_box = thermal_prompts(thermal_img, low=low)
    if thermal_box is None: return np.zeros(rgb_img.shape[:2], dtype=np.uint8)
    t_x1, t_y1, t_x2, t_y2 = thermal_box
    thermal_area = (t_x2 - t_x1) * (t_y2 - t_y1)
    
    valid_boxes = []
    for box in boxes:
        b_x1, b_y1, b_x2, b_y2 = box
        
        ix1, iy1 = max(t_x1, b_x1), max(t_y1, b_y1)
        ix2, iy2 = min(t_x2, b_x2), min(t_y2, b_y2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        
        if inter > 0:
            box_area = (b_x2 - b_x1) * (b_y2 - b_y1)
            union = thermal_area + box_area - inter
            iou = inter / union
            
            # Relaxed filter: Allow larger boxes (up to 4.0x) but require overlap
            if iou > 0.05 and box_area < 4.0 * thermal_area:
                valid_boxes.append(box)
            
    if not valid_boxes: return np.zeros(rgb_img.shape[:2], dtype=np.uint8)
    
    sam_results = sam(rgb_img, bboxes=valid_boxes, verbose=False)
    final_mask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)
    if sam_results[0].masks is not None:
        masks = sam_results[0].masks.data.cpu().numpy()
        for m in masks:
            final_mask = np.maximum(final_mask, m.astype(np.uint8))
            
    return refine_mask_morphology(final_mask)


def generate_mask_by_method(rgb_img, thermal_img, method, args):
    mask = None
    
    if method == "thermal":
        mask = thermal_mask(thermal_img, low=args.thermal_low)
    
    elif method == "thermal_cluster":
        mask = thermal_cluster_mask(
            thermal_img, k=args.cluster_k, iters=args.cluster_iters,
            min_ratio=args.cluster_min_ratio, low_fallback=args.thermal_low
        )

    elif method in ["sam", "sam_v2"]:
        sam_low = args.sam_low if args.sam_low is not None else args.thermal_low
        mask = sam_prompt_mask(
            rgb_img, thermal_img, root=args.root, low=sam_low, topk=args.sam_topk,
            model_type=args.sam_model_type, checkpoint=args.sam_checkpoint
        )
        
    elif method == "hq_sam":
        sam_low = args.sam_low if args.sam_low is not None else args.thermal_low
        hq_ckpt = args.hq_sam_checkpoint or (args.root / "weights" / "sam_hq_vit_b.pth")
        mask = hq_sam_prompt_mask(
            rgb_img, thermal_img, root=args.root, low=sam_low, topk=args.sam_topk,
            model_type=args.sam_model_type, checkpoint=hq_ckpt
        )

    elif method == "sam2":
        sam_low = args.sam_low if args.sam_low is not None else args.thermal_low
        sam2_ckpt = args.sam2_checkpoint or (args.root / "weights" / "sam2_hiera_large.pt")
        mask = sam2_prompt_mask(
            rgb_img, thermal_img, config_file=args.sam2_config, checkpoint=sam2_ckpt,
            low=sam_low, topk=args.sam_topk
        )

    elif method == "groundingdino":
        mask = grounding_dino_optimized(
            rgb_img, thermal_img, root=args.root, model_type=args.sam_model_type,
            checkpoint=args.sam_checkpoint, dino_config=args.dino_config,
            dino_checkpoint=args.dino_checkpoint, text_prompt=args.dino_text_prompt,
            box_threshold=args.dino_box_threshold, text_threshold=args.dino_text_threshold,
            low=args.thermal_low
        )

    elif method == "florence2":
        mask = florence2_optimized(
            rgb_img, thermal_img, text_prompt=args.florence2_text_prompt,
            model_id=args.florence2_model_id, device=getattr(args, 'device', None),
            low=args.thermal_low
        )

    elif method == "yolo_world_sam":
        classes = args.yolo_world_classes.split(",") if args.yolo_world_classes else ["black wok", "cooking pot"]
        mask = yolo_world_optimized(
            rgb_img, thermal_img, classes=classes, yolo_model=args.yolo_world_model,
            sam_model=args.yolo_world_sam_model, conf=args.yolo_world_conf,
            device=getattr(args, 'device', 'cuda'), low=args.thermal_low
        )
    
    else:
        raise ValueError(f"Unknown method: {method}")

    # --- SMART FALLBACK: The Lid Solution ---
    # If the advanced model fails (empty mask), but Thermal shows significant heat,
    # fallback to Thermal Cluster. This saves the day for "Silver Lid" or "Reflective Pan" scenarios.
    if mask is not None:
        if mask.sum() == 0:
            ref = thermal_mask(thermal_img, low=args.thermal_low)
            # Check if thermal has a decent sized object (e.g., > 100 pixels)
            if ref.sum() > 200: 
                print(f"[{method}] Failed to detect object, but heat detected. Fallback to thermal cluster.")
                mask = thermal_cluster_mask(
                    thermal_img, k=args.cluster_k, iters=args.cluster_iters,
                    min_ratio=args.cluster_min_ratio, low_fallback=args.thermal_low
                )
    
    return mask