"""Mask generation methods for RGB + thermal data with Multi-Class Support & Optimized Caching."""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import torch
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


def refine_mask_morphology(mask: np.ndarray) -> np.ndarray:
    """
    Applies morphology to a single binary layer.
    """
    if mask.sum() == 0:
        return mask
    
    mask = mask.astype(np.uint8)

    # 1. Filter tiny components (Noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    new_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Allow smaller objects (like egg bits), threshold lowered to 20
        if area > 20:
            new_mask[labels == i] = 1
            
    mask = new_mask
    
    # 2. Fill Holes (Closing)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # Smaller kernel for finer details
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 3. Fill Internal Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)
    
    return mask


def filter_huge_masks(mask: np.ndarray, max_ratio: float = 0.35) -> bool:
    """
    Returns True if mask is too big (likely the workbench/table).
    Stricter threshold: 35% of image.
    """
    total_pixels = mask.shape[0] * mask.shape[1]
    mask_pixels = np.sum(mask > 0)
    return (mask_pixels / total_pixels) > max_ratio


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
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
    # Ensure memory is contiguous for Tensor conversion
    points = coords[top_idx][:, ::-1].copy() 
    labels = np.ones(len(points), dtype=np.int32)
    box = np.array([x1, y1, x2, y2])
    return points, labels, box


# --- Selection Logic ---

def select_best_mask_by_thermal(
    masks: np.ndarray, scores: np.ndarray, thermal_ref_mask: np.ndarray
) -> np.ndarray:
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
        
        area_ratio = mask_area / (thermal_area + 1e-6)
        penalty = 0.0
        # Increased penalty for massive masks (backgrounds)
        if area_ratio > 3.0: penalty = 2.0 
        elif area_ratio < 0.1: penalty = 0.5
            
        score = iou - penalty + (0.05 * scores[i])
        if score > best_score:
            best_score = score
            best_idx = i
            
    return refine_mask_morphology(masks[best_idx].astype(np.uint8))


def select_best_mask_by_confidence(
    masks: np.ndarray, scores: np.ndarray
) -> np.ndarray:
    if masks.shape[0] == 0:
        return np.zeros((2,2), dtype=np.uint8)
    if masks.shape[0] == 1:
        return refine_mask_morphology(masks[0].astype(np.uint8))
    
    best_idx = int(np.argmax(scores))
    return refine_mask_morphology(masks[best_idx].astype(np.uint8))


# --- Model Loaders ---
_SAM_PREDICTOR = None
_HQ_SAM_PREDICTOR = None
_SAM2_PREDICTOR = None
_YOLO_WORLD_MODEL = None
_YOLO_WORLD_CACHE_KEY = None 

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
    from segment_anything import sam_model_registry, SamPredictor
    if not checkpoint.exists(): raise FileNotFoundError(f"Missing: {checkpoint}")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint)).to("cuda" if torch.cuda.is_available() else "cpu")
    _SAM_PREDICTOR = SamPredictor(sam)
    return _SAM_PREDICTOR

def get_hq_sam_predictor(root, model_type, checkpoint):
    global _HQ_SAM_PREDICTOR
    if _HQ_SAM_PREDICTOR: return _HQ_SAM_PREDICTOR
    from segment_anything_hq import sam_model_registry, SamPredictor
    if not checkpoint.exists(): raise FileNotFoundError(f"Missing: {checkpoint}")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint)).to("cuda" if torch.cuda.is_available() else "cpu")
    _HQ_SAM_PREDICTOR = SamPredictor(sam)
    return _HQ_SAM_PREDICTOR

def get_sam2_predictor(config_file, checkpoint):
    global _SAM2_PREDICTOR
    if _SAM2_PREDICTOR: return _SAM2_PREDICTOR
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    ckpt_path = str(checkpoint)
    if not Path(ckpt_path).exists(): 
        raise FileNotFoundError(f"Missing SAM2 checkpoint: {ckpt_path}")
    sam2_model = build_sam2(config_file, ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu")
    _SAM2_PREDICTOR = SAM2ImagePredictor(sam2_model)
    return _SAM2_PREDICTOR

def get_yolo_world_cached(yolo_model_name, classes, device="cuda"):
    global _YOLO_WORLD_MODEL, _YOLO_WORLD_CACHE_KEY
    normalized_device = _normalize_device(device)
    classes_tuple = tuple(sorted(classes))
    current_key = (yolo_model_name, classes_tuple)

    if _YOLO_WORLD_MODEL is not None and _YOLO_WORLD_CACHE_KEY == current_key:
        return _YOLO_WORLD_MODEL

    print(f"--- Loading YOLO-World ({yolo_model_name}) ---")
    print(f"--- Setting Classes: {classes} ---")
    
    from ultralytics import YOLOWorld
    model = YOLOWorld(yolo_model_name)
    model.to(normalized_device)
    if classes:
        model.set_classes(classes)
        
    _YOLO_WORLD_MODEL = model
    _YOLO_WORLD_CACHE_KEY = current_key
    return model


# --- Prompt Generation Helpers ---

def get_yolo_cold_boxes(
    rgb_img: np.ndarray,
    yolo_model: str,
    classes: list[str],
    conf: float,
    device: str
) -> list[tuple[np.ndarray, int]]:
    
    yolo = get_yolo_world_cached(yolo_model, classes, device)
    # Exclude keywords that refer to the thermal source itself
    SKIP_KEYWORDS = ["wok", "pot", "cooker"] 
    
    results = yolo.predict(rgb_img, conf=conf, device=_normalize_device(device), verbose=False)
    
    cold_items = []
    if len(results[0].boxes) > 0:
        boxes_data = results[0].boxes.data.cpu().numpy()
        names = results[0].names
        
        for det in boxes_data:
            x1, y1, x2, y2, conf_score, cls_id = det
            label = names[int(cls_id)].lower()
            
            # Detect everything except the pot itself (let thermal handle the pot main body)
            # BUT if detection is 'spatula' or 'egg', we definitely want it.
            if any(k in label for k in SKIP_KEYWORDS):
                continue
                
            cold_items.append((np.array([x1, y1, x2, y2]), int(cls_id)))
            
    return cold_items


# --- Hybrid SAM Methods (Multi-Class) ---

def run_sam_hybrid(
    predictor, rgb_img, thermal_img, 
    yolo_params: dict, thermal_params: dict
) -> np.ndarray:
    
    final_mask = np.zeros(rgb_img.shape[:2], dtype=np.int32)
    predictor.set_image(rgb_img)
    
    # --- Step 1: Thermal Objects (The Wok) ---
    # Usually ID 1 (Wok)
    t_points, t_labels, t_box = thermal_prompts(
        thermal_img, low=thermal_params['low'], topk=thermal_params['topk']
    )
    t_ref_mask = thermal_mask(thermal_img, low=thermal_params['low'])
    
    wok_binary_mask = None
    
    if t_box is not None:
        masks, scores, _ = predictor.predict(
            point_coords=t_points, point_labels=t_labels, 
            box=t_box[None, :], multimask_output=True
        )
        wok_binary_mask = select_best_mask_by_thermal(masks, scores, t_ref_mask)
        
        # Stricter Filter for huge table masks (0.35)
        if filter_huge_masks(wok_binary_mask, max_ratio=0.35):
            wok_binary_mask = None 
        else:
            final_mask[wok_binary_mask > 0] = 1 
        
    # --- Step 2: YOLO Objects (Spatula, Egg, Bowl) ---
    c_items = get_yolo_cold_boxes(
        rgb_img, 
        yolo_model=yolo_params['model'],
        classes=yolo_params['classes'],
        conf=yolo_params['conf'],
        device=yolo_params['device']
    )
    
    for box, cls_idx in c_items:
        masks, scores, _ = predictor.predict(
            point_coords=None, point_labels=None,
            box=box[None, :], multimask_output=True
        )
        obj_mask = select_best_mask_by_confidence(masks, scores)
        
        # Filter huge masks (Workbench/Table false positives)
        if filter_huge_masks(obj_mask, max_ratio=0.35):
            continue
        
        # === KEY FIX: Logical Subtraction ===
        # If we found a spatula/egg/bowl, it MUST NOT be part of the Wok.
        # "Cut out" this object from the Wok mask.
        if wok_binary_mask is not None:
             # Remove object pixels from wok (set to 0 where object is 1)
             final_mask[(final_mask == 1) & (obj_mask > 0)] = 0

        # Then draw the object
        final_mask[obj_mask > 0] = cls_idx + 1

    return final_mask


def naive_sam_prompt_mask(rgb_img, thermal_img, root, args):
    predictor = get_sam_predictor(root, args.sam_model_type, args.sam_checkpoint)
    yolo_params = {
        'model': args.yolo_world_model,
        'classes': [c.strip() for c in args.yolo_world_classes.split(',')],
        'conf': args.yolo_world_conf,
        'device': getattr(args, 'device', 'cuda')
    }
    thermal_params = {'low': args.sam_low or args.thermal_low, 'topk': args.sam_topk}
    return run_sam_hybrid(predictor, rgb_img, thermal_img, yolo_params, thermal_params)


def hq_sam_prompt_mask(rgb_img, thermal_img, root, args):
    hq_ckpt = args.hq_sam_checkpoint or (args.root / "weights" / "sam_hq_vit_b.pth")
    predictor = get_hq_sam_predictor(root, args.sam_model_type, hq_ckpt)
    yolo_params = {
        'model': args.yolo_world_model,
        'classes': [c.strip() for c in args.yolo_world_classes.split(',')],
        'conf': args.yolo_world_conf,
        'device': getattr(args, 'device', 'cuda')
    }
    thermal_params = {'low': args.sam_low or args.thermal_low, 'topk': args.sam_topk}
    return run_sam_hybrid(predictor, rgb_img, thermal_img, yolo_params, thermal_params)


def sam2_prompt_mask(rgb_img, thermal_img, args):
    sam2_ckpt = args.sam2_checkpoint or (args.root / "weights" / "sam2_hiera_large.pt")
    predictor = get_sam2_predictor(args.sam2_config, sam2_ckpt)
    yolo_params = {
        'model': args.yolo_world_model,
        'classes': [c.strip() for c in args.yolo_world_classes.split(',')],
        'conf': args.yolo_world_conf,
        'device': getattr(args, 'device', 'cuda')
    }
    thermal_params = {'low': args.sam_low or args.thermal_low, 'topk': args.sam_topk}
    return run_sam_hybrid(predictor, rgb_img, thermal_img, yolo_params, thermal_params)


def yolo_world_mixed_logic(rgb_img, thermal_img, args):
    """
    Optimized: Uses YOLO-World (Cached) + SAM 2 (Stronger).
    """
    device = getattr(args, 'device', 'cuda')
    classes = [c.strip() for c in args.yolo_world_classes.split(',')]
    
    yolo = get_yolo_world_cached(args.yolo_world_model, classes, device)
    results = yolo.predict(rgb_img, conf=args.yolo_world_conf, device=_normalize_device(device), verbose=False)
    
    final_mask = np.zeros(rgb_img.shape[:2], dtype=np.int32)
    if len(results[0].boxes) == 0:
        return final_mask

    sam2_ckpt = args.sam2_checkpoint or (args.root / "weights" / "sam2_hiera_tiny.pt") 
    sam2_config = "sam2_hiera_t.yaml" if "tiny" in str(sam2_ckpt) else args.sam2_config
    
    try:
        predictor = get_sam2_predictor(sam2_config, sam2_ckpt)
        predictor.set_image(rgb_img)
        
        boxes_data = results[0].boxes.data.cpu().numpy()
        input_boxes = boxes_data[:, :4]
        class_ids = boxes_data[:, 5].astype(int)
        
        masks, scores, _ = predictor.predict(
            point_coords=None, point_labels=None,
            box=input_boxes, multimask_output=False 
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # Composite Mask with filtering
        for i, m in enumerate(masks):
            cls_idx = class_ids[i]
            binary_m = (m > 0.0).astype(np.uint8)
            binary_m = refine_mask_morphology(binary_m)
            
            # Filter huge masks (Workbench) - STRICTER (0.35)
            if filter_huge_masks(binary_m, max_ratio=0.35):
                continue

            final_mask[binary_m > 0] = cls_idx + 1

    except Exception as e:
        print(f"Warning: SAM 2 inference failed in yolo_world_sam: {e}")
        pass
        
    return final_mask


def generate_mask_by_method(rgb_img, thermal_img, method, args):
    mask = None
    
    if method == "thermal":
        mask = thermal_mask(thermal_img, low=args.thermal_low)
    elif method == "thermal_cluster":
        mask = thermal_cluster_mask(
            thermal_img, k=args.cluster_k, iters=args.cluster_iters,
            min_ratio=args.cluster_min_ratio, low_fallback=args.thermal_low
        )
    elif method == "naive_sam":
        mask = naive_sam_prompt_mask(rgb_img, thermal_img, root=args.root, args=args)
    elif method == "hq_sam":
        mask = hq_sam_prompt_mask(rgb_img, thermal_img, root=args.root, args=args)
    elif method == "sam2":
        mask = sam2_prompt_mask(rgb_img, thermal_img, args=args)
    elif method == "yolo_world_sam":
        mask = yolo_world_mixed_logic(rgb_img, thermal_img, args=args)
    else:
        raise ValueError(f"Unknown method: {method}")

    if mask is not None:
        if mask.sum() == 0:
            ref = thermal_mask(thermal_img, low=args.thermal_low)
            if ref.sum() > 200:
                mask = thermal_cluster_mask(
                    thermal_img, k=args.cluster_k, iters=args.cluster_iters,
                    min_ratio=args.cluster_min_ratio, low_fallback=args.thermal_low
                ).astype(np.int32)
                
    return mask