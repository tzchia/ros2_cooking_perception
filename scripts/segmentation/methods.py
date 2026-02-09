"""Mask generation methods: RGB+Thermal Hybrid with Qwen2.5-VL, SAHI & DINO."""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings
import re

import numpy as np
import torch
import cv2
from PIL import Image

# --- GroundingDINO Patch ---
try:
    from transformers.models.bert.modeling_bert import BertModel
    if not hasattr(BertModel, 'get_head_mask'):
        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is not None:
                return self.get_extended_attention_mask(head_mask, (num_hidden_layers,), device=self.device)
            return None
        BertModel.get_head_mask = get_head_mask
except: pass

# --- Device Helper ---
def _normalize_device(device: str | int | None) -> str:
    if device is None: return "cpu"
    d = str(device).strip()
    if d == "-1" or d.lower() == "cpu": return "cpu"
    if d.isdigit(): return f"cuda:{d}"
    if d.startswith("cuda") or d.startswith("mps"): return d
    return d

# --- Helper Functions ---
def normalize_thermal(thermal_img: np.ndarray) -> np.ndarray:
    tmin = float(thermal_img.min())
    tmax = float(thermal_img.max())
    if tmax == tmin: return np.zeros_like(thermal_img, dtype=np.float32)
    return (thermal_img - tmin) / (tmax - tmin + 1e-6)

def thermal_mask(thermal_img: np.ndarray, low: float = 0.6) -> np.ndarray:
    norm = normalize_thermal(thermal_img)
    return (norm >= low).astype(np.uint8)

def refine_mask_morphology(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0: return mask
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    new_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 20: new_mask[labels == i] = 1
    mask = new_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)
    return mask

def filter_huge_masks(mask: np.ndarray, max_ratio: float = 0.40) -> bool:
    total = mask.shape[0] * mask.shape[1]
    return (np.sum(mask > 0) / total) > max_ratio

def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    return float(inter) / float(union) if union > 0 else 0.0

# --- Thermal Prompts ---
def thermal_prompts(thermal_img: np.ndarray, low: float = 0.6, topk: int = 20):
    norm = normalize_thermal(thermal_img)
    mask = norm >= low
    ys, xs = np.where(mask)
    if len(xs) == 0: return None, None, None
    coords = np.column_stack([ys, xs])
    vals = norm[mask]
    top_idx = np.argsort(vals)[-min(topk, len(vals)) :]
    points = coords[top_idx][:, ::-1].copy() 
    labels = np.ones(len(points), dtype=np.int32)
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
    return points, labels, box

def thermal_cluster_mask(thermal_img, k=3, iters=10, min_ratio=0.002, low_fallback=0.6):
    return thermal_mask(thermal_img, low=low_fallback)

def select_best_mask_by_thermal(masks, scores, ref_mask):
    if masks.shape[0] == 0: return np.zeros_like(ref_mask)
    best_score = -float('inf')
    best_idx = 0
    ref_area = ref_mask.sum()
    for i in range(masks.shape[0]):
        m = masks[i].astype(np.uint8)
        iou = calculate_iou(m, ref_mask)
        ratio = m.sum() / (ref_area + 1e-6)
        penalty = 0.0
        if ratio > 3.0: penalty = 2.0
        elif ratio < 0.1: penalty = 0.5
        score = iou - penalty + (0.05 * scores[i])
        if score > best_score: best_score = score; best_idx = i
    return refine_mask_morphology(masks[best_idx].astype(np.uint8))

def select_best_mask_by_confidence(masks, scores):
    if masks.shape[0] == 0: return np.zeros((2,2), dtype=np.uint8)
    best_idx = int(np.argmax(scores))
    return refine_mask_morphology(masks[best_idx].astype(np.uint8))

# --- Model Loaders ---
_MODELS = {}

def get_predictor(args, method_name):
    device = _normalize_device(args.device)
    key = f"predictor_{method_name}"
    if key in _MODELS: return _MODELS[key]
    
    if "sam2" in method_name or "qwen" in method_name:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2(args.sam2_config, str(args.sam2_checkpoint), device=device)
        pred = SAM2ImagePredictor(model)
    elif "hq_sam" in method_name:
        from segment_anything_hq import sam_model_registry, SamPredictor
        ckpt = args.hq_sam_checkpoint or args.root/"weights"/"sam_hq_vit_b.pth"
        sam = sam_model_registry[args.sam_model_type](checkpoint=str(ckpt)).to(device)
        pred = SamPredictor(sam)
    else: 
        from segment_anything import sam_model_registry, SamPredictor
        ckpt = args.sam_checkpoint
        sam = sam_model_registry[args.sam_model_type](checkpoint=str(ckpt)).to(device)
        pred = SamPredictor(sam)
    _MODELS[key] = pred
    return pred

def get_yolo_world(model_path, device):
    device = _normalize_device(device)
    if "yolo" in _MODELS: return _MODELS["yolo"]
    from ultralytics import YOLOWorld
    model = YOLOWorld(model_path)
    if device != "cpu": model.to(device)
    _MODELS["yolo"] = model
    return model

def get_yolo_seg(model_path, device):
    device = _normalize_device(device)
    key = f"yolo_seg_{model_path}"
    if key in _MODELS: return _MODELS[key]
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    if device != "cpu": model.to(device)
    _MODELS[key] = model
    return model

def get_grounding_dino(config, ckpt, device):
    device = _normalize_device(device)
    if "dino" in _MODELS: return _MODELS["dino"]
    from groundingdino.util.inference import load_model
    model = load_model(str(config), str(ckpt)).to(device)
    _MODELS["dino"] = model
    return model

def get_qwen_vl(model_id, device):
    device = _normalize_device(device)
    if "qwen" in _MODELS: return _MODELS["qwen"]
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    # Qwen2.5-VL is heavy, ensure bfloat16/float16 if cuda
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if device == "cpu": dtype = torch.float32
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_id)
    _MODELS["qwen"] = (model, processor)
    return model, processor

# --- Cold Prompt Generators ---

def get_yolo_cold_boxes(rgb, model_path, classes, conf, device):
    device = _normalize_device(device)
    yolo = get_yolo_world(model_path, device)
    SKIP = ["wok", "pot", "cooker", "pan"] 
    try: yolo.set_classes(classes)
    except: pass
    res = yolo.predict(rgb, conf=conf, device=device, verbose=False)
    boxes = []
    if len(res[0].boxes) > 0:
        data = res[0].boxes.data.cpu().numpy()
        names = res[0].names
        for det in data:
            x1, y1, x2, y2, _, cls_id = det
            label = names[int(cls_id)].lower()
            if any(k in label for k in SKIP): continue
            boxes.append((np.array([x1, y1, x2, y2]), int(cls_id)))
    return boxes

def get_yolo_sahi_boxes(rgb, model_path, classes, conf, slice_size, overlap, device):
    device = _normalize_device(device)
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8', model_path=model_path, confidence_threshold=conf, device=device
    )
    try: detection_model.model.set_classes(classes)
    except: pass
    result = get_sliced_prediction(
        rgb, detection_model, slice_height=slice_size, slice_width=slice_size,
        overlap_height_ratio=overlap, overlap_width_ratio=overlap
    )
    SKIP = ["wok", "pot", "cooker", "pan"]
    boxes = []
    for pred in result.object_prediction_list:
        label = pred.category.name.lower()
        if any(k in label for k in SKIP): continue
        cls_idx = 0
        try: cls_idx = classes.index(pred.category.name)
        except: 
            for i, c in enumerate(classes):
                if label in c or c in label: cls_idx = i; break
        bbox = pred.bbox.to_xyxy()
        boxes.append((np.array(bbox), cls_idx))
    return boxes

def get_dino_boxes(rgb, config, ckpt, classes, box_thresh, text_thresh, device="cuda"):
    device = _normalize_device(device)
    from groundingdino.util.inference import predict
    import groundingdino.datasets.transforms as T
    model = get_grounding_dino(config, ckpt, device)
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_pil = Image.fromarray(rgb).convert("RGB")
    img_tensor, _ = transform(img_pil, None)
    SKIP = ["wok", "pot", "cooker"]
    cold_classes = [c for c in classes if not any(k in c for k in SKIP)]
    prompt = ". ".join(cold_classes)
    boxes, logits, phrases = predict(
        model=model, image=img_tensor, caption=prompt,
        box_threshold=box_thresh, text_threshold=text_thresh, device=device
    )
    h, w = rgb.shape[:2]
    out_boxes = []
    for i, box in enumerate(boxes):
        phrase = phrases[i]
        cx, cy, bw, bh = box.numpy()
        x1, x2 = (cx - bw/2) * w, (cx + bw/2) * w
        y1, y2 = (cy - bh/2) * h, (cy + bh/2) * h
        cls_idx = 0
        best_match_len = 0
        for idx, cname in enumerate(classes):
            if phrase in cname or cname in phrase:
                if len(cname) > best_match_len: cls_idx = idx; best_match_len = len(cname)
        out_boxes.append((np.array([x1, y1, x2, y2]), cls_idx))
    return out_boxes

def get_qwen_boxes(rgb, model_id, classes, text_prompt, device="cuda"):
    """
    Use Qwen2.5-VL to detect objects and return boxes for SAM2.
    Output parsing depends on Qwen2.5-VL's detection format.
    """
    from qwen_vl_utils import process_vision_info
    
    model, processor = get_qwen_vl(model_id, device)
    
    # Construct prompts for each class or all at once?
    # Qwen VL supports "Detect {obj}" style prompts.
    SKIP = ["wok", "pot", "cooker"]
    cold_classes = [c for c in classes if not any(k in c for k in SKIP)]
    
    out_boxes = []
    h, w = rgb.shape[:2]
    
    # Batch detection prompt
    prompt_text = "Detect " + ", ".join(cold_classes)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": Image.fromarray(rgb)},
            {"type": "text", "text": prompt_text}
        ]}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]
    
    # Parse Qwen2.5-VL detection output
    # Format typically: <|box_start|>(y1,x1),(y2,x2)<|box_end|>label
    # Coordinates are normalized 0-1000
    
    pattern = r"<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>([\w\s]+)"
    matches = re.findall(pattern, output_text)
    
    for match in matches:
        y1_n, x1_n, y2_n, x2_n, label = match
        label = label.strip().lower()
        
        # Filter classes
        cls_idx = -1
        for i, c in enumerate(classes):
            if c in label or label in c:
                cls_idx = i
                break
        
        if cls_idx == -1: continue # Unknown class
        if any(k in classes[cls_idx] for k in SKIP): continue # Skip hot objects
        
        # Convert 1000-scale to pixels
        x1 = int(x1_n) / 1000 * w
        y1 = int(y1_n) / 1000 * h
        x2 = int(x2_n) / 1000 * w
        y2 = int(y2_n) / 1000 * h
        
        out_boxes.append((np.array([x1, y1, x2, y2]), cls_idx))
        
    return out_boxes

# --- Execution Core ---

def run_hybrid_pipeline(rgb, thermal, predictor, method_cfg, args):
    final_mask = np.zeros(rgb.shape[:2], dtype=np.int32)
    predictor.set_image(rgb)
    
    # 1. Hot Object (Wok) - ID 1
    t_points, t_labels, t_box = thermal_prompts(thermal, low=args.thermal_low)
    t_ref = thermal_mask(thermal, low=args.thermal_low)
    wok_binary = None
    
    if t_box is not None:
        masks, scores, _ = predictor.predict(
            point_coords=t_points, point_labels=t_labels, box=t_box[None,:], multimask_output=True
        )
        wok_binary = select_best_mask_by_thermal(masks, scores, t_ref)
        if not filter_huge_masks(wok_binary):
            final_mask[wok_binary > 0] = 1 
            
    # 2. Cold Objects
    cold_type = method_cfg['type']
    classes_list = [c.strip() for c in args.yolo_world_classes.split(',')]
    
    boxes = []
    if cold_type == 'sahi':
        boxes = get_yolo_sahi_boxes(
            rgb, args.yolo_world_model, classes_list, args.yolo_world_conf,
            args.sahi_slice_size, args.sahi_overlap, args.device
        )
    elif cold_type == 'dino':
        boxes = get_dino_boxes(
            rgb, args.dino_config, args.dino_checkpoint, classes_list,
            args.dino_box_threshold, args.dino_text_threshold, args.device
        )
    elif cold_type == 'qwen':
        boxes = get_qwen_boxes(
            rgb, args.qwen_model, classes_list, args.qwen_text_prompt, args.device
        )
    else: # Standard YOLO
        boxes = get_yolo_cold_boxes(
            rgb, args.yolo_world_model, classes_list, args.yolo_world_conf, args.device
        )
        
    # Segment Cold Objects
    for box, cls_id in boxes:
        masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=box[None,:], multimask_output=True)
        obj_mask = select_best_mask_by_confidence(masks, scores)
        
        if filter_huge_masks(obj_mask): continue
        if wok_binary is not None: final_mask[(final_mask==1) & (obj_mask>0)] = 0
        final_mask[obj_mask > 0] = cls_id + 1
        
    return final_mask

def yolo_seg_mask(rgb: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    model = get_yolo_seg(args.yolo_seg_model, args.device)
    preds = model.predict(rgb, conf=args.yolo_seg_conf, iou=args.yolo_seg_iou, imgsz=args.yolo_seg_imgsz, max_det=args.yolo_seg_max_det, device=_normalize_device(args.device), verbose=False)
    if not preds or preds[0].masks is None: return np.zeros(rgb.shape[:2], dtype=np.int32)
    res = preds[0]
    masks = res.masks.data.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    order = np.argsort(-confs)
    final_mask = np.zeros(rgb.shape[:2], dtype=np.int32)
    for idx in order:
        m_resized = cv2.resize(masks[idx], (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        final_mask[(final_mask == 0) & (m_resized > 0.5)] = int(cls_ids[idx]) + 1
    return final_mask

def generate_mask_by_method(rgb, thermal, method, args):
    if method == "thermal": return thermal_mask(thermal, args.thermal_low)
    if method == "thermal_cluster": return thermal_cluster_mask(thermal, low_fallback=args.thermal_low)
    if method == "yolo_seg": return yolo_seg_mask(rgb, args)
    
    predictor = get_predictor(args, method)
    cfg = {'type': 'yolo'} 
    if 'sahi' in method: cfg['type'] = 'sahi'
    if 'dino' in method: cfg['type'] = 'dino'
    if 'qwen' in method: cfg['type'] = 'qwen'
    
    mask = run_hybrid_pipeline(rgb, thermal, predictor, cfg, args)
    if mask.sum() == 0 and thermal_mask(thermal, args.thermal_low).sum() > 200:
        return thermal_cluster_mask(thermal, low_fallback=args.thermal_low).astype(np.int32)
    return mask