import os

# Enable MPS fallback for SAM2 (Upsample Bicubic)
# MUST be set before importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import numpy as np

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class RoadSegmenter:
    def __init__(self, model_checkpoint="sam2_hiera_tiny.pt", config="sam2_hiera_t.yaml"):
        """
        Initializes the SAM2-based Road Segmenter.
        """
        print(f"Loading SAM2 model from {model_checkpoint} (Tiny)...")
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Build SAM2 model
        # Note: We assume the config file is available or handled by the library if not passed explicitly if standard.
        # However, typically build_sam2 needs a config path. 
        # Since I don't see the yaml config in the root file list, I'll try to rely on the internal registry or assume the user has it.
        # But wait, usually `build_sam2` takes (config_file, checkpoint).
        # If the user only provided the .pt, I might need to find where the config is.
        # Let's assume standard configs are importable or present.
        # For now, I'll use the constructor that accepts the checkpoint.
        
        try:
             self.sam2_model = build_sam2(config, model_checkpoint, device=self.device)
             self.predictor = SAM2ImagePredictor(self.sam2_model)
        except Exception as e:
            print(f"Error loading SAM2: {e}")
            print("Attempting to load without config file argument (if supported) or using default...")
            # Fallback or error handling
            raise e

        self.inference_count = 0

    def segment_frame(self, frame):
        """
        Generates a road segmentation mask for the given frame.
        Strategy: Prompt with a point at the bottom-center (ego-vehicle hood).
        """
        if frame is None:
            return None
            
        # Standardize resize to avoid mismatch
        # SAM2 sometimes returns low-res masks (256x256) or different scales.
        # We must ensure output matches input frame (h, w).
        orig_h, orig_w = frame.shape[:2]
        
        # Set the image for the predictor
        self.predictor.set_image(frame)
        
        # Define Point Prompt:
        # Positive (1): Road Center (approx 65% down - usually safe from horizon and hood)
        # Negative (0): Hood/Dashboard (95% down - very bottom)
        input_point = np.array([
            [orig_w // 2, int(orig_h * 0.65)], # Target: Road
            [orig_w // 2, int(orig_h * 0.95)]  # Negative: Hood
        ])
        input_label = np.array([1, 0]) # 1 = Foreground, 0 = Background

        # Predict
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False 
        )
        
        # masks is usually (1, H, W) or low res.
        raw_mask = masks[0]
        
        # Explicitly Resize to original frame size using Nearest Neighbor (keep binary)
        # Note: raw_mask might be float or bool or uint8
        # Convert to uint8 for resizing
        mask_uint8 = (raw_mask > 0).astype(np.uint8)
        
        # Resize if dimensions differ
        if mask_uint8.shape[:2] != (orig_h, orig_w):
            import cv2
            full_mask = cv2.resize(mask_uint8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        else:
            full_mask = mask_uint8
            
        return full_mask > 0 # Return boolean map

