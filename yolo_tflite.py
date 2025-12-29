import cv2
import numpy as np
import time

try:
    # On Android, we often use tflite_runtime
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Fallback for testing on Desktop if tensorflow is installed
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("TFLite not found")
        tflite = None

class YoloTFLite:
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Load Interpreter
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape'] # [1, 640, 640, 3] usually
        self.input_h = self.input_shape[1]
        self.input_w = self.input_shape[2]
        
        # COCO names (standard) or Custom
        # For this prototype we use standard COCO map or pass it in.
        # RREM uses partial map but let's assume full map
        self.names = {
             0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck',
             10: 'traffic light', 15: 'cat', 16: 'dog'
        }
        # In real usage, pass names dict
        
    def preprocess(self, img):
        """
        Resize image to input_shape with letterbox (padding)
        Returns: processed_img, ratio, (dw, dh)
        """
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = (self.input_w, self.input_h)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border

        # Convert to float32 and normalize
        img = img.astype(np.float32) / 255.0
        
        # Add Batch Dimension [1, H, W, 3]
        img = np.expand_dims(img, axis=0)
        
        return img, ratio, (dw, dh)

    def detect(self, image):
        # Preprocess
        input_data, ratio, pad = self.preprocess(image)
        
        # Set input
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index']) 
        # Output shape: [1, 84, 8400] (for YOLOv8/11 default) usually
        # 84 = x, y, w, h + 80 classes
        
        if output_data.ndim == 3 and output_data.shape[1] < output_data.shape[2]:
            # Convert [1, 84, 8400] back to [1, 8400, 84]
            output_data = np.transpose(output_data, (0, 2, 1))
            
        predictions = output_data[0] # [8400, 84]
        
        # NMS
        boxes = []
        confidences = []
        class_ids = []
        
        # Optimize iteratively?
        # Vectorized filtering
        # predictions: [N, 84]
        # x, y, w, h = 0,1,2,3
        # classes = 4..83
        
        box_data = predictions[:, :4]
        cls_data = predictions[:, 4:]
        
        # Get max confidence and class index
        max_scores = np.max(cls_data, axis=1)
        argmax_ids = np.argmax(cls_data, axis=1)
        
        # Filter by confidence
        mask = max_scores > self.conf_thres
        
        filtered_boxes = box_data[mask]
        filtered_scores = max_scores[mask]
        filtered_ids = argmax_ids[mask]
        
        # Convert XYWH to XYXY for NMS
        # box_data is center_x, center_y, w, h
        if len(filtered_boxes) == 0:
            return []

        cx = filtered_boxes[:, 0]
        cy = filtered_boxes[:, 1]
        w = filtered_boxes[:, 2]
        h = filtered_boxes[:, 3]
        
        x1 = cx - w/2
        y1 = cy - h/2
        
        # Scale back to original image size
        # (x - pad) / ratio
        pad_w, pad_h = pad
        ratio_w, ratio_h = ratio
        
        x1 = (x1 - pad_w) / ratio_w
        y1 = (y1 - pad_h) / ratio_h
        w = w / ratio_w
        h = h / ratio_h
        
        # Create boxes for cv2.NMSBoxes: [x, y, w, h] (top-left)
        # We already have x1, y1.
        
        nms_boxes = []
        nms_scores = []
        
        for i in range(len(filtered_boxes)):
            nms_boxes.append([int(x1[i]), int(y1[i]), int(w[i]), int(h[i])])
            nms_scores.append(float(filtered_scores[i]))
            
        # Run NMS
        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, self.conf_thres, self.iou_thres)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                # Format: {'box': [x1,y1,x2,y2], 'class_id': int, 'conf': float}
                bx, by, bw, bh = nms_boxes[i]
                results.append({
                    'box': [bx, by, bx+bw, by+bh],
                    'class_id': int(filtered_ids[i]),
                    'conf': float(filtered_scores[i]),
                    'id': None # No tracker in this basic version
                })
                
        return results
