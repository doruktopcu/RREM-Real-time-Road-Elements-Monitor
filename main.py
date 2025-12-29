from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.core.window import Window

import cv2
import numpy as np

# Import our mobile modules
from mobile_utils import HazardAnalyzer, HazardStabilizer, DistanceMonitor, BoxesShim
from yolo_tflite import YoloTFLite

class RREMApp(App):
    def build(self):
        # UI Structure: Video Feed with Alert Overlay
        self.layout = BoxLayout(orientation='vertical')
        
        # Dashboard Header
        self.header = Label(text="RREM Mobile", size_hint=(1, 0.1), font_size='20sp')
        self.layout.add_widget(self.header)
        
        # Video Feed
        self.img1 = Image(size_hint=(1, 0.8))
        self.layout.add_widget(self.img1)
        
        # Alert Footer
        self.alert_label = Label(text="Status: Initializing...", size_hint=(1, 0.1), color=(1, 0, 0, 1), font_size='18sp')
        self.layout.add_widget(self.alert_label)
        
        # Initialize Logic
        # Note: On Android, camera index 0 usually works for back camera if configured.
        # Sometimes 1. Users might need to toggle.
        self.capture = cv2.VideoCapture(0)
        
        # Initialize RREM Components
        try:
            self.detector = YoloTFLite("assets/yolo11n.tflite")
            self.analyzer = HazardAnalyzer()
            self.stabilizer = HazardStabilizer() # Will init with default 1920x1080, updated later
            self.msg = "System Ready"
        except Exception as e:
            self.msg = f"Error: {str(e)}"
            self.detector = None

        Clock.schedule_interval(self.update, 1.0/30.0) # 30 FPS target
        
        return self.layout

    def update(self, dt):
        if not self.capture.isOpened():
            return
            
        ret, frame = self.capture.read()
        if not ret:
            return
            
        # Flip if needed (front cam usually mirrored, back cam fine)
        # frame = cv2.flip(frame, 0) 
        
        # Update Stabilizer dims if first run
        h, w = frame.shape[:2]
        if hasattr(self, 'stabilizer') and (self.stabilizer.width != w or self.stabilizer.height != h):
             self.stabilizer = HazardStabilizer(frame_width=w, frame_height=h)

        if self.detector:
            # 1. Detect
            detections = self.detector.detect(frame)
            
            # 2. Stabilize & Filter (Red Zone)
            # track_ids mock (no tracking in basic tflite wrapper yet, so id=None)
            # If we want tracking, we need simple_tracker port. 
            # For now, pass empty IDs list so stabilizer treats all as new/transient or handles them.
            # Stabilizer expects list of IDs.
            valid_dets = self.stabilizer.update(detections, []) 
            
            # 3. Analyze
            # Create Shim
            # Analyzer expects BoxesShim object with attributes
            shim_list = [BoxesShim.BoxShimShim(d) for d in valid_dets] # Need to fix Shim instantiation
            # QuickShim definition inline or fix mobile_utils
            # Let's fix passing data to mobile_utils.BoxesShim
            
            class MiniShim:
                def __init__(self, d):
                    self.cls = np.array([float(d['class_id'])])
                    self.id = np.array([d['id']]) if d['id'] else None
                    self.xyxy = np.array([d['box']])
                    self.conf = np.array([d['conf']])
            
            shim_objs = [MiniShim(d) for d in valid_dets]
            boxes_shim = BoxesShim(shim_objs)
            
            alerts = self.analyzer.analyze(boxes_shim, self.detector.names, frame_shape=frame.shape, lane_bounds=(0, w))
            
            # 4. Visualization
            # Draw Red Zone
            frame = self.stabilizer.draw_debug_zone(frame)
            
            # Draw Boxes
            for det in valid_dets:
                 x1, y1, x2, y2 = map(int, det['box'])
                 label = f"{self.detector.names.get(det['class_id'], 'Obj')} {det['conf']:.2f}"
                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                 cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                 
            # Update Alerts
            if alerts:
                display_txt = " | ".join(alerts)
                self.alert_label.text = display_txt
            else:
                self.alert_label.text = "Scanning..."

        # Convert to Kivy Texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img1.texture = image_texture

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    RREMApp().run()
