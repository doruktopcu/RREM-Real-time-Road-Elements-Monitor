import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import threading
import time
from rrem_monitor import RREMMonitor

# Fix for macOS/OpenCV threading crash (libavcodec)
cv2.setNumThreads(0)

class RREMGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RREM: Real-time Road Elements Monitor")
        self.root.geometry("1000x800")
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Header
        header = ttk.Label(root, text="RREM Dashboard", font=("Helvetica", 20, "bold"))
        header.pack(pady=10)
        
        # Control Frame
        control_frame = ttk.Frame(root)
        control_frame.pack(pady=5, fill=tk.X, padx=10)
        
        self.btn_load = ttk.Button(control_frame, text="Load Video File", command=self.load_video)
        self.btn_load.pack(side=tk.LEFT, padx=5)
        
        self.btn_webcam = ttk.Button(control_frame, text="Use Webcam", command=self.use_webcam)
        self.btn_webcam.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = ttk.Button(control_frame, text="Stop", command=self.stop_monitor, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        # Playback Controls
        self.btn_prev = ttk.Button(control_frame, text="<<", command=lambda: self.seek_relative(-50), state=tk.DISABLED)
        self.btn_prev.pack(side=tk.LEFT, padx=2)
        
        self.btn_pause = ttk.Button(control_frame, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=2)
        
        self.btn_next = ttk.Button(control_frame, text=">>", command=lambda: self.seek_relative(50), state=tk.DISABLED)
        self.btn_next.pack(side=tk.LEFT, padx=2)
        
        # Zoom Controls
        self.btn_zoom_out = ttk.Button(control_frame, text="-", width=3, command=lambda: self.change_zoom(-0.1), state=tk.DISABLED)
        self.btn_zoom_out.pack(side=tk.LEFT, padx=5)
        
        self.lbl_zoom = ttk.Label(control_frame, text="100%")
        self.lbl_zoom.pack(side=tk.LEFT, padx=2)
        
        self.btn_zoom_in = ttk.Button(control_frame, text="+", width=3, command=lambda: self.change_zoom(0.1), state=tk.DISABLED)
        self.btn_zoom_in.pack(side=tk.LEFT, padx=2)
        
        self.lbl_status = ttk.Label(control_frame, text="Status: Idle", foreground="gray")
        self.lbl_status.pack(side=tk.RIGHT, padx=5)

        # Main Content Area (Video + Logs)
        content_frame = ttk.Frame(root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Video Display
        # Center alignment using anchor and pack options
        self.video_label = ttk.Label(content_frame, text="Video Feed", background="black", foreground="white", anchor=tk.CENTER)
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Alert Log
        log_frame = ttk.LabelFrame(content_frame, text="Alert Log")
        log_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        self.log_tree = ttk.Treeview(log_frame, columns=("Time", "Alert"), show="headings", height=20)
        self.log_tree.heading("Time", text="Time")
        self.log_tree.heading("Alert", text="Hazard")
        self.log_tree.column("Time", width=100)
        self.log_tree.column("Alert", width=200)
        self.log_tree.pack(fill=tk.BOTH, expand=True)
        
        # Monitor Instance
        self.monitor = RREMMonitor()
        self.running = False
        self.paused = False
        self.zoom_level = 1.0
        self.current_frame_img = None # Store last frame for redraw during pause
        self.thread = None
        
    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.avi")])
        if file_path:
            self.start_monitor(file_path)
            
    def use_webcam(self):
        self.start_monitor(0) # 0 for webcam
        
    def start_monitor(self, source):
        if self.running:
            self.stop_monitor()
            
        try:
            self.monitor.start_capture(source)
            self.running = True
            self.paused = False
            self.zoom_level = 1.0
            self.update_controls_state(tk.NORMAL)
            self.lbl_status.config(text=f"Monitoring: {source}", foreground="green")
            self.lbl_zoom.config(text="100%")
            
            # Run loop in separate thread to not block GUI
            self.thread = threading.Thread(target=self.update_feed, daemon=True)
            self.thread.start()
            
        except Exception as e:
            self.lbl_status.config(text=f"Error: {str(e)}", foreground="red")
            
    def stop_monitor(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.monitor.stop_capture()
        self.update_controls_state(tk.DISABLED)
        self.btn_load.config(state=tk.NORMAL)
        self.btn_webcam.config(state=tk.NORMAL)
        self.lbl_status.config(text="Status: Stopped", foreground="gray")
        
    def update_controls_state(self, state):
        self.btn_stop.config(state=state)
        self.btn_pause.config(state=state)
        self.btn_prev.config(state=state)
        self.btn_next.config(state=state)
        self.btn_zoom_in.config(state=state)
        self.btn_zoom_out.config(state=state)
        if state == tk.NORMAL:
            self.btn_load.config(state=tk.DISABLED)
            self.btn_webcam.config(state=tk.DISABLED)
            self.btn_pause.config(text="Pause")
            
    def toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.config(text="Play" if self.paused else "Pause")
        
    def seek_relative(self, frames):
        curr, total, _ = self.monitor.get_video_info()
        self.monitor.seek_frame(curr + frames)
        if self.paused:
            # If paused, we need to grab the new frame specifically to update UI
            frame, _ = self.monitor.process_frame()
            if frame is not None:
                # Convert CV2 frame (BGR) to RGB for Tkinter
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                self.current_frame_img = img
                self.root.after(0, self.update_gui_image, img)
    
    def change_zoom(self, delta):
        self.zoom_level = max(0.2, min(5.0, self.zoom_level + delta))
        self.lbl_zoom.config(text=f"{int(self.zoom_level * 100)}%")
        # Force redraw if paused
        if self.paused and self.current_frame_img:
            self.root.after(0, self.update_gui_image, self.current_frame_img)

    def update_feed(self):
        while self.running:
            if not self.paused:
                frame, alerts = self.monitor.process_frame()
                
                if frame is None:
                    self.running = False
                    self.lbl_status.config(text="Status: Finished", foreground="blue")
                    self.root.after(0, self.reset_buttons)
                    break
                
                # Convert CV2 frame (BGR) to RGB for Tkinter
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                self.current_frame_img = img # Store for redraws
                
                # Update GUI in main thread - passing the PIL image object
                self.root.after(0, self.update_gui_image, img)
                
                if alerts:
                    timestamp = time.strftime("%H:%M:%S")
                    for alert in alerts:
                        self.root.after(0, self.log_alert, timestamp, alert)
            else:
                time.sleep(0.1) # low usage when paused
                
            # Control frame rate roughly
            time.sleep(0.01)

    def update_gui_image(self, img_pil):
        # RESIZE LOGIC HERE (Main Thread)
        win_w = self.root.winfo_width()
        win_h = self.root.winfo_height()
        
        if win_w > 1 and win_h > 1:
            target_w = int(win_w * 0.6 * self.zoom_level) # Apply Zoom
            target_h = int(win_h * 0.6 * self.zoom_level)
            
            aspect_ratio = img_pil.width / img_pil.height
            new_h = int(target_w / aspect_ratio)
            
            # Bounds logic (optional: limit max size? nah let them zoom)
            
            # Use resize (not thumbnail) to force exact fit
            img_pil = img_pil.resize((target_w, new_h), Image.Resampling.LANCZOS)
            
        imgtk = ImageTk.PhotoImage(image=img_pil)
        
        self.video_label.imgtk = imgtk # Keep reference
        self.video_label.configure(image=imgtk)

    def log_alert(self, timestamp, alert):
        # Insert at top
        self.log_tree.insert("", 0, values=(timestamp, alert))
        # Keep log size manageable
        if len(self.log_tree.get_children()) > 100:
            self.log_tree.delete(self.log_tree.get_children()[-1])
            
    def reset_buttons(self):
        self.stop_monitor()

if __name__ == "__main__":
    root = tk.Tk()
    app = RREMGUI(root)
    root.mainloop()
