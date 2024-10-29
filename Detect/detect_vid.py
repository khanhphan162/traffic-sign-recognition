import tkinter as tk
from tkinter import filedialog
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
import sys
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize image with unchanged aspect ratio using padding
    img_h, img_w = img.shape[0], img.shape[1]
    new_w, new_h = new_shape
    scale = min(new_w / img_w, new_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((new_shape[1], new_shape[0], 3), color, dtype=np.uint8)
    canvas[(new_h - img_h) // 2:(new_h - img_h) // 2 + img_h, (new_w - img_w) // 2:(new_w - img_w) // 2 + img_w, :] = resized_image
    return canvas, scale

def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, :4] = np.clip(coords[:, :4], 0, img0_shape[:2])
    return coords

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection with YOLOv5")
        
        self.video_source = None
        self.cap = None
        self.model = None
        
        self.load_model()
        
        self.label = tk.Label(self.root)
        self.label.pack()
        
        self.detect_button = tk.Button(self.root, text="Detect Objects", command=self.detect_objects)
        self.detect_button.pack()
        
        self.browse_button = tk.Button(self.root, text="Browse Video", command=self.browse_video)
        self.browse_button.pack()
        
        self.webcam_button = tk.Button(self.root, text="Use Webcam", command=self.use_webcam)
        self.webcam_button.pack()
        
        self.exit_button = tk.Button(self.root, text="Exit", command=self.exit_app)
        self.exit_button.pack()
    
    def load_model(self):
        # Load YOLOv5 model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = attempt_load("yolov5s.pt",device=self.device)
        self.model.eval()
    
    def browse_video(self):
        video_file = filedialog.askopenfilename()
        if video_file:
            self.video_source = video_file
            self.cap = cv2.VideoCapture(self.video_source)
            self.detect_objects()
    
    def use_webcam(self):
        self.video_source = 0
        self.cap = cv2.VideoCapture(self.video_source)
        self.detect_objects()
    
    def detect_objects(self):
        if self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Convert OpenCV BGR image to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize and normalize image for YOLOv5
        img0, scale = letterbox(frame, new_shape=(640, 640))
        img = img0[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(torch.float32)
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Object detection
        img = img.to(self.device)
        pred = self.model(img)[0]
        pred = non_max_suppression(pred, 0.4, 0.5)
        
        # Process detections
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.model.names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert the frame back to PIL format and display in GUI
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=img)
        self.label.imgtk = img
        self.label.config(image=img)
        
        # Repeat the detection process
        self.root.after(10, self.detect_objects)
    
    def exit_app(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
