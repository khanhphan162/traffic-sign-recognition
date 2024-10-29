import cv2
import torch
from pathlib import Path
import argparse
import tkinter as tk
from tkinter import filedialog
import sys

# Import yolov5
yolov5_path = Path("F:/NCKH/yolov5-live/Detect/yolov5")
sys.path.append(str(yolov5_path))
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection")
        
        self.source_label = tk.Label(root, text="Choose Source:")
        self.source_label.pack()

        self.source_var = tk.StringVar(root, "webcam")
        self.source_radio1 = tk.Radiobutton(root, text="Webcam", variable=self.source_var, value="webcam")
        self.source_radio1.pack()
        self.source_radio2 = tk.Radiobutton(root, text="Video", variable=self.source_var, value="video")
        self.source_radio2.pack()

        self.choose_button = tk.Button(root, text="Choose Video", command=self.choose_video)
        self.choose_button.pack()

        self.detect_button = tk.Button(root, text="Start Detection", command=self.start_detection)
        self.detect_button.pack()

        self.model_weights = "best.pt"
        
    def choose_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
        self.video_path = file_path
        
    def start_detection(self):
        source = self.source_var.get()
        if source == "video" and not hasattr(self, 'video_path'):
            tk.messagebox.showerror("Error", "Please choose a video file.")
            return

        # Load YOLOv5 model
        weights = self.model_weights
        device = select_device('')

        model = attempt_load(weights, device=device)
        model.eval()

        # Capture video or webcam
        if source == 'video':
            cap = cv2.VideoCapture(self.video_path)
        elif source == 'webcam':
            cap = cv2.VideoCapture(0)
        else:
            raise ValueError("Invalid source. Please choose 'video' or 'webcam'.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect objects
            results = detect_objects(frame, model, device)

            # Display results
            if results is not None:
                for x1, y1, x2, y2, conf, cls in results:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, f'{cls}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def detect_objects(img, model, device, conf_thres=0.25, iou_thres=0.45):
    # Convert image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    # Inference
    results = model(img, size=640)

    # Postprocess
    results = non_max_suppression(results, conf_thres, iou_thres)[0]

    # If there are detections, format them
    if results is not None and len(results):
        results[:, :4] = scale_boxes(img.shape[2:], results[:, :4], img.shape).round()

    return results

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
