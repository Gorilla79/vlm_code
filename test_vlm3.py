import pyrealsense2 as rs
import numpy as np
import cv2
import time
import torch
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import re

class ASFMNavigator:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo = YOLO('yolov8n.pt').to(self.device)

        # ✅ Public BLIP 모델 사용 (비공개 모델 사용 안함)
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.vlm_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)

    def run(self):
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_image)

                prompt = "Describe the environment."

                inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.vlm_model.generate(**inputs, max_new_tokens=50)

                result = self.processor.decode(outputs[0], skip_special_tokens=True)
                print("\n[🤖 VLM Interpretation]:", result)

                annotated = cv2.putText(color_image.copy(), result[:80], (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow("BLIP2 RealSense Feed", annotated)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    navigator = ASFMNavigator()
    navigator.run()
